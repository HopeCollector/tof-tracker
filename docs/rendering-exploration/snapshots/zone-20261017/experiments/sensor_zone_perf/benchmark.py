"""One reproducible, isolated performance cell for the formal GPU sensors.

Run inside Docker. This script does not call the online review service and does
not change production sampling. Inputs and measured outputs stay on CUDA.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from devtools.config import load_config
from devtools.rendering import BatchRenderer
from devtools.scene import Pose, Scene
from devtools.sensors import (CameraBatch, RigBatch, _depth, _same_snapshot,
                              mounted_poses, poses_to_tensor, sample_camera,
                              sample_depth, sample_rig)
from devtools._sensor_processing import postprocess_camera


SEED = 20261016
FIRST_POSITION = (.00400000019, -17.60400009, 1.5)
FIRST_YAW_DEG = 78.6839981
TARGET_POSITION = (.7369999886, -13.31799984, 1.5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() if result.returncode == 0 else None


def allocator():
    if not torch.cuda.is_initialized():
        return {}
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def quaternion(roll, pitch, yaw):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def make_poses(batch, pattern="varied"):
    """Prefix-stable poses: exact review case first, nearby full-RPY views next."""
    random = np.random.default_rng(SEED)
    result = [Pose(FIRST_POSITION, quaternion(0., 0., math.radians(FIRST_YAW_DEG)))]
    if pattern == "repeat":
        return result * batch
    for _ in range(batch - 1):
        delta = random.uniform(-1., 1., 6)
        position = np.asarray(FIRST_POSITION) + delta[:3] * (.25, .25, .15)
        rpy = np.radians(delta[3:] * (8., 6., 12.) + (0., 0., FIRST_YAW_DEG))
        result.append(Pose(tuple(float(x) for x in position), quaternion(*rpy)))
    return result


def tensors(value, prefix=""):
    """Flatten only tensors, keeping class/shape metadata outside timers."""
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if is_dataclass(value):
        result = {}
        for field in fields(value):
            result.update(tensors(getattr(value, field.name),
                                  f"{prefix}.{field.name}".lstrip(".")))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(tensors(item, f"{prefix}.{index}".lstrip(".")))
        return result
    return {}


def describe(value):
    return {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "device": str(tensor.device),
                   "bytes": tensor.numel() * tensor.element_size()}
            for name, tensor in tensors(value).items()}


def verify_equal(actual, expected):
    """Explicit diagnostic download outside all timers; exact IDs/classes/masks."""
    left, right = tensors(actual), tensors(expected)
    if left.keys() != right.keys():
        raise AssertionError(f"output tensor fields differ: {left.keys()} / {right.keys()}")
    report = {}
    for name in left:
        a, b = left[name].detach().cpu(), right[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{name}: shape or dtype differs")
        if a.is_floating_point():
            category = lambda x: torch.where(torch.isfinite(x), 0,
                                  torch.where(torch.isposinf(x), 1,
                                  torch.where(torch.isnan(x), 2, 3)))
            classes = int((category(a) != category(b)).sum())
            finite = torch.isfinite(a) & torch.isfinite(b)
            error = float((a[finite].double() - b[finite].double()).abs().max()) if bool(finite.any()) else 0.
            equal = torch.allclose(a, b, rtol=0., atol=0., equal_nan=True)
            report[name] = {"category_disagreements": classes,
                            "finite_max_abs": error, "passed": bool(equal)}
        else:
            errors = int((a != b).sum())
            report[name] = {"disagreements": errors, "passed": errors == 0}
    if not all(item["passed"] for item in report.values()):
        raise AssertionError(f"decomposed stages disagree with production API: {report}")
    return {"passed": True, "comparison": "exact, including NaN/+inf classes, masks and IDs",
            "tensors": report}


def measure(action, args, *, units, expected_draws, renderer):
    """Single-call synchronized wall latency; event spans are separate probes."""
    before_geometry = renderer.geometry_info
    for _ in range(args.warmup):
        output = action()
        torch.cuda.synchronize()
        del output
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    memory_before = allocator()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = action()
    torch.cuda.synchronize()
    pilot_ms = (time.perf_counter() - started) * 1000
    del output
    repeats = min(args.max_repeats, max(args.min_repeats,
                  math.ceil(args.target_budget_ms / max(pilot_ms, .001))))
    wall_samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = action()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - started) * 1000)
        del output
    memory_after = allocator()
    event_samples = []
    for _ in range(min(repeats, args.event_repeats)):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = action()
        end.record()
        end.synchronize()
        event_samples.append(start.elapsed_time(end))
        del output
    after_geometry = renderer.geometry_info
    actions = args.warmup + 1 + repeats + len(event_samples)
    draw_delta = after_geometry["instanced_draw_count"] - before_geometry["instanced_draw_count"]
    render_delta = after_geometry["render_calls"] - before_geometry["render_calls"]
    contract = {"action_count": actions, "expected_draws_per_action": expected_draws,
                "draw_delta": draw_delta, "render_delta": render_delta,
                "geometry_uploads_before": before_geometry["geometry_uploads"],
                "geometry_uploads_after": after_geometry["geometry_uploads"],
                "geometry_buffer_unchanged": before_geometry["scene_geometry_buffer_name"] == after_geometry["scene_geometry_buffer_name"]}
    contract["passed"] = (draw_delta == render_delta == actions * expected_draws and
                          before_geometry["geometry_uploads"] == after_geometry["geometry_uploads"] == 1 and
                          contract["geometry_buffer_unchanged"])
    if not contract["passed"]:
        raise AssertionError(f"draw/shared geometry contract failed: {contract}")
    median = statistics.median(wall_samples)
    return {"wall_median_ms": median,
            "wall_p95_ms": float(np.quantile(wall_samples, .95)),
            "wall_min_ms": min(wall_samples), "wall_samples_ms": wall_samples,
            "warmup": args.warmup, "pilot_ms": pilot_ms,
            "repeats": repeats, "target_budget_ms": args.target_budget_ms,
            "actual_measured_total_ms": sum(wall_samples),
            "units_per_second": units * 1000 / median,
            "cuda_event_span_ms": event_samples,
            "cuda_event_span_median_ms": statistics.median(event_samples) if event_samples else None,
            "cuda_event_scope": "whole CUDA stream span, including host submission/interop gaps; not pure kernel execution",
            "torch_memory_before": memory_before, "torch_memory_after": memory_after,
            "torch_peak_increment_bytes": memory_after["peak_allocated_bytes"] - memory_before["allocated_bytes"],
            "gl_memory_after": renderer.memory_info, "draw_contract": contract}


def gl_draw_probe(action, renderer, repeats=3):
    """Timer queries wrap only the existing glDrawArraysInstanced call."""
    draw = renderer._gl.glDrawArraysInstanced
    samples, live_queries = [], []

    def instrumented(*draw_args):
        # Keep outstanding pose-pack/copy work outside the GL timing interval.
        # This changes scheduling only in the independent optional probe.
        torch.cuda.synchronize(renderer.device)
        query = renderer._ctx.query(time=True)
        live_queries.append(query)
        with query:
            draw(*draw_args)

    renderer._gl.glDrawArraysInstanced = instrumented
    try:
        for _ in range(repeats):
            torch.cuda.synchronize()
            output = action()
            torch.cuda.synchronize()
            samples.append([int(query.elapsed) / 1e6 for query in live_queries])
            live_queries.clear()
            del output
    finally:
        renderer._gl.glDrawArraysInstanced = draw
    return {"scope": "GL_TIME_ELAPSED around each glDrawArraysInstanced only; excludes clear/transfer/interop/Torch work",
            "repeats": repeats, "per_draw_samples_ms": samples,
            "per_action_sum_samples_ms": [sum(sample) for sample in samples],
            "per_action_sum_median_ms": statistics.median(sum(sample) for sample in samples)}


def synchronized_first_forward(action, renderer):
    """Diagnostic setup only: finish each GL draw before first CUDA consumers.

    The default large-batch first call can wait indefinitely at the GL/CUDA
    boundary on the test machine. This option tests whether fully synchronized
    initialization changes subsequent unmodified production calls; it does not
    guarantee that those calls complete. The original function is restored
    before every reported timing stage.
    """
    draw = renderer._gl.glDrawArraysInstanced
    def initial_draw(*parameters):
        draw(*parameters)
        renderer._ctx.finish()
    renderer._gl.glDrawArraysInstanced = initial_draw
    try:
        return action()
    finally:
        renderer._gl.glDrawArraysInstanced = draw


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.set_grad_enabled(False)
    renderer = None
    run_started = time.perf_counter()

    def progress(stage):
        # Diagnostics stay outside measured blocks and flush for timeout traces.
        print(json.dumps({"stage": stage, "elapsed_s": time.perf_counter() - run_started,
                          "mode": args.mode, "batch": args.batch}), file=sys.stderr, flush=True)

    record = {"status": "running", "mode": args.mode, "sampling": args.sampling,
              "synchronized_first_forward": args.synchronize_first_forward,
              "pose_pattern": args.pose_pattern,
              "batch": args.batch, "grad_enabled": False,
              "scope": {"pipeline": "prepared CUDA sensor poses to raw owned CUDA distance/identity/hit/triangle; includes GL draw, GL/CUDA interop, D2D copies and Torch output formatting",
                        "postprocess": "fixed raw CUDA images and prepared mounted poses to formal GPU observation; no rendering",
                        "end_to_end": "unmodified formal sample_depth/sample_camera/sample_rig API on CUDA input poses; rig includes mounts",
                        "excluded": "map load, source hashing, CPU pose generation/upload, UI HTTP/base64/image downloads, scene motion and renderer construction",
                        "nonadditive": "stages measured independently with synchronization; their medians need not sum to complete-call latency"},
              "unit": "body_rigs" if args.mode == "rig" else "views",
              "expected_draws_per_action": {"tof": 1, "camera": 2, "rig": 3}[args.mode]}
    try:
        if args.mode != "tof" and args.sampling != "zone":
            raise ValueError("--sampling pixel is only supported with --mode tof (non-equivalent cost baseline)")
        map_path, config_path = Path(args.map).resolve(), Path(args.config).resolve()
        config = load_config(config_path)
        sensor = config.sensors
        if sensor is None:
            raise ValueError("configuration has no sensors")
        progress("provenance")
        record["source"] = {"head": command(["git", "rev-parse", "HEAD"]),
                            "working_tree_status": command(["git", "status", "--short"]),
                            "files_sha256": {name: sha256(ROOT / name) for name in
                                ("devtools/rendering.py", "devtools/_zone_shaders.py", "devtools/_gl_cuda.py",
                                 "devtools/sensors.py", "devtools/_sensor_processing.py", "devtools/config.py",
                                 "experiments/sensor_zone_perf/benchmark.py")}}
        record["map"] = {"path": str(map_path), "sha256": sha256(map_path)}
        record["config"] = {"path": str(config_path), "sha256": sha256(config_path),
                            "values": asdict(config)}
        record["environment"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                 "python": sys.version, "numpy": np.__version__,
                                 "cpu_threads": torch.get_num_threads(),
                                 "torch_cuda_arch_list": torch.cuda.get_arch_list(),
                                 "cuda_gl_environment": {key: os.environ.get(key) for key in
                                     ("CUDA_MODULE_LOADING", "CUDA_MODULE_DATA_LOADING", "CUDA_CACHE_PATH",
                                      "CUDA_CACHE_DISABLE", "CUDA_FORCE_PTX_JIT", "CUDA_LAUNCH_BLOCKING",
                                      "__GL_SHADER_DISK_CACHE", "__GL_SHADER_DISK_CACHE_PATH",
                                      "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "HOME", "XDG_CACHE_HOME")},
                                 "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"])}
        progress("cuda_init")
        started = time.perf_counter()
        torch.cuda.init()
        torch.cuda.synchronize()
        record["cuda_init_ms"] = (time.perf_counter() - started) * 1000
        record["environment"]["gpu"] = torch.cuda.get_device_name()
        record["environment"]["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        progress("scene_load")
        started = time.perf_counter()
        scene = Scene.from_obj(map_path, config, make_poses(1)[0], TARGET_POSITION, t_ref=12.5)
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        record["map_load_ms"] = (time.perf_counter() - started) * 1000
        progress("pose_upload")
        poses_cpu = make_poses(args.batch, args.pose_pattern)
        pose_array = np.stack([pose.as_transform().matrix for pose in poses_cpu]).astype(np.float32)
        poses = poses_to_tensor(poses_cpu)
        torch.cuda.synchronize()
        record["input"] = {"seed": SEED, "pattern": args.pose_pattern,
                           "algorithm": "repeat: duplicate first case; varied: numpy.default_rng.uniform(-1,1,6) per view after first, translation offsets scaled by (.25,.25,.15)m and RPY offsets (8,6,12)deg; prefixes identical across batch sizes",
                           "first_position_m": list(FIRST_POSITION), "first_rpy_deg": [0., 0., FIRST_YAW_DEG],
                           "target_position_m": list(TARGET_POSITION),
                           "float32_pose_sha256": hashlib.sha256(pose_array.tobytes()).hexdigest(),
                           "pose_shape": list(poses.shape), "pose_device": str(poses.device)}
        progress("renderer_setup")
        started = time.perf_counter()
        renderer = BatchRenderer(snapshot)
        torch.cuda.synchronize()
        record["renderer_setup_ms"] = (time.perf_counter() - started) * 1000
        record["geometry"] = renderer.geometry_info
        progress("renderer_ready")
        max_views = args.batch * (4 if args.mode == "rig" else 1)
        if max_views > renderer.geometry_info["max_array_texture_layers"]:
            record.update(status="capacity_limit", error=f"requires {max_views} array layers; device permits {renderer.geometry_info['max_array_texture_layers']}; no hidden chunking")
            return record

        if args.mode == "tof":
            full_call = lambda: sample_depth(renderer, sensor.tof, poses, sampling=args.sampling)
            mount_call = None
        elif args.mode == "camera":
            full_call = lambda: sample_camera(renderer, sensor.camera, poses, sensor.min_visible_fraction)
            mount_call = None
        else:
            full_call = lambda: sample_rig(renderer, sensor, poses)
            mount_call = lambda: (mounted_poses(poses, sensor.tof_mounts).reshape(-1, 4, 4),
                                  mounted_poses(poses, (sensor.camera_mount,))[:, 0])

        before = renderer.geometry_info
        progress("first_forward")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        first = (synchronized_first_forward(full_call, renderer)
                 if args.synchronize_first_forward else full_call())
        torch.cuda.synchronize()
        record["first_forward_ms"] = (time.perf_counter() - started) * 1000
        record["first_forward_scope"] = ("setup with explicit GL finish after each draw; not default cold-call latency"
                                         if args.synchronize_first_forward else "default cold formal API")
        record["first_forward_memory"] = allocator()
        record["first_forward_gl_memory"] = renderer.memory_info
        record["output_tensors"] = describe(first)
        after = renderer.geometry_info
        record["first_forward_contract"] = {"draw_delta": after["instanced_draw_count"] - before["instanced_draw_count"],
                                             "render_delta": after["render_calls"] - before["render_calls"],
                                             "geometry_uploads": after["geometry_uploads"],
                                             "geometry_buffer_unchanged": before["scene_geometry_buffer_name"] == after["scene_geometry_buffer_name"]}
        first_contract = record["first_forward_contract"]
        if not (first_contract["draw_delta"] == first_contract["render_delta"] == record["expected_draws_per_action"]
                and first_contract["geometry_uploads"] == 1 and first_contract["geometry_buffer_unchanged"]):
            raise AssertionError(f"first-forward contract failed: {first_contract}")
        del first
        progress("first_forward_done")
        if args.mode == "rig":
            tof_poses, camera_poses = mount_call()
        else:
            tof_poses = camera_poses = poses
        torch.cuda.synchronize()

        def raw():
            if args.mode == "tof":
                return (renderer.render(tof_poses, sensor.tof, sampling=args.sampling),)
            if args.mode == "camera":
                return (renderer.render(camera_poses, sensor.camera),
                        renderer.render(camera_poses, sensor.camera, only_object="target"))
            return (renderer.render(tof_poses, sensor.tof, sampling="zone"),
                    renderer.render(camera_poses, sensor.camera),
                    renderer.render(camera_poses, sensor.camera, only_object="target"))

        def post(data):
            if args.mode == "tof":
                return _depth(data[0], tof_poses, sensor.tof)
            full, target = data[-2:]
            _same_snapshot(full, target)
            depth = _depth(full, camera_poses, sensor.camera)
            measurements = postprocess_camera(depth.range_m, depth.object_index,
                            target.distance_m, target.object_index, camera_poses,
                            sensor.camera, full.map_bounds_m, full.object_ids.index("target"),
                            sensor.min_visible_fraction)
            camera = CameraBatch(depth, **measurements)
            if args.mode == "camera":
                return camera
            tof = _depth(data[0], tof_poses, sensor.tof)
            _same_snapshot(data[0], full)
            return RigBatch(tof.range_m.reshape(args.batch, 4, sensor.tof.height, sensor.tof.width),
                            camera.target_hint, camera, tof.scene_version, tof.t_ref)

        # Isolated measurements. Fixed raw storage exists only in postprocess.
        progress("pipeline_measure")
        record["pipeline"] = measure(raw, args, units=args.batch,
                                      expected_draws=record["expected_draws_per_action"], renderer=renderer)
        progress("production_parity")
        fixed = raw()
        torch.cuda.synchronize()
        record["fixed_raw_tensors"] = describe(fixed)
        decomposed = post(fixed)
        production = full_call()
        torch.cuda.synchronize()
        record["production_parity"] = verify_equal(decomposed, production)
        del decomposed, production
        progress("postprocess_measure")
        record["postprocess"] = measure(lambda: post(fixed), args, units=args.batch,
                                         expected_draws=0, renderer=renderer)
        del fixed
        gc.collect()
        if mount_call:
            progress("mount_measure")
            record["mount"] = measure(mount_call, args, units=args.batch,
                                       expected_draws=0, renderer=renderer)
        progress("end_to_end_measure")
        record["end_to_end"] = measure(full_call, args, units=args.batch,
                                        expected_draws=record["expected_draws_per_action"], renderer=renderer)
        if args.probe_gl_draw:
            progress("gl_draw_probe")
            record["gl_draw_probe"] = gl_draw_probe(raw, renderer)
        record["geometry_after_warm"] = renderer.geometry_info
        record["status"] = "ok"
        progress("complete")
    except torch.cuda.OutOfMemoryError as error:
        record.update(status="oom", error=str(error), traceback=traceback.format_exc(),
                      memory_at_failure=allocator())
    except Exception as error:
        record.update(status="error", error=str(error), traceback=traceback.format_exc(),
                      memory_at_failure=allocator())
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as error:
                record["cleanup_error"] = str(error)
                if record["status"] == "ok":
                    record["status"] = "cleanup_error"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/devtools.json"))
    parser.add_argument("--mode", choices=("tof", "camera", "rig"), required=True)
    parser.add_argument("--sampling", choices=("zone", "pixel"), default="zone",
                        help="ToF only: pixel is a non-equivalent historical cost baseline; camera stays pixel and rig stays zone+pixel")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--pose-pattern", choices=("varied", "repeat"), default="varied")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-budget-ms", type=float, default=500.)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--event-repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--probe-gl-draw", action="store_true")
    parser.add_argument("--synchronize-first-forward", action="store_true",
                        help="Only first setup call finishes each GL draw; warm measured calls remain unmodified")
    args = parser.parse_args()
    if args.batch < 1 or args.min_repeats < 1 or args.max_repeats < args.min_repeats or args.warmup < 0 or args.event_repeats < 0 or args.target_budget_ms <= 0 or args.cpu_threads < 1:
        parser.error("invalid batch, repeat, timing budget or thread count")
    result = run(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "mode": args.mode, "sampling": args.sampling,
                      "batch": args.batch, "output": args.output,
                      "end_to_end_median_ms": result.get("end_to_end", {}).get("wall_median_ms"),
                      "error": result.get("error")}), flush=True)
    return 0 if result["status"] in ("ok", "capacity_limit", "oom") else 1


if __name__ == "__main__":
    raise SystemExit(main())
