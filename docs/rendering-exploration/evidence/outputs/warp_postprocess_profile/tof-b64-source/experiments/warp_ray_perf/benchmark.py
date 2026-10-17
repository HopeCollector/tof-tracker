"""Measure one Docker-only Warp sensor cell, keeping all timed data on CUDA.

Example: python experiments/warp_ray_perf/benchmark.py --map MAP.obj --mode tof
--batch 64 --output outputs/warp_ray_perf/tof-64.json

Run cells sequentially on the GPU. Each stage is independently synchronized;
its median is not an additive component of another stage. CUDA events span
the complete stream interval, including host submission gaps, not only kernels.
The varied Pose generator is extracted from sensor_zone_perf's fixed seed and
prefix-stable formula; this script never imports the historical GL backend.
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
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from devtools.config import load_config
from devtools.ray_rendering import RayRenderer
from devtools.scene import Pose, Scene
from devtools.sensors import (CameraBatch, RigBatch, generate_rays,
                              mounted_poses, poses_to_tensor, process_samples,
                              sample_camera, sample_depth, sample_rig)
from devtools._ray_processing import camera_measurements


SEED = 20261016
FIRST_POSITION = (.00400000019, -17.60400009, 1.5)
FIRST_YAW_DEG = 78.6839981
HISTORICAL_TARGET_POSITION = (.7369999886, -13.31799984, 1.5)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(arguments):
    try:
        result = subprocess.run(arguments, cwd=ROOT, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def allocator():
    if not torch.cuda.is_initialized():
        return {}
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def global_memory():
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total, "used_bytes": total - free,
            "scope": "whole CUDA device including desktop/other processes; shared-device approximation"}


def quaternion(roll, pitch, yaw):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def make_poses(batch, pattern="varied"):
    """Same prefixes and full-RPY distribution as the prior zone benchmark."""
    random = np.random.default_rng(SEED)
    first = Pose(FIRST_POSITION, quaternion(0., 0., math.radians(FIRST_YAW_DEG)))
    if pattern == "repeat":
        return [first] * batch
    result = [first]
    for _ in range(batch - 1):
        delta = random.uniform(-1., 1., 6)
        position = np.asarray(FIRST_POSITION) + delta[:3] * (.25, .25, .15)
        rpy = np.radians(delta[3:] * (8., 6., 12.) + (0., 0., FIRST_YAW_DEG))
        result.append(Pose(tuple(float(x) for x in position), quaternion(*rpy)))
    return result


def tensors(value, prefix=""):
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
    result = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                     "device": str(tensor.device),
                     "bytes": tensor.numel() * tensor.element_size()}
              for name, tensor in tensors(value).items()}
    if any(not tensor.is_cuda for tensor in tensors(value).values()):
        raise AssertionError("sensor computation returned a CPU tensor")
    return result


def verify_equal(actual, expected):
    """Download only outside timers; compare every tensor field, including raw."""
    def metadata(value):
        if isinstance(value, torch.Tensor):
            return None
        if is_dataclass(value):
            return {field.name: metadata(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, (tuple, list)):
            return tuple(metadata(item) for item in value)
        return value

    if type(actual) is not type(expected) or metadata(actual) != metadata(expected):
        raise AssertionError("decomposed/formal output classes or scene metadata differ")
    left, right = tensors(actual), tensors(expected)
    if left.keys() != right.keys():
        raise AssertionError(f"different tensor fields: {left.keys()} / {right.keys()}")
    report = {}
    for name in left:
        a, b = left[name].detach().cpu(), right[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{name}: different shape or dtype")
        if a.is_floating_point():
            finite = torch.isfinite(a) & torch.isfinite(b)
            maximum = float((a[finite].double() - b[finite].double()).abs().max()) if bool(finite.any()) else 0.
            equal = bool(torch.allclose(a, b, rtol=0., atol=0., equal_nan=True))
            report[name] = {"passed": equal, "finite_max_abs": maximum,
                            "nonfinite_class_disagreements": int(((torch.isnan(a) != torch.isnan(b))
                                | (torch.isposinf(a) != torch.isposinf(b))
                                | (torch.isneginf(a) != torch.isneginf(b))).sum())}
        else:
            errors = int((a != b).sum())
            report[name] = {"passed": errors == 0, "disagreements": errors}
    if not all(value["passed"] for value in report.values()):
        raise AssertionError(f"decomposed/formal outputs disagree: {report}")
    return {"passed": True, "comparison": "exact all tensor fields; matching NaN/Inf classes and scene metadata",
            "host_reads": "verification only, outside every measured block", "fields": report}


def measure(action, args, renderer, expected_traces, units, queried_rays):
    before = renderer.geometry_info
    for _ in range(args.warmup):
        output = action()
        torch.cuda.synchronize()
        del output
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory_before = allocator()
    global_before = global_memory()
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
    after = renderer.geometry_info
    calls = args.warmup + 1 + repeats + len(event_samples)
    contract = {"action_count": calls, "expected_traces_per_action": expected_traces,
                "trace_call_delta": after["trace_calls"] - before["trace_calls"],
                "rays_traced_delta": after["rays_traced"] - before["rays_traced"],
                "geometry_uploads_before": before["geometry_uploads"],
                "geometry_uploads_after": after["geometry_uploads"]}
    contract["passed"] = (contract["trace_call_delta"] == expected_traces * calls
                           and contract["rays_traced_delta"] == queried_rays * calls
                           and before["geometry_uploads"] == after["geometry_uploads"] == 1)
    if not contract["passed"]:
        raise AssertionError(f"shared scene / batch trace contract failed: {contract}")
    memory_after = allocator()
    median = statistics.median(wall_samples)
    return {"wall_median_ms": median, "wall_p95_ms": float(np.quantile(wall_samples, .95)),
            "wall_min_ms": min(wall_samples), "wall_samples_ms": wall_samples,
            "pilot_ms": pilot_ms, "repeats": repeats, "warmup": args.warmup,
            "units_per_second": units * 1000 / median,
            "queried_rays_per_second": queried_rays * 1000 / median if queried_rays else None,
            "cuda_event_span_ms": event_samples,
            "cuda_event_span_median_ms": statistics.median(event_samples) if event_samples else None,
            "cuda_event_scope": "entire current-stream interval, including CPU submission gaps; not pure kernel time",
            "torch_memory_before": memory_before, "torch_memory_after": memory_after,
            "torch_peak_increment_bytes": memory_after["peak_allocated_bytes"] - memory_before["allocated_bytes"],
            "global_device_memory_before": global_before, "global_device_memory_after": global_memory(),
            "scene_memory_after": renderer.memory_info, "batch_contract": contract}


def run(args):
    torch.set_num_threads(args.cpu_threads)
    torch.set_grad_enabled(False)
    renderer = None
    started_run = time.perf_counter()
    record = {"status": "running", "mode": args.mode, "batch": args.batch,
              "pose_pattern": args.pose_pattern, "grad_enabled": False,
              "unit": "body_rigs" if args.mode == "rig" else "views",
              "scope": {
                  "ray_generation": "prepared CUDA mounted sensor poses to contiguous world origin/direction arrays",
                  "trace": "fixed CUDA rays to owned CUDA closest-hit results; camera includes target-only query",
                  "postprocess": "fixed raw hits/rays to formal sensor output; no tracing",
                  "pipeline": "ray generation plus trace; prepared mounted poses, no sensor postprocess",
                  "end_to_end": "formal sample_depth(weighted_quantile)/sample_camera/sample_rig on CUDA poses; rig includes mount composition",
                  "excluded": "map parsing, CPU pose generation/upload, renderer/BVH setup, motion, GUI/HTTP/image encoding",
                  "nonadditive": "stages independently synchronized and timed; medians need not add",
                  "residency": "all timed inputs/outputs remain CUDA tensors; parity downloads happen outside timers"}}

    def progress(stage):
        print(json.dumps({"stage": stage, "mode": args.mode, "batch": args.batch,
                          "elapsed_s": time.perf_counter() - started_run}), file=sys.stderr, flush=True)

    try:
        progress("provenance")
        map_path, config_path = Path(args.map).resolve(), Path(args.config).resolve()
        config = load_config(config_path)
        sensor = config.sensors
        if sensor is None:
            raise ValueError("configuration has no sensors")
        if (sensor.tof.hfov_deg, sensor.tof.vfov_deg, sensor.tof.min_range_m,
                sensor.tof.max_range_m) != (45., 45., .015, 4.):
            raise ValueError("this benchmark requires the agreed ToF 45/45 degree, .015..4m configuration")
        if sensor.tof.samples_per_axis != 4 or sensor.camera.samples_per_axis != 2:
            raise ValueError("this benchmark requires 4x4 ToF and 2x2 camera subrays")
        files = ("devtools/ray_rendering.py", "devtools/sensors.py", "devtools/_ray_processing.py",
                 "devtools/config.py", "devtools/scene.py", "devtools/mesh.py",
                 "experiments/warp_ray_perf/benchmark.py")
        record["source"] = {"head": command(["git", "rev-parse", "HEAD"]),
                            "working_tree_status": command(["git", "status", "--short"]),
                            "files_sha256": {name: sha256(ROOT / name) for name in files}}
        record["map"] = {"path": str(map_path), "sha256": sha256(map_path)}
        record["config"] = {"path": str(config_path), "sha256": sha256(config_path), "values": asdict(config)}
        record["environment"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                 "warp": wp.__version__, "numpy": np.__version__, "python": sys.version,
                                 "cpu_threads": torch.get_num_threads(),
                                 "environment": {key: os.environ.get(key) for key in (
                                     "CUDA_MODULE_LOADING", "CUDA_MODULE_DATA_LOADING", "CUDA_CACHE_PATH",
                                     "CUDA_LAUNCH_BLOCKING", "WP_CACHE_DIR")},
                                 "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"])}
        progress("cuda_init")
        start = time.perf_counter()
        torch.cuda.init()
        torch.cuda.synchronize()
        record["cuda_init_ms"] = (time.perf_counter() - start) * 1000
        record["environment"]["gpu"] = torch.cuda.get_device_name()
        record["environment"]["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        record["environment"]["torch_cuda_arch_list"] = torch.cuda.get_arch_list()
        progress("warp_init")
        start = time.perf_counter()
        wp.init()
        wp.get_device("cuda:0")
        torch.cuda.synchronize()
        record["warp_init_ms"] = (time.perf_counter() - start) * 1000
        progress("scene_load")
        first_pose = make_poses(1)[0]
        target = first_pose.body_to_world((args.target_distance_m, 0., 0.))
        start = time.perf_counter()
        scene = Scene.from_obj(map_path, config, first_pose, target, t_ref=12.5)
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        record["map_load_ms"] = (time.perf_counter() - start) * 1000
        progress("pose_upload")
        poses_cpu = make_poses(args.batch, args.pose_pattern)
        pose_array = np.stack([pose.as_transform().matrix for pose in poses_cpu]).astype(np.float32)
        poses = poses_to_tensor(poses_cpu)
        torch.cuda.synchronize()
        record["input"] = {"seed": SEED, "pattern": args.pose_pattern,
                           "algorithm": "first historical pose; subsequent numpy.default_rng.uniform(-1,1,6), translation (.25,.25,.15)m, RPY offsets (8,6,12)deg; prefix-stable across B",
                           "first_position_m": list(FIRST_POSITION), "first_rpy_deg": [0., 0., FIRST_YAW_DEG],
                           "target_position_m": list(target), "target_distance_from_first_origin_m": args.target_distance_m,
                           "target_policy": "center lies on first view forward axis; geometry still participates in normal background occlusion",
                           "historical_target_position_m": list(HISTORICAL_TARGET_POSITION),
                           "historical_case_note": "historical target center is about 4.348m away, radius .3m; its nearest surface exceeds the new 4m ToF range and is deliberately not used as the default performance target",
                           "float32_pose_sha256": hashlib.sha256(pose_array.tobytes()).hexdigest(),
                           "pose_shape": list(poses.shape), "pose_device": str(poses.device)}
        progress("scene_bvh_setup")
        torch.cuda.synchronize()
        before_memory, before_allocator = global_memory(), allocator()
        start = time.perf_counter()
        renderer = RayRenderer(snapshot)
        torch.cuda.synchronize()
        record["scene_bvh_setup_ms"] = (time.perf_counter() - start) * 1000
        after_memory = global_memory()
        record["scene_bvh_setup_scope"] = "CPU geometry assembly, scene tensor upload and native Warp BVH construction; excludes prior explicit Warp init; not exclusive BVH-kernel timing"
        record["scene_memory"] = {"torch_before": before_allocator, "torch_after": allocator(),
                                  "global_before": before_memory, "global_after": after_memory,
                                  "global_used_delta_bytes": after_memory["used_bytes"] - before_memory["used_bytes"],
                                  "global_delta_scope": "shared device approximate delta, includes allocator reservations and other processes; not exact Warp BVH size",
                                  "explicit": renderer.memory_info, "warp_bvh_bytes": None}
        record["geometry"] = renderer.geometry_info
        expected_traces = {"tof": 1, "camera": 2, "rig": 3}[args.mode]
        tof_rays = args.batch * sensor.tof.height * sensor.tof.width * sensor.tof.samples_per_axis ** 2
        camera_rays = args.batch * sensor.camera.height * sensor.camera.width * sensor.camera.samples_per_axis ** 2
        queried_rays = (tof_rays if args.mode == "tof" else 2 * camera_rays if args.mode == "camera"
                        else 4 * tof_rays + 2 * camera_rays)
        record["expected_traces_per_action"] = expected_traces
        record["queried_rays_per_action"] = queried_rays

        if args.mode == "tof":
            full_call = lambda: sample_depth(renderer, sensor.tof, poses, reduction="weighted_quantile",
                                             quantile=sensor.tof_depth_quantile, signal_floor=sensor.tof_signal_floor)
        elif args.mode == "camera":
            full_call = lambda: sample_camera(renderer, sensor.camera, poses, sensor.min_visible_fraction)
        else:
            full_call = lambda: sample_rig(renderer, sensor, poses)
        progress("first_forward")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = renderer.geometry_info
        start = time.perf_counter()
        first = full_call()
        torch.cuda.synchronize()
        record["first_forward_ms"] = (time.perf_counter() - start) * 1000
        record["first_forward_scope"] = "first formal API call; includes any uncached Warp JIT/module load, lazy sensor constants and PyTorch initialization"
        record["first_forward_memory"] = allocator()
        record["output_tensors"] = describe(first)
        after = renderer.geometry_info
        record["first_forward_contract"] = {"trace_call_delta": after["trace_calls"] - before["trace_calls"],
                                            "geometry_uploads": after["geometry_uploads"]}
        if record["first_forward_contract"] != {"trace_call_delta": expected_traces, "geometry_uploads": 1}:
            raise AssertionError("first formal call violated batch/scene sharing contract")
        del first
        if args.mode == "rig":
            tof_poses = mounted_poses(poses, sensor.tof_mounts).reshape(-1, 4, 4)
            camera_poses = mounted_poses(poses, (sensor.camera_mount,))[:, 0]
        else:
            tof_poses = camera_poses = poses
        torch.cuda.synchronize()

        def generate():
            if args.mode == "tof":
                return (generate_rays(tof_poses, sensor.tof),)
            if args.mode == "camera":
                return (generate_rays(camera_poses, sensor.camera),)
            return (generate_rays(tof_poses, sensor.tof), generate_rays(camera_poses, sensor.camera))

        def trace(rays):
            if args.mode == "tof":
                return (renderer.trace(*rays[0], sensor.tof.max_range_m),)
            tof = None
            if args.mode == "rig":
                tof = renderer.trace(*rays[0], sensor.tof.max_range_m)
            camera = rays[-1]
            full = renderer.trace(*camera, sensor.camera.max_range_m)
            target = renderer.trace(*camera, sensor.camera.max_range_m, only_object="target")
            if args.mode == "camera":
                return full, target
            return tof, full, target

        def post(data, rays):
            if args.mode == "tof":
                return process_samples(data[0], *rays[0], tof_poses, sensor.tof,
                                       reduction="weighted_quantile", quantile=sensor.tof_depth_quantile,
                                       signal_floor=sensor.tof_signal_floor)
            full, target = data[-2:]
            camera_depth = process_samples(full, *rays[-1], camera_poses, sensor.camera)
            target_depth = process_samples(target, *rays[-1], camera_poses, sensor.camera)
            measurements = camera_measurements(camera_depth.subray_range_m, camera_depth.raw.subray_object_index,
                                               target_depth.subray_range_m, target_depth.raw.subray_object_index,
                                               sensor.camera, full.object_ids.index("target"), sensor.min_visible_fraction)
            camera = CameraBatch(camera_depth, **measurements)
            if args.mode == "camera":
                return camera
            tof = process_samples(data[0], *rays[0], tof_poses, sensor.tof,
                                  reduction="weighted_quantile", quantile=sensor.tof_depth_quantile,
                                  signal_floor=sensor.tof_signal_floor)
            return RigBatch(tof.range_m.reshape(args.batch, 4, sensor.tof.height, sensor.tof.width),
                            camera.target_hint, camera, tof.scene_version, tof.t_ref)

        progress("ray_generation_measure")
        record["ray_generation"] = measure(generate, args, renderer, 0, args.batch, 0)
        progress("fixed_rays")
        fixed_rays = generate()
        record["fixed_ray_tensors"] = describe(fixed_rays)
        progress("trace_measure")
        record["trace"] = measure(lambda: trace(fixed_rays), args, renderer,
                                   expected_traces, args.batch, queried_rays)
        progress("production_parity")
        fixed_raw = trace(fixed_rays)
        decomposed, production = post(fixed_raw, fixed_rays), full_call()
        torch.cuda.synchronize()
        record["fixed_raw_tensors"] = describe(fixed_raw)
        record["production_parity"] = verify_equal(decomposed, production)
        del decomposed, production
        progress("postprocess_measure")
        record["postprocess"] = measure(lambda: post(fixed_raw, fixed_rays), args,
                                         renderer, 0, args.batch, 0)
        del fixed_raw, fixed_rays
        gc.collect()
        progress("pipeline_measure")
        record["pipeline"] = measure(lambda: trace(generate()), args, renderer,
                                      expected_traces, args.batch, queried_rays)
        progress("end_to_end_measure")
        record["end_to_end"] = measure(full_call, args, renderer,
                                        expected_traces, args.batch, queried_rays)
        record["geometry_after_warm"] = renderer.geometry_info
        record["status"] = "ok"
        progress("complete")
    except torch.cuda.OutOfMemoryError as error:
        record.update(status="oom", error=str(error), traceback=traceback.format_exc(), memory_at_failure=allocator())
    except Exception as error:
        record.update(status="error", error=str(error), traceback=traceback.format_exc(), memory_at_failure=allocator())
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
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--pose-pattern", choices=("varied", "repeat"), default="varied")
    parser.add_argument("--target-distance-m", type=float, default=3.0,
                        help="target center distance on first pose forward axis; default lies inside the agreed ToF range")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=30)
    parser.add_argument("--event-repeats", type=int, default=3)
    parser.add_argument("--target-budget-ms", type=float, default=500.)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if (args.batch < 1 or args.warmup < 0 or args.min_repeats < 1 or args.max_repeats < args.min_repeats
            or args.event_repeats < 0 or args.target_budget_ms <= 0 or args.cpu_threads < 1
            or not math.isfinite(args.target_distance_m) or args.target_distance_m <= 0):
        parser.error("invalid batch, repeats, budget, threads or target distance")
    record = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "mode": args.mode, "batch": args.batch,
                      "output": args.output, "end_to_end_median_ms": record.get("end_to_end", {}).get("wall_median_ms"),
                      "error": record.get("error")}), flush=True)
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
