"""Supplementary real-budget, stream-lifetime and float32 sphere probes.

Run only inside Docker, sequentially with other GPU measurements. This script
never patches the chunk planner. Tangent discrepancies are preserved as concrete
witnesses, not clamped or silently changed into passing measurements.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from devtools.config import DepthConfig, PerceptionConfig, SamplingRuntimeConfig, load_config
from devtools.scene import GeometrySnapshot, Pose, Scene, SceneObject, Sphere
from devtools.ray_rendering import RayRenderer
from devtools._ray_budget import plan
from devtools.sensors import poses_to_tensor, process_samples, sample_rig, sample_rig_diagnostics
from devtools.perception import (measurement_output_bytes, process_camera_perception,
                                 sample_measurements, sample_measurement_diagnostics)
from experiments.unified_ray_perf.benchmark import make_poses, tensor_fields
from verify_ray_sensors import oracle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host(value):
    return value.detach().cpu().numpy()


@contextmanager
def no_readback():
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU image readback in sampling")), \
         patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy image readback in sampling")), \
         patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar readback in sampling")):
        yield


def compare(left, right, atol=3e-5):
    left, right = tensor_fields(left), tensor_fields(right)
    assert left.keys() == right.keys()
    report = {}
    for name in left:
        a, b = host(left[name]), host(right[name])
        assert a.dtype == b.dtype and a.shape == b.shape, name
        if np.issubdtype(a.dtype, np.floating):
            assert np.array_equal(np.isnan(a), np.isnan(b)), name
            assert np.array_equal(np.isposinf(a), np.isposinf(b)), name
            assert np.array_equal(np.isneginf(a), np.isneginf(b)), name
            finite = np.isfinite(a)
            maximum = float(np.abs(a[finite] - b[finite]).max(initial=0))
            assert maximum <= atol, (name, maximum)
            report[name] = {"max_abs": maximum}
        else:
            assert np.array_equal(a, b), name
            report[name] = {"exact": True}
    return {"passed": True, "float_atol": atol, "fields": report}


def real_budget_probe(renderer, scene, configuration, poses):
    sensors = configuration.sensors
    count = len(poses)
    rays_per_body = (4 * sensors.tof.width * sensors.tof.height * sensors.tof.samples_per_axis ** 2
                     + sensors.camera.width * sensors.camera.height * sensors.camera.samples_per_axis ** 2)
    retained = measurement_output_bytes(sensors, count)
    _, initial = plan(renderer, count, rays_per_body, retained, SamplingRuntimeConfig(.5))
    fixed = initial["sampling_workset_limit_bytes"] - initial["fraction_workspace_bytes"]
    # A real fraction strictly between the cost of one and two body worksets
    # avoids a floor boundary and leaves physical-free checking fully enabled.
    fraction = (fixed + 1.5 * initial["workspace_bytes_per_item"]) / initial["cuda_total_bytes"]
    small = SamplingRuntimeConfig(fraction)
    actual_chunk, planned = plan(renderer, count, rays_per_body, retained, small)
    assert 1 <= actual_chunk <= 2, planned
    before = renderer.geometry_info["trace_calls"]
    with no_readback():
        control = sample_measurements(renderer, sensors, poses, scene,
                                      configuration.perception, runtime=SamplingRuntimeConfig(.5))
    assert renderer.geometry_info["trace_calls"] - before == 1
    before = renderer.geometry_info["trace_calls"]
    with no_readback():
        chunked = sample_measurements(renderer, sensors, poses, scene,
                                      configuration.perception, runtime=small)
    trace_delta = renderer.geometry_info["trace_calls"] - before
    assert trace_delta == chunked.execution_info["chunks"] == math.ceil(count / actual_chunk)
    parity = compare(control, chunked)
    # Reject retaining all large-batch diagnostic buffers before a query or
    # any output reservation. Only the small Pose input is materialized.
    large = poses[:1].expand(10000, -1, -1).contiguous()
    rejected = []
    for name, action in (
        ("physical_diagnostics", lambda: sample_rig_diagnostics(renderer, sensors, large, runtime=small)),
        ("measurement_diagnostics", lambda: sample_measurement_diagnostics(
            renderer, sensors, large, scene, configuration.perception, runtime=small))):
        before = renderer.geometry_info["trace_calls"]
        try:
            action()
        except MemoryError as error:
            assert renderer.geometry_info["trace_calls"] == before
            rejected.append({"name": name, "query_delta": 0, "error": str(error),
                             "budget": getattr(error, "budget_info", None)})
        else:
            raise AssertionError(name + " accepted a 10000-body retained diagnostic batch")
    return {"status": "passed", "batch": count, "default_camera": {
                "width": sensors.camera.width, "height": sensors.camera.height,
                "samples_per_axis": sensors.camera.samples_per_axis},
            "configured_fraction": fraction, "plan": planned,
            "control_execution": control.execution_info, "chunked_execution": chunked.execution_info,
            "parity": parity, "sampling_cpu_readbacks": 0,
            "large_diagnostic_prequery_rejections": rejected}


def storage_lifetime_probe(renderer, scene, configuration):
    import devtools.sensors as sensor_module

    cpu_poses = make_poses(4)
    baseline_poses = poses_to_tensor(cpu_poses)
    baseline = sample_rig(renderer, configuration.sensors, baseline_poses)
    torch.cuda.synchronize()
    rows = []
    for repetition in range(3):
        stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
        stream_a.wait_stream(torch.cuda.current_stream())
        # Initialize stream-specific library handles/workspaces before the
        # pending-work assertion. Lazy initialization may synchronize streams.
        with torch.cuda.stream(stream_b):
            warm_poses = poses_to_tensor(cpu_poses)
            warm_output = sample_rig(renderer, configuration.sensors, warm_poses)
        stream_b.synchronize()
        del warm_poses, warm_output
        # Seed two cache blocks before testing so reuse pressure need not
        # introduce a new driver allocation that could synchronize the device.
        with torch.cuda.stream(stream_a):
            spare_a = torch.empty((4, 4, 4), dtype=torch.float32, device=renderer.device)
            spare_b = torch.empty_like(spare_a)
        stream_a.synchronize()
        del spare_a, spare_b
        with torch.cuda.stream(stream_a):
            values = poses_to_tensor(cpu_poses)
        original_pointer = values.data_ptr()
        # Caller-provided readiness is distinct from consumer storage lifetime.
        stream_b.wait_stream(stream_a)
        done = torch.cuda.Event()
        original_mounted = sensor_module.mounted_poses
        delayed_once = False

        def delayed_mount(*args, **kwargs):
            nonlocal delayed_once
            if not delayed_once:
                # The planner has already run. Avoid consuming the artificial
                # delay in setup-time host synchronization; only this test
                # injects work into the physical sensor's consumption stream.
                torch.cuda._sleep(500_000_000)
                delayed_once = True
            return original_mounted(*args, **kwargs)

        with torch.cuda.stream(stream_b):
            with patch.object(sensor_module, "mounted_poses", side_effect=delayed_mount):
                sampled = sample_rig(renderer, configuration.sensors, values)
            done.record()
        assert delayed_once
        pending_before_reuse = not done.query()
        assert pending_before_reuse, "stream finished before reuse stress was queued"
        del values
        reused = 0
        with torch.cuda.stream(stream_a):
            for i in range(128):
                replacement = torch.empty((4, 4, 4), dtype=torch.float32, device=renderer.device)
                reused += int(replacement.data_ptr() == original_pointer)
                replacement.fill_(100. + i)
                del replacement
        done.synchronize()
        stream_a.synchronize()
        rows.append({"repetition": repetition, "pending_before_reuse": pending_before_reuse,
                     "same_pointer_reuses_observed": reused, "parity": compare(sampled, baseline)})
        del sampled
    return {"status": "passed", "scope": "physical sample_rig only; test injects GPU delay at first mounted_poses after planning; A produces Pose storage, B waits for readiness and consumes it, A releases/reuses while B is pending; correctness depends on input record_stream, not an API synchronization guarantee",
            "repetitions": rows}


def tangent_probe():
    records, discrepancies = [], []
    offsets = (-4e-6, -1e-6, -2e-7, -5e-8, -1e-8, 0., 1e-8, 5e-8, 2e-7, 1e-6, 4e-6)
    for case_index, (origin, axis, distance, radius) in enumerate((
        ((0., 0., 2.), (1., 0., 0.), 3., .3),
        ((0., 0., 2.), (1., 0., 0.), 4., .2),
        ((0., 0., 2.), (1., 1., 0.), 3., .3),
        ((7.123, -5.456, 1.5), (1., .3, .2), 3., .3),
        ((1000.125, -300.25, 50.), (1., .3, .2), 3., .3),
        ((0., 0., 2.), (1., .2, .3), 10., .1),
    )):
        origin = np.asarray(origin, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        normal = np.cross(axis, (0., 0., 1.))
        normal /= np.linalg.norm(normal)
        center = origin + distance * axis
        angle = math.asin(radius / distance)
        directions = np.asarray([axis * math.cos(angle + delta) + normal * math.sin(angle + delta)
                                 for delta in offsets], dtype=np.float32)
        source = GeometrySnapshot((SceneObject("target", "target", Sphere(
            Pose(tuple(float(v) for v in center)), radius)),), 3., None)
        uploaded_source = GeometrySnapshot((SceneObject("target", "target", Sphere(
            Pose(tuple(float(v) for v in center.astype(np.float32))), float(np.float32(radius)))),), 3., None)
        config = DepthConfig(len(directions), 1, 45., 45., .015, 30.)
        actual_origins = np.repeat(origin.astype(np.float32)[None], len(directions), axis=0)
        _, ideal_i, _ = oracle(source, actual_origins, directions, 30.)
        _, uploaded_i, _ = oracle(uploaded_source, actual_origins, directions, 30.)
        with RayRenderer(source) as renderer:
            origins = torch.as_tensor(actual_origins, device=renderer.device)
            ray_dirs = torch.as_tensor(directions, device=renderer.device)
            hits = renderer.trace(origins, ray_dirs, 30.)
            matrices = poses_to_tensor((Pose(tuple(float(v) for v in origin.astype(np.float32))),))
            depth = process_samples(hits, origins, ray_dirs, matrices, config)
            perceived, target_raw = process_camera_perception(
                depth, depth.raw, matrices, config, source, PerceptionConfig(0.), diagnostics=True)
            warp_hit = host(hits.object_index) >= 0
            torch_hit = host(target_raw.subray_object_index).reshape(-1) >= 0
            warp_valid = np.isfinite(host(depth.subray_range_m).reshape(-1))
            target_valid = torch_hit & (host(target_raw.subray_distance_m).reshape(-1) >= config.min_range_m)
            missed_denominator = warp_valid & ~target_valid
            mismatch = warp_hit != torch_hit
            row = {"case": case_index, "origin_m": origin.tolist(), "center_m": center.tolist(),
                   "radius_m": radius, "nominal_tangent_angle_rad": angle,
                   "angle_offsets_rad": list(offsets), "input_directions_float32": directions.tolist(),
                   "public_double_hit": (ideal_i >= 0).tolist(),
                   "uploaded_geometry_double_hit": (uploaded_i >= 0).tolist(),
                   "warp_hit": warp_hit.tolist(), "torch_target_hit": torch_hit.tolist(),
                   "warp_distance_m": host(hits.distance_m).tolist(),
                   "torch_target_distance_m": host(target_raw.subray_distance_m).reshape(-1).tolist(),
                   "unoccluded_samples": int(host(perceived.unoccluded_samples)[0]),
                   "visible_samples": int(warp_valid.sum()),
                   "visible_fraction": float(host(perceived.visible_fraction)[0]),
                   "fraction_above_one": bool(host(perceived.visible_fraction)[0] > 1.),
                   "denominator_missed_indices": np.flatnonzero(missed_denominator).tolist(),
                   "hit_disagreement_indices": np.flatnonzero(mismatch).tolist()}
            records.append(row)
            for i in np.flatnonzero(mismatch | missed_denominator):
                discrepancies.append({"case": case_index, "ray": int(i), "offset_rad": offsets[i],
                                      "direction_float32": directions[i].tolist(),
                                      "warp_hit": bool(warp_hit[i]), "torch_hit": bool(torch_hit[i]),
                                      "denominator_missed": bool(missed_denominator[i])})
    return {"status": "observed" if discrepancies else "passed",
            "scope": "nominal tangent and both angular sides, then quantized float32; double oracles distinguish public and uploaded geometry; no ideal nextafter correctness guarantee",
            "case_count": len(records), "discrepancy_count": len(discrepancies),
            "any_fraction_above_one": any(row["fraction_above_one"] for row in records),
            "any_denominator_missed": any(row["denominator_missed_indices"] for row in records),
            "witnesses": discrepancies, "cases": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=str(ROOT / "tests/fixtures/sensor-review-pillars-20.obj"))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--results", default=str(ROOT / "outputs/ray_unified/runtime-probe-stream-final.json"))
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    source_paths = ("devtools/ray_rendering.py", "devtools/_ray_budget.py", "devtools/sensors.py",
                    "devtools/_ray_processing.py", "devtools/perception.py", "devtools/config.py")
    hashes = {name: digest(ROOT / name) for name in source_paths}
    report = {"status": "running", "source_sha256": hashes, "groups": []}
    renderer = None
    try:
        config = load_config(ROOT / "configs/devtools.json")
        cpu_poses = make_poses(args.batch)
        first = cpu_poses[0]
        scene = Scene.from_obj(args.map, config, first, first.body_to_world((3., 0., 0.)), t_ref=12.5)
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        renderer = RayRenderer(snapshot)
        poses = poses_to_tensor(cpu_poses)
        for name, action in (("actual_runtime_fraction_chunking", lambda: real_budget_probe(renderer, snapshot, config, poses)),
                             ("pose_cross_stream_storage_lifetime", lambda: storage_lifetime_probe(renderer, snapshot, config)),
                             ("warp_vs_torch_tangent_spheres", tangent_probe)):
            try:
                result = action()
                report["groups"].append({"name": name, **result})
                print(name + ": " + result["status"], flush=True)
            except Exception as error:
                report["groups"].append({"name": name, "status": "failed", "error": str(error),
                                          "traceback": traceback.format_exc()})
                print(name + ": failed: " + str(error), flush=True)
        report["status"] = "failed" if any(row["status"] == "failed" for row in report["groups"]) else "passed"
        report["needs_attention"] = any(row.get("discrepancy_count", 0) for row in report["groups"])
    finally:
        if renderer is not None:
            renderer.close()
        report["formal_source_unchanged"] = all(digest(ROOT / name) == hashes[name] for name in source_paths)
        destination = Path(args.results)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
