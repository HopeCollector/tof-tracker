"""Additional all-layer parity check, run in the experiment Docker image.

This script is deliberately outside the frozen timing-source directory. It was
added after timing started; it changes no renderer, worker, or timing result.
For each scene/workload it renders the reference at max(B) once, then compares
every batch size against the identical deterministic prefix. No timing claims
are derived from this validation, whose final comparisons download GPU images.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback


def _exact(comparison):
    exact = (comparison["category_disagreements"] == 0
             and comparison["identity_disagreements"] == 0
             and comparison["finite_max_abs_m"] in (None, 0.))
    camera = comparison.get("camera")
    if camera:
        exact &= (camera["mask_disagreements"] == 0
                  and camera["hint_disagreements"] == 0
                  and camera["fraction_max_abs"] == 0.
                  and camera["count_disagreements"] == 0)
    return bool(exact)


def _reference_raw(sensor, case, np, target_only=False):
    frames = [sensor._rasterize(case.config, pose, target_only=target_only)
              for pose in case.poses]
    return {"range_m": np.stack([frame[0] for frame in frames]),
            "object_index": np.stack([frame[1] for frame in frames])}


def _workload(scene, metadata, mode, batches, modules, device):
    common, fixtures, GLBatchRenderer, MeshSensor, torch, np = modules
    config = getattr(fixtures.CONFIG.sensors, mode)
    poses = fixtures.make_poses(max(batches), mode)
    case = common.Case("all_batches", f"pillars_{metadata['pillars']}_{mode}",
                       scene, config, poses, mode == "camera",
                       fixtures.CONFIG.sensors.min_visible_fraction)
    bounds = scene.get_geometry(exclude_ids={"vehicle"}).map_bounds_m
    # Public reference and raw reference are independent sensor calls. Keeping
    # their max(B) results and slicing known prefixes avoids repeated work.
    with MeshSensor(scene) as reference:
        object_ids = tuple(reference.object_ids)
        public = common._public_frames(reference, case)
        raw_reference = _reference_raw(reference, case, np)
        target_reference = _reference_raw(reference, case, np, True) if case.camera else None
        reference_info = dict(reference.renderer_info)
    _, matrices = common._pose_matrices(poses, device)
    renderer = GLBatchRenderer(scene, device=device)
    records = []
    try:
        if tuple(renderer.object_ids) != object_ids:
            raise AssertionError("public geometry object ID order differs")
        shared_geometry = renderer.geometry_info
        for batch in sorted(set(batches)):
            record = {"pillars": metadata["pillars"], "mode": mode, "batch_views": batch,
                      "status": "running", "scene": metadata, "config": asdict(config),
                      "reference_renderer": reference_info}
            try:
                renderer.prepare(batch, include_target=case.camera)
                before = renderer.geometry_info
                pose_batch = matrices[:batch]
                raw = renderer.render(pose_batch, config)
                target = renderer.render(pose_batch, config, target_only=True) if case.camera else None
                output = common._post(raw, target, pose_batch, case, bounds,
                                      object_ids.index("target"))
                after = renderer.geometry_info
                record["output_tensors"] = {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "device": str(value.device)} for key, value in output.items()}
                shape = (batch, config.height, config.width)
                output_contract = (raw[0].shape == raw[1].shape == shape
                                   and raw[0].dtype == torch.float32
                                   and raw[1].dtype == torch.int64
                                   and all(value.is_cuda and value.device == pose_batch.device
                                           for value in output.values()))
                expected_draws = 2 if case.camera else 1
                record["draws_per_action"] = (after["instanced_draw_count"]
                                              - before["instanced_draw_count"])
                record["geometry_buffer_bytes"] = after["scene_geometry_buffer_bytes"]
                record["geometry_uploads"] = after["geometry_uploads"]
                geometry_contract = (
                    after["scene_geometry_buffer_name"] == shared_geometry["scene_geometry_buffer_name"]
                    and after["scene_geometry_buffer_bytes"] == shared_geometry["scene_geometry_buffer_bytes"]
                    and after["geometry_uploads"] == 1)
                record["gpu_and_geometry_contract_passed"] = bool(
                    output_contract and geometry_contract and record["draws_per_action"] == expected_draws)
                expected = {key: value[:batch] for key, value in public.items()}
                record["public_parity"] = common.compare(common._host(output), expected,
                                                         atol_m=common.RASTER_ATOL_M)
                raw_host = common._host({"range_m": raw[0], "object_index": raw[1]})
                record["raw_parity"] = common.compare(
                    raw_host, {key: value[:batch] for key, value in raw_reference.items()},
                    atol_m=common.RASTER_ATOL_M)
                comparisons = [record["public_parity"], record["raw_parity"]]
                if target is not None:
                    target_host = common._host({"range_m": target[0], "object_index": target[1]})
                    record["target_raw_parity"] = common.compare(
                        target_host, {key: value[:batch] for key, value in target_reference.items()},
                        atol_m=common.RASTER_ATOL_M)
                    comparisons.append(record["target_raw_parity"])
                record["exact_all_outputs"] = all(_exact(comparison) for comparison in comparisons)
                record["status"] = ("passed" if record["gpu_and_geometry_contract_passed"]
                                    and all(comparison["status"] == "passed" for comparison in comparisons)
                                    else "failed")
                del output, raw, target
            except Exception as error:
                record.update(status="error", error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
            records.append(record)
            print(json.dumps({key: record.get(key) for key in
                              ("pillars", "mode", "batch_views", "status", "exact_all_outputs")}), flush=True)
            gc.collect()
    finally:
        renderer.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pillars", type=int, nargs="+", default=[20, 500, 5000])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256])
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches):
        parser.error("batch sizes must be positive")
    sys.path.insert(0, str(args.root.resolve()))
    if args.map_dir is not None:
        import os
        os.environ["PYTORCH3D_EVAL_MAP_DIR"] = str(args.map_dir.resolve())
    import numpy as np
    import torch
    from devtools.sensors import MeshSensor
    from experiments.opengl_batch_eval.backend import GLBatchRenderer
    from experiments.opengl_batch_eval.run_eval import source_hashes
    from experiments.pytorch3d_render_eval import fixtures, validate as common
    modules = (common, fixtures, GLBatchRenderer, MeshSensor, torch, np)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    report = {"status": "running", "backend": "gl_batch", "cases": [],
              "scope": "additional all-batch/all-layer exact reference comparison; no timing",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": source_hashes(), "arguments": {
                  key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()},
              "acceptance": {"finite_atol_m": common.RASTER_ATOL_M,
                             "category_disagreements_allowed": 0, "identity_disagreements_allowed": 0,
                             "camera_mask_hint_count_disagreements_allowed": 0,
                             "camera_fraction_atol": 1e-15},
              "limitations": ["reference behavior, not an independent analytic correctness proof",
                              "independent oracles and single/reverse/stream contracts are covered by preflight",
                              "reference uses original f64 Pose; batch input is quantized to f32",
                              "added after frozen timing began; renderer/worker sources are unchanged"]}
    try:
        for pillars in args.pillars:
            scene, metadata = fixtures.load_scene(pillars)
            for mode in ("tof", "camera"):
                try:
                    report["cases"].extend(_workload(scene, metadata, mode, args.batches, modules, args.device))
                except Exception as error:
                    report["cases"].append({"pillars": pillars, "mode": mode, "status": "error",
                                            "error_type": type(error).__name__, "error": str(error),
                                            "traceback": traceback.format_exc()})
        report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
        report["exact_all_outputs"] = all(case.get("exact_all_outputs", False) for case in report["cases"])
        report["case_count"] = len(report["cases"])
        report["passed_cases"] = sum(case["status"] == "passed" for case in report["cases"])
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(common._json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "case_count": report.get("case_count"),
                      "exact_all_outputs": report.get("exact_all_outputs"),
                      "output": str(args.output)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
