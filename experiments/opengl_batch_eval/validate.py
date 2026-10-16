"""Validate the isolated OpenGL batch backend in Docker, outside timing.

Fixtures, independent oracles, comparisons, and CUDA postprocessing are reused
from the previous experiment.  Reference and experimental GL contexts are
opened sequentially.  Public-sensor parity and independent geometric oracles
are reported separately: a pre-existing reference failure is not silently
called a batch regression.  Result downloads happen only at comparison time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from devtools.scene import Pose
from devtools.sensors import MeshSensor
from experiments.pytorch3d_render_eval import validate as common
from experiments.pytorch3d_render_eval.fixtures import CONFIG, load_scene, make_poses


def _backend_class():
    # Lazy import lets the runner describe/report setup errors before loading GL.
    from experiments.opengl_batch_eval.backend import GLBatchRenderer
    return GLBatchRenderer


def _tensor_contract(raw, matrices, config):
    distance, identity = raw
    shape = (matrices.shape[0], config.height, config.width)
    assert isinstance(distance, torch.Tensor) and isinstance(identity, torch.Tensor)
    assert distance.device == identity.device == matrices.device
    assert distance.is_cuda and identity.is_cuda
    assert distance.shape == identity.shape == shape
    assert distance.dtype == torch.float32 and identity.dtype == torch.int64
    return {"distance": {"shape": list(distance.shape), "dtype": str(distance.dtype),
                         "device": str(distance.device)},
            "identity": {"shape": list(identity.shape), "dtype": str(identity.dtype),
                         "device": str(identity.device)}}


def _render(renderer, matrices, case, bounds):
    renderer.prepare(matrices.shape[0], include_target=case.camera)
    raw = renderer.render(matrices, case.config)
    _tensor_contract(raw, matrices, case.config)
    target = renderer.render(matrices, case.config, target_only=True) if case.camera else None
    if target is not None:
        _tensor_contract(target, matrices, case.config)
    output = common._post(raw, target, matrices, case, bounds,
                          renderer.object_ids.index("target"))
    return output, raw


@torch.no_grad()
def _validate_case(case, device):
    record = {"group": case.group, "name": case.name, "status": "running",
              "config": asdict(case.config), "batch_size": len(case.poses),
              "camera": case.camera, "threshold": case.threshold if case.camera else None,
              "poses": [{"position_m": list(p.position_m),
                         "quaternion_xyzw": list(p.quaternion_xyzw)} for p in case.poses],
              "metadata": case.metadata}
    renderer = None
    try:
        bounds = case.scene.get_geometry(exclude_ids={"vehicle"}).map_bounds_m
        precise, quantized = common._pose_matrices(case.poses, device)
        with MeshSensor(case.scene) as sensor:
            object_ids = tuple(sensor.object_ids)
            record.update(reference_renderer_info=sensor.renderer_info,
                          object_ids=list(object_ids), map_bounds_m=bounds)
            expected = common._public_frames(sensor, case)
            raw_reference = common._raw_gl(sensor, case, device)
            target_reference = common._raw_gl(sensor, case, device, True) if case.camera else None
        # No reference GL context remains current when the batch context opens.
        replay = common._post(raw_reference, target_reference, precise, case, bounds,
                              object_ids.index("target"))
        replay_host = common._host(replay)
        record["gl_raw_postprocess_f64pose"] = common.compare(replay_host, expected, atol_m=0.)
        record["pose_quantization_diagnostic"] = common.compare(
            common._host(common._post(raw_reference, target_reference, quantized, case,
                                     bounds, object_ids.index("target"))), expected, atol_m=0.)
        renderer = _backend_class()(case.scene, device=device)
        assert tuple(renderer.object_ids) == object_ids, "public geometry object order changed"
        output, raw = _render(renderer, quantized, case, bounds)
        record["cuda_output_contract"] = _tensor_contract(raw, quantized, case.config)
        actual = common._host(output)
        record["batch_vs_public_gl"] = common.compare(actual, expected, atol_m=common.RASTER_ATOL_M)
        record["batch_raw_vs_reference_raw"] = common.compare(
            common._host({"range_m": raw[0], "object_index": raw[1]}),
            common._host({"range_m": raw_reference[0], "object_index": raw_reference[1]}),
            atol_m=common.RASTER_ATOL_M)
        record["batch_consistency"] = common._consistency(
            lambda matrices: _render(renderer, matrices, case, bounds)[0],
            quantized, output, atol_m=common.RASTER_ATOL_M)
        record["geometry"] = renderer.geometry_info
        checks = [record["gl_raw_postprocess_f64pose"], record["batch_vs_public_gl"],
                  record["batch_raw_vs_reference_raw"], *record["batch_consistency"].values()]
        if case.oracle is not None:
            record["independent_oracle"] = {
                "public_gl": case.oracle(expected, object_ids),
                "gl_raw_postprocess": case.oracle(replay_host, object_ids),
                "opengl_batch": case.oracle(actual, object_ids)}
            baseline = record["independent_oracle"]["public_gl"]["status"]
            batch = record["independent_oracle"]["opengl_batch"]["status"]
            record["independent_oracle_regression"] = baseline == "passed" and batch != "passed"
        record["status"] = ("passed" if all(row["status"] == "passed" for row in checks)
                            and not record.get("independent_oracle_regression", False) else "failed")
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        if renderer is not None:
            renderer.close()
    return record


def _raw_host(raw):
    return common._host({"range_m": raw[0], "object_index": raw[1]})


@torch.no_grad()
def _runtime_contract(case, device):
    """Check live CUDA pose values, output ownership, streams, and one VBO."""
    record = {"group": "runtime_contract", "name": "cuda_pose_storage_and_stream",
              "status": "running"}
    renderer = None
    try:
        precise, matrices = common._pose_matrices(case.poses, device)
        shifted_poses = [Pose((p.position_m[0] + .25, *p.position_m[1:]), p.quaternion_xyzw)
                         for p in case.poses]
        shifted = replace(case, poses=shifted_poses)
        with MeshSensor(case.scene) as sensor:
            expected_raw = common._raw_gl(sensor, shifted, device)
        renderer = _backend_class()(case.scene, device=device)
        geometry = []
        for batch in (1, 4, 16):
            renderer.prepare(batch, include_target=False)
            geometry.append(dict(renderer.geometry_info))
        renderer.prepare(len(case.poses), include_target=False)
        original = renderer.render(matrices, case.config)
        record["cuda_output_contract"] = _tensor_contract(original, matrices, case.config)
        saved = tuple(value.clone() for value in original)
        # This writes existing CUDA storage, rather than uploading a new pose or
        # giving the renderer a different Python tensor/storage pointer.
        matrices[:, 0, 3].add_(.25)
        changed = renderer.render(matrices, case.config)
        changed_saved = tuple(value.clone() for value in changed)
        current = torch.cuda.current_stream(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            alternate_stream = renderer.render(matrices, case.config)
        current.wait_stream(stream)
        matrices[:, 0, 3].sub_(.25)
        restored = renderer.render(matrices, case.config)
        record["pose_mutation_matches_reference"] = common.compare(
            _raw_host(changed_saved), _raw_host(expected_raw), atol_m=common.RASTER_ATOL_M)
        record["pose_restore"] = common.compare(_raw_host(restored), _raw_host(saved), atol_m=0.)
        record["earlier_output_not_overwritten"] = common.compare(
            _raw_host(original), _raw_host(saved), atol_m=0.)
        record["alternate_cuda_stream"] = common.compare(
            _raw_host(alternate_stream), _raw_host(changed_saved), atol_m=0.)
        before, after = _raw_host(saved), _raw_host(changed_saved)
        changed_pixels = int(np.count_nonzero(
            (before["range_m"] != after["range_m"])
            | (before["object_index"] != after["object_index"])))
        record["pose_mutation_changed_pixels"] = changed_pixels
        fields = ("scene_geometry_buffer_name", "scene_geometry_buffer_bytes", "geometry_uploads")
        record["shared_geometry_prepare_batches"] = [1, 4, 16]
        record["shared_geometry_snapshots"] = [{key: item.get(key) for key in fields}
                                                for item in geometry]
        geometry_passed = (all(all(key in item for key in fields) for item in geometry)
                           and all(item["geometry_uploads"] == 1 for item in geometry)
                           and all(item[fields[0]] == geometry[0][fields[0]]
                                   and item[fields[1]] == geometry[0][fields[1]] for item in geometry))
        record["shared_geometry_status"] = "passed" if geometry_passed else "failed"
        checks = [record[key]["status"] == "passed" for key in
                  ("pose_mutation_matches_reference", "pose_restore",
                   "earlier_output_not_overwritten", "alternate_cuda_stream")]
        record["status"] = "passed" if all(checks) and geometry_passed and changed_pixels else "failed"
        record["scope"] = ("CUDA tensor contract and behavior, not a complete transfer trace; "
                           "single thread/context with sequential stream use only")
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        if renderer is not None:
            renderer.close()
    return record


def run_validation(pillars=(20, 500, 5000), device="cuda"):
    report = {"status": "running", "device": str(device), "backend": "gl_batch",
              "cases": [], "acceptance": {
                  "raw_replay_finite_atol_m": 0., "batch_finite_atol_m": common.RASTER_ATOL_M,
                  "category_disagreements_allowed": 0, "identity_disagreements_allowed": 0,
                  "camera_mask_hint_count_disagreements_allowed": 0, "camera_fraction_atol": 1e-15,
                  "pose_quantization_diagnostic_is_acceptance": False,
                  "independent_oracle_policy": "report baseline failures separately; fail new regressions"},
              "scope": "fixed public scene snapshot, f32 CUDA poses, geometry parity plus runtime contracts",
              "limitations": ["not a timing run; comparisons synchronize/download outputs",
                              "existing GL independent-oracle failures remain visible",
                              "sphere oracle accounts for its inscribed triangle mesh",
                              "no simultaneous thread/context or stream concurrency claim"]}
    try:
        torch.set_grad_enabled(False)
        cases = common._cases(common._reference_helpers())
        for count in pillars:
            try:
                scene, info = load_scene(count)
                for mode, batch in (("tof", 8), ("camera", 4)):
                    cases.append(common.Case("maps", f"pillars_{count}_{mode}", scene,
                                             getattr(CONFIG.sensors, mode), make_poses(batch, mode),
                                             mode == "camera", CONFIG.sensors.min_visible_fraction,
                                             metadata=info))
            except Exception as error:
                report["cases"].append({"group": "maps", "name": f"pillars_{count}",
                                        "status": "error", "error": str(error)})
        for case in cases:
            result = _validate_case(case, device)
            report["cases"].append(result)
            print(json.dumps({key: result[key] for key in ("group", "name", "status")}), flush=True)
            gc.collect()
        contract = _runtime_contract(next(case for case in cases if case.name == "asymmetric"), device)
        report["cases"].append(contract)
        print(json.dumps({key: contract[key] for key in ("group", "name", "status")}), flush=True)
        report["groups"] = {
            group: {status: sum(row["group"] == group and row["status"] == status
                                for row in report["cases"])
                    for status in ("passed", "failed", "error")}
            for group in sorted({row["group"] for row in report["cases"]})}
        report["reference_oracle_failures"] = [row["name"] for row in report["cases"]
            if row.get("independent_oracle", {}).get("public_gl", {}).get("status") == "failed"]
        report["batch_oracle_failures"] = [row["name"] for row in report["cases"]
            if row.get("independent_oracle", {}).get("opengl_batch", {}).get("status") == "failed"]
        report["status"] = "passed" if all(row["status"] == "passed" for row in report["cases"]) else "failed"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    return common._json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pillars", nargs="+", type=int, default=[20, 500, 5000])
    args = parser.parse_args()
    result = run_validation(args.pillars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
