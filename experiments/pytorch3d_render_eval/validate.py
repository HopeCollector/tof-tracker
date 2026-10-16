"""Bounded correctness experiment; run only in the graphics/CUDA Docker image.

Two comparisons stay separate: GL raw images replayed through CUDA postprocess
against public MeshSensor frames, and the complete P3D path against those same
frames. Original float64 Pose matrices isolate the first comparison; P3D uses
its actual float32 input poses throughout. This is validation, not a timer.
GPU outputs are copied to CPU only at the final comparison boundary.

An unsuccessful case is retained in JSON and does not prevent later cases or
the performance runner from running. Inspect JSON ``status``; CLI exit status
is zero after a result file was successfully written, including failed checks.
"""

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from devtools.config import DepthConfig
from devtools.scene import Pose
from devtools.sensors import MeshSensor
from experiments.pytorch3d_render_eval.backend import P3DRenderer
from experiments.pytorch3d_render_eval.fixtures import (
    CONFIG, depth_comparison, load_scene, make_poses,
)
from experiments.pytorch3d_render_eval.postprocess import (
    postprocess_camera, postprocess_depth,
)


RASTER_ATOL_M = 2e-5


@dataclass
class Case:
    group: str
    name: str
    scene: object
    config: DepthConfig
    poses: list
    camera: bool = False
    threshold: float = .2
    oracle: object = None
    metadata: object = None


def _reference_helpers():
    """Reuse independent fixtures/oracles, never the verification script main."""
    spec = importlib.util.spec_from_file_location(
        "sensor_independent_oracles", ROOT / "tests/verify_sensors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pose_matrices(poses, device):
    matrices = np.repeat(np.eye(4)[None], len(poses), axis=0)
    for index, pose in enumerate(poses):
        matrices[index, :3, :3] = pose.as_transform().rotation
        matrices[index, :3, 3] = pose.position_m
    precise = torch.as_tensor(matrices, dtype=torch.float64, device=device)
    return precise, precise.to(dtype=torch.float32)


def _host(output):
    # The only device-to-host conversion point for experimental outputs.
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


def _category(array):
    return np.where(np.isfinite(array), 0,
                    np.where(np.isposinf(array), 1, np.where(np.isnan(array), 2, 3)))


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _json_safe(value):
    """Retain bad numerical outputs as evidence instead of losing the report."""
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def compare(actual, expected, *, atol_m):
    """No silhouette or nonfinite disagreements are removed from acceptance."""
    result = depth_comparison(actual["range_m"], expected["range_m"],
                              actual["object_index"], expected["object_index"])
    result["finite_atol_m"] = atol_m
    result["forbidden_negative_inf_actual"] = int(np.isneginf(actual["range_m"]).sum())
    result["forbidden_negative_inf_expected"] = int(np.isneginf(expected["range_m"]).sum())
    result["boundary_pixels"] = result["pixels"] - result["interior_pixels"]
    result["boundary_identity_disagreements"] = (
        result["identity_disagreements"] - result["interior_identity_disagreements"])
    a, e = actual["range_m"], expected["range_m"]
    common = np.isfinite(a) & np.isfinite(e)
    errors = np.zeros(a.shape, dtype=np.float64)
    errors[common] = np.abs(a[common].astype(np.float64) - e[common].astype(np.float64))
    mismatch = ((_category(a) != _category(e)) | (errors > atol_m)
                | (actual["object_index"] != expected["object_index"]))
    result["first_disagreements"] = [
        {"pixel_brc": index.tolist(), "actual_m": _number(a[tuple(index)]),
         "expected_m": _number(e[tuple(index)]),
         "actual_id": int(actual["object_index"][tuple(index)]),
         "expected_id": int(expected["object_index"][tuple(index)])}
        for index in np.argwhere(mismatch)[:12]
    ]
    passed = (result["category_disagreements"] == 0
              and result["identity_disagreements"] == 0
              and result["forbidden_negative_inf_actual"] == 0
              and result["forbidden_negative_inf_expected"] == 0
              and not np.any(errors > atol_m))
    if "target_mask" in expected:
        mask_diff = actual["target_mask"] != expected["target_mask"]
        hint_error = np.abs(actual["target_hint"].astype(np.float64)
                            - expected["target_hint"].astype(np.float64))
        fraction_error = np.abs(actual["visible_fraction"] - expected["visible_fraction"])
        count_diff = actual["unoccluded_pixels"] != expected["unoccluded_pixels"]
        nonfinite_hint = int(np.count_nonzero(~np.isfinite(actual["target_hint"])))
        nonfinite_fraction = int(np.count_nonzero(~np.isfinite(actual["visible_fraction"])))
        result["camera"] = {
            "mask_disagreements": int(mask_diff.sum()),
            "hint_max_abs": float(hint_error.max()),
            "hint_disagreements": int(np.count_nonzero(hint_error)),
            "fraction_max_abs": float(fraction_error.max()),
            "count_disagreements": int(count_diff.sum()),
            "forbidden_nonfinite_hint_values": nonfinite_hint,
            "forbidden_nonfinite_fraction_values": nonfinite_fraction,
            "actual_visible_pixels": actual["target_mask"].sum(axis=(-2, -1)).tolist(),
            "expected_visible_pixels": expected["target_mask"].sum(axis=(-2, -1)).tolist(),
            "actual_unoccluded_pixels": actual["unoccluded_pixels"].tolist(),
            "expected_unoccluded_pixels": expected["unoccluded_pixels"].tolist(),
            "actual_fraction": actual["visible_fraction"].tolist(),
            "expected_fraction": expected["visible_fraction"].tolist(),
            "actual_hint": actual["target_hint"].tolist(),
            "expected_hint": expected["target_hint"].tolist(),
        }
        passed &= (not np.any(mask_diff) and not np.any(hint_error)
                   and not np.any(fraction_error > 1e-15) and not np.any(count_diff)
                   and nonfinite_hint == 0 and nonfinite_fraction == 0)
    result["status"] = "passed" if passed else "failed"
    return result


def _public_frames(sensor, case):
    if case.camera:
        frames = [sensor.camera(case.config, pose, case.threshold) for pose in case.poses]
        return {"range_m": np.stack([f.depth.range_m for f in frames]),
                "object_index": np.stack([f.depth.object_index for f in frames]),
                "target_mask": np.stack([f.target_mask for f in frames]),
                "target_hint": np.stack([f.target_hint for f in frames]),
                "visible_fraction": np.array([f.visible_fraction for f in frames]),
                "unoccluded_pixels": np.array([f.unoccluded_pixels for f in frames])}
    frames = sensor.sample_many(case.config, case.poses)
    return {"range_m": np.stack([f.range_m for f in frames]),
            "object_index": np.stack([f.object_index for f in frames])}


def _raw_gl(sensor, case, device, target_only=False):
    frames = [sensor._rasterize(case.config, pose, target_only=target_only)
              for pose in case.poses]
    return tuple(torch.as_tensor(np.stack([frame[i] for frame in frames]), device=device)
                 for i in range(2))


def _post(raw, target, matrices, case, bounds, target_index):
    output = {"range_m": postprocess_depth(*raw, matrices, case.config, bounds),
              "object_index": raw[1]}
    if case.camera:
        output.update(postprocess_camera(output["range_m"], raw[1], *target,
                                         matrices, case.config, bounds,
                                         target_index, case.threshold))
    return output


def _p3d(renderer, matrices, case, bounds):
    renderer.prepare(matrices.shape[0])
    raw = renderer.render(matrices, case.config)
    target = renderer.render(matrices, case.config, target_only=True) if case.camera else None
    return _post(raw, target, matrices, case, bounds, renderer.object_ids.index("target"))


def _consistency(render, matrices, batched, *, atol_m):
    # Per-view iteration belongs to this independent validator, never the CUDA
    # postprocessor or measured batch path. Do not confuse it with GPU batching.
    singles = [render(matrices[i:i+1]) for i in range(matrices.shape[0])]
    single = {key: torch.cat([row[key] for row in singles]) for key in batched}
    reverse = torch.arange(matrices.shape[0] - 1, -1, -1, device=matrices.device)
    permuted = render(matrices.index_select(0, reverse))
    restored = {key: value.index_select(0, reverse) for key, value in permuted.items()}
    expected = _host(batched)
    return {"batch_vs_single": compare(expected, _host(single), atol_m=atol_m),
            "batch_vs_reversed_restored": compare(expected, _host(restored), atol_m=atol_m)}


def _range_oracle(expected, expected_ids=None, atol=RASTER_ATOL_M):
    expected = np.asarray(expected)

    def check(output, object_ids):
        actual = output["range_m"]
        common = np.isfinite(actual) & np.isfinite(expected)
        error = np.abs(actual[common].astype(float) - expected[common])
        category_errors = int(np.count_nonzero(_category(actual) != _category(expected)))
        negative_inf = int(np.isneginf(actual).sum())
        id_errors = 0
        if expected_ids is not None:
            names = np.array((*object_ids, "miss"), dtype=object)
            actual_names = names[np.where(output["object_index"] < 0,
                                          len(object_ids), output["object_index"])]
            id_errors = int(np.count_nonzero(actual_names != expected_ids))
        maximum = float(error.max()) if error.size else None
        return {"status": "passed" if category_errors == 0 and id_errors == 0 and negative_inf == 0
                and (maximum is None or maximum <= atol) else "failed",
                "category_disagreements": category_errors,
                "identity_disagreements": id_errors, "finite_max_abs_m": maximum,
                "forbidden_negative_inf_actual": negative_inf,
                "finite_atol_m": atol,
                "expected_first_pixel_m": _number(expected.reshape(-1)[0]),
                "actual_first_pixel_m": _number(actual.reshape(-1)[0])}
    return check


def _camera_oracle(ref, config, poses, center, radius, kind, threshold):
    def check(output, object_ids):
        checks, details = [], []
        for index, pose in enumerate(poses):
            mask = output["target_mask"][index]
            count = int(output["unoccluded_pixels"][index])
            fraction = float(output["visible_fraction"][index])
            visible = int(mask.sum())
            checks.append(fraction == (visible / count if count else 0.))
            detected = count > 0 and visible > 0 and fraction >= threshold
            expected_hint = np.zeros(5, dtype=np.float32)
            if detected:
                rows, cols = np.nonzero(mask)
                expected_hint[:] = (1, cols.min()/config.width, rows.min()/config.height,
                                    (cols.max()+1)/config.width, (rows.max()+1)/config.height)
            checks.append(np.array_equal(output["target_hint"][index], expected_hint))
            detail = {"visible_pixels": visible, "unoccluded_pixels": count,
                      "fraction": fraction, "half_open_hint": expected_hint.tolist()}
            if kind in ("clear", "half"):
                analytic, ranges, discriminant = ref.analytic_sphere(
                    config, center, radius, np.asarray(pose.position_m))
                expected_mask = analytic
                if kind == "half":
                    expected_mask = analytic & (ref.reference_rays(config)[..., 1] < 0)
                    checks.append(.48 <= fraction <= .52)
                else:
                    checks.append(fraction == 1 and count == visible and count > 0)
                disagreement = int(np.count_nonzero(mask ^ expected_mask))
                # The target is a fixed inscribed triangle mesh, not an exact
                # sphere. This explicit oracle allowance is not used in parity.
                checks.append(disagreement <= 4)
                interior = mask & analytic & (discriminant > .1)
                error = output["range_m"][index][interior] - ranges[interior]
                checks.append(bool(error.size) and float(error.min()) >= -2e-5
                              and float(error.max()) < .006)
                detail.update(analytic_silhouette_disagreements=disagreement,
                              inscribed_sphere_error_min_m=float(error.min()) if error.size else None,
                              inscribed_sphere_error_max_m=float(error.max()) if error.size else None)
            elif kind == "full":
                checks.append(count > 0 and visible == 0 and fraction == 0)
            elif kind == "zero":
                checks.append(count == 0 and visible == 0 and fraction == 0)
            elif kind == "boundary":
                checks.append(count == visible == 1 and fraction == 1)
                distance = 5 - pose.position_m[0]
                checks.append(abs(float(output["range_m"][index, 0, 0]) - distance) < 2e-5)
                detail["analytic_boundary_distance_m"] = distance
            details.append(detail)
        return {"status": "passed" if all(checks) else "failed",
                "checks": len(checks), "failed_checks": sum(not value for value in checks),
                "views": details}
    return check


def _cases(ref):
    origin = ref.ORIGIN
    poses = [Pose((x, 0, 20)) for x in (0., .125, -.125)]
    one = DepthConfig(1, 1, 60, 60, 0, 8)
    wall = ("wall", ((4, 5), (-8, 8), (10, 30)))
    wall_scene = ref.scene((wall,))
    cases = []
    for declared in (True, False):
        config = DepthConfig(5, 3, 60, 40, 0, 10)
        angles = (0., math.pi/2, 0.)
        down = [Pose((0, 0, height), ref.quaternion(*angles)) for height in (4., 3.75, 4.125)]
        directions = np.array([[ref.rotate_euler(ray, *angles) for ray in row]
                               for row in ref.reference_rays(config)])
        expected = np.stack([p.position_m[2] / -directions[..., 2] for p in down])
        cases.append(Case("ground", f"ground_declared_{declared}", ref.scene(declared=declared),
                          config, down, oracle=_range_oracle(expected)))
    boxes = (("far_wall", ((6, 7), (-10, 10), (10, 30))),
             ("upper_left", ((3, 3.5), (.4, 1.8), (20.4, 21.8))))
    for label, angles, config, geometry in (
        ("asymmetric", (0., 0., 0.), DepthConfig(9, 7, 60, 45, 0, 12), boxes),
        ("full_rpy", tuple(map(math.radians, (18, -14, 12))),
         DepthConfig(7, 5, 30, 20, 0, 12), (wall,))):
        views = [Pose(p.position_m, ref.quaternion(*angles)) for p in poses]
        analytic = [ref.box_image(config, p.position_m, geometry, angles) for p in views]
        ids = np.stack([np.array(["map/" + box[0] for box in geometry], dtype=object)[row[1]]
                        for row in analytic])
        cases.append(Case("closed_boxes", label, ref.scene(geometry), config, views,
                          oracle=_range_oracle(np.stack([row[0] for row in analytic]), ids)))
    centered = [Pose((0, y, 20)) for y in (0., .125, -.125)]
    for label, config, value in (("inclusive_min", replace(one, min_range_m=4), 4.),
                                  ("inclusive_max", replace(one, max_range_m=4), 4.),
                                  ("too_close", replace(one, min_range_m=4.25), math.nan),
                                  ("beyond_max", replace(one, max_range_m=3.5), math.inf)):
        cases.append(Case("range_coverage", label, wall_scene, config, centered,
                          oracle=_range_oracle(np.full((3, 1, 1), value))))
    for declared in (True, False):
        cases.append(Case("range_coverage", f"empty_declared_{declared}",
                          ref.scene(declared=declared), one, centered,
                          oracle=_range_oracle(np.full((3, 1, 1), math.inf if declared else math.nan))))
    small = ref.scene(size=(10, 10, 10), target=(-3, -3, 5))
    cases.append(Case("range_coverage", "uncovered_empty_segment", small, one,
                      [Pose((0, 0, 5))], oracle=_range_oracle([[[math.nan]]])))
    within = ("within_domain", ((4, 4.5), (-1, 1), (4, 6)))
    cases.append(Case("range_coverage", "partial_domain_hit_and_outside_origin",
                      ref.scene((within,), size=(10, 10, 10), target=(-3, -3, 5)), one,
                      [Pose((0, 0, 5)), Pose((6, 0, 5), ref.quaternion(yaw=math.pi))],
                      oracle=_range_oracle([[[4.]], [[math.nan]]])))
    crossing = ("crossing_camera_plane", ((-1, .05), (-1, 1), (19, 21)))
    config = DepthConfig(3, 3, 60, 60, .1, 8)
    cases.append(Case("eye_plane", "near_occluder_crossing_eye", ref.scene((wall, crossing)),
                      config, [Pose(origin)], oracle=_range_oracle(
                          np.full((1, 3, 3), math.nan),
                          np.full((1, 3, 3), "map/crossing_camera_plane", dtype=object))))
    near_max = ("near_max_wall", ((12.000001, 12.5), (-8, 8), (10, 30)))
    cases.append(Case("numerical_regressions", "uncertain_near_max_hit", ref.scene((near_max,)),
                      replace(one, max_range_m=12), centered,
                      oracle=_range_oracle(np.full((3, 1, 1), math.nan),
                          np.full((3, 1, 1), "map/near_max_wall", dtype=object))))
    config = DepthConfig(64, 48, 120, 100, .05, 12)
    position = (-.5481651582549905, -1.4890260899279797, 11.00635970341665)
    quaternion = (-.28231015743329757, .26836274025745954,
                  -.45667413895892944, .799831948260142)
    q = np.asarray(quaternion) / math.hypot(*quaternion)
    ray = ref.reference_rays(config)[8, 55]
    direction = ray + 2 * np.cross(q[:3], np.cross(q[:3], ray) + q[3] * ray)
    witness_expected = (1 - position[0]) / direction[0]

    def witness(output, object_ids):
        value = float(output["range_m"][0, 8, 55])
        identity = int(output["object_index"][0, 8, 55])
        passed = (math.isfinite(value) and abs(value - witness_expected) < 2e-5
                  and identity == object_ids.index("map/wide_fov_wall"))
        return {"status": "passed" if passed else "failed", "pixel_brc": [0, 8, 55],
                "analytic_m": witness_expected, "actual_m": _number(value),
                "abs_error_m": abs(value - witness_expected) if math.isfinite(value) else None,
                "strict_abs_error_bound_m": 2e-5}
    cases.append(Case("numerical_regressions", "wide_fov_clipped_wall_witness",
                      ref.scene((("wide_fov_wall", ((1, 1.5), (-15, 15), (0, 20))),),
                                size=(40, 40, 20)), config,
                      [Pose(position, quaternion)], oracle=witness))
    config = DepthConfig(80, 60, 60, 45, .01, 15)
    center, radius = (6., 0., 20.), 1.2
    views = [Pose((x, 0, 20)) for x in (0., .125)]
    half = ("half_occluder", ((3, 3.2), (0, 3), (16, 24)))
    full = ("full_occluder", ((3, 3.2), (-3, 3), (16, 24)))
    for label, geometry, kind, threshold, depth in (
        ("clear", (), "clear", 1., config),
        ("half_detect", (half,), "half", .4, config),
        ("half_threshold_equal", (half,), "half", .5, config),
        ("half_reject", (half,), "half", .6, config),
        ("full_occlusion", (full,), "full", 0., config),
        ("zero_denominator", (), "zero", 0., replace(config, max_range_m=2))):
        cases.append(Case("camera_occlusion", label,
                          ref.scene(geometry, target=center, radius=radius), depth, views, True,
                          threshold, _camera_oracle(ref, depth, views, center, radius, kind, threshold)))
    views = [Pose((0, 0, 5)), Pose((6, 0, 5), ref.quaternion(yaw=math.pi))]
    cases.append(Case("camera_coverage", "target_outside_declared_domain",
                      ref.scene(size=(10, 10, 10), target=(6, 0, 5), radius=.25), one, views, True, 0.,
                      _camera_oracle(ref, one, views, (6, 0, 5), .25, "zero", 0.)))
    views = [Pose((x, 0, 5)) for x in (-.123, -.37)]
    cases.append(Case("camera_coverage", "target_surface_exact_domain_boundary",
                      ref.scene(size=(10, 10, 10), target=(5.5, 0, 5), radius=.5), one, views, True, 1.,
                      _camera_oracle(ref, one, views, (5.5, 0, 5), .5, "boundary", 1.)))
    return cases


@torch.no_grad()
def _validate_case(case, device, renderer_class=P3DRenderer, bin_capacity=None):
    record = {"group": case.group, "name": case.name, "status": "running",
              "config": asdict(case.config), "batch_size": len(case.poses),
              "camera": case.camera, "threshold": case.threshold if case.camera else None,
              "poses": [{"position_m": list(p.position_m),
                         "quaternion_xyzw": list(p.quaternion_xyzw)} for p in case.poses],
              "metadata": case.metadata}
    renderer = None
    try:
        snapshot = case.scene.get_geometry(exclude_ids={"vehicle"})
        bounds = snapshot.map_bounds_m
        precise, quantized = _pose_matrices(case.poses, device)
        with MeshSensor(case.scene) as sensor:
            record.update(renderer_info=sensor.renderer_info, object_ids=list(sensor.object_ids),
                          map_bounds_m=bounds)
            expected = _public_frames(sensor, case)
            raw = _raw_gl(sensor, case, device)
            target = _raw_gl(sensor, case, device, True) if case.camera else None
            target_index = sensor.object_ids.index("target")
            replay = _post(raw, target, precise, case, bounds, target_index)
            replay_host = _host(replay)
            record["gl_raw_postprocess_f64pose"] = compare(replay_host, expected, atol_m=0.)
            # Diagnostic only: quantizing a pose cannot be undone by .double().
            # This explicitly attributes any strict coverage-boundary changes.
            record["pose_quantization_diagnostic"] = compare(
                _host(_post(raw, target, quantized, case, bounds, target_index)), expected, atol_m=0.)

            def replay_slice(indices):
                full = tuple(value.index_select(0, indices) for value in raw)
                isolated = tuple(value.index_select(0, indices) for value in target) if target else None
                return _post(full, isolated, precise.index_select(0, indices), case, bounds, target_index)
            # Index matrices provide unique view labels; no CUDA-value inspection.
            indices = torch.arange(len(case.poses), device=device)[:, None, None]
            record["gl_postprocess_consistency"] = _consistency(
                lambda value: replay_slice(value.reshape(-1)), indices, replay, atol_m=0.)
            renderer = renderer_class(case.scene, device=device, max_faces_per_bin=bin_capacity)
            if tuple(renderer.object_ids) != tuple(sensor.object_ids):
                raise AssertionError("P3D and GL public geometry object order differs")
            p3d = _p3d(renderer, quantized, case, bounds)
            p3d_host = _host(p3d)
            record["p3d_vs_gl"] = compare(p3d_host, expected, atol_m=RASTER_ATOL_M)
            record["p3d_consistency"] = _consistency(
                lambda value: _p3d(renderer, value, case, bounds), quantized, p3d,
                atol_m=RASTER_ATOL_M)
            record["geometry"] = renderer.geometry_info
            if case.oracle is not None:
                record["independent_oracle"] = {
                    "public_gl": case.oracle(expected, sensor.object_ids),
                    "gl_raw_postprocess": case.oracle(replay_host, sensor.object_ids),
                    "p3d": case.oracle(p3d_host, sensor.object_ids)}
            checks = [record["gl_raw_postprocess_f64pose"], record["p3d_vs_gl"],
                      *record["gl_postprocess_consistency"].values(),
                      *record["p3d_consistency"].values(),
                      *record.get("independent_oracle", {}).values()]
            record["status"] = "passed" if all(row["status"] == "passed" for row in checks) else "failed"
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        if renderer is not None:
            renderer.release_batch()
    return record


def run_validation(pillars=(20, 500, 5000), device="cuda", backend="p3d", bin_capacity=None):
    """Return a JSON-compatible report; individual failures never stop the sweep."""
    report = {"status": "running", "device": str(device), "backend": backend,
              "bin_capacity": bin_capacity, "cases": [],
              "acceptance": {"raw_replay_finite_atol_m": 0.,
                             "p3d_finite_atol_m": RASTER_ATOL_M,
                             "category_disagreements_allowed": 0,
                             "identity_disagreements_allowed": 0,
                             "camera_mask_hint_count_disagreements_allowed": 0,
                             "camera_fraction_atol": 1e-15,
                             "pose_quantization_diagnostic_is_acceptance": False},
              "scope": "GL raw replay isolated with original float64 Pose matrices; P3D rendered/postprocessed with float32 poses; no performance inference",
              "limitations": ["P3D positive optical z clip may lose near-eye surfaces; failures remain visible",
                              "sphere analytic oracle explicitly accounts for its inscribed triangle mesh",
                              "final comparisons synchronize/copy outputs to CPU; not part of timing"]}
    try:
        torch.set_grad_enabled(False)
        renderer_class = P3DRenderer
        if backend == "p3d_conditioned":
            from experiments.pytorch3d_render_eval.conditioned_backend import P3DRenderer as renderer_class
        ref = _reference_helpers()
        cases = _cases(ref)
        for count in pillars:
            try:
                scene, info = load_scene(count)
                for mode, batch in (("tof", 8), ("camera", 4)):
                    cases.append(Case("maps", f"pillars_{count}_{mode}", scene,
                                      getattr(CONFIG.sensors, mode), make_poses(batch, mode),
                                      mode == "camera", CONFIG.sensors.min_visible_fraction,
                                      metadata=info))
            except Exception as error:
                report["cases"].append({"group": "maps", "name": f"pillars_{count}",
                                        "status": "error", "error": str(error),
                                        "traceback": traceback.format_exc()})
        for case in cases:
            result = _validate_case(case, device, renderer_class, bin_capacity)
            report["cases"].append(result)
            print(json.dumps({key: result[key] for key in ("group", "name", "status")}), flush=True)
            gc.collect()
        report["groups"] = {
            group: {status: sum(row["group"] == group and row["status"] == status
                                for row in report["cases"])
                    for status in ("passed", "failed", "error")}
            for group in sorted({row["group"] for row in report["cases"]})}
        report["status"] = "passed" if all(row["status"] == "passed" for row in report["cases"]) else "failed"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    return _json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pillars", nargs="+", type=int, default=[20, 500, 5000])
    parser.add_argument("--backend", choices=("p3d", "p3d_conditioned"), default="p3d")
    parser.add_argument("--bin-capacity", default=None)
    args = parser.parse_args()
    capacity = args.bin_capacity
    if capacity is not None and capacity != "pytorch3d_default":
        capacity = int(capacity)
    result = run_validation(args.pillars, backend=args.backend, bin_capacity=capacity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
