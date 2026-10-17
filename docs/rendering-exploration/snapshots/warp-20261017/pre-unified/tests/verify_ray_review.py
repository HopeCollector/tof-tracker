"""Real HTTP checks of the online Warp sensor review adapter, in Docker.

The service alone owns the GPU renderer. Client assertions use independent
float64 ray/triangle and sphere facts, and explicitly decode the UI boundary's
binary arrays. The original GL HTTP verification remains untouched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import traceback
from urllib.request import urlopen

import numpy as np

from devtools.config import DepthConfig
from devtools.mesh import MeshObject
from devtools.scene import GeometrySnapshot, Pose, SceneObject
from verify_ray_sensors import cpu_selection, independent_rays, oracle, sphere
from verify_sensor_review import (Service, BOX_FACES, array, check_map,
                                  close, geometry, pose_matrix, quaternion_matrix,
                                  verify_bootstrap, verify_request_validation,
                                  verify_transition, verify_upload_and_map_invalidation,
                                  views)


ROOT = Path(__file__).resolve().parents[1]
ATOL = 5e-5
TARGET_A = {"position_m": [3., 0., 1.5]}
TARGET_B = {"position_m": [3., 5., 1.5]}


def foreground_fixture():
    """A narrow front occluder creates mixed target/background camera pixels."""
    x0, x1, y0, y1, z0, z1 = 1.5, 1.6, .04, .16, .2, 3.8
    objects = [
        {"name": "ground", "kind": "ground",
         "vertices": [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)],
         "triangles": [(0, 1, 2), (0, 2, 3)]},
        {"name": "wall", "kind": "obstacle",
         "vertices": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                      (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
         "triangles": BOX_FACES},
    ]
    lines, offset = ["# unit metre", "# size_xyz 40 40 10"], 1
    for item in objects:
        lines.append("o " + item["name"])
        lines.extend("v " + " ".join(str(v) for v in vertex) for vertex in item["vertices"])
        lines.extend("f " + " ".join(str(v + offset) for v in face) for face in item["triangles"])
        offset += len(item["vertices"])
    return "\n".join(lines) + "\n", objects


def source_from_map(asset, target, radius):
    objects = []
    for item in asset["objects"]:
        name = item["id"].split("/", 1)[1]
        mesh = MeshObject(name, tuple(tuple(v) for v in item["vertices"]),
                          tuple(tuple(v) for v in item["triangles"]), item["kind"])
        objects.append(SceneObject(item["id"], item["kind"], mesh))
    if target is not None:
        objects.append(SceneObject("target", "target", sphere(tuple(target["position_m"]), radius)))
    bounds = asset["declared_bounds_m"]
    return GeometrySnapshot(tuple(objects), None,
                            None if bounds is None else tuple(tuple(v) for v in bounds))


def grid_to_samples(value, height, width, side):
    return value.reshape(height, side, width, side).transpose(0, 2, 1, 3).reshape(height, width, side * side)


def samples_to_grid(value, height, width, side):
    return value.reshape(height, width, side, side).transpose(0, 2, 1, 3).reshape(height * side, width * side)


def measurement_ranges(distance, identity, origins, directions, config, bounds):
    """Independent radial validity and convex-domain segment knowledge."""
    hit = identity >= 0
    limits = np.asarray(bounds, dtype=float) if bounds is not None else None
    if limits is None:
        known_hit = hit.copy()
        known_empty = np.zeros(len(distance), dtype=bool)
    else:
        origin_known = ((origins >= limits[:, 0]) & (origins <= limits[:, 1])).all(axis=-1)
        points = origins + distance[:, None] * directions
        known_hit = origin_known & ((points >= limits[:, 0] - ATOL)
                                    & (points <= limits[:, 1] + ATOL)).all(axis=-1)
        endpoints = origins + config.max_range_m * directions
        known_empty = origin_known & ((endpoints >= limits[:, 0])
                                      & (endpoints <= limits[:, 1])).all(axis=-1)
    valid = hit & known_hit & (distance >= config.min_range_m) & (distance <= config.max_range_m)
    ranges = np.where(valid, distance, np.nan)
    ranges = np.where(~hit & known_empty, np.inf, ranges)
    known = np.where(hit, known_hit, known_empty)
    return ranges, known


def verify_view(view, source, sensor_settings, *, target_present=False):
    config = DepthConfig(**view["config"])
    height, width, side = config.height, config.width, config.samples_per_axis
    shape = (height, width)
    subshape = (height * side, width * side)
    pose = Pose(tuple(view["pose"]["position_m"]), tuple(view["pose"]["quaternion_xyzw"]))
    origins, directions = independent_rays((pose,), config)
    distance, identity, _ = oracle(source, origins, directions, config.max_range_m)
    expected_subdistance = distance.reshape(height, width, side * side)
    expected_subidentity = identity.reshape(height, width, side * side)
    actual_subdistance = grid_to_samples(array(view["subray_distance"], "float32", subshape), height, width, side)
    actual_subidentity = grid_to_samples(array(view["subray_object_index"], "int32", subshape), height, width, side)
    close(actual_subdistance, expected_subdistance, ATOL)
    assert np.array_equal(actual_subidentity, expected_subidentity), view["id"]
    subconfig = view["subray_config"]
    assert (subconfig["height"], subconfig["width"]) == subshape
    assert subconfig["hfov_deg"] == config.hfov_deg and subconfig["vfov_deg"] == config.vfov_deg
    assert view["object_ids"] == list(obj.object_id for obj in source.objects)
    is_tof = view["id"].startswith("tof/")
    assert view["sampling"] == ("weighted_quantile_subrays" if is_tof else "nearest_subrays")
    selected = array(view["selected_subray"], "int32", shape)
    assert selected.min() >= 0 and selected.max() < side * side
    # Evaluate the reduction separately on the actual ray samples: this tests
    # the rule without making face-edge float32 tie breaking a geometry claim.
    if is_tof:
        expected_selected = np.asarray([
            cpu_selection(d, i, config.max_range_m,
                          sensor_settings["tof_depth_quantile"], sensor_settings["tof_signal_floor"])
            for d, i in zip(actual_subdistance.reshape(-1, side * side),
                            actual_subidentity.reshape(-1, side * side))]).reshape(shape)
    else:
        effective = np.where(actual_subidentity >= 0, actual_subdistance, config.max_range_m)
        expected_selected = effective.argmin(axis=-1)
    assert np.array_equal(selected, expected_selected), (view["id"], "subray reduction")
    selected_distance = np.take_along_axis(actual_subdistance, selected[..., None], axis=-1)[..., 0]
    selected_identity = np.take_along_axis(actual_subidentity, selected[..., None], axis=-1)[..., 0]
    raw = array(view["raw_distance"], "float32", shape)
    raw_identity = array(view["raw_object_index"], "int32", shape)
    close(raw, selected_distance, 0)
    assert np.array_equal(raw_identity, selected_identity)
    points = array(view["hit_point_sensor_m"], "float32", (*shape, 3))
    triangles = array(view["triangle_index"], "int32", shape)
    close(np.linalg.norm(points, axis=-1), raw, ATOL)
    missed = raw_identity < 0
    assert np.all(points[missed] == 0) and np.all(triangles[missed] == -1)
    if target_present:
        assert np.all(triangles[raw_identity == view["object_ids"].index("target")] == -1)
    ranges, known = measurement_ranges(distance, identity, origins, directions, config, source.map_bounds_m)
    ranges, known = ranges.reshape(height, width, side * side), known.reshape(height, width, side * side)
    expected_final = np.take_along_axis(ranges, selected[..., None], axis=-1)[..., 0]
    expected_final = np.where(known.all(axis=-1), expected_final, np.nan)
    measured = array(view["range_m"], "float32", shape)
    for classify in (np.isfinite, np.isnan, np.isposinf):
        assert np.array_equal(classify(measured), classify(expected_final)), (view["id"], classify.__name__)
    finite = np.isfinite(expected_final)
    close(measured[finite], expected_final[finite], ATOL)
    assert not np.isneginf(measured).any()
    return {"view": view["id"], "subrays": int(distance.size),
            "finite": int(np.isfinite(measured).sum()), "unknown": int(np.isnan(measured).sum()),
            "empty": int(np.isposinf(measured).sum())}


def verify_static_assets(service):
    sizes = {}
    for path, content_type in (("/", "text/html"), ("/app.js", "text/javascript"),
                               ("/style.css", "text/css"), ("/vendor/three.module.js", "text/javascript"),
                               ("/vendor/three.core.js", "text/javascript"),
                               ("/vendor/OrbitControls.js", "text/javascript"),
                               ("/vendor/three-LICENSE.txt", "text/plain"),
                               ("/vendor/manifest.json", "application/json")):
        with urlopen(service.url + path, timeout=15) as response:
            data = response.read()
            assert response.status == 200 and response.headers["Content-Type"].startswith(content_type)
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert len(data) > 50, path
        sizes[path] = len(data)
        if path == "/":
            assert b"/app.js" in data and b"/style.css" in data
        if path.endswith("manifest.json"):
            assert isinstance(json.loads(data), dict)
    for path in ("/../etc/passwd", "/%2e%2e/etc/passwd", "/unknown.js"):
        service.request("GET", path, expected="4xx")
    return {"assets": sizes, "unregistered_asset_paths_rejected": 3}


def verify_no_target_samples(service):
    loaded = service.upload()
    map_id = check_map(loaded, service.objects)
    sample = service.sample(map_id, request_id="no-target-rays")
    source = source_from_map(loaded["map"], None, service.bootstrap["config"]["target_radius_m"])
    info = geometry(sample)
    assert info["backend"] == "warp_mesh_bvh_analytic_spheres"
    assert info["trace_calls"] == 2 and info["geometry_uploads"] == 1
    assert info["cpu_image_readbacks"] == 0 and info["per_ray_python_calls"] == 0
    assert "target" not in info["object_ids"]
    settings = service.bootstrap["config"]["sensors"]
    details = [verify_view(view, source, settings) for view in views(sample).values()]
    camera = views(sample)["camera"]
    assert not array(camera["target_mask"], "uint8").any()
    assert not array(camera["target_coverage"], "float32").any()
    assert not np.any(camera["target_hint"])
    assert camera["visible_fraction"] == camera["unoccluded_pixels"] == camera["unoccluded_samples"] == 0
    assert "target_only_subray_distance" not in camera and "target_only_subray_object_index" not in camera
    repeated = service.sample(map_id, request_id="no-target-repeat")
    assert geometry(repeated)["trace_calls"] == 4
    assert repeated["scene_version"] == sample["scene_version"]
    assert repeated["sample_id"] != sample["sample_id"]
    return {"views": details, "queries_per_no_target_sample": 2}


def verify_mounted_rays(service):
    loaded = service.upload()
    map_id = check_map(loaded, service.objects)
    source = source_from_map(loaded["map"], None, service.bootstrap["config"]["target_radius_m"])
    body = {"position_m": [.5, -.3, 1.8],
            "quaternion_xyzw": [.1, -.2, .3, math.sqrt(1 - .1 ** 2 - .2 ** 2 - .3 ** 2)]}
    sample = service.sample(map_id, request_id="nonlevel-rays", body=body)
    settings = service.bootstrap["config"]["sensors"]
    mounts = settings["tof_mounts"]
    if isinstance(mounts, dict):
        mounts = [mounts[name] for name in ("right", "front", "left", "rear")]
    details = []
    for view, mount in zip(views(sample).values(), [*mounts, settings["camera_mount"]]):
        expected_matrix = pose_matrix(body) @ pose_matrix(mount)
        close(view["pose"]["matrix"], expected_matrix, 2e-7)
        close(quaternion_matrix(view["pose"]["quaternion_xyzw"]), expected_matrix[:3, :3], 2e-7)
        details.append(verify_view(view, source, settings))
    return {"nonlevel_world_poses_and_subrays": details}


def check_camera_coverage(view, source, settings):
    config = DepthConfig(**view["config"])
    height, width, side = config.height, config.width, config.samples_per_axis
    shape = (height, width, side * side)
    origins, directions = independent_rays((Pose(tuple(view["pose"]["position_m"]),
                                                tuple(view["pose"]["quaternion_xyzw"])),), config)
    target_index = view["object_ids"].index("target")
    target_source = GeometrySnapshot((source.get_object("target"),), source.t_ref, source.map_bounds_m)
    target_d, target_id, _ = oracle(target_source, origins, directions, config.max_range_m)
    # Target-only raw object indices still address the full scene object table.
    expected_target_i = np.where(target_id >= 0, target_index, -1)
    subshape = (height * side, width * side)
    close(array(view["target_only_subray_distance"], "float32", subshape),
          samples_to_grid(target_d, height, width, side), ATOL)
    assert np.array_equal(array(view["target_only_subray_object_index"], "int32", subshape),
                          samples_to_grid(expected_target_i, height, width, side))
    full_d, full_id, _ = oracle(source, origins, directions, config.max_range_m)
    full_ranges, _ = measurement_ranges(full_d, full_id, origins, directions, config, source.map_bounds_m)
    target_ranges, _ = measurement_ranges(target_d, target_id, origins, directions, config, source.map_bounds_m)
    visible = ((full_id == target_index) & np.isfinite(full_ranges)).reshape(shape)
    potential = ((target_id >= 0) & np.isfinite(target_ranges)).reshape(shape)
    coverage = visible.mean(axis=-1)
    close(array(view["target_coverage"], "float32", shape[:2]), coverage, 0)
    mask = array(view["target_mask"], "uint8", shape[:2]).astype(bool)
    assert np.array_equal(mask, visible.any(axis=-1))
    numerator, denominator = int(visible.sum()), int(potential.sum())
    assert view["unoccluded_samples"] == denominator
    assert view["unoccluded_pixels"] == int(potential.any(axis=-1).sum())
    expected_fraction = numerator / denominator if denominator else 0.
    assert abs(view["visible_fraction"] - expected_fraction) <= 1e-15
    detected = numerator > 0 and denominator > 0 and expected_fraction >= settings["min_visible_fraction"]
    if detected:
        rows, cols = np.nonzero(mask)
        expected_hint = (1., cols.min() / width, rows.min() / height,
                         (cols.max() + 1) / width, (rows.max() + 1) / height)
    else:
        expected_hint = (0., 0., 0., 0., 0.)
    close(view["target_hint"], expected_hint, 1e-7)
    representative = array(view["raw_object_index"], "int32", shape[:2])
    return {"visible_subrays": numerator, "target_only_subrays": denominator,
            "partial_pixels": int(np.count_nonzero((coverage > 0) & (coverage < 1))),
            "mask_pixels_with_other_representative": int(np.count_nonzero(mask & (representative != target_index)))}


def verify_target_motion_and_same_pass(service):
    text, expected_objects = foreground_fixture()
    loaded = service.upload("foreground.obj", text)
    map_id = check_map(loaded, expected_objects)
    first = service.sample(map_id, TARGET_A, "target-A")
    saved_response = json.dumps(first, sort_keys=True)
    moved = service.sample(map_id, TARGET_B, "target-B")
    restored = service.sample(map_id, TARGET_A, "target-A-again")
    initial = geometry(first)
    assert first["scene_version"] < moved["scene_version"] < restored["scene_version"]
    settings = service.bootstrap["config"]["sensors"]
    radius = service.bootstrap["config"]["target_radius_m"]
    source = source_from_map(loaded["map"], TARGET_A, radius)
    for i, result in enumerate((first, moved, restored)):
        info = geometry(result)
        assert info["trace_calls"] == 3 * (i + 1)
        assert info["geometry_uploads"] == initial["geometry_uploads"] == 1
        assert info["vertex_count"] == initial["vertex_count"]
        assert info["triangle_count"] == initial["triangle_count"]
        assert info["analytic_sphere_count"] == 1
        assert info["object_transform_updates"] == initial["object_transform_updates"] + i
        assert info["object_transform_upload_bytes"] == initial["object_transform_upload_bytes"] + 12 * i
        if "mesh_bvh_builds" in info:
            assert info["mesh_bvh_builds"] == 1
    for view in views(first).values():
        verify_view(view, source, settings, target_present=True)
    camera_a, camera_b, camera_again = (views(value)["camera"] for value in (first, moved, restored))
    details = check_camera_coverage(camera_a, source, settings)
    assert details["visible_subrays"] > 0 and details["partial_pixels"] > 0
    assert details["mask_pixels_with_other_representative"] > 0
    assert not array(camera_b["target_mask"], "uint8").any()
    for name in ("raw_distance", "raw_object_index", "range_m", "target_mask", "target_coverage",
                 "subray_distance", "subray_object_index", "selected_subray"):
        assert camera_a[name] == camera_again[name], name
    assert json.dumps(first, sort_keys=True) == saved_response
    return {**details, "queries_per_sample": 3, "transform_bytes_per_target_move": 12,
            "geometry_uploads": 1, "A_B_A_owned_response": "passed"}


def verify_concurrent_snapshots(service):
    loaded = service.upload()
    map_id = check_map(loaded, service.objects)
    service.sample(map_id, TARGET_A, "warm")
    targets = [TARGET_A, TARGET_B, TARGET_A, TARGET_B]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(service.sample, map_id, target, f"parallel-{index}")
                   for index, target in enumerate(targets)]
        results = [future.result() for future in futures]
    assert len({item["sample_id"] for item in results}) == 4
    counts = sorted(geometry(item)["trace_calls"] for item in results)
    assert counts == [6, 9, 12, 15]
    for index, (target, item) in enumerate(zip(targets, results)):
        assert item["request_id"] == f"parallel-{index}" and item["map_id"] == map_id
        assert bool(array(views(item)["camera"]["target_mask"], "uint8").any()) == (target == TARGET_A)
        assert item["scene_version"] == geometry(item)["scene_version"]
        assert geometry(item)["geometry_uploads"] == 1
    return {"concurrent_requests": 4, "serialized_query_counts": counts}


def verify_sensor_ranges_and_target_cases(service):
    loaded = service.upload()
    map_id = check_map(loaded, service.objects)
    settings = service.bootstrap["config"]["sensors"]
    tof = settings["tof"]
    assert (tof["hfov_deg"], tof["vfov_deg"], tof["min_range_m"], tof["max_range_m"],
            tof["samples_per_axis"]) == (45., 45., .015, 4., 4)
    assert settings["camera"]["samples_per_axis"] == 2
    near = service.sample(map_id, TARGET_A, "target-3m")
    front = views(near)["tof/front"]
    target_index = front["object_ids"].index("target")
    assert np.any(array(front["subray_object_index"], "int32") == target_index)
    assert np.any((array(front["raw_object_index"], "int32") == target_index)
                  & np.isfinite(array(front["range_m"], "float32")))
    # Reproduce the original screenshot's actual center and sensor attitude.
    position = [.004000000189989805, -17.604000091552734, 1.5]
    target = {"position_m": [.7369999885559082, -13.317999839782717, 1.5]}
    yaw = math.radians(78.68399810791016)
    body = {"position_m": position, "quaternion_xyzw": [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]}
    far = service.sample(map_id, target, "original-4m-case-new-range", body=body)
    front = views(far)["tof/front"]
    target_index = front["object_ids"].index("target")
    assert not np.any(array(front["subray_object_index"], "int32") == target_index)
    nearest = float(np.linalg.norm(np.asarray(target["position_m"]) - position)
                    - service.bootstrap["config"]["target_radius_m"])
    assert nearest > 4.
    return {"tof_fov_deg": 45., "tof_range_m": [.015, 4.], "3m_target": "valid sampled return",
            "old_case_nearest_surface_m": nearest, "old_case_target_beyond_new_range": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "outputs" / "ray_sensors" / "review-verification.json")
    args = parser.parse_args()
    groups = (verify_bootstrap, verify_static_assets, verify_upload_and_map_invalidation,
              verify_no_target_samples, verify_mounted_rays, verify_target_motion_and_same_pass,
              verify_request_validation, verify_concurrent_snapshots, verify_sensor_ranges_and_target_cases,
              verify_transition)
    report = {"passed": False, "checks": [], "failures": []}
    try:
        with Service() as service:
            for action in groups:
                try:
                    detail = action(service)
                    report["checks"].append({"name": action.__name__, "passed": True, "detail": detail})
                    print(f"PASS {action.__name__}", flush=True)
                except Exception as error:
                    failure = {"name": action.__name__, "passed": False, "error": str(error),
                               "traceback": traceback.format_exc()}
                    report["checks"].append(failure)
                    report["failures"].append(failure)
                    print(f"FAIL {action.__name__}: {type(error).__name__}: {error}", flush=True)
            service.log.flush()
            service.log.seek(0)
            report["server_log"] = service.log.read()
    except Exception as error:
        report["failures"].append({"name": "service_lifecycle", "error": str(error),
                                   "traceback": traceback.format_exc()})
    report["passed"] = not report["failures"]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "failures": len(report["failures"]), "results": str(args.results)}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
