"""Docker checks for public batched CUDA sensors and transform-only scene sync.

Independent ray/box/plane/sphere facts and the frozen sequential GL reference
are separate oracles. CPU downloads below are explicit assertion boundaries;
the public observation path returns owned CUDA tensors throughout.
"""

from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import traceback
from unittest.mock import patch

import numpy as np
import torch

from devtools.config import DepthConfig, MountConfig, SensorConfig, load_config
from devtools import _sensor_processing as processing
from devtools.mesh import MeshObject
from devtools.rendering import BatchRenderer, RasterBatch
from devtools.scene import GeometrySnapshot, Pose, SceneObject, Sphere
from devtools.sensors import (CameraBatch, DepthBatch, RigBatch, mounted_poses,
                             poses_to_tensor, sample_camera, sample_depth, sample_rig)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_SPEC = importlib.util.spec_from_file_location("frozen_sensor_checks", ROOT / "tests" / "verify_sensor_reference.py")
ref = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ref)
ATOL_M = 2e-5


def host(value):
    """Explicit consumer-side tensor download for assertions only."""
    return value.detach().cpu().numpy()


def rejected(action, errors=(TypeError, ValueError)):
    try:
        action()
    except errors:
        return
    raise AssertionError("invalid operation was accepted")


def close(actual, expected, atol=ATOL_M):
    assert np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True), (
        np.asarray(actual).shape, np.asarray(expected).shape,
        float(np.nanmax(np.abs(np.asarray(actual) - np.asarray(expected)))))


def same_ranges(actual, expected, atol=ATOL_M):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    categories = {
        "finite": np.isfinite,
        "positive_infinity": np.isposinf,
        "unavailable": np.isnan,
    }
    for name, classify in categories.items():
        differing = np.argwhere(classify(actual) != classify(expected))
        assert not len(differing), (name, differing[:12].tolist(), len(differing))
    assert not np.isneginf(actual).any()
    finite = np.isfinite(expected)
    difference = np.abs(actual[finite] - expected[finite])
    maximum = float(difference.max(initial=0))
    assert maximum <= atol, (maximum, atol)
    return {"finite_max_abs_m": maximum, "finite_pixels": int(finite.sum()),
            "empty_pixels": int(np.isposinf(actual).sum()),
            "unavailable_pixels": int(np.isnan(actual).sum())}


def depth_host(value):
    return {"range_m": host(value.range_m), "object_index": host(value.object_index),
            "object_ids": value.object_ids, "t_ref": value.t_ref,
            "scene_version": value.scene_version}


def camera_host(value):
    return {**depth_host(value.depth), "target_mask": host(value.target_mask),
            "target_hint": host(value.target_hint),
            "visible_fraction": host(value.visible_fraction),
            "unoccluded_pixels": host(value.unoccluded_pixels)}


def reference(scene, config, poses, *, camera=False, threshold=0):
    with ref.MeshSensor(scene) as sensor:
        if camera:
            frames = [sensor.camera(config, pose, threshold) for pose in poses]
            return {"range_m": np.stack([frame.depth.range_m for frame in frames]),
                    "object_index": np.stack([frame.depth.object_index for frame in frames]),
                    "object_ids": frames[0].depth.object_ids,
                    "target_mask": np.stack([frame.target_mask for frame in frames]),
                    "target_hint": np.stack([frame.target_hint for frame in frames]),
                    "visible_fraction": np.asarray([frame.visible_fraction for frame in frames]),
                    "unoccluded_pixels": np.asarray([frame.unoccluded_pixels for frame in frames])}
        frames = sensor.sample_many(config, poses)
        return {"range_m": np.stack([frame.range_m for frame in frames]),
                "object_index": np.stack([frame.object_index for frame in frames]),
                "object_ids": frames[0].object_ids}


def compare(actual, expected, *, atol=ATOL_M):
    record = same_ranges(actual["range_m"], expected["range_m"], atol)
    assert actual["object_ids"] == expected["object_ids"]
    differences = np.argwhere(actual["object_index"] != expected["object_index"])
    assert not len(differences), ("object_index", differences[:12].tolist(), len(differences))
    for name in ("target_mask", "target_hint", "unoccluded_pixels"):
        if name in expected:
            assert np.array_equal(actual[name], expected[name]), name
    if "visible_fraction" in expected:
        close(actual["visible_fraction"], expected["visible_fraction"], 1e-15)
    return record


def tensor_contract(value, batch, config, device):
    depth = value.depth if isinstance(value, CameraBatch) else value
    assert isinstance(depth, DepthBatch)
    assert depth.range_m.shape == depth.object_index.shape == (batch, config.height, config.width)
    assert depth.range_m.device == depth.object_index.device == device
    assert depth.range_m.dtype == torch.float32 and depth.object_index.dtype == torch.int64
    assert depth.range_m.is_cuda and depth.object_index.is_cuda
    assert not depth.range_m.requires_grad
    assert isinstance(depth.object_ids, tuple) and "vehicle" not in depth.object_ids
    assert isinstance(depth.scene_version, int)
    if isinstance(value, CameraBatch):
        for item, shape, dtype in ((value.target_mask, depth.range_m.shape, torch.bool),
                                   (value.target_hint, (batch, 5), torch.float32),
                                   (value.visible_fraction, (batch,), torch.float64),
                                   (value.unoccluded_pixels, (batch,), torch.int64)):
            assert item.shape == shape and item.device == device and item.dtype == dtype
    rejected(lambda: setattr(depth, "range_m", None), (FrozenInstanceError,))


def snapshot(scene):
    return scene.get_geometry(exclude_ids={"vehicle"})


def render_depth(scene, config, poses):
    expected = reference(scene, config, poses)
    with BatchRenderer(snapshot(scene)) as renderer:
        matrices = poses_to_tensor(poses, device=renderer.device)
        value = sample_depth(renderer, config, matrices)
        tensor_contract(value, len(poses), config, renderer.device)
        assert value.t_ref == scene.t_ref
        actual = depth_host(value)
        record = compare(actual, expected)
    return actual, expected, record


def independent_box_result(actual, expected_reference, config, poses, boxes, angles):
    """Keep pre-existing edge disagreements explicit and reject new ones."""
    distances, indices = [], []
    for pose in poses:
        distance, index = ref.box_image(config, pose.position_m, boxes, angles)
        distances.append(distance)
        indices.append(index)
    distance, index = np.stack(distances), np.stack(indices)
    for classify in (np.isfinite, np.isnan, np.isposinf):
        assert np.array_equal(classify(actual["range_m"]), classify(distance))
        assert np.array_equal(classify(expected_reference["range_m"]), classify(distance))
    finite_distance = np.isfinite(distance)
    baseline_error = np.zeros(distance.shape)
    actual_error = np.zeros(distance.shape)
    baseline_error[finite_distance] = np.abs(expected_reference["range_m"][finite_distance] - distance[finite_distance])
    actual_error[finite_distance] = np.abs(actual["range_m"][finite_distance] - distance[finite_distance])
    known_distance_failures = baseline_error > ATOL_M
    assert np.array_equal(actual_error > ATOL_M, known_distance_failures)
    assert np.all(actual_error[~known_distance_failures] <= ATOL_M)
    expected_names = np.asarray(["map/" + name for name, _ in boxes], dtype=object)
    finite = index >= 0
    oracle_names = expected_names[index[finite]]
    baseline_names = np.asarray(expected_reference["object_ids"], dtype=object)[expected_reference["object_index"][finite]]
    actual_names = np.asarray(actual["object_ids"], dtype=object)[actual["object_index"][finite]]
    baseline_mismatches = baseline_names != oracle_names
    actual_mismatches = actual_names != oracle_names
    assert np.array_equal(actual_mismatches, baseline_mismatches)
    examples = []
    for location in np.argwhere(known_distance_failures):
        at = tuple(location)
        examples.append({"batch_row_col": location.tolist(),
                         "analytic_range_m": float(distance[at]),
                         "frozen_reference_range_m": float(expected_reference["range_m"][at]),
                         "formal_sensor_range_m": float(actual["range_m"][at]),
                         "analytic_object_id": str(expected_names[index[at]]),
                         "frozen_reference_object_id": expected_reference["object_ids"][expected_reference["object_index"][at]],
                         "formal_sensor_object_id": actual["object_ids"][actual["object_index"][at]]})
    return {"oracle_identity_disagreement_pixels": int(actual_mismatches.sum()),
            "frozen_reference_identity_disagreement_pixels": int(baseline_mismatches.sum()),
            "oracle_distance_disagreement_pixels": int(known_distance_failures.sum()),
            "independent_oracle_status": "known_reference_failure" if examples else "passed",
            "known_reference_failures": examples,
            "policy": "same per-pixel errors as frozen reference; all other pixels retain 2e-5 m acceptance; all new disagreements fail"}


def verify_planes_and_boxes():
    records = {}
    config = DepthConfig(9, 7, 60, 45, 0, 12)
    boxes = (("far_wall", ((6, 7), (-10, 10), (10, 30))),
             ("upper_left", ((3, 3.5), (0.4, 1.8), (20.4, 21.8))))
    poses = (Pose(ref.ORIGIN), Pose((0.125, 0, 20)), Pose((0.25, 0, 20)))
    actual, expected, records["boxes_vs_frozen_reference"] = render_depth(ref.scene(boxes), config, poses)
    records["boxes_independent"] = independent_box_result(actual, expected, config, poses, boxes, (0, 0, 0))
    assert [item["batch_row_col"] for item in records["boxes_independent"]["known_reference_failures"]] == [
        [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 2, 3]]
    assert actual["range_m"][0, 0, -1] > 6.1
    assert actual["object_ids"][actual["object_index"][0, 0, 0]] == "map/upper_left"

    config = DepthConfig(5, 3, 60, 40, 0, 10)
    angles = (0, math.pi / 2, 0)
    poses = (Pose((0, 0, 4), ref.quaternion(*angles)),)
    analytic = np.asarray([[4 / -ref.rotate_euler(ray, *angles)[2]
                            for ray in row] for row in ref.reference_rays(config)])
    for declared in (True, False):
        actual, _, records[f"ground_declared_{declared}"] = render_depth(ref.scene(declared=declared), config, poses)
        same_ranges(actual["range_m"][0], analytic)
        assert np.all(actual["object_index"] == actual["object_ids"].index("map/ground"))

    config = DepthConfig(64, 48, 120, 100, .05, 12)
    origin = (-.5481651582549905, -1.4890260899279797, 11.00635970341665)
    quaternion = (-.28231015743329757, .26836274025745954, -.45667413895892944, .799831948260142)
    unit = np.asarray(quaternion) / math.hypot(*quaternion)
    ray = ref.reference_rays(config)[8, 55]
    direction = ray + 2 * np.cross(unit[:3], np.cross(unit[:3], ray) + unit[3] * ray)
    expected_distance = (1 - origin[0]) / direction[0]
    wall = ("wide_fov_wall", ((1, 1.5), (-15, 15), (0, 20)))
    actual, _, records["wide_fov_witness"] = render_depth(ref.scene((wall,), size=(40, 40, 20)), config, (Pose(origin, quaternion),))
    close(actual["range_m"][0, 8, 55], expected_distance)
    records["witness_independent_abs_m"] = float(abs(actual["range_m"][0, 8, 55] - expected_distance))
    return records


def verify_range_and_coverage():
    records = {}
    wall = ("wall", ((4, 5), (-8, 8), (10, 30)))
    base = DepthConfig(1, 1, 60, 60, 0, 8)
    poses = (Pose(ref.ORIGIN),)
    for name, config, classify in (("inclusive_min", replace(base, min_range_m=4), np.isfinite),
                                    ("inclusive_max", replace(base, max_range_m=4), np.isfinite),
                                    ("too_close", replace(base, min_range_m=4.25), np.isnan),
                                    ("beyond_range", replace(base, max_range_m=3.5), np.isposinf)):
        actual, _, records[name] = render_depth(ref.scene((wall,)), config, poses)
        assert classify(actual["range_m"]).all()
    crossing = ("crossing_camera_plane", ((-1, .05), (-1, 1), (19, 21)))
    actual, _, records["camera_plane_crossing_occluder"] = render_depth(
        ref.scene((wall, crossing)), DepthConfig(3, 3, 60, 60, .1, 8), poses)
    assert np.isnan(actual["range_m"]).all()
    assert np.all(actual["object_index"] == actual["object_ids"].index("map/crossing_camera_plane"))
    near_max = ("near_max_wall", ((12.000001, 12.5), (-8, 8), (10, 30)))
    actual, _, records["uncertain_near_max"] = render_depth(ref.scene((near_max,)), replace(base, max_range_m=12), poses)
    assert np.isnan(actual["range_m"]).all()
    for declared in (True, False):
        actual, _, records[f"empty_declared_{declared}"] = render_depth(ref.scene(declared=declared), base, poses)
        assert (np.isposinf if declared else np.isnan)(actual["range_m"]).all()
        _, _, records[f"hit_declared_{declared}"] = render_depth(ref.scene((wall,), declared=declared), base, poses)
    small = (10, 10, 10)
    origin = (Pose((0, 0, 5)),)
    actual, _, records["uncovered_empty"] = render_depth(ref.scene(size=small, target=(-3, -3, 5)), base, origin)
    assert np.isnan(actual["range_m"]).all()
    within = ("within_domain", ((4, 4.5), (-1, 1), (4, 6)))
    domain_scene = ref.scene((within,), size=small, target=(-3, -3, 5))
    poses = (Pose((0, 0, 5)), Pose((6, 0, 5), ref.quaternion(yaw=math.pi)))
    actual, _, records["in_domain_hit_outside_origin"] = render_depth(domain_scene, base, poses)
    assert actual["range_m"][0, 0, 0] == 4 and np.isnan(actual["range_m"][1, 0, 0])
    return records


def render_camera(scene, config, poses, threshold):
    expected = reference(scene, config, poses, camera=True, threshold=threshold)
    with BatchRenderer(snapshot(scene)) as renderer:
        matrices = poses_to_tensor(poses, device=renderer.device)
        value = sample_camera(renderer, config, matrices, threshold)
        tensor_contract(value, len(poses), config, renderer.device)
        actual = camera_host(value)
        record = compare(actual, expected)
    return actual, expected, record


def verify_camera():
    records = {}
    config = DepthConfig(80, 60, 60, 45, .01, 15)
    center, radius = (6, 0, 20), 1.2
    poses = (Pose(ref.ORIGIN), Pose((0, .125, 20)), Pose((0, -.125, 20)))
    clear_scene = ref.scene(target=center, radius=radius)
    actual, _, records["clear"] = render_camera(clear_scene, config, poses, 1)
    analytic_mask, analytic_distance, discriminant = ref.analytic_sphere(config, center, radius)
    disagreements = int(np.count_nonzero(actual["target_mask"][0] ^ analytic_mask))
    assert disagreements <= 4, disagreements
    interior = analytic_mask & actual["target_mask"][0] & (discriminant > .1)
    delta = actual["range_m"][0][interior] - analytic_distance[interior]
    assert len(delta) and delta.min() >= -ATOL_M and delta.max() < .006
    records["sphere_independent"] = {"silhouette_disagreement_pixels": disagreements,
                                      "interior_max_abs_m": float(np.abs(delta).max())}
    assert np.all(actual["visible_fraction"] == 1) and np.all(actual["target_hint"][:, 0] == 1)
    actual, _, records["zero_denominator"] = render_camera(clear_scene, replace(config, max_range_m=2), poses, 0)
    assert not actual["target_mask"].any() and not actual["target_hint"].any()
    assert not actual["unoccluded_pixels"].any() and not actual["visible_fraction"].any()

    half = ("half_occluder", ((3, 3.2), (0, 3), (16, 24)))
    half_scene = ref.scene((half,), target=center, radius=radius)
    actual, _, records["half_occlusion"] = render_camera(half_scene, config, poses, .4)
    visible = actual["target_mask"][0]
    independent_visible = analytic_mask & (ref.reference_rays(config)[..., 1] < 0)
    assert np.count_nonzero(visible ^ independent_visible) <= 4
    assert .48 <= actual["visible_fraction"][0] <= .52
    assert actual["visible_fraction"][0] == np.count_nonzero(visible) / actual["unoccluded_pixels"][0]
    rows, cols = np.nonzero(visible)
    close(actual["target_hint"][0], (1, cols.min()/80, rows.min()/60, (cols.max()+1)/80, (rows.max()+1)/60), 1e-7)
    assert np.all(cols >= 40)
    for name, threshold, detected in (("inclusive_threshold", float(actual["visible_fraction"][0]), True),
                                       ("threshold_reject", .6, False)):
        observed, _, records[name] = render_camera(half_scene, config, (poses[0],), threshold)
        assert bool(observed["target_hint"][0, 0]) == detected
    full = ("full_occluder", ((3, 3.2), (-3, 3), (16, 24)))
    actual, _, records["full_occlusion"] = render_camera(ref.scene((full,), target=center, radius=radius), config, poses, 0)
    assert actual["unoccluded_pixels"].min() > 0
    assert not actual["visible_fraction"].any() and not actual["target_hint"].any() and not actual["target_mask"].any()

    one = DepthConfig(1, 1, 60, 60, 0, 8)
    actual, _, records["target_outside_declared_domain"] = render_camera(
        ref.scene(size=(10, 10, 10), target=(6, 0, 5), radius=.25), one, (Pose((0, 0, 5)),), 0)
    assert not actual["unoccluded_pixels"].any() and not actual["target_hint"].any()
    actual, _, records["target_exact_domain_boundary"] = render_camera(
        ref.scene(size=(10, 10, 10), target=(5.5, 0, 5), radius=.5), one, (Pose((-.123, 0, 5)),), 1)
    close(actual["range_m"][0, 0, 0], 5.123)
    assert actual["target_mask"].all() and actual["visible_fraction"][0] == 1
    assert np.array_equal(actual["target_hint"][0], (1, 0, 0, 1, 1))
    return records


def verify_batch_and_runtime():
    records = {}
    wall = ("wall", ((4, 5), (-8, 8), (10, 30)))
    base = ref.scene((wall,))
    config = DepthConfig(7, 5, 30, 20, 0, 12)
    angles = tuple(math.radians(x) for x in (18, -14, 12))
    poses = tuple(Pose((.125*(index % 3), 0, 20), ref.quaternion(*angles)) for index in range(48))
    expected = reference(base, config, poses)
    with BatchRenderer(snapshot(base)) as renderer:
        records["renderer"] = renderer.renderer_info
        assert not any(tag in renderer.renderer_info["GL_RENDERER"].lower()
                       for tag in ("llvmpipe", "softpipe", "software", "lavapipe"))
        matrices = poses_to_tensor(poses, device=renderer.device)
        close(host(matrices), np.stack([pose.as_transform().matrix for pose in poses]), 1e-7)
        before = dict(renderer.geometry_info)
        value = sample_depth(renderer, config, matrices)
        assert renderer.geometry_info["instanced_draw_count"] - before["instanced_draw_count"] == 1
        tensor_contract(value, 48, config, renderer.device)
        saved = depth_host(value)
        records["48_views_frozen_reference"] = compare(saved, expected)
        records["48_views_independent"] = independent_box_result(saved, expected, config, poses, (wall,), angles)
        reverse = depth_host(sample_depth(renderer, config, matrices.flip(0)))
        for key in ("range_m", "object_index"):
            reverse[key] = reverse[key][::-1]
        compare(reverse, saved, atol=0)
        single = depth_host(sample_depth(renderer, config, matrices[7:8]))
        selected = {key: (array[7:8] if isinstance(array, np.ndarray) else array) for key, array in saved.items()}
        compare(single, selected, atol=0)
        raw = renderer.render(matrices[:1], config, only_object="map/wall")
        assert isinstance(raw, RasterBatch) and raw.distance_m.shape == raw.object_index.shape == (1, 5, 7)
        assert raw.distance_m.device == raw.object_index.device == renderer.device
        assert raw.distance_m.dtype == torch.float32 and raw.object_index.dtype == torch.int64
        assert raw.map_bounds_m == snapshot(base).map_bounds_m
        assert raw.scene_version == value.scene_version and raw.t_ref == value.t_ref
        same_ranges(host(raw.distance_m), saved["range_m"][:1], 0)
        assert np.array_equal(host(raw.object_index), saved["object_index"][:1])
        shifted_poses = tuple(Pose((pose.position_m[0]+.25, 0, 20), pose.quaternion_xyzw) for pose in poses)
        shifted_expected = reference(base, config, shifted_poses)
        matrices[:, 0, 3].add_(.25)
        shifted = depth_host(sample_depth(renderer, config, matrices))
        records["mutated_cuda_pose_storage"] = compare(shifted, shifted_expected)
        assert np.any(shifted["range_m"] != saved["range_m"])
        current, alternate = torch.cuda.current_stream(renderer.device), torch.cuda.Stream(device=renderer.device)
        alternate.wait_stream(current)
        with torch.cuda.stream(alternate):
            streamed = sample_depth(renderer, config, matrices)
        current.wait_stream(alternate)
        compare(depth_host(streamed), shifted, atol=0)
        matrices[:, 0, 3].sub_(.25)
        restored = depth_host(sample_depth(renderer, config, matrices))
        compare(restored, saved, atol=0)
        compare(depth_host(value), saved, atol=0)
        large_poses = tuple(Pose((.125*(index % 3), 0, 20)) for index in range(256))
        large_matrices = poses_to_tensor(large_poses, device=renderer.device)
        large = sample_depth(renderer, DepthConfig(8, 8, 60, 60, .05, 12), large_matrices)
        tensor_contract(large, 256, DepthConfig(8, 8, 60, 60, .05, 12), renderer.device)
        large_expected = reference(base, DepthConfig(8, 8, 60, 60, .05, 12), large_poses)
        records["256_views_frozen_reference"] = compare(depth_host(large), large_expected)
        after = dict(renderer.geometry_info)
        for key in ("scene_geometry_buffer_name", "scene_geometry_buffer_bytes", "geometry_uploads"):
            assert before[key] == after[key], key
        assert after["geometry_uploads"] == 1 and after["cpu_image_readbacks"] == after["per_view_draw_calls"] == 0
        thread_failures = []
        def other_thread():
            for operation in (lambda: renderer.render(matrices, config), renderer.close):
                try:
                    operation()
                except RuntimeError:
                    continue
                except Exception as error:
                    thread_failures.append(f"wrong cross-thread error: {type(error).__name__}: {error}")
                    continue
                thread_failures.append("cross-thread operation accepted")
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
        assert not thread_failures, thread_failures
        for invalid in (matrices.cpu(), matrices.to(torch.float64), matrices[:, :3, :], matrices[:0]):
            rejected(lambda invalid=invalid: sample_depth(renderer, config, invalid))
        rejected(lambda: sample_depth(renderer, None, matrices))
        rejected(lambda: sample_camera(renderer, config, matrices, True))
        rejected(lambda: renderer.render(matrices, config, only_object="missing"))
        rejected(lambda: poses_to_tensor((ref.ORIGIN,)))
        rejected(lambda: poses_to_tensor(()))
        compare(depth_host(value), saved, atol=0)
        records["shared_geometry"] = {key: after[key] for key in
                                     ("scene_geometry_buffer_name", "scene_geometry_buffer_bytes", "geometry_uploads",
                                      "cpu_image_readbacks", "per_view_draw_calls")}
    renderer.close()
    compare(depth_host(value), saved, atol=0)
    rejected(lambda: sample_depth(renderer, config, matrices), (RuntimeError,))
    return records


def verify_rig():
    from verify_tof_zones import analytic_box_zones

    walls = (("right", ((-20, 20), (-2.5, -2), (1, 39))),
             ("front", ((4, 4.5), (-20, 20), (1, 39))),
             ("left", ((-20, 20), (6, 6.5), (1, 39))),
             ("rear", ((-8.5, -8), (-20, 20), (1, 39))))
    base = ref.scene(walls, target=(30, 30, 20), radius=.5)
    config = replace(load_config(ROOT / "configs" / "devtools.json").sensors,
                     camera=DepthConfig(24, 18, 80, 60, .01, 30))
    body_angles = tuple(math.radians(x) for x in (23, -31, 48))
    mount_angles = tuple(math.radians(x) for x in (-17, 21, -12))
    custom = MountConfig((.4, -.2, .1), ref.quaternion(*mount_angles))
    config = replace(config, tof_mounts=(custom, *config.tof_mounts[1:]), camera_mount=custom)
    poses = (Pose((0, 0, 20), ref.quaternion(*body_angles)), Pose((.25, -.125, 20), ref.quaternion(*body_angles)))
    with ref.MeshSensor(base) as sensor:
        expected = [ref.sample_rig(sensor, config, pose) for pose in poses]
    with BatchRenderer(snapshot(base)) as renderer:
        matrices = poses_to_tensor(poses, device=renderer.device)
        mounts = host(mounted_poses(matrices, config.tof_mounts))
        for body_index, body in enumerate(poses):
            for mount_index, mount in enumerate(config.tof_mounts):
                close(mounts[body_index, mount_index], ref.mounted_pose(body, mount).as_transform().matrix, 2e-6)
        first_mount = config.tof_mounts[0]
        independent_origin = np.asarray(poses[0].position_m) + ref.rotate_euler(first_mount.position_m, *body_angles)
        close(mounts[0, 0, :3, 3], independent_origin, 2e-6)
        for vector in np.eye(3):
            independent_direction = ref.rotate_euler(ref.rotate_euler(vector, *mount_angles), *body_angles)
            close(mounts[0, 0, :3, :3] @ vector, independent_direction, 2e-7)
        before = renderer.geometry_info["instanced_draw_count"]
        rig = sample_rig(renderer, config, matrices)
        assert isinstance(rig, RigBatch) and rig.tof_m.shape == (2, 4, 8, 8)
        assert rig.tof_m.dtype == torch.float32 and rig.tof_m.device == renderer.device
        assert rig.target_hint.shape == (2, 5) and rig.target_hint.device == renderer.device
        assert rig.scene_version == rig.camera.depth.scene_version and rig.t_ref == rig.camera.depth.t_ref == 12.5
        assert renderer.geometry_info["instanced_draw_count"] - before == 3
        close(host(rig.target_hint), np.stack([item.target_hint for item in expected]), 1e-7)
        for body_index, pose in enumerate(poses):
            for mount_index in range(4):
                independent, _ = analytic_box_zones(config.tof, mounts[body_index, mount_index], walls)
                same_ranges(host(rig.tof_m)[body_index, mount_index], independent, 3e-5)
        tensor_contract(rig.camera, len(poses), config.camera, renderer.device)
    return {"body_poses": 2, "tof_views": 8, "full_and_target_camera_views": 4,
            "instanced_draws": 3,
            "mount_geometry": "independent Euler composition and continuous box-face/zone active-set quadratic programs",
            "camera_reference": "unchanged center-pixel GL visibility and hints"}


def verify_metadata_and_allocation_failure():
    """Invalid metadata must not allocate; partial GPU setup fails closed."""
    wall = ("wall", ((4, 5), (-8, 8), (10, 30)))
    config = DepthConfig(3, 3, 60, 60, 0, 12)
    with BatchRenderer(snapshot(ref.scene((wall,)))) as renderer:
        matrices = poses_to_tensor((Pose(ref.ORIGIN),), device=renderer.device)
        original = sample_depth(renderer, config, matrices)
        saved = depth_host(original)
        memory_before = dict(renderer.memory_info)
        draws_before = renderer.geometry_info["instanced_draw_count"]
        for invalid_config in (replace(config, width=4, max_range_m=1e40),
                               replace(config, width=4, hfov_deg=1e-300),
                               replace(config, width=4, vfov_deg=1e-300)):
            rejected(lambda invalid_config=invalid_config: renderer.render(matrices, invalid_config),
                     (ValueError,))
            assert renderer.memory_info == memory_before
            assert renderer.geometry_info["instanced_draw_count"] == draws_before
        compare(depth_host(sample_depth(renderer, config, matrices)), saved, atol=0)
        with patch.object(renderer._cuda, "register", side_effect=RuntimeError("injected registration failure")):
            try:
                renderer.render(matrices, replace(config, width=5, height=4))
            except RuntimeError as error:
                assert "injected registration failure" in str(error)
            else:
                raise AssertionError("injected allocation failure was not propagated")
        rejected(lambda: renderer.render(matrices, config), (RuntimeError,))
        compare(depth_host(original), saved, atol=0)
    renderer.close()
    return {"invalid_metadata_cases": 3, "allocation_failure": "propagated and closed",
            "previous_output": "retained after failed allocation and close"}


def verify_postprocess_cache_streams():
    """Consume a still-pending cache from an unrelated stream, without GL."""
    config = DepthConfig(5, 3, 60, 60, 1, 8)
    bounds = ((-5, 5), (-5, 5), (0, 10))
    distances = np.full((2, 3, 5), 2, dtype=np.float32)
    identities = np.zeros((2, 3, 5), dtype=np.int64)
    identities[0, 0, 0], distances[0, 0, 0] = -1, 0
    distances[0, 0, 1], distances[0, 0, 2] = .5, 8.000001
    expected = np.full((2, 3, 5), math.nan, dtype=np.float32)
    expected[0] = 2
    expected[0, 0, 0], expected[0, 0, 1], expected[0, 0, 2] = math.inf, math.nan, math.nan
    # Slab/segment facts independently establish these values: the first
    # camera is at x=-4, where all 8 m segments are covered; the second is at
    # x=3.5 and every 2 m hit is outside x<=5. The 0.5 m hit is below min, while
    # 8.000001 m is inside the float32 uncertainty band beyond max.
    matrices = poses_to_tensor((Pose((-4, 0, 5)), Pose((3.5, 0, 5))))
    distance_a = torch.tensor(distances, device=matrices.device)
    identity_a = torch.tensor(identities, device=matrices.device)
    distance_b, identity_b, matrices_b = distance_a.clone(), identity_a.clone(), matrices.clone()
    first = torch.cuda.Stream(device=matrices.device)
    second = torch.cuda.Stream(device=matrices.device)
    if not hasattr(torch.cuda, "_sleep"):
        raise RuntimeError("CUDA _sleep is required for this deterministic pending-cache race probe")
    torch.cuda.synchronize(matrices.device)
    # Warm the kernels and each stream's allocator before creating the race;
    # a first cudaMalloc/kernel load can otherwise synchronize a short delay.
    # Use different FOV values so recycled stale rays cannot equal the intended
    # constants. With 120-degree rays several x=3.5 hits lie inside the domain;
    # all corresponding 60-degree rays lie outside it.
    warm_config = replace(config, hfov_deg=120, vfov_deg=120)
    with torch.cuda.stream(first):
        warm_a = processing.postprocess_depth(distance_a, identity_a, matrices, warm_config, bounds)
        torch.cuda._sleep(1)
    with torch.cuda.stream(second):
        warm_b = processing.postprocess_depth(distance_b, identity_b, matrices_b, warm_config, bounds)
    torch.cuda.synchronize(matrices.device)
    del warm_a, warm_b
    processing.clear_caches()
    original_arange = torch.arange
    inserted = []
    def delayed_first_arange(*args, **kwargs):
        # Insert after the small bounds upload, so a pageable host-to-device
        # constant copy cannot accidentally drain the intended delay.
        if not inserted:
            torch.cuda._sleep(100_000_000)
            inserted.append(True)
        return original_arange(*args, **kwargs)
    with torch.cuda.stream(first):
        with patch.object(processing.torch, "arange", side_effect=delayed_first_arange):
            output_a = processing.postprocess_depth(distance_a, identity_a, matrices, config, bounds)
        # Inspect readiness only; do not wait for the producer event here.
        _, cache_ready = processing._pixel_constants(config.height, config.width,
                                                     config.hfov_deg, config.vfov_deg,
                                                     matrices.device)
    assert inserted and not cache_ready.query(), "probe failed to create a genuinely pending cache"
    with torch.cuda.stream(second):
        output_b = processing.postprocess_depth(distance_b, identity_b, matrices_b, config, bounds)
    # No second.wait_stream(first) is issued: the module's ready event is the
    # only cross-stream dependency for its shared pixel and bounds constants.
    second.synchronize()
    first.synchronize()
    same_ranges(host(output_a), expected, 0)
    same_ranges(host(output_b), expected, 0)
    return {"independent_input_buffers": True, "producer_cache_was_pending": True,
            "explicit_consumer_wait_stream": False, "GL_interop": False,
            "finite_pixels": int(np.isfinite(expected).sum()),
            "empty_pixels": int(np.isposinf(expected).sum()),
            "unavailable_pixels": int(np.isnan(expected).sum())}


def verify_target_sync():
    config = DepthConfig(80, 60, 60, 45, .01, 15)
    occluder = ("half_occluder", ((3, 3.2), (0, 3), (16, 24)))
    first = ref.scene((occluder,), target=(6, 0, 20), radius=1.2)
    moved = first.with_target_position((6, -2.5, 20))
    final = moved.with_target_position(first.target.pose.position_m)
    poses = (Pose(ref.ORIGIN), Pose((0, -.125, 20)), Pose((.125, .125, 20)))
    snapshots = tuple(snapshot(scene) for scene in (first, moved, final))
    for current_scene, current_snapshot in zip((first, moved, final), snapshots):
        assert current_scene.background is first.background
        assert current_snapshot.frame == "W" and current_snapshot.unit == "metre"
        assert current_snapshot.get_object("target").geometry.pose.position_m == current_scene.target.pose.position_m
        for before, after in zip(snapshots[0].objects[:-1], current_snapshot.objects[:-1]):
            assert before.geometry is after.geometry
    cold, baseline = [], []
    for current_scene, current_snapshot in zip((first, moved, final), snapshots):
        baseline.append(reference(current_scene, config, poses, camera=True, threshold=.4))
        with BatchRenderer(current_snapshot) as renderer:
            matrices = poses_to_tensor(poses, device=renderer.device)
            cold.append(camera_host(sample_camera(renderer, config, matrices, .4)))
    records = {}
    with BatchRenderer(snapshots[0]) as renderer:
        matrices = poses_to_tensor(poses, device=renderer.device)
        immutable_geometry = dict(renderer.geometry_info)
        value = sample_camera(renderer, config, matrices, .4)
        saved = camera_host(value)
        initial_version = value.depth.scene_version
        assert renderer.sync(snapshots[0]) == initial_version
        initial_updates = renderer.geometry_info["object_transform_updates"]
        initial_bytes = renderer.geometry_info["object_transform_upload_bytes"]
        versions, fractions = [initial_version], [float(host(value.visible_fraction)[0])]
        compare(saved, cold[0], atol=0)
        for index, current_snapshot in enumerate(snapshots[1:], 1):
            version = renderer.sync(current_snapshot)
            assert version == versions[-1] + 1
            versions.append(version)
            measured = sample_camera(renderer, config, matrices, .4)
            tensor_contract(measured, len(poses), config, renderer.device)
            actual = camera_host(measured)
            assert measured.depth.scene_version == version and measured.depth.t_ref == first.t_ref
            records[f"sync_{index}_cold_renderer"] = compare(actual, cold[index], atol=0)
            records[f"sync_{index}_frozen_reference"] = compare(actual, baseline[index])
            fractions.append(float(actual["visible_fraction"][0]))
            info = renderer.geometry_info
            for name in ("scene_geometry_buffer_name", "scene_geometry_buffer_bytes", "geometry_uploads"):
                assert info[name] == immutable_geometry[name], name
            assert info["object_transform_updates"] == initial_updates + index
            assert info["object_transform_upload_bytes"] == initial_bytes + index * 64
            compare(camera_host(value), saved, atol=0)
        assert fractions[1] > fractions[0] + .2, fractions
        assert fractions[0] == fractions[2]
        records["versions"] = versions
        records["visible_fraction_A_B_A"] = fractions
        records["moving_objects_uploaded_bytes"] = renderer.geometry_info["object_transform_upload_bytes"] - initial_bytes
        stable = camera_host(sample_camera(renderer, config, matrices, .4))
        stable_info = dict(renderer.geometry_info)
        target = snapshots[-1].get_object("target")
        changed_radius = replace(target, geometry=Sphere(target.geometry.pose, 1.3))
        changed_id = replace(target, object_id="renamed_target")
        map_object = snapshots[-1].objects[0]
        map_shape = map_object.geometry
        changed_mesh = replace(map_object, geometry=MeshObject(map_shape.name, map_shape.vertices_m,
                                                              ((0, 1, 3), (1, 2, 3)), map_shape.kind))
        invalid = (replace(snapshots[-1], objects=(*snapshots[-1].objects[:-1], changed_radius)),
                   replace(snapshots[-1], objects=(*snapshots[-1].objects[:-1], changed_id)),
                   replace(snapshots[-1], objects=(changed_mesh, *snapshots[-1].objects[1:])),
                   replace(snapshots[-1], objects=snapshots[-1].objects[:-1]))
        for current_snapshot in invalid:
            rejected(lambda current_snapshot=current_snapshot: renderer.sync(current_snapshot))
            now = renderer.geometry_info
            for name in ("scene_version", "geometry_uploads", "object_transform_updates", "object_transform_upload_bytes"):
                assert now[name] == stable_info[name], name
            compare(camera_host(sample_camera(renderer, config, matrices, .4)), stable, atol=0)
        timed = replace(snapshots[-1], t_ref=99)
        new_version = renderer.sync(timed)
        assert new_version == versions[-1] + 1
        timed_output = sample_camera(renderer, config, matrices, .4)
        assert timed_output.depth.t_ref == 99 and value.depth.t_ref == 12.5
        assert timed_output.depth.scene_version == new_version and value.depth.scene_version == initial_version
        for name in ("geometry_uploads", "object_transform_updates", "object_transform_upload_bytes"):
            assert renderer.geometry_info[name] == stable_info[name], name
        compare(camera_host(timed_output), stable, atol=0)
        compare(camera_host(value), saved, atol=0)
    compare(camera_host(value), saved, atol=0)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    result = {"status": "running", "backend": "formal OpenGL layered CUDA sensor",
              "device": torch.cuda.get_device_name(), "checks": [],
              "acceptance": {"finite_atol_m": ATOL_M, "new_category_or_identity_errors": 0,
                             "camera_mask_hint_count_errors": 0, "moving_vs_cold_atol_m": 0},
              "scope": "noiseless mesh pinhole observations; explicit assertion downloads only"}
    def run(name, action):
        try:
            detail = action()
            result["checks"].append({"name": name, "status": "passed", "detail": detail})
            print(f"{name}: passed", flush=True)
        except Exception as error:
            result["checks"].append({"name": name, "status": "failed", "error_type": type(error).__name__,
                                     "error": str(error), "traceback": traceback.format_exc()})
            print(f"{name}: failed: {type(error).__name__}: {error}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sensor-config-", dir=args.results.parent) as folder:
        config_checks = []
        run("strict_sensor_config", lambda: (ref.verify_config(Path(folder), config_checks), config_checks)[1])
    for name, action in (("independent_planes_boxes_wide_fov", verify_planes_and_boxes),
                         ("range_and_coverage", verify_range_and_coverage),
                         ("batched_camera_visibility", verify_camera),
                         ("batch_order_storage_stream_thread", verify_batch_and_runtime),
                         ("external_rig_mounts", verify_rig),
                         ("metadata_and_allocation_failure", verify_metadata_and_allocation_failure),
                         ("postprocess_pending_cache_streams", verify_postprocess_cache_streams),
                         ("target_A_B_A_transform_sync", verify_target_sync)):
        run(name, action)
    result["status"] = "passed" if all(check["status"] == "passed" for check in result["checks"]) else "failed"
    result["check_groups"] = len(result["checks"])
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"sensors: {result['status']}; result: {args.results}")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
