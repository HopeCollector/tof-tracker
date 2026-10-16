"""Verify GPU sensing against independent pinhole and analytic geometry facts.

Fixtures are constructed here from metre-valued boxes, a plane and spheres.
No old renderer or private rasterization routine supplies expected results.
Run inside the project's graphics-capable Docker environment.
"""

import argparse
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

from devtools.config import DepthConfig, MountConfig, SceneConfig, SensorConfig, load_config
from devtools.mesh import MeshMap, MeshObject
from devtools.scene import Pose, Scene, Sphere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.sensor_reference import MeshSensor, mounted_pose, pixel_directions, sample_rig


ORIGIN = (0.0, 0.0, 20.0)
IDENTITY = (0.0, 0.0, 0.0, 1.0)
BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))


def rejected(action, error=ValueError):
    try:
        action()
    except error:
        return
    raise AssertionError("invalid operation was accepted")


def close(actual, expected, tolerance=2e-5):
    assert np.allclose(actual, expected, rtol=0, atol=tolerance), (actual, expected)


def readonly(array):
    assert not array.flags.writeable
    rejected(lambda: array.setflags(write=True))
    rejected(lambda: array.__setitem__((...,), 0))


def quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """Independent elementary Euler-angle quaternion, with radian inputs."""
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


def rotate_euler(vector, roll, pitch, yaw):
    """Rotate by Rx, then Ry, then Rz, independently of the shared Transform."""
    x, y, z = vector
    y, z = math.cos(roll)*y - math.sin(roll)*z, math.sin(roll)*y + math.cos(roll)*z
    x, z = math.cos(pitch)*x + math.sin(pitch)*z, -math.sin(pitch)*x + math.cos(pitch)*z
    return np.array((math.cos(yaw)*x - math.sin(yaw)*y,
                     math.sin(yaw)*x + math.cos(yaw)*y, z))


def reference_rays(config):
    """Explicit pinhole pixel centers: rows down, columns right, FLU axes."""
    rays = np.empty((config.height, config.width, 3), dtype=float)
    for row in range(config.height):
        for col in range(config.width):
            left = (1 - 2 * (col + 0.5) / config.width) * math.tan(math.radians(config.hfov_deg) / 2)
            up = (1 - 2 * (row + 0.5) / config.height) * math.tan(math.radians(config.vfov_deg) / 2)
            length = math.sqrt(1 + left*left + up*up)
            rays[row, col] = (1 / length, left / length, up / length)
    return rays


def box(name, bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    vertices = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    return MeshObject(name, vertices, BOX_FACES, "obstacle")


def scene(boxes=(), *, size=(100.0, 100.0, 100.0), target=(-20.0, 0.0, 20.0),
          radius=0.3, declared=True):
    sx, sy, _ = size
    ground = MeshObject("ground", ((-sx/2, -sy/2, 0), (sx/2, -sy/2, 0),
                                   (sx/2, sy/2, 0), (-sx/2, sy/2, 0)),
                        ((0, 1, 2), (0, 2, 3)), "ground")
    background = MeshMap(Path("analytic-sensor-fixture.obj"),
                         (ground, *(box(name, bounds) for name, bounds in boxes)),
                         size_xyz_m=size if declared else None)
    return Scene(background, Sphere(Pose(ORIGIN), 0.2), Sphere(Pose(target), radius), 12.5)


def ray_box(origin, direction, bounds):
    """Analytic slab intersection with an axis-aligned closed box."""
    entering, leaving = -math.inf, math.inf
    for value, delta, (low, high) in zip(origin, direction, bounds):
        if abs(delta) < 1e-14:
            if value < low or value > high:
                return math.inf
            continue
        first, second = (low - value) / delta, (high - value) / delta
        entering, leaving = max(entering, min(first, second)), min(leaving, max(first, second))
    if entering > leaving or leaving <= 0:
        return math.inf
    return entering if entering > 0 else leaving


def box_image(config, origin, boxes, angles=(0.0, 0.0, 0.0)):
    ranges = np.full((config.height, config.width), math.inf)
    identities = np.full(ranges.shape, -1, dtype=int)
    rays = reference_rays(config)
    for row in range(config.height):
        for col in range(config.width):
            direction = rotate_euler(rays[row, col], *angles)
            candidates = [ray_box(origin, direction, bounds) for _, bounds in boxes]
            nearest = min(candidates)
            if nearest <= config.max_range_m:
                ranges[row, col] = nearest
                identities[row, col] = candidates.index(nearest)
    return ranges, identities


def finite_summary(values):
    """Consumer-side example: unavailable ranges never enter a finite mean."""
    array = np.asarray(values)
    valid = array[np.isfinite(array)]
    return {"finite_count": int(valid.size), "mean_m": float(valid.mean()) if valid.size else None,
            "empty_count": int(np.count_nonzero(np.isposinf(array))),
            "unavailable_count": int(np.count_nonzero(np.isnan(array)))}


def record_renderer(sensor, renderers):
    assert sensor.renderer_info["GL_VENDOR"] and sensor.renderer_info["GL_VERSION"]
    assert not any(tag in sensor.renderer_info["GL_RENDERER"].lower()
                   for tag in ("llvmpipe", "softpipe", "swrast", "software", "lavapipe"))
    if sensor.renderer_info not in renderers:
        renderers.append(dict(sensor.renderer_info))


def verify_config(folder, checks):
    source = Path(__file__).resolve().parents[1] / "configs" / "devtools.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    config = load_config(source)
    sensors = config.sensors
    assert isinstance(sensors, SensorConfig)
    assert sensors.tof == DepthConfig(8, 8, 60, 60, 0.05, 12)
    assert sensors.camera == DepthConfig(160, 120, 80, 60, 0.01, 30)
    assert sensors.min_visible_fraction == 0.2
    for mount, direction in zip(sensors.tof_mounts, ((0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0))):
        close(mount.position_m, (0, 0, 0))
        close(mount.as_transform().apply_vectors((1, 0, 0)), direction, 1e-10)
    close(sensors.camera_mount.as_transform().apply_vectors((1, 0, 0)), (1, 0, 0))
    for instance, name, other in ((sensors.tof, "width", 10), (sensors, "tof_mounts", ()),
                                  (sensors.camera_mount, "position_m", (1, 2, 3)), (config, "sensors", None)):
        rejected(lambda instance=instance, name=name, other=other: setattr(instance, name, other), FrozenInstanceError)
    position, rotation = [1, 2, 3], [0, 0, 0, 1.0000005]
    copied_mount = MountConfig(position, rotation)
    position[0], rotation[3] = 99, 0
    assert copied_mount.position_m == (1, 2, 3)
    close(copied_mount.quaternion_xyzw, IDENTITY, 1e-10)
    mounts = list(sensors.tof_mounts)
    copied_sensors = replace(sensors, tof_mounts=mounts)
    mounts.clear()
    assert copied_sensors.tof_mounts == sensors.tof_mounts
    for field in ("width", "height"):
        for bad in (True, False, 0, -1, 1.0, 4097, "8", None, math.inf, math.nan):
            rejected(lambda field=field, bad=bad: replace(sensors.tof, **{field: bad}))
    rejected(lambda: replace(sensors.camera, width=4096, height=4096))
    assert DepthConfig(4096, 1, 60, 60, 0, 1).width == 4096
    assert DepthConfig(1024, 1024, 60, 60, 0, 1).height == 1024
    for field in ("hfov_deg", "vfov_deg", "min_range_m", "max_range_m"):
        for bad in (True, False, "1", None, math.nan, math.inf, -math.inf, 10**400):
            rejected(lambda field=field, bad=bad: replace(sensors.tof, **{field: bad}))
    for field in ("hfov_deg", "vfov_deg"):
        for bad in (0, -1, 179, 180):
            rejected(lambda field=field, bad=bad: replace(sensors.tof, **{field: bad}))
    for minimum, maximum in ((-1, 2), (1, 1), (2, 1), (0, 0)):
        rejected(lambda minimum=minimum, maximum=maximum:
                 DepthConfig(8, 8, 60, 60, minimum, maximum))
    for bad in (True, False, -0.1, 1.1, math.nan, math.inf, "0.2", None):
        rejected(lambda bad=bad: replace(sensors, min_visible_fraction=bad))
    assert replace(sensors, min_visible_fraction=0).min_visible_fraction == 0
    assert replace(sensors, min_visible_fraction=1).min_visible_fraction == 1
    rejected(lambda: replace(sensors, tof=replace(sensors.tof, width=7)))
    for bad in ((), sensors.tof_mounts[:3], (*sensors.tof_mounts, sensors.camera_mount),
                (None, *sensors.tof_mounts[1:]), "four", None):
        rejected(lambda bad=bad: replace(sensors, tof_mounts=bad))
    rejected(lambda: replace(sensors, camera_mount=None))
    rejected(lambda: replace(sensors, camera={}))
    rejected(lambda: SceneConfig(0.2, 0.3, {}))
    for bad in ((0, 0), (0, 0, 0, 0), (0, math.inf, 0), (0, False, 0), "000"):
        rejected(lambda bad=bad: MountConfig(bad, IDENTITY))
    for bad in ((0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 2),
                (0, 0, 0, 1.000002), (0, 0, True, 0), (0, 0, math.nan, 1)):
        rejected(lambda bad=bad: MountConfig((0, 0, 0), bad))

    path = folder / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    reads, original = [], Path.read_text

    def count_read(file_path, *args, **kwargs):
        reads.append(file_path)
        return original(file_path, *args, **kwargs)

    with patch.object(Path, "read_text", count_read):
        assert load_config(path) == config
    assert reads == [path]
    legacy = {name: value[name] for name in ("vehicle", "target")}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_config(path) == SceneConfig(0.2, 0.3)
    invalid = []
    for key in value["sensors"]:
        altered = deepcopy(value)
        del altered["sensors"][key]
        invalid.append(altered)
    for route in ((), ("sensors",), ("sensors", "tof"), ("sensors", "camera"),
                  ("sensors", "tof_mounts"), ("sensors", "tof_mounts", "right"),
                  ("sensors", "camera_mount")):
        altered = deepcopy(value)
        node = altered
        for key in route:
            node = node[key]
        node["unknown"] = 1
        invalid.append(altered)
    for name in ("tof", "camera"):
        for key in value["sensors"][name]:
            altered = deepcopy(value)
            del altered["sensors"][name][key]
            invalid.append(altered)
    for name in ("position_m", "quaternion_xyzw"):
        altered = deepcopy(value)
        del altered["sensors"]["camera_mount"][name]
        invalid.append(altered)
    for key in ("width", "hfov_deg", "min_range_m"):
        altered = deepcopy(value)
        altered["sensors"]["tof"][key] = True
        invalid.append(altered)
    altered = deepcopy(value)
    del altered["sensors"]["tof_mounts"]["rear"]
    invalid.append(altered)
    for bad in (None, [], {}, True):
        altered = deepcopy(value)
        altered["sensors"] = bad
        invalid.append(altered)
    for altered in invalid:
        path.write_text(json.dumps(altered), encoding="utf-8")
        rejected(lambda: load_config(path))
    text = json.dumps(value)
    path.write_text(text.replace('"width": 8', '"width": 8, "width": 8', 1), encoding="utf-8")
    rejected(lambda: load_config(path))
    checks.append("immutable explicit sensor facts; strict finite/nonboolean values, ranges/FOV/resolution/8x8/mounts; one JSON read, legacy compatibility and complete exact-key sensor schema")
    return sensors


def verify_planes_and_pixels(checks, errors, renderers):
    config = DepthConfig(9, 7, 60, 45, 0, 12)
    rays = reference_rays(config)
    close(pixel_directions(config), rays, 1e-12)
    assert rays[0, 0, 1] > 0 and rays[0, 0, 2] > 0
    assert rays[-1, -1, 1] < 0 and rays[-1, -1, 2] < 0
    wall = ("far_wall", ((6, 7), (-10, 10), (10, 30)))
    nearer = ("upper_left", ((3, 3.5), (0.4, 1.8), (20.4, 21.8)))
    boxes = (wall, nearer)
    with MeshSensor(scene(boxes)) as sensor:
        record_renderer(sensor, renderers)
        frame = sensor.sample(config, Pose(ORIGIN))
        expected, identity = box_image(config, ORIGIN, boxes)
        assert np.all(np.isfinite(frame.range_m))
        close(frame.range_m, expected)
        errors["box_nearest_surface_max_abs_m"] = float(np.max(abs(frame.range_m - expected)))
        actual_names = np.array(frame.object_ids, dtype=object)[frame.object_index]
        expected_names = np.where(identity == 0, "map/far_wall", "map/upper_left")
        assert np.array_equal(actual_names, expected_names)
        near_pixels = np.argwhere(identity == 1)
        assert len(near_pixels) and np.all(near_pixels[:, 0] < config.height // 2)
        assert np.all(near_pixels[:, 1] < config.width // 2)
        close(frame.range_m[config.height//2, config.width//2], 6)
        assert frame.range_m[0, -1] > 6.1
        assert "vehicle" not in frame.object_ids and frame.t_ref == 12.5
        readonly(frame.range_m)
        readonly(frame.object_index)
        rejected(lambda: setattr(frame, "range_m", None), FrozenInstanceError)
        assert frame.range_m.shape == frame.object_index.shape == (7, 9)

    downward = DepthConfig(5, 3, 60, 40, 0, 10)
    angles = (0.0, math.pi/2, 0.0)
    ground_errors = []
    for declared in (True, False):
        with MeshSensor(scene(declared=declared)) as sensor:
            record_renderer(sensor, renderers)
            frame = sensor.sample(downward, Pose((0, 0, 4), quaternion(*angles)))
            expected = np.array([[4 / -rotate_euler(ray, *angles)[2] for ray in row]
                                 for row in reference_rays(downward)])
            assert np.all(np.isfinite(frame.range_m))
            close(frame.range_m, expected)
            assert np.all(frame.object_index == frame.object_ids.index("map/ground"))
            close(frame.range_m[1, 2], 4)
            ground_errors.append(float(np.max(abs(frame.range_m - expected))))
    errors["ground_plane_max_abs_m"] = max(ground_errors)

    # A wide-FOV triangle clipped across the camera plane previously lost
    # millimetres in interpolated positions, making this in-range hit look empty.
    # Compute the world ray independently by quaternion-vector multiplication,
    # then intersect the known x=1 wall plane analytically.
    witness_config = DepthConfig(64, 48, 120, 100, 0.05, 12)
    witness_origin = (-0.5481651582549905, -1.4890260899279797, 11.00635970341665)
    witness_quaternion = (-0.28231015743329757, 0.26836274025745954,
                          -0.45667413895892944, 0.799831948260142)
    normalized = np.asarray(witness_quaternion) / math.hypot(*witness_quaternion)
    vector = reference_rays(witness_config)[8, 55]
    direction = vector + 2 * np.cross(normalized[:3],
                                      np.cross(normalized[:3], vector) + normalized[3] * vector)
    expected_distance = (1 - witness_origin[0]) / direction[0]
    close(expected_distance, 11.9949247515, 1e-8)
    witness_wall = ("wide_fov_wall", ((1, 1.5), (-15, 15), (0, 20)))
    with MeshSensor(scene((witness_wall,), size=(40, 40, 20))) as sensor:
        record_renderer(sensor, renderers)
        witness = sensor.sample(witness_config, Pose(witness_origin, witness_quaternion))
        actual_distance = witness.range_m[8, 55]
        assert np.isfinite(actual_distance)
        close(actual_distance, expected_distance, 2e-5)
        assert abs(actual_distance - expected_distance) < 2e-5
        assert witness.object_ids[witness.object_index[8, 55]] == "map/wide_fov_wall"
        errors["wide_fov_clipped_wall_witness_abs_m"] = float(abs(actual_distance - expected_distance))
    checks.append("independent pinhole row/column signs, plane/closed-box Euclidean distances, nearest surfaces and asymmetric upper-left occlusion; metadata-boundary ground and wide-FOV clipped-wall regressions; vehicle self-geometry excluded")


def verify_range_and_coverage(checks, renderers):
    wall = ("wall", ((4, 5), (-8, 8), (10, 30)))
    base = DepthConfig(1, 1, 60, 60, 0, 8)
    measured = []
    with MeshSensor(scene((wall,))) as sensor:
        record_renderer(sensor, renderers)
        for config in (replace(base, min_range_m=4), replace(base, max_range_m=4)):
            frame = sensor.sample(config, Pose(ORIGIN))
            close(frame.range_m[0, 0], 4)
            measured.append(frame.range_m[0, 0])
        too_close = sensor.sample(replace(base, min_range_m=4.25), Pose(ORIGIN))
        assert np.isnan(too_close.range_m[0, 0])
        assert too_close.object_ids[too_close.object_index[0, 0]] == "map/wall"
        beyond = sensor.sample(replace(base, max_range_m=3.5), Pose(ORIGIN))
        assert np.isposinf(beyond.range_m[0, 0])
        measured.extend((too_close.range_m[0, 0], beyond.range_m[0, 0]))
    crossing_near = ("crossing_camera_plane", ((-1, 0.05), (-1, 1), (19, 21)))
    with MeshSensor(scene((wall, crossing_near))) as sensor:
        record_renderer(sensor, renderers)
        frame = sensor.sample(DepthConfig(3, 3, 60, 60, 0.1, 8), Pose(ORIGIN))
        assert np.all(np.isnan(frame.range_m))
        assert np.all(frame.object_index == frame.object_ids.index("map/crossing_camera_plane"))
    # This observed surface is just beyond max and within float32 uncertainty.
    # It must remain unavailable, rather than becoming a supposedly clear ray.
    near_max = ("near_max_wall", ((12.000001, 12.5), (-8, 8), (10, 30)))
    with MeshSensor(scene((near_max,))) as sensor:
        near_max_frame = sensor.sample(replace(base, max_range_m=12), Pose(ORIGIN))
        assert np.isnan(near_max_frame.range_m[0, 0])
        assert near_max_frame.object_ids[near_max_frame.object_index[0, 0]] == "map/near_max_wall"
    for declared in (True, False):
        with MeshSensor(scene(declared=declared)) as sensor:
            record_renderer(sensor, renderers)
            frame = sensor.sample(base, Pose(ORIGIN))
            assert (np.isposinf if declared else np.isnan)(frame.range_m[0, 0])
            measured.append(frame.range_m[0, 0])
        with MeshSensor(scene((wall,), declared=declared)) as sensor:
            close(sensor.sample(base, Pose(ORIGIN)).range_m[0, 0], 4)
    small = (10.0, 10.0, 10.0)
    with MeshSensor(scene(size=small, target=(-3, -3, 5))) as sensor:
        record_renderer(sensor, renderers)
        assert np.isnan(sensor.sample(base, Pose((0, 0, 5))).range_m[0, 0])
    within = ("within_domain", ((4, 4.5), (-1, 1), (4, 6)))
    with MeshSensor(scene((within,), size=small, target=(-3, -3, 5))) as sensor:
        close(sensor.sample(base, Pose((0, 0, 5))).range_m[0, 0], 4)
        # Even a surface inside the domain cannot establish the unknown segment
        # from an out-of-domain camera origin to that surface.
        outside = sensor.sample(base, Pose((6, 0, 5), quaternion(yaw=math.pi)))
        assert np.isnan(outside.range_m[0, 0]) and outside.object_index[0, 0] >= 0
    with MeshSensor(scene(size=small, target=(6, 0, 5), radius=0.25)) as sensor:
        frame = sensor.sample(base, Pose((0, 0, 5)))
        assert np.isnan(frame.range_m[0, 0])
        assert frame.object_ids[frame.object_index[0, 0]] == "target"
        camera = sensor.camera(base, Pose((0, 0, 5)), 0)
        assert camera.unoccluded_pixels == 0 and camera.visible_fraction == 0
        assert not np.any(camera.target_mask) and not np.any(camera.target_hint)
        outside_camera = sensor.camera(base, Pose((6, 0, 5), quaternion(yaw=math.pi)), 0)
        assert not np.any(outside_camera.target_hint) and outside_camera.unoccluded_pixels == 0
    # The nearest target point lies exactly on x=5, while its center is outside
    # the domain. A non-binary origin exposes float32 hit reconstruction error;
    # finite-hit boundary rounding must agree for full and target-only views.
    with MeshSensor(scene(size=small, target=(5.5, 0, 5), radius=0.5)) as sensor:
        camera = sensor.camera(base, Pose((-0.123, 0, 5)), 1)
        close(camera.depth.range_m[0, 0], 5.123)
        assert camera.unoccluded_pixels == 1 and camera.visible_fraction == 1
        assert camera.target_mask[0, 0] and np.array_equal(camera.target_hint, (1, 0, 0, 1, 1))
    summary = finite_summary(measured)
    assert summary == {"finite_count": 2, "mean_m": 4.0, "empty_count": 2, "unavailable_count": 2}
    assert finite_summary([math.inf, math.nan])["mean_m"] is None
    assert finite_summary([])["mean_m"] is None
    checks.append("inclusive min/max and uncertain near-max hits remain NaN; too-close and camera-plane-crossing occluders block farther surfaces; covered empty=inf, unknown=NaN; out-of-domain hits/origins rejected and partial-domain valid hits retained; finite-only aggregation")
    return summary


def analytic_sphere(config, center, radius, origin=ORIGIN):
    rays = reference_rays(config)
    relative = np.asarray(center) - origin
    projection = rays @ relative
    discriminant = projection*projection - (relative @ relative - radius*radius)
    hits = (discriminant >= 0) & (projection > 0)
    ranges = np.full(hits.shape, math.inf)
    ranges[hits] = projection[hits] - np.sqrt(discriminant[hits])
    return hits, ranges, discriminant


def verify_camera(checks, errors, renderers):
    config = DepthConfig(80, 60, 60, 45, 0.01, 15)
    center, radius = (6.0, 0.0, 20.0), 1.2
    analytic_mask, analytic_range, discriminant = analytic_sphere(config, center, radius)
    with MeshSensor(scene(target=center, radius=radius)) as sensor:
        record_renderer(sensor, renderers)
        camera = sensor.camera(config, Pose(ORIGIN), 1)
        assert camera.visible_fraction == 1 and camera.target_hint[0] == 1
        assert camera.unoccluded_pixels == int(np.count_nonzero(camera.target_mask)) > 0
        disagreements = int(np.count_nonzero(camera.target_mask ^ analytic_mask))
        assert disagreements <= 4, disagreements
        interior = analytic_mask & camera.target_mask & (discriminant > 0.1)
        differences = camera.depth.range_m[interior] - analytic_range[interior]
        assert len(differences) and float(np.min(differences)) >= -2e-5
        assert float(np.max(differences)) < 0.006
        errors["sphere_interior_max_abs_m"] = float(np.max(abs(differences)))
        errors["sphere_silhouette_disagreement_pixels"] = disagreements
        empty = sensor.camera(replace(config, max_range_m=2), Pose(ORIGIN), 0)
        assert empty.unoccluded_pixels == 0 and empty.visible_fraction == 0
        assert not np.any(empty.target_hint) and not np.any(empty.target_mask)
        for bad in (True, False, -0.1, 1.1, math.nan, math.inf, "0.2", None, 10**400):
            rejected(lambda bad=bad: sensor.camera(config, Pose(ORIGIN), bad))

    half = ("half_occluder", ((3, 3.2), (0, 3), (16, 24)))
    with MeshSensor(scene((half,), target=center, radius=radius)) as sensor:
        record_renderer(sensor, renderers)
        camera = sensor.camera(config, Pose(ORIGIN), 0.4)
        visible = camera.target_mask
        expected_visible = analytic_mask & (reference_rays(config)[..., 1] < 0)
        assert np.count_nonzero(visible ^ expected_visible) <= 4
        assert 0.48 <= camera.visible_fraction <= 0.52
        assert camera.visible_fraction == np.count_nonzero(visible) / camera.unoccluded_pixels
        assert camera.unoccluded_pixels > 10 * np.count_nonzero(visible) / 6
        assert camera.target_hint[0] == 1
        rows, cols = np.nonzero(visible)
        expected_box = (1, cols.min()/80, rows.min()/60, (cols.max()+1)/80, (rows.max()+1)/60)
        close(camera.target_hint, expected_box, 1e-7)
        assert np.all(cols >= 40) and np.all(camera.target_hint[1:] >= 0)
        assert np.all(camera.target_hint[1:] <= 1)
        assert np.all(camera.depth.object_index[visible] == camera.depth.object_ids.index("target"))
        assert np.all(np.isfinite(camera.depth.range_m[visible]))
        assert sensor.camera(config, Pose(ORIGIN), camera.visible_fraction).target_hint[0] == 1
        assert not np.any(sensor.camera(config, Pose(ORIGIN), 0.6).target_hint)
        readonly(camera.target_mask)
        readonly(camera.target_hint)
        rejected(lambda: setattr(camera, "visible_fraction", 0), FrozenInstanceError)
    full = ("full_occluder", ((3, 3.2), (-3, 3), (16, 24)))
    with MeshSensor(scene((full,), target=center, radius=radius)) as sensor:
        camera = sensor.camera(config, Pose(ORIGIN), 0)
        assert camera.unoccluded_pixels > 0 and camera.visible_fraction == 0
        assert not np.any(camera.target_mask) and not np.any(camera.target_hint)
    checks.append("analytic sphere interior/silhouette bounds account for declared mesh approximation; visible target mask, target-only denominator, half-open normalized box, inclusive visibility threshold, full occlusion and zero-denominator rejection")


class PublicOnlyScene(Scene):
    """Fail if a sensing backend reaches past the public geometry boundary."""

    def __getattribute__(self, name):
        if name in {"background", "vehicle", "target", "_geometry"}:
            try:
                armed = object.__getattribute__(self, "boundary_armed")
            except AttributeError:
                armed = False
            if armed:
                raise AssertionError(f"sensor read internal scene attribute {name}")
        return super().__getattribute__(name)

    def get_geometry(self, *, include_ids=None, exclude_ids=()):
        assert include_ids is None and set(exclude_ids) == {"vehicle"}
        self.boundary_calls.append((include_ids, tuple(sorted(exclude_ids))))
        return self.boundary_snapshot


def verify_poses_and_batch(checks, errors, renderers):
    wall = ("wall", ((4, 5), (-8, 8), (10, 30)))
    base = scene((wall,))
    guarded = PublicOnlyScene(base.background, base.vehicle, base.target, base.t_ref)
    snapshot = Scene.get_geometry(guarded, exclude_ids={"vehicle"})
    object.__setattr__(guarded, "boundary_snapshot", snapshot)
    object.__setattr__(guarded, "boundary_calls", [])
    object.__setattr__(guarded, "boundary_armed", True)
    config = DepthConfig(7, 5, 30, 20, 0, 12)
    angles = tuple(math.radians(value) for value in (18, -14, 12))
    pose = Pose(ORIGIN, quaternion(*angles))
    poses = tuple(Pose((0.1 * (index % 3), 0, 20), quaternion(*angles)) for index in range(48))
    with MeshSensor(guarded) as sensor:
        record_renderer(sensor, renderers)
        first = sensor.sample(config, pose)
        expected, _ = box_image(config, ORIGIN, (wall,), angles)
        close(first.range_m, expected)
        errors["full_roll_pitch_yaw_box_max_abs_m"] = float(np.max(abs(first.range_m - expected)))
        before = first.range_m.copy()
        started = time.perf_counter()
        frames = sensor.sample_many(config, poses)
        elapsed = time.perf_counter() - started
        assert len(frames) == len(poses)
        for result, current_pose in zip(frames, poses):
            expected, _ = box_image(config, current_pose.position_m, (wall,), angles)
            close(result.range_m, expected)
            assert result.t_ref == 12.5
        assert np.array_equal(first.range_m, before)
        assert sensor.geometry_uploads == 1 and sensor.triangle_count > 14
        assert guarded.boundary_calls == [(None, ("vehicle",))]
        assert sensor.sample_many(config, ()) == ()
        rejected(lambda: sensor.sample(None, pose), TypeError)
        rejected(lambda: sensor.sample(config, ORIGIN), TypeError)
        renderer_info = dict(sensor.renderer_info)
    sensor.close()
    rejected(lambda: sensor.sample(config, pose), RuntimeError)
    rejected(lambda: sensor.sample_many(config, (pose,)), RuntimeError)
    rejected(lambda: sensor.camera(config, pose, 0.2), RuntimeError)
    assert np.array_equal(first.range_m, before)
    assert elapsed > 0 and math.isfinite(elapsed)

    body_angles = tuple(math.radians(value) for value in (23, -31, 48))
    mount_angles = tuple(math.radians(value) for value in (-17, 21, -12))
    body = Pose((1, 2, 3), quaternion(*body_angles))
    mount = MountConfig((0.4, -0.2, 0.1), quaternion(*mount_angles))
    composed = mounted_pose(body, mount)
    expected_origin = np.array(body.position_m) + rotate_euler(mount.position_m, *body_angles)
    close(composed.position_m, expected_origin, 1e-10)
    for vector in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        expected_direction = rotate_euler(rotate_euler(vector, *mount_angles), *body_angles)
        close(composed.as_transform().apply_vectors(vector), expected_direction, 1e-10)
    rejected(lambda: mounted_pose(ORIGIN, mount), TypeError)
    rejected(lambda: mounted_pose(body, Pose((0, 0, 0))), TypeError)
    checks.append("full roll/pitch/yaw and translated mounts preserve independent Euler geometry; sample_many preserves pose order/time, reuses one public snapshot/upload and leaves earlier images immutable; explicit/idempotent close rejects further sampling")
    return {"poses": len(poses), "pixels_per_pose": config.width * config.height,
            "sample_many_seconds": elapsed, "poses_per_second": len(poses) / elapsed,
            "geometry_uploads": 1, "renderer_info": renderer_info,
            "scope": "measured only for this small analytic fixture; sequential GPU render/readback loop"}


def verify_rig(sensors, checks, errors, renderers):
    walls = (("right", ((-20, 20), (-2.5, -2), (1, 39))),
             ("front", ((4, 4.5), (-20, 20), (1, 39))),
             ("left", ((-20, 20), (6, 6.5), (1, 39))),
             ("rear", ((-8.5, -8), (-20, 20), (1, 39))))
    # Keep the target behind this fixture's walls, so all ToF expected values
    # come solely from analytic boxes. The camera remains externally assembled.
    with MeshSensor(scene(walls, target=(30, 30, 20), radius=0.5)) as sensor:
        record_renderer(sensor, renderers)
        rig_config = replace(sensors, camera=DepthConfig(24, 18, 80, 60, 0.01, 30))
        rig = sample_rig(sensor, rig_config, Pose(ORIGIN))
        assert rig.tof_m.shape == (4, 8, 8) and rig.target_hint.shape == (5,)
        assert rig.t_ref == rig.camera.depth.t_ref == 12.5
        max_error = 0.0
        for channel, yaw, distance in zip(rig.tof_m, (-math.pi/2, 0, math.pi/2, math.pi), (2, 4, 6, 8)):
            expected, _ = box_image(rig_config.tof, ORIGIN, walls, (0, 0, yaw))
            close(channel, expected, 3e-5)
            max_error = max(max_error, float(np.max(abs(channel - expected))))
            local = reference_rays(rig_config.tof)[3, 3]
            close(channel[3, 3], distance / local[0], 3e-5)
        errors["right_front_left_rear_rig_max_abs_m"] = max_error
        readonly(rig.tof_m)
        readonly(rig.target_hint)
        rejected(lambda: setattr(rig, "tof_m", None), FrozenInstanceError)
        rejected(lambda: sample_rig(sensor, None, Pose(ORIGIN)), TypeError)
        assert sensor.geometry_uploads == 1
    checks.append("external rig assembly yields raw (right,front,left,rear)x8x8 row-major ranges, matches independent four-wall intersections and common scene time; camera and immutable hint remain separate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    checks, errors, renderers = [], {}, []
    with tempfile.TemporaryDirectory(prefix="sensor-check-", dir=args.results.parent) as temporary:
        sensors = verify_config(Path(temporary), checks)
        verify_planes_and_pixels(checks, errors, renderers)
        finite_ranges = verify_range_and_coverage(checks, renderers)
        verify_camera(checks, errors, renderers)
        performance = verify_poses_and_batch(checks, errors, renderers)
        verify_rig(sensors, checks, errors, renderers)
    result = {"status": "passed", "checks": checks, "check_groups": len(checks),
              "renderer_info": renderers, "measured_errors": errors,
              "finite_range_example": finite_ranges, "batch_fixture_performance": performance,
              "scope": "noiseless GPU pinhole pixel-center mesh sensing; no hardware/noise/zone-response or training claim"}
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"sensors: {len(checks)} check groups passed; result: {args.results}")


if __name__ == "__main__":
    main()
