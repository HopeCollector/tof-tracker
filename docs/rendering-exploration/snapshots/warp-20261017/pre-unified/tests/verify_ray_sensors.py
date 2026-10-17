"""Docker-only checks of sampled GPU ray geometry and sensor reductions.

The geometry oracle is independent NumPy float64 Moller-Trumbore plus analytic
spheres. It neither imports a GL renderer nor uses a denser reference image.
Every CPU tensor download below is an explicit test assertion boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch

from devtools.config import DepthConfig, load_config
from devtools.mesh import MeshMap, MeshObject
from devtools.ray_rendering import RayRenderer
from devtools.scene import GeometrySnapshot, Pose, Scene, SceneObject, Sphere
from devtools.sensors import (generate_rays, poses_to_tensor, sample_camera,
                             sample_depth, sample_rig, sample_rig_diagnostics,
                             process_samples)
from devtools._ray_processing import select_subray


ROOT = Path(__file__).resolve().parents[1]
ATOL = 3e-5
BOX_FACES = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))


def host(value):
    return value.detach().cpu().numpy()


def close(actual, expected, atol=ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True), (
        actual.tolist(), expected.tolist())


def rejected(action, errors=(TypeError, ValueError, RuntimeError)):
    try:
        action()
    except errors:
        return
    raise AssertionError("invalid operation was accepted")


def box(name, bounds):
    (x0, x1), (y0, y1), (z0, z1) = bounds
    vertices = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    return MeshObject(name, vertices, BOX_FACES, "obstacle")


def ground(half=10.):
    return MeshObject("ground", ((-half, -half, 0.), (half, -half, 0.),
                                  (half, half, 0.), (-half, half, 0.)),
                      ((0, 1, 2), (0, 2, 3)), "ground")


def snapshot(*geometries, bounds=None, t_ref=7.5):
    objects = []
    for name, geometry in geometries:
        role = geometry.kind if isinstance(geometry, MeshObject) else "target"
        objects.append(SceneObject(name, role, geometry))
    return GeometrySnapshot(tuple(objects), t_ref, bounds)


def sphere(center=(3., 0., 2.), radius=.3):
    return Sphere(Pose(tuple(float(v) for v in center)), radius)


def oracle(source, origins, directions, maximum):
    """Return radial distance, object ID and world point without GPU code.

    Mesh hits are two-sided, barycentrically bounded triangle intersections;
    analytic sphere hits choose the nearest nonnegative quadratic root. Shared
    face edges can select either triangle, so IDs/distances are the oracle and
    the returned triangle is checked separately for surface membership.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    maximum = np.broadcast_to(np.asarray(maximum, dtype=np.float64), (len(origins),))
    distance = np.full(len(origins), np.inf)
    identity = np.full(len(origins), -1, dtype=np.int64)
    for object_index, obj in enumerate(source.objects):
        geometry = obj.geometry
        candidate = np.full(len(origins), np.inf)
        if isinstance(geometry, Sphere):
            offset = origins - np.asarray(geometry.pose.position_m)
            b = np.sum(offset * directions, axis=-1)
            c = np.sum(offset * offset, axis=-1) - geometry.radius_m ** 2
            discriminant = b * b - c
            valid = discriminant >= -1e-12
            root = np.sqrt(np.maximum(discriminant, 0.))
            near, far = -b - root, -b + root
            candidate = np.where(valid & (near >= 0), near,
                                 np.where(valid & (far >= 0), far, np.inf))
        else:
            vertices = np.asarray(geometry.vertices_m, dtype=np.float64)
            for face in geometry.triangles:
                a, b, c = vertices[np.asarray(face)]
                edge1, edge2 = b - a, c - a
                p = np.cross(directions, edge2)
                determinant = p @ edge1
                valid = np.abs(determinant) > 1e-12
                inverse = np.zeros(len(origins))
                inverse[valid] = 1. / determinant[valid]
                delta = origins - a
                u = np.sum(delta * p, axis=-1) * inverse
                q = np.cross(delta, edge1)
                v = np.sum(directions * q, axis=-1) * inverse
                t = (q @ edge2) * inverse
                valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (t >= 0)
                candidate = np.minimum(candidate, np.where(valid, t, np.inf))
        nearer = (candidate < distance) & (candidate <= maximum)
        distance[nearer], identity[nearer] = candidate[nearer], object_index
    hit = identity >= 0
    distance[~hit] = 0
    point = origins + distance[:, None] * directions
    point[~hit] = 0
    return distance, identity, point


def check_trace(raw, source, origins, directions, maximum, atol=ATOL):
    origins, directions = np.asarray(origins), np.asarray(directions)
    count = len(origins)
    for name, shape, dtype in (("distance_m", (count,), torch.float32),
                               ("object_index", (count,), torch.int64),
                               ("hit_point_world_m", (count, 3), torch.float32),
                               ("triangle_index", (count,), torch.int64)):
        value = getattr(raw, name)
        assert value.shape == shape and value.dtype == dtype and value.is_cuda, (name, value.shape, value.dtype)
    assert raw.object_ids == tuple(obj.object_id for obj in source.objects)
    assert raw.t_ref == source.t_ref and raw.map_bounds_m == source.map_bounds_m
    expected_distance, expected_id, expected_point = oracle(source, origins, directions, maximum)
    actual_distance, actual_id = host(raw.distance_m), host(raw.object_index)
    assert np.array_equal(actual_id, expected_id), (actual_id.tolist(), expected_id.tolist())
    close(actual_distance, expected_distance, atol)
    close(host(raw.hit_point_world_m), expected_point, atol)
    missed = actual_id < 0
    assert np.all(host(raw.triangle_index)[missed] == -1)
    offsets, next_offset = {}, 0
    for i, obj in enumerate(source.objects):
        if isinstance(obj.geometry, MeshObject):
            offsets[i] = next_offset
            next_offset += len(obj.geometry.triangles)
    for i in np.flatnonzero(~missed):
        geometry = source.objects[int(actual_id[i])].geometry
        triangle = int(host(raw.triangle_index)[i])
        if isinstance(geometry, Sphere):
            assert triangle == -1
        else:
            local = triangle - offsets[int(actual_id[i])]
            assert 0 <= local < len(geometry.triangles), (triangle, local)
            a, b, c = np.asarray(geometry.vertices_m)[np.asarray(geometry.triangles[local])]
            point = host(raw.hit_point_world_m)[i]
            uv = np.linalg.lstsq(np.column_stack((b - a, c - a)), point - a, rcond=None)[0]
            close(a + uv[0] * (b - a) + uv[1] * (c - a), point, atol)
            assert uv.min() >= -atol and uv.sum() <= 1 + atol
    return {"rays": count, "hits": int(np.count_nonzero(~missed)),
            "max_abs_distance_m": float(np.abs(actual_distance - expected_distance).max(initial=0))}


def trace(renderer, origins, directions, maximum=12., **kwargs):
    origins = torch.as_tensor(origins, dtype=torch.float32, device=renderer.device)
    directions = torch.as_tensor(directions, dtype=torch.float32, device=renderer.device)
    return renderer.trace(origins, directions, maximum, **kwargs)


def independent_rays(poses, config):
    """Pinhole FLU rays, cell-major and subrow-major within each cell."""
    side = config.samples_per_axis
    rays = []
    for row in range(config.height):
        for col in range(config.width):
            for subrow in range(side):
                for subcol in range(side):
                    u = (col + (subcol + .5) / side) / config.width
                    v = (row + (subrow + .5) / side) / config.height
                    ray = np.asarray((1., (1 - 2 * u) * math.tan(math.radians(config.hfov_deg) / 2),
                                      (1 - 2 * v) * math.tan(math.radians(config.vfov_deg) / 2)))
                    rays.append(ray / np.linalg.norm(ray))
    rays = np.asarray(rays)
    origins, directions = [], []
    for pose in poses:
        matrix = np.asarray(pose.as_transform().matrix, dtype=float)
        origins.append(np.broadcast_to(matrix[:3, 3], rays.shape))
        directions.append(rays @ matrix[:3, :3].T)
    return np.concatenate(origins), np.concatenate(directions)


def verify_raw_geometry():
    source = snapshot(("ground", ground()),
                      ("thin", box("thin", ((1.7, 1.701), (-.018, .018), (1.95, 2.05)))),
                      ("offset", box("offset", ((3.1, 3.8), (.3, 1.2), (1.4, 2.5)))),
                      ("target", sphere((4., -.8, 2.), .45)))
    rng = np.random.default_rng(8421)
    origins = np.broadcast_to(np.asarray((0., 0., 2.)), (300, 3)).copy()
    directions = np.column_stack((rng.uniform(.4, 3., 300), rng.uniform(-1.2, 1.2, 300),
                                  rng.uniform(-1., .4, 300)))
    # Explicit narrow-surface and cross-triangle witnesses precede random rays.
    directions[:6] = ((9., 0., 0.), (1., 0., -.5), (1., .2, 0.),
                       (4., -.8, 0.), (-1., 0., 0.), (1., .001, .002))
    maximum = rng.uniform(1., 10., len(origins)).astype(np.float32)
    maximum[:6] = 12
    with RayRenderer(source) as renderer:
        maximum_cuda = torch.as_tensor(maximum, device=renderer.device)
        raw = trace(renderer, origins, directions, maximum_cuda)
        record = check_trace(raw, source, origins, directions, maximum)
        assert raw.object_ids[int(host(raw.object_index)[0])] == "thin"
        # The public core consumes rays, so nonunit inputs must still report metres.
        close(host(raw.distance_m)[:1], np.asarray((1.7,)))
    return record


def verify_spheres_and_limits():
    source = snapshot(("target", sphere((3., 0., 2.), .5)))
    origins = np.asarray(((0., 0., 2.), (0., .5, 2.), (3., 0., 2.),
                          (4., 0., 2.), (0., 0., 2.), (0., 0., 2.)))
    directions = np.asarray(((2., 0., 0.), (1., 0., 0.), (3., 0., 0.),
                             (1., 0., 0.), (1., 0., 0.), (1., 0., 0.)))
    maximum = np.asarray((2.5, 3., 1., 10., 2.49, 2.51), dtype=np.float32)
    with RayRenderer(source) as renderer:
        raw = trace(renderer, origins, directions, torch.as_tensor(maximum, device=renderer.device))
        record = check_trace(raw, source, origins, directions, maximum)
        close(host(raw.distance_m), np.asarray((2.5, 3., .5, 0., 0., 2.5)))
        filtered = trace(renderer, origins, directions, 12, only_object="target")
        check_trace(filtered, source, origins, directions, 12)
        empty = renderer.trace(torch.empty((0, 3), dtype=torch.float32, device=renderer.device),
                               torch.empty((0, 3), dtype=torch.float32, device=renderer.device), 4.)
        assert empty.distance_m.shape == (0,) and empty.hit_point_world_m.shape == (0, 3)
        rejected(lambda: renderer.trace(torch.zeros((1, 3)), torch.ones((1, 3)), 4.))
        rejected(lambda: renderer.trace(torch.zeros((1, 3), device=renderer.device, dtype=torch.float64),
                                        torch.ones((1, 3), device=renderer.device), 4.))
        rejected(lambda: renderer.trace(torch.zeros((1, 2), device=renderer.device),
                                        torch.ones((1, 3), device=renderer.device), 4.))
        rejected(lambda: renderer.trace(torch.zeros((2, 6), device=renderer.device)[:, :3],
                                        torch.ones((2, 3), device=renderer.device), 4.))
        rejected(lambda: trace(renderer, origins, directions, -1.))
        rejected(lambda: trace(renderer, origins, directions, 1e-100))
    # Exercise the documented float32 query guard with representable, well
    # separated axis-aligned witnesses. The float64 sphere oracle describes
    # their ideal geometry; production intersections retain float32 roundoff.
    # These checks deliberately avoid treating nextafter inputs as an exact
    # ideal-geometry guarantee at a single floating-point ULP.
    guard = 4e-6
    guard_cases = []
    config = DepthConfig(1, 1, 45., 45., .015, 4.)
    pose = Pose((0., 0., 2.))
    matrices = poses_to_tensor((pose,))
    for offset in (0., .5 * guard, 2 * guard):
        source = snapshot(("target", sphere((4.25 + offset, 0., 2.), .25)),
                          bounds=((-10., 10.), (-10., 10.), (0., 4.)))
        expected_d, expected_i, _ = oracle(source, ((0., 0., 2.),), ((1., 0., 0.),), 4. + guard)
        with RayRenderer(source) as renderer:
            raw = trace(renderer, ((0., 0., 2.),), ((1., 0., 0.),), 4.)
            actual_d = float(host(raw.distance_m)[0])
            actual_i = int(host(raw.object_index)[0])
            assert actual_i == int(expected_i[0])
            close(np.asarray((actual_d,)), expected_d, 1e-6)
            formal = sample_depth(renderer, config, matrices)
            value = float(host(formal.range_m)[0, 0, 0])
            if offset == 0:
                assert value == 4. and actual_d == 4.
                category = "finite_inclusive_4m"
            elif offset < guard:
                assert actual_i >= 0 and actual_d > 4.
                assert math.isnan(value), "guard hit beyond sensor range must not become a valid 4 m return"
                assert int(host(formal.object_index)[0, 0, 0]) == actual_i
                category = "guard_hit_out_of_range_NaN"
            else:
                assert actual_i == -1 and actual_d == 0.
                assert math.isinf(value) and value > 0
                category = "outside_guard_known_empty_inf"
            guard_cases.append({"ideal_surface_m": 4. + offset,
                                "raw_distance_m": actual_d, "measurement": category})
    return {**record, "tangent_range_m": 3., "inside_exit_range_m": .5,
            "inclusive_max_range_m": 2.5, "empty_batch": "passed",
            "scalar_limit_underflow_rejected": True, "float32_guard_m": guard,
            "guard_boundary_cases": guard_cases}


def verify_scene_updates_and_ownership():
    background = MeshMap(Path("synthetic-ray-check.obj"), (ground(),))
    scene = Scene(background, Sphere(Pose((0., 0., 2.)), .2), sphere(), t_ref=1.)
    source_a = scene.get_geometry(exclude_ids={"vehicle"})
    source_b = scene.with_target_position((6., 0., 2.)).get_geometry(exclude_ids={"vehicle"})
    assert source_a.get_object("target").pose.position_m == (3., 0., 2.)
    assert source_b.get_object("target").pose.position_m == (6., 0., 2.)
    with RayRenderer(source_a) as renderer:
        initial_geometry = renderer.geometry_info
        first = trace(renderer, ((0., 0., 2.),), ((1., 0., 0.),))
        first_values = tuple(host(getattr(first, name)).copy() for name in
                             ("distance_m", "object_index", "hit_point_world_m", "triangle_index"))
        renderer.sync(source_b)
        second = trace(renderer, ((0., 0., 2.),), ((1., 0., 0.),))
        check_trace(second, source_b, ((0., 0., 2.),), ((1., 0., 0.),), 12.)
        renderer.sync(source_a)
        third = trace(renderer, ((0., 0., 2.),), ((1., 0., 0.),))
        check_trace(third, source_a, ((0., 0., 2.),), ((1., 0., 0.),), 12.)
        assert second.scene_version > first.scene_version and third.scene_version > second.scene_version
        final_geometry = renderer.geometry_info
        assert final_geometry["geometry_uploads"] == initial_geometry["geometry_uploads"] == 1
        assert final_geometry["object_transform_updates"] == 2
        assert final_geometry["object_transform_upload_bytes"] - initial_geometry["object_transform_upload_bytes"] == 24
        for name, saved in zip(("distance_m", "object_index", "hit_point_world_m", "triangle_index"), first_values):
            assert np.array_equal(host(getattr(first, name)), saved), name
        assert first.distance_m.data_ptr() != third.distance_m.data_ptr()
    close(host(first.distance_m), np.asarray((2.7,)))
    rejected(lambda: trace(renderer, ((0., 0., 2.),), ((1., 0., 0.),)))
    renderer.close()
    return {"target_centres_A_B_A_m": [3., 6., 3.], "saved_output_after_close": "passed",
            "scene_versions": [first.scene_version, second.scene_version, third.scene_version],
            "public_geometry_world_coordinates": "passed"}


def verify_streams():
    source_a = snapshot(("target", sphere()))
    source_b = snapshot(("target", sphere((5., 0., 2.))))
    default = torch.cuda.current_stream()
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    origins = torch.tensor(((0., 0., 2.),), dtype=torch.float32, device="cuda")
    directions = torch.tensor(((1., 0., 0.),), dtype=torch.float32, device="cuda")
    stream_a.wait_stream(default)
    stream_b.wait_stream(default)
    with torch.cuda.stream(stream_a):
        renderer = RayRenderer(source_a)
    with torch.cuda.stream(stream_b):
        first = renderer.trace(origins, directions, 12.)
        # Same-stream consumption must see the query output without CPU waits.
        first_value = first.distance_m * 2
    with torch.cuda.stream(stream_a):
        renderer.sync(source_b)
    with torch.cuda.stream(stream_b):
        second = renderer.trace(origins, directions, 12.)
        second_value = second.distance_m * 2
    torch.cuda.synchronize()
    close(host(first_value), np.asarray((5.4,)))
    close(host(second_value), np.asarray((9.4,)))
    renderer.close()
    return {"construction_A_trace_B_sync_A_trace_B": "passed", "same_stream_consumption": "passed"}


def cpu_selection(distance, identity, maximum, quantile=.2, floor=.08):
    distance = np.asarray(distance, dtype=float)
    effective = np.where(np.asarray(identity) >= 0, distance, maximum)
    weight = np.where(effective >= maximum - 1e-4, floor,
                      np.maximum(floor, 1 / (effective ** 2 + .05)))
    weight *= np.maximum(.05, 1 - .35 * (effective / maximum) ** 2)
    order = np.argsort(effective, kind="stable")
    selected = np.searchsorted(np.cumsum(weight[order]), quantile * weight.sum(), side="left")
    return int(order[min(selected, len(order) - 1)])


def verify_weighted_reduction():
    distance = np.asarray(([2.] + [3.] * 15, [2.] * 2 + [3.] * 14,
                           [0.] * 16, [.005] * 16), dtype=np.float32).reshape(4, 1, 1, 16)
    identity = np.asarray(([0] + [1] * 15, [0] * 2 + [1] * 14,
                           [-1] * 16, [0] * 16), dtype=np.int64).reshape(4, 1, 1, 16)
    expected_indices = np.asarray([cpu_selection(d, i, 4.) for d, i in zip(distance[:, 0, 0], identity[:, 0, 0])])
    selected = select_subray(torch.as_tensor(distance, device="cuda"), torch.as_tensor(identity, device="cuda"),
                             4., reduction="weighted_quantile", quantile=.2, signal_floor=.08)
    actual_indices = host(selected).reshape(-1)
    # Equal-distance records may select different stable representatives; the
    # selected distance and ID must still belong to the same original record.
    flat_d, flat_i = distance.reshape(4, 16), identity.reshape(4, 16)
    actual_d = flat_d[np.arange(4), actual_indices]
    actual_i = flat_i[np.arange(4), actual_indices]
    expected_d = flat_d[np.arange(4), expected_indices]
    expected_i = flat_i[np.arange(4), expected_indices]
    close(actual_d, expected_d)
    assert np.array_equal(actual_i, expected_i)
    assert actual_d[0] == 3. and actual_d[1] == 2.
    nearest = select_subray(torch.as_tensor(distance[:2], device="cuda"),
                            torch.as_tensor(identity[:2], device="cuda"), 4., reduction="nearest")
    assert np.all(flat_d[:2][np.arange(2), host(nearest).reshape(-1)] == 2.)
    rng = np.random.default_rng(913)
    mixed_d = rng.uniform(.1, 3.9, (12, 1, 1, 16)).astype(np.float32)
    mixed_i = rng.integers(-1, 3, mixed_d.shape, dtype=np.int64)
    mixed_d[mixed_i < 0] = 0
    for quantile in (0., .2, .65, 1.):
        expected = np.asarray([cpu_selection(d, i, 4., quantile)
                               for d, i in zip(mixed_d[:, 0, 0], mixed_i[:, 0, 0])])
        actual = host(select_subray(torch.as_tensor(mixed_d, device="cuda"),
                                     torch.as_tensor(mixed_i, device="cuda"), 4.,
                                     reduction="weighted_quantile", quantile=quantile,
                                     signal_floor=.08)).reshape(-1)
        assert np.array_equal(actual, expected), (quantile, actual.tolist(), expected.tolist())
    return {"one_near_return_of_16_m": float(actual_d[0]), "two_near_returns_of_16_m": float(actual_d[1]),
            "no_return_keeps_identity_minus_one": int(actual_i[2]), "nearest_is_separate_policy": "passed",
            "mixed_miss_quantile_cases": 48}


def verify_camera_coverage():
    config = DepthConfig(4, 4, 60., 60., .015, 8., samples_per_axis=2)
    target = sphere((3., 0., 2.), .7)
    source = snapshot(("occluder", box("occluder", ((1.5, 1.6), (.04, .16), (1.2, 2.8)))),
                      ("target", target))
    poses = (Pose((0., 0., 2.)), Pose((0., -.1, 2.)))
    origins, directions = independent_rays(poses, config)
    distance, identity, _ = oracle(source, origins, directions, config.max_range_m)
    only_target = snapshot(("target", target))
    target_distance, target_identity, _ = oracle(only_target, origins, directions, config.max_range_m)
    shape = (len(poses), config.height, config.width, 4)
    visible = (identity.reshape(shape) == 1) & (distance.reshape(shape) >= config.min_range_m)
    potential = (target_identity.reshape(shape) >= 0) & (target_distance.reshape(shape) >= config.min_range_m)
    coverage = visible.mean(axis=-1)
    denominator = potential.sum(axis=(1, 2, 3))
    fraction = np.divide(visible.sum(axis=(1, 2, 3)), denominator,
                         out=np.zeros(len(poses)), where=denominator > 0)
    with RayRenderer(source) as renderer:
        matrices = poses_to_tensor(poses, renderer.device)
        actual_origins, actual_directions = generate_rays(matrices, config)
        close(host(actual_origins), origins)
        close(host(actual_directions), directions)
        result = sample_camera(renderer, config, matrices, 0.)
        shaped_d, shaped_i = distance.reshape(shape), identity.reshape(shape)
        nearest = np.argmin(np.where(shaped_i >= 0, shaped_d, config.max_range_m), axis=-1)
        expected_d = np.take_along_axis(shaped_d, nearest[..., None], axis=-1)[..., 0]
        expected_i = np.take_along_axis(shaped_i, nearest[..., None], axis=-1)[..., 0]
        close(host(result.depth.raw.distance_m), expected_d)
        assert np.array_equal(host(result.depth.object_index), expected_i)
        close(host(result.target_coverage), coverage)
        assert np.array_equal(host(result.target_mask), coverage > 0)
        close(host(result.visible_fraction), fraction, 1e-12)
        assert np.array_equal(host(result.unoccluded_samples), denominator)
        assert np.array_equal(host(result.unoccluded_pixels), potential.any(axis=-1).sum(axis=(1, 2)))
        assert np.any((coverage > 0) & (coverage < 1)), coverage
        # Subray occupancy supervises the mask even when a nearer subray gives
        # the pixel's representative depth a different object's identity.
        other_representative = (coverage > 0) & (host(result.depth.object_index) != 1)
        assert np.any(other_representative)
        for batch in range(len(poses)):
            rows, cols = np.nonzero(coverage[batch] > 0)
            expected_hint = np.asarray((1., cols.min() / config.width, rows.min() / config.height,
                                        (cols.max() + 1) / config.width, (rows.max() + 1) / config.height))
            close(host(result.target_hint)[batch], expected_hint)
        queries_before = renderer.geometry_info["trace_calls"]
        no_target = sample_camera(renderer, config, matrices, 0., target_id=None)
        assert renderer.geometry_info["trace_calls"] - queries_before == 1
        assert not bool(no_target.target_mask.any()) and not bool(no_target.target_coverage.any())
        assert not bool(no_target.target_hint.any()) and not bool(no_target.visible_fraction.any())
    return {"visible_subrays": visible.sum(axis=(1, 2, 3)).tolist(),
            "target_only_subrays": denominator.tolist(), "visible_fraction": fraction.tolist(),
            "partial_coverage_pixels": int(np.count_nonzero((coverage > 0) & (coverage < 1))),
            "mask_pixels_with_other_representative_id": int(np.count_nonzero(other_representative))}


def verify_tof_ranges_and_small_target():
    config = load_config(ROOT / "configs" / "devtools.json").sensors
    assert (config.tof.hfov_deg, config.tof.vfov_deg) == (45., 45.)
    assert (config.tof.min_range_m, config.tof.max_range_m) == (.015, 4.)
    assert config.tof.samples_per_axis == 4 and config.camera.samples_per_axis == 2
    pose = Pose((0., 0., 2.))
    matrices = poses_to_tensor((pose,))
    # A tiny opaque front box is nearer than the minimum measurement range.
    # It must stay the nearest hit and cause NaN, never reveal the far target.
    near = box("near", ((.003, .005), (-.02, .02), (1.98, 2.02)))
    source = snapshot(("near", near), ("target", sphere((2., 0., 2.), .3)))
    with RayRenderer(source) as renderer:
        result = sample_depth(renderer, config.tof, matrices, reduction="weighted_quantile")
        assert np.isnan(host(result.range_m)).all()
        assert np.all(host(result.object_index) == 0)
    # A sphere centered across both grid axes supplies genuine subray hits in
    # several cells, with a radius small enough to expose partial coverage.
    source = snapshot(("target", sphere((3., 0., 2.), .16)))
    origins, directions = independent_rays((pose,), config.tof)
    expected_distance, expected_id, _ = oracle(source, origins, directions, 4.)
    with RayRenderer(source) as renderer:
        result = sample_depth(renderer, config.tof, matrices, reduction="weighted_quantile")
        raw = result.raw
        assert raw is not None
        close(host(raw.subray_distance_m).reshape(-1), expected_distance)
        assert np.array_equal(host(raw.subray_object_index).reshape(-1), expected_id)
        hit_cells = (expected_id.reshape(8, 8, 16) >= 0).any(axis=-1)
        assert int(hit_cells.sum()) == 4
        assert np.count_nonzero(expected_id >= 0) > 0
    # The original screenshot's nearest sphere surface is beyond the newly
    # agreed 4 m range, so it is now correctly absent even with more samples.
    origin = np.asarray((.004000000189989805, -17.604000091552734, 1.5))
    target = np.asarray((.7369999885559082, -13.317999839782717, 1.5))
    closest = float(np.linalg.norm(target - origin) - .3)
    assert closest > 4.
    yaw = math.radians(78.68399810791016)
    old_pose = Pose(tuple(origin), (0., 0., math.sin(yaw / 2), math.cos(yaw / 2)))
    source = snapshot(("target", sphere(tuple(target), .3)),
                      bounds=((-100., 100.), (-100., 100.), (-100., 100.)))
    with RayRenderer(source) as renderer:
        old_result = sample_depth(renderer, config.tof, poses_to_tensor((old_pose,)), reduction="weighted_quantile")
        assert np.isposinf(host(old_result.range_m)).all()
        assert np.all(host(old_result.raw.subray_object_index) == -1)
    return {"tof_fov_deg": 45., "tof_range_m": [.015, 4.],
            "small_target_cells_hit_by_samples": int(hit_cells.sum()),
            "small_target_subray_hits": int(np.count_nonzero(expected_id >= 0)),
            "old_screenshot_nearest_sphere_surface_m": closest,
            "old_screenshot_new_range_result": "no return; target outside 4 m"}


def verify_known_domain_and_batch_rig():
    # With declared bounds, unknown portions of a sampled cell cannot become
    # a confident empty return or silently disappear during nearest reduction.
    source = snapshot(bounds=((-1., 1.), (-1., 1.), (0., 4.)))
    config = DepthConfig(2, 2, 45., 45., .015, 4., samples_per_axis=2)
    poses = (Pose((0., 0., 2.)), Pose((0., .5, 2.)))
    with RayRenderer(source) as renderer:
        result = sample_depth(renderer, config, poses_to_tensor(poses))
        assert np.isnan(host(result.range_m)).all()
    no_domain = snapshot()
    with RayRenderer(no_domain) as renderer:
        result = sample_depth(renderer, config, poses_to_tensor(poses))
        assert np.isnan(host(result.range_m)).all(), "unknown domain must not establish empty space"
    # The first selected ray is fully known and empty, while the two rays on
    # the other side leave the map. The cell still has to remain unknown.
    mixed_config = DepthConfig(1, 1, 45., 45., .015, 4., samples_per_axis=2)
    mixed_source = snapshot(bounds=((-1., 5.), (-.1, 5.), (0., 5.)))
    origins, directions = independent_rays((poses[0],), mixed_config)
    endpoints = origins + mixed_config.max_range_m * directions
    limits = np.asarray(mixed_source.map_bounds_m)
    known = ((endpoints >= limits[:, 0]) & (endpoints <= limits[:, 1])).all(axis=-1)
    assert known[0] and int(known.sum()) == 2
    with RayRenderer(mixed_source) as renderer:
        result = sample_depth(renderer, mixed_config, poses_to_tensor((poses[0],)))
        assert np.isnan(host(result.range_m)).all()
        # The raw kernel safely misses invalid input rays. Measurement policy
        # must not reinterpret those safe misses as known free-space returns.
        matrix = poses_to_tensor((poses[0],))
        bad_origins = torch.as_tensor(origins, dtype=torch.float32, device=renderer.device).contiguous()
        for invalid in (0., math.nan):
            bad_directions = torch.full_like(bad_origins, invalid)
            hits = renderer.trace(bad_origins, bad_directions, mixed_config.max_range_m)
            assert np.all(host(hits.object_index) == -1)
            depth = process_samples(hits, bad_origins, bad_directions, matrix, mixed_config)
            assert np.isnan(host(depth.range_m)).all()
    # A hit contained in the known domain remains usable even when that ray's
    # max-range endpoint would be outside: visibility stops at its first hit.
    source = snapshot(("wall", box("wall", ((.5, .6), (-.9, .9), (1.1, 2.9)))),
                      bounds=((-1., 1.), (-1., 1.), (0., 4.)))
    with RayRenderer(source) as renderer:
        result = sample_depth(renderer, config, poses_to_tensor((poses[0],)))
        assert np.isfinite(host(result.range_m)).all()
    source_without_bounds = GeometrySnapshot(source.objects, source.t_ref, None)
    with RayRenderer(source_without_bounds) as renderer:
        result = sample_depth(renderer, config, poses_to_tensor((poses[0],)))
        assert np.isfinite(host(result.range_m)).all(), "real hits remain known without declared bounds"
    # Formal rig batching and diagnostics must refer to identical ray samples.
    settings = load_config(ROOT / "configs" / "devtools.json").sensors
    source = snapshot(("ground", ground()), ("target", sphere((3., 0., 2.), .3)))
    with RayRenderer(source) as renderer:
        matrices = poses_to_tensor(poses)
        batched = sample_rig(renderer, settings, matrices)
        diagnostic = sample_rig_diagnostics(renderer, settings, matrices)
        close(host(batched.tof_m), host(diagnostic.rig.tof_m))
        close(host(batched.target_hint), host(diagnostic.rig.target_hint))
        assert batched.tof_m.shape == (2, 4, 8, 8)
        assert diagnostic.tof_raw.subray_distance_m.shape == (8, 8, 8, 16)
        for i in range(2):
            single = sample_rig(renderer, settings, matrices[i:i + 1])
            close(host(batched.tof_m)[i:i + 1], host(single.tof_m))
            close(host(batched.camera.target_coverage)[i:i + 1], host(single.camera.target_coverage))
        assert batched.tof_m.is_cuda and batched.camera.target_mask.is_cuda
    return {"sampled_unknown_coverage": "NaN", "known_first_hit_before_domain_exit": "passed",
            "mixed_known_and_unknown_cell": {"known_rays": 2, "unknown_rays": 2, "result": "NaN"},
            "missing_domain_miss": "NaN", "invalid_zero_nan_directions": "NaN",
            "rig_batch_vs_single": "passed", "same_pass_diagnostics": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "outputs" / "ray_sensors" / "verification.json")
    parser.add_argument("--only", nargs="*", help="Run named groups only; default runs all groups.")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    checks = (("raw_mesh_sphere_oracle", verify_raw_geometry),
              ("sphere_tangent_inside_and_limits", verify_spheres_and_limits),
              ("world_geometry_updates_output_ownership", verify_scene_updates_and_ownership),
              ("construction_and_query_streams", verify_streams),
              ("tof_weighted_quantile", verify_weighted_reduction),
              ("camera_subray_coverage_and_visibility", verify_camera_coverage),
              ("tof_new_range_small_and_old_target", verify_tof_ranges_and_small_target),
              ("known_domain_and_batch_rig", verify_known_domain_and_batch_rig))
    if args.only:
        unknown = set(args.only) - {name for name, _ in checks}
        if unknown:
            parser.error("unknown groups: " + ", ".join(sorted(unknown)))
    records = []
    for name, action in checks:
        if args.only and name not in args.only:
            continue
        try:
            records.append({"name": name, "status": "passed", "detail": action()})
            print(f"{name}: passed", flush=True)
        except Exception as error:
            records.append({"name": name, "status": "failed", "error_type": type(error).__name__,
                            "error": str(error), "traceback": traceback.format_exc()})
            print(f"{name}: failed: {type(error).__name__}: {error}", flush=True)
    result = {"status": "passed" if all(record["status"] == "passed" for record in records) else "failed",
              "device": torch.cuda.get_device_name(), "checks": records,
              "oracle": "independent float64 ray-triangle and analytic sphere; independent subray pinhole and weighted CDF"}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
