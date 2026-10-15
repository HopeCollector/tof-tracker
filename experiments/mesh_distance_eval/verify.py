"""Independent geometry, first-derivative and scene-update verification.

Run inside the experiment Docker image. Raw inputs/outputs are retained in NPZ;
JSON distinguishes verified claims from known geometric limitations. This is
an experiment, not a new production ESDF API or a claim of exact union SDFs.
"""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from devtools.config import SceneConfig
from devtools.mesh import MeshObject
from devtools.scene import Pose, Scene

from .backend import MeshDistance, SampledField, triangle_distance as candidate_triangle_distance
from .fixtures import (box_sdf, box_union_sdf, ground_distance, joined_mesh,
                       make_box, make_scene, mesh_signed_distance, snapshot_oracle,
                       sphere_sdf, transform_mesh, triangle_distance, winding_number,
                       write_obj)


def _stats(error):
    error = np.asarray(error, dtype=np.float64)
    finite = error[np.isfinite(error)]
    return {"count": int(error.size), "nonfinite": int(error.size - finite.size),
            "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
            "mean_abs": float(np.mean(np.abs(finite))) if finite.size else None,
            "p95_abs": float(np.quantile(np.abs(finite), 0.95)) if finite.size else None}


def _tensor(points, grad=False):
    return torch.tensor(np.asarray(points), dtype=torch.float32, device="cuda", requires_grad=grad)


def _numpy(value):
    return value.detach().cpu().numpy()


def _evaluate(query, points):
    with torch.no_grad():
        return _numpy(query(_tensor(points)))


def _gradient(query, points):
    p = _tensor(points, grad=True)
    d = query(p)
    gradient, = torch.autograd.grad(torch.nan_to_num(d).sum(), p)
    return _numpy(d), _numpy(gradient)


def _finite_difference(query, points, epsilon=1e-4):
    points = np.asarray(points, dtype=np.float64)
    result = np.zeros_like(points)
    for axis in range(3):
        delta = np.eye(3)[axis] * epsilon
        result[:, axis] = (_evaluate(query, points + delta) - _evaluate(query, points - delta)) / (2 * epsilon)
    return result


def _cpu_difference(query, points, epsilon=1e-5):
    points = np.asarray(points, dtype=np.float64)
    return np.stack([(query(points + np.eye(3)[axis] * epsilon) -
                      query(points - np.eye(3)[axis] * epsilon)) / (2 * epsilon)
                     for axis in range(3)], axis=-1)


class _Results:
    def __init__(self):
        self.checks = {}
        self.arrays = {}
        self.details = {}
        self.limitations = []

    def check(self, name, condition, **detail):
        self.checks[name] = {"passed": bool(condition), **detail}

    def close(self, name, actual, expected, atol):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = actual - expected
        self.check(name, np.isfinite(error).all() and np.max(np.abs(error), initial=0) <= atol,
                   tolerance=atol, error=_stats(error))

    def save(self, name, **arrays):
        for key, value in arrays.items():
            self.arrays[f"{name}__{key}"] = np.asarray(value)


def _oracle_self_checks(result):
    lo, hi = (-1.0, -0.8, 1.0), (1.0, 0.8, 3.0)
    box = make_box("oracle_box", lo, hi)
    rng = np.random.default_rng(8128)
    points = rng.uniform((-2, -2, 0.2), (2, 2, 4.0), size=(256, 3))
    analytical = box_sdf(points, lo, hi)
    vertices, faces = joined_mesh((box,))
    unsigned, closest = triangle_distance(points, vertices, faces)
    winding = winding_number(points, vertices, faces)
    result.close("oracle_triangle_vs_analytic_box", unsigned, np.abs(analytical), 2e-12)
    result.check("oracle_winding_vs_analytic_inside", np.array_equal(winding > 0.5, analytical < 0))
    result.close("oracle_closest_point_distance", np.linalg.norm(points - closest, axis=1), unsigned, 2e-12)
    # A single-box union must reduce to the independent closed-form box SDF.
    result.close("oracle_union_single_box", box_union_sdf(points, ((lo, hi),)), analytical, 2e-12)
    result.save("oracle", points=points, analytic=analytical, triangle_unsigned=unsigned,
                closest=closest, winding=winding)


def _box_geometry(result, output_dir):
    lo, hi = (-1.0, -0.8, 1.0), (1.0, 0.8, 3.0)
    box = make_box("box", lo, hi)
    obj_path = write_obj(output_dir / "fixtures" / "box.obj", (box,))
    scene = Scene.from_obj(obj_path, SceneConfig(0.2, 0.3), Pose((-3, -3, 1)), (3, 3, 2))
    snapshot = scene.get_geometry(exclude_ids={"vehicle"})
    backend = MeshDistance(snapshot, device="cuda")
    feature_points = np.array(((1.4, 0.13, 1.8), (1.2, 1.0, 2.0), (1.2, 1.0, 3.2),
                               (0.8, 0.13, 2.0), (1.0001, 0.13, 1.8), (0.9999, 0.13, 1.8),
                               (1.0, 0.13, 1.8), (0.0, 0.0, 2.0), (1.0, 0.8, 3.0)), dtype=np.float64)
    rng = np.random.default_rng(1729)
    random_points = rng.uniform((-2.5, -2.5, 0.2), (2.5, 2.5, 4.5), size=(384, 3))
    points = np.concatenate((feature_points, random_points))
    expected_mesh = box_sdf(points, lo, hi)
    expected_query = snapshot_oracle(points, snapshot)
    result.save("box", points=points, expected_mesh=expected_mesh, expected_query=expected_query,
                feature_labels=np.array(("face", "edge", "vertex", "inside", "near_outside", "near_inside",
                                         "on_face", "medial_tie", "on_vertex")))
    for method in ("torch", "tape"):
        mesh = _evaluate(lambda p: backend.mesh_query(p, method=method), points)
        query = _evaluate(lambda p: backend.query(p, method=method), points)
        normal = _evaluate(lambda p: backend.mesh_query(p, sign="normal", method=method), points)
        result.close(f"box_{method}_mesh_analytic", mesh, expected_mesh, 3e-5)
        result.close(f"box_{method}_public_composition", query, expected_query, 3e-5)
        result.close(f"box_{method}_normal_disjoint", normal, expected_mesh, 3e-5)
        result.check(f"box_{method}_near_surface_sign", mesh[4] > 0 and mesh[5] < 0)
        result.save("box", **{f"mesh_{method}": mesh, f"query_{method}": query, f"normal_{method}": normal})
    # Smooth features retain unique nearest points, even when the nearest feature
    # is an edge or vertex. Exact surface/medial ties are reported separately.
    smooth_points = np.array(((1.37, 0.13, 1.83), (1.23, 1.07, 2.17), (1.23, 1.07, 3.31),
                              (0.81, 0.13, 2.17), (-1.39, -0.17, 2.21)))
    analytic_gradient = _cpu_difference(lambda p: box_sdf(p, lo, hi), smooth_points)
    gradients = {}
    for method in ("torch", "tape"):
        query = lambda p: backend.mesh_query(p, method=method)
        distance, gradient = _gradient(query, smooth_points)
        fd = _finite_difference(query, smooth_points)
        result.close(f"box_{method}_gradient_analytic", gradient, analytic_gradient, 3e-5)
        result.close(f"box_{method}_gradient_own_fd", gradient, fd, 3e-3)
        result.save("box_gradient", **{f"distance_{method}": distance, f"gradient_{method}": gradient,
                                        f"fd_{method}": fd})
        gradients[method] = gradient
        tie_d, tie_grad = _gradient(query, feature_points[[6, 7, 8]])
        result.check(f"box_{method}_nonsmooth_finite", np.isfinite(tie_d).all() and np.isfinite(tie_grad).all(),
                     note="Exact surface, medial ties and vertex: selected derivative only; no uniqueness claim.")
        result.save("box_nonsmooth", **{f"distance_{method}": tie_d, f"gradient_{method}": tie_grad})
    result.close("box_two_routes_gradient_agree", gradients["torch"], gradients["tape"], 3e-5)
    result.save("box_gradient", points=smooth_points, analytic=analytic_gradient)
    return scene, backend


def _sphere_geometry(result):
    center, radius = (0.0, 0.0, 2.0), 0.6
    scene = make_scene(target_position=center, target_radius=radius)
    snapshot = scene.get_geometry(exclude_ids={"vehicle"})
    backend = MeshDistance(snapshot, device="cuda")
    directions = np.array(((1, .2, .3), (-.3, 1, .2), (.2, -.3, 1), (.2, .3, -1)), dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = np.concatenate([np.asarray(center) + directions * r for r in (.2, .5999, .6001, 1.1)])
    expected = sphere_sdf(points, center, radius)
    for method in ("torch", "tape"):
        query = lambda p: backend.query(p, method=method)
        distance, gradient = _gradient(query, points)
        expected_grad = np.tile(directions, (4, 1))
        result.close(f"sphere_exact_{method}_distance", distance, expected, 3e-6)
        result.close(f"sphere_exact_{method}_gradient", gradient, expected_grad, 3e-6)
        result.close(f"sphere_exact_{method}_gradient_fd", gradient, _finite_difference(query, points), 3e-3)
        result.save("sphere_exact", **{f"distance_{method}": distance, f"gradient_{method}": gradient})
    result.save("sphere_exact", points=points, expected=expected)
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.vertices += center
    sphere_mesh = MeshObject("faceted_sphere", tuple(map(tuple, mesh.vertices.tolist())),
                             tuple(map(tuple, mesh.faces.tolist())), "obstacle")
    backend_mesh = MeshDistance(make_scene((sphere_mesh,)).get_geometry(exclude_ids={"vehicle"}), device="cuda")
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(320, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points = np.asarray(center) + directions * rng.uniform(.3, 1.2, size=(320, 1))
    oracle = mesh_signed_distance(points, (sphere_mesh,))
    exact_sphere = sphere_sdf(points, center, radius)
    values = {}
    for method in ("torch", "tape"):
        values[method] = _evaluate(lambda p: backend_mesh.mesh_query(p, method=method), points)
        result.close(f"sphere_mesh_{method}_vs_triangle_oracle", values[method], oracle, 3e-5)
    result.details["sphere_mesh_approximation"] = {
        "vertices": len(sphere_mesh.vertices_m), "triangles": len(sphere_mesh.triangles),
        "mesh_vs_exact_sphere_error_m": _stats(oracle - exact_sphere),
        "note": "Inscribed triangulated sphere shape error is separate from backend query error."}
    result.save("sphere_mesh", points=points, triangle_oracle=oracle, exact_sphere=exact_sphere, **values)


def _rotation_and_gap(result):
    local_lo, local_hi = (-.8, -.5, -.35), (.8, .5, .35)
    axis = np.array((1., 2., 3.)) / np.sqrt(14)
    quaternion = tuple(float(v) for v in (*tuple(axis * np.sin(.4)), np.cos(.4)))
    pose = Pose((.1, -.2, 2.3), quaternion)
    box = transform_mesh(make_box("rotated", local_lo, local_hi), pose)
    backend = MeshDistance(make_scene((box,)).get_geometry(exclude_ids={"vehicle"}), device="cuda")
    local = np.array(((1.1, .1, .07), (.91, .67, .17), (.91, .67, .51),
                      (.6, .11, .03), (-1.07, .13, -.07)))
    world = np.array([pose.body_to_world(tuple(row)) for row in local])
    expected = box_sdf(local, local_lo, local_hi)
    rotation = np.column_stack([np.asarray(pose.body_to_world(tuple(np.eye(3)[i]))) - pose.position_m
                                for i in range(3)])
    expected_gradient = _cpu_difference(lambda p: box_sdf(p, local_lo, local_hi), local) @ rotation.T
    for method in ("torch", "tape"):
        distance, gradient = _gradient(lambda p: backend.mesh_query(p, method=method), world)
        result.close(f"full_3d_pose_{method}_distance", distance, expected, 3e-5)
        result.close(f"full_3d_pose_{method}_gradient", gradient, expected_gradient, 3e-5)
        result.save("full_3d_pose", **{f"distance_{method}": distance, f"gradient_{method}": gradient})
    result.save("full_3d_pose", local=local, world=world, quaternion=quaternion,
                expected=expected, expected_gradient=expected_gradient)
    boxes = (((-1.5, -1.0, .5), (-.06, 1.0, 3.5)), ((.06, -1.0, .5), (1.5, 1.0, 3.5)))
    obstacles = tuple(make_box(f"gap_{i}", *box) for i, box in enumerate(boxes))
    backend = MeshDistance(make_scene(obstacles).get_geometry(exclude_ids={"vehicle"}), device="cuda")
    points = np.array(((-.04, .13, 1.7), (.04, .13, 1.7), (-.07, .13, 1.7),
                       (.07, .13, 1.7), (0., .13, 1.7)))
    expected = np.min([box_sdf(points, *box) for box in boxes], axis=0)
    for method in ("torch", "tape"):
        actual = _evaluate(lambda p: backend.query(p, method=method), points)
        result.close(f"narrow_gap_{method}", actual, expected, 3e-5)
        result.save("narrow_gap", **{method: actual})
    result.save("narrow_gap", points=points, expected=expected)


def _sliver_triangle(result):
    # Long, thin triangles stress the selected-face Torch recomputation without
    # relying on BVH selection or sign. Face, edge and vertex queries are smooth.
    vertices = np.array(((0., 0., 0.), (8., 0., 0.), (8., .001, 0.)))
    points = np.array(((6., .00025, .03), (6., .002, .03), (8.3, .002, .03),
                       (2., .0001, -.017), (4., -.003, .011)))
    expected, closest = triangle_distance(points, vertices, ((0, 1, 2),))
    expected_gradient = (points - closest) / expected[:, None]
    triangles = _tensor(np.repeat(vertices[None], len(points), axis=0))
    distance, gradient = _gradient(lambda p: candidate_triangle_distance(p, triangles), points)
    result.close("sliver_triangle_torch_distance", distance, expected, 3e-6)
    result.close("sliver_triangle_torch_gradient", gradient, expected_gradient, 3e-5)
    result.save("sliver_triangle", vertices=vertices, points=points, distance=distance,
                expected=expected, closest=closest, gradient=gradient, expected_gradient=expected_gradient)
    unsupported = make_box("unsupported_sliver", (-.0005, -.8, 1.), (.0005, .8, 3.))
    rejection = None
    try:
        MeshDistance(make_scene((unsupported,)).get_geometry(exclude_ids={"vehicle"}), device="cuda")
    except ValueError as error:
        rejection = str(error)
    result.check("warp_sliver_mesh_rejected_explicitly", rejection is not None,
                 error=rejection,
                 note="Robust selected-triangle recomputation cannot repair a native BVH selecting the wrong triangle.")
    result.limitations.append({"name": "sliver_mesh_quality_guard", "status": "explicitly_rejected_input",
                               "box_width_m": .001, "box_height_m": 2., "error": rejection,
                               "note": "Standalone triangle recomputation correctness does not establish native mesh-query correctness on slivers."})


def _overlap_limit(result):
    boxes = (((-2., -.5, 1.), (2., .5, 4.)), ((-.5, -2., 1.), (.5, 2., 4.)))
    obstacles = tuple(make_box(f"cross_{i}", *box) for i, box in enumerate(boxes))
    backend = MeshDistance(make_scene(obstacles).get_geometry(exclude_ids={"vehicle"}), device="cuda")
    points = np.array(((0., 0., 2.5), (.1, .1, 2.5), (.6, .2, 2.5), (.8, .1, 2.5),
                       (1.5, .7, 2.5), (.7, .7, 2.5), (2.3, 2.3, 2.5)))
    nearest_original = mesh_signed_distance(points, obstacles)
    true_union = box_union_sdf(points, boxes)
    per_object_min = np.min([box_sdf(points, *box) for box in boxes], axis=0)
    result.close("overlap_union_oracle_center", true_union[:2], (-np.sqrt(.5), -np.sqrt(.32)), 1e-12)
    outputs = {}
    for method in ("torch", "tape"):
        outputs[f"winding_{method}"] = _evaluate(lambda p: backend.mesh_query(p, method=method), points)
        outputs[f"normal_{method}"] = _evaluate(lambda p: backend.mesh_query(p, sign="normal", method=method), points)
        result.close(f"overlap_{method}_stated_original_face_semantics", outputs[f"winding_{method}"], nearest_original, 3e-5)
        result.check(f"overlap_{method}_winding_inside_sign", np.array_equal(outputs[f"winding_{method}"] < 0, true_union < 0))
        result.close(f"overlap_{method}_external_union_distance", outputs[f"winding_{method}"][4:], true_union[4:], 3e-5)
    result.check("overlap_interior_nonexactness_exposed", np.max(np.abs(outputs["winding_torch"] - true_union)) > .2)
    result.limitations.append({
        "name": "overlapping_shells_are_not_exact_union_interior_sdf",
        "status": "expected_limitation",
        "center_m": points[0].tolist(), "true_union_sdf_m": float(true_union[0]),
        "combined_mesh_sdf_m": float(outputs["winding_torch"][0]),
        "min_object_sdf_m": float(per_object_min[0]),
        "interior_error_m": _stats(outputs["winding_torch"][:4] - true_union[:4]),
        "normal_sign_disagreements": int(np.count_nonzero((outputs["normal_torch"] < 0) != (true_union < 0))),
        "note": "Winding classifies overlapping interiors, but nearest original triangles include hidden faces."
    })
    result.save("overlap", points=points, nearest_original=nearest_original, true_union=true_union,
                min_object_sdf=per_object_min, **outputs)


def _field_checks(result, scene, backend):
    field = SampledField(backend, (-2., -2., .5), (41, 41, 36), .1)
    result.check("field_xyz_shape", tuple(field.values.shape) == (41, 41, 36))
    rng = np.random.default_rng(922)
    indices = rng.integers((0, 0, 0), (41, 41, 36), size=(128, 3))
    nodes = np.array((-2, -2, .5)) + indices * .1
    expected_nodes = _evaluate(backend.query, nodes)
    node_query = _evaluate(field.query, nodes)
    result.close("field_grid_nodes_match_source", node_query, expected_nodes, 3e-6)
    points = rng.uniform((-1.8, -1.8, .7), (1.8, 1.8, 3.8), size=(256, 3))
    sampled = _evaluate(field.query, points)
    continuous = snapshot_oracle(points, scene.get_geometry(exclude_ids={"vehicle"}))
    # No overlapping objects in this fixture; the composed field is 1-Lipschitz.
    result.close("field_continuous_distance_lipschitz_bound", sampled, continuous, np.sqrt(3) * .1 + 3e-5)
    result.details["field_continuous_error"] = {"spacing_m": .1, "distance_error_m": _stats(sampled - continuous),
                                               "note": "Interpolation error relative to continuous geometry; not oracle query error."}
    smooth_points = np.array(((1.37, .13, 1.83), (1.23, 1.07, 2.17), (1.23, 1.07, 3.31),
                              (.81, .13, 2.17), (-1.39, -.17, 2.21)))
    distance, gradient = _gradient(field.query, smooth_points)
    fd = _finite_difference(field.query, smooth_points)
    oracle_gradient = _cpu_difference(lambda p: snapshot_oracle(p, scene.get_geometry(exclude_ids={"vehicle"})), smooth_points)
    result.close("field_gradient_vs_own_interpolation_fd", gradient, fd, 3e-3)
    result.details["field_gradient_continuous_geometry_bias"] = {
        "error": _stats(gradient - oracle_gradient),
        "note": "Descriptive bias relative to continuous geometry, separate from interpolation derivative correctness."}
    result.save("field", nodes=nodes, node_query=node_query, expected_nodes=expected_nodes,
                points=points, sampled=sampled, continuous=continuous)
    result.save("field_gradient", points=smooth_points, distance=distance, autograd=gradient,
                interpolation_fd=fd, continuous_oracle=oracle_gradient)
    # A thin solid wholly between sampled X nodes can disappear from B entirely.
    thin = make_box("thin", (.02, -.7, 1.1), (.04, .7, 3.1))
    thin_backend = MeshDistance(make_scene((thin,)).get_geometry(exclude_ids={"vehicle"}), device="cuda")
    thin_field = SampledField(thin_backend, (-.2, -1., 1.), (6, 21, 23), .1)
    thin_points = np.array(((.03, .13, 2.17), (.025, .13, 2.17), (.035, .13, 2.17)))
    continuous_thin = _evaluate(thin_backend.query, thin_points)
    sampled_thin = _evaluate(thin_field.query, thin_points)
    result.check("field_thin_obstacle_aliasing_exposed", np.all(continuous_thin < 0) and np.all(sampled_thin > 0))
    result.limitations.append({"name": "sampling_can_miss_subcell_obstacles", "status": "expected_limitation",
                               "spacing_m": .1, "obstacle_thickness_m": .02,
                               "continuous_m": continuous_thin.tolist(), "sampled_m": sampled_thin.tolist()})
    result.save("field_thin_obstacle", points=thin_points, continuous=continuous_thin, sampled=sampled_thin)
    return field


def _invalid_and_boundary(result, backend, field):
    invalid_points = np.array(((np.nan, 0, 2), (0, np.inf, 2), (0, 0, -np.inf),
                               (5.001, 0, 2), (0, 0, -.001), (0, 0, 6.001)))
    for method in ("torch", "tape"):
        d, gradient = _gradient(lambda p: backend.query(p, method=method), invalid_points)
        result.check(f"invalid_{method}_nan_and_zero_gradient", np.isnan(d).all() and np.array_equal(gradient, np.zeros_like(gradient)))
        result.save("invalid", **{f"distance_{method}": d, f"gradient_{method}": gradient})
    boundaries = np.array(((-5, 0, 2), (5, 0, 2), (0, -5, 2), (0, 5, 2), (0, 0, 0), (0, 0, 6)))
    result.check("map_boundary_inclusive", np.isfinite(_evaluate(backend.query, boundaries)).all())
    origin, high = _numpy(field.origin).astype(float), _numpy(field.high).astype(float)
    middle = (origin + high) / 2
    inside_edges = np.stack((origin, high, (origin[0], middle[1], middle[2]),
                            (high[0], middle[1], middle[2])))
    invalid_field = np.stack((origin - (1e-3, 0, 0), high + (0, 1e-3, 0),
                             (np.nan, 0, 2), (0, np.inf, 2)))
    edge_d, edge_g = _gradient(field.query, inside_edges)
    invalid_d, invalid_g = _gradient(field.query, invalid_field)
    result.check("field_boundary_inclusive_finite", np.isfinite(edge_d).all() and np.isfinite(edge_g).all())
    result.check("field_outside_nan_zero_gradient", np.isnan(invalid_d).all() and np.array_equal(invalid_g, np.zeros_like(invalid_g)))
    result.details["domain_policy"] = {
        "map_domain": "A requires declared or explicit bounds; outside and nonfinite inputs return NaN and zero derivative.",
        "field_domain": "B is valid on the inclusive sample-node convex hull; padding is never exposed as valid geometry.",
        "boundary_derivative": "B border interpolation selects zero normal derivative exactly at its support boundary; no exterior derivative claim."}
    result.save("invalid", points=invalid_points, map_boundaries=boundaries)
    result.save("field_boundary", points=inside_edges, distance=edge_d, gradient=edge_g,
                invalid_points=invalid_field, invalid_distance=invalid_d, invalid_gradient=invalid_g)
    # Explicitly extend the query domain to test the finite ground edge. The
    # public map's declared domain would otherwise mask all such queries.
    empty = make_scene(target_position=(3, 3, 4))
    extended = MeshDistance(empty.get_geometry(exclude_ids={"vehicle"}), device="cuda",
                            bounds=((-8, 8), (-8, 8), (-2, 8)))
    p = np.array(((6, 0, .2), (6, 6, .2), (0, 0, -.7), (5, 0, .2), (0, 0, 0)))
    expected = np.minimum(ground_distance(p), sphere_sdf(p, (3, 3, 4), .3))
    actual = _evaluate(extended.query, p)
    result.close("finite_ground_rectangle_not_infinite_plane", actual, expected, 3e-6)
    result.save("ground", points=p, expected=expected, actual=actual)


def _training_chain(result, backend, field):
    # A shared learnable tensor supplies query points through a full 3D rotation.
    # Optimizing exp(-d) should move the resulting points away from obstacles.
    initial = np.array(((1.37, .13, 1.83), (1.23, 1.07, 2.17), (-1.39, -.17, 2.21),
                        (.81, .13, 2.17)), dtype=np.float32)
    rotation = np.array(((.8, -.48, .36), (.6, .64, -.48), (0., .6, .8)), dtype=np.float32)
    rot = _tensor(rotation)
    shift = _tensor((.11, -.17, .23))
    for name, query in (("torch", lambda p: backend.query(p, method="torch")),
                         ("tape", lambda p: backend.query(p, method="tape")), ("field", field.query)):
        latent = torch.nn.Parameter((_tensor(initial) - shift) @ rot)
        optimizer = torch.optim.SGD([latent], lr=.08)
        losses, grad_norms, positions = [], [], []
        for step in range(5):
            optimizer.zero_grad()
            points = latent @ rot.T + shift
            d = query(points)
            loss = torch.exp(-d).mean()
            loss.backward()
            if step == 0:
                # Verify the actual learnable parameters through the complete
                # rotation -> query -> nonlinear cost chain, beyond point-only
                # gradient checks or observing that SGD happens to descend.
                initial_parameters = _numpy(latent).astype(np.float64)
                parameter_gradient = _numpy(latent.grad)
                parameter_fd = np.empty_like(initial_parameters)
                epsilon = 1e-3
                with torch.no_grad():
                    for row in range(len(initial_parameters)):
                        for axis in range(3):
                            delta = np.zeros_like(initial_parameters)
                            delta[row, axis] = epsilon
                            plus_points = _tensor(initial_parameters + delta) @ rot.T + shift
                            minus_points = _tensor(initial_parameters - delta) @ rot.T + shift
                            plus = float(torch.exp(-query(plus_points)).mean().cpu())
                            minus = float(torch.exp(-query(minus_points)).mean().cpu())
                            parameter_fd[row, axis] = (plus - minus) / (2 * epsilon)
                result.close(f"training_parameter_{name}_gradient_fd", parameter_gradient, parameter_fd, 2e-4)
                result.save(f"training_{name}", initial_parameters=initial_parameters,
                            parameter_gradient=parameter_gradient, parameter_fd=parameter_fd)
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(latent.grad.norm().cpu()))
            positions.append(_numpy(points))
            optimizer.step()
        with torch.no_grad():
            final_loss = float(torch.exp(-query(latent @ rot.T + shift)).mean().cpu())
        losses.append(final_loss)
        result.check(f"training_parameter_query_exp_loss_sgd_{name}",
                     np.isfinite(losses).all() and np.isfinite(grad_norms).all() and min(grad_norms) > 0
                     and losses[-1] < losses[0] and np.all(np.diff(losses) < 0),
                     losses=losses, gradient_norms=grad_norms,
                     note="First-order position gradients only; no geometry or second-order differentiation claim.")
        result.save(f"training_{name}", positions=positions, losses=losses, gradient_norms=grad_norms,
                    final_parameters=_numpy(latent))


def _dynamic_updates(result, output_dir):
    obstacle = make_box("static", (-1, -.8, 1), (1, .8, 3))
    scene = make_scene((obstacle,), target_position=(2.5, 1.2, 2.0), target_radius=.45)
    snapshot = scene.get_geometry(exclude_ids={"vehicle"})
    backend = MeshDistance(snapshot, device="cuda")
    mesh_handle = backend.mesh.id
    field = SampledField(backend, (-3., -2., .5), (61, 41, 36), .1)
    centers = ((2.5, 1.2, 2.0), (.85, 0., 2.), (-2.2, 1.2, 2.4),
               (-.85, 0., 2.), (2.5, 1.2, 2.0))
    rng = np.random.default_rng(993)
    probes = np.concatenate((np.array(centers), np.array(((.7, .13, 2.1), (-.7, .13, 2.1), (1.25, 0, 2))),
                             rng.uniform((-2.8, -1.8, .7), (2.8, 1.8, 3.8), size=(64, 3))))
    static_probes = np.array(((.7, .13, 2.1), (-.7, .13, 2.1)))
    expected_static = box_sdf(static_probes, (-1, -.8, 1), (1, .8, 3))
    for index, center in enumerate(centers):
        moved_scene = scene.with_target_position(center)
        moved = moved_scene.get_geometry(exclude_ids={"vehicle"})
        backend.update(moved)
        field.rebuild(backend)
        fresh = MeshDistance(moved, device="cuda")
        fresh_field = SampledField(fresh, (-3., -2., .5), (61, 41, 36), .1)
        oracle = snapshot_oracle(probes, moved)
        for method in ("torch", "tape"):
            actual = _evaluate(lambda p: backend.query(p, method=method), probes)
            reference = _evaluate(lambda p: fresh.query(p, method=method), probes)
            result.close(f"update_{index}_{method}_fresh_equivalence", actual, reference, 3e-6)
            result.close(f"update_{index}_{method}_oracle", actual, oracle, 3e-5)
            result.save(f"update_{index}", **{f"actual_{method}": actual, f"fresh_{method}": reference})
        values, new_values = _numpy(field.values), _numpy(fresh_field.values)
        result.close(f"update_{index}_field_fresh_equivalence", values, new_values, 3e-6)
        sampled, fresh_sampled = _evaluate(field.query, probes), _evaluate(fresh_field.query, probes)
        result.close(f"update_{index}_field_queries_fresh", sampled, fresh_sampled, 3e-6)
        result.check(f"update_{index}_static_bvh_reused", backend.mesh.id == mesh_handle)
        static_only = _evaluate(lambda p: backend.mesh_query(p), static_probes)
        result.close(f"update_{index}_static_surface_preserved", static_only, expected_static, 3e-6)
        result.save(f"update_{index}", center=center, points=probes, oracle=oracle,
                    field=sampled, fresh_field=fresh_sampled, field_values=values)
    result.details["dynamic_update"] = {
        "positions": [list(center) for center in centers], "static_bvh_reused": True,
        "field_update": "Full resampling from the replacement snapshot; no incremental voxel-clearing claim.",
        "overlap_semantics": "Sphere/mesh min-composition, also not an exact Boolean-union interior SDF.",
        "check": "Every move, including overlap and departure, compared with a fresh backend and full fresh field."}


def run_verification(output_dir: Path) -> dict:
    """Run checks, save all numerical evidence, and return a JSON-safe report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _Results()
    stages = [
        ("oracle_self_checks", lambda: _oracle_self_checks(result)),
        ("sphere_geometry", lambda: _sphere_geometry(result)),
        ("rotation_and_narrow_gap", lambda: _rotation_and_gap(result)),
        ("sliver_triangle", lambda: _sliver_triangle(result)),
        ("overlap_limit", lambda: _overlap_limit(result)),
        ("dynamic_updates", lambda: _dynamic_updates(result, output_dir)),
    ]
    # Keep independent stages running after a failed assertion; retain the first
    # error and raw evidence rather than losing all diagnostic output.
    try:
        scene, backend = _box_geometry(result, output_dir)
        field = _field_checks(result, scene, backend)
        _invalid_and_boundary(result, backend, field)
        _training_chain(result, backend, field)
    except Exception:
        result.check("box_field_chain_execution", False, traceback=traceback.format_exc())
    for name, call in stages:
        try:
            call()
        except Exception:
            result.check(f"{name}_execution", False, traceback=traceback.format_exc())
    np.savez_compressed(output_dir / "verification_arrays.npz", **result.arrays)
    report = {
        "passed": bool(result.checks) and all(entry["passed"] for entry in result.checks.values()),
        "checks_passed": sum(entry["passed"] for entry in result.checks.values()),
        "checks_total": len(result.checks),
        "scope": "First-order position derivatives; world-metre public scene geometry; float32 CUDA queries against float64 independent CPU references.",
        "checks": result.checks, "details": result.details, "expected_limitations": result.limitations,
        "raw_arrays": "verification_arrays.npz",
    }
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_verification(args.output)
    print(json.dumps({"passed": report["passed"], "checks_passed": report["checks_passed"],
                      "checks_total": report["checks_total"], "output": str(args.output)}, indent=2))
    if not report["passed"]:
        for name, check in report["checks"].items():
            if not check["passed"]:
                print(name, json.dumps(check, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
