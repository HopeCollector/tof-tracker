"""Actual-map accuracy and resident query timings; not a training loss API.

The rectangle oracle measures the union's exposed surfaces independently of the
triangle BVH and interpolation. Its geometry preparation shares the documented
axis-aligned-box model; analytic geometry regressions live in tests/.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from devtools.esdf import ESDFGrid
from docs.explorations.esdf.adaptive_distance_eval.code.real_map_geometry import load_real_map


def stats(estimated, truth):
    error = np.asarray(estimated, dtype=np.float64) - truth
    return {"count": len(error), "mean_abs_m": float(np.abs(error).mean()),
            "p95_abs_m": float(np.quantile(np.abs(error), .95)),
            "max_abs_m": float(np.abs(error).max()),
            "max_overestimate_m": float(max(0., error.max())),
            "max_underestimate_m": float(max(0., -error.min())),
            "sign_disagreement_outside_1mm": int(((estimated*truth < 0) & (np.abs(truth) > .001)).sum()),
            "sign_disagreement_outside_1cm": int(((estimated*truth < 0) & (np.abs(truth) > .01)).sum())}


def query_batch(field, points, *, gradient=False, dtype=torch.float64):
    values, derivatives = [], []
    for start in range(0, len(points), 4096):
        p = torch.tensor(points[start:start+4096], device=field.device, dtype=dtype, requires_grad=gradient)
        d, valid = field.query(p, return_valid=True)
        assert bool(valid.all()) and bool(torch.isfinite(d).all())
        if gradient:
            g, = torch.autograd.grad(d.sum(), p)
            assert bool(torch.isfinite(g).all())
            derivatives.append(g.detach().cpu().numpy())
        values.append(d.detach().cpu().numpy())
    value = np.concatenate(values)
    return (value, np.concatenate(derivatives)) if gradient else value


def make_points(geometry, field):
    rng = np.random.default_rng(80123)
    bounds = np.asarray(field.bounds_m)
    groups = {"uniform": rng.uniform(bounds[0], bounds[1], (8192, 3))}
    lo, hi = geometry.box_lower, geometry.box_upper
    # All pillars, all faces, offsets both sides. Occluded faces are retained;
    # the reference is the exposed union, rather than a nearest raw object.
    faces = []
    for axis in range(3):
        for side in (0, 1):
            p = rng.uniform(lo, hi)
            p[:, axis] = (lo if side == 0 else hi)[:, axis]
            for offset in (-.05, -.005, .005, .05):
                q = p.copy()
                q[:, axis] += offset
                faces.append(q)
    groups["near_original_box_faces"] = np.concatenate(faces)
    groups["pillar_interiors"] = np.concatenate((rng.uniform(lo, hi), (lo+hi)/2))
    intersections = []
    for a, b in geometry.overlap_pairs:
        lower, upper = np.maximum(lo[a], lo[b]), np.minimum(hi[a], hi[b])
        intersections.extend(((lower+upper)/2, .75*lower+.25*upper, .25*lower+.75*upper))
    groups["overlap_interiors"] = np.asarray(intersections)
    ground = rng.uniform(bounds[0], bounds[1], (1024, 3))
    ground[:, 2] = rng.choice([0., .001, .01, .05], len(ground))
    groups["ground_near_surface"] = ground
    # Actual stored node positions, chosen inside the logical domain.
    indices = np.column_stack([rng.integers(1, n-1, 2048) for n in field.shape])
    groups["shared_nodes"] = np.asarray(field.sample_origin_m) + indices*field.spacing_m
    return {name: p[((p >= bounds[0]) & (p <= bounds[1])).all(axis=1)] for name, p in groups.items()}


def seam_check(field):
    rng = np.random.default_rng(2403)
    lo, hi = np.asarray(field.bounds_m)
    origin, h = np.asarray(field.sample_origin_m), field.spacing_m
    points, axes = [], []
    for axis in range(3):
        p = rng.uniform(lo+.2*h, hi-.2*h, (128, 3))
        face = rng.integers(2, field.shape[axis]-2, len(p))
        p[:, axis] = origin[axis]+face*h
        points.extend(p)
        axes.extend([axis]*len(p))
    p, axes = np.asarray(points), np.asarray(axes)
    delta = np.zeros_like(p)
    delta[np.arange(len(p)), axes] = h*1e-6
    sides = []
    for sign in (-1, 1):
        v1, g1 = query_batch(field, p+sign*delta, gradient=True)
        v2, g2 = query_batch(field, p+2*sign*delta, gradient=True)
        sides.append((2*v1-v2, 2*g1-g2))
    value_jump = float(np.abs(sides[0][0]-sides[1][0]).max())
    gradient_jump = float(np.abs(sides[0][1]-sides[1][1]).max())
    assert value_jump < 1e-8 and gradient_jump < 1e-6
    return {"face_points": len(p), "float64_linear_extrapolated_value_jump_m": value_jump,
            "float64_linear_extrapolated_gradient_component_jump": gradient_jump,
            "scope": "sampled actual-map internal faces; independent same-point Hermite limits and exhaustive fixture faces are checked by tests/verify_esdf.py"}


def timings(field):
    rng = np.random.default_rng(159)
    bounds = np.asarray(field.bounds_m)
    rows = []
    is_cuda = field.device.type == "cuda"
    for count in (1, 100_000, 262_144):
        p = torch.tensor(rng.uniform(bounds[0], bounds[1], (count, 3)), device=field.device,
                         dtype=torch.float32, requires_grad=True)
        for backward in (False, True):
            measured = []
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
            for repeat in range(12):
                if is_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if backward:
                    g, = torch.autograd.grad(field.query(p).sum(), p)
                else:
                    with torch.no_grad():
                        d = field.query(p)
                if is_cuda:
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                if repeat >= 2:
                    measured.append(elapsed)
            row = {"points": count, "operation": "query_and_position_backward" if backward else "query_no_grad",
                   "median_wall_ms": float(np.median(measured)), "p95_wall_ms": float(np.quantile(measured, .95)),
                   "repeats": len(measured)}
            if is_cuda:
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                row["extra_peak_over_resident_bytes"] = row["peak_allocated_bytes"]-baseline
            rows.append(row)
    return {"rows": rows, "scope": "float32 resident map and positions, synchronized Python wall clock; excludes disk load, transfer, mesh and trajectory decoding"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    start = time.perf_counter()
    field = ESDFGrid.load(args.grid, device=args.device)
    if field.device.type == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter()-start
    geometry = load_real_map(args.map)
    groups = make_points(geometry, field)
    raw, errors, errors32 = {}, {}, {}
    for name, p in groups.items():
        truth = geometry.distance_torch(torch.tensor(p, dtype=torch.float64, device=field.device)).cpu().numpy()
        d, g = query_batch(field, p, gradient=True)
        errors[name] = stats(d, truth)
        raw[name+"_points"] = p
        raw[name+"_reference"] = truth
        raw[name+"_distance"] = d
        raw[name+"_gradient"] = g
        # The double oracle receives the exact represented FP32 positions.
        p32 = p.astype(np.float32)
        truth32 = geometry.distance_torch(torch.tensor(p32.astype(np.float64), dtype=torch.float64,
                                                     device=field.device)).cpu().numpy()
        d32, g32 = query_batch(field, p32, gradient=True, dtype=torch.float32)
        errors32[name] = stats(d32, truth32)
        raw[name+"_points_float32"] = p32
        raw[name+"_reference_at_float32_positions"] = truth32
        raw[name+"_distance_float32"] = d32
        raw[name+"_gradient_float32"] = g32
    all_d = np.concatenate([raw[name+"_distance"] for name in groups])
    all_truth = np.concatenate([raw[name+"_reference"] for name in groups])
    # Gradient agreement is with this interpolant, away from cell boundaries.
    p = groups["uniform"][:128]
    d, g = query_batch(field, p, gradient=True)
    fd = np.empty_like(g)
    for axis in range(3):
        delta = np.zeros_like(p)
        delta[:, axis] = 1e-5
        fd[:, axis] = (query_batch(field, p+delta)-query_batch(field, p-delta))/(2e-5)
    fd_error = float(np.abs(fd-g).max())
    assert fd_error < 2e-7
    seams = seam_check(field)
    report = {"completed": True, "grid": str(args.grid), "source_map_sha256": hashlib.sha256(args.map.read_bytes()).hexdigest(),
              "samples_sha256": hashlib.sha256((args.grid/"samples.npy").read_bytes()).hexdigest(),
              "device": str(field.device), "torch_version": torch.__version__, "numpy_version": np.__version__,
              "python_version": platform.python_version(), "gpu": torch.cuda.get_device_name() if field.device.type == "cuda" else None,
              "load_and_transfer_seconds": load_seconds, "node_bytes": field.storage_bytes,
              "bounds_m": field.bounds_m, "shape_xyz_with_halo": field.shape,
              "accuracy_query_dtype": "float64 (float32 stored nodes)",
              "error_groups": errors, "all_points": stats(all_d, all_truth),
              "float32_error_groups_at_represented_positions": errors32, "seams": seams,
              "float64_position_gradient_fd_max_abs": fd_error, "timing": timings(field),
              "limits": "Distance errors are sampled, not a global bound. Cubic overshoot/sign changes are measured, not excluded. C1 describes the interpolated field; no unit-gradient or exact-mesh-clearance claim. This does not measure trajectory optimization."}
    np.savez_compressed(args.output/"raw.npz", **raw)
    (args.output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
