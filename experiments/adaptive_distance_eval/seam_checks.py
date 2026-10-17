"""Measure both leaf polynomials at shared faces, including world gradients.

The lookup table defines the canonical dyadic topology. Float32-exported leaf
origins can differ slightly at a shared face; both polynomials are nevertheless
evaluated at the same canonical world point. This tests the actual Torch field.
Finite face probes are evidence, not a certificate over every point of a face.
"""

import json
from pathlib import Path

import numpy as np
import torch


def _faces(lookup: np.ndarray) -> np.ndarray:
    """All unique oriented (left leaf, right leaf, normal axis) adjacencies."""
    groups = []
    for axis in range(3):
        pairs = []
        for plane in range(lookup.shape[axis] - 1):
            left = np.take(lookup, plane, axis=axis)
            right = np.take(lookup, plane + 1, axis=axis)
            different = left != right
            if different.any():
                pairs.append(np.unique(np.stack((left[different], right[different]), axis=1), axis=0))
        if pairs:
            unique = np.unique(np.concatenate(pairs), axis=0)
            groups.append(np.column_stack((unique, np.full(len(unique), axis))))
    return np.concatenate(groups).astype(np.int64) if groups else np.empty((0, 3), dtype=np.int64)


def _counts(faces, coarse_fine):
    return {"total": len(faces), "same_level": int((~coarse_fine).sum()),
            "coarse_fine": int(coarse_fine.sum()),
            "by_axis": {axis: int((faces[:, 2] == i).sum()) for i, axis in enumerate("xyz")}}


def _select_faces(coarse_fine, max_faces):
    count = len(coarse_fine)
    if max_faces is None or count <= max_faces:
        return np.arange(count, dtype=np.int64)
    if max_faces < 2:
        raise ValueError("max_faces must be at least two or None")
    same, mixed = np.flatnonzero(~coarse_fine), np.flatnonzero(coarse_fine)
    if not len(same) or not len(mixed):
        return np.linspace(0, count - 1, max_faces, dtype=np.int64)
    same_budget = max(1, min(max_faces - 1, round(max_faces * len(same) / count)))
    return np.sort(np.concatenate((same[np.linspace(0, len(same) - 1, same_budget, dtype=np.int64)],
                                   mixed[np.linspace(0, len(mixed) - 1, max_faces - same_budget,
                                                     dtype=np.int64)])))


def _evaluate(field, points, ids, batch_points):
    values = np.empty(len(points), dtype=np.float64)
    gradients = np.empty((len(points), 3), dtype=np.float64)
    for start in range(0, len(points), batch_points):
        stop = min(start + batch_points, len(points))
        query = torch.tensor(points[start:stop], dtype=torch.float64, requires_grad=True)
        if ids is None:
            value = field.query(query)
        else:
            selected = torch.tensor(ids[start:stop], dtype=torch.int64)
            value = field.evaluate_leaf(query, selected)
        if value.shape != (stop - start,) or value.dtype != torch.float64:
            raise TypeError("field evaluation must preserve float64 Tensor[N]")
        derivative, = torch.autograd.grad(value.sum(), query)
        values[start:stop] = value.detach().numpy()
        gradients[start:stop] = derivative.detach().numpy()
    return values, gradients


def _distribution(values):
    finite = np.isfinite(values)
    return {"maximum": float(values.max()) if values.size and finite.all() else None,
            "p95": float(np.percentile(values, 95)) if values.size and finite.all() else None,
            "nonfinite_count": int((~finite).sum())}


def run_seam_checks(field, output_dir, *, samples_per_axis=5, max_faces=50000,
                    batch_points=4096) -> dict:
    """Save JSON and raw NPZ; reject incomplete coverage as a full acceptance.

    CPU-only, float64 arithmetic over the field's exported coefficients. At each
    shared overlap face, a fixed grid includes the edges/corners. Gradients are
    obtained from autograd with respect to the *world* points, so the local-cell
    scale is included. Value threshold is 5e-6 m and gradient L2 threshold 2e-3.

    A cap deterministically selects faces proportionally from same-level and
    coarse/fine categories. `sampled_passed` reports those probes; `passed` also
    requires every enumerated face to have been checked. Neither is a rigorous
    all-points continuity certificate. Use max_faces=None to remove the cap.
    """
    if torch.device(field.device).type != "cpu":
        raise ValueError("run_seam_checks requires a CPU field; no implicit device fallback")
    if samples_per_axis < 3 or samples_per_axis % 2 != 1 or batch_points < 1:
        raise ValueError("use an odd face grid >= 3 and a positive batch_points")
    lookup = np.asarray(field.lookup_numpy)
    if lookup.ndim != 3 or len(set(lookup.shape)) != 1 or (lookup < 0).any():
        raise ValueError("lookup must be a fully covered cubic 3D leaf-ID table")
    rows = np.asarray(field.rows)
    all_faces = _faces(lookup)
    if len(all_faces) and int(all_faces[:, :2].max()) >= len(rows):
        raise ValueError("lookup contains a leaf ID outside field.rows")
    depths = rows[:, 4].astype(np.int64)
    all_mixed = depths[all_faces[:, 0]] != depths[all_faces[:, 1]]
    selected = _select_faces(all_mixed, max_faces)
    faces, mixed = all_faces[selected], all_mixed[selected]
    lo, hi = (bound.detach().cpu().numpy().astype(np.float64) for bound in field.bounds)
    cell_size = (hi - lo) / lookup.shape[0]
    lattice_lo = np.rint((rows[:, :3].astype(np.float64) - lo) / cell_size).astype(np.int64)
    widths = np.rint(rows[:, 3].astype(np.float64) / cell_size[0]).astype(np.int64)
    left, right, axes = faces.T
    lower_grid = np.maximum(lattice_lo[left], lattice_lo[right])
    upper_grid = np.minimum(lattice_lo[left] + widths[left, None],
                            lattice_lo[right] + widths[right, None])
    indices = np.arange(len(faces))
    face_planes = lattice_lo[left, axes] + widths[left]
    topology_valid = (face_planes == lattice_lo[right, axes])
    tangent_widths = upper_grid - lower_grid
    tangent_widths[indices, axes] = 1
    topology_valid &= (tangent_widths > 0).all(axis=1)
    if not topology_valid.all():
        raise ValueError("lookup adjacency disagrees with the exported leaf boxes")
    lower_grid[indices, axes] = face_planes
    upper_grid[indices, axes] = face_planes
    lower, upper = lo + lower_grid * cell_size, lo + upper_grid * cell_size
    tangents = np.array([[axis for axis in range(3) if axis != normal]
                         for normal in axes], dtype=np.int64).reshape(-1, 2)
    grid = np.linspace(0., 1., samples_per_axis)
    uv = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    points = np.broadcast_to(lower[:, None], (len(faces), len(uv), 3)).copy()
    for tangent in range(2):
        coordinate = tangents[:, tangent]
        points[indices[:, None], np.arange(len(uv))[None, :], coordinate[:, None]] += (
            (upper - lower)[indices, coordinate, None] * uv[None, :, tangent])
    flat_points = points.reshape(-1, 3)
    left_value, left_gradient = _evaluate(field, flat_points, np.repeat(left, len(uv)), batch_points)
    right_value, right_gradient = _evaluate(field, flat_points, np.repeat(right, len(uv)), batch_points)
    values = np.stack((left_value, right_value), axis=-1).reshape(len(faces), len(uv), 2)
    gradients = np.stack((left_gradient, right_gradient), axis=1).reshape(len(faces), len(uv), 2, 3)
    value_jump = np.abs(values[:, :, 1] - values[:, :, 0])
    gradient_delta = gradients[:, :, 1] - gradients[:, :, 0]
    gradient_jump = np.linalg.norm(gradient_delta, axis=-1)

    # Only the overlap-face centre is used for side lookup: face-grid edges and
    # corners can legitimately belong to a third neighbouring leaf.
    centre_index = len(uv) // 2
    centres = points[:, centre_index].copy()
    epsilon = np.minimum(rows[left, 3], rows[right, 3]).astype(np.float64) * 1e-5
    offsets = np.zeros_like(centres)
    offsets[indices, axes] = epsilon
    lookup_ids = np.empty((len(faces), 3), dtype=np.int64)
    with torch.no_grad():
        for column, locations in enumerate((centres - offsets, centres, centres + offsets)):
            for start in range(0, len(faces), batch_points):
                stop = min(start + batch_points, len(faces))
                lookup_ids[start:stop, column] = field.leaf_ids(
                    torch.tensor(locations[start:stop], dtype=torch.float64)).numpy()
    side_lookup_ok = (lookup_ids[:, 0] == left) & (lookup_ids[:, 2] == right)
    exact_lookup_ok = (lookup_ids[:, 1] == left) | (lookup_ids[:, 1] == right)
    centre_values, centre_gradients = _evaluate(field, centres, None, batch_points)
    centre_side = (lookup_ids[:, 1] == right).astype(np.int64)
    centre_value_error = np.abs(centre_values - values[indices, centre_index, centre_side])
    centre_gradient_error = np.linalg.norm(
        centre_gradients - gradients[indices, centre_index, centre_side], axis=-1)

    value_limit, gradient_limit = 5e-6, 2e-3
    def worst(jumps, mask):
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            return None
        scores = np.where(np.isfinite(jumps[candidates]), jumps[candidates], np.inf)
        local_face, probe = np.unravel_index(int(np.argmax(scores)), scores.shape)
        face = int(candidates[local_face])
        return {"selected_face_index": face, "enumerated_face_index": int(selected[face]),
                "left_leaf": int(left[face]), "right_leaf": int(right[face]),
                "axis": "xyz"[axes[face]], "depths": depths[[left[face], right[face]]].tolist(),
                "probe_index": int(probe), "point": points[face, probe].tolist(),
                "value_jump_m": float(value_jump[face, probe]) if np.isfinite(value_jump[face, probe]) else None,
                "gradient_jump_l2": float(gradient_jump[face, probe]) if np.isfinite(gradient_jump[face, probe]) else None}

    def summary(mask):
        v, g = value_jump[mask], gradient_jump[mask]
        return {"face_count": int(mask.sum()), "probe_count": int(v.size),
                "value_jump_m": _distribution(v), "gradient_jump_l2": _distribution(g),
                "gradient_jump_linf": _distribution(np.abs(gradient_delta[mask]).max(axis=-1)),
                "value_fail_faces": int((~np.isfinite(v) | (v > value_limit)).any(axis=1).sum()),
                "gradient_fail_faces": int((~np.isfinite(g) | (g > gradient_limit)).any(axis=1).sum()),
                "worst_value_face": worst(value_jump, mask),
                "worst_gradient_face": worst(gradient_jump, mask)}

    coverage_complete = len(selected) == len(all_faces)
    checks = {"has_shared_faces": len(faces) > 0,
              "finite_values_and_world_gradients": bool(np.isfinite(values).all() & np.isfinite(gradients).all()),
              "value_jump_within_5e_6_m": bool((value_jump <= value_limit).all()),
              "gradient_l2_jump_within_2e_3": bool((gradient_jump <= gradient_limit).all()),
              "offset_lookup_matches_both_sides": bool(side_lookup_ok.all()),
              "exact_face_lookup_selects_an_adjacent_leaf": bool(exact_lookup_ok.all()),
              "query_matches_selected_polynomial_value": bool((centre_value_error <= 1e-12).all()),
              "query_matches_selected_polynomial_world_gradient": bool((centre_gradient_error <= 1e-12).all())}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path = output_dir / "seam-checks-raw.npz", output_dir / "seam-checks.json"
    np.savez_compressed(raw_path, all_faces=all_faces, selected_face_indices=selected,
                        face_columns=np.array(["left_leaf", "right_leaf", "axis"]),
                        coarse_fine=mixed, face_lower=lower, face_upper=upper,
                        points=points, values=values, world_gradients=gradients,
                        value_jump=value_jump, gradient_jump_l2=gradient_jump,
                        centre_lookup_ids=lookup_ids, centre_offset_epsilon_m=epsilon,
                        centre_query_values=centre_values, centre_query_gradients=centre_gradients,
                        centre_value_error=centre_value_error, centre_gradient_error=centre_gradient_error)
    result = {"passed": all(checks.values()) and coverage_complete,
              "sampled_passed": all(checks.values()), "checks": checks,
              "coverage_complete": coverage_complete, "maximum_faces": max_faces,
              "selection": "all faces" if coverage_complete else "deterministic proportional same-level/coarse-fine subsample",
              "enumerated_faces": _counts(all_faces, all_mixed), "checked_faces": _counts(faces, mixed),
              "samples_per_face": len(uv), "samples_per_axis": samples_per_axis,
              "dtype": "torch.float64", "device": "cpu",
              "thresholds": {"value_m": value_limit, "gradient_l2": gradient_limit},
              "scope": "finite exact-face probes; not an all-points continuity or rounding certificate",
              "all": summary(np.ones(len(faces), dtype=bool)),
              "same_level": summary(~mixed), "coarse_fine": summary(mixed),
              "lookup": {"offset_failure_faces": int((~side_lookup_ok).sum()),
                         "exact_failure_faces": int((~exact_lookup_ok).sum()),
                         "selected_query_value_error": _distribution(centre_value_error),
                         "selected_query_gradient_error": _distribution(centre_gradient_error)},
              "artifacts": {"raw": str(raw_path), "summary": str(summary_path)}}
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
