"""Deterministic 8 x 15 world-frame quintic boundary fixtures for seed 23.

This is a deliberately selected geometric stress set, not a random population
sample or a planner. Selection never queries a field, a loss, or its gradient.
The categories describe the intended local geometry; other pillars may still
intersect a candidate, and no candidate is labelled collision-free in advance.
"""

from __future__ import annotations

import numpy as np


DURATION_S = 10.0 / 6.0
SOURCE_SHA256 = "da25463989072ef10ceaa5c6cbddd55697c9c16a8d9c2e3c5df3cb5f523ba6e5"


def make_fixtures(geometry):
    """Return ``initial, terminal, metadata`` using public geometry arrays.

    Both arrays are float64 [B=8,K=15,pva=3,xyz=3]. Each group shares its
    complete initial state. Duration is 10/6 seconds; callers must use that
    duration. The fixed pillar indices refer to the existing seed-23 OBJ.
    Boundaries stay inside the logical domain by checking all six quintic
    Bernstein control points, rather than a finite set of trajectory samples.
    This domain check says nothing about collision clearance.
    """
    boxes = np.asarray(geometry.boxes, dtype=np.float64)
    bounds = np.asarray(geometry.logical_bounds, dtype=np.float64)
    source_sha = geometry.metadata.get("source_sha256")
    if source_sha != SOURCE_SHA256:
        raise ValueError("these fixed pillar fixtures require the recorded seed-23 source OBJ")
    if (boxes.shape != (500, 2, 3) or bounds.shape != (2, 3)
            or not np.isfinite(boxes).all() or not np.isfinite(bounds).all()
            or np.any(boxes[:, 1] <= boxes[:, 0]) or np.any(bounds[1] <= bounds[0])):
        raise ValueError("expected finite 500-pillar geometry and increasing logical bounds")

    # Indices were chosen from box geometry before evaluating any distance
    # field. 247 and 329 have large nearest-footprint gaps; 26/43 overlap another
    # pillar. Different locations, heights and side orientations are retained.
    specifications = (
        (247, "sparse_positive_y_side", "side", 1),
        (329, "sparse_negative_y_side", "side", -1),
        (26, "overlapping_positive_y_side", "side", 1),
        (43, "overlapping_negative_y_side", "side", -1),
        (262, "low_ground_positive_y_side", "ground", 1),
        (12, "low_pillar_top", "top", 1),
        (13, "higher_pillar_top", "top", -1),
        (2, "ordinary_negative_y_side", "side", -1),
    )
    initial = np.zeros((8, 15, 3, 3), dtype=np.float64)
    terminal = np.zeros_like(initial)
    groups = []
    for group, (index, name, anchor_type, side) in enumerate(specifications):
        lo, hi = boxes[index]
        centre = (lo + hi) * .5
        xy_margin = np.minimum(lo[:2] - bounds[0, :2], bounds[1, :2] - hi[:2])
        if np.any(xy_margin <= 5):
            raise ValueError(f"pillar {index} is not more than 5 m from the XY domain boundary")
        if anchor_type == "top":
            y0, z0 = centre[1], hi[2] + .02
            y_offsets, z_offsets = np.linspace(-.9, .9, 5), np.array([-.7, 0., .7])
        elif anchor_type == "ground":
            y0, z0 = hi[1], .18
            y_offsets, z_offsets = np.linspace(-1.2, 1.2, 5), np.array([-.12, 0., .24])
        else:
            y0 = hi[1] if side > 0 else lo[1]
            z0 = min(3., .45 * hi[2])
            y_offsets, z_offsets = np.linspace(-1.2, 1.2, 5), np.array([-.24, 0., .24])

        initial[group, :, 0] = (lo[0] - 2., y0, z0)
        initial[group, :, 1] = (2., .02 * side, 0.)
        initial[group, :, 2] = (.04, .02 * side, .015)
        candidates = []
        for iy, dy in enumerate(y_offsets):
            for iz, dz in enumerate(z_offsets):
                k = 3 * iy + iz
                terminal[group, k, 0] = (hi[0] + 2., y0 + dy, z0 + dz)
                terminal[group, k, 1] = (2. + .05 * (iy - 2), .04 * dy, .015 * (iz - 1))
                terminal[group, k, 2] = (-.03, -.02 * side, -.012)
                candidates.append({"candidate": k, "y_offset_m": float(dy), "z_offset_m": float(dz),
                                   "terminal_velocity_mps": terminal[group, k, 1].tolist(),
                                   "terminal_acceleration_mps2": terminal[group, k, 2].tolist()})

        separation = np.maximum(np.maximum(boxes[:, 0, :2] - hi[:2], lo[:2] - boxes[:, 1, :2]), 0.)
        gap = np.linalg.norm(separation, axis=1)
        gap[index] = np.inf
        neighbour = int(gap.argmin())
        overlap = np.all((boxes[:, 0, :2] < hi[:2]) & (boxes[:, 1, :2] > lo[:2]), axis=1)
        overlap[index] = False
        groups.append({"group": group, "name": name, "pillar_index": index,
                       "pillar_bounds_m": boxes[index].tolist(), "anchor_type": anchor_type,
                       "anchor_side_y": side, "anchor_yz_m": [float(y0), float(z0)],
                       "xy_boundary_margin_m": xy_margin.tolist(),
                       "nearest_pillar_index_by_xy_gap": neighbour,
                       "nearest_pillar_xy_gap_m": float(gap[neighbour]),
                       "xy_overlap_pillar_indices": np.flatnonzero(overlap).tolist(),
                       "candidates": candidates})

    # Exact boundary-to-Bernstein identities for a quintic over u=t/T.
    t = DURATION_S
    p0, v0, a0 = (initial[..., i, :] for i in range(3))
    p1, v1, a1 = (terminal[..., i, :] for i in range(3))
    control = np.stack((p0, p0 + t*v0/5., p0 + 2*t*v0/5. + t*t*a0/20.,
                        p1 - 2*t*v1/5. + t*t*a1/20., p1 - t*v1/5., p1), axis=-2)
    if (not np.isfinite(control).all() or np.any(control < bounds[0])
            or np.any(control > bounds[1])):
        raise ValueError("a complete quintic Bernstein hull leaves the logical domain")
    metadata = {
        "selection": "fixed geometry-selected stress fixtures; not random population statistics",
        "source_map_sha256": source_sha, "duration_s": t,
        "shape": list(initial.shape), "state_layout": "[group,candidate,p/v/a,world_xyz]",
        "shared_initial_per_group": True, "logical_bounds_m": bounds.tolist(),
        "candidate_rule": "5 terminal y offsets x 3 terminal z offsets; small p/v/a variations",
        "classification": "side, top and ground are intended anchors; free/near/penetrating labels are not assumed",
        "selection_uses_field_or_loss": False,
        "hull_bounds_m": np.stack((control.min(axis=(0, 1, 2)), control.max(axis=(0, 1, 2)))).tolist(),
        "hull_domain_check": "all six boundary-derived Bernstein control points lie inside logical bounds",
        "groups": groups,
    }
    return initial, terminal, metadata
