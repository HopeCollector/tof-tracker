"""Explicit box-union adaptation of the existing 500-pillar map.

Only axis-aligned closed boxes and the declared rectangular z=0 ground are
accepted.  Geometry enters through Scene.get_geometry(), never scene storage.
The source OBJ is read only.  Overlaps are retained as a geometric union: faces
inside another box and duplicate coplanar faces are removed before triangulation.

The source ground is an open, finite sheet.  For SdfLib's closed-solid sign
convention this adapter extends it to a slab covering the whole query cube.
Its sides and bottom are sufficiently far away to never be nearest within that
cube.  In the original logical domain (including its XY boundary, z >= 0), the
result is the occupied union of the pillars and ground below z=0.  Outside that
domain the extended ground is an explicit new modelling convention.

The float64 oracle measures distance to exposed rectangles, independently of
SdfLib and of the exported triangulation, and gets sign from box occupancy.
It is the SDF of this union, not sign times distance to the original raw faces.
Its Torch version deliberately does not build an autograd graph.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import product
from pathlib import Path

import numpy as np

from devtools.config import load_config
from devtools.mesh import MeshObject
from devtools.scene import Pose, Scene


_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MAP = _ROOT / "docs/explorations/esdf/artifacts/outputs/mesh_distance_eval/reviewed-20261015/pillar_omni.obj"
DEFAULT_CUBE_BOUNDS = np.array([[-51.2, -51.2, -43.7], [51.2, 51.2, 58.7]], dtype=np.float64)


def _bounds(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all() or np.any(result[1] <= result[0]):
        raise ValueError(f"{name} must be finite increasing bounds with shape (2,3): [lo,hi]")
    return result.copy()


def _as_box(mesh):
    """Reject shapes for which interpreting only their extrema would be false."""
    vertices = np.asarray(mesh.vertices_m, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.shape != (8, 3) or triangles.shape != (12, 3):
        raise ValueError(f"{mesh.name}: expected an eight-vertex, twelve-triangle box")
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    expected = set(product(*zip(lo, hi)))
    if np.any(hi <= lo) or set(map(tuple, vertices)) != expected:
        raise ValueError(f"{mesh.name}: obstacle is not an axis-aligned box")
    plane_areas = np.zeros((3, 2), dtype=np.float64)
    for triangle in vertices[triangles]:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        nonzero = np.flatnonzero(normal != 0)
        if len(nonzero) != 1:
            raise ValueError(f"{mesh.name}: triangle is not on an axis-aligned box face")
        axis = int(nonzero[0])
        side = int(normal[axis] > 0)
        coordinate = hi[axis] if side else lo[axis]
        if not np.all(triangle[:, axis] == coordinate):
            raise ValueError(f"{mesh.name}: box triangle has an inward or misplaced normal")
        plane_areas[axis, side] += abs(normal[axis]) * 0.5
    extent = hi - lo
    expected_areas = np.array([extent[1] * extent[2], extent[2] * extent[0], extent[0] * extent[1]])
    if not np.allclose(plane_areas, expected_areas[:, None], rtol=1e-12, atol=0):
        raise ValueError(f"{mesh.name}: triangles do not cover all six box faces once")
    return np.stack((lo, hi))


def _subtract_rectangle(rectangle, cover):
    """Nonoverlapping rectangle pieces after subtracting one closed rectangle.

    Coordinates are (u0,v0,u1,v1).  All cuts reuse input coordinates, so adjacent
    faces get exactly equal endpoints rather than separately rounded estimates.
    Boundary-only intersections do not change the two-dimensional surface.
    """
    u0, v0, u1, v1 = rectangle
    a0, b0 = max(u0, cover[0]), max(v0, cover[1])
    a1, b1 = min(u1, cover[2]), min(v1, cover[3])
    if a0 >= a1 or b0 >= b1:
        return [rectangle]
    pieces = []
    if u0 < a0:
        pieces.append((u0, v0, a0, v1))
    if a1 < u1:
        pieces.append((a1, v0, u1, v1))
    if v0 < b0:
        pieces.append((a0, v0, a1, b0))
    if b1 < v1:
        pieces.append((a0, b1, a1, v1))
    return pieces


def _exposed_rectangles(boxes):
    """Exact AABB face subtraction; no voxelization or offset-point sign test.

    The B^2 broadphase checks boxes against each source face.  Subtraction work
    depends on local overlaps and the resulting surface arrangement; it is not
    a global Cartesian product of all x/y/z coordinates.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    lo, hi = boxes[:, 0], boxes[:, 1]
    planes = defaultdict(list)
    records = []
    for source in range(len(boxes)):
        for axis in range(3):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            footprint = (lo[source, u], lo[source, v], hi[source, u], hi[source, v])
            for sign in (-1, 1):
                coordinate = boxes[source, int(sign > 0), axis]
                # These strict/non-strict pairs ask about the immediate outward
                # side, including touching boxes, without a numerical epsilon.
                if sign > 0:
                    covers = (lo[:, axis] <= coordinate) & (hi[:, axis] > coordinate)
                else:
                    covers = (lo[:, axis] < coordinate) & (hi[:, axis] >= coordinate)
                covers &= (lo[:, u] < footprint[2]) & (hi[:, u] > footprint[0])
                covers &= (lo[:, v] < footprint[3]) & (hi[:, v] > footprint[1])
                covers[source] = False
                pieces = [footprint]
                for other in np.flatnonzero(covers):
                    cover = (lo[other, u], lo[other, v], hi[other, u], hi[other, v])
                    pieces = [part for piece in pieces for part in _subtract_rectangle(piece, cover)]
                    if not pieces:
                        break
                key = (axis, sign, float(coordinate))
                # Two boxes can expose overlapping regions of the same plane.
                # Keep the region once, even when their face extents differ.
                for accepted in planes[key]:
                    pieces = [part for piece in pieces for part in _subtract_rectangle(piece, accepted)]
                    if not pieces:
                        break
                planes[key].extend(pieces)
                for piece in pieces:
                    lower = np.zeros(3, dtype=np.float64)
                    upper = np.zeros(3, dtype=np.float64)
                    lower[axis] = upper[axis] = coordinate
                    lower[u], lower[v], upper[u], upper[v] = piece
                    records.append((lower, upper, axis, sign))
    return (np.array([item[0] for item in records]), np.array([item[1] for item in records]),
            np.array([item[2] for item in records], dtype=np.int8),
            np.array([item[3] for item in records], dtype=np.int8))


def _triangulate_rectangles(lower, upper, axes, signs):
    """Split T-junction edges globally, then fan each rectangle from its centre."""
    polygons = []
    line_points = defaultdict(set)
    for lo, hi, axis, sign in zip(lower, upper, axes, signs):
        axis = int(axis)
        u, v = (axis + 1) % 3, (axis + 2) % 3
        corners = []
        for high_u, high_v in ((False, False), (True, False), (True, True), (False, True)):
            corner = lo.copy()
            corner[u] = hi[u] if high_u else lo[u]
            corner[v] = hi[v] if high_v else lo[v]
            corners.append(tuple(corner))
        if sign < 0:
            corners.reverse()
        polygons.append(corners)
        # A corner can meet an edge from any of the three axis directions.
        for corner in corners:
            for direction in range(3):
                fixed = tuple(corner[i] for i in range(3) if i != direction)
                line_points[(direction, fixed)].add(corner[direction])
    line_points = {key: sorted(values) for key, values in line_points.items()}
    vertices, triangles, vertex_ids = [], [], {}

    def vertex_id(point):
        key = tuple(point)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]

    for corners, lo, hi in zip(polygons, lower, upper):
        boundary = []
        for start, end in zip(corners, corners[1:] + corners[:1]):
            direction = next(i for i in range(3) if start[i] != end[i])
            fixed = tuple(start[i] for i in range(3) if i != direction)
            values = line_points[(direction, fixed)]
            low, high = sorted((start[direction], end[direction]))
            splits = values[bisect_left(values, low):bisect_right(values, high)]
            if start[direction] > end[direction]:
                splits.reverse()
            for position in splits[:-1]:
                point = list(start)
                point[direction] = position
                boundary.append(vertex_id(point))
        centre = vertex_id((lo + hi) * 0.5)
        for start, end in zip(boundary, boundary[1:] + boundary[:1]):
            triangles.append((centre, start, end))
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    edges = defaultdict(lambda: [0, 0])
    for triangle in triangles:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edges[key][0] += 1
            edges[key][1] += 1 if a < b else -1
    bad = [(key, value) for key, value in edges.items() if value != [2, 0]]
    if bad:
        raise ValueError(f"union boundary is not a closed oriented two-manifold: {len(bad)} bad edges; {bad[:4]}")
    triangle_points = vertices[triangles]
    double_area = np.linalg.norm(np.cross(triangle_points[:, 1] - triangle_points[:, 0],
                                         triangle_points[:, 2] - triangle_points[:, 0]), axis=1)
    if not np.all(double_area > 0):
        raise ValueError("union triangulation contains a degenerate triangle")
    return vertices, triangles, {
        "closed_oriented_edge_check": True,
        "unique_edge_count": len(edges),
        "bad_edge_count": 0,
        "minimum_triangle_area_m2": float(double_area.min() * 0.5),
        "surface_area_m2": float(double_area.sum() * 0.5),
        "t_junctions": "rectangle edges split at every coincident axis-line corner before triangulation",
    }


@dataclass
class RealMapGeometry:
    """Prepared geometry; arrays describe world coordinates in metres.

    boxes/box_lower/box_upper contain only the original pillars.  solid_boxes
    additionally contains the extended ground slab.  Rectangles have one equal
    lower/upper coordinate, specified by rectangle_axes and rectangle_signs.
    """

    logical_bounds: np.ndarray
    cube_bounds: np.ndarray
    boxes: np.ndarray
    ground_slab_bounds: np.ndarray
    rectangle_lower: np.ndarray
    rectangle_upper: np.ndarray
    rectangle_axes: np.ndarray
    rectangle_signs: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    overlap_pairs: np.ndarray
    metadata: dict

    @property
    def box_lower(self):
        return self.boxes[:, 0]

    @property
    def box_upper(self):
        return self.boxes[:, 1]

    @property
    def solid_boxes(self):
        return np.concatenate((self.boxes, self.ground_slab_bounds[None]), axis=0)

    def write_union_obj(self, path):
        """Write a separate derived asset, refusing to overwrite the source."""
        path = Path(path)
        if path.resolve() == Path(self.metadata["source_obj"]).resolve():
            raise ValueError("the source OBJ must remain unchanged")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            stream.write("# unit metre\n# exposed boundary of original pillar union and extended ground slab\n")
            stream.write(f"# source_sha256 {self.metadata['source_sha256']}\n")
            stream.write("o closed_union\n")
            for point in self.vertices:
                stream.write("v " + " ".join(format(float(value), ".17g") for value in point) + "\n")
            for triangle in self.triangles:
                stream.write("f " + " ".join(str(int(value) + 1) for value in triangle) + "\n")
        return path

    def distance_numpy(self, points, *, point_chunk=1024, rectangle_chunk=512):
        """Float64 signed point distance; nonfinite inputs produce NaN."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points must have shape [...,3]")
        if point_chunk <= 0 or rectangle_chunk <= 0:
            raise ValueError("chunk sizes must be positive")
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        result = np.full(len(flat), np.nan, dtype=np.float64)
        solids = self.solid_boxes
        for start in range(0, len(flat), point_chunk):
            chunk = flat[start:start + point_chunk]
            valid = np.isfinite(chunk).all(axis=1)
            p = chunk[valid]
            squared = np.full(len(p), np.inf, dtype=np.float64)
            for offset in range(0, len(self.rectangle_lower), rectangle_chunk):
                lo = self.rectangle_lower[offset:offset + rectangle_chunk]
                hi = self.rectangle_upper[offset:offset + rectangle_chunk]
                delta = np.maximum(np.maximum(lo[None] - p[:, None], p[:, None] - hi[None]), 0)
                squared = np.minimum(squared, np.einsum("ijk,ijk->ij", delta, delta).min(axis=1))
            inside = np.zeros(len(p), dtype=bool)
            for offset in range(0, len(solids), rectangle_chunk):
                boxes = solids[offset:offset + rectangle_chunk]
                inside |= np.all((p[:, None] >= boxes[None, :, 0]) &
                                 (p[:, None] <= boxes[None, :, 1]), axis=2).any(axis=1)
            result[start:start + len(chunk)][valid] = np.sqrt(squared) * np.where(inside, -1, 1)
        return result.reshape(shape)

    def distance_torch(self, points, *, point_chunk=1024, rectangle_chunk=512):
        """Independent no-grad float64 oracle on the input Torch device."""
        import torch

        if not isinstance(points, torch.Tensor) or points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points must be a Torch tensor with shape [...,3]")
        if point_chunk <= 0 or rectangle_chunk <= 0:
            raise ValueError("chunk sizes must be positive")
        with torch.no_grad():
            shape = points.shape[:-1]
            flat = points.detach().to(dtype=torch.float64).reshape(-1, 3)
            result = torch.full((len(flat),), float("nan"), dtype=torch.float64, device=points.device)
            lower = torch.as_tensor(self.rectangle_lower, dtype=torch.float64, device=points.device)
            upper = torch.as_tensor(self.rectangle_upper, dtype=torch.float64, device=points.device)
            solids = torch.as_tensor(self.solid_boxes, dtype=torch.float64, device=points.device)
            for start in range(0, len(flat), point_chunk):
                chunk = flat[start:start + point_chunk]
                valid = torch.isfinite(chunk).all(dim=1)
                p = chunk[valid]
                squared = torch.full((len(p),), float("inf"), dtype=torch.float64, device=points.device)
                for offset in range(0, len(lower), rectangle_chunk):
                    lo, hi = lower[offset:offset + rectangle_chunk], upper[offset:offset + rectangle_chunk]
                    delta = torch.maximum(lo[None] - p[:, None], p[:, None] - hi[None]).clamp_min(0)
                    squared = torch.minimum(squared, delta.square().sum(dim=2).amin(dim=1))
                inside = torch.zeros(len(p), dtype=torch.bool, device=points.device)
                for offset in range(0, len(solids), rectangle_chunk):
                    boxes = solids[offset:offset + rectangle_chunk]
                    inside |= ((p[:, None] >= boxes[None, :, 0]) &
                               (p[:, None] <= boxes[None, :, 1])).all(dim=2).any(dim=1)
                result[start:start + len(chunk)][valid] = squared.sqrt() * torch.where(inside, -1., 1.)
            return result.reshape(shape)


def load_real_map(path=DEFAULT_MAP, *, cube_bounds=DEFAULT_CUBE_BOUNDS,
                  ground_margin_m=None, expected_pillars=500):
    """Load the complete existing map and prepare its closed union boundary.

    Bounds use shape (2,3), [lower_xyz, upper_xyz].  The cube must contain the
    complete declared map.  ground_margin_m, if supplied, may only increase the
    conservative margin; it cannot move remote slab faces close to the cube.
    expected_pillars=None permits other explicitly supplied box-only fixtures.
    """
    path = Path(path).resolve()
    cube = _bounds(cube_bounds, "cube_bounds")
    if not np.allclose(cube[1] - cube[0], (cube[1] - cube[0])[0], rtol=0, atol=1e-10):
        raise ValueError("SdfLib query bounds must be a cube")
    scene = Scene.from_obj(path, load_config(_ROOT / "configs/devtools.json"),
                           Pose((0., 0., 1.)), (0., 0., 1.))
    snapshot = scene.get_geometry(exclude_ids={"vehicle", "target"})
    if snapshot.map_bounds_m is None:
        raise ValueError("the source map must declare its logical bounds")
    logical = _bounds(np.asarray(snapshot.map_bounds_m).T, "logical bounds")
    if np.any(cube[0] > logical[0]) or np.any(cube[1] < logical[1]):
        raise ValueError("cube_bounds must contain the entire declared logical map")
    objects = snapshot.objects
    if any(not isinstance(obj.geometry, MeshObject) or obj.role not in {"obstacle", "ground"}
           for obj in objects):
        raise ValueError("only public obstacle and ground MeshObjects are supported")
    obstacles = [obj for obj in objects if obj.role == "obstacle"]
    grounds = [obj for obj in objects if obj.role == "ground"]
    if not obstacles or len(grounds) != 1:
        raise ValueError("expected at least one pillar and exactly one rectangular ground")
    if expected_pillars is not None and len(obstacles) != expected_pillars:
        raise ValueError(f"expected all {expected_pillars} pillars; received {len(obstacles)}")
    boxes = np.stack([_as_box(obj.geometry) for obj in obstacles])
    if np.any(boxes[:, 0] < logical[0]) or np.any(boxes[:, 1] > logical[1]) or np.any(boxes[:, 0, 2] != 0):
        raise ValueError("pillars must stand at z=0 and stay within the declared map")
    ground = np.asarray(grounds[0].geometry.vertices_m, dtype=np.float64)
    expected_ground = {(x, y, 0.) for x, y in product(logical[:, 0], logical[:, 1])}
    if logical[0, 2] != 0 or ground.shape != (4, 3) or set(map(tuple, ground)) != expected_ground:
        raise ValueError("ground must span exactly the logical XY rectangle at z=0")

    # An exposed top-slab witness at x=cube_hi_x+1, y=query_y, z=0 is outside
    # all pillars. Its maximum possible distance from any cube point is W.
    # Every slab side/bottom is at least margin > W away from the cube.
    width_x = cube[1, 0] - cube[0, 0]
    witness_bound = float(np.hypot(width_x + 1., max(abs(cube[0, 2]), abs(cube[1, 2]))))
    minimum_margin = witness_bound + 1.
    margin = minimum_margin if ground_margin_m is None else float(ground_margin_m)
    if not np.isfinite(margin) or margin < minimum_margin:
        raise ValueError(f"ground_margin_m must be at least {minimum_margin:.17g}")
    slab = np.array([[cube[0, 0] - margin, cube[0, 1] - margin, min(cube[0, 2], 0.) - margin],
                     [cube[1, 0] + margin, cube[1, 1] + margin, 0.]], dtype=np.float64)
    solids = np.concatenate((boxes, slab[None]), axis=0)
    rectangles = _exposed_rectangles(solids)
    vertices, triangles, topology = _triangulate_rectangles(*rectangles)
    overlap_pairs, overlap_records = [], []
    for i, box in enumerate(boxes):
        lower = np.maximum(box[0], boxes[i + 1:, 0])
        upper = np.minimum(box[1], boxes[i + 1:, 1])
        for offset in np.flatnonzero(np.all(upper > lower, axis=1)):
            j = i + 1 + int(offset)
            overlap_pairs.append((i, j))
            overlap_records.append({"indices": [i, j], "ids": [obstacles[i].object_id, obstacles[j].object_id],
                                    "bounds": [lower[offset].tolist(), upper[offset].tolist()],
                                    "volume_m3": float(np.prod(upper[offset] - lower[offset]))})
    metadata = {
        "source_obj": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "public_interface": "Scene.from_obj(...).get_geometry(exclude_ids={'vehicle','target'})",
        "geometry_scope": "complete original axis-aligned pillar union plus explicitly extended closed ground slab",
        "original_object_count": len(objects),
        "original_pillar_count": len(boxes),
        "original_obstacle_vertex_count": sum(len(obj.geometry.vertices_m) for obj in obstacles),
        "original_obstacle_triangle_count": sum(len(obj.geometry.triangles) for obj in obstacles),
        "original_ground_vertex_count": len(ground),
        "original_ground_triangle_count": len(grounds[0].geometry.triangles),
        "object_ids": [obj.object_id for obj in obstacles],
        "logical_bounds": logical.tolist(), "cube_bounds": cube.tolist(),
        "boxes": boxes.tolist(), "ground_slab_bounds": slab.tolist(),
        "ground_adaptation": "source finite open sheet extended to a closed slab; physical ground convention retained only in declared logical XY domain and z>=0",
        "ground_margin_m": margin,
        "slab_remote_face_min_distance_from_cube_m": margin,
        "exposed_ground_witness_max_distance_m": witness_bound,
        "slab_remote_faces_cannot_be_nearest_in_cube": margin > witness_bound,
        "ground_union_semantics": "pillar bottom faces are removed; occupied-side distances measure exposed union boundary, not raw internal faces",
        "overlap_pair_count": len(overlap_pairs), "overlap_pairs": overlap_records,
        "rectangle_count": len(rectangles[0]), "vertex_count": len(vertices), "triangle_count": len(triangles),
        "boundary_construction": "axis-aligned face rectangle subtraction; no voxel approximation; T-junction-conforming centre-fan triangulation",
        "oracle": "float64 distance to exposed rectangles; union occupancy sign; independent of SdfLib and triangulation",
        **topology,
    }
    return RealMapGeometry(logical, cube, boxes, slab, *rectangles, vertices, triangles,
                           np.asarray(overlap_pairs, dtype=np.int64).reshape(-1, 2), metadata)
