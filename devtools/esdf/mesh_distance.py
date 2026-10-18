"""Offline CPU distances to explicitly prepared, closed mesh boundaries.

GeometrySnapshot enters through public world-frame data. Box unions and the
ground convention are prepared here; the C++ query is not a Boolean operation.
No Torch, octree, SdfLib runtime, automatic compilation, or process-wide cache.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import ctypes
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import threading
import weakref

import numpy as np

from ..mesh import MeshObject
from ..scene import GeometrySnapshot


def _bounds(value):
    bounds = np.asarray(value, dtype=np.float64)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0]) or not np.isfinite(bounds[1]-bounds[0]).all()):
        raise ValueError("query_bounds_m must be finite increasing [lo_xyz,hi_xyz], shape (2,3)")
    return bounds.copy()


def _mesh_arrays(vertices, triangles):
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or vertices.dtype.kind not in "iuf":
        raise ValueError("vertices must be a numeric array with shape (N,3), N>=4")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 4 or triangles.dtype.kind not in "iu":
        raise ValueError("triangles must be an integer array with shape (M,3), M>=4")
    if not np.isfinite(vertices).all() or len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("mesh vertices must be finite and indexable by int32")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangle indices are outside the vertex array")
    return np.array(vertices, dtype=np.float64, order="C", copy=True), np.array(triangles, dtype=np.int32, order="C", copy=True)


def _as_box(mesh):
    """Only accept complete outward axis-aligned boxes, never just extrema."""
    vertices = np.asarray(mesh.vertices_m, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.shape != (8, 3) or triangles.shape != (12, 3):
        raise ValueError(f"{mesh.name}: expected an eight-vertex, twelve-triangle box")
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(hi <= lo) or set(map(tuple, vertices)) != set(product(*zip(lo, hi))):
        raise ValueError(f"{mesh.name}: obstacle is not an axis-aligned box")
    areas = np.zeros((3, 2), dtype=np.float64)
    for triangle in vertices[triangles]:
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        nonzero = np.flatnonzero(normal != 0)
        if len(nonzero) != 1:
            raise ValueError(f"{mesh.name}: triangle is not on a box face")
        axis = int(nonzero[0])
        side = int(normal[axis] > 0)
        if not np.all(triangle[:, axis] == (hi[axis] if side else lo[axis])):
            raise ValueError(f"{mesh.name}: inward or misplaced box face")
        areas[axis, side] += abs(normal[axis]) * .5
    d = hi-lo
    if not np.allclose(areas, np.array([d[1]*d[2], d[2]*d[0], d[0]*d[1]])[:, None], rtol=1e-12, atol=0):
        raise ValueError(f"{mesh.name}: triangles must cover every box face once")
    return np.stack((lo, hi))


def _subtract_rectangle(rectangle, cover):
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
    """Subtract covered faces and duplicate coplanar regions before triangulation."""
    lo, hi = boxes[:, 0], boxes[:, 1]
    planes, records = defaultdict(list), []
    for source in range(len(boxes)):
        for axis in range(3):
            u, v = (axis+1) % 3, (axis+2) % 3
            footprint = (lo[source,u], lo[source,v], hi[source,u], hi[source,v])
            for sign in (-1, 1):
                coordinate = boxes[source, int(sign > 0), axis]
                # Test the immediate outward side, including touching boxes,
                # with exact inequalities rather than an offset epsilon.
                if sign > 0:
                    covers = (lo[:,axis] <= coordinate) & (hi[:,axis] > coordinate)
                else:
                    covers = (lo[:,axis] < coordinate) & (hi[:,axis] >= coordinate)
                covers &= (lo[:,u] < footprint[2]) & (hi[:,u] > footprint[0])
                covers &= (lo[:,v] < footprint[3]) & (hi[:,v] > footprint[1])
                covers[source] = False
                pieces = [footprint]
                for other in np.flatnonzero(covers):
                    cover = (lo[other,u], lo[other,v], hi[other,u], hi[other,v])
                    pieces = [part for piece in pieces for part in _subtract_rectangle(piece, cover)]
                    if not pieces:
                        break
                key = (axis, sign, float(coordinate))
                for accepted in planes[key]:
                    pieces = [part for piece in pieces for part in _subtract_rectangle(piece, accepted)]
                    if not pieces:
                        break
                planes[key].extend(pieces)
                for piece in pieces:
                    lower, upper = np.zeros(3), np.zeros(3)
                    lower[axis] = upper[axis] = coordinate
                    lower[u], lower[v], upper[u], upper[v] = piece
                    records.append((lower, upper, axis, sign))
    return (np.array([r[0] for r in records]), np.array([r[1] for r in records]),
            np.array([r[2] for r in records]), np.array([r[3] for r in records]))


def _triangulate_rectangles(lower, upper, axes, signs):
    """Conform every rectangle edge to shared T-junctions, then centre-fan it."""
    polygons, line_points = [], defaultdict(set)
    for lo, hi, axis, sign in zip(lower, upper, axes, signs):
        axis = int(axis)
        u, v = (axis+1) % 3, (axis+2) % 3
        corners = []
        for high_u, high_v in ((False,False), (True,False), (True,True), (False,True)):
            point = lo.copy()
            point[u], point[v] = hi[u] if high_u else lo[u], hi[v] if high_v else lo[v]
            corners.append(tuple(point))
        if sign < 0:
            corners.reverse()
        polygons.append(corners)
        for corner in corners:
            for direction in range(3):
                fixed = tuple(corner[i] for i in range(3) if i != direction)
                line_points[direction, fixed].add(corner[direction])
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
        for start, end in zip(corners, corners[1:]+corners[:1]):
            direction = next(i for i in range(3) if start[i] != end[i])
            fixed = tuple(start[i] for i in range(3) if i != direction)
            values = line_points[direction, fixed]
            low, high = sorted((start[direction], end[direction]))
            splits = values[bisect_left(values, low):bisect_right(values, high)]
            if start[direction] > end[direction]:
                splits.reverse()
            for position in splits[:-1]:
                point = list(start)
                point[direction] = position
                boundary.append(vertex_id(point))
        centre = vertex_id(lo + (hi-lo)*.5)
        triangles.extend((centre, a, b) for a,b in zip(boundary, boundary[1:]+boundary[:1]))
    vertices, triangles = np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)
    edges = defaultdict(lambda: [0, 0])
    for triangle in triangles:
        for a,b in zip(triangle, np.roll(triangle,-1)):
            edges[min(int(a),int(b)), max(int(a),int(b))][0] += 1
            edges[min(int(a),int(b)), max(int(a),int(b))][1] += 1 if a < b else -1
    if any(value != [2,0] for value in edges.values()):
        raise ValueError("box union has a nonmanifold or nonconforming boundary")
    p = vertices[triangles]
    area2 = np.linalg.norm(np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]), axis=1)
    if not np.isfinite(area2).all() or np.any(area2 <= 0):
        raise ValueError("box union triangulation is degenerate")
    return vertices, triangles, {"closed_oriented_edge_check": True, "unique_edge_count": len(edges),
        "minimum_triangle_area_m2": float(area2.min()*.5)}


@dataclass(frozen=True)
class PreparedMesh:
    """Owned world-metre arrays; complete exposed boundary, not an occupancy grid.

    Shape/index checks run here; complete topology/orientation checks also run
    in MeshDistanceQuery before the BVH is constructed. Direct boundary callers
    must supply non-self-intersecting exposed surfaces without nested components.
    Arrays are read-only float64 vertices / int32 triangles; metadata is JSON
    compatible and records geometry hashes, object IDs, bounds and conventions.
    query_bounds_m bounds the ground extension guarantee, not mesh cropping.
    """
    vertices_m: np.ndarray
    triangles: np.ndarray
    query_bounds_m: np.ndarray
    metadata: dict

    def __post_init__(self):
        vertices, triangles = _mesh_arrays(self.vertices_m, self.triangles)
        bounds = _bounds(self.query_bounds_m)
        for array in (vertices, triangles, bounds):
            array.flags.writeable = False
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "query_bounds_m", bounds)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_boundary_mesh(cls, vertices, triangles, *, query_bounds_m):
        """Accept a caller-prepared closed external boundary; performs no Boolean."""
        return cls(vertices, triangles, query_bounds_m, {
            "geometry_semantics": "caller-supplied closed exposed boundary; no Boolean operation",
            "input_contract": "non-self-intersecting, outward components; no overlapping or nested components",
        })

    @classmethod
    def from_geometry(cls, snapshot, *, query_bounds_m):
        """Prepare all snapshot boxes and optional ground as a geometric union.

        Spheres must be excluded by the caller. A single arbitrary closed mesh
        without ground is accepted as an already exposed boundary. Other shapes
        mixed with ground or other objects require a caller-prepared Boolean.
        Ground becomes occupied z<=0 over the complete query bounds (including
        any requested halo); this explicitly extends the finite source sheet.
        A closed slab covers all objects and query bounds, with side/bottom
        faces farther away than a known exposed-ground witness. Internal box
        and ground faces are removed by rectangle subtraction, followed by
        shared T-junction splits and centre-fan triangulation. This computes
        the exposed union boundary without voxelization or min-of-SDF errors.
        Edge/point contacts producing nonmanifold boundaries are rejected.
        """
        if not isinstance(snapshot, GeometrySnapshot):
            raise TypeError("snapshot must be the public GeometrySnapshot")
        bounds = _bounds(query_bounds_m)
        objects = snapshot.objects
        if not objects or any(not isinstance(obj.geometry, MeshObject) for obj in objects):
            raise ValueError("nonempty mesh geometry required; exclude vehicle/target spheres explicitly")
        obstacles = [obj for obj in objects if obj.role == "obstacle"]
        grounds = [obj for obj in objects if obj.role == "ground"]
        if len(grounds) > 1 or len(obstacles)+len(grounds) != len(objects):
            raise ValueError("only obstacle meshes and at most one rectangular ground are supported")
        source = [{"id": obj.object_id, "role": obj.role, "vertices_m": obj.geometry.vertices_m,
                   "triangles": obj.geometry.triangles} for obj in objects]
        provenance = {"source_geometry_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
            "object_ids": [obj.object_id for obj in objects], "obstacle_count": len(obstacles),
            "source_map_bounds_m": snapshot.map_bounds_m, "query_bounds_m": bounds.tolist()}
        try:
            boxes = np.array([_as_box(obj.geometry) for obj in obstacles], dtype=np.float64).reshape(-1,2,3)
        except ValueError as error:
            if len(obstacles) == 1 and not grounds:
                mesh = obstacles[0].geometry
                return cls(mesh.vertices_m, mesh.triangles, bounds, {**provenance,
                    "geometry_semantics": "one caller-supplied closed obstacle; no Boolean operation"})
            raise ValueError("mixed/arbitrary meshes require an exposed Boolean boundary; min of object SDFs is not supported") from error
        solids = boxes
        ground_metadata = {"ground_convention": "none"}
        if grounds:
            ground = np.asarray(grounds[0].geometry.vertices_m, dtype=np.float64)
            # MeshObject already validates the +Z rectangle at z=0. Include all
            # boxes, the source ground, and the query halo when extending it.
            extents = np.vstack((bounds, ground, boxes.reshape(-1,3)))
            lo, hi = extents.min(axis=0), extents.max(axis=0)
            witness_x = hi[0] + 1.0
            witness_bound = float(np.hypot(witness_x-bounds[0,0], np.abs(bounds[:,2]).max()))
            margin = witness_bound + 1.0
            slab = np.array([[lo[0]-margin,lo[1]-margin,min(lo[2],0.)-margin],
                             [hi[0]+margin,hi[1]+margin,0.]], dtype=np.float64)
            solids = np.concatenate((boxes, slab[None]), axis=0)
            ground_metadata = {"ground_convention": "occupied z<=0; finite source rectangle explicitly extended across query bounds",
                "source_ground_bounds_m": [ground.min(axis=0).tolist(), ground.max(axis=0).tolist()],
                "ground_slab_bounds_m": slab.tolist(), "remote_face_min_distance_m": margin,
                "exposed_ground_witness_max_distance_m": witness_bound,
                "remote_faces_cannot_be_nearest_in_query_bounds": bool(margin > witness_bound),
                "ground_boundary_semantics": "pillar bottom/internal faces removed with all other union-internal faces"}
        rectangles = _exposed_rectangles(solids)
        vertices, triangles, topology = _triangulate_rectangles(*rectangles)
        overlap_pairs = []
        for i, box in enumerate(boxes):
            overlap = np.minimum(box[1], boxes[i+1:,1]) > np.maximum(box[0], boxes[i+1:,0])
            overlap_pairs.extend([i, i+1+int(j)] for j in np.flatnonzero(overlap.all(axis=1)))
        return cls(vertices, triangles, bounds, {**provenance, **ground_metadata, **topology,
            "geometry_semantics": "exposed boundary of axis-aligned box union; no voxel approximation",
            "boxes_m": boxes.tolist(), "overlap_pair_count": len(overlap_pairs), "overlap_pairs": overlap_pairs,
            "rectangle_count": len(rectangles[0]), "vertex_count": len(vertices), "triangle_count": len(triangles),
            "triangulation": "rectangle subtraction plus shared T-junction splits and centre fans"})


class MeshDistanceQuery:
    """One owned CPU BVH; callable finite Nx3 positions -> float64 N distances.

    Construction checks finite/nondegenerate faces, closed edges, manifold vertex
    fans and outward positive-volume components. It does not detect all triangle
    intersections or resolve overlapping/nested components. Queries and close
    are serialized; no gradients are built here. Use an explicit library path.
    Each instance copies its input, owns one BVH and runs a single-threaded
    double-precision batch loop. Noncontiguous/float32 positions are converted
    to contiguous float64; empty batches are allowed, nonfinite input rejected.
    C ABI exceptions become Python errors without partially returned results.
    Use a context manager or close(); closing twice is harmless.

    The native wrapper uses the third-party BVH's selected triangle, then
    recomputes projection/edge residuals in double precision. Strict face
    interiors use the oriented-plane sign; edges/vertices use pseudonormals.
    This is numerical mesh distance, not an exact-arithmetic certificate.
    """
    def __init__(self, vertices, triangles, *, library_path):
        vertices, triangles = _mesh_arrays(vertices, triangles)
        library = ctypes.CDLL(str(Path(library_path).resolve(strict=True)))
        f64 = ctypes.POINTER(ctypes.c_double)
        i32 = ctypes.POINTER(ctypes.c_int32)
        library.mdq_abi_version.argtypes = []
        library.mdq_abi_version.restype = ctypes.c_int
        if library.mdq_abi_version() != 1:
            raise RuntimeError("incompatible mesh-distance C ABI")
        library.mdq_create.argtypes = [f64, ctypes.c_uint64, i32, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        library.mdq_create.restype = ctypes.c_int
        library.mdq_query.argtypes = [ctypes.c_void_p, f64, ctypes.c_uint64, f64, ctypes.c_char_p, ctypes.c_size_t]
        library.mdq_query.restype = ctypes.c_int
        library.mdq_destroy.argtypes = [ctypes.c_void_p]
        library.mdq_destroy.restype = None
        handle, error = ctypes.c_void_p(), ctypes.create_string_buffer(2048)
        status = library.mdq_create(vertices.ctypes.data_as(f64), len(vertices),
            triangles.ctypes.data_as(i32), len(triangles), ctypes.byref(handle), error, len(error))
        if status:
            raise ValueError(error.value.decode("utf-8", errors="replace"))
        self._library, self._handle, self._lock = library, handle, threading.Lock()
        self._finalizer = weakref.finalize(self, library.mdq_destroy, handle)

    def __call__(self, points):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "iuf":
            raise ValueError("query points must be numeric Nx3")
        points = np.ascontiguousarray(points, dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("query points must be finite")
        output = np.empty(len(points), dtype=np.float64)
        pointer = ctypes.POINTER(ctypes.c_double)
        error = ctypes.create_string_buffer(2048)
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("mesh-distance query is closed")
            status = self._library.mdq_query(self._handle, points.ctypes.data_as(pointer), len(points),
                output.ctypes.data_as(pointer), error, len(error))
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        return output

    def close(self):
        """Release the BVH; repeated close calls are harmless."""
        with self._lock:
            self._finalizer()

    def __enter__(self):
        if not self._finalizer.alive:
            raise RuntimeError("mesh-distance query is closed")
        return self

    def __exit__(self, *_):
        self.close()
