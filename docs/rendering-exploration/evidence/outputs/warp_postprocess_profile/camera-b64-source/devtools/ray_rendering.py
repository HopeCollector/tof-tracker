"""One persistent world scene and batched, GPU-only closest ray queries.

The core knows no sensor, image layout, installation transform or measurement
noise. Meshes share one Warp BVH; physical spheres remain analytic primitives.
All reported distances are radial metres along normalized input directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import numpy as np
import torch
import warp as wp

from .mesh import MeshObject
from .scene import GeometrySnapshot, Sphere


@wp.kernel
def _trace_rays(mesh: wp.uint64, has_mesh: int,
                face_objects: wp.array(dtype=wp.int64),
                centers: wp.array(dtype=wp.vec3),
                radii: wp.array(dtype=float),
                sphere_objects: wp.array(dtype=wp.int64), sphere_count: int,
                origins: wp.array(dtype=wp.vec3),
                directions: wp.array(dtype=wp.vec3),
                limits: wp.array(dtype=float), scalar_limit: float,
                per_ray_limit: int, selected_object: int,
                distances: wp.array(dtype=float),
                objects: wp.array(dtype=wp.int64),
                points: wp.array(dtype=wp.vec3),
                faces: wp.array(dtype=wp.int64)):
    i = wp.tid()
    distances[i] = 0.0
    objects[i] = wp.int64(-1)
    points[i] = wp.vec3(0.0)
    faces[i] = wp.int64(-1)
    origin = origins[i]
    direction = directions[i]
    limit = scalar_limit
    if per_ray_limit != 0:
        limit = limits[i]
    finite_origin = (wp.isfinite(origin[0]) and wp.isfinite(origin[1])
                     and wp.isfinite(origin[2]))
    finite_direction = (wp.isfinite(direction[0]) and wp.isfinite(direction[1])
                        and wp.isfinite(direction[2]))
    if finite_origin and finite_direction and wp.isfinite(limit) and limit > 0.0:
        # Scale before normalization so finite, unusually large/small direction
        # components cannot overflow/underflow the length calculation.
        scale = wp.max(wp.abs(direction[0]),
                       wp.max(wp.abs(direction[1]), wp.abs(direction[2])))
        if scale > 0.0:
            direction = wp.normalize(direction / scale)
            # Float32 intersection arithmetic can round a boundary hit upward.
            # This guard is numerical, not a sensor near/far validity policy.
            query_limit = limit + wp.max(1.0e-6, limit * 1.0e-6)
            nearest = query_limit
            identity = wp.int64(-1)
            face = wp.int64(-1)
            if has_mesh != 0 and selected_object < 0:
                hit = wp.mesh_query_ray(mesh, origin, direction, query_limit)
                if hit.result and hit.t >= 0.0 and wp.isfinite(hit.t):
                    nearest = hit.t
                    identity = face_objects[hit.face]
                    face = wp.int64(hit.face)
            for j in range(sphere_count):
                sphere_identity = sphere_objects[j]
                if selected_object < 0 or sphere_identity == wp.int64(selected_object):
                    relative = origin - centers[j]
                    b = wp.dot(relative, direction)
                    # Perpendicular-distance form avoids subtracting b*b from
                    # squared center distance for small distant spheres.
                    perpendicular = relative - b * direction
                    discriminant = radii[j] * radii[j] - wp.dot(perpendicular, perpendicular)
                    if discriminant >= 0.0:
                        root = wp.sqrt(discriminant)
                        t = -b - root
                        if t < 0.0:
                            t = -b + root
                        if t >= 0.0 and t <= nearest and wp.isfinite(t):
                            # At exact equal distances preserve the earlier
                            # winner, keeping metadata and hit point coherent.
                            if t < nearest or identity < wp.int64(0):
                                nearest = t
                                identity = sphere_identity
                                face = wp.int64(-1)
            if identity >= wp.int64(0):
                distances[i] = nearest
                objects[i] = identity
                points[i] = origin + nearest * direction
                faces[i] = face


@dataclass(frozen=True)
class RayHits:
    """Owned CUDA results plus the immutable scene metadata for this query.

    A miss has distance/point zero and object/triangle -1. Triangle indices
    address the combined static mesh; analytic sphere hits have triangle -1.
    Object indices address ``object_ids``. Storage survives later calls, scene
    updates and renderer closure. No gradient through closest-hit selection is
    promised. An origin inside a solid sphere returns its forward exit surface.
    """

    distance_m: torch.Tensor
    object_index: torch.Tensor
    hit_point_world_m: torch.Tensor
    triangle_index: torch.Tensor
    object_ids: tuple[str, ...]
    scene_version: int
    t_ref: float | None
    map_bounds_m: tuple[tuple[float, float], ...] | None


class RayRenderer:
    """Thread-confined Warp 1.15 mesh BVH with analytic moving spheres.

    Calls run on PyTorch's current CUDA stream. The caller provides the normal
    stream dependency for externally produced input tensors; this class records
    input lifetimes and orders its own geometry updates across streams. A query
    is one GPU launch, never one Python call per ray. Static mesh selection is
    deliberately unsupported; target-only queries of analytic spheres are
    supported without tracing the background.
    """

    def __init__(self, snapshot: GeometrySnapshot, device="cuda"):
        if not isinstance(snapshot, GeometrySnapshot):
            raise TypeError("snapshot must be a GeometrySnapshot")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("RayRenderer requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._owner_thread = threading.get_ident()
        self._closed = False
        self._snapshot = snapshot
        self.object_ids = tuple(obj.object_id for obj in snapshot.objects)
        self._scene_version = 0
        self._geometry_uploads = 1
        self._transform_updates = 0
        self._transform_uploads = 1 if any(isinstance(o.geometry, Sphere)
                                          for o in snapshot.objects) else 0
        self._transform_upload_bytes = 0
        self._trace_calls = 0
        self._ray_count = 0
        self._last_output_bytes = 0
        self._last_event = None
        vertices, triangles, face_objects = [], [], []
        centers, radii, sphere_objects = [], [], []
        self._object_triangle_ranges = {}
        for index, obj in enumerate(snapshot.objects):
            geometry = obj.geometry
            if isinstance(geometry, MeshObject):
                first = len(triangles)
                offset = len(vertices)
                vertices.extend(geometry.vertices_m)
                triangles.extend(tuple(v + offset for v in face) for face in geometry.triangles)
                face_objects.extend([index] * len(geometry.triangles))
                self._object_triangle_ranges[obj.object_id] = [first, len(triangles) - first]
            elif isinstance(geometry, Sphere):
                centers.append(geometry.pose.position_m)
                radii.append(geometry.radius_m)
                sphere_objects.append(index)
        self.vertex_count, self.triangle_count = len(vertices), len(triangles)
        self._sphere_count = len(centers)
        # CPU world facts may contain values representable as float64 only.
        # Reject them at scene creation rather than passing invalid BVH input.
        for values in (vertices, centers, radii):
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.asarray(values, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError("scene geometry must be representable in float32")
        if radii and np.any(np.asarray(radii, dtype=np.float32) <= 0):
            raise ValueError("sphere radii must remain positive in float32")
        if triangles:
            converted = np.asarray(vertices, dtype=np.float32).astype(np.float64)[np.asarray(triangles)]
            cross = np.cross(converted[:, 1] - converted[:, 0],
                             converted[:, 2] - converted[:, 0])
            if np.any(np.all(cross == 0, axis=1)):
                raise ValueError("mesh triangles must remain nondegenerate in float32")
        self._centers_host = tuple(centers)
        wp.init()
        with torch.cuda.device(device), self._stream():
            self._vertices = torch.tensor(vertices, dtype=torch.float32, device=device).reshape(-1, 3)
            self._triangles = torch.tensor(triangles, dtype=torch.int32, device=device).reshape(-1, 3)
            self._face_objects = torch.tensor(face_objects, dtype=torch.int64, device=device)
            self._centers = torch.tensor(centers, dtype=torch.float32, device=device).reshape(-1, 3)
            self._radii = torch.tensor(radii, dtype=torch.float32, device=device)
            self._sphere_objects = torch.tensor(sphere_objects, dtype=torch.int64, device=device)
            self._mesh = None
            if triangles:
                self._mesh = wp.Mesh(points=wp.from_torch(self._vertices, dtype=wp.vec3),
                                     indices=wp.from_torch(self._triangles.reshape(-1)))
            self._record_event()
        self._transform_upload_bytes = len(centers) * 12
        self._mesh_bytes = sum(t.numel() * t.element_size() for t in (
            self._vertices, self._triangles, self._face_objects))
        self._center_bytes = self._centers.numel() * self._centers.element_size()
        self._geometry_bytes = self._mesh_bytes + sum(t.numel() * t.element_size() for t in (
            self._centers, self._radii, self._sphere_objects))

    def _stream(self):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def _record_event(self):
        self._last_event = torch.cuda.Event()
        self._last_event.record(torch.cuda.current_stream(self.device))

    def _order_scene(self):
        if self._last_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._last_event)

    def _require_open(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("RayRenderer must be used and closed on its creating thread")
        if self._closed:
            raise RuntimeError("RayRenderer is closed")

    @property
    def scene_version(self):
        return self._scene_version

    @property
    def geometry_info(self):
        return {
            "backend": "warp_mesh_bvh_analytic_spheres", "warp_version": wp.__version__,
            "device": str(self.device), "object_ids": list(self.object_ids),
            "vertex_count": self.vertex_count, "triangle_count": self.triangle_count,
            "analytic_sphere_count": self._sphere_count,
            "mesh_bvh_builds": 1 if self.triangle_count else 0,
            "geometry_uploads": self._geometry_uploads,
            "object_transform_uploads": self._transform_uploads,
            "object_transform_updates": self._transform_updates,
            "object_transform_upload_bytes": self._transform_upload_bytes,
            "scene_version": self._scene_version,
            "object_triangle_ranges": {k: list(v) for k, v in self._object_triangle_ranges.items()},
            "batch_geometry_policy": "one static mesh BVH and shared analytic spheres; independent of ray count",
            "trace_calls": self._trace_calls, "rays_traced": self._ray_count,
            "cpu_image_readbacks": 0, "per_ray_python_calls": 0,
            "input": "world origins/directions float32 CUDA N3; radial max distance scalar or CUDA N",
            "output": "owned CUDA radial distance, object/triangle index and world hit point",
            "near_clip_policy": "none; nearest nonnegative surface before measurement validity",
            "selection_policy": "full scene or one analytic sphere; static mesh selection unsupported",
        }

    @property
    def memory_info(self):
        geometry = self._geometry_bytes if not self._closed else 0
        return {
            "scene_geometry_buffer_bytes": geometry,
            "scene_mesh_buffer_bytes": self._mesh_bytes if not self._closed else 0,
            "object_transform_buffer_bytes": self._center_bytes if not self._closed else 0,
            "tracked_total_bytes": geometry,
            "tracked_total_scope": "owned scene tensor storage; excludes Warp BVH/allocator/driver and query outputs",
            "warp_bvh_bytes": None,
            "last_output_bytes": self._last_output_bytes,
            "image_cpu_transfer_bytes_per_trace": 0,
        }

    def sync(self, snapshot: GeometrySnapshot) -> int:
        """Update sphere positions/metadata only; reject other geometry changes.

        Static geometry is compared on the CPU using the public immutable
        snapshot. Center uploads touch only changed spheres; there is no BVH
        rebuild or geometry copy when rigid spheres move.
        """
        self._require_open()
        if not isinstance(snapshot, GeometrySnapshot):
            raise TypeError("snapshot must be a GeometrySnapshot")
        if len(snapshot.objects) != len(self._snapshot.objects):
            raise ValueError("geometry structure changed; rebuild RayRenderer")
        changed = (snapshot.t_ref != self._snapshot.t_ref
                   or snapshot.map_bounds_m != self._snapshot.map_bounds_m)
        centers = []
        for old, new in zip(self._snapshot.objects, snapshot.objects):
            if (old.object_id != new.object_id or old.role != new.role
                    or type(old.geometry) is not type(new.geometry)):
                raise ValueError("object identity/type/order changed; rebuild RayRenderer")
            if isinstance(old.geometry, MeshObject):
                if old.geometry is not new.geometry and old.geometry != new.geometry:
                    raise ValueError("mesh geometry changed; rebuild RayRenderer")
            else:
                if old.geometry.radius_m != new.geometry.radius_m:
                    raise ValueError("sphere radius changed; rebuild RayRenderer")
                centers.append(new.geometry.pose.position_m)
                changed |= old.geometry.pose != new.geometry.pose
        updates = [index for index, (old, new) in enumerate(zip(self._centers_host, centers))
                   if old != new]
        with np.errstate(over="ignore", invalid="ignore"):
            center_values = np.asarray(centers, dtype=np.float32)
        if not np.isfinite(center_values).all():
            raise ValueError("sphere positions must be representable in float32")
        if updates:
            with torch.cuda.device(self.device):
                self._order_scene()
                current = torch.cuda.current_stream(self.device)
                for index in updates:
                    source = torch.tensor(centers[index], device=self.device, dtype=torch.float32)
                    self._centers[index].copy_(source)
                    source.record_stream(current)
                self._record_event()
            self._transform_updates += len(updates)
            self._transform_uploads += len(updates)
            self._transform_upload_bytes += len(updates) * 12
            self._centers_host = tuple(centers)
        if changed:
            self._scene_version += 1
        self._snapshot = snapshot
        return self._scene_version

    def trace(self, origins: torch.Tensor, directions: torch.Tensor,
              max_distance: float | torch.Tensor, *, only_object: str | None = None) -> RayHits:
        """Query a batch without host reads or per-ray Python dispatch.

        Inputs must be contiguous float32 CUDA tensors. Direction normalization
        happens inside the kernel, including after an external perturbation.
        Invalid per-ray origins, directions or limits produce a miss. Positive
        range boundaries use a float32 guard of max(1e-6 m, 1e-6 * range).
        """
        self._require_open()
        for name, value in (("origins", origins), ("directions", directions)):
            if (not isinstance(value, torch.Tensor) or value.ndim != 2
                    or value.shape[1] != 3 or value.dtype != torch.float32
                    or value.device != self.device or not value.is_contiguous()):
                raise ValueError(f"{name} must be contiguous float32 CUDA [N,3] on {self.device}")
        if origins.shape != directions.shape:
            raise ValueError("origins and directions must have equal shapes")
        count = len(origins)
        selected = -1
        if only_object is not None:
            if only_object not in self.object_ids:
                raise ValueError(f"unknown object ID {only_object!r}")
            selected = self.object_ids.index(only_object)
            if isinstance(self._snapshot.objects[selected].geometry, MeshObject):
                raise NotImplementedError("selected static mesh queries require a separate filtered RayRenderer")
        per_ray_limit = isinstance(max_distance, torch.Tensor)
        if per_ray_limit:
            if (max_distance.shape != (count,) or max_distance.dtype != torch.float32
                    or max_distance.device != self.device or not max_distance.is_contiguous()):
                raise ValueError("max_distance tensor must be contiguous float32 CUDA [N] on the renderer device")
            limit_value = 0.0
        else:
            if isinstance(max_distance, bool) or not isinstance(max_distance, (float, int)):
                raise ValueError("max_distance must be a positive finite scalar or float32 CUDA [N]")
            limit_value = float(max_distance)
            if (not math.isfinite(limit_value)
                    or not 0 < limit_value < np.finfo(np.float32).max / 2
                    or np.float32(limit_value) <= 0):
                raise ValueError("max_distance must be positive and safely representable in float32")
        with torch.cuda.device(self.device):
            self._order_scene()
            current = torch.cuda.current_stream(self.device)
            distances = torch.empty(count, dtype=torch.float32, device=self.device)
            objects = torch.empty(count, dtype=torch.int64, device=self.device)
            points = torch.empty((count, 3), dtype=torch.float32, device=self.device)
            faces = torch.empty(count, dtype=torch.int64, device=self.device)
            limits = max_distance if per_ray_limit else self._radii
            if count:
                with self._stream():
                    wp.launch(_trace_rays, dim=count, inputs=[
                        self._mesh.id if self._mesh is not None else 0, int(self._mesh is not None),
                        wp.from_torch(self._face_objects),
                        wp.from_torch(self._centers, dtype=wp.vec3), wp.from_torch(self._radii),
                        wp.from_torch(self._sphere_objects), self._sphere_count,
                        wp.from_torch(origins.detach(), dtype=wp.vec3),
                        wp.from_torch(directions.detach(), dtype=wp.vec3), wp.from_torch(limits.detach()),
                        limit_value, int(per_ray_limit), selected,
                        wp.from_torch(distances), wp.from_torch(objects),
                        wp.from_torch(points, dtype=wp.vec3), wp.from_torch(faces)], device=str(self.device))
                for tensor in (origins, directions, limits, distances, objects, points, faces,
                               self._vertices, self._triangles, self._face_objects,
                               self._centers, self._radii, self._sphere_objects):
                    tensor.record_stream(current)
                self._record_event()
        self._trace_calls += 1
        self._ray_count += count
        self._last_output_bytes = count * (4 + 8 + 12 + 8)
        return RayHits(distances, objects, points, faces, self.object_ids,
                       self._scene_version, self._snapshot.t_ref, self._snapshot.map_bounds_m)

    def close(self):
        """Wait for this renderer's queued work before releasing its native BVH."""
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("RayRenderer must be used and closed on its creating thread")
        if self._closed:
            return
        if self._last_event is not None:
            self._last_event.synchronize()
        self._mesh = None
        self._vertices = self._triangles = self._face_objects = None
        self._centers = self._radii = self._sphere_objects = None
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()
