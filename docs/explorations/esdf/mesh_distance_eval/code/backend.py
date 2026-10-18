"""Two experimental position-differentiable routes over public scene geometry.

The signed mesh magnitude is distance to *all original faces*. It is NOT an
exact union SDF inside overlapping solids. Ground is a finite unsigned sheet.
Only point derivatives are supported; geometry is fixed within each forward.
"""
from contextlib import nullcontext
import math

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from devtools.mesh import MeshObject
from devtools.scene import GeometrySnapshot, Sphere


@wp.kernel
def _select_winding(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                    faces: wp.array(dtype=int), signs: wp.array(dtype=float),
                    distances: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, points[i], 1.0e10)
    faces[i] = -1
    signs[i] = q.sign
    d = float(wp.nan)
    if q.result:
        faces[i] = q.face
        closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = q.sign * wp.length(points[i] - closest)
    distances[i] = d


@wp.kernel
def _select_normal(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
                   faces: wp.array(dtype=int), signs: wp.array(dtype=float),
                   distances: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, points[i], 1.0e10)
    faces[i] = -1
    signs[i] = q.sign
    d = float(wp.nan)
    if q.result:
        faces[i] = q.face
        closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = q.sign * wp.length(points[i] - closest)
    distances[i] = d


def _stream(points):
    if points.is_cuda:
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(points.device)))
    return nullcontext()


def triangle_distance(points, triangles):
    """Recompute the selected triangle's exact distance using original tensors.

    A valid plane projection and three clamped segments cover face, edge and
    vertex features. Discrete global face selection is outside the Torch graph.
    """
    output_dtype = points.dtype
    points, triangles = points.double(), triangles.double()
    a, b, c = triangles.unbind(1)
    ab, ac, ap = b - a, c - a, points - a
    normal = torch.linalg.cross(ab, ac)
    normal2 = normal.square().sum(-1)
    v = (torch.linalg.cross(ap, ac) * normal).sum(-1) / normal2
    w = (torch.linalg.cross(ab, ap) * normal).sum(-1) / normal2
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    candidates = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = (((points - start) * edge).sum(-1) / edge.square().sum(-1)).clamp(0, 1)
        candidates.append(torch.linalg.vector_norm(points - start - t[:, None] * edge, dim=-1))
    plane = (ap * normal).sum(-1).abs() / normal2.sqrt()
    candidates.append(torch.where(inside, plane, torch.full_like(plane, float('inf'))))
    return torch.stack(candidates, dim=-1).min(-1).values.to(output_dtype)


class _TapeDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, owner, sign):
        record = ctx.needs_input_grad[0]
        p = wp.from_torch(points.detach(), dtype=wp.vec3, requires_grad=record)
        faces = wp.empty(len(points), dtype=int, device=owner.device)
        signs = wp.empty(len(points), dtype=float, device=owner.device)
        distances = wp.empty(len(points), dtype=float, device=owner.device, requires_grad=record)
        tape = wp.Tape()
        with _stream(points), (tape if record else nullcontext()):
            wp.launch(owner.kernels[sign], dim=len(points),
                      inputs=[owner.mesh.id, p, faces, signs, distances], device=owner.device)
        ctx.tape, ctx.p, ctx.distances = tape, p, distances
        ctx.owner, ctx.points = owner, points  # retain mesh and shared tensor lifetimes
        return wp.to_torch(distances)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        with _stream(ctx.points):
            ctx.tape.zero()
            ctx.tape.backward(grads={ctx.distances: wp.from_torch(grad_output.contiguous())})
        return wp.to_torch(ctx.p.grad).clone(), None, None


class MeshDistance:
    """Experimental static combined BVH + analytic target and ground.

    Input is a filtered GeometrySnapshot, never scene/background internals.
    NaN/Inf and out-of-domain points return NaN with zero position derivative.
    A missing declared domain remains unknown and must be supplied explicitly.
    """
    kernels = {'winding': _select_winding, 'normal': _select_normal}

    def __init__(self, snapshot: GeometrySnapshot, device='cuda', *, bounds=None):
        wp.init()
        self.device = str(device)
        self.bounds = snapshot.map_bounds_m if bounds is None else bounds
        if self.bounds is None:
            raise ValueError('unknown map domain: pass explicit experiment bounds')
        limits = np.asarray(self.bounds, dtype=float)
        if limits.shape != (3, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('bounds must be three increasing finite intervals')
        if any(o.role == 'vehicle' for o in snapshot.objects):
            raise ValueError('exclude vehicle through the public scene selection')
        obstacles = [o for o in snapshot.objects if o.role == 'obstacle']
        self.static_signature = tuple((o.object_id, o.geometry) for o in snapshot.objects
                                      if isinstance(o.geometry, MeshObject))
        vertices, indices = [], []
        for obj in obstacles:
            geometry = obj.geometry
            offset = len(vertices)
            vertices.extend(geometry.vertices_m)
            indices.extend(tuple(i + offset for i in face) for face in geometry.triangles)
        self.min_triangle_quality = None
        if indices:
            # Native float32 nearest-triangle selection is unreliable on very
            # thin faces. Reject the reproduced failure class rather than
            # claiming Torch recomputation can repair a wrong selected face.
            tri = np.asarray(vertices, dtype=np.float32).astype(np.float64)[np.asarray(indices)]
            edges = [tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]]
            cross = np.linalg.norm(np.cross(edges[0], -edges[2]), axis=-1)
            longest2 = np.max([np.sum(e * e, axis=-1) for e in edges], axis=0)
            quality = cross / longest2
            self.min_triangle_quality = float(quality.min())
            if not np.isfinite(quality).all() or self.min_triangle_quality < 1e-3:
                raise ValueError('unsupported thin/float32-degenerate mesh triangles (quality < 1e-3)')
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device).reshape(-1, 3)
        self.indices = torch.tensor(indices, dtype=torch.int32, device=device).reshape(-1, 3)
        self.mesh = None
        if len(indices):
            self.mesh = wp.Mesh(points=wp.from_torch(self.vertices, dtype=wp.vec3),
                                indices=wp.from_torch(self.indices.flatten()),
                                support_winding_number=True)
        self.limits = torch.tensor(limits, dtype=torch.float32, device=device)
        grounds = [o.geometry for o in snapshot.objects if o.role == 'ground']
        if len(grounds) != 1:
            raise ValueError('one finite ground rectangle required')
        ground = torch.tensor(grounds[0].vertices_m, dtype=torch.float32, device=device)
        self.ground_lo, self.ground_hi = ground.amin(0), ground.amax(0)
        self._set_target(snapshot)

    def _set_target(self, snapshot):
        targets = [o.geometry for o in snapshot.objects if isinstance(o.geometry, Sphere)]
        if len(targets) != 1:
            raise ValueError('this experiment requires exactly one target sphere')
        self.target = torch.tensor(targets[0].pose.position_m, dtype=torch.float32, device=self.device)
        self.radius = targets[0].radius_m

    def update(self, snapshot):
        signature = tuple((o.object_id, o.geometry) for o in snapshot.objects
                          if isinstance(o.geometry, MeshObject))
        if any(o.role == 'vehicle' for o in snapshot.objects) or signature != self.static_signature:
            raise ValueError('update only supports target state, with the same static geometry')
        if snapshot.map_bounds_m is not None and snapshot.map_bounds_m != self.bounds:
            raise ValueError('update cannot change the declared domain')
        self._set_target(snapshot)

    def _points(self, points):
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float32:
            raise ValueError('points must be float32 Nx3 tensors')
        if points.device != self.vertices.device:
            raise ValueError('points and backend must use the same device')
        valid = torch.isfinite(points).all(-1)
        valid = valid & (points >= self.limits[:, 0]).all(-1) & (points <= self.limits[:, 1]).all(-1)
        safe = torch.where(valid[:, None], points, self.limits.mean(-1)).contiguous()
        return safe, valid

    def _mesh_query(self, points, sign, method):
        if sign not in self.kernels or method not in ('torch', 'tape'):
            raise ValueError('unknown sign or differentiation route')
        if self.mesh is None:
            return points.sum(-1) * 0 + float('inf')
        if method == 'tape':
            return _TapeDistance.apply(points, self, sign)
        faces = torch.empty(len(points), dtype=torch.int32, device=points.device)
        signs = torch.empty(len(points), dtype=torch.float32, device=points.device)
        distances = torch.empty_like(signs)
        with _stream(points):
            wp.launch(self.kernels[sign], dim=len(points), inputs=[
                self.mesh.id, wp.from_torch(points.detach(), dtype=wp.vec3),
                wp.from_torch(faces), wp.from_torch(signs), wp.from_torch(distances)], device=self.device)
        triangles = self.vertices[self.indices[faces.clamp(min=0).long()].long()]
        d = triangle_distance(points, triangles) * signs
        return torch.where(faces >= 0, d, torch.full_like(d, float('nan')))

    def mesh_query(self, points, sign='winding', method='torch'):
        safe, valid = self._points(points)
        d = self._mesh_query(safe, sign, method)
        return torch.where(valid, d, torch.full_like(d, float('nan')))

    def query(self, points, method='torch'):
        safe, valid = self._points(points)
        mesh = self._mesh_query(safe, 'winding', method)
        sphere = torch.linalg.vector_norm(safe - self.target, dim=-1) - self.radius
        closest = torch.minimum(torch.maximum(safe, self.ground_lo), self.ground_hi)
        ground = torch.linalg.vector_norm(safe - closest, dim=-1)
        d = torch.stack((mesh, sphere, ground), dim=-1).min(-1).values
        return torch.where(valid, d, torch.full_like(d, float('nan')))


class SampledField:
    """GPU samples of the identical query, then Torch trilinear interpolation.

    Values are XYZ; grid_sample sees ZYX. Node convex hull is the valid domain.
    No padding/clamp values are reported as physical obstacles or known space.
    Exact domain-edge queries select zero normal derivative through border mode.
    """
    def __init__(self, backend, origin_xyz, shape_xyz, spacing):
        self.origin = torch.tensor(origin_xyz, dtype=torch.float32, device=backend.device)
        self.shape = tuple(shape_xyz)
        self.spacing = float(spacing)
        if (self.origin.shape != (3,) or not torch.isfinite(self.origin).all()
                or len(self.shape) != 3 or any(type(s) is not int or s < 2 for s in self.shape)
                or not math.isfinite(self.spacing) or self.spacing <= 0):
            raise ValueError('invalid field specification')
        self.high = self.origin + torch.tensor([s - 1 for s in self.shape], device=backend.device) * self.spacing
        if (self.origin < backend.limits[:, 0]).any() or (self.high > backend.limits[:, 1]).any():
            raise ValueError('field support must be inside declared geometry domain')
        self.rebuild(backend)

    def rebuild(self, backend):
        axes = [self.origin[i] + torch.arange(s, device=backend.device) * self.spacing
                for i, s in enumerate(self.shape)]
        with torch.no_grad():
            # Chunking bounds query workspace independently of logical grid size.
            nodes = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
            self.values = torch.cat([backend.query(p, method='tape') for p in nodes.split(131072)]).reshape(self.shape)
            self.volume = self.values.permute(2, 1, 0).contiguous()[None, None]

    def query(self, points):
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype != torch.float32:
            raise ValueError('points must be float32 Nx3')
        valid = torch.isfinite(points).all(-1)
        valid &= (points >= self.origin).all(-1) & (points <= self.high).all(-1)
        safe = torch.where(valid[:, None], points, self.origin)
        normalized = 2 * (safe - self.origin) / (self.high - self.origin) - 1
        sampled = F.grid_sample(self.volume, normalized[None, None, None], mode='bilinear',
                                padding_mode='border', align_corners=True).flatten()
        return torch.where(valid, sampled, torch.full_like(sampled, float('nan')))
