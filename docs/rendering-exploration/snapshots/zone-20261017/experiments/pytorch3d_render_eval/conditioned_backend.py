"""Bounded experimental conditioning of the standard PyTorch3D backend.

This retains upstream mesh projection, clipping, rasterization, and face
selection.  It is a separate candidate, not a claim of exact GL equivalence:
optical depth below 1 mm is unsupported and boundary coverage can differ.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings

from devtools.config import DepthConfig
from devtools.mesh import MeshObject
from devtools.scene import Sphere
from experiments.sensor_reference import _sphere_mesh

from .backend import P3DRenderer as BaseP3DRenderer


Z_CLIP_M = 1.0e-3
BLUR_RADIUS_NDC_SQUARED = 1.0e-8


class P3DRenderer(BaseP3DRenderer):
    """Standard P3D selection plus float64 original-face plane reconstruction.

    The constructor/prepare/render contract matches backend.P3DRenderer,
    including prepare(B, include_target=True).  No library code is modified.
    Planes are computed once from the same public snapshot, using the same
    sphere tessellation; all plane arithmetic and render outputs stay on CUDA.
    The additional float64 plane/ray workspace is part of this candidate's cost.
    """

    def __init__(self, scene, device="cuda", *, max_faces_per_bin=None):
        super().__init__(scene, device, max_faces_per_bin=max_faces_per_bin)
        normals, points = [], []
        target_range = None
        face_start = 0
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        if tuple(obj.object_id for obj in snapshot.objects) != self.object_ids:
            raise ValueError("scene geometry changed during renderer construction")
        for obj in snapshot.objects:
            if isinstance(obj.geometry, MeshObject):
                vertices = np.asarray(obj.geometry.vertices_m, dtype=np.float64)
                faces = np.asarray(obj.geometry.triangles, dtype=np.int64)
            elif isinstance(obj.geometry, Sphere):
                vertices, faces = _sphere_mesh(obj.geometry)
                vertices = np.asarray(vertices, dtype=np.float64)
                faces = np.asarray(faces, dtype=np.int64)
            else:
                raise TypeError("unsupported public geometry")
            verts_cuda = torch.as_tensor(vertices, device=self.device, dtype=torch.float64)
            faces_cuda = torch.as_tensor(faces, device=self.device, dtype=torch.int64)
            triangle = verts_cuda[faces_cuda]
            normals.append(torch.cross(triangle[:, 1] - triangle[:, 0],
                                       triangle[:, 2] - triangle[:, 0], dim=-1))
            points.append(triangle[:, 0])
            if obj.object_id == "target":
                target_range = (face_start, face_start + len(faces))
            face_start += len(faces)
        self._plane_normals = torch.cat(normals, dim=0)
        self._plane_points = torch.cat(points, dim=0)
        if target_range is None or face_start != self.triangle_count:
            raise ValueError("plane geometry must match the standard renderer's faces")
        start, end = target_range
        self._target_plane_normals = self._plane_normals[start:end]
        self._target_plane_points = self._plane_points[start:end]

    @property
    def geometry_info(self) -> dict:
        info = super().geometry_info
        info.update({
            "backend": "pytorch3d_conditioned_experimental_adapter",
            "experimental_adapter": True,
            "z_clip_m": Z_CLIP_M,
            "z_clip_policy": "fixed 1 mm optical near clip, independent of min_range",
            "blur_radius_ndc_squared": BLUR_RADIUS_NDC_SQUARED,
            "coverage_filter": "selected fragment dists <= 0; reject positive-distance candidates",
            "clip_barycentric_coords": True,
            "selection_precision": "upstream float32 rasterization and z-buffer",
            "distance_precision": "CUDA float64 original-face plane/ray intersection, cast to float32",
            "plane_storage_bytes": self.triangle_count * 6 * 8,
            "target_planes_share_full_storage": True,
            "limitations": [
                "geometry portions at optical depth below 1 mm are unsupported",
                "float64 reconstruction cannot repair an incorrectly selected or missing face",
                "tiny blur and closed-edge filtering do not implement the GL top-left rule",
                "a positive-distance K=1 candidate can hide an interior candidate before filtering",
                "rasterization remains float32 and uses quantized float32 input poses",
                "upstream clipping scalar synchronization is retained",
                "additional float64 plane/ray workspace is included in measured memory and time",
            ],
        })
        return info

    def _view(self, config: DepthConfig, target_only: bool):
        key = (config.width, config.height, config.hfov_deg, config.vfov_deg, target_only)
        if key not in self._view_cache:
            width, height = config.width, config.height
            tan_h = math.tan(math.radians(config.hfov_deg) / 2)
            tan_v = math.tan(math.radians(config.vfov_deg) / 2)
            scale = min(width, height)
            focal = torch.tensor([[width / scale / tan_h, height / scale / tan_v]],
                                 device=self.device, dtype=torch.float32)
            principal = torch.zeros((1, 2), device=self.device, dtype=torch.float32)
            col = 2 * (torch.arange(width, device=self.device, dtype=torch.float64) + .5) / width - 1
            row = 1 - 2 * (torch.arange(height, device=self.device, dtype=torch.float64) + .5) / height
            y = (-col * tan_h)[None, :].expand(height, -1)
            z = (row * tan_v)[:, None].expand(-1, width)
            rays = torch.stack((torch.ones_like(y), y, z), dim=-1)
            rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
            count = self.target_triangle_count if target_only else self.triangle_count
            settings = RasterizationSettings(
                image_size=(height, width), blur_radius=BLUR_RADIUS_NDC_SQUARED,
                faces_per_pixel=1, perspective_correct=True,
                clip_barycentric_coords=True, cull_backfaces=False,
                z_clip_value=Z_CLIP_M, cull_to_frustum=False,
                bin_size=None, max_faces_per_bin=self._capacity(count),
            )
            self._view_cache[key] = (focal, principal, rays,
                                     MeshRasterizer(raster_settings=settings))
        return self._view_cache[key]

    @torch.no_grad()
    def render(self, poses_cuda: torch.Tensor, config: DepthConfig,
               target_only: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(config, DepthConfig):
            raise TypeError("config must be a DepthConfig")
        if not isinstance(target_only, bool):
            raise TypeError("target_only must be bool")
        if (not isinstance(poses_cuda, torch.Tensor) or poses_cuda.device != self.device
                or poses_cuda.dtype != torch.float32 or poses_cuda.ndim != 3
                or poses_cuda.shape[1:] != (4, 4)):
            raise ValueError("poses must be float32 [B,4,4] on the renderer CUDA device")
        if poses_cuda.shape[0] != self._batch_size:
            raise ValueError("call prepare(B) for this exact pose batch size before render")
        if target_only and self._batch_target is None:
            raise ValueError("call prepare(B, include_target=True) before target-only rendering")
        focal, principal, rays, rasterizer = self._view(config, target_only)
        origin = poses_cuda[:, :3, 3]
        rotation = poses_cuda[:, :3, :3]
        rotation_p3d = torch.stack((rotation[:, :, 1], rotation[:, :, 2],
                                    rotation[:, :, 0]), dim=-1)
        translation = -torch.bmm(origin[:, None, :], rotation_p3d).squeeze(1)
        cameras = PerspectiveCameras(device=self.device, R=rotation_p3d, T=translation,
                                     focal_length=focal, principal_point=principal,
                                     in_ndc=True)
        mesh = self._batch_target if target_only else self._batch_full
        fragments = rasterizer(mesh, cameras=cameras, eps=Z_CLIP_M)
        face = fragments.pix_to_face[..., 0]
        face_ids = self._target_face_ids if target_only else self._full_face_ids
        local_face = face.clamp_min(0).remainder(face_ids.shape[0])
        plane_normals = self._target_plane_normals if target_only else self._plane_normals
        plane_points = self._target_plane_points if target_only else self._plane_points
        normal = plane_normals[local_face]
        point = plane_points[local_face]

        # The pose convention is camera-FLU -> world.  Intersect the actual
        # pixel-center world ray with the original face, not the clipped bary.
        rays_world = torch.einsum("bij,hwj->bhwi", rotation.to(torch.float64), rays)
        rays_world = rays_world / torch.linalg.vector_norm(rays_world, dim=-1, keepdim=True)
        numerator = (normal * (point - origin.to(torch.float64)[:, None, None, :])).sum(dim=-1)
        denominator = (normal * rays_world).sum(dim=-1)
        nonparallel = denominator != 0
        safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
        distance64 = numerator / safe_denominator
        distance = distance64.to(torch.float32)

        scale = origin.abs().amax(dim=1).clamp_min(max(1.0, config.max_range_m))
        roundoff = 32 * torch.finfo(torch.float32).eps * scale
        shell = config.max_range_m + torch.maximum(
            4 * roundoff, roundoff.new_full(roundoff.shape, .001 * config.max_range_m))
        hit = ((face >= 0) & (fragments.dists[..., 0] <= 0) & nonparallel
               & torch.isfinite(distance) & (distance >= 0)
               & (distance <= shell[:, None, None]))
        identity = torch.where(hit, face_ids[local_face], torch.full_like(face, -1))
        distance = torch.where(hit, distance, torch.zeros_like(distance))
        return distance, identity
