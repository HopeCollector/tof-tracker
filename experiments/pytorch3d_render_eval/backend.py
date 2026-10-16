"""Isolated PyTorch3D rasterization experiment, not a sensor replacement.

This uses the standard MeshRasterizer/PerspectiveCameras path from PyTorch3D
88e182f989c80836f4bd744e0d9cb1852762ce01.  Geometry is copied by Meshes.extend;
neither its batch memory nor its near-clipping scalar synchronizations are
hidden.  Run and measure this module only in the experiment's Docker image.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

from devtools.config import DepthConfig
from devtools.mesh import MeshObject
from devtools.scene import Sphere
from devtools.sensors import _sphere_mesh


PYTORCH3D_SHA = "88e182f989c80836f4bd744e0d9cb1852762ce01"
Z_CLIP_M = 1.0e-5


def _known_mesh_storage(mesh: Meshes) -> int:
    """Count existing list tensor storages without forcing lazy render caches.

    The byte count excludes small internal bookkeeping tensors, rasterization
    scratch, camera transforms, fragments, outputs, and allocator reservation.
    It is a known-storage count, not a prediction of peak CUDA allocation.
    """
    tensors = [*mesh.verts_list(), *mesh.faces_list()]
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(storages.values())


class P3DRenderer:
    """One immutable public scene snapshot, rendered for a prepared pose batch.

    Poses are float32 CUDA ``[B,4,4]`` camera-FLU-to-world column-vector rigid
    transforms.  Call ``prepare(B)`` before ``render`` to keep mesh preparation
    separate from steady-state measurements.  Inputs must already contain
    finite rigid transforms; validation does not read GPU values onto the CPU.

    Outputs are CUDA ``distance[B,H,W]`` float32 and ``identity[B,H,W]`` int64.
    Misses are 0/-1; nonnegative IDs index ``object_ids`` even in target-only
    mode.  Distances retain too-close occluders and the current sensor's guard
    shell.  Minimum-range and map-coverage semantics belong to the caller.

    ``max_faces_per_bin=None`` reserves a conservative 2F faces per bin, since
    near clipping can split one triangle into two.  A positive integer permits
    capacity experiments; ``'pytorch3d_default'`` selects the library heuristic.
    Smaller capacities can overflow: runner logs must never discard warnings.
    """

    def __init__(self, scene, device="cuda", *, max_faces_per_bin=None):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("this experiment requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if max_faces_per_bin is not None and max_faces_per_bin != "pytorch3d_default":
            if (isinstance(max_faces_per_bin, bool)
                    or not isinstance(max_faces_per_bin, int)
                    or max_faces_per_bin <= 0):
                raise ValueError("max_faces_per_bin must be None, a positive integer, "
                                 "or 'pytorch3d_default'")
        self.device = device
        self._bin_capacity = max_faces_per_bin
        snapshot = scene.get_geometry(exclude_ids={"vehicle"})
        self.object_ids = tuple(obj.object_id for obj in snapshot.objects)
        vertices, faces, face_ids = [], [], []
        target_vertices = target_faces = None
        offset = 0
        target_index = None
        for identity, obj in enumerate(snapshot.objects):
            if isinstance(obj.geometry, MeshObject):
                verts = np.asarray(obj.geometry.vertices_m, dtype=np.float32)
                tris = np.asarray(obj.geometry.triangles, dtype=np.int64)
            elif isinstance(obj.geometry, Sphere):
                verts, tris = _sphere_mesh(obj.geometry)
                verts = np.asarray(verts, dtype=np.float32)
                tris = np.asarray(tris, dtype=np.int64)
            else:
                raise TypeError(f"unsupported public geometry: {type(obj.geometry).__name__}")
            vertices.append(verts)
            faces.append(tris + offset)
            face_ids.append(np.full(len(tris), identity, dtype=np.int64))
            offset += len(verts)
            if obj.object_id == "target":
                target_vertices, target_faces, target_index = verts, tris, identity
        if not vertices or target_vertices is None:
            raise ValueError("scene must include nonempty geometry and object ID 'target'")

        verts = torch.as_tensor(np.concatenate(vertices), device=device, dtype=torch.float32)
        tris = torch.as_tensor(np.concatenate(faces), device=device, dtype=torch.int64)
        target_verts = torch.as_tensor(target_vertices, device=device, dtype=torch.float32)
        target_tris = torch.as_tensor(target_faces, device=device, dtype=torch.int64)
        self._base_full = Meshes(verts=[verts], faces=[tris])
        self._base_target = Meshes(verts=[target_verts], faces=[target_tris])
        self._full_face_ids = torch.as_tensor(np.concatenate(face_ids), device=device,
                                              dtype=torch.int64)
        self._target_face_ids = torch.full((len(target_tris),), target_index,
                                           device=device, dtype=torch.int64)
        self.vertex_count, self.triangle_count = len(verts), len(tris)
        self.target_vertex_count = len(target_verts)
        self.target_triangle_count = len(target_tris)
        self._raw_mesh_bytes = ((self.vertex_count + self.target_vertex_count) * 3 * 4
                                + (self.triangle_count + self.target_triangle_count) * 3 * 8)
        self._face_identity_bytes = (self.triangle_count + self.target_triangle_count) * 8
        self._batch_size = None
        self._batch_full = self._batch_target = None
        self._view_cache = {}
        self._prepare_info = {"batch_size": None}

    def _capacity(self, triangle_count: int) -> int:
        if self._bin_capacity is None:
            return 2 * triangle_count
        if self._bin_capacity == "pytorch3d_default":
            return int(max(10000, triangle_count / 5))
        return self._bin_capacity

    @property
    def geometry_info(self) -> dict:
        return {
            "backend": "pytorch3d_standard_mesh_rasterizer",
            "pytorch3d_sha": PYTORCH3D_SHA,
            "device": str(self.device),
            "object_ids": list(self.object_ids),
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "target_vertex_count": self.target_vertex_count,
            "target_triangle_count": self.target_triangle_count,
            "base_raw_mesh_arrays_bytes": self._raw_mesh_bytes,
            "base_face_identity_bytes": self._face_identity_bytes,
            "batch_geometry_policy": "Meshes.extend clones full and target geometry",
            "z_clip_m": Z_CLIP_M,
            "z_clip_policy": "fixed tiny positive optical depth, independent of min_range",
            "backend_internal_scalar_sync": True,
            "scalar_sync_reason": "upstream clip.py uses item() for clipping/control flow",
            "max_faces_per_bin_policy": (
                "safe_2F" if self._bin_capacity is None else self._bin_capacity),
            "full_max_faces_per_bin": self._capacity(self.triangle_count),
            "target_max_faces_per_bin": self._capacity(self.target_triangle_count),
            "faces_per_pixel": 1,
            "perspective_correct": True,
            "backface_culling": False,
            "range_policy": "nearest surface then current MeshSensor guard-shell cutoff",
            "min_range_filter": False,
            "map_coverage_filter": False,
            "limitations": [
                "surfaces with optical depth below z_clip_m can be clipped away",
                "camera-plane crossing and vertices exactly on the plane require validation",
                "no geometric input validation by GPU-to-CPU scalar readback",
                "known storage estimates exclude raster scratch and allocator reservation",
            ],
        }

    @property
    def prepare_info(self) -> dict:
        return dict(self._prepare_info)

    @torch.no_grad()
    def prepare(self, batch_size: int, *, include_target=True) -> dict:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size == self._batch_size and (not include_target or self._batch_target is not None):
            return self.prepare_info
        self.release_batch()
        self._batch_full = self._base_full.extend(batch_size)
        self._batch_target = self._base_target.extend(batch_size) if include_target else None
        full_bytes = _known_mesh_storage(self._batch_full)
        target_bytes = _known_mesh_storage(self._batch_target) if include_target else 0
        self._batch_size = batch_size
        self._prepare_info = {
            "batch_size": batch_size,
            "include_target": include_target,
            "batch_raw_mesh_arrays_bytes": full_bytes + target_bytes,
            "full_known_cached_tensor_storage_bytes": full_bytes,
            "target_known_cached_tensor_storage_bytes": target_bytes,
            "known_cached_tensor_storage_bytes": full_bytes + target_bytes,
            "estimate_scope": "existing list tensor storages only; lazy render caches are not forced; "
                              "excludes internal metadata, raster scratch and reserved memory",
            "full_max_faces_per_bin": self._capacity(self.triangle_count),
            "target_max_faces_per_bin": self._capacity(self.target_triangle_count),
        }
        return self.prepare_info

    def release_batch(self) -> None:
        """Drop prepared meshes/camera caches; leave PyTorch's allocator untouched."""
        self._batch_full = self._batch_target = None
        self._batch_size = None
        self._view_cache.clear()
        self._prepare_info = {"batch_size": None}

    def _view(self, config: DepthConfig, target_only: bool):
        key = (config.width, config.height, config.hfov_deg, config.vfov_deg, target_only)
        if key not in self._view_cache:
            width, height = config.width, config.height
            tan_h = math.tan(math.radians(config.hfov_deg) / 2)
            tan_v = math.tan(math.radians(config.vfov_deg) / 2)
            # PyTorch3D NDC uses [-1,1] on the shorter image side, with +X left
            # and +Y up. Scale each focal length by its image side / min(H,W).
            scale = min(width, height)
            focal = torch.tensor([[width / scale / tan_h, height / scale / tan_v]],
                                 device=self.device, dtype=torch.float32)
            principal = torch.zeros((1, 2), device=self.device, dtype=torch.float32)
            col = 2 * (torch.arange(width, device=self.device, dtype=torch.float32) + .5) / width - 1
            row = 1 - 2 * (torch.arange(height, device=self.device, dtype=torch.float32) + .5) / height
            ray_norm = torch.sqrt(1 + (row[:, None] * tan_v).square()
                                  + (col[None, :] * tan_h).square())
            count = self.target_triangle_count if target_only else self.triangle_count
            settings = RasterizationSettings(
                image_size=(height, width), blur_radius=0.0, faces_per_pixel=1,
                perspective_correct=True, clip_barycentric_coords=False,
                cull_backfaces=False, z_clip_value=Z_CLIP_M, cull_to_frustum=False,
                bin_size=None, max_faces_per_bin=self._capacity(count),
            )
            self._view_cache[key] = (focal, principal, ray_norm,
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
        focal, principal, ray_norm, rasterizer = self._view(config, target_only)
        origin = poses_cuda[:, :3, 3]
        rotation = poses_cuda[:, :3, :3]
        # P3D row-vector world->view: X_view = X_world R + T.  Its camera axes
        # (left, up, forward) are FLU (Y,Z,X), including full roll/pitch/yaw.
        rotation_p3d = torch.stack((rotation[:, :, 1], rotation[:, :, 2],
                                    rotation[:, :, 0]), dim=-1)
        translation_p3d = -torch.bmm(origin[:, None, :], rotation_p3d).squeeze(1)
        # NDC intrinsics avoid screen-camera construction's GPU image_size.any()
        # validation. The standard near-clipping path still has upstream item().
        cameras = PerspectiveCameras(device=self.device, R=rotation_p3d,
                                     T=translation_p3d, focal_length=focal,
                                     principal_point=principal, in_ndc=True)
        mesh = self._batch_target if target_only else self._batch_full
        fragments = rasterizer(mesh, cameras=cameras, eps=Z_CLIP_M)
        face = fragments.pix_to_face[..., 0]
        # MeshRasterizer deliberately retains view-space Z in zbuf.  With
        # perspective-correct interpolation this is FLU optical X in metres.
        distance = fragments.zbuf[..., 0] * ray_norm[None, :, :]
        scale = origin.abs().amax(dim=1).clamp_min(max(1.0, config.max_range_m))
        roundoff = 32 * torch.finfo(torch.float32).eps * scale
        shell = config.max_range_m + torch.maximum(
            4 * roundoff, roundoff.new_full(roundoff.shape, .001 * config.max_range_m))
        hit = ((face >= 0) & torch.isfinite(distance) & (distance >= 0)
               & (distance <= shell[:, None, None]))
        face_ids = self._target_face_ids if target_only else self._full_face_ids
        local_face = face.clamp_min(0).remainder(face_ids.shape[0])
        identity = torch.where(hit, face_ids[local_face], torch.full_like(face, -1))
        distance = torch.where(hit, distance, torch.zeros_like(distance))
        return distance, identity
