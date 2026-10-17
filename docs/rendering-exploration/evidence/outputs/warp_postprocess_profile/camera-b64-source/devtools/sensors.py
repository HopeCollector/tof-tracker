"""Batched GPU sensor policies over a shared, sensor-agnostic ray core.

This layer generates rays from complete sensor-to-world FLU poses, then applies
ToF or camera reductions. Ray directions can be changed before tracing and raw
returns can be changed before process_samples: the core has no noise hooks.
Single views use B=1; no sampling function downloads tensors to the CPU.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

from .config import DepthConfig, MountConfig, SensorConfig, _finite_number
from .ray_rendering import RayRenderer, RayHits
from .scene import Pose
from ._ray_processing import (camera_measurements, gather_subray, range_by_sample,
                              select_subray)


TOF_ORDER = ("right", "front", "left", "rear")


@dataclass(frozen=True)
class SampleBatch:
    """Same-pass raw geometric samples and the selected representative ray.

    Representative fields have shape [B,H,W]; subray fields have [B,H,W,S].
    Vec3 fields append a final dimension of three. Raw misses remain zero/-1.
    The subray axis is subrow-major within each pixel, then pixels are ordered
    by row and column. All tensors share the tracing CUDA device.
    """

    distance_m: torch.Tensor
    object_index: torch.Tensor
    object_ids: tuple[str, ...]
    scene_version: int
    t_ref: float | None
    map_bounds_m: tuple | None
    hit_point_sensor_m: torch.Tensor
    triangle_index: torch.Tensor
    subray_distance_m: torch.Tensor
    subray_object_index: torch.Tensor
    subray_hit_point_world_m: torch.Tensor
    subray_triangle_index: torch.Tensor
    ray_origins_world_m: torch.Tensor
    ray_directions_world: torch.Tensor
    selected_subray: torch.Tensor


@dataclass(frozen=True)
class DepthBatch:
    """CUDA radial ranges [B,H,W] and matching representative identities.

    A known selected hit in [min,max] is finite; a known no-return is +inf.
    Any unknown constituent ray makes the cell unavailable (NaN), as does a
    selected too-close hit. This is conservative over the finite samples,
    without claiming continuous angular coverage between them. Identities and
    raw geometry remain development diagnostics, never online model inputs.
    """

    range_m: torch.Tensor
    object_index: torch.Tensor
    object_ids: tuple[str, ...]
    scene_version: int
    t_ref: float | None
    hit_point_sensor_m: torch.Tensor | None = None
    triangle_index: torch.Tensor | None = None
    raw: SampleBatch | None = None
    subray_range_m: torch.Tensor | None = None


@dataclass(frozen=True)
class CameraBatch:
    """CUDA masks, boxes and visibility for the same B views as depth.

    target_mask is bool [B,H,W], target_hint is float32 [B,5] containing
    (detected, left, top, right, bottom), visible_fraction is float64 [B], and
    unoccluded_pixels is int64 [B]. target_coverage is the visible target
    subray fraction per pixel, and unoccluded_samples counts valid target-only
    rays used as the visible_fraction denominator. A representative depth can
    come from another object in a partially covered target pixel.
    """

    depth: DepthBatch
    target_mask: torch.Tensor
    target_hint: torch.Tensor
    visible_fraction: torch.Tensor
    unoccluded_pixels: torch.Tensor
    target_coverage: torch.Tensor | None = None
    unoccluded_samples: torch.Tensor | None = None


@dataclass(frozen=True)
class RigBatch:
    """Synthetic simultaneous observations for N body poses on one snapshot.

    tof_m is CUDA float32 [N,4,8,8] in right/front/left/rear order. target_hint
    is float32 [N,5]. Invalid ToF values still require online-model adaptation.
    Sampling does not simulate asynchronous hardware timing or sensor noise.
    """

    tof_m: torch.Tensor
    target_hint: torch.Tensor
    camera: CameraBatch
    scene_version: int
    t_ref: float | None


@dataclass(frozen=True)
class RigDiagnostics:
    """Same-pass development images alongside the formal rig observation.

    Raw images retain the renderer's zero-for-miss radial distances and -1
    identities, before range and known-map policy. Pose tensors are the exact
    CUDA float32 mounted poses used by those queries. No image is rendered again
    for inspection and no tensor is downloaded by this interface.
    """

    rig: RigBatch
    tof_raw: SampleBatch
    camera_raw: SampleBatch
    target_raw: SampleBatch | None
    tof_poses: torch.Tensor
    camera_poses: torch.Tensor


def pixel_directions(config: DepthConfig):
    """CPU diagnostic FLU unit rays at pixel centers, shape [H,W,3]."""
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    u = 2 * (np.arange(config.width) + 0.5) / config.width - 1
    v = 1 - 2 * (np.arange(config.height) + 0.5) / config.height
    y, z = np.meshgrid(-u * math.tan(math.radians(config.hfov_deg) / 2),
                       v * math.tan(math.radians(config.vfov_deg) / 2))
    rays = np.stack((np.ones_like(y), y, z), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def poses_to_tensor(poses: Sequence[Pose], device="cuda") -> torch.Tensor:
    """Explicitly upload CPU Pose inputs as CUDA float32 [B,4,4].

    This convenience boundary validates CPU values once. GPU callers can
    provide their own already-valid rigid-transform tensor directly.
    """
    if (isinstance(poses, (str, bytes)) or not isinstance(poses, Sequence)
            or len(poses) < 1 or not all(isinstance(pose, Pose) for pose in poses)):
        raise TypeError("poses must be a nonempty sequence of Pose values")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("sensor poses must reside on CUDA")
    matrices = np.stack([pose.as_transform().matrix for pose in poses])
    if np.any(np.abs(matrices) > float(np.finfo(np.float32).max) / 2):
        raise ValueError("poses must be representable by the float32 renderer")
    return torch.as_tensor(matrices, dtype=torch.float32, device=device)


def mounted_pose(body_pose: Pose, mount: MountConfig) -> Pose:
    """CPU helper: complete sensor-to-body mount followed by body-to-world."""
    if not isinstance(body_pose, Pose) or not isinstance(mount, MountConfig):
        raise TypeError("expected Pose and MountConfig")
    position = body_pose.body_to_world(mount.position_m)
    a, b = body_pose.quaternion_xyzw, mount.quaternion_xyzw
    xyz = a[3] * np.array(b[:3]) + b[3] * np.array(a[:3]) + np.cross(a[:3], b[:3])
    w = a[3] * b[3] - np.dot(a[:3], b[:3])
    return Pose(position, tuple(float(value) for value in (*xyz, w)))


def _check_poses(poses: torch.Tensor) -> None:
    if not isinstance(poses, torch.Tensor) or not poses.is_cuda:
        raise TypeError("poses must be a CUDA tensor")
    if poses.dtype != torch.float32:
        raise TypeError("poses must use float32")
    if poses.ndim != 3 or poses.shape[0] < 1 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [B,4,4] with B >= 1")


@lru_cache(maxsize=16)
def _mount_constants(mounts: tuple[MountConfig, ...], device: torch.device):
    matrices = np.stack([mount.as_transform().matrix for mount in mounts])
    tensor = torch.as_tensor(matrices, dtype=torch.float32, device=device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return tensor, ready


def mounted_poses(body_poses: torch.Tensor, mounts: Sequence[MountConfig]) -> torch.Tensor:
    """Compose all N body poses with M mounts on GPU, returning [N,M,4,4].

    Input transforms must be finite sensor/body-to-world rigid matrices; their
    values are not copied to CPU for validation. Only configuration constants
    are uploaded and cached. All roll, pitch, yaw and translations are retained.
    """
    _check_poses(body_poses)
    if (isinstance(mounts, (str, bytes)) or not isinstance(mounts, Sequence)
            or len(mounts) < 1 or not all(isinstance(mount, MountConfig) for mount in mounts)):
        raise TypeError("mounts must be a nonempty sequence of MountConfig values")
    tensor, ready = _mount_constants(tuple(mounts), body_poses.device)
    stream = torch.cuda.current_stream(body_poses.device)
    stream.wait_event(ready)
    tensor.record_stream(stream)
    return torch.matmul(body_poses[:, None], tensor[None])


@lru_cache(maxsize=24)
def _ray_constants(config: DepthConfig, device: torch.device):
    """Uniform subcell image-plane centres, cached independently of B."""
    samples = config.samples_per_axis
    rows = np.arange(config.height)[:, None, None, None]
    cols = np.arange(config.width)[None, :, None, None]
    subrows = np.arange(samples)[None, None, :, None]
    subcols = np.arange(samples)[None, None, None, :]
    y = (1 - 2 * (cols + (subcols + 0.5) / samples) / config.width)
    z = (1 - 2 * (rows + (subrows + 0.5) / samples) / config.height)
    y, z = np.broadcast_arrays(y * math.tan(math.radians(config.hfov_deg) / 2),
                                z * math.tan(math.radians(config.vfov_deg) / 2))
    directions = np.stack((np.ones_like(y), y, z), axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    tensor = torch.as_tensor(directions.reshape(-1, 3), dtype=torch.float32, device=device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return tensor, ready


def generate_rays(poses: torch.Tensor, config: DepthConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate contiguous world origins/directions [B*H*W*S,3] on CUDA.

    Pixel order is row-major; within each pixel, subray order is subrow-major.
    Directions are unit length for valid rigid input poses. The returned arrays
    are caller-owned and can be perturbed before invoking RayRenderer.trace.
    """
    _check_poses(poses)
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    directions, ready = _ray_constants(config, poses.device)
    stream = torch.cuda.current_stream(poses.device)
    stream.wait_event(ready)
    directions.record_stream(stream)
    world = torch.matmul(directions[None], poses[:, :3, :3].transpose(-1, -2))
    origins = poses[:, None, :3, 3].expand(-1, directions.shape[0], -1)
    return origins.reshape(-1, 3).contiguous(), world.reshape(-1, 3).contiguous()


def process_samples(hits: RayHits, origins: torch.Tensor, directions: torch.Tensor,
                    poses: torch.Tensor, config: DepthConfig, *,
                    reduction: str = "nearest", quantile: float = 0.2,
                    signal_floor: float = 0.08) -> DepthBatch:
    """Reduce one raw query without rendering again or downloading its data.

    The inputs may be changed by an outer perturbation model. Directions must
    match the world rays passed to trace; they are normalized here for the
    known-domain check, matching the core's ray-direction normalization.
    """
    _check_poses(poses)
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    shape = (poses.shape[0], config.height, config.width, config.samples_per_axis ** 2)
    rays = math.prod(shape)
    for name, values in (("origins", origins), ("directions", directions)):
        if (not isinstance(values, torch.Tensor) or values.device != poses.device
                or values.dtype != torch.float32 or values.shape != (rays, 3)):
            raise TypeError(f"{name} must be CUDA float32 [{rays},3] on poses.device")
    if hits.distance_m.shape != (rays,) or hits.distance_m.device != poses.device:
        raise ValueError("hits must match the generated ray count and device")
    distance = hits.distance_m.reshape(shape)
    identity = hits.object_index.reshape(shape)
    points = hits.hit_point_world_m.reshape(*shape, 3)
    triangles = hits.triangle_index.reshape(shape)
    origins = origins.reshape(*shape, 3)
    # Match the core's scale-first normalization for externally perturbed rays.
    scale = directions.abs().amax(dim=-1, keepdim=True)
    scaled = directions / scale.clamp_min(torch.finfo(torch.float32).tiny)
    directions = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    directions = directions.reshape(*shape, 3)
    selected = select_subray(distance, identity, config.max_range_m,
                            reduction=reduction, quantile=quantile,
                            signal_floor=signal_floor, min_distance=config.min_range_m)
    ranges, known = range_by_sample(distance, identity, origins, directions, config,
                                   hits.map_bounds_m, return_known=True)
    selected_distance = gather_subray(distance, selected)
    selected_identity = gather_subray(identity, selected)
    selected_triangles = gather_subray(triangles, selected)
    selected_world = gather_subray(points, selected)
    relative = selected_world - poses[:, None, None, :3, 3]
    sensor_points = torch.matmul(relative.reshape(len(poses), -1, 3),
                                 poses[:, :3, :3]).reshape(*shape[:3], 3)
    sensor_points = torch.where((selected_identity >= 0)[..., None], sensor_points, 0.0)
    raw = SampleBatch(selected_distance, selected_identity, hits.object_ids,
                      hits.scene_version, hits.t_ref, hits.map_bounds_m,
                      sensor_points, selected_triangles, distance, identity, points,
                      triangles, origins, directions, selected)
    representative = gather_subray(ranges, selected)
    representative = torch.where(known.all(dim=-1), representative, math.nan)
    return DepthBatch(representative, selected_identity, hits.object_ids, hits.scene_version,
                      hits.t_ref, sensor_points, selected_triangles, raw, ranges)


def _same_snapshot(left: SampleBatch, right: SampleBatch) -> None:
    if (left.scene_version != right.scene_version or left.object_ids != right.object_ids
            or left.t_ref != right.t_ref or left.map_bounds_m != right.map_bounds_m):
        raise RuntimeError("scene changed during sensor sampling; update only between batches")


def sample_depth(renderer: RayRenderer, config: DepthConfig,
                 poses: torch.Tensor, *, reduction: str = "nearest",
                 quantile: float = 0.2, signal_floor: float = 0.08) -> DepthBatch:
    """Trace an entire batch and apply its configured per-cell ray reduction."""
    origins, directions = generate_rays(poses, config)
    hits = renderer.trace(origins, directions, config.max_range_m)
    return process_samples(hits, origins, directions, poses, config,
                           reduction=reduction, quantile=quantile, signal_floor=signal_floor)


def sample_camera(renderer: RayRenderer, config: DepthConfig,
                  poses: torch.Tensor, min_visible_fraction: float,
                  target_id: str | None = "target") -> CameraBatch:
    """Trace full/target-only batches, then reduce camera coverage on CUDA.

    The representative depth is the nearest subray. Target coverage and its
    mask include all valid visible target subrays, so a mixed pixel may have an
    occluder representative while retaining nonzero target coverage.
    Explicit target_id=None disables recognition and the target-only query.
    """
    camera, _, _ = _sample_camera(renderer, config, poses, min_visible_fraction, target_id)
    return camera


def _sample_camera(renderer: RayRenderer, config: DepthConfig,
                   poses: torch.Tensor, min_visible_fraction: float,
                   target_id: str | None):
    _check_poses(poses)
    threshold = _finite_number(min_visible_fraction, "min_visible_fraction")
    if not 0 <= threshold <= 1:
        raise ValueError("min_visible_fraction must be in [0,1]")
    if target_id is not None and (not isinstance(target_id, str) or not target_id):
        raise ValueError("target_id must be a nonempty object ID or None")
    origins, directions = generate_rays(poses, config)
    hits = renderer.trace(origins, directions, config.max_range_m)
    depth = process_samples(hits, origins, directions, poses, config)
    full = depth.raw
    if target_id is None:
        count, height, width = depth.range_m.shape
        camera = CameraBatch(
            depth,
            torch.zeros((count, height, width), dtype=torch.bool, device=poses.device),
            torch.zeros((count, 5), dtype=torch.float32, device=poses.device),
            torch.zeros(count, dtype=torch.float64, device=poses.device),
            torch.zeros(count, dtype=torch.int64, device=poses.device),
            torch.zeros((count, height, width), dtype=torch.float32, device=poses.device),
            torch.zeros(count, dtype=torch.int64, device=poses.device),
        )
        return camera, full, None
    if target_id not in full.object_ids:
        raise ValueError(f"target object {target_id!r} is absent from the scene")
    target_hits = renderer.trace(origins, directions, config.max_range_m, only_object=target_id)
    target_depth = process_samples(target_hits, origins, directions, poses, config)
    target = target_depth.raw
    _same_snapshot(full, target)
    measurements = camera_measurements(depth.subray_range_m, full.subray_object_index,
                                      target_depth.subray_range_m, target.subray_object_index,
                                      config, hits.object_ids.index(target_id), threshold)
    return CameraBatch(depth, **measurements), full, target


def sample_rig(renderer: RayRenderer, config: SensorConfig,
               body_poses: torch.Tensor) -> RigBatch:
    """Assemble all mounted observations with three whole-batch ray queries."""
    return sample_rig_diagnostics(renderer, config, body_poses).rig


def sample_rig_diagnostics(renderer: RayRenderer, config: SensorConfig,
                           body_poses: torch.Tensor, *,
                           target_id: str | None = "target") -> RigDiagnostics:
    """Sample a batch of rigs and retain exactly the same geometric returns.

    Four mounted ToF views form one query; full cameras form another; optional
    target-only cameras form the third. Scene updates belong between calls.
    """
    if not isinstance(config, SensorConfig):
        raise TypeError("config must be a SensorConfig")
    _check_poses(body_poses)
    count = body_poses.shape[0]
    tof_poses = mounted_poses(body_poses, config.tof_mounts).reshape(-1, 4, 4)
    camera_poses = mounted_poses(body_poses, (config.camera_mount,))[:, 0]
    tof = sample_depth(renderer, config.tof, tof_poses, reduction="weighted_quantile",
                       quantile=config.tof_depth_quantile, signal_floor=config.tof_signal_floor)
    camera, camera_raw, target_raw = _sample_camera(
        renderer, config.camera, camera_poses, config.min_visible_fraction, target_id)
    _same_snapshot(tof.raw, camera_raw)
    tof_m = tof.range_m.reshape(count, 4, config.tof.height, config.tof.width)
    rig = RigBatch(tof_m, camera.target_hint, camera, tof.scene_version, tof.t_ref)
    return RigDiagnostics(rig, tof.raw, camera_raw, target_raw, tof_poses, camera_poses)


def clear_caches() -> None:
    """Drop cached mount/layout constants without invalidating queued consumers."""
    _mount_constants.cache_clear()
    _ray_constants.cache_clear()
