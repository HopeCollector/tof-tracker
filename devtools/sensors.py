"""Batched development observations backed by one GPU scene.

The renderer accepts complete sensor-to-world FLU poses. This module assembles
mounts and applies the measurement policy to whole CUDA batches: radial ranges,
known-map coverage, visible target masks and normalized half-open boxes. This
is a noiseless geometric sensor, not a physical ToF or RGB appearance model.

Single views use B=1. CPU conversion is explicit at a caller's diagnostic or
export boundary; none is performed by the sampling functions here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch

from .config import DepthConfig, MountConfig, SensorConfig, _finite_number
from .rendering import BatchRenderer, RasterBatch
from .scene import Pose
from ._sensor_processing import postprocess_camera, postprocess_depth


TOF_ORDER = ("right", "front", "left", "rear")


@dataclass(frozen=True)
class DepthBatch:
    """Radial ranges and development identities for B simultaneous views.

    range_m is CUDA float32 [B,H,W]: finite within inclusive [min,max], +inf
    only for a fully covered empty segment, and NaN for unknown coverage,
    too-close surfaces or unavailable returns. object_index is CUDA int64 with
    -1 for no rasterized hit and otherwise indexes object_ids, including
    too-close hits. Identities and scene metadata are development diagnostics,
    never online model inputs. Frozen fields do not make tensors immutable.
    """

    range_m: torch.Tensor
    object_index: torch.Tensor
    object_ids: tuple[str, ...]
    scene_version: int
    t_ref: float | None


@dataclass(frozen=True)
class CameraBatch:
    """CUDA masks, boxes and visibility for the same B views as depth.

    target_mask is bool [B,H,W], target_hint is float32 [B,5] containing
    (detected, left, top, right, bottom), visible_fraction is float64 [B], and
    unoccluded_pixels is int64 [B]. A rejected detection has an all-zero hint;
    its visible-pixel mask remains available for development supervision.
    """

    depth: DepthBatch
    target_mask: torch.Tensor
    target_hint: torch.Tensor
    visible_fraction: torch.Tensor
    unoccluded_pixels: torch.Tensor


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


def _depth(raw: RasterBatch, poses: torch.Tensor, config: DepthConfig) -> DepthBatch:
    ranges = postprocess_depth(raw.distance_m, raw.object_index, poses, config,
                               raw.map_bounds_m)
    return DepthBatch(ranges, raw.object_index, raw.object_ids,
                      raw.scene_version, raw.t_ref)


def _same_snapshot(left: RasterBatch, right: RasterBatch) -> None:
    if (left.scene_version != right.scene_version or left.object_ids != right.object_ids
            or left.t_ref != right.t_ref or left.map_bounds_m != right.map_bounds_m):
        raise RuntimeError("scene changed during sensor sampling; update only between batches")


def sample_depth(renderer: BatchRenderer, config: DepthConfig,
                 poses: torch.Tensor) -> DepthBatch:
    """Render B complete view poses once and apply range/coverage on CUDA."""
    _check_poses(poses)
    return _depth(renderer.render(poses, config), poses, config)


def sample_camera(renderer: BatchRenderer, config: DepthConfig,
                  poses: torch.Tensor, min_visible_fraction: float,
                  target_id: str = "target") -> CameraBatch:
    """Two batched draws followed by CUDA target mask and box reductions.

    The denominator is target-only coverage at exactly the same poses, image,
    valid radial range and snapshot. Zero unoccluded or visible pixels means
    no detection; threshold equality is accepted. Updates belong between calls.
    """
    _check_poses(poses)
    threshold = _finite_number(min_visible_fraction, "min_visible_fraction")
    if not 0 <= threshold <= 1:
        raise ValueError("min_visible_fraction must be in [0,1]")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("target_id must be a nonempty object ID")
    full = renderer.render(poses, config)
    if target_id not in full.object_ids:
        raise ValueError(f"target object {target_id!r} is absent from the scene")
    target = renderer.render(poses, config, only_object=target_id)
    _same_snapshot(full, target)
    depth = _depth(full, poses, config)
    measurements = postprocess_camera(depth.range_m, depth.object_index,
                                      target.distance_m, target.object_index,
                                      poses, config, full.map_bounds_m,
                                      full.object_ids.index(target_id), threshold)
    return CameraBatch(depth, **measurements)


def sample_rig(renderer: BatchRenderer, config: SensorConfig,
               body_poses: torch.Tensor) -> RigBatch:
    """Assemble N rigs with three draws total: ToF, camera, target-only camera.

    Four mounted ToF poses per body are flattened into one 4N batch. Camera
    processing returns one observation per body. The scene stays fixed across
    all three draws; explicit synchronization of changed scene facts is the
    caller's responsibility before the next sample call.
    """
    if not isinstance(config, SensorConfig):
        raise TypeError("config must be a SensorConfig")
    _check_poses(body_poses)
    count = body_poses.shape[0]
    tof_poses = mounted_poses(body_poses, config.tof_mounts).reshape(-1, 4, 4)
    camera_poses = mounted_poses(body_poses, (config.camera_mount,))[:, 0]
    tof = sample_depth(renderer, config.tof, tof_poses)
    camera = sample_camera(renderer, config.camera, camera_poses, config.min_visible_fraction)
    if (tof.scene_version != camera.depth.scene_version or tof.object_ids != camera.depth.object_ids
            or tof.t_ref != camera.depth.t_ref):
        raise RuntimeError("scene changed during rig sampling; update only between batches")
    tof_m = tof.range_m.reshape(count, 4, config.tof.height, config.tof.width)
    return RigBatch(tof_m, camera.target_hint, camera, tof.scene_version, tof.t_ref)
