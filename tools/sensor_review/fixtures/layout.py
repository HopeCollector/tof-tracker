"""Review fixture: mounts, four ToFs and a forward depth camera.

Physical sensors accept world poses only. This module owns body-relative
installation and packages independent sensor requests into one shared batch.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from devtools.config import MountConfig, RigConfig
from devtools.scene import Pose
from devtools.sensors import CameraSensor, DepthBatch, SensorSampler, ToFSensor
from devtools.sensors._ops import SampleBatch, _check_poses, poses_to_tensor
from .capture import capture_sample


TOF_ORDER = ("right", "front", "left", "rear")


@dataclass(frozen=True)
class LayoutBatch:
    tof_m: torch.Tensor
    camera: DepthBatch
    scene_version: int
    t_ref: float | None
    execution_info: dict | None = None


@dataclass(frozen=True)
class LayoutDiagnostics:
    layout: LayoutBatch
    tof_raw: SampleBatch
    camera_raw: SampleBatch
    tof_poses: torch.Tensor
    camera_poses: torch.Tensor


def mounted_pose(body_pose: Pose, mount: MountConfig) -> Pose:
    """Compose a full sensor-to-body mount with a CPU body-to-world pose."""
    if not isinstance(body_pose, Pose) or not isinstance(mount, MountConfig):
        raise TypeError("expected Pose and MountConfig")
    position = body_pose.body_to_world(mount.position_m)
    a, b = body_pose.quaternion_xyzw, mount.quaternion_xyzw
    xyz = a[3] * np.array(b[:3]) + b[3] * np.array(a[:3]) + np.cross(a[:3], b[:3])
    w = a[3] * b[3] - np.dot(a[:3], b[:3])
    return Pose(position, tuple(float(value) for value in (*xyz, w)))


@lru_cache(maxsize=16)
def _mount_constants(mounts: tuple[MountConfig, ...], device: torch.device):
    matrices = np.stack([mount.as_transform().matrix for mount in mounts])
    tensor = torch.as_tensor(matrices, dtype=torch.float32, device=device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return tensor, ready


def mounted_poses(body_poses: torch.Tensor, mounts: Sequence[MountConfig]) -> torch.Tensor:
    """Compose N world body poses and M mounts, retaining all rigid axes."""
    _check_poses(body_poses)
    if (isinstance(mounts, (str, bytes)) or not isinstance(mounts, Sequence)
            or not mounts or not all(isinstance(mount, MountConfig) for mount in mounts)):
        raise TypeError("mounts must be a nonempty sequence of MountConfig values")
    tensor, ready = _mount_constants(tuple(mounts), body_poses.device)
    stream = torch.cuda.current_stream(body_poses.device)
    stream.wait_event(ready)
    tensor.record_stream(stream)
    return torch.matmul(body_poses[:, None], tensor[None])


def _collect_layout(sampler, config, body_poses, *, diagnostics=False,
                    camera_consumer=None, retained_output_bytes=0):
    if not isinstance(sampler, SensorSampler) or not isinstance(config, RigConfig):
        raise TypeError("expected SensorSampler and outer RigConfig")
    _check_poses(body_poses)
    tof_poses = mounted_poses(body_poses, config.tof_mounts).reshape(-1, 4, 4)
    camera_poses = mounted_poses(body_poses, (config.camera_mount,))[:, 0]
    tof = ToFSensor(sampler, config.tof, quantile=config.tof_depth_quantile,
                    signal_floor=config.tof_signal_floor)
    camera = CameraSensor(sampler, config.camera)
    batch = sampler.batch()
    tof_handle = capture_sample(tof, tof_poses, batch=batch, diagnostics=diagnostics)
    camera_handle = capture_sample(camera, camera_poses, batch=batch, diagnostics=diagnostics,
                                   consumer=camera_consumer,
                                   retained_output_bytes=(retained_output_bytes
                                       + 4 * (tof_poses.numel() + camera_poses.numel())))
    return batch, tof_handle, camera_handle, tof_poses, camera_poses


def _layout_result(tof, camera, count):
    if (tof.object_ids != camera.object_ids or tof.scene_version != camera.scene_version
            or tof.t_ref != camera.t_ref):
        raise RuntimeError("scene changed between review layout requests")
    return LayoutBatch(tof.range_m.reshape(count, 4, *tof.range_m.shape[-2:]), camera,
                    tof.scene_version, tof.t_ref, camera.execution_info)


def sample_layout(sampler: SensorSampler, config: RigConfig, body_poses: torch.Tensor) -> LayoutBatch:
    """Sample this outer recipe through two requests in one mixed ray batch."""
    batch, tof, camera, _, _ = _collect_layout(sampler, config, body_poses)
    batch.execute()
    return _layout_result(tof.result(), camera.result(), len(body_poses))


def sample_layout_diagnostics(sampler: SensorSampler, config: RigConfig,
                              body_poses: torch.Tensor) -> LayoutDiagnostics:
    """Explicitly retain same-pass raw for a memory-budgeted review request."""
    batch, tof, camera, tof_poses, camera_poses = _collect_layout(
        sampler, config, body_poses, diagnostics=True)
    batch.execute()
    td, cd = tof.result(), camera.result()
    return LayoutDiagnostics(_layout_result(td, cd, len(body_poses)), td.raw, cd.raw,
                          tof_poses, camera_poses)


def clear_caches():
    """Release cached mount constants without invalidating queued consumers."""
    _mount_constants.cache_clear()
