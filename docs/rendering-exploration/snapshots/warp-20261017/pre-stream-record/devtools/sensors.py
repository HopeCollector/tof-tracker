"""Batched GPU sensor policies over a shared, sensor-agnostic ray core.

This layer generates rays from complete sensor-to-world FLU poses, then applies
ToF or camera reductions. Ray directions can be changed before tracing and raw
returns can be changed before process_samples: the core has no noise hooks.
Single views use B=1; no sampling function downloads tensors to the CPU.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np
import torch

from .config import DepthConfig, MountConfig, SensorConfig, SamplingRuntimeConfig
from ._ray_budget import plan
from .ray_rendering import RayRenderer, RayHits
from .scene import Pose
from ._ray_processing import gather_subray, range_by_sample, select_subray


TOF_ORDER = ("right", "front", "left", "rear")


@dataclass(frozen=True)
class SampleBatch:
    """Same-pass raw geometric samples and the selected representative ray.

    Representative fields have shape [B,H,W]; subray fields have [B,H,W,S].
    Vec3 sample fields append three. Origins retain compact [B,3], shared [1,3],
    or caller-indexed [M,3] storage; sample_ray_origins gives its broadcast grid.
    Raw misses remain zero/-1.
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
    ray_origin_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class DepthBatch:
    """CUDA radial ranges [B,H,W] and matching representative identities.

    A known selected hit in [min,max] is finite; a known no-return is +inf.
    Any unknown constituent ray makes the cell unavailable (NaN), as does a
    selected too-close hit. This is conservative over the finite samples,
    without claiming continuous angular coverage between them. Identities and
    raw geometry remain development diagnostics, never online model inputs.
    Compact sampling leaves identity/point/triangle fields None and keeps only
    the physical range tensor and scene metadata.
    """

    range_m: torch.Tensor
    object_index: torch.Tensor | None
    object_ids: tuple[str, ...]
    scene_version: int
    t_ref: float | None
    hit_point_sensor_m: torch.Tensor | None = None
    triangle_index: torch.Tensor | None = None
    raw: SampleBatch | None = None
    subray_range_m: torch.Tensor | None = None
    execution_info: dict | None = None


@dataclass(frozen=True)
class RigBatch:
    """Synthetic simultaneous observations for N body poses on one snapshot.

    tof_m is CUDA float32 [N,4,8,8] in right/front/left/rear order.
    Camera carries physical depth only. Invalid ToF values require adaptation.
    Sampling does not simulate asynchronous hardware timing or sensor noise.
    """

    tof_m: torch.Tensor
    camera: DepthBatch
    scene_version: int
    t_ref: float | None
    execution_info: dict | None = None


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
    tof_poses: torch.Tensor
    camera_poses: torch.Tensor
    body_start: int = 0
    body_stop: int = 0


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
                    signal_floor: float = 0.08,
                    origin_indices: torch.Tensor | None = None) -> DepthBatch:
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
    if (not isinstance(directions, torch.Tensor) or directions.device != poses.device
            or directions.dtype != torch.float32 or directions.shape != (rays, 3)):
        raise TypeError(f"directions must be CUDA float32 [{rays},3] on poses.device")
    if (not isinstance(origins, torch.Tensor) or origins.device != poses.device
            or origins.dtype != torch.float32 or origins.ndim != 2 or origins.shape[1] != 3):
        raise TypeError("origins must be CUDA float32 [M,3] on poses.device")
    if hits.distance_m.shape != (rays,) or hits.distance_m.device != poses.device:
        raise ValueError("hits must match the generated ray count and device")
    distance = hits.distance_m.reshape(shape)
    identity = hits.object_index.reshape(shape)
    points = hits.hit_point_world_m.reshape(*shape, 3)
    triangles = hits.triangle_index.reshape(shape)
    origin_grid = _origin_grid(origins, shape, origin_indices)
    # Match the core's scale-first normalization for externally perturbed rays.
    scale = directions.abs().amax(dim=-1, keepdim=True)
    scaled = directions / scale.clamp_min(torch.finfo(torch.float32).tiny)
    directions = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    directions = directions.reshape(*shape, 3)
    selected = select_subray(distance, identity, config.max_range_m,
                            reduction=reduction, quantile=quantile,
                            signal_floor=signal_floor, min_distance=config.min_range_m)
    ranges, known = range_by_sample(distance, identity, origin_grid, directions, config,
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
                      triangles, origins, directions, selected, origin_indices)
    representative = gather_subray(ranges, selected)
    representative = torch.where(known.all(dim=-1), representative, math.nan)
    return DepthBatch(representative, selected_identity, hits.object_ids, hits.scene_version,
                      hits.t_ref, sensor_points, selected_triangles, raw, ranges)


def _same_snapshot(left: SampleBatch, right: SampleBatch) -> None:
    if (left.scene_version != right.scene_version or left.object_ids != right.object_ids
            or left.t_ref != right.t_ref or left.map_bounds_m != right.map_bounds_m):
        raise RuntimeError("scene changed during sensor sampling; update only between batches")


def _origin_grid(origins, shape, indices=None):
    count = math.prod(shape)
    if indices is not None:
        if (not isinstance(indices, torch.Tensor) or indices.device != origins.device
                or indices.dtype not in (torch.int32, torch.int64) or indices.shape != (count,)):
            raise ValueError("origin_indices must be CUDA int32/int64 [ray_count]")
        if len(origins) < 1:
            raise ValueError("nonempty samples require at least one origin")
        valid = (indices >= 0) & (indices < len(origins))
        expanded = origins.index_select(0, indices.clamp(0, len(origins) - 1).to(torch.int64))
        return torch.where(valid[:, None], expanded, math.nan).reshape(*shape, 3)
    if len(origins) == count:
        return origins.reshape(*shape, 3)
    if len(origins) == shape[0]:
        return origins[:, None, None, None, :]
    if len(origins) == 1:
        return origins.reshape(1, 1, 1, 1, 3)
    raise ValueError("origins must have one, B, or ray_count rows, or supply origin_indices")


def sample_ray_origins(raw: SampleBatch) -> torch.Tensor:
    """Return a broadcastable origin grid for compact or externally indexed rays."""
    return _origin_grid(raw.ray_origins_world_m, raw.subray_distance_m.shape, raw.ray_origin_indices)


def _compact_rays(poses, config):
    _check_poses(poses)
    directions, ready = _ray_constants(config, poses.device)
    current = torch.cuda.current_stream(poses.device)
    current.wait_event(ready)
    directions.record_stream(current)
    world = torch.matmul(directions[None], poses[:, :3, :3].transpose(-1, -2)).reshape(-1, 3)
    origins = poses[:, :3, 3].contiguous()
    indices = torch.arange(len(poses), dtype=torch.int32, device=poses.device).repeat_interleave(len(directions))
    return origins, world, indices


def _plan_chunks(renderer, item_count, rays_per_item, retained_output_bytes, runtime,
                 extra_fixed_bytes=0):
    if runtime is None:
        runtime = SamplingRuntimeConfig()
    if not isinstance(runtime, SamplingRuntimeConfig):
        raise TypeError("runtime must be a SamplingRuntimeConfig")
    return plan(renderer, item_count, rays_per_item, retained_output_bytes, runtime,
                extra_fixed_bytes=extra_fixed_bytes)


def depth_output_bytes(config: DepthConfig, count: int, *, diagnostics=False) -> int:
    """Conservative retained tensor reservation, including concatenation copies."""
    cells = count * config.height * config.width
    rays = cells * config.samples_per_axis ** 2
    retained = 4 * cells
    if diagnostics:
        retained = 32 * cells + 64 * rays + 64 * cells + 12 * count
    return 2 * retained


def rig_output_bytes(config: SensorConfig, count: int, *, diagnostics=False) -> int:
    result = 2 * count * 4 * config.tof.height * config.tof.width * 4
    result += depth_output_bytes(config.camera, count, diagnostics=diagnostics)
    if diagnostics:
        result += depth_output_bytes(config.tof, 4 * count, diagnostics=True)
        result += 2 * count * 5 * 16 * 4
    return result


def compact_depth(depth: DepthBatch) -> DepthBatch:
    """Keep only physical depth and metadata, releasing geometric diagnostics."""
    return replace(depth, object_index=None, hit_point_sensor_m=None,
                   triangle_index=None, raw=None, subray_range_m=None)


def _slice_hits(hits, start, stop):
    return RayHits(hits.distance_m[start:stop], hits.object_index[start:stop],
                   hits.hit_point_world_m[start:stop], hits.triangle_index[start:stop],
                   hits.object_ids, hits.scene_version, hits.t_ref, hits.map_bounds_m)


def concat_samples(samples):
    if len(samples) == 1:
        return samples[0]
    first = samples[0]
    for raw in samples[1:]:
        _same_snapshot(first, raw)
    def cat(name):
        return torch.cat([getattr(raw, name) for raw in samples], dim=0)
    if any(raw.ray_origin_indices is not None for raw in samples):
        raise ValueError("concatenated diagnostics require internal compact-per-view origins")
    return SampleBatch(cat("distance_m"), cat("object_index"), first.object_ids,
                       first.scene_version, first.t_ref, first.map_bounds_m,
                       cat("hit_point_sensor_m"), cat("triangle_index"),
                       cat("subray_distance_m"), cat("subray_object_index"),
                       cat("subray_hit_point_world_m"), cat("subray_triangle_index"),
                       cat("ray_origins_world_m"), cat("ray_directions_world"),
                       cat("selected_subray"))


def concat_depth(depths, *, diagnostics=False, execution_info=None):
    first = depths[0]
    for depth in depths[1:]:
        if (depth.object_ids != first.object_ids or depth.scene_version != first.scene_version
                or depth.t_ref != first.t_ref):
            raise RuntimeError("scene changed between sampling chunks")
    def cat(name):
        values = [getattr(depth, name) for depth in depths]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("cannot concatenate compact and diagnostic depth fields")
        return values[0] if len(values) == 1 else torch.cat(values, dim=0)
    raw = concat_samples([depth.raw for depth in depths]) if diagnostics else None
    return DepthBatch(cat("range_m"), cat("object_index"), first.object_ids,
                      first.scene_version, first.t_ref, cat("hit_point_sensor_m"),
                      cat("triangle_index"), raw, cat("subray_range_m") if diagnostics else None,
                      execution_info)


def sample_depth(renderer: RayRenderer, config: DepthConfig,
                 poses: torch.Tensor, *, reduction: str = "nearest",
                 quantile: float = 0.2, signal_floor: float = 0.08,
                 runtime: SamplingRuntimeConfig | None = None,
                 diagnostics: bool = False) -> DepthBatch:
    """Sample independent depth views in memory-budgeted, whole-view chunks.

    Default results keep only pixel-size outputs. Explicit diagnostics retains
    all raw samples and is reserved in the budget before any query begins.
    """
    _check_poses(poses)
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    count, rays_per_item = len(poses), config.height * config.width * config.samples_per_axis ** 2
    chunk_size, budget = _plan_chunks(renderer, count, rays_per_item,
                                     depth_output_bytes(config, count, diagnostics=diagnostics), runtime)
    pieces = []
    for start in range(0, count, chunk_size):
        current = poses[start:start + chunk_size]
        origins, directions, indices = _compact_rays(current, config)
        hits = renderer.trace(origins, directions, config.max_range_m, origin_indices=indices)
        result = process_samples(hits, origins, directions, current, config, reduction=reduction,
                                 quantile=quantile, signal_floor=signal_floor)
        pieces.append(result if diagnostics else compact_depth(result))
        del result, hits, origins, directions, indices
    info = {"chunks": len(pieces), "chunk_count": len(pieces), "chunk_items": chunk_size,
            "trace_calls": len(pieces), "rays_per_item": rays_per_item,
            "compact_origins": True, "planner": budget, "diagnostics": diagnostics}
    return concat_depth(pieces, diagnostics=diagnostics, execution_info=info)


def sample_camera(renderer: RayRenderer, config: DepthConfig, poses: torch.Tensor, *,
                  runtime: SamplingRuntimeConfig | None = None,
                  diagnostics: bool = False) -> DepthBatch:
    """Physical depth only, using the nearest configured subray per pixel."""
    return sample_depth(renderer, config, poses, runtime=runtime, diagnostics=diagnostics)


def iter_rig_samples(renderer: RayRenderer, config: SensorConfig, body_poses: torch.Tensor, *,
                     runtime: SamplingRuntimeConfig | None = None,
                     retained_output_bytes: int = 0):
    """Yield same-pass physical outputs/raw for one body chunk at a time.

    Each chunk concatenates every ToF and camera full-scene ray into exactly
    one trace call, with per-ray maxima and compact origins. Consumers process
    and release each raw chunk before advancing; retaining all yielded raw
    bypasses the streaming memory contract. Reserve any accumulated compact
    outputs using retained_output_bytes before the first chunk is allocated.
    """
    if not isinstance(config, SensorConfig):
        raise TypeError("config must be a SensorConfig")
    _check_poses(body_poses)
    count = len(body_poses)
    tof_rays = config.tof.height * config.tof.width * config.tof.samples_per_axis ** 2
    camera_rays = config.camera.height * config.camera.width * config.camera.samples_per_axis ** 2
    rays_per_item = 4 * tof_rays + camera_rays
    chunk_size, budget = _plan_chunks(renderer, count, rays_per_item, retained_output_bytes, runtime)
    chunks = math.ceil(count / chunk_size)
    info = {"chunks": chunks, "chunk_count": chunks, "chunk_items": chunk_size,
            "trace_calls": chunks, "rays_per_item": rays_per_item,
            "compact_origins": True, "planner": budget, "full_scene_queries_per_chunk": 1}
    for start in range(0, count, chunk_size):
        body = body_poses[start:start + chunk_size]
        n = len(body)
        tof_poses = mounted_poses(body, config.tof_mounts).reshape(-1, 4, 4)
        camera_poses = mounted_poses(body, (config.camera_mount,))[:, 0]
        to, td, ti = _compact_rays(tof_poses, config.tof)
        co, cd, ci = _compact_rays(camera_poses, config.camera)
        origins = torch.cat((to, co), dim=0)
        directions = torch.cat((td, cd), dim=0)
        indices = torch.cat((ti, ci + len(to)), dim=0)
        limits = torch.cat((torch.full((len(td),), config.tof.max_range_m, device=body.device, dtype=torch.float32),
                            torch.full((len(cd),), config.camera.max_range_m, device=body.device, dtype=torch.float32)))
        boundary = len(td)
        hits = renderer.trace(origins, directions, limits, origin_indices=indices)
        # Groups keep only their B origins; no origin is expanded to N rays.
        tof = process_samples(_slice_hits(hits, 0, boundary), to, directions[:boundary],
                              tof_poses, config.tof, reduction="weighted_quantile",
                              quantile=config.tof_depth_quantile, signal_floor=config.tof_signal_floor)
        camera = process_samples(_slice_hits(hits, boundary, len(directions)), co, directions[boundary:],
                                 camera_poses, config.camera)
        _same_snapshot(tof.raw, camera.raw)
        rig = RigBatch(tof.range_m.reshape(n, 4, config.tof.height, config.tof.width),
                       camera, tof.scene_version, tof.t_ref, info)
        yield RigDiagnostics(rig, tof.raw, camera.raw, tof_poses, camera_poses, start, start + n)
        del rig, tof, camera, hits, origins, directions, indices, limits, to, td, ti, co, cd, ci


def sample_rig(renderer: RayRenderer, config: SensorConfig, body_poses: torch.Tensor, *,
               runtime: SamplingRuntimeConfig | None = None) -> RigBatch:
    """Collect compact physical observations; never retain the batch's raw rays."""
    _check_poses(body_poses)
    tofs, cameras, version, t_ref, info = [], [], None, None, None
    iterator = iter_rig_samples(renderer, config, body_poses, runtime=runtime,
                                retained_output_bytes=rig_output_bytes(config, len(body_poses)))
    for chunk in iterator:
        if version is not None and (version != chunk.rig.scene_version or t_ref != chunk.rig.t_ref):
            raise RuntimeError("scene changed between sampling chunks")
        version, t_ref, info = chunk.rig.scene_version, chunk.rig.t_ref, chunk.rig.execution_info
        tofs.append(chunk.rig.tof_m)
        cameras.append(compact_depth(chunk.rig.camera))
        del chunk
    return RigBatch(tofs[0] if len(tofs) == 1 else torch.cat(tofs),
                    concat_depth(cameras, execution_info=info), version, t_ref, info)


def sample_rig_diagnostics(renderer: RayRenderer, config: SensorConfig, body_poses: torch.Tensor, *,
                           runtime: SamplingRuntimeConfig | None = None) -> RigDiagnostics:
    """Retain raw for small review batches, rejecting oversized reservations early."""
    _check_poses(body_poses)
    parts = list(iter_rig_samples(renderer, config, body_poses, runtime=runtime,
                                  retained_output_bytes=rig_output_bytes(config, len(body_poses), diagnostics=True)))
    first, info = parts[0], parts[0].rig.execution_info
    camera = concat_depth([part.rig.camera for part in parts], diagnostics=True, execution_info=info)
    tof_raw = concat_samples([part.tof_raw for part in parts])
    tofs = [part.rig.tof_m for part in parts]
    rig = RigBatch(tofs[0] if len(tofs) == 1 else torch.cat(tofs), camera,
                   first.rig.scene_version, first.rig.t_ref, info)
    return RigDiagnostics(rig, tof_raw, camera.raw,
                          torch.cat([part.tof_poses for part in parts]),
                          torch.cat([part.camera_poses for part in parts]), 0, len(body_poses))


def clear_caches() -> None:
    """Drop cached mount/layout constants without invalidating queued consumers."""
    _mount_constants.cache_clear()
    _ray_constants.cache_clear()
