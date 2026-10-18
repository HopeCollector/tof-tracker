"""Development environment's simulated recognition, outside physical sensors.

Visibility comes from the full-scene closest-hit identities. The unoccluded
denominator uses Torch analytic sphere intersections of those same camera
subrays, never a second renderer query or a projected bounding-box estimate.
All tensors remain on CUDA; no noise or learned RGB recognition is implied.
"""

from dataclasses import dataclass, replace
import math

import torch

from .config import DepthConfig, PerceptionConfig, RigConfig, _finite_number
from .scene import GeometrySnapshot, Sphere
from .sensors import DepthBatch, SensorSampler
from .rig import _submit_rig
from ._sensor_ops import (SampleBatch, compact_depth, concat_samples, gather_subray,
                          sample_ray_origins, select_subray)
from ._ray_processing import _check_samples, range_by_sample


@dataclass(frozen=True)
class PerceivedCameraBatch:
    """Physical depth plus environment-generated recognition supervision."""

    depth: DepthBatch
    target_mask: torch.Tensor
    target_hint: torch.Tensor
    visible_fraction: torch.Tensor
    unoccluded_pixels: torch.Tensor
    target_coverage: torch.Tensor
    unoccluded_samples: torch.Tensor


@dataclass(frozen=True)
class MeasurementBatch:
    """Compact algorithm-facing synthetic measurements, grouped on CUDA."""

    tof_m: torch.Tensor
    camera: PerceivedCameraBatch
    scene_version: int
    t_ref: float | None
    execution_info: dict | None = None

    @property
    def target_hint(self):
        return self.camera.target_hint


@dataclass(frozen=True)
class MeasurementDiagnostics:
    """Small-batch review of physical sampling and outer analytic target facts."""

    rig: MeasurementBatch
    tof_raw: SampleBatch
    camera_raw: SampleBatch
    target_raw: SampleBatch | None
    tof_poses: torch.Tensor
    camera_poses: torch.Tensor


def camera_measurements(full_ranges: torch.Tensor, full_identity: torch.Tensor,
                        target_ranges: torch.Tensor, target_identity: torch.Tensor,
                        config: DepthConfig, target_index: int,
                        threshold: float) -> dict[str, torch.Tensor]:
    """Finite-sample target coverage, visible fraction and normalized boxes."""
    _check_samples(full_ranges, full_identity)
    _check_samples(target_ranges, target_identity)
    if full_ranges.shape != target_ranges.shape or full_ranges.device != target_ranges.device:
        raise ValueError("full and target-only samples must have the same shape and device")
    threshold = _finite_number(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
        raise ValueError("target_index must be a nonnegative object index")
    visible = (full_identity == target_index) & torch.isfinite(full_ranges)
    unoccluded = (target_identity == target_index) & torch.isfinite(target_ranges)
    coverage = visible.to(torch.float32).mean(dim=-1)
    mask = visible.any(dim=-1)
    count = unoccluded.sum(dim=(-3, -2, -1), dtype=torch.int64)
    visible_count = visible.sum(dim=(-3, -2, -1), dtype=torch.int64)
    fraction = visible_count.to(torch.float64) / count.clamp_min(1).to(torch.float64)
    fraction = torch.where(count > 0, fraction, 0.0)
    detected = (count > 0) & (visible_count > 0) & (fraction >= threshold)
    rows = torch.arange(config.height, device=full_ranges.device)
    cols = torch.arange(config.width, device=full_ranges.device)
    row_occupied, col_occupied = mask.any(dim=-1), mask.any(dim=-2)
    first_row = torch.where(row_occupied, rows, config.height).amin(dim=-1)
    last_row = torch.where(row_occupied, rows, -1).amax(dim=-1)
    first_col = torch.where(col_occupied, cols, config.width).amin(dim=-1)
    last_col = torch.where(col_occupied, cols, -1).amax(dim=-1)
    box = torch.stack((first_col / config.width, first_row / config.height,
                       (last_col + 1) / config.width, (last_row + 1) / config.height), dim=-1)
    hint = torch.cat((detected[:, None].to(torch.float32), box), dim=-1)
    hint = torch.where(detected[:, None], hint, 0.0)
    return {"target_mask": mask, "target_coverage": coverage, "target_hint": hint,
            "visible_fraction": fraction,
            "unoccluded_pixels": unoccluded.any(dim=-1).sum(dim=(-2, -1), dtype=torch.int64),
            "unoccluded_samples": count}


def _empty_perception(depth):
    count, height, width = depth.range_m.shape
    device = depth.range_m.device
    return PerceivedCameraBatch(
        depth, torch.zeros((count, height, width), dtype=torch.bool, device=device),
        torch.zeros((count, 5), dtype=torch.float32, device=device),
        torch.zeros(count, dtype=torch.float64, device=device),
        torch.zeros(count, dtype=torch.int64, device=device),
        torch.zeros((count, height, width), dtype=torch.float32, device=device),
        torch.zeros(count, dtype=torch.int64, device=device))


def _sphere_returns(raw, geometry, config, target_index):
    origins = sample_ray_origins(raw)
    directions = raw.ray_directions_world
    center = torch.tensor(geometry.pose.position_m, dtype=torch.float32, device=directions.device)
    radius = torch.tensor(geometry.radius_m, dtype=torch.float32, device=directions.device)
    relative = origins - center
    b = (relative * directions).sum(dim=-1)
    perpendicular = relative - b[..., None] * directions
    discriminant = radius * radius - perpendicular.square().sum(dim=-1)
    root = discriminant.clamp_min(0).sqrt()
    near = -b - root
    distance = torch.where(near >= 0, near, -b + root)
    limit = config.max_range_m + max(1e-6, config.max_range_m * 1e-6)
    valid = ((discriminant >= 0) & (distance >= 0) & (distance <= limit)
             & torch.isfinite(distance))
    distance = torch.where(valid, distance, 0.0)
    identity = torch.where(valid, target_index, -1).to(torch.int64)
    ranges = range_by_sample(distance, identity, origins, directions, config, raw.map_bounds_m)
    return distance, identity, ranges


def _target_diagnostics(raw, poses, config, distance, identity):
    """Assemble only requested target diagnostics; no second generic process."""
    selected = select_subray(distance, identity, config.max_range_m)
    representative_distance = gather_subray(distance, selected)
    representative_identity = gather_subray(identity, selected)
    origins = sample_ray_origins(raw)
    points = origins + distance[..., None] * raw.ray_directions_world
    points = torch.where((identity >= 0)[..., None], points, 0.0)
    selected_world = gather_subray(points, selected)
    relative = selected_world - poses[:, None, None, :3, 3]
    sensor_points = torch.matmul(relative.reshape(len(poses), -1, 3), poses[:, :3, :3])
    sensor_points = sensor_points.reshape(*distance.shape[:3], 3)
    sensor_points = torch.where((representative_identity >= 0)[..., None], sensor_points, 0.0)
    return SampleBatch(
        representative_distance, representative_identity, raw.object_ids, raw.scene_version,
        raw.t_ref, raw.map_bounds_m, sensor_points, torch.full_like(representative_identity, -1),
        distance, identity, points, torch.full_like(identity, -1), raw.ray_origins_world_m,
        raw.ray_directions_world, selected, raw.ray_origin_indices)


def process_camera_perception(depth: DepthBatch, camera_raw: SampleBatch, poses: torch.Tensor,
                              camera_config: DepthConfig, snapshot: GeometrySnapshot,
                              config: PerceptionConfig, *, target_id: str | None = "target",
                              diagnostics: bool = False):
    """Interpret one physical camera batch without calling any scene renderer.

    The caller supplies the same immutable public geometry snapshot as the
    physical query. Only sphere targets are supported. Missing/disabled targets
    produce zero recognition and no target-only diagnostic buffers.
    """
    if not isinstance(snapshot, GeometrySnapshot) or not isinstance(config, PerceptionConfig):
        raise TypeError("expected GeometrySnapshot and PerceptionConfig")
    if target_id is not None and (not isinstance(target_id, str) or not target_id):
        raise ValueError("target_id must be a nonempty string or None")
    if (tuple(obj.object_id for obj in snapshot.objects) != camera_raw.object_ids
            or snapshot.map_bounds_m != camera_raw.map_bounds_m or snapshot.t_ref != camera_raw.t_ref):
        raise ValueError("perception snapshot metadata must match the physical sample")
    if target_id is None or target_id not in camera_raw.object_ids:
        return _empty_perception(depth), None
    geometry = snapshot.get_object(target_id).geometry
    if not isinstance(geometry, Sphere):
        raise NotImplementedError("analytic perception currently supports sphere targets only")
    target_index = camera_raw.object_ids.index(target_id)
    distance, identity, ranges = _sphere_returns(camera_raw, geometry, camera_config, target_index)
    full_ranges = depth.subray_range_m
    if full_ranges is None:
        full_ranges = range_by_sample(camera_raw.subray_distance_m, camera_raw.subray_object_index,
                                      sample_ray_origins(camera_raw), camera_raw.ray_directions_world,
                                      camera_config, camera_raw.map_bounds_m)
    values = camera_measurements(full_ranges, camera_raw.subray_object_index, ranges, identity,
                                 camera_config, target_index, config.min_visible_fraction)
    target_raw = _target_diagnostics(camera_raw, poses, camera_config, distance, identity) if diagnostics else None
    return PerceivedCameraBatch(depth, **values), target_raw


def measurement_output_bytes(sensors, count, *, diagnostics=False):
    """Reserve only outer perception outputs; sensor handles budget their own."""
    camera_cells = count * sensors.camera.height * sensors.camera.width
    retained = 2 * (5 * camera_cells + count * (20 + 8 + 16))
    if diagnostics:
        rays = camera_cells * sensors.camera.samples_per_axis ** 2
        retained += 2 * (64 * rays + 64 * camera_cells)
    return retained


def _concat_cameras(cameras, depth):
    def cat(name):
        values = [getattr(camera, name) for camera in cameras]
        return values[0] if len(values) == 1 else torch.cat(values)
    return PerceivedCameraBatch(depth, cat("target_mask"), cat("target_hint"), cat("visible_fraction"),
                                cat("unoccluded_pixels"), cat("target_coverage"), cat("unoccluded_samples"))


def _sample(sampler, sensors, body_poses, snapshot, config, target_id, diagnostics):
    if not isinstance(sampler, SensorSampler) or not isinstance(sensors, RigConfig):
        raise TypeError("expected SensorSampler and outer RigConfig")
    if not isinstance(config, PerceptionConfig):
        raise TypeError("config must be PerceptionConfig")
    cameras, target_raws = [], []

    def consume_camera(depth, raw, poses, start, stop):
        perceived, target_raw = process_camera_perception(
            depth, raw, poses, sensors.camera, snapshot, config,
            target_id=target_id, diagnostics=diagnostics)
        # Keep only compact physical output and recognition fields. All
        # full-scene raw stays chunk-local unless diagnostics was explicit.
        cameras.append(replace(perceived, depth=compact_depth(depth)))
        if target_raw is not None:
            target_raws.append(target_raw)

    batch, tof, camera, tof_poses, camera_poses = _submit_rig(
        sampler, sensors, body_poses, diagnostics=diagnostics,
        camera_consumer=consume_camera,
        retained_output_bytes=measurement_output_bytes(sensors, len(body_poses), diagnostics=diagnostics))
    batch.execute()
    td, cd = tof.result(), camera.result()
    if td.scene_version != cd.scene_version or td.t_ref != cd.t_ref or td.object_ids != cd.object_ids:
        raise RuntimeError("scene changed between measurement requests")
    info = {**batch.execution_info, "analytic_perception": "sphere_torch",
            "perception_scene_queries": 0, "diagnostics": diagnostics}
    cd = replace(cd, execution_info=info)
    perceived = _concat_cameras(cameras, cd)
    result = MeasurementBatch(td.range_m.reshape(len(body_poses), 4, *td.range_m.shape[-2:]),
                              perceived, td.scene_version, td.t_ref, info)
    if not diagnostics:
        return result
    return MeasurementDiagnostics(result, td.raw, cd.raw,
                                  concat_samples(target_raws) if target_raws else None,
                                  tof_poses, camera_poses)


def sample_measurements(sampler: SensorSampler, sensors: RigConfig, body_poses: torch.Tensor,
                        snapshot: GeometrySnapshot, config: PerceptionConfig, *,
                        target_id: str | None = "target") -> MeasurementBatch:
    """One shared sensor batch, streaming camera raw into outer recognition."""
    return _sample(sampler, sensors, body_poses, snapshot, config, target_id, False)


def sample_measurement_diagnostics(sampler: SensorSampler, sensors: RigConfig,
                                  body_poses: torch.Tensor, snapshot: GeometrySnapshot,
                                  config: PerceptionConfig, *,
                                  target_id: str | None = "target") -> MeasurementDiagnostics:
    """Explicit same-pass raw retention, budgeted before any query starts."""
    return _sample(sampler, sensors, body_poses, snapshot, config, target_id, True)
