"""Batched CUDA range, coverage and target-visibility measurement rules.

All image/pose-dependent work stays on the input CUDA device. Functions use no
per-view loop, CPU tensor conversion, scalar tensor extraction or ``nonzero``.
Configuration and tuple-valued bounds are validated host metadata; only small
device constants and FLU pixel rays are cached, independently of batch size.

Inputs use top-down image rows, columns right, sensor FLU (+X forward), and
complete sensor-to-world transforms. ``identity`` is an object index with -1
for no hit; a rasterizer's packed face index must first be mapped to this index.
``distance`` is radial Euclidean metres, not normalized/NDC or optical depth.
Pose matrices must already have valid finite rigid-transform values;
checking their contents on the host would introduce a synchronization point.
"""

from collections.abc import Sequence
from functools import lru_cache
import math

import torch

from .config import DepthConfig, _finite_number


_FLOAT32_EPS = torch.finfo(torch.float32).eps
_IDENTITY_DTYPES = (torch.int32, torch.int64)


def _check_image(image: torch.Tensor, config: DepthConfig, name: str) -> None:
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    if not isinstance(image, torch.Tensor) or not image.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if image.ndim != 3 or image.shape[0] < 1 or image.shape[1:] != (config.height, config.width):
        raise ValueError(f"{name} must have shape (B, {config.height}, {config.width}) with B >= 1")
    if not image.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")


def _check_inputs(distance: torch.Tensor, identity: torch.Tensor,
                  poses_cuda: torch.Tensor, config: DepthConfig) -> None:
    _check_image(distance, config, "distance")
    if not isinstance(identity, torch.Tensor) or identity.device != distance.device:
        raise TypeError("identity must be a CUDA tensor on distance.device")
    if identity.shape != distance.shape or identity.dtype not in _IDENTITY_DTYPES:
        raise ValueError("identity must have distance.shape and int32 or int64 dtype")
    if not isinstance(poses_cuda, torch.Tensor) or poses_cuda.device != distance.device:
        raise TypeError("poses_cuda must be a CUDA tensor on distance.device")
    if poses_cuda.shape != (distance.shape[0], 4, 4):
        raise ValueError("poses_cuda must have shape (B, 4, 4)")
    if poses_cuda.dtype not in (torch.float32, torch.float64):
        raise TypeError("poses_cuda must use float32 or float64")


@lru_cache(maxsize=16)
def _pixel_constants(height: int, width: int, hfov_deg: float, vfov_deg: float,
                     device: torch.device):
    # Coverage arithmetic retains the previous NumPy implementation's float64
    # precision; the radial measurements themselves remain float32.
    rows = torch.arange(height, device=device, dtype=torch.int64)
    cols = torch.arange(width, device=device, dtype=torch.int64)
    left = (1 - 2 * (cols.to(torch.float64) + 0.5) / width) * math.tan(math.radians(hfov_deg) / 2)
    up = (1 - 2 * (rows.to(torch.float64) + 0.5) / height) * math.tan(math.radians(vfov_deg) / 2)
    z, y = torch.meshgrid(up, left, indexing="ij")
    rays = torch.stack((torch.ones_like(y), y, z), dim=-1)
    rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return (rays, rows, cols), ready


def _use_constants(cached, device: torch.device):
    tensors, ready = cached
    stream = torch.cuda.current_stream(device)
    stream.wait_event(ready)
    # Cache eviction may occur while another stream still uses a constant.
    # record_stream keeps its allocation alive through every queued consumer.
    for tensor in tensors:
        tensor.record_stream(stream)
    return tensors


def _pixels(config: DepthConfig, device: torch.device):
    return _use_constants(_pixel_constants(config.height, config.width,
                                          config.hfov_deg, config.vfov_deg, device), device)


@lru_cache(maxsize=8)
def _bounds_constant(bounds: tuple, device: torch.device):
    tensor = torch.tensor(bounds, device=device, dtype=torch.float64)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return (tensor,), ready


def _limits(bounds, device: torch.device) -> torch.Tensor:
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence) or len(bounds) != 3:
        raise ValueError("bounds must contain three increasing finite intervals")
    intervals = []
    for interval in bounds:
        if isinstance(interval, (str, bytes)) or not isinstance(interval, Sequence) or len(interval) != 2:
            raise ValueError("bounds must contain three increasing finite intervals")
        low, high = (_finite_number(value, "bounds") for value in interval)
        if low >= high:
            raise ValueError("bounds intervals must be increasing")
        intervals.append((low, high))
    return _use_constants(_bounds_constant(tuple(intervals), device), device)[0]


def roundoff_m(poses_cuda: torch.Tensor, config: DepthConfig) -> torch.Tensor:
    """Per-view float32 measurement budget, shape (B, 1, 1), on CUDA."""
    origin_size = poses_cuda[:, :3, 3].to(torch.float64).abs().amax(dim=-1)
    scale = origin_size.clamp_min(max(1.0, config.max_range_m))
    return (32 * _FLOAT32_EPS * scale)[:, None, None]


def _coverage(distance: torch.Tensor, poses_cuda: torch.Tensor,
              config: DepthConfig, bounds):
    if bounds is None:
        return torch.ones_like(distance, dtype=torch.bool), torch.zeros_like(distance, dtype=torch.bool)
    limits = _limits(bounds, distance.device)
    poses = poses_cuda.to(dtype=torch.float64)
    origin = poses[:, :3, 3]
    rotation = poses[:, :3, :3]
    rays, _, _ = _pixels(config, distance.device)
    directions = torch.matmul(rays.reshape(-1, 3), rotation.transpose(-1, -2))
    directions = directions.reshape(distance.shape[0], config.height, config.width, 3)
    origin_inside = ((origin >= limits[:, 0]) & (origin <= limits[:, 1])).all(dim=-1)
    origin_image = origin[:, None, None, :]
    inside_image = origin_inside[:, None, None]
    end = origin_image + config.max_range_m * directions
    full = inside_image & ((end >= limits[:, 0]) & (end <= limits[:, 1])).all(dim=-1)
    point = origin_image + distance.to(torch.float64)[..., None] * directions
    tolerance = roundoff_m(poses, config)[..., None]
    known = inside_image & ((point >= limits[:, 0] - tolerance)
                            & (point <= limits[:, 1] + tolerance)).all(dim=-1)
    return known, full


def postprocess_depth(distance: torch.Tensor, identity: torch.Tensor,
                      poses_cuda: torch.Tensor, config: DepthConfig, bounds=None) -> torch.Tensor:
    """Apply production range/coverage policy to raw batched CUDA hits.

    A measured nearest hit below ``min_range_m`` or just beyond max within the
    roundoff budget stays NaN and continues to occlude farther geometry. +inf
    requires strict coverage of the full max-range segment and no possibly
    in-range surface. Unknown coverage never establishes free space.
    """
    _check_inputs(distance, identity, poses_cuda, config)
    distance = distance.to(dtype=torch.float32)
    hit = identity >= 0
    known, full = _coverage(distance, poses_cuda, config, bounds)
    valid = hit & known & (distance >= config.min_range_m) & (distance <= config.max_range_m)
    unavailable = torch.full_like(distance, math.nan)
    result = torch.where(valid, distance, unavailable)
    empty = (~hit | (distance.to(torch.float64) > config.max_range_m + roundoff_m(poses_cuda, config))) & full
    return torch.where(empty, torch.full_like(distance, math.inf), result)


def postprocess_camera(full_ranges: torch.Tensor, full_identity: torch.Tensor,
                       target_distance: torch.Tensor, target_identity: torch.Tensor,
                       poses_cuda: torch.Tensor, config: DepthConfig, bounds,
                       target_index: int, threshold: float) -> dict[str, torch.Tensor]:
    """Derive visible target mask, fraction and gated half-open boxes on CUDA.

    ``full_ranges`` is the output of ``postprocess_depth`` for the complete
    scene. ``target_distance`` is the raw radial target-only image at exactly
    the same poses/configuration/snapshot. ``target_identity`` uses the same
    object-index mapping as ``full_identity``.

    Returns target_mask bool (B,H,W), target_hint float32 (B,5),
    visible_fraction float64 (B,), and unoccluded_pixels int64 (B,). The mask
    remains visible-pixel truth even when the detection threshold rejects it.
    """
    _check_inputs(full_ranges, full_identity, poses_cuda, config)
    _check_inputs(target_distance, target_identity, poses_cuda, config)
    if target_distance.shape != full_ranges.shape or target_distance.device != full_ranges.device:
        raise ValueError("full and target-only images must have the same shape and device")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or target_index < 0:
        raise ValueError("target_index must be a nonnegative object index")
    threshold = _finite_number(threshold, "threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    target_distance = target_distance.to(dtype=torch.float32)
    known_target, _ = _coverage(target_distance, poses_cuda, config, bounds)
    unoccluded = ((target_identity == target_index) & known_target
                  & (target_distance >= config.min_range_m) & (target_distance <= config.max_range_m))
    visible = (full_identity == target_index) & torch.isfinite(full_ranges)
    count = unoccluded.sum(dim=(-2, -1), dtype=torch.int64)
    visible_count = visible.sum(dim=(-2, -1), dtype=torch.int64)
    ratio = visible_count.to(torch.float64) / count.clamp_min(1).to(torch.float64)
    fraction = torch.where(count > 0, ratio, torch.zeros_like(ratio))
    detection = (count > 0) & (visible_count > 0) & (fraction >= threshold)

    # Reducing row/column occupancy avoids ragged nonzero output and its CUDA
    # synchronization. Sentinels make entirely empty images well-defined.
    _, rows, cols = _pixels(config, full_ranges.device)
    occupied_rows = visible.any(dim=-1)
    occupied_cols = visible.any(dim=-2)
    first_row = torch.where(occupied_rows, rows, config.height).amin(dim=-1)
    last_row = torch.where(occupied_rows, rows, -1).amax(dim=-1)
    first_col = torch.where(occupied_cols, cols, config.width).amin(dim=-1)
    last_col = torch.where(occupied_cols, cols, -1).amax(dim=-1)
    box = torch.stack((first_col.to(torch.float64) / config.width,
                       first_row.to(torch.float64) / config.height,
                       (last_col + 1).to(torch.float64) / config.width,
                       (last_row + 1).to(torch.float64) / config.height), dim=-1)
    hint = torch.cat((detection[:, None].to(torch.float32), box.to(torch.float32)), dim=-1)
    hint = torch.where(detection[:, None], hint, torch.zeros_like(hint))
    return {"target_mask": visible, "target_hint": hint,
            "visible_fraction": fraction, "unoccluded_pixels": count}


def clear_caches() -> None:
    """Release this module's references to cached CUDA constants."""
    _pixel_constants.cache_clear()
    _bounds_constant.cache_clear()
