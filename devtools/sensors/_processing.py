"""CUDA-only measurement rules over a batch of independent ray returns.

The ray dimension is last, with shapes [B,H,W,S]. Misses retain identity -1
and raw distance zero. A selected return always carries the matching identity
and point; weighting never synthesizes a surface between two real returns.

The signal/CDF equations follow the read-only Omni_ToF-YOPO reference at
cf0b7f4b17712667a8bbdb1126f5e914904ece9d, renderToFSenseMImage and
weightedQuantileDepth. These are new batched tensor operations, with no
reference-project runtime dependency or copied native rendering code.
"""

from functools import lru_cache
import math

import torch

from ..config import DepthConfig, _finite_number


def _check_samples(distance, identity):
    if (not isinstance(distance, torch.Tensor) or not distance.is_cuda
            or distance.dtype != torch.float32 or distance.ndim != 4):
        raise TypeError("distance must be CUDA float32 [B,H,W,S]")
    if (not isinstance(identity, torch.Tensor) or identity.device != distance.device
            or identity.shape != distance.shape or identity.dtype != torch.int64):
        raise TypeError("identity must be int64 on distance.device with distance.shape")
    if min(distance.shape) < 1:
        raise ValueError("sample dimensions must be nonempty")


def select_subray(distance: torch.Tensor, identity: torch.Tensor, max_distance: float, *,
                  reduction: str = "nearest", quantile: float = 0.2,
                  signal_floor: float = 0.08, min_distance: float = 0.0) -> torch.Tensor:
    """Choose one original ray per cell, returning CUDA int64 [B,H,W].

    Nearest and weighted-quantile reductions use row-major subray order to
    break equal-distance ties. The latter follows the reference signal model,
    but retains the original radial distance and identity after selection.
    No-return samples participate at max_distance with a floor signal.
    """
    _check_samples(distance, identity)
    maximum = _finite_number(max_distance, "max_distance")
    minimum = _finite_number(min_distance, "min_distance")
    if not 0 <= minimum < maximum:
        raise ValueError("ranges must satisfy 0 <= min_distance < max_distance")
    effective = torch.where(identity >= 0, distance, maximum)
    if reduction == "nearest":
        return effective.argmin(dim=-1)
    if reduction != "weighted_quantile":
        raise ValueError("reduction must be 'nearest' or 'weighted_quantile'")
    quantile = _finite_number(quantile, "quantile")
    floor = _finite_number(signal_floor, "signal_floor")
    if not 0 <= quantile <= 1 or floor <= 0:
        raise ValueError("quantile must be in [0,1] and signal_floor must be positive")
    effective = effective.clamp(min=minimum, max=maximum)
    ordered, order = torch.sort(effective, dim=-1, stable=True)
    inverse_square = (1 / (ordered.square() + 0.05)).clamp_min(floor)
    signal = torch.where(ordered >= maximum - 1e-4, floor, inverse_square)
    signal = signal * (1 - 0.35 * (ordered / maximum).square()).clamp_min(0.05)
    cumulative = signal.cumsum(dim=-1)
    # Using the last cumulative value makes q=1 select the final record even
    # when a differently ordered reduction would round the total upward.
    selected = (cumulative < quantile * cumulative[..., -1:]).sum(dim=-1)
    selected = selected.clamp_max(distance.shape[-1] - 1)
    return order.gather(-1, selected[..., None]).squeeze(-1)


def gather_subray(values: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
    """Gather scalar [B,H,W,S] or vector [B,H,W,S,3] records together."""
    if values.ndim == selection.ndim + 1:
        return values.gather(-1, selection[..., None]).squeeze(-1)
    if values.ndim == selection.ndim + 2 and values.shape[-1] == 3:
        index = selection[..., None, None].expand(*selection.shape, 1, 3)
        return values.gather(-2, index).squeeze(-2)
    raise ValueError("values must be scalar or vec3 subray records")


@lru_cache(maxsize=8)
def _bounds_constant(bounds: tuple, device: torch.device):
    tensor = torch.tensor(bounds, device=device, dtype=torch.float32)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return tensor, ready


def _bounds(bounds, device):
    try:
        intervals = tuple(tuple(_finite_number(x, "bounds") for x in axis) for axis in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must contain three increasing finite intervals") from exc
    if len(intervals) != 3 or any(len(axis) != 2 or axis[0] >= axis[1] for axis in intervals):
        raise ValueError("bounds must contain three increasing finite intervals")
    tensor, ready = _bounds_constant(intervals, device)
    stream = torch.cuda.current_stream(device)
    stream.wait_event(ready)
    tensor.record_stream(stream)
    return tensor


def range_by_sample(distance: torch.Tensor, identity: torch.Tensor,
                    origins: torch.Tensor, directions: torch.Tensor,
                    config: DepthConfig, bounds=None, *, return_known: bool = False):
    """Apply radial range and known-domain rules independently to each ray.

    A too-close closest hit stays NaN and continues to occlude farther surfaces.
    +inf establishes an empty sampled segment only when both its origin and
    max-range endpoint are in the known convex box. Without declared bounds,
    actual hits can be measured but a miss cannot establish known free space.
    This makes no claim about directions between the finite samples.
    """
    _check_samples(distance, identity)
    if not isinstance(config, DepthConfig):
        raise TypeError("config must be a DepthConfig")
    for name, vectors in (("origins", origins), ("directions", directions)):
        if (not isinstance(vectors, torch.Tensor) or vectors.device != distance.device
                or vectors.dtype != torch.float32):
            raise TypeError(f"{name} must be float32 on distance.device")
        try:
            shape = torch.broadcast_shapes(vectors.shape, (*distance.shape, 3))
        except RuntimeError as exc:
            raise ValueError(f"{name} must broadcast to [B,H,W,S,3]") from exc
        if shape != (*distance.shape, 3):
            raise ValueError(f"{name} must broadcast to [B,H,W,S,3]")
    hit = identity >= 0
    valid_ray = (torch.isfinite(origins).all(dim=-1) & torch.isfinite(directions).all(dim=-1)
                 & (directions.abs().amax(dim=-1) > 0))
    if bounds is None:
        known = valid_ray
        full = torch.zeros_like(hit)
    else:
        limits = _bounds(bounds, distance.device)
        origin_inside = (valid_ray & ((origins >= limits[:, 0])
                                     & (origins <= limits[:, 1])).all(dim=-1))
        end = origins + config.max_range_m * directions
        full = origin_inside & ((end >= limits[:, 0]) & (end <= limits[:, 1])).all(dim=-1)
        point = origins + distance[..., None] * directions
        # Ray geometry uses float32; admit a bounded roundoff budget for hit
        # endpoints, but require strict known coverage for an empty segment.
        tolerance = (32 * torch.finfo(torch.float32).eps
                     * origins.abs().amax(dim=-1).clamp_min(max(1.0, config.max_range_m)))[..., None]
        known = origin_inside & ((point >= limits[:, 0] - tolerance)
                                 & (point <= limits[:, 1] + tolerance)).all(dim=-1)
    valid = hit & known & (distance >= config.min_range_m) & (distance <= config.max_range_m)
    ranges = torch.where(valid, distance, math.nan)
    ranges = torch.where(~hit & full, math.inf, ranges)
    if return_known:
        # Distinguish an unknown ray from a known, too-close surface. Only the
        # former invalidates other subrays before selecting a measurement.
        available = torch.where(hit, known & torch.isfinite(distance) & (distance >= 0), full)
        return ranges, available
    return ranges


def clear_caches() -> None:
    """Release cached device constants after their queued stream consumers."""
    _bounds_constant.cache_clear()
