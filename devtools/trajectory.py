"""Differentiable boundary-value polynomials for training and development.

No frame conversion, environment query, controller or optimizer lives here.
All translation data in one call uses the same fixed frame (normally B_ref).
Quintic boundary equations reference YOPO-tracker's Poly5Solver; see
docs/origins.md and tools/mapgen/third_party_notices/YOPO-tracker-LICENSE.
Yaw's cubic Hermite polynomial is derived from its four boundary conditions.
"""

from dataclasses import dataclass
from math import factorial, pi
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryState:
    """Explicit translation and yaw boundary values, or sampled states.

    position/velocity/acceleration have shape (*batch, 3), in m, m/s, m/s².
    yaw/yaw_rate have shape (*batch), in rad and rad/s. Initial yaw and rate
    are required; they are not inferred from position, velocity or a pose.
    Fitting validates tensors and broadcasts their batch dimensions.
    Sampling adds a final sample dimension before the vector dimension.
    """

    position: Tensor
    velocity: Tensor
    acceleration: Tensor
    yaw: Tensor
    yaw_rate: Tensor


def _tensor(name: str, value: Tensor, reference: Tensor | None = None) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a strided float32 or float64 tensor")
    if reference is not None and (value.dtype != reference.dtype or value.device != reference.device):
        raise ValueError(f"{name} must have the same dtype and device as position")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _broadcast_shape(*shapes: torch.Size) -> torch.Size:
    try:
        return torch.broadcast_shapes(*shapes)
    except RuntimeError as error:
        raise ValueError("batch dimensions are not broadcastable") from error


def _evaluate(coefficients: Tensor, u: Tensor, derivative: int = 0) -> Tensor:
    """Horner evaluation, coefficients (*batch, powers, channels), u (*batch,S)."""
    degree = coefficients.shape[-2] - 1
    result = coefficients[..., degree, :].unsqueeze(-2) * (
        factorial(degree) // factorial(degree - derivative)
    )
    for power in range(degree - 1, derivative - 1, -1):
        multiplier = factorial(power) // factorial(power - derivative)
        result = result * u.unsqueeze(-1) + coefficients[..., power, :].unsqueeze(-2) * multiplier
    return result


@dataclass(frozen=True)
class PolynomialTrajectory:
    """Polynomials in u=t/duration, with coefficients in ascending powers.

    Normally created by fit_trajectory. position_coefficients is (*batch,6,3),
    yaw_coefficients is (*batch,4), duration is (*batch) seconds. All have an
    identical dtype/device and exactly the same batch shape. Tensor storage
    remains mutable: callers must treat fields and source tensors as read-only
    while using the trajectory or its autograd graph.
    """

    position_coefficients: Tensor
    yaw_coefficients: Tensor
    duration: Tensor

    def __post_init__(self) -> None:
        _tensor("position_coefficients", self.position_coefficients)
        _tensor("yaw_coefficients", self.yaw_coefficients, self.position_coefficients)
        _tensor("duration", self.duration, self.position_coefficients)
        if self.position_coefficients.shape[-2:] != (6, 3):
            raise ValueError("position_coefficients must have shape (*batch, 6, 3)")
        if self.yaw_coefficients.shape[-1:] != (4,):
            raise ValueError("yaw_coefficients must have shape (*batch, 4)")
        if (self.position_coefficients.shape[:-2] != self.duration.shape
                or self.yaw_coefficients.shape[:-1] != self.duration.shape):
            raise ValueError("coefficient and duration batch shapes must match exactly")
        if not bool((self.duration > 0).all()):
            raise ValueError("duration must be strictly positive")

    def sample(self, time: Tensor) -> TrajectoryState:
        """Sample at elapsed seconds in [0,T], with no clipping/extrapolation.

        time may be scalar (one sample) or (*time_batch,S). time_batch and
        trajectory batch dimensions broadcast using normal Torch rules. A
        common grid is (S,); one distinct time per trajectory is (*batch,1).
        Translation outputs are (*broadcast_batch,S,3), yaw/rate outputs are
        (*broadcast_batch,S). Derivatives are with respect to physical seconds.
        Gradients to time, duration and all boundary values are preserved.
        """
        _tensor("time", time, self.position_coefficients)
        if time.ndim == 0:
            time = time.unsqueeze(0)
        _broadcast_shape(self.duration.shape, time.shape[:-1])
        duration = self.duration.unsqueeze(-1)
        if not bool(((time >= 0) & (time <= duration)).all()):
            raise ValueError("time must lie in [0, duration] for every trajectory")
        u = time / duration
        coefficients = self.position_coefficients
        position = _evaluate(coefficients, u)
        velocity = _evaluate(coefficients, u, 1) / duration.unsqueeze(-1)
        acceleration = _evaluate(coefficients, u, 2) / duration.unsqueeze(-1).square()
        yaw = _evaluate(self.yaw_coefficients.unsqueeze(-1), u).squeeze(-1)
        yaw_rate = (_evaluate(self.yaw_coefficients.unsqueeze(-1), u, 1).squeeze(-1)
                    / duration)
        result = TrajectoryState(position, velocity, acceleration, yaw, yaw_rate)
        # Positive finite T can still exceed a dtype's numerical range. Fail
        # explicitly instead of silently returning overflow or clipping T.
        for name in TrajectoryState.__dataclass_fields__:
            _tensor(f"sample.{name}", getattr(result, name))
        return result


def fit_trajectory(
    initial: TrajectoryState,
    terminal: TrajectoryState,
    duration: Tensor,
    *,
    yaw_mode: Literal["unwrapped", "shortest"] = "unwrapped",
) -> PolynomialTrajectory:
    """Fit a 3D quintic (p/v/a) and a yaw cubic (yaw/yaw_rate).

    Every input must be a finite, strided Torch float32/float64 tensor with
    the same dtype/device. No implicit casts, default initial state, frame
    conversions, thresholds or constraints are applied. duration > 0 seconds;
    all input batch dimensions broadcast, with a trailing 3 only on vectors.

    unwrapped uses both yaw values literally, preserving intentional turns.
    shortest replaces terminal yaw by initial yaw + delta, delta in [-pi,pi).
    An exact half-turn chooses -pi. The chosen branch is differentiable away
    from odd-pi differences; branch selection is discontinuous there. Sampled
    yaw remains continuous/unwrapped, including after crossing +/-pi. This
    scalar yaw polynomial does not define a full 3D attitude or body yaw axis.
    """
    if not isinstance(initial, TrajectoryState) or not isinstance(terminal, TrajectoryState):
        raise TypeError("initial and terminal must be TrajectoryState instances")
    if yaw_mode not in ("unwrapped", "shortest"):
        raise ValueError("yaw_mode must be 'unwrapped' or 'shortest'")

    reference = initial.position
    _tensor("initial.position", reference)
    batch_shapes = []
    for prefix, state in (("initial", initial), ("terminal", terminal)):
        for name in TrajectoryState.__dataclass_fields__:
            value = getattr(state, name)
            _tensor(f"{prefix}.{name}", value, reference)
            if name in ("position", "velocity", "acceleration"):
                if value.shape[-1:] != (3,):
                    raise ValueError(f"{prefix}.{name} must have shape (*batch, 3)")
                batch_shapes.append(value.shape[:-1])
            else:
                batch_shapes.append(value.shape)
    _tensor("duration", duration, reference)
    if not bool((duration > 0).all()):
        raise ValueError("duration must be strictly positive")
    batch = _broadcast_shape(*batch_shapes, duration.shape)
    duration = duration.expand(batch)
    t = duration.unsqueeze(-1)
    p0, v0, a0 = (getattr(initial, name).expand(*batch, 3)
                  for name in ("position", "velocity", "acceleration"))
    p1, v1, a1 = (getattr(terminal, name).expand(*batch, 3)
                  for name in ("position", "velocity", "acceleration"))
    c0, c1, c2 = p0, t * v0, 0.5 * t.square() * a0
    rp = p1 - c0 - c1 - c2
    rv = t * v1 - c1 - 2 * c2
    ra = t.square() * a1 - 2 * c2
    position_coefficients = torch.stack((
        c0, c1, c2,
        10 * rp - 4 * rv + 0.5 * ra,
        -15 * rp + 7 * rv - ra,
        6 * rp - 3 * rv + 0.5 * ra,
    ), dim=-2)

    y0, w0 = initial.yaw.expand(batch), initial.yaw_rate.expand(batch)
    y1, w1 = terminal.yaw.expand(batch), terminal.yaw_rate.expand(batch)
    delta = y1 - y0
    if yaw_mode == "shortest":
        delta = torch.remainder(delta + pi, 2 * pi) - pi
    d0, d1 = duration * w0, duration * w1
    yaw_coefficients = torch.stack((y0, d0, 3 * delta - 2 * d0 - d1,
                                    -2 * delta + d0 + d1), dim=-1)
    return PolynomialTrajectory(position_coefficients, yaw_coefficients, duration)
