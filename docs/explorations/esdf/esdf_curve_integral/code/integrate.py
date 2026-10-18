"""Experimental cell-split quadrature of a batch of quintic curve penalties.

The objective is integral_0^1 exp(-(field(p(u))-d0)/r) du, a time average.
There is no explicit duration factor. Coefficients use u=t/T; any dependence on
physical duration comes through the caller's boundary-to-coefficient map.

Face roots adapt the numerical splitter from esdf_grid_eval: derivative roots
form monotone intervals and simple crossings are bisected. Stationary roots
within 1e-10 of u=0/1 are classified as endpoints. This is neither
analytic integration of an exponential nor certified continuous collision
checking. For transverse crossings in a C0 field, moving-boundary terms cancel
between adjacent exact integrals. Detached partitions therefore give the
continuous objective's first derivative in the quadrature limit. At finite GL
order this derivative need not exactly equal differentiating the numerical
quadrature rule including moving nodes; re-split finite differences test that
discrepancy. Root degeneracies and out-of-domain curves are rejected.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from docs.explorations.esdf.esdf_grid_eval.code.path_integrals import _extrema, _real_roots


class CurveSplitError(ValueError):
    """A failed curve makes the entire batch fail; no curves are skipped."""

    def __init__(self, curve_index, cause):
        self.curve_index = int(curve_index)
        super().__init__(f"curve {curve_index} could not be split: {cause}")


def _geometry_signature(field):
    return (tuple(field.shape), tuple(field.sample_origin_m), float(field.spacing_m),
            tuple(tuple(row) for row in field.bounds_m))


def _check_coefficients(coefficients):
    if (not isinstance(coefficients, torch.Tensor) or coefficients.layout != torch.strided
            or coefficients.dtype not in (torch.float32, torch.float64)
            or coefficients.ndim != 3 or coefficients.shape[1:] != (6, 3)
            or coefficients.shape[0] == 0):
        raise ValueError("coefficients must be a nonempty strided float32/64 tensor [B,6,3]")


def _split_curve_endpoint_aware(field, coefficients):
    """Adapt the earlier experiment's split_curve without editing that source.

    Reuse the earlier derivative/extrema solvers, tolerances and final interval
    coverage checks. Replace the quintic companion-matrix roots by bisection
    between derivative roots: an endpoint double root otherwise splits into
    spurious internal roots. Endpoint value normalization is limited to a
    recorded coefficient-scaled roundoff tolerance. Interior near-tangencies
    still fail. None of these numerical decisions is a certified classifier.
    """
    c = coefficients.detach().cpu().numpy()
    if c.shape != (6, 3) or not np.isfinite(c).all():
        raise ValueError("expected one finite quintic with coefficients (6,3)")
    origin = np.asarray(field.sample_origin_m, dtype=np.float64)
    bounds = np.asarray(field.bounds_m, dtype=np.float64)
    shape, h = np.asarray(field.shape), float(field.spacing_m)
    endpoint_tolerance = 1e-10
    roots, events, on_faces, endpoint_stationary, endpoint_normalizations = [0., 1.], [], [], [], []
    for axis in range(3):
        minimum, maximum = _extrema(c[:, axis])
        if minimum < bounds[0, axis]-1e-10 or maximum > bounds[1, axis]+1e-10:
            raise ValueError("curve leaves the declared query domain")
        for index in range(1, int(shape[axis])-1):
            level = origin[axis]+index*h
            if level < minimum-1e-10 or level > maximum+1e-10:
                continue
            equation = c[:, axis].copy()
            equation[0] -= level
            if np.max(np.abs(equation)) <= 1e-13:
                on_faces.append({"axis": axis, "index": index})
                continue
            derivative = np.arange(1, 6)*equation[1:]
            stationary_roots = _real_roots(derivative)
            for stationary in stationary_roots:
                residual = abs(np.polynomial.polynomial.polyval(stationary, equation))
                if residual >= 1e-9:
                    continue
                endpoint = 0. if abs(stationary) <= endpoint_tolerance else (
                    1. if abs(stationary-1.) <= endpoint_tolerance else None)
                if endpoint is not None:
                    endpoint_stationary.append({"axis": axis, "index": index,
                                                "computed_u": stationary, "endpoint_u": endpoint,
                                                "face_residual_m": float(residual)})
                elif 0. < stationary < 1.:
                    raise ValueError(f"internal near-tangent grid-face contact at axis={axis}, "
                                     f"index={index}, u={stationary:.17g}; rejected")
            knots = [0.] + sorted(r for r in stationary_roots
                                  if endpoint_tolerance < r < 1.-endpoint_tolerance) + [1.]
            values = np.polynomial.polynomial.polyval(knots, equation)
            roundoff = 32.*np.finfo(np.float64).eps*max(1., float(np.abs(equation).sum()))
            for endpoint_index in (0, len(knots)-1):
                if abs(values[endpoint_index]) <= roundoff:
                    endpoint_normalizations.append({"axis": axis, "index": index,
                                                    "u": knots[endpoint_index],
                                                    "original_residual_m": float(values[endpoint_index]),
                                                    "roundoff_tolerance_m": roundoff})
                    values[endpoint_index] = 0.
            crossing_roots = []
            for left, right, fleft, fright in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
                if fleft == 0. or fright == 0. or np.signbit(fleft) == np.signbit(fright):
                    continue
                # Derivative roots bracket a monotone polynomial interval.
                # Bisection never invokes the ill-conditioned repeated-root
                # companion eigensolve for the quintic face equation.
                for _ in range(64):
                    middle = (left+right)/2.
                    fmiddle = np.polynomial.polynomial.polyval(middle, equation)
                    if fmiddle == 0. or right-left <= 8.*np.finfo(np.float64).eps:
                        left = right = middle
                        break
                    if np.signbit(fmiddle) == np.signbit(fleft):
                        left, fleft = middle, fmiddle
                    else:
                        right = middle
                crossing_roots.append((left+right)/2.)
            for root in crossing_roots:
                if not endpoint_tolerance < root < 1.-endpoint_tolerance:
                    continue
                residual = abs(np.polynomial.polynomial.polyval(root, equation))
                slope = abs(np.polynomial.polynomial.polyval(root, derivative))
                if residual > 1e-8 or slope < 1e-7:
                    raise ValueError("unreliable or nearly multiple internal face root")
                roots.append(root)
                events.append({"u": root, "axis": axis, "index": index,
                               "kind": "transverse_crossing", "root_residual_m": float(residual),
                               "slope_m_per_u": float(slope)})
    roots.sort()
    unique = [roots[0]]
    for root in roots[1:]:
        if root-unique[-1] > 1e-10:
            unique.append(root)
    pieces = []
    for a, b in zip(unique[:-1], unique[1:]):
        midpoint = np.polynomial.polynomial.polyval((a+b)/2., c)
        cell = np.floor((midpoint-origin)/h).astype(np.int64)
        cell = np.minimum(cell, shape-3)
        if np.any(cell < 1) or np.any(cell > shape-3):
            raise ValueError("cell has no complete 4^3 public sample stencil")
        for axis in range(3):
            minimum, maximum = _extrema(c[:, axis], a, b)
            if (minimum < origin[axis]+cell[axis]*h-1e-8
                    or maximum > origin[axis]+(cell[axis]+1)*h+1e-8):
                raise ValueError("a segment crosses a cell face: root splitting failed")
        pieces.append((a, b, tuple(int(i) for i in cell)))
    return pieces, {"face_events": events, "identically_on_faces": on_faces,
                    "endpoint_stationary_contacts": endpoint_stationary,
                    "endpoint_value_normalizations": endpoint_normalizations,
                    "state_counts": {"transverse_crossings": len(events),
                                     "endpoint_stationary_contacts": len(endpoint_stationary),
                                     "endpoint_value_normalizations": len(endpoint_normalizations),
                                     "whole_coordinate_on_faces": len(on_faces),
                                     "accepted_internal_tangencies": 0},
                    "coverage_check": "coordinate extrema via numerical degree-four roots; not certified",
                    "root_method": "numerical derivative roots, monotone brackets, bisection; internal near-tangencies rejected",
                    "endpoint_time_tolerance": endpoint_tolerance,
                    "near_tangency_policy": "reject internal stationary face residual < 1e-9 m; classify endpoint within 1e-10"}


@dataclass(frozen=True)
class CurveSegments:
    """CPU partition data. Arrays and diagnostics must be treated as read-only.

    ``intervals`` is [S,2], ``cells_xyz`` [S,3], ``curve_indices`` [S].
    ``coefficients`` stores the detached represented input values as float64;
    ``dtype`` retains their original Torch dtype. Diagnostics are per curve.
    """

    intervals: np.ndarray
    cells_xyz: np.ndarray
    curve_indices: np.ndarray
    coefficients: np.ndarray
    dtype: torch.dtype
    geometry_signature: tuple
    diagnostics: tuple

    @property
    def curve_count(self):
        return len(self.coefficients)

    @property
    def segment_count(self):
        return len(self.intervals)


@dataclass(frozen=True)
class QuadraturePlan:
    """Device-resident nodes/weights for one unchanged represented curve batch.

    Tensors must be treated as read-only. ``u`` and ``weights`` are [S,order];
    all interval scale factors are included in weights. Changing coefficients
    requires splitting again, even when the caller expects the same cells.
    """

    u: torch.Tensor
    weights: torch.Tensor
    curve_indices: torch.Tensor
    cells_xyz: torch.Tensor
    coefficients: torch.Tensor
    geometry_signature: tuple
    order: int

    @property
    def curve_count(self):
        return len(self.coefficients)

    @property
    def segment_count(self):
        return len(self.curve_indices)

    @property
    def query_count(self):
        return self.u.numel()


def split_batch(field, coefficients):
    """Detach/copy once to CPU float64, then split every curve numerically.

    Timing this call includes input validation, a possible GPU-to-CPU transfer,
    numerical face roots and extrema checks. It does not integrate the field.
    No public node snapshot or coefficient interpolation construction is used.
    """
    _check_coefficients(coefficients)
    cpu = coefficients.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
    intervals, cells, indices, diagnostics = [], [], [], []
    for curve_index, c in enumerate(cpu):
        try:
            pieces, diagnostic = _split_curve_endpoint_aware(field, c)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            raise CurveSplitError(curve_index, error) from error
        intervals.extend((a, b) for a, b, _ in pieces)
        cells.extend(cell for _, _, cell in pieces)
        indices.extend([curve_index] * len(pieces))
        diagnostics.append(diagnostic)
    arrays = (np.asarray(intervals, dtype=np.float64), np.asarray(cells, dtype=np.int64),
              np.asarray(indices, dtype=np.int64), cpu.numpy().copy())
    for array in arrays:
        array.setflags(write=False)
    return CurveSegments(*arrays, coefficients.dtype, _geometry_signature(field), tuple(diagnostics))


def prepare_quadrature(segments, *, order=16, device="cpu", dtype=None):
    """Build/transfer flattened-batch GL4/8/16/32 quadrature separately from roots.

    The coefficient dtype must match the original split input. To benchmark or
    validate float32, first split the actual float32 coefficients, rather than
    silently rounding a partition computed for a different represented curve.
    """
    if not isinstance(segments, CurveSegments):
        raise TypeError("segments must come from split_batch")
    if isinstance(order, bool) or order not in (4, 8, 16, 32):
        raise ValueError("order must be one of 4, 8, 16, 32")
    if dtype is None:
        dtype = segments.dtype
    if dtype != segments.dtype:
        raise ValueError("quadrature dtype must match split input; split the converted coefficients again")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    a, b = segments.intervals.T
    u = a[:, None] + (b-a)[:, None] * ((nodes+1.) / 2.)
    w = (b-a)[:, None] * (weights / 2.)
    return QuadraturePlan(torch.tensor(u, dtype=dtype, device=device),
                          torch.tensor(w, dtype=dtype, device=device),
                          torch.tensor(segments.curve_indices, dtype=torch.int64, device=device),
                          torch.tensor(segments.cells_xyz, dtype=torch.int64, device=device),
                          torch.tensor(segments.coefficients, dtype=dtype, device=device),
                          segments.geometry_signature, int(order))


def integrate_segments(field, coefficients, plan, *, d0=1.2, r=.6):
    """Return one differentiable normalized-time mean per curve, shape [B].

    All segments are evaluated in one batched Horner/public-query operation.
    Plan setup/transfers are excluded, but coefficient/field checks and finite
    result checks are included. CUDA checks synchronize; do not report this
    function's timing as an unchecked kernel timing. The default query uses
    the public field's actual cell lookup, not forced extrapolated cell values.
    """
    _check_coefficients(coefficients)
    if not isinstance(plan, QuadraturePlan):
        raise TypeError("plan must come from prepare_quadrature")
    if (coefficients.device != plan.coefficients.device
            or coefficients.dtype != plan.coefficients.dtype
            or coefficients.shape != plan.coefficients.shape
            or not torch.equal(coefficients.detach(), plan.coefficients)):
        raise ValueError("stale or incompatible quadrature plan; split the current coefficients again")
    if _geometry_signature(field) != plan.geometry_signature or field.device != coefficients.device:
        raise ValueError("field geometry/device does not match the quadrature plan")
    if not math.isfinite(d0) or not math.isfinite(r) or r <= 0:
        raise ValueError("d0 must be finite and r finite and positive")
    c = coefficients[plan.curve_indices]
    point = c[:, 5, None, :]
    for power in range(4, -1, -1):
        point = point * plan.u[..., None] + c[:, power, None, :]
    distance = field.query(point)
    value = torch.exp(-(distance-d0) / r)
    segment_values = (value * plan.weights).sum(-1)
    result = coefficients.new_zeros(plan.curve_count).index_add(0, plan.curve_indices, segment_values)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite field/penalty/integral; the batch is not partially accepted")
    return result


def integrate_curves(field, coefficients, *, order=16, d0=1.2, r=.6):
    """Split, prepare and integrate anew; includes all stages, with autograd.

    Use this path for finite differences and costs at updated coefficients.
    Use the three separate functions to report split, setup and integration
    costs; caching a plan is legitimate only for unchanged curve values.
    """
    segments = split_batch(field, coefficients)
    plan = prepare_quadrature(segments, order=order, device=coefficients.device)
    return integrate_segments(field, coefficients, plan, d0=d0, r=r)
