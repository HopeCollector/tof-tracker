"""Independent continuous *spatial* geometry reference for the ESDF experiment.

The boundary is a supplied list of exposed axis-aligned rectangles of a box
union. Occupancy comes from the original solid boxes (including the explicit
ground slab), never from a minimum of per-object signed distances. There is no
mesh BVH, voxel field, SdfLib, or production distance query in this module.
``from_geometry`` consumes the public arrays of experimental RealMapGeometry;
it does not establish the correctness of an arbitrary supplied boundary.

Queries support first position derivatives. At a medial tie or a nonsmooth
surface point the selected branch is deterministic, not a unique derivative;
``evaluate`` identifies competing near-nearest gradients. Numerical integration
below is a convergence study, not certified continuous collision detection.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.autograd.function import once_differentiable


def _bounds(value):
    result = np.array(value, dtype=np.float64, copy=True)
    if (result.shape != (2, 3) or not np.isfinite(result).all()
            or np.any(result[1] < result[0])
            or not np.isfinite(result[1] - result[0]).all()):
        raise ValueError("bounds must be finite [lo_xyz, hi_xyz] with lo <= hi")
    return result


class RectangleUnionOracle:
    """Float64 analytic rectangle closest points, union sign, and branch gradient.

    ``lower/upper`` are (R,3), ``axes/signs`` are (R,), ``solid_boxes`` is
    (B,2,3). The rectangles must already be the outward, exposed union boundary:
    original faces inside another solid must have been removed. These inputs
    are copied. Box/rectangle correspondence is a caller precondition, not a
    general mesh Boolean validation performed by this class.

    ``query`` accepts strided float32/float64 tensors (...,3), returns (...) in
    the input dtype/device, and connects the first position derivative. All
    geometry calculations use float64. ``evaluate`` returns detached float64
    diagnostics, plus boolean ``valid/inside/tie`` and original rectangle IDs.
    Invalid/out-of-bounds points produce NaN distance and zero backward.
    """

    def __init__(self, lower, upper, axes, signs, solid_boxes, *, bounds_m=None,
                 point_chunk=1024, rectangle_chunk=256, tie_atol=1e-10,
                 tie_rtol=1e-12, gradient_tie_atol=1e-7, rectangle_ids=None):
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        axes_array, signs_array = np.asarray(axes), np.asarray(signs)
        n = len(self.lower)
        if (self.lower.shape != (n, 3) or n == 0 or self.upper.shape != (n, 3)
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or axes_array.shape != (n,) or not np.isin(axes_array, [0, 1, 2]).all()
                or signs_array.shape != (n,) or not np.isin(signs_array, [-1, 1]).all()):
            raise ValueError("expected nonempty finite axis-aligned rectangle arrays")
        self.axes = axes_array.astype(np.int64, copy=True)
        self.signs = signs_array.astype(np.float64, copy=True)
        extent = self.upper - self.lower
        mask = np.arange(3)[None, :] == self.axes[:, None]
        if np.any(extent[mask] != 0) or np.any(extent[~mask] <= 0):
            raise ValueError("each rectangle must have exactly one flat declared axis")
        self.solid_boxes = np.array(solid_boxes, dtype=np.float64, copy=True)
        if (self.solid_boxes.ndim != 3 or self.solid_boxes.shape[1:] != (2, 3)
                or not np.isfinite(self.solid_boxes).all()
                or np.any(self.solid_boxes[:, 1] <= self.solid_boxes[:, 0])):
            raise ValueError("solid_boxes must have shape (B,2,3) and positive extent")
        self.bounds_m = None if bounds_m is None else _bounds(bounds_m)
        self.rectangle_ids = (np.arange(n, dtype=np.int64) if rectangle_ids is None
                              else np.asarray(rectangle_ids))
        if (self.rectangle_ids.shape != (n,) or self.rectangle_ids.dtype.kind not in "iu"
                or len(np.unique(self.rectangle_ids)) != n or np.any(self.rectangle_ids < 0)):
            raise ValueError("rectangle_ids must be distinct nonnegative integers")
        self.rectangle_ids = self.rectangle_ids.astype(np.int64, copy=True)
        if (isinstance(point_chunk, bool) or int(point_chunk) != point_chunk or point_chunk < 1
                or isinstance(rectangle_chunk, bool) or int(rectangle_chunk) != rectangle_chunk
                or rectangle_chunk < 1):
            raise ValueError("chunk sizes must be positive integers")
        self.point_chunk, self.rectangle_chunk = int(point_chunk), int(rectangle_chunk)
        if not all(np.isfinite(x) and x >= 0 for x in (tie_atol, tie_rtol, gradient_tie_atol)):
            raise ValueError("tie tolerances must be finite and nonnegative")
        self.tie_atol, self.tie_rtol = float(tie_atol), float(tie_rtol)
        self.gradient_tie_atol = float(gradient_tie_atol)
        self.restriction = None
        self._buffers = {}
        for array in (self.lower, self.upper, self.axes, self.signs, self.solid_boxes,
                      self.rectangle_ids, self.bounds_m):
            if array is not None:
                array.setflags(write=False)

    @classmethod
    def from_geometry(cls, geometry, **kwargs):
        """Use RealMapGeometry public arrays; default to its logical query domain.

        The ground extension is a deliberate solid convention. A caller wishing
        to query below the logical z=0 boundary must explicitly pass bounds_m
        inside the domain for which that slab was constructed.
        """
        kwargs.setdefault("bounds_m", geometry.logical_bounds)
        return cls(geometry.rectangle_lower, geometry.rectangle_upper,
                   geometry.rectangle_axes, geometry.rectangle_signs,
                   geometry.solid_boxes, **kwargs)

    def restricted(self, bounds_m):
        """Cull impossible nearest rectangles for a known whole-curve AABB.

        For center c and radius r, distance-to-boundary is <= d(c)+r everywhere
        in this box (the distance function is 1-Lipschitz). Rectangle-to-AABB
        distance is a lower bound. Thus lower_bound > d(c)+r cannot win. A
        float64 roundoff/near-tie padding is retained; this is algebraically
        safe culling, not a claim of machine-certified interval arithmetic.
        Occupancy only needs original solids that intersect the query box.
        """
        bounds = _bounds(bounds_m)
        if self.bounds_m is not None and (np.any(bounds[0] < self.bounds_m[0])
                                         or np.any(bounds[1] > self.bounds_m[1])):
            raise ValueError("restriction must lie inside the parent query bounds")
        center = bounds[0] + (bounds[1] - bounds[0]) * .5
        radius = np.linalg.norm((bounds[1] - bounds[0]) * .5)
        offset = center - np.clip(center, self.lower, self.upper)
        upper_bound = float(np.linalg.norm(offset, axis=1).min() + radius)
        separation = np.maximum(np.maximum(self.lower - bounds[1], bounds[0] - self.upper), 0)
        lower_bound = np.linalg.norm(separation, axis=1)
        scale = max(1., upper_bound, float(np.abs(bounds).max()),
                    float(np.abs(self.lower).max()), float(np.abs(self.upper).max()))
        padding = 64 * np.finfo(np.float64).eps * scale + self.tie_atol + self.tie_rtol * upper_bound
        keep = lower_bound <= upper_bound + padding
        boxes = self.solid_boxes
        occupied = np.all((boxes[:, 1] >= bounds[0]) & (boxes[:, 0] <= bounds[1]), axis=1)
        result = type(self)(self.lower[keep], self.upper[keep], self.axes[keep], self.signs[keep],
                            boxes[occupied], bounds_m=bounds, point_chunk=self.point_chunk,
                            rectangle_chunk=self.rectangle_chunk, tie_atol=self.tie_atol,
                            tie_rtol=self.tie_rtol, gradient_tie_atol=self.gradient_tie_atol,
                            rectangle_ids=self.rectangle_ids[keep])
        result.restriction = {"input_rectangles": len(self.lower), "kept_rectangles": int(keep.sum()),
                              "input_solids": len(boxes), "kept_solids": int(occupied.sum()),
                              "uniform_distance_upper_bound_m": upper_bound, "padding_m": padding}
        return result

    def _device_buffers(self, device):
        key = str(device)
        if key not in self._buffers:
            normals = np.eye(3, dtype=np.float64)[self.axes] * self.signs[:, None]
            self._buffers[key] = tuple(torch.tensor(a, device=device) for a in
                                      (self.lower, self.upper, normals, self.solid_boxes,
                                       self.rectangle_ids))
        return self._buffers[key]

    @staticmethod
    def _check_points(points):
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a Torch tensor")
        if (points.layout != torch.strided or points.dtype not in (torch.float32, torch.float64)
                or points.shape[-1:] != (3,)):
            raise ValueError("points must be strided float32/float64 (...,3)")

    @torch.no_grad()
    def evaluate(self, points, *, detect_ties=True):
        """Detached diagnostics; tie means different gradients among near minima.

        Coplanar fragments sharing the same closest point/gradient are not a
        medial tie. ``tie`` also marks a surface corner's competing normals.
        False only excludes alternatives within the stated tolerances; this
        numerical flag is not a formal differentiability certificate.
        """
        self._check_points(points)
        shape = points.shape[:-1]
        p = points.detach().reshape(-1, 3).to(torch.float64)
        lo, hi, normals, boxes, original_ids = self._device_buffers(p.device)
        valid = torch.isfinite(p).all(dim=-1)
        if self.bounds_m is not None:
            bounds = torch.tensor(self.bounds_m, device=p.device)
            valid &= ((p >= bounds[0]) & (p <= bounds[1])).all(dim=-1)
        distance = torch.full((len(p),), torch.nan, dtype=p.dtype, device=p.device)
        gradient = torch.zeros_like(p)
        closest = torch.full_like(p, torch.nan)
        ids = torch.full((len(p),), -1, dtype=torch.int64, device=p.device)
        inside_result, ties = torch.zeros_like(valid), torch.zeros_like(valid)
        selected = torch.nonzero(valid, as_tuple=False).flatten()
        for start in range(0, len(selected), self.point_chunk):
            rows = selected[start:start + self.point_chunk]
            point = p[rows]
            best = torch.full((len(point),), torch.inf, dtype=p.dtype, device=p.device)
            winner = torch.zeros(len(point), dtype=torch.int64, device=p.device)
            for begin in range(0, len(lo), self.rectangle_chunk):
                end = begin + self.rectangle_chunk
                q = torch.maximum(torch.minimum(point[:, None], hi[None, begin:end]), lo[None, begin:end])
                dsq = (point[:, None] - q).square().sum(dim=-1)
                local, index = dsq.min(dim=1)
                replace = local < best
                winner = torch.where(replace, begin + index, winner)
                best = torch.minimum(best, local)
            inside = torch.zeros(len(point), dtype=torch.bool, device=p.device)
            for begin in range(0, len(boxes), self.rectangle_chunk):
                solid = boxes[begin:begin + self.rectangle_chunk]
                inside |= ((point[:, None] >= solid[None, :, 0])
                           & (point[:, None] <= solid[None, :, 1])).all(dim=-1).any(dim=-1)
            sign = torch.where(inside, -1., 1.)
            q = torch.maximum(torch.minimum(point, hi[winner]), lo[winner])
            radius = best.sqrt()
            grad = sign[:, None] * (point - q) / torch.where(radius > 0, radius, 1.)[:, None]
            grad = torch.where((radius > 0)[:, None], grad, normals[winner])
            finite = torch.isfinite(radius) & torch.isfinite(grad).all(dim=-1)
            # Huge finite coordinates can overflow squared norms: never invent
            # a valid distance or gradient in that unsupported numerical range.
            distance[rows] = torch.where(finite, sign * radius, torch.nan)
            gradient[rows] = torch.where(finite[:, None], grad, 0.)
            closest[rows] = torch.where(finite[:, None], q, torch.nan)
            ids[rows] = torch.where(finite, original_ids[winner], -1)
            inside_result[rows] = inside & finite
            valid[rows] &= finite
            if detect_ties:
                ambiguous = torch.zeros(len(point), dtype=torch.bool, device=p.device)
                tolerance = self.tie_atol + self.tie_rtol * radius
                for begin in range(0, len(lo), self.rectangle_chunk):
                    end = begin + self.rectangle_chunk
                    candidate_q = torch.maximum(torch.minimum(point[:, None], hi[None, begin:end]),
                                                lo[None, begin:end])
                    residual = point[:, None] - candidate_q
                    candidate_radius = residual.square().sum(dim=-1).sqrt()
                    candidate_grad = sign[:, None, None] * residual / torch.where(
                        candidate_radius > 0, candidate_radius, 1.)[..., None]
                    candidate_grad = torch.where((candidate_radius > 0)[..., None], candidate_grad,
                                                 normals[None, begin:end])
                    nearby = (candidate_radius - radius[:, None]).abs() <= tolerance[:, None]
                    different = (candidate_grad - grad[:, None]).norm(dim=-1) > self.gradient_tie_atol
                    ambiguous |= (nearby & different).any(dim=-1)
                ties[rows] = ambiguous & finite
        return {"distance": distance.reshape(shape), "gradient": gradient.reshape(*shape, 3),
                "closest_point": closest.reshape(*shape, 3), "nearest_rectangle": ids.reshape(shape),
                "inside": inside_result.reshape(shape), "tie": ties.reshape(shape),
                "valid": valid.reshape(shape)}

    def query(self, points):
        """Signed distances with first position derivatives, no Hessian contract."""
        self._check_points(points)
        return _SignedDistance.apply(points, self)

    __call__ = query


class _SignedDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, oracle):
        result = oracle.evaluate(points, detect_ties=False)
        ctx.save_for_backward(result["gradient"].to(points.dtype), result["valid"])
        return result["distance"].to(points.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        gradient, valid = ctx.saved_tensors
        # Mask the upstream first: 0 * NaN is NaN for an unmasked invalid loss.
        upstream = torch.where(valid, upstream, 0.)
        return upstream[..., None] * gradient, None


def reference_penalty(distance):
    """Common Tracker penalty exp(-(d - 1.2 m) / 0.6 m)."""
    return torch.exp(-(distance - 1.2) / .6)


def tracker_fractions(*, dtype=torch.float64, device="cpu", count=30):
    """Right endpoints k/count, k=1..count; there is no u=0 sample."""
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("count must be a positive integer")
    return torch.arange(1, int(count) + 1, dtype=dtype, device=device) / count


def sampled_mean(oracle, positions, penalty=reference_penalty):
    """Finite sample mean over final time axis; no duration multiplier.

    positions is (...,N,3), with N>0. The caller supplies exactly the same
    positions for both fields and this geometry reference. For Tracker's main
    experiment use tracker_fractions() and fixed T=10/6 seconds.
    """
    if positions.ndim < 2 or positions.shape[-2] == 0:
        raise ValueError("positions must be (...,N,3) with N > 0")
    return penalty(oracle.query(positions)).mean(dim=-1)


def numerical_integral(oracle: RectangleUnionOracle, position_at: Callable,
                       penalty=reference_penalty, *, parameters: torch.Tensor,
                       duration=None, order=8, initial_panels=1, max_panels=1024,
                       atol=1e-8, rtol=1e-6, grad_atol=1e-8, grad_rtol=1e-5,
                       required_refinements=2):
    """Composite Gauss-Legendre reference with value AND gradient refinement.

    position_at(u) must freshly build (...,len(u),3) from ``parameters``.
    parameters is a float64 requires_grad tensor (e.g. Bx9 terminal p/v/a).
    With duration=None the objective is integral_0^1 penalty(d(p(u))) du,
    a normalized time average. Supplying T gives the physical time integral
    T*integral_0^1; a Tensor T can be connected to parameters by the caller.

    ``parameter_gradient`` is d(value.sum())/d(parameters), detached. ``value``
    retains the final graph. Gradients of separate batch rows therefore require
    independent per-row parameters; this is not a full batched Jacobian.
    Convergence means successive sampled quadratures agree within tolerances,
    not rigorous error bounds; a sufficiently narrow feature can be missed.
    No minimization over continuous time or collision guarantee is performed.
    """
    if (not isinstance(parameters, torch.Tensor) or parameters.dtype != torch.float64
            or not parameters.requires_grad or not torch.isfinite(parameters).all()):
        raise ValueError("parameters must be a finite float64 requires_grad tensor")
    for name, number in (("order", order), ("initial_panels", initial_panels),
                         ("max_panels", max_panels), ("required_refinements", required_refinements)):
        if isinstance(number, bool) or int(number) != number or number < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_panels < initial_panels:
        raise ValueError("max_panels must be >= initial_panels")
    if not all(np.isfinite(x) and x >= 0 for x in (atol, rtol, grad_atol, grad_rtol)):
        raise ValueError("convergence tolerances must be finite and nonnegative")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes = torch.tensor(nodes, dtype=parameters.dtype, device=parameters.device)
    weights = torch.tensor(weights, dtype=parameters.dtype, device=parameters.device)
    if duration is not None:
        duration = torch.as_tensor(duration, dtype=parameters.dtype, device=parameters.device)
        if not bool((torch.isfinite(duration) & (duration > 0)).all()):
            raise ValueError("duration must be finite and positive")
    history, previous, consecutive, panels = [], None, 0, int(initial_panels)
    value_delta, gradient_delta = None, None
    converged = False
    while True:
        panel = torch.arange(panels, dtype=parameters.dtype, device=parameters.device)
        u = ((panel[:, None] + (nodes[None] + 1) * .5) / panels).flatten()
        w = (weights[None].expand(panels, -1) / (2 * panels)).flatten()
        positions = position_at(u)
        if positions.shape[-2:] != (len(u), 3):
            raise ValueError("position_at(u) must return (...,len(u),3)")
        value = (penalty(oracle.query(positions)) * w).sum(dim=-1)
        if duration is not None:
            if torch.broadcast_shapes(value.shape, duration.shape) != value.shape:
                raise ValueError("duration must broadcast to, not expand, the trajectory batch")
            value = duration * value
        gradient = torch.autograd.grad(value.sum(), parameters, retain_graph=True)[0]
        v, g = value.detach(), gradient.detach()
        finite = bool(torch.isfinite(v).all() and torch.isfinite(g).all())
        record = {"panels": panels, "evaluations": len(u), "finite": finite,
                  "value_delta_max": None, "gradient_delta_max": None, "within_tolerance": False}
        if previous is not None and finite:
            pv, pg = previous
            dv, dg = (v - pv).abs(), (g - pg).abs()
            value_delta, gradient_delta = dv, dg
            record.update(value_delta_max=float(dv.max()), gradient_delta_max=float(dg.max()),
                          within_tolerance=bool((dv <= atol + rtol * v.abs()).all()
                                                and (dg <= grad_atol + grad_rtol * g.abs()).all()))
        consecutive = consecutive + 1 if record["within_tolerance"] else 0
        history.append(record)
        converged = finite and consecutive >= required_refinements
        if converged or not finite or panels >= max_panels:
            break
        previous, panels = (v, g), min(2 * panels, int(max_panels))
    return {"value": value, "parameter_gradient": g, "converged": converged,
            "history": history, "objective": "normalized_time_integral" if duration is None
            else "physical_time_integral", "total_evaluations": sum(row["evaluations"] for row in history),
            "value_delta": value_delta, "gradient_delta": gradient_delta,
            "certified": False}
