"""Small CPU experiment: split a quintic at grid faces and integrate polynomials.

This is not a certified root isolator, a collision checker, or a public ESDF API.
Cell polynomial coefficients use Bernstein form to avoid expanding degree-90
integrands in global power coordinates. All field access is public.
"""

from functools import lru_cache
from math import comb

import numpy as np
import torch


def _real_roots(coefficients):
    coefficients = np.trim_zeros(np.asarray(coefficients, dtype=np.float64), "b")
    if len(coefficients) <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(coefficients)
    return sorted(float(r.real) for r in roots if abs(r.imag) <= 1e-8)


def _extrema(coefficients, lo=0.0, hi=1.0):
    derivative = np.arange(1, len(coefficients)) * coefficients[1:]
    times = [lo, hi] + [r for r in _real_roots(derivative) if lo < r < hi]
    values = np.polynomial.polynomial.polyval(times, coefficients)
    return float(values.min()), float(values.max())


def split_curve(field, coefficients):
    """Find numerical face roots, diagnose tangencies, and check cell extrema.

    Exact-on-face coordinate polynomials are allowed. An isolated tangency or
    nearly multiple crossing is rejected rather than silently omitted. The
    degree-four extrema check is an independent numerical coverage check, not
    an interval-arithmetic/root-isolation certificate.
    """
    c = coefficients.detach().cpu().numpy()
    if c.shape != (6, 3) or not np.isfinite(c).all():
        raise ValueError("expected one finite quintic with coefficients (6,3)")
    origin = np.asarray(field.sample_origin_m, dtype=np.float64)
    bounds = np.asarray(field.bounds_m, dtype=np.float64)
    h = float(field.spacing_m)
    shape = np.asarray(field.shape)
    roots, events, on_faces = [0.0, 1.0], [], []
    for axis in range(3):
        minimum, maximum = _extrema(c[:, axis])
        if minimum < bounds[0, axis] - 1e-10 or maximum > bounds[1, axis] + 1e-10:
            raise ValueError("curve leaves the declared query domain")
        for index in range(1, int(shape[axis]) - 1):
            level = origin[axis] + index * h
            if level < minimum - 1e-10 or level > maximum + 1e-10:
                continue
            equation = c[:, axis].copy()
            equation[0] -= level
            if np.max(np.abs(equation)) <= 1e-13:
                on_faces.append({"axis": axis, "index": index})
                continue
            derivative = np.arange(1, 6) * equation[1:]
            for stationary in _real_roots(derivative):
                if 0 < stationary < 1 and abs(np.polynomial.polynomial.polyval(stationary, equation)) < 1e-9:
                    raise ValueError("near-tangent grid-face contact; this experiment rejects it")
            for r in _real_roots(equation):
                if not 1e-10 < r < 1 - 1e-10:
                    continue
                residual = abs(np.polynomial.polynomial.polyval(r, equation))
                slope = abs(np.polynomial.polynomial.polyval(r, derivative))
                if residual > 1e-8 or slope < 1e-7:
                    raise ValueError("unreliable or nearly multiple face root")
                roots.append(r)
                events.append({"u": r, "axis": axis, "index": index,
                               "root_residual_m": residual, "slope_m_per_u": slope})
    roots.sort()
    unique = [roots[0]]
    for r in roots[1:]:
        if r - unique[-1] > 1e-10:
            unique.append(r)
    pieces = []
    for a, b in zip(unique[:-1], unique[1:]):
        midpoint = np.polynomial.polynomial.polyval((a + b) / 2, c)
        cell = np.floor((midpoint - origin) / h).astype(np.int64)
        # A whole curve on the last valid node uses the final interior cell.
        cell = np.minimum(cell, shape - 3)
        if np.any(cell < 1) or np.any(cell > shape - 3):
            raise ValueError("cell has no complete 4^3 public sample stencil")
        for axis in range(3):
            minimum, maximum = _extrema(c[:, axis], a, b)
            if minimum < origin[axis] + cell[axis] * h - 1e-8 or maximum > origin[axis] + (cell[axis] + 1) * h + 1e-8:
                raise ValueError("a segment crosses a cell face: root splitting failed")
        pieces.append((a, b, tuple(int(i) for i in cell)))
    return pieces, {"face_events": events, "identically_on_faces": on_faces,
                    "coverage_check": "coordinate extrema via numerical degree-four roots; not certified",
                    "near_tangency_policy": "reject stationary face residual < 1e-9 m"}


@lru_cache(maxsize=None)
def _product_data(n, m):
    indices, weights = [], []
    for i in range(n + 1):
        for j in range(m + 1):
            indices.append(i + j)
            weights.append(comb(n, i) * comb(m, j) / comb(n + m, i + j))
    return torch.tensor(indices), torch.tensor(weights, dtype=torch.float64)


def bernstein_product(a, b):
    """Multiply Bernstein polynomials; last dimension contains coefficients."""
    n, m = a.shape[-1] - 1, b.shape[-1] - 1
    indices, weights = _product_data(n, m)
    products = (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2) * weights
    result = products.new_zeros(products.shape[:-1] + (n + m + 1,))
    return result.scatter_add(-1, indices.expand_as(products), products)


def elevate(a, degree):
    n = a.shape[-1] - 1
    if n == degree:
        return a
    return bernstein_product(a, a.new_ones(degree - n + 1))


def restricted_bernstein(coefficients, a, b):
    """Degree-five world-coordinate curve over u=a+(b-a)*s, s in [0,1]."""
    transform = coefficients.new_zeros((6, 6))
    for k in range(6):
        for j in range(k + 1):
            transform[j, k] = comb(k, j) * a ** (k - j) * (b - a) ** j
    local_power = transform @ coefficients
    to_bernstein = coefficients.new_zeros((6, 6))
    for i in range(6):
        for j in range(i + 1):
            to_bernstein[i, j] = comb(i, j) / comb(5, j)
    return (to_bernstein @ local_power).transpose(0, 1)


def cell_distance_bernstein(field, samples, coefficients, piece):
    """Compose public Catmull–Rom node data with a quintic, degree <= 45."""
    a, b, cell = piece
    origin = coefficients.new_tensor(field.sample_origin_m)
    h = float(field.spacing_m)
    q = (restricted_bernstein(coefficients, a, b)
         - (origin + coefficients.new_tensor(cell) * h).unsqueeze(-1)) / h
    q2 = bernstein_product(q, q)
    q3 = bernstein_product(q2, q)
    powers = torch.stack((q.new_ones((3, 16)), elevate(q, 15), elevate(q2, 15), q3), dim=-2)
    # Rows correspond to node offsets -1,0,1,2; ascending cubic powers.
    basis = coefficients.new_tensor([[0, -.5, 1, -.5], [1, 0, -2.5, 1.5],
                                     [0, .5, 2, -1.5], [0, 0, -.5, .5]])
    weights = torch.einsum("jp,apn->ajn", basis, powers)
    xy = bernstein_product(weights[0, :, None], weights[1, None, :])
    xyz = bernstein_product(xy[:, :, None], weights[2, None, None, :])
    i, j, k = cell
    stencil = samples[i-1:i+3, j-1:j+3, k-1:k+3]
    return torch.sum(stencil[..., None] * xyz, dim=(0, 1, 2))


def polynomial_integral(field, coefficients, duration, *, target=0.1, pieces=None, samples=None):
    """Integrate 0.5*(field(p(t))-target)^2 dt, not an obstacle penalty.

    Crossing times/cells are discrete detached calculations. For transverse
    crossings, continuity makes adjacent moving-limit terms cancel. Backward
    differentiates the polynomial coefficients and the explicit duration.
    """
    if coefficients.dtype != torch.float64 or coefficients.device.type != "cpu":
        raise ValueError("the experiment requires CPU float64 coefficients")
    if pieces is None:
        pieces, _ = split_curve(field, coefficients)
    if samples is None:
        samples = torch.as_tensor(field.samples).to(dtype=coefficients.dtype, device="cpu")
    terms = []
    for piece in pieces:
        distance = cell_distance_bernstein(field, samples, coefficients, piece)
        residual = distance - target
        integrand = .5 * bernstein_product(residual, residual)
        terms.append((piece[1] - piece[0]) * integrand.mean())
    return duration * torch.stack(terms).sum()


def gauss_integral(field, trajectory, pieces, *, order=64, penalty="square", target=.1):
    """Independent public-query quadrature; also used for non-polynomial exp."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    us, ws = [], []
    for a, b, _ in pieces:
        us.extend(a + (nodes + 1) * (b - a) / 2)
        ws.extend(weights * (b - a) / 2)
    u = trajectory.position_coefficients.new_tensor(us)
    weights = u.new_tensor(ws)
    positions = trajectory.sample(u * trajectory.duration).position
    distance = field.query(positions)
    values = .5 * (distance - target).square() if penalty == "square" else torch.exp(-distance / .4)
    return trajectory.duration * torch.sum(weights * values)
