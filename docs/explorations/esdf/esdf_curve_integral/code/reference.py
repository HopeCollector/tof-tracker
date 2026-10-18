"""Independent time-quadrature reference for the same C1 spatial field.

Uniform composite Gauss-Legendre nodes are chosen without consulting cells,
crossing times, or the partition under evaluation. The field's public query
supplies distance and its position derivative; independent analytic quintic
boundary sensitivities then give all nine terminal p/v/a derivatives.

The result is numerical, not an exact exponential integral or a collision
certificate. Refinement deltas are observed disagreements, NOT rigorous error
bounds (and no Richardson factor is assumed across piecewise cubics). Keep
every invalid or unconverged row. Mesh integrals are a separate spatial-error
comparison, with their own nonunique-gradient and convergence qualifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


DURATION_S = 10. / 6.


def gauss_nodes(panels, *, order=8, device="cpu"):
    """Float64 normalized-time nodes/weights; weights sum to one."""
    for name, value in (("panels", panels), ("order", order)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = (np.arange(panels)[:, None] + .5 * (nodes[None] + 1)) / panels
    w = np.broadcast_to(weights[None] / (2 * panels), u.shape)
    return (torch.tensor(u.ravel(), device=device, dtype=torch.float64),
            torch.tensor(w.ravel(), device=device, dtype=torch.float64))


def positions(coefficients, u):
    """Horner evaluation of ascending normalized-time coefficients (...,6,3)."""
    value = coefficients[..., 5, :].unsqueeze(-2)
    for power in range(4, -1, -1):
        value = value * u[:, None] + coefficients[..., power, :].unsqueeze(-2)
    return value


def terminal_basis(u, duration_s=DURATION_S):
    """dp(u)/d(p1,v1,a1) scalar factors, shape (...,N,3).

    The initial p/v/a and physical duration are held fixed. Each factor acts
    independently on x/y/z; flatten [p/v/a,xyz] for the nine-parameter order.
    This does not give a duration derivative.
    """
    t = torch.as_tensor(duration_s, dtype=u.dtype, device=u.device)[..., None]
    u3, u4, u5 = u**3, u**4, u**5
    factors = torch.broadcast_tensors(10*u3 - 15*u4 + 6*u5,
                                     t*(-4*u3 + 7*u4 - 3*u5),
                                     .5*t.square()*(u3 - 2*u4 + u5))
    return torch.stack(factors, dim=-1)


def composite_gauss(field, coefficients, *, panels=512, order=8,
                    duration_s=DURATION_S, node_chunk=256, clearance_m=1.2,
                    radius_m=.6, normalization="average"):
    """Return detached cost (...), gradient (...,9), coefficient_gradient (...,6,3).

    field is an object exposing query(points), or that differentiable callable.
    coefficients must be float64 (...,6,3), ascending powers of u=t/T. The
    values are assumed to come from the specified fixed-duration boundary fit;
    the nine sensitivities refer to its terminal p/v/a, not arbitrary 18
    coefficients. No curve fitting or grid partition is reused here.

    average: integral_0^1 exp(-(d(p(u))-clearance)/radius) du.
    time_integral: duration times that integral, still holding duration fixed
    in the nine-parameter gradient. Invalid samples poison their whole row's
    cost AND gradient with NaN; zero gradients never mean a failed query is safe.
    Returned valid is sampled numerical validity, not whole-curve certification.
    """
    if (not isinstance(coefficients, torch.Tensor) or coefficients.dtype != torch.float64
            or coefficients.shape[-2:] != (6, 3) or not torch.isfinite(coefficients).all()):
        raise ValueError("coefficients must be finite float64 (...,6,3)")
    if isinstance(node_chunk, bool) or not isinstance(node_chunk, int) or node_chunk < 1:
        raise ValueError("node_chunk must be a positive integer")
    if (not np.isfinite(clearance_m) or not np.isfinite(radius_m) or radius_m <= 0
            or normalization not in ("average", "time_integral")):
        raise ValueError("require finite clearance, positive radius, and known normalization")
    c = coefficients.detach()
    shape = c.shape[:-2]
    duration = torch.as_tensor(duration_s, dtype=c.dtype, device=c.device)
    if (torch.broadcast_shapes(duration.shape, shape) != shape
            or not bool((torch.isfinite(duration) & (duration > 0)).all())):
        raise ValueError("duration must be finite positive and broadcast to the curve batch")
    duration = duration.expand(shape)
    u, weights = gauss_nodes(panels, order=order, device=c.device)
    query = field.query if hasattr(field, "query") else field
    cost = c.new_zeros(shape)
    gradient = c.new_zeros((*shape, 9))
    coefficient_gradient = c.new_zeros((*shape, 6, 3))
    invalid_count = torch.zeros(shape, device=c.device, dtype=torch.int64)
    for begin in range(0, len(u), node_chunk):
        t, w = u[begin:begin+node_chunk], weights[begin:begin+node_chunk]
        with torch.enable_grad():
            p = positions(c, t).detach().requires_grad_(True)
            distance = query(p)
            if not isinstance(distance, torch.Tensor) or distance.shape != p.shape[:-1]:
                raise ValueError("field query must return one distance per position")
            spatial_gradient, = torch.autograd.grad(distance.sum(), p)
        with torch.no_grad():
            penalty = torch.exp(-(distance - clearance_m) / radius_m)
            penalty_gradient = -penalty[..., None] * spatial_gradient / radius_m
            valid = (torch.isfinite(distance) & torch.isfinite(penalty)
                     & torch.isfinite(spatial_gradient).all(-1)
                     & torch.isfinite(penalty_gradient).all(-1))
            invalid_count += (~valid).sum(-1)
            # Mask temporary terms to stop a NaN from obscuring failure counts;
            # the whole affected row is explicitly poisoned after accumulation.
            safe_penalty = torch.where(valid, penalty, 0.)
            safe_gradient = torch.where(valid[..., None], penalty_gradient, 0.)
            cost += (safe_penalty*w).sum(-1)
            basis = terminal_basis(t, duration)
            gradient += torch.einsum("...nc,...nj,n->...jc", safe_gradient, basis, w).flatten(-2)
            powers = t[:, None] ** torch.arange(6, device=t.device)
            coefficient_gradient += torch.einsum("...nc,nk,n->...kc", safe_gradient, powers, w)
    if normalization == "time_integral":
        cost, gradient = cost * duration, gradient * duration[..., None]
        coefficient_gradient = coefficient_gradient * duration[..., None, None]
    valid = ((invalid_count == 0) & torch.isfinite(cost) & torch.isfinite(gradient).all(-1)
             & torch.isfinite(coefficient_gradient).all((-2,-1)))
    cost = torch.where(valid, cost, torch.nan)
    gradient = torch.where(valid[..., None], gradient, torch.nan)
    coefficient_gradient = torch.where(valid[..., None, None], coefficient_gradient, torch.nan)
    return {"cost": cost.detach(), "gradient": gradient.detach(), "valid": valid,
            "coefficient_gradient": coefficient_gradient.detach(),
            "invalid_count": invalid_count, "panels": int(panels), "order": int(order),
            "points_per_curve": len(u), "normalization": normalization}


def refinement_delta(coarse, fine, *, cost_atol=1e-8, cost_rtol=1e-6,
                     gradient_atol=1e-7, gradient_rtol=1e-5):
    """Rowwise observed cost/9-vector changes; no mathematical error guarantee."""
    if not all(np.isfinite(x) and x >= 0 for x in (cost_atol, cost_rtol, gradient_atol, gradient_rtol)):
        raise ValueError("refinement tolerances must be finite and nonnegative")
    for key in ("cost", "gradient", "coefficient_gradient", "valid"):
        if coarse[key].shape != fine[key].shape:
            raise ValueError("refinement rows must have identical shapes")
    cost_delta = (fine["cost"] - coarse["cost"]).abs()
    gradient_delta = torch.linalg.vector_norm(fine["gradient"] - coarse["gradient"], dim=-1)
    cost_limit = cost_atol + cost_rtol*fine["cost"].abs()
    gradient_limit = gradient_atol + gradient_rtol*torch.linalg.vector_norm(fine["gradient"], dim=-1)
    coefficient_delta = torch.linalg.vector_norm(
        (fine["coefficient_gradient"]-coarse["coefficient_gradient"]).flatten(-2), dim=-1)
    coefficient_limit = gradient_atol + gradient_rtol*torch.linalg.vector_norm(
        fine["coefficient_gradient"].flatten(-2), dim=-1)
    finite = (coarse["valid"] & fine["valid"] & torch.isfinite(cost_delta)
              & torch.isfinite(gradient_delta) & torch.isfinite(coefficient_delta))
    cost_ok, gradient_ok = finite & (cost_delta <= cost_limit), finite & (gradient_delta <= gradient_limit)
    coefficient_ok = finite & (coefficient_delta <= coefficient_limit)
    return {"cost_delta": cost_delta, "gradient_delta_l2": gradient_delta,
            "coefficient_gradient_delta_l2": coefficient_delta,
            "cost_limit": cost_limit, "gradient_limit_l2": gradient_limit,
            "coefficient_gradient_limit_l2": coefficient_limit,
            "cost_converged": cost_ok, "gradient_converged": gradient_ok,
            "coefficient_gradient_converged": coefficient_ok,
            "converged": cost_ok & gradient_ok & coefficient_ok}


def refine_reference(field, coefficients, *, panels=(256,512,1024,2048),
                     cost_atol=1e-8, cost_rtol=1e-6, gradient_atol=1e-7,
                     gradient_rtol=1e-5, required_refinements=2, **kwargs):
    """Run every level, retaining all rows and consecutive convergence masks.

    levels[-1] is the numerical reference, including failed rows. history keeps
    rowwise comparisons. Each cost/terminal9/coeff18 mask requires the specified
    number of consecutive successful refinements, not just the last pair. There
    is no early stopping or deletion of difficult curves. All tensors stay on
    the query device; callers decide when to export raw arrays.
    """
    panels = tuple(panels)
    if (len(panels) < 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in panels)
            or any(b != 2*a for a,b in zip(panels, panels[1:]))):
        raise ValueError("panels must contain at least two positive, doubling integer levels")
    if (isinstance(required_refinements, bool) or not isinstance(required_refinements, int)
            or not 1 <= required_refinements < len(panels)):
        raise ValueError("required_refinements must be between 1 and number of comparisons")
    levels, history = [], []
    cost_runs = torch.zeros(coefficients.shape[:-2], device=coefficients.device, dtype=torch.int64)
    gradient_runs = torch.zeros_like(cost_runs)
    coefficient_runs = torch.zeros_like(cost_runs)
    for count in panels:
        current = composite_gauss(field, coefficients, panels=count, **kwargs)
        if levels:
            delta = refinement_delta(levels[-1], current, cost_atol=cost_atol, cost_rtol=cost_rtol,
                                     gradient_atol=gradient_atol, gradient_rtol=gradient_rtol)
            cost_runs = torch.where(delta["cost_converged"], cost_runs + 1, 0)
            gradient_runs = torch.where(delta["gradient_converged"], gradient_runs + 1, 0)
            coefficient_runs = torch.where(delta["coefficient_gradient_converged"], coefficient_runs + 1, 0)
            history.append({"coarse_panels": levels[-1]["panels"], "fine_panels": count, **delta})
        levels.append(current)
    cost_ok, gradient_ok = cost_runs >= required_refinements, gradient_runs >= required_refinements
    coefficient_ok = coefficient_runs >= required_refinements
    return {"reference": levels[-1], "levels": levels, "history": history,
            "cost_converged": cost_ok, "gradient_converged": gradient_ok,
            "coefficient_gradient_converged": coefficient_ok,
            "converged": cost_ok & gradient_ok & coefficient_ok, "certified": False,
            "tolerances": {"cost_atol": cost_atol, "cost_rtol": cost_rtol,
                           "gradient_atol": gradient_atol, "gradient_rtol": gradient_rtol,
                           "required_refinements": required_refinements}}


def previous_mesh_reference(raw_path, *, expected_coefficients=None):
    """Load existing secondary mesh evidence, keeping its original masks/ties.

    This does no new geometry integration and cannot promote the old numerical
    estimate to exact truth. Optional exact coefficient comparison (allowing
    only batch-prefix flattening, with row order unchanged) prevents accidentally
    comparing different fixtures. Arrays retain their original batch prefix.
    Original tolerance definitions
    belong to the accompanying previous results.json, not this field reference.
    """
    path = Path(raw_path)
    with np.load(path, allow_pickle=False) as saved:
        if expected_coefficients is not None:
            expected = (expected_coefficients.detach().cpu().numpy() if isinstance(expected_coefficients, torch.Tensor)
                        else np.asarray(expected_coefficients))
            stored = saved["coefficients"]
            if (expected.shape[-2:] != (6,3) or stored.shape[-2:] != (6,3)
                    or not np.array_equal(expected.reshape(-1,6,3), stored.reshape(-1,6,3))):
                raise ValueError("secondary mesh evidence belongs to different curve coefficients")
        result = {name: saved[key].copy() for name,key in (
            ("cost", "mesh_dense_cost"), ("gradient", "mesh_dense_gradient"),
            ("cost_converged", "mesh_cost_converged"), ("gradient_converged", "mesh_gradient_converged"),
            ("tie", "mesh_dense_tie"), ("cost_delta", "mesh_final_cost_delta"),
            ("gradient_delta_l2", "mesh_final_gradient_delta_l2"))}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    result.update(source_path=str(path), source_sha256=digest.hexdigest(),
                  role="secondary spatial-geometry comparison only", certified=False)
    return result


def verify_cpu():
    """Small reproducible plane/nonlinear-field and nine-parameter CPU checks."""
    from docs.explorations.esdf.esdf_grid_eval.code.c1_esdf import ESDFGrid
    from devtools.trajectory import TrajectoryState, fit_trajectory

    torch.set_num_threads(1)
    t = torch.tensor(DURATION_S, dtype=torch.float64)
    zero = torch.tensor(0., dtype=torch.float64)
    initial = TrajectoryState(torch.tensor([.1,.15,.3],dtype=torch.float64),
                              torch.tensor([.03,-.02,.04],dtype=torch.float64),
                              torch.tensor([-.01,.01,.02],dtype=torch.float64),zero,zero)
    theta = torch.tensor([.8,.6,.7,.2,.1,.15,-.03,.04,-.02],dtype=torch.float64,requires_grad=True)
    def fit(parameters):
        terminal = TrajectoryState(parameters[:3],parameters[3:6],parameters[6:9],zero,zero)
        return fit_trajectory(initial,terminal,t).position_coefficients
    axis = np.arange(-1,7,dtype=np.float64)*.25
    xyz = np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),axis=-1)
    # Binary-exact polynomial node values include a nonlinear spatial term.
    values = (xyz[...,0]/4 + xyz[...,1]/2 + xyz[...,2] + xyz[...,0]*xyz[...,1]/2).astype(np.float32)
    field = ESDFGrid(values,sample_origin_m=(-.25,)*3,spacing_m=.25,bounds_m=((0,)*3,(1.25,)*3))
    c = fit(theta)
    u,w = gauss_nodes(64)
    reference = composite_gauss(field,c,panels=64,node_chunk=37)
    independent_coefficients = c.detach().requires_grad_()
    coefficient_value = (torch.exp(-(field.query(positions(independent_coefficients,u))-1.2)/.6)*w).sum()
    coefficient_ad, = torch.autograd.grad(coefficient_value,independent_coefficients)
    coefficient_ad_error = float((coefficient_ad-reference["coefficient_gradient"]).abs().max())
    assert coefficient_ad_error < 3e-14
    def objective(parameters):
        d = field.query(positions(fit(parameters),u))
        return (torch.exp(-(d-1.2)/.6)*w).sum()
    objective_value = objective(theta)
    direct_gradient, = torch.autograd.grad(objective_value,theta)
    chain_error = float((reference["gradient"]-direct_gradient).abs().max())
    assert chain_error < 3e-14 and abs(float(reference["cost"]-objective_value.detach())) < 3e-14
    fd_errors = []
    for h in (1e-4,1e-5):
        fd = torch.zeros_like(theta)
        for i in range(9):
            plus,minus = theta.detach().clone(),theta.detach().clone()
            plus[i] += h; minus[i] -= h
            fd[i] = (objective(plus)-objective(minus))/(2*h)
        fd_errors.append(float((fd-reference["gradient"]).abs().max()))
    assert max(fd_errors) < 3e-8
    refined = refine_reference(field,c,panels=(16,32,64,128),node_chunk=43,
                               cost_atol=1e-12,cost_rtol=1e-12,gradient_atol=1e-12,gradient_rtol=1e-12)
    assert bool(refined["converged"])
    physical = composite_gauss(field,c,panels=64,normalization="time_integral")
    assert abs(float(physical["cost"]-reference["cost"]*t)) < 3e-14
    assert float((physical["gradient"]-reference["gradient"]*t).abs().max()) < 3e-14
    assert float((physical["coefficient_gradient"]-reference["coefficient_gradient"]*t).abs().max()) < 3e-14
    # Exact exponential of an affine-in-time plane distance has closed form.
    plane = lambda p: p[...,2]
    line = torch.zeros((6,3),dtype=torch.float64)
    line[0,2],line[1,2] = .3,.4
    line_ref = composite_gauss(plane,line,panels=2)
    closed = np.exp((1.2-.3)/.6)*(-np.expm1(-.4/.6))/(.4/.6)
    assert abs(float(line_ref["cost"])-closed) < 2e-15
    # Closed moments test all 18 coefficients, including B=0 and near zero.
    normal = np.array([.25,.5,-.125])
    plane_values = (xyz @ normal + .125).astype(np.float32)
    plane_field = ESDFGrid(plane_values,sample_origin_m=(-.25,)*3,spacing_m=.25,
                          bounds_m=((0,)*3,(1.25,)*3))
    closed_errors = []
    for b in (-.15,0.,1e-12,.15):
        line = torch.zeros((6,3),dtype=torch.float64)
        line[0] = torch.tensor([.5,.25,.5],dtype=torch.float64)
        line[1,0] = -b*.6/normal[0]
        measured = composite_gauss(plane_field,line,panels=2)
        n = normal.astype(np.longdouble)
        c0,c1 = line[:2].numpy().astype(np.longdouble)
        a = (np.longdouble(1.2)-n@c0-np.longdouble(.125))/np.longdouble(.6)
        exponent = -(n@c1)/np.longdouble(.6)
        moments = []
        for k in range(6):
            term = np.longdouble(1.)
            moment = term/(k+1)
            for j in range(1,80):
                term *= exponent/j
                moment += term/(j+k+1)
            moments.append(moment)
        expected = -np.exp(a)*np.asarray(moments)[:,None]*n[None]/np.longdouble(.6)
        closed_errors.append(float(np.max(np.abs(measured["coefficient_gradient"].numpy()-expected))))
        assert abs(float(measured["cost"])-float(np.exp(a)*moments[0])) < 3e-15
    assert max(closed_errors) < 2e-14, closed_errors
    # Rowwise validity and convergence: do not discard a failing path.
    invalid = c.detach().clone(); invalid[0,0] += 5
    mixed = refine_reference(field,torch.stack((c.detach(),invalid)),panels=(4,8,16))
    assert bool(mixed["converged"][0]) and not bool(mixed["converged"][1])
    assert mixed["reference"]["invalid_count"][1] == 128
    assert torch.isnan(mixed["reference"]["gradient"][1]).all()
    # Explicit formula test, including equality and nonfinite rejection.
    first = {"cost":torch.tensor([10.,10.,torch.nan]), "gradient":torch.zeros((3,9)),
             "coefficient_gradient":torch.zeros((3,6,3)),
             "valid":torch.tensor([True,True,False])}
    second = {"cost":torch.tensor([10.25,10.5,torch.nan]), "gradient":torch.zeros((3,9)),
              "coefficient_gradient":torch.zeros((3,6,3)),
              "valid":torch.tensor([True,True,False])}
    second["gradient"][0,0],second["gradient"][1,0] = .1,.2
    delta = refinement_delta(first,second,cost_atol=.25,cost_rtol=0,gradient_atol=.1,gradient_rtol=0)
    assert delta["converged"].tolist() == [True,False,False]
    return {"spatial_field":"small float32 C1 node grid", "terminal9_chain_max_abs":chain_error,
            "coefficient18_same_nodes_ad_max_abs":coefficient_ad_error,
            "coefficient18_closed_moment_max_abs":closed_errors,
            "terminal9_fd_steps":[1e-4,1e-5],"terminal9_fd_max_abs":fd_errors,
            "closed_plane_cost_error":abs(float(line_ref["cost"])-closed),
            "same_nodes_ad_value_error":abs(float(reference["cost"]-objective_value.detach())),
            "rowwise_failure_preserved":True,"refinement_formula_checked":True,
            "physical_duration_factor_checked":True,"certified":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cpu",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if not args.verify_cpu:
        parser.error("this standalone entry point only supports --verify-cpu")
    result = verify_cpu()
    encoded = json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded+"\n")
    print(encoded)
