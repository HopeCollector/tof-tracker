"""CPU checks for the experimental spatial/trajectory oracle; run in Docker.

Analytic box, extruded cross polygon, and ground answers are independent of
rectangle closest-point code. Boundary generation is the existing experimental
box-union adapter; it is not a query oracle and no production BVH is called.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from devtools.trajectory import TrajectoryState, fit_trajectory
from experiments.adaptive_distance_eval.real_map_geometry import _exposed_rectangles, load_real_map
from experiments.esdf_reference_eval.oracle import (
    RectangleUnionOracle, numerical_integral, reference_penalty, sampled_mean, tracker_fractions,
)


def make_oracle(boxes, **kwargs):
    boxes = np.asarray(boxes, dtype=np.float64)
    return RectangleUnionOracle(*_exposed_rectangles(boxes), boxes, **kwargs)


def box_sdf(points):
    q = np.abs(points) - 1.
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def cross_sdf(points):
    polygon = np.array([[-1,-.2],[-.2,-.2],[-.2,-1],[.2,-1],[.2,-.2],[1,-.2],
                        [1,.2],[.2,.2],[.2,1],[-.2,1],[-.2,.2],[-1,.2]])
    distance = np.full(len(points), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        t = np.clip((points[:, :2] - a) @ edge / (edge @ edge), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points[:, :2] - a - t[:, None] * edge, axis=1))
    xy = np.abs(points[:, :2])
    inside = ((xy[:, 0] <= 1) & (xy[:, 1] <= .2)) | ((xy[:, 0] <= .2) & (xy[:, 1] <= 1))
    q = np.stack((distance * np.where(inside, -1, 1), np.abs(points[:, 2]) - 1), axis=1)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def finite_difference(function, parameters, step):
    result = torch.zeros_like(parameters)
    for index in range(parameters.numel()):
        plus, minus = parameters.detach().clone(), parameters.detach().clone()
        plus.reshape(-1)[index] += step
        minus.reshape(-1)[index] -= step
        result.reshape(-1)[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def run(map_path=None):
    torch.set_num_threads(1)
    dtype = torch.float64
    rng = np.random.default_rng(5219)
    result = {}
    unit = make_oracle([[[-1,-1,-1],[1,1,1]]], point_chunk=17, rectangle_chunk=2)
    points = rng.uniform(-2, 2, (300, 3))
    points = np.vstack((points, [[0,0,0],[1,0,0],[1,1,0],[1,1,1],[1+1e-10,0,0],[1-1e-10,0,0]]))
    actual = unit.evaluate(torch.tensor(points))
    error = float(np.abs(actual["distance"].numpy() - box_sdf(points)).max())
    assert error < 5e-15, error
    assert bool(actual["tie"][-6])  # center has six different nearest faces
    assert not bool(actual["tie"][-5])  # smooth zero-distance face
    assert bool(actual["tie"][-4]) and bool(actual["tie"][-3])  # zero edge and vertex
    result["analytic_box"] = {"count": len(points), "distance_max_abs_m": error}

    ordinary = torch.tensor([[1.3,.1,.2],[1.3,1.4,.2],[1.2,1.3,1.6],
                             [.2,.3,.7],[-.8,.2,.1]], dtype=dtype, requires_grad=True)
    grad = torch.autograd.grad(unit.query(ordinary).sum(), ordinary)[0]
    q = np.maximum(np.abs(ordinary.detach().numpy()) - 1, 0)
    expected = q / np.where(np.linalg.norm(q, axis=1) > 0, np.linalg.norm(q, axis=1), 1)[:, None]
    expected *= np.sign(ordinary.detach().numpy())
    expected[-2:] = [[0,0,1],[-1,0,0]]
    assert np.max(np.abs(grad.numpy() - expected)) < 2e-15
    assert torch.autograd.gradcheck(unit.query, (ordinary,), eps=1e-6, atol=2e-8, rtol=2e-7)
    result["box_gradient_and_gradcheck"] = {"max_abs": float(np.max(np.abs(grad.numpy() - expected)))}

    cross = make_oracle([[[-1,-.2,-1],[1,.2,1]],[[-.2,-1,-1],[.2,1,1]]])
    points = np.vstack((rng.uniform(-1.5, 1.5, (257, 3)), [[0,0,0],[.1,.05,0]]))
    diagnostic = cross.evaluate(torch.tensor(points))
    error = float(np.abs(diagnostic["distance"].numpy() - cross_sdf(points)).max())
    assert error < 2e-15
    assert abs(float(diagnostic["distance"][-2]) + np.sqrt(.08)) < 1e-15
    assert bool(diagnostic["tie"][-2])
    selected = torch.tensor(points[~diagnostic["tie"].numpy()][:16], requires_grad=True)
    assert torch.autograd.gradcheck(cross.query, (selected,), eps=1e-6, atol=2e-8, rtol=2e-7)
    result["overlap_true_union"] = {"distance_max_abs_m": error,
                                   "center_true_distance_m": float(diagnostic["distance"][-2]),
                                   "incorrect_min_object_distance_m": -.2}

    slab = [[-10,-10,-10],[10,10,0]]
    ground = make_oracle([slab], bounds_m=[[-2,-2,-.2],[2,2,3]])
    pillar_ground = make_oracle([slab, [[-1,-1,0],[1,1,2]]])
    p = torch.tensor([[0,0,0],[0,0,.1],[0,0,1],[0,0,2],[1,0,0],[1.2,0,.2],[0,0,-.1]], dtype=dtype)
    expected = np.array([-1,-1,-1,0,0,.2,-np.sqrt(1.01)])
    assert np.max(np.abs(pillar_ground.query(p).numpy() - expected)) < 1e-15
    p = torch.tensor([[.2,.3,z] for z in (0,1e-12,-1e-12,.2,-.2)], dtype=dtype, requires_grad=True)
    d = ground.query(p)
    g = torch.autograd.grad(d.sum(), p)[0]
    assert torch.equal(d, p[:, 2]) and torch.equal(g, torch.tensor([[0.,0.,1.]]).expand(5, -1))
    result["ground_union_and_near_zero"] = {"passed": True}

    # Duplicate coplanar fragments do not create a geometric medial-axis tie.
    duplicate = RectangleUnionOracle(np.repeat(unit.lower, 2, axis=0), np.repeat(unit.upper, 2, axis=0),
                                    np.repeat(unit.axes, 2), np.repeat(unit.signs, 2), unit.solid_boxes)
    assert not bool(duplicate.evaluate(torch.tensor([[1.3,.1,.2]], dtype=dtype))["tie"][0])
    result["duplicate_same_gradient_not_tie"] = {"passed": True}

    restrictions = [([[-.12,-.1,-.3],[.16,.09,.3]], cross),
                    ([[1.3,-.3,-.3],[1.7,.3,.3]], unit),
                    ([[-1.5,-1.5,-.1],[1.5,1.5,.5]], pillar_ground),
                    ([[1.3,-.3,0],[1.7,.3,0]], unit)]
    for bounds, source in restrictions:
        restricted = source.restricted(bounds)
        p = torch.tensor(rng.uniform(*np.array(bounds), size=(100, 3)))
        a, b = source.evaluate(p), restricted.evaluate(p)
        for name in ("distance", "gradient", "closest_point", "nearest_rectangle", "tie", "inside"):
            assert torch.equal(a[name], b[name]), name
    assert len(unit.restricted([[1.3,-.3,-.3],[1.7,.3,.3]]).solid_boxes) == 0
    result["restricted_matches_global"] = {"points": 400, "bitwise_distance_gradient_and_ids": True}

    bounded = unit.restricted([[-2,-2,-2],[2,2,2]])
    p = torch.tensor([[2,0,0],[2.1,0,0],[float("nan"),0,0],[0,float("inf"),0]], dtype=dtype, requires_grad=True)
    d = bounded.query(p)
    # Test unmasked NaN loss too: invalid incoming gradients remain zero.
    g = torch.autograd.grad(reference_penalty(d).sum(), p)[0]
    assert torch.isfinite(d[0]) and torch.isnan(d[1:]).all() and torch.equal(g[1:], torch.zeros_like(g[1:]))
    assert bounded.query(torch.empty(0,3)).shape == (0,)
    p32 = ordinary.detach().float().reshape(1,5,3).requires_grad_()
    d32 = unit.query(p32)
    g32 = torch.autograd.grad(d32.sum(), p32)[0]
    assert d32.dtype == g32.dtype == torch.float32 and d32.shape == (1,5)
    assert torch.max(torch.abs(g32.double().reshape(5,3) - grad)) < 2e-7
    result["invalid_empty_dtype_batch"] = {"passed": True}

    duration = torch.tensor(10/6, dtype=dtype)
    zero = torch.tensor(0., dtype=dtype)
    initial = TrajectoryState(torch.tensor([1.4,1.3,1.2], dtype=dtype),
                              torch.tensor([.1,.1,.1], dtype=dtype), torch.zeros(3,dtype=dtype), zero, zero)
    theta = torch.tensor([1.8,1.6,1.7,.15,.12,.1,.02,-.03,.01], dtype=dtype, requires_grad=True)
    def positions(parameters, u):
        terminal = TrajectoryState(parameters[:3],parameters[3:6],parameters[6:9],zero,zero)
        return fit_trajectory(initial,terminal,duration).sample(duration*u).position
    fractions = tracker_fractions()
    assert fractions.shape == (30,) and fractions[0] == 1/30 and fractions[-1] == 1
    def objective(parameters):
        return sampled_mean(unit, positions(parameters, fractions))
    value = objective(theta)
    derivative = torch.autograd.grad(value, theta)[0]
    fd_errors = []
    for step in (2e-5, 1e-5):
        fd_errors.append(float((finite_difference(objective, theta, step) - derivative).abs().max()))
    assert max(fd_errors) < 2e-8, fd_errors
    assert bool((derivative.abs() > 1e-5).all())  # All nine parameters exercised.
    analytic_points = positions(theta, fractions)
    analytic = reference_penalty(torch.linalg.vector_norm(analytic_points - 1., dim=-1)).mean()
    assert abs(float((value - analytic).detach())) < 1e-15
    decreased = objective((theta - .01 * derivative).detach())
    assert decreased < value
    result["tracker_30_mean_terminal9_fd"] = {"value": float(value.detach()), "steps": [2e-5,1e-5],
        "gradient_fd_max_abs": fd_errors, "sgd_value": float(decreased), "duration_s": float(duration)}

    theta_i = torch.tensor([.8,.4,1.7],dtype=dtype,requires_grad=True)
    def plane_path(u):
        return torch.stack((u*.2,u*.1,theta_i[0]+theta_i[1]*u),dim=-1)
    def analytic_integral(parameters):
        a,b,_ = parameters
        return torch.exp((1.2-a)/.6)*(-torch.expm1(-b/.6))/(b/.6)
    normalized = numerical_integral(ground,plane_path,parameters=theta_i,order=4,max_panels=16,
                                    atol=1e-11,rtol=1e-11,grad_atol=1e-11,grad_rtol=1e-11)
    physical = numerical_integral(ground,plane_path,parameters=theta_i,duration=theta_i[2],order=4,max_panels=16,
                                  atol=1e-11,rtol=1e-11,grad_atol=1e-11,grad_rtol=1e-11)
    closed = analytic_integral(theta_i)
    closed_g = torch.autograd.grad(closed,theta_i)[0]
    closed_physical_g = torch.autograd.grad(analytic_integral(theta_i)*theta_i[2],theta_i)[0]
    assert normalized["converged"] and physical["converged"]
    assert abs(float((normalized["value"]-closed).detach())) < 2e-14
    assert float((normalized["parameter_gradient"]-closed_g).abs().max()) < 3e-14
    assert float((physical["parameter_gradient"]-closed_physical_g).abs().max()) < 5e-14
    finite_mean = sampled_mean(ground,plane_path(fractions))
    assert abs(float((finite_mean - closed).detach())) > 1e-3  # Deliberately different objectives.
    result["quadrature_plane_closed_form"] = {"normalized": float(normalized["value"].detach()),
        "physical": float(physical["value"].detach()),"finite_mean_30": float(finite_mean.detach()),
        "value_abs_error": abs(float(normalized["value"].detach()-closed.detach())),
        "gradient_max_abs_error": float((normalized["parameter_gradient"]-closed_g).abs().max()),
        "history": normalized["history"], "duration_gradient_extension": True}

    y = torch.tensor([1.3],dtype=dtype,requires_grad=True)
    def corner_path(u):
        return torch.stack((3*u-1.5,y[0]+u*0,u*0),dim=-1)
    integral = numerical_integral(unit,corner_path,parameters=y,order=8,max_panels=512,
                                  atol=2e-8,rtol=2e-8,grad_atol=2e-8,grad_rtol=2e-8)
    assert integral["converged"], integral["history"]
    dense_u = (torch.arange(32768,dtype=dtype)+.5)/32768
    dense = sampled_mean(unit,corner_path(dense_u))
    dense_gradient = torch.autograd.grad(dense,y)[0]
    assert abs(float((dense-integral["value"]).detach())) < 5e-8
    assert float((dense_gradient-integral["parameter_gradient"]).abs().max()) < 5e-8
    unconverged = numerical_integral(unit,corner_path,parameters=y,max_panels=1)
    assert not unconverged["converged"] and not unconverged["certified"]
    result["quadrature_feature_changes"] = {"history": integral["history"],
        "dense32768_value_abs_difference": abs(float(dense.detach()-integral["value"].detach())),
        "dense32768_gradient_abs_difference": float((dense_gradient-integral["parameter_gradient"]).abs().max()),
        "budget_exhaustion_reports_unconverged": True}

    if map_path:
        geometry = load_real_map(map_path)
        oracle = RectangleUnionOracle.from_geometry(geometry)
        roi = np.array([[-3.,-3.,0.],[3.,3.,3.]])
        restricted = oracle.restricted(roi)
        p = rng.uniform(roi[0],roi[1],size=(128,3))
        a,b = oracle.evaluate(torch.tensor(p)),restricted.evaluate(torch.tensor(p))
        assert torch.equal(a["distance"],b["distance"]) and torch.equal(a["gradient"],b["gradient"])
        expected = geometry.distance_numpy(p)
        assert np.max(np.abs(a["distance"].numpy()-expected)) < 2e-15
        ordinary = torch.tensor(p[~a["tie"].numpy()][:8],requires_grad=True)
        assert torch.autograd.gradcheck(restricted.query,(ordinary,),eps=1e-6,atol=1e-7,rtol=1e-6)
        result["real_map_public_arrays"] = {"points":len(p),"rectangles":len(oracle.lower),
            "solids":len(oracle.solid_boxes),"restriction":restricted.restriction,
            "source_sha256":geometry.metadata["source_sha256"],
            "note":"Same rectangle construction as earlier geometry; analytic tests above are independent query truth."}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = run(args.map)
    encoded = json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded+"\n")
    print(encoded)
