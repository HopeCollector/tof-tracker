"""CPU checks for experimental cell-split exponential trajectory means.

References: an analytic plane/linear-curve exponential and a uniform composite
Gauss rule over the whole time domain that does not use cell roots. Parameter
finite differences rebuild the partition for every perturbation. These checks
do not certify root completeness, collision safety or geometric SDF accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch

from devtools.esdf import build_esdf
from devtools.trajectory import TrajectoryState, fit_trajectory
from experiments.esdf_curve_integral.integrate import (
    CurveSplitError, integrate_curves, integrate_segments, prepare_quadrature, split_batch,
)


def _value_gradient(c, callback):
    c = c.detach().clone().requires_grad_(True)
    value = callback(c)
    gradient, = torch.autograd.grad(value.sum(), c)
    return value.detach().numpy(), gradient.detach().numpy()


def _close(actual, expected, *, atol, rtol=0.):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def _independent_mean(field, c, panels):
    """Uniform composite GL8: independent time nodes, no partition input."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    u = c.new_tensor(((np.arange(panels)[:, None]+(nodes+1.)/2.)/panels).ravel())
    w = c.new_tensor(np.tile(weights/(2.*panels), panels))
    point = c[:, 5, None, :]
    for power in range(4, -1, -1):
        point = point*u[None, :, None]+c[:, power, None, :]
    return (torch.exp(-(field.query(point)-1.2)/.6)*w).sum(-1)


def _save_partition(raw, name, c, segments):
    raw[name+"_coefficients"] = c.detach().numpy()
    raw[name+"_intervals"] = segments.intervals
    raw[name+"_cells_xyz"] = segments.cells_xyz
    raw[name+"_curve_indices"] = segments.curve_indices
    for order in (4, 8, 16, 32):
        plan = prepare_quadrature(segments, order=order)
        raw[f"{name}_gl{order}_u"] = plan.u.numpy()
        raw[f"{name}_gl{order}_weights"] = plan.weights.numpy()


def _plane_check(raw):
    field = build_esdf(lambda p: p[:, 2]-.125, bounds_m=((-1.5,)*3, (1.5,)*3), voxel_size_m=.25)
    c = torch.zeros((3, 6, 3), dtype=torch.float64)
    c[:, 0] = c.new_tensor([[-.83, -.37, .125], [-.9, -.7, .625], [-.81, -.31, .375]])
    c[:, 1] = c.new_tensor([[1.64, .91, .5], [1.6, 1.1, -.5], [1.55, .82, 0.]])
    expected, expected_gradient = [], np.zeros(c.shape)
    for i in range(len(c)):
        A, B = (1.2-(float(c[i, 0, 2])-.125))/.6, -float(c[i, 1, 2])/.6
        if B == 0.:
            moments = np.array([1./(k+1) for k in range(6)])
        else:
            moments = [np.expm1(B)/B]
            for k in range(1, 6):
                moments.append(np.exp(B)/B-k*moments[-1]/B)
            moments = np.array(moments)
        expected.append(np.exp(A)*moments[0])
        expected_gradient[i, :, 2] = -np.exp(A)*moments/.6
    segments = split_batch(field, c)
    _save_partition(raw, "plane", c, segments)
    records = []
    for order in (4, 8, 16, 32):
        value, gradient = _value_gradient(c, lambda x: integrate_curves(field, x, order=order))
        raw[f"plane_gl{order}_value"], raw[f"plane_gl{order}_gradient"] = value, gradient
        records.append({"order": order, "cost_abs_error": _close(value, expected, atol=2e-10),
                        "coefficient_gradient_abs_error": _close(gradient, expected_gradient, atol=2e-8)})
    raw["plane_closed_form_value"] = np.array(expected)
    raw["plane_closed_form_gradient"] = expected_gradient
    return {"passed": True, "segments": segments.segment_count, "curves": len(c), "orders": records,
            "objective": "normalized_time_mean; no duration factor", "split_diagnostics": segments.diagnostics}


def _curves():
    return torch.tensor([
        [[-.9, -.8, -.6], [.25, .4, .3], [.1, -.1, .05],
         [11., 9., 8.], [-16.5, -13.5, -12.], [6.6, 5.4, 4.8]],
        [[-.7, -.8, -.6], [6., 1.6, 1.2], [-6., 0., 0.],
         [.3, .2, -.1], [-.6, -.4, .2], [.3, .2, -.1]],
    ], dtype=torch.float64)


def _finite_difference(theta, callback, step=2e-6):
    result = np.zeros(theta.shape)
    for index in np.ndindex(tuple(theta.shape)):
        plus, minus = theta.clone(), theta.clone()
        plus[index] += step
        minus[index] -= step
        with torch.no_grad():
            result[index] = float((callback(plus).sum()-callback(minus).sum())/(2.*step))
    return result


def _boundary_parameters(c, duration):
    p0, v0, a0 = c[0], c[1]/duration, 2*c[2]/duration**2
    p1 = c.sum(0)
    v1 = (c[1:]*c.new_tensor([1, 2, 3, 4, 5])[:, None]).sum(0)/duration
    a1 = (c[2:]*c.new_tensor([2, 6, 12, 20])[:, None]).sum(0)/duration**2
    return torch.cat((p0, v0, a0, p1, v1, a1, c.new_tensor([duration])))


def _fit_parameters(theta):
    zero = theta.new_zeros(())
    initial = TrajectoryState(theta[:3], theta[3:6], theta[6:9], zero, zero)
    terminal = TrajectoryState(theta[9:12], theta[12:15], theta[15:18], zero, zero)
    return fit_trajectory(initial, terminal, theta[18]).position_coefficients[None]


def _general_check(raw):
    field = build_esdf(lambda p: np.linalg.norm(p-np.array([.2, -.15, .1]), axis=1)-.55,
                       bounds_m=((-1.5,)*3, (1.5,)*3), voxel_size_m=.25)
    c = _curves()
    segments = split_batch(field, c)
    _save_partition(raw, "general", c, segments)
    raw["general_field_samples"] = field.samples.numpy()
    refs = [_value_gradient(c, lambda x, n=n: _independent_mean(field, x, n)) for n in (512, 1024)]
    ref_value, ref_gradient = refs[-1]
    independent_convergence = {"cost_abs_change": _close(refs[0][0], ref_value, atol=2e-8),
                               "coefficient_gradient_abs_change": _close(refs[0][1], ref_gradient, atol=2e-6)}
    raw["general_independent_value"], raw["general_independent_gradient"] = ref_value, ref_gradient
    records = []
    for order in (4, 8, 16, 32):
        value, gradient = _value_gradient(c, lambda x: integrate_curves(field, x, order=order))
        raw[f"general_gl{order}_value"], raw[f"general_gl{order}_gradient"] = value, gradient
        row = {"order": order, "cost_abs_error": float(np.max(np.abs(value-ref_value))),
               "coefficient_gradient_abs_error": float(np.max(np.abs(gradient-ref_gradient)))}
        if order == 32:
            _close(value, ref_value, atol=5e-8)
            _close(gradient, ref_gradient, atol=3e-6)
        records.append(row)

    one = c[:1].clone()
    cost = lambda x: integrate_curves(field, x, order=32)
    _, coefficient_gradient = _value_gradient(one, cost)
    coefficient_fd = _finite_difference(one, cost)
    coefficient_error = _close(coefficient_gradient, coefficient_fd, atol=2e-6, rtol=2e-5)
    theta = _boundary_parameters(c[0], 1.4)
    endpoint_cost = lambda x: integrate_curves(field, _fit_parameters(x), order=32)
    _, boundary_gradient = _value_gradient(theta, endpoint_cost)
    boundary_fd = _finite_difference(theta, endpoint_cost)
    boundary_error = _close(boundary_gradient, boundary_fd, atol=2e-6, rtol=2e-5)
    raw.update(general_coefficient_fd=coefficient_fd, general_boundary_parameters=theta.numpy(),
               general_boundary_gradient=boundary_gradient, general_boundary_fd=boundary_fd)

    # Float32 has its own represented curve, numerical partition and plan.
    c32 = c.float()
    s32 = split_batch(field, c32)
    p32 = prepare_quadrature(s32, order=32, dtype=torch.float32)
    c32 = c32.requires_grad_(True)
    v32 = integrate_segments(field, c32, p32)
    g32, = torch.autograd.grad(v32.sum(), c32)
    assert bool(torch.isfinite(v32).all() and torch.isfinite(g32).all())
    # A one-curve plan must agree with that row of the batched calculation.
    one_value, one_gradient = _value_gradient(one, cost)
    batch_error = _close(one_value, raw["general_gl32_value"][:1], atol=2e-12)
    _close(one_gradient, raw["general_gl32_gradient"][:1], atol=2e-11)
    return {"passed": True, "segments": segments.segment_count, "orders": records,
            "independent_composite_gl8_panels": [512, 1024], "independent_convergence": independent_convergence,
            "coefficient_fd_max_abs_error": coefficient_error, "boundary_and_duration_fd_max_abs_error": boundary_error,
            "fd_step": 2e-6, "fd_rebuilds_every_partition": True,
            "boundary_parameter_order": "p0[3],v0[3],a0[3],p1[3],v1[3],a1[3],T",
            "float32_own_partition_segments": s32.segment_count, "float32_value_gradient_finite": True,
            "single_vs_batch_cost_abs_error": batch_error, "split_diagnostics": segments.diagnostics}


def _rejection_check():
    field = build_esdf(lambda p: p[:, 2], bounds_m=((-1.5,)*3, (1.5,)*3), voxel_size_m=.25)
    c = _curves()[:1]
    segments = split_batch(field, c)
    plan = prepare_quadrature(segments, order=8)
    tangent = torch.tensor([[[.75, .125, .125], [-1, 0, 0], [1, 0, 0], [0]*3, [0]*3, [0]*3]], dtype=torch.float64)
    outside = c.clone(); outside[:, 0, 0] = 3.
    nonfinite = c.clone(); nonfinite[:, 0, 0] = torch.nan
    changed = c.clone(); changed[:, 0, 0] += 1e-4
    # The recorded real-map polynomial has p'(1)=0, but polyroots returns
    # 0.9999999999999987. Translating its z=3 face to z=.5 preserves that root.
    endpoint = torch.zeros((1, 6, 3), dtype=torch.float64)
    endpoint[0, 0] = endpoint.new_tensor([-.8, .125, .5])
    endpoint[0, 1, 0] = 1.6
    endpoint[0, 2:, 2] = endpoint.new_tensor([
        .020833333333333336, -.07916666666666669, .09583333333333338, -.037500000000000006])
    endpoint_segments = split_batch(field, endpoint)
    contacts = endpoint_segments.diagnostics[0]["endpoint_stationary_contacts"]
    assert any(row["endpoint_u"] == 1. and row["computed_u"] < 1. for row in contacts)
    endpoint_value, endpoint_gradient = _value_gradient(endpoint, lambda x: integrate_curves(field, x, order=32))
    assert np.isfinite(endpoint_value).all() and np.isfinite(endpoint_gradient).all()
    rejected = {}
    cases = {"near_tangency_batch_index_1": lambda: split_batch(field, torch.cat((c, tangent))),
             "outside_domain": lambda: split_batch(field, outside),
             "nonfinite_coefficients": lambda: split_batch(field, nonfinite),
             "empty_batch": lambda: split_batch(field, c[:0]),
             "stale_plan": lambda: integrate_segments(field, changed, plan),
             "changing_plan_dtype": lambda: prepare_quadrature(segments, dtype=torch.float32),
             "unsupported_order": lambda: prepare_quadrature(segments, order=12),
             "penalty_overflow": lambda: integrate_segments(field, c, plan, d0=10000.),
             "nonpositive_penalty_radius": lambda: integrate_segments(field, c, plan, r=0.)}
    for name, action in cases.items():
        try:
            action()
        except (ValueError, FloatingPointError) as error:
            if name == "near_tangency_batch_index_1":
                assert isinstance(error, CurveSplitError) and error.curve_index == 1
            rejected[name] = {"exception": type(error).__name__, "message": str(error)}
        else:
            raise AssertionError(f"{name} was not rejected")
    return {"passed": True, "cases": rejected,
            "near_endpoint_stationary_accepted": contacts,
            "near_endpoint_integral_and_gradient_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    raw, records = {}, {}
    started = time.perf_counter()
    error = None
    try:
        for name, callback in (("plane_closed_form", lambda: _plane_check(raw)),
                               ("general_field_and_fd", lambda: _general_check(raw)),
                               ("rejections", _rejection_check)):
            records[name] = callback()
            print(json.dumps({"check": name, "passed": True}), flush=True)
    except Exception:
        error = traceback.format_exc()
    source_paths = ["experiments/esdf_curve_integral/integrate.py", "tests/verify_esdf_curve_integral.py",
                    "experiments/esdf_grid_eval/path_integrals.py", "devtools/esdf.py", "devtools/trajectory.py"]
    report = {"passed": error is None, "device": "cpu", "python": platform.python_version(),
              "torch": torch.__version__, "numpy": np.__version__, "wall_seconds": time.perf_counter()-started,
              "checks": records, "failure": error,
              "limits": "numerical GL convergence and first gradients; no exact integral, certified roots or collision safety",
              "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}}
    np.savez(args.output / "raw.npz", **raw)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if error is not None:
        raise RuntimeError(error)
    print(json.dumps({"passed": True, "output": str(args.output), "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
