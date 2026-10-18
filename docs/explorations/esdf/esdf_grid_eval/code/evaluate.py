"""Run the bounded CPU path-integral experiment and preserve its evidence."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from docs.explorations.esdf.esdf_grid_eval.code.c1_esdf import build_esdf
from devtools.trajectory import TrajectoryState, fit_trajectory
from .path_integrals import (cell_distance_bernstein, gauss_integral,
                             polynomial_integral, split_curve)


def quadratic_numpy(p):
    x, y, z = p.T
    return .125 + .125*x + .0625*y*y + .03125*z*z - .0625*x*z


def _convolve(a, b):
    result = a.new_zeros(a.numel() + b.numel() - 1)
    for i in range(a.numel()):
        result = result + torch.nn.functional.pad(a[i] * b, (i, a.numel() - i - 1))
    return result


def quadratic_closed_form(trajectory):
    """Independent low-degree power integral of the known analytic fixture."""
    x, y, z = trajectory.position_coefficients.T
    d = .0625*_convolve(y, y) + .03125*_convolve(z, z) - .0625*_convolve(x, z)
    d = d + torch.nn.functional.pad(.125*x, (0, 5))
    constant = d.new_zeros(11)
    constant[0] = .125 - .1
    squared = .5 * _convolve(d + constant, d + constant)
    return trajectory.duration * torch.sum(squared / torch.arange(1, 22, dtype=d.dtype))


def fixture_curves():
    # Explicit normalized-time polynomials; converted to boundary p/v/a below.
    return {
        "three_axis_quintic": np.array([
            [-.9, -.8, -.6], [.25, .4, .3], [.1, -.1, .05],
            [11., 9., 8.], [-16.5, -13.5, -12.], [6.6, 5.4, 4.8]]),
        "return_across_faces": np.array([
            [-.7, -.8, -.6], [6., 1.6, 1.2], [-6., 0., 0.],
            [.3, .2, -.1], [-.6, -.4, .2], [.3, .2, -.1]]),
        "constant_z_off_faces": np.array([
            [-.85, -.65, .375], [1.7, 1.3, 0.], [.1, -.2, 0.],
            [.3, .2, 0.], [-.7, .2, 0.], [.3, -.2, 0.]])}


def boundaries(c, duration=1.4):
    zero = torch.tensor(0., dtype=torch.float64)
    c = torch.tensor(c, dtype=torch.float64)
    initial = TrajectoryState(c[0], c[1]/duration, 2*c[2]/duration**2, zero, zero)
    terminal_p = c.sum(0)
    terminal_v = (c[1:] * torch.arange(1, 6)[:, None]).sum(0)/duration
    terminal_a = (c[2:] * torch.tensor([2, 6, 12, 20])[:, None]).sum(0)/duration**2
    return initial, torch.cat((terminal_p, terminal_v, terminal_a, zero.new_tensor([duration])))


def fit(initial, theta):
    zero = theta.new_tensor(0.)
    return fit_trajectory(initial, TrajectoryState(theta[:3], theta[3:6], theta[6:9], zero, zero), theta[9])


def scalar_and_gradient(initial, theta, callback):
    theta = theta.detach().clone().requires_grad_(True)
    value = callback(fit(initial, theta))
    grad, = torch.autograd.grad(value, theta)
    return float(value.detach()), grad.detach().numpy()


def run_case(field, samples, field_name, name, coefficients, raw):
    initial, theta = boundaries(coefficients)
    trajectory = fit(initial, theta)
    pieces, split_diagnostics = split_curve(field, trajectory.position_coefficients)
    integral = lambda t: polynomial_integral(field, t.position_coefficients, t.duration, samples=samples)
    value, gradient = scalar_and_gradient(initial, theta, integral)
    checks, comparisons = {}, {}
    for order in (48, 64):
        reference, reference_grad = scalar_and_gradient(
            initial, theta, lambda t: gauss_integral(field, t, pieces, order=order))
        comparisons[f"gauss_{order}"] = {"cost": reference, "gradient": reference_grad.tolist(),
            "absolute_cost_difference": abs(value-reference),
            "max_gradient_difference": float(np.max(np.abs(gradient-reference_grad)))}
        checks[f"gauss_{order}"] = bool(abs(value-reference) < 2e-10 and
                                       np.max(np.abs(gradient-reference_grad)) < 2e-9)
    finite_difference = []
    step = 1e-6
    for parameter in range(10):
        plus, minus = theta.clone(), theta.clone()
        plus[parameter] += step
        minus[parameter] -= step
        with torch.no_grad():
            finite_difference.append(float((integral(fit(initial, plus))-integral(fit(initial, minus)))/(2*step)))
    finite_difference = np.array(finite_difference)
    fd_error = np.abs(gradient-finite_difference)
    checks["parameter_fd"] = bool(np.all(fd_error < 2e-7 + 2e-5*np.abs(finite_difference)))
    if field_name == "quadratic_scalar_not_sdf":
        analytic, analytic_grad = scalar_and_gradient(initial, theta, quadratic_closed_form)
        comparisons["independent_quadratic_closed_form"] = {
            "cost": analytic, "gradient": analytic_grad.tolist(),
            "absolute_cost_difference": abs(value-analytic),
            "max_gradient_difference": float(np.max(np.abs(gradient-analytic_grad)))}
        checks["independent_quadratic"] = bool(abs(value-analytic) < 2e-10 and
                                               np.max(np.abs(gradient-analytic_grad)) < 2e-9)
    exp_values = {}
    for order in (32, 64):
        exp_value, exp_gradient = scalar_and_gradient(initial, theta,
            lambda t: gauss_integral(field, t, pieces, order=order, penalty="exp"))
        exp_values[str(order)] = {"cost": exp_value, "gradient": exp_gradient.tolist()}
    exp_difference = abs(exp_values["32"]["cost"] - exp_values["64"]["cost"])
    exp_grad_difference = np.max(np.abs(np.array(exp_values["32"]["gradient"])
                                          - np.array(exp_values["64"]["gradient"])))
    checks["exp_quadrature_convergence_observed"] = bool(exp_difference < 2e-9 and exp_grad_difference < 2e-8)
    distances = [cell_distance_bernstein(field, samples, trajectory.position_coefficients, p).detach().numpy()
                 for p in pieces]
    jumps = [abs(a[-1]-b[0]) for a, b in zip(distances[:-1], distances[1:])]
    checks["crossing_c0"] = bool(max(jumps, default=0) < 2e-10)
    # Coefficient-level autograd: compare public query against the composed polynomial.
    c = trajectory.position_coefficients.detach().clone().requires_grad_(True)
    direct = polynomial_integral(field, c, trajectory.duration, pieces=pieces, samples=samples)
    direct_grad, = torch.autograd.grad(direct, c)
    from devtools.trajectory import PolynomialTrajectory
    c_query = c.detach().clone().requires_grad_(True)
    t_query = PolynomialTrajectory(c_query, trajectory.yaw_coefficients, trajectory.duration)
    q_grad, = torch.autograd.grad(gauss_integral(field, t_query, pieces), c_query)
    coef_error = float((direct_grad-q_grad).abs().max())
    checks["coefficient_gradient"] = coef_error < 2e-9
    key = field_name + "__" + name
    raw[key+"__pieces"] = np.array([(a, b, *cell) for a, b, cell in pieces])
    raw[key+"__distance_bernstein"] = np.array(distances)
    raw[key+"__coefficients"] = trajectory.position_coefficients.detach().numpy()
    raw[key+"__coefficient_gradient"] = direct_grad.detach().numpy()
    return {"field": field_name, "curve": name, "segments": len(pieces),
        "cost": value, "parameter_order": ["px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az", "T"],
        "gradient": gradient.tolist(), "finite_difference": finite_difference.tolist(),
        "fd_step": step, "max_fd_difference": float(fd_error.max()),
        "max_coefficient_gradient_difference": coef_error,
        "max_crossing_value_jump": max(jumps, default=0), "comparisons": comparisons,
        "exp_quadrature_only": {"orders": exp_values, "cost_difference": exp_difference,
                                 "max_gradient_difference": float(exp_grad_difference)},
        "split_diagnostics": split_diagnostics, "checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise FileExistsError("use a fresh output directory; evidence is not overwritten")
    torch.set_num_threads(1)
    callbacks = {
        "quadratic_scalar_not_sdf": quadratic_numpy,
        "sphere_sdf_samples": lambda p: np.linalg.norm(p-np.array([.2, -.15, .1]), axis=1)-.55}
    records, raw, failure = [], {}, None
    for field_name, callback in callbacks.items():
        field = build_esdf(callback, bounds_m=((-1.5,)*3, (1.5,)*3), voxel_size_m=.25)
        public_samples = field.samples
        raw[field_name+"__samples"] = public_samples.numpy()
        samples = public_samples.to(dtype=torch.float64)
        for name, coefficients in fixture_curves().items():
            try:
                result = run_case(field, samples, field_name, name, coefficients, raw)
            except Exception as error:
                failure = {"field": field_name, "curve": name,
                           "exception": type(error).__name__, "message": str(error)}
                break
            records.append(result)
            print(json.dumps({"field": field_name, "curve": name, "segments": result["segments"],
                              "passed": result["passed"], "checks": result["checks"]}), flush=True)
        if failure is not None:
            break
    rejected = {}
    for name, c in {"tangent": [[.75, .125, .125], [-1, 0, 0], [1, 0, 0], [0]*3, [0]*3, [0]*3],
                    "outside_domain": [[2, 0, 0], [0]*3, [0]*3, [0]*3, [0]*3, [0]*3]}.items():
        try:
            split_curve(field, torch.tensor(c, dtype=torch.float64))
        except ValueError as error:
            rejected[name] = str(error)
    source_paths = [Path(__file__), Path(__file__).with_name("path_integrals.py"),
                    Path("docs/explorations/esdf/esdf_grid_eval/code/c1_esdf.py"), Path("devtools/trajectory.py")]
    result = {"schema": "esdf-grid-path-integral-experiment-v1", "device": "cpu", "dtype": "float64",
        "samples_dtype": "float32", "spacing_m": .25, "bounds_m": [[-1.5]*3, [1.5]*3],
        "integral": "T * integral_0^1 0.5*(F(p(u))-0.1)^2 du",
        "meaning": "polynomial integration demonstration, not an obstacle loss or collision guarantee",
        "method": "numerical grid-face roots; per-cell Bernstein composition and coefficient-average integral",
        "root_limit": "not certified; near tangencies rejected; no sampling-based coverage certificate",
        "gauss_limit": "48-point segment GL integrates degree<=95 in exact arithmetic; still finite evaluations",
        "exp_limit": "32/64-point numerical quadrature comparison only; no analytic exp integral",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "image_id": os.environ.get("ESDF_EXPERIMENT_IMAGE_ID", "unrecorded"),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "cases": records, "rejected_cases": rejected, "failure": failure,
        "passed": failure is None and len(records) == 6 and all(r["passed"] for r in records) and len(rejected) == 2}
    np.savez_compressed(args.output / "raw.npz", **raw)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    if not result["passed"]:
        raise SystemExit("one or more checks failed; results preserved")


if __name__ == "__main__":
    main()
