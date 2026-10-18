"""Check the sampled trajectory -> field -> scalar-cost autograd chain.

Run through the experiment driver inside Docker. This is neither a continuous
clearance solver nor an analytic trajectory integral. The default fixture passes
above the box centred at (.08, -.06, .02), half-size (.27, .21, .32), inside
[-.8, .8]^3. Its nonzero boundary derivatives make duration affect the path even
though sample fractions u are fixed. Only the public field query is used.
"""

from dataclasses import asdict, dataclass
import json
from math import comb, isfinite
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import softplus

from devtools.trajectory import TrajectoryState, fit_trajectory


@dataclass(frozen=True)
class TrajectoryCheckFixture:
    initial_position: tuple[float, float, float] = (-.65, .43, .48)
    initial_velocity: tuple[float, float, float] = (.10, -.06, -.04)
    initial_acceleration: tuple[float, float, float] = (.03, .02, -.01)
    terminal_position: tuple[float, float, float] = (.65, .40, .45)
    terminal_velocity: tuple[float, float, float] = (.14, .05, -.02)
    terminal_acceleration: tuple[float, float, float] = (-.02, .03, .04)
    duration_s: float = 1.4
    sample_count: int = 73
    clearance_m: float = .10
    cost_scale_m: float = .15


PARAMETER_NAMES = tuple(f"terminal_{name}_{axis}"
                        for name in ("position", "velocity", "acceleration")
                        for axis in "xyz") + ("duration_s",)


def _bernstein_controls(coefficients: Tensor) -> Tensor:
    """Convert the existing normalized-u power polynomial to Bernstein form.

    If every control point lies in the domain box, the entire curve does too.
    This sufficient convex-hull bound says nothing about obstacle clearance.
    """
    degree = coefficients.shape[0] - 1
    return torch.stack([
        sum(coefficients[j] * (comb(k, j) / comb(degree, j))
            for j in range(k + 1))
        for k in range(degree + 1)
    ])


def _json_safe(value):
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_trajectory_checks(field, device, output_dir, *,
                          fixture: TrajectoryCheckFixture | None = None) -> dict:
    """Return and save float64 gradient, cross-leaf and descent evidence.

    field.bounds must be (lo[3], hi[3]); field.query(points[N,3]) must return
    distances[N] without casting away float64; field.leaf_ids(points) returns
    nonnegative integer leaf IDs. The driver calls this separately for each
    supported device. No field moves, rebuilding or fallback occur here.

    Ten differentiated scalars are terminal p/v/a (xyz each) and duration T.
    The exact checked objective is mean(softplus((clearance-d(x(u*T)))/scale)).
    Fractions u are fixed; both the fit and the sample times depend on T.
    It is a finite sampled cost, with no T multiplier and no direct parameter
    regularizer that could conceal a disconnected field query.
    """
    fixture = fixture or TrajectoryCheckFixture()
    device = torch.device(device)
    dtype = torch.float64
    if fixture.sample_count < 3 or fixture.cost_scale_m <= 0:
        raise ValueError("need at least three samples and positive cost scale")
    lo, hi = (torch.as_tensor(value, dtype=dtype, device=device)
              for value in field.bounds)
    if (lo.shape != (3,) or hi.shape != (3,)
            or not bool(torch.isfinite(lo).all() & torch.isfinite(hi).all())
            or not bool((hi > lo).all())):
        raise ValueError("field.bounds must contain finite ordered xyz bounds")
    initial = TrajectoryState(
        *(torch.tensor(getattr(fixture, f"initial_{name}"), dtype=dtype, device=device)
          for name in ("position", "velocity", "acceleration")),
        torch.tensor(.13, dtype=dtype, device=device),
        torch.tensor(.07, dtype=dtype, device=device),
    )
    base = torch.tensor((*fixture.terminal_position, *fixture.terminal_velocity,
                         *fixture.terminal_acceleration, fixture.duration_s),
                        dtype=dtype, device=device)
    fractions = torch.linspace(0., 1., fixture.sample_count, dtype=dtype, device=device)
    minimum_domain_margin = float("inf")

    def evaluate(parameters):
        nonlocal minimum_domain_margin
        terminal = TrajectoryState(parameters[:3], parameters[3:6], parameters[6:9],
                                   initial.yaw + .2, initial.yaw_rate - .03)
        trajectory = fit_trajectory(initial, terminal, parameters[9])
        controls = _bernstein_controls(trajectory.position_coefficients)
        margin = float(torch.minimum(controls - lo, hi - controls).min().detach())
        if margin <= 0:
            raise ValueError("trajectory's Bernstein hull is not strictly inside field bounds")
        minimum_domain_margin = min(minimum_domain_margin, margin)
        points = trajectory.sample(fractions * parameters[9]).position
        distances = field.query(points)
        if not isinstance(distances, Tensor) or distances.shape != (fixture.sample_count,):
            raise TypeError("field.query must return one Tensor distance per sample")
        if distances.dtype != dtype or distances.device != points.device:
            raise TypeError("field.query must preserve float64 and the query device")
        if not bool(torch.isfinite(distances).all()):
            raise ValueError("field query returned nonfinite trajectory distances")
        cost = softplus((fixture.clearance_m - distances) / fixture.cost_scale_m).mean()
        return cost, points, distances, controls

    parameters = base.clone().requires_grad_(True)
    cost, points, distances, controls = evaluate(parameters)
    points.retain_grad()
    if not distances.requires_grad or not cost.requires_grad:
        raise RuntimeError("field.query disconnected the trajectory autograd chain")
    distances.retain_grad()
    cost.backward()
    if parameters.grad is None or points.grad is None or distances.grad is None:
        raise RuntimeError("backward did not reach all required chain stages")
    gradient = parameters.grad.detach().clone()

    # Two central-difference step sizes expose an unstable comparison instead of
    # accepting a single accidentally favourable step. All probes refit and query.
    relative_step, atol, rtol = 2e-5, 2e-6, 1e-3
    steps = relative_step * torch.maximum(base.abs(), torch.ones_like(base))
    finite_differences, probe_costs = [], []
    with torch.no_grad():
        for multiplier in (1., .5):
            differences, pairs = [], []
            for index in range(base.numel()):
                delta = torch.zeros_like(base)
                delta[index] = steps[index] * multiplier
                plus = evaluate(base + delta)[0]
                minus = evaluate(base - delta)[0]
                differences.append((plus - minus) / (2 * delta[index]))
                pairs.append([float(minus), float(plus)])
            finite_differences.append(torch.stack(differences))
            probe_costs.append(pairs)
    fd, fd_half = finite_differences
    errors = torch.stack([(gradient - item).abs() for item in finite_differences])
    tolerances = torch.stack([atol + rtol * item.abs() for item in finite_differences])
    per_parameter_pass = (errors <= tolerances).all(dim=0)
    stable_fd = bool(((fd - fd_half).abs() <= 2 * (atol + rtol * fd_half.abs())).all())

    leaf_ids = field.leaf_ids(points.detach())
    if (not isinstance(leaf_ids, Tensor) or leaf_ids.shape != (fixture.sample_count,)
            or leaf_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                                      torch.uint8)):
        raise TypeError("field.leaf_ids must return an integer Tensor[N]")
    unique_leaf_count = int(torch.unique(leaf_ids).numel())
    spans = points.detach().amax(dim=0) - points.detach().amin(dim=0)
    active = {name: bool(torch.linalg.vector_norm(gradient[start:end]) > 1e-9)
              for name, start, end in (("terminal_position", 0, 3),
                                       ("terminal_velocity", 3, 6),
                                       ("terminal_acceleration", 6, 9),
                                       ("duration", 9, 10))}

    # A deliberately small optimization smoke check, not a planner. Backtracking
    # rejects out-of-domain hulls and requires Armijo decrease of the same cost.
    current, cost_history, optimization_steps = base.clone(), [float(cost.detach())], []
    for iteration in range(3):
        current = current.detach().requires_grad_(True)
        before = evaluate(current)[0]
        derivative, = torch.autograd.grad(before, current)
        if not bool(torch.isfinite(derivative).all()):
            break
        squared_norm = float(derivative.square().sum().detach())
        step_size, accepted = .03, False
        for backtracks in range(12):
            candidate = current.detach() - step_size * derivative.detach()
            if float(candidate[9]) <= 0:
                step_size *= .5
                continue
            try:
                with torch.no_grad():
                    after = evaluate(candidate)[0]
            except ValueError as error:
                if "Bernstein hull" not in str(error):
                    raise
                step_size *= .5
                continue
            if (float(after) <= float(before.detach()) - 1e-4 * step_size * squared_norm
                    and float(after) < cost_history[-1] - 1e-12):
                accepted = True
                current = candidate
                cost_history.append(float(after))
                optimization_steps.append({"iteration": iteration + 1,
                                           "step_size": step_size, "backtracks": backtracks,
                                           "cost": float(after),
                                           "parameters": candidate.cpu().tolist()})
                break
            step_size *= .5
        if not accepted:
            break

    checks = {
        "float64_chain": True,
        "finite_gradients_all_stages": bool(torch.isfinite(gradient).all()
                                            & torch.isfinite(points.grad).all()
                                            & torch.isfinite(distances.grad).all()),
        "all_parameter_groups_active": all(active.values()),
        "central_fd_all_ten_parameters_both_steps": bool(per_parameter_pass.all()),
        "central_fd_stable_between_steps": stable_fd,
        "xyz_position_variation": bool((spans > 1e-4).all()),
        "valid_leaf_ids": bool((leaf_ids >= 0).all()),
        "trajectory_crosses_leaves": unique_leaf_count > 1,
        "entire_curve_hulls_in_domain_for_all_accepted_evaluations": minimum_domain_margin > 0,
        "three_accepted_descent_steps": len(optimization_steps) == 3,
        "sampled_cost_decreased": cost_history[-1] < cost_history[0] - 1e-10,
    }
    rows = [{"parameter": name, "autograd": float(gradient[i]),
             "step": float(steps[i]), "fd": float(fd[i]), "fd_half_step": float(fd_half[i]),
             "absolute_error": errors[:, i].cpu().tolist(),
             "tolerance": tolerances[:, i].cpu().tolist(),
             "minus_plus_costs": [pairs[i] for pairs in probe_costs],
             "passed": bool(per_parameter_pass[i])}
            for i, name in enumerate(PARAMETER_NAMES)]
    result = {
        "passed": all(checks.values()), "checks": checks, "device": str(device),
        "dtype": str(dtype), "fixture": asdict(fixture),
        "scope": "sampled cost gradient and descent only; no continuous clearance or integral claim",
        "cost_definition": "mean(softplus((clearance_m - field.query(x(u*T))) / cost_scale_m))",
        "duration_semantics": "u fixed; T differentiated through both fit_trajectory and sample(u*T)",
        "parameters_initial": base.cpu().tolist(),
        "gradients": {"rows": rows, "active_groups": active,
                      "fd_atol": atol, "fd_rtol": rtol,
                      "maximum_absolute_error": float(errors.max())},
        "trajectory": {
            "fractions": fractions.cpu().tolist(), "positions": points.detach().cpu().tolist(),
            "distances": distances.detach().cpu().tolist(), "leaf_ids": leaf_ids.cpu().tolist(),
            "unique_leaf_count": unique_leaf_count,
            "leaf_transitions": int((leaf_ids[1:] != leaf_ids[:-1]).sum()),
            "position_spans_xyz_m": spans.cpu().tolist(),
            "minimum_sampled_distance_m": float(distances.detach().min()),
            "cost_position_gradients": points.grad.cpu().tolist(),
            "initial_bernstein_controls": controls.detach().cpu().tolist(),
            "minimum_hull_domain_margin_all_evaluations_m": minimum_domain_margin,
        },
        "optimization": {"accepted_steps": optimization_steps, "cost_history": cost_history,
                         "initial_cost": cost_history[0], "final_cost": cost_history[-1]},
    }
    result = _json_safe(result)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"trajectory-checks-{str(device).replace(':', '-')}-float64.json"
    result["artifact"] = str(path)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
