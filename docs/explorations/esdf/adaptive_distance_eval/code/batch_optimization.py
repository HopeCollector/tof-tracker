"""Bounded batch-trajectory optimization experiment, not a flight planner.

The independent oracle is used for task selection and before/after observation,
never for the optimization gradient. All execution belongs inside Docker. The
line search enforces sampled-cost descent; descent is not an empirical safety
guarantee. Dense oracle checks also remain discrete, not continuous clearance.
"""

import json
from math import comb, isfinite
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.functional import softplus

from devtools.trajectory import TrajectoryState, fit_trajectory


PARAMETER_NAMES = tuple(f"terminal_{name}_{axis}"
                        for name in ("position", "velocity", "acceleration")
                        for axis in "xyz") + ("duration",)
DATA_KEYS = ("p0", "v0", "a0", "p_goal", "v_goal", "a_goal", "duration0")
GROUPS = ("clearance_violating", "near", "safe")
WEIGHTS = {"position_anchor": .2, "velocity_anchor": .02,
           "acceleration_anchor": .01, "duration_anchor": .05, "jerk": .01}


def _state(p, v, a):
    scalar = torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
    return TrajectoryState(p, v, a, scalar, scalar.clone())


def _subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def _to(data, *, device, dtype):
    return {key: torch.as_tensor(value, device=device, dtype=dtype)
            for key, value in data.items()}


def _fit(q, data):
    p = data["p_goal"] + .5 * q[:, :3]
    v = data["v_goal"] + 2. * q[:, 3:6]
    a = data["a_goal"] + 2. * q[:, 6:9]
    duration = data["duration0"] * (1. + .2 * q[:, 9])
    return fit_trajectory(_state(data["p0"], data["v0"], data["a0"]),
                          _state(p, v, a), duration)


def _controls(coefficients):
    return torch.stack([sum(coefficients[:, j] * (comb(k, j) / comb(5, j))
                            for j in range(k + 1)) for k in range(6)], dim=1)


def _inside(trajectory, bounds):
    lo, hi = (bound.to(dtype=trajectory.duration.dtype) for bound in bounds)
    controls = _controls(trajectory.position_coefficients)
    return (torch.isfinite(controls).all(dim=(1, 2))
            & (controls > lo).all(dim=(1, 2)) & (controls < hi).all(dim=(1, 2)))


def _oracle_values(oracle, points, chunk_size=4096):
    parts = []
    with torch.no_grad():
        for chunk in points.split(chunk_size):
            result = oracle(chunk)
            if not isinstance(result, torch.Tensor) or result.shape != (len(chunk),):
                raise TypeError("oracle must return Tensor[N] signed distances")
            if result.device != points.device:
                raise ValueError("oracle distances must remain on the query device")
            parts.append(result.detach())
    return torch.cat(parts) if parts else points.new_empty((0,))


def _sample_tasks(oracle, bounds, device, count, seed):
    rng = np.random.default_rng(seed)
    lower, upper = (bound.detach().cpu().numpy() for bound in bounds)
    if (upper - lower <= 1.6).any():
        raise ValueError("logical map must have room for 0.8 m endpoint boundary margins")
    quota = np.full(3, count // 3, dtype=int)
    quota[:count % 3] += 1
    collected = {key: [] for key in DATA_KEYS}
    groups, observed_minima = [], []
    generated = domain_candidates = endpoint_free = hull_free = 0
    classify_u = torch.linspace(0., 1., 17, device=device, dtype=torch.float64)
    counts = np.zeros(3, dtype=int)
    # Endpoint batches contain at most 1024 points; there is no per-pair query.
    while generated < 2048 and not np.all(counts >= quota):
        number = min(512, 2048 - generated)
        generated += number
        p0 = rng.uniform(lower + .8, upper - .8, (number, 3))
        direction = rng.normal(size=(number, 3))
        direction[:, 2] *= .5
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        length = rng.uniform(4., 8., number)
        goal = p0 + length[:, None] * direction
        valid = ((goal > lower + .8) & (goal < upper - .8)).all(axis=1)
        p0, goal, direction, length = p0[valid], goal[valid], direction[valid], length[valid]
        domain_candidates += len(p0)
        if not len(p0):
            continue
        endpoints = torch.tensor(np.concatenate((p0, goal)), device=device, dtype=torch.float64)
        distances = _oracle_values(oracle, endpoints).reshape(2, -1)
        free = (torch.isfinite(distances).all(dim=0) & (distances > .65).all(dim=0)).cpu().numpy()
        p0, goal, direction, length = p0[free], goal[free], direction[free], length[free]
        endpoint_free += len(p0)
        if not len(p0):
            continue
        arrays = {"p0": p0, "v0": direction + rng.normal(0., .08, p0.shape),
                  "a0": rng.uniform(-.15, .15, p0.shape), "p_goal": goal,
                  "v_goal": .8 * direction + rng.normal(0., .08, p0.shape),
                  "a_goal": rng.uniform(-.15, .15, p0.shape), "duration0": length / 2.}
        candidate = _to(arrays, device=device, dtype=torch.float64)
        with torch.no_grad():
            trajectory = _fit(torch.zeros((len(p0), 10), device=device, dtype=torch.float64), candidate)
            inside = _inside(trajectory, bounds)
            keep = inside.cpu().numpy()
            hull_free += int(keep.sum())
            if not keep.any():
                continue
            positions = trajectory.sample(trajectory.duration[:, None] * classify_u).position[inside]
            values = _oracle_values(oracle, positions.reshape(-1, 3)).reshape(-1, 17)
            minima = values.amin(dim=1)
            finite = torch.isfinite(values).all(dim=1).cpu().numpy()
            minima = minima.cpu().numpy()[finite]
        selected_arrays = {key: value[keep][finite] for key, value in arrays.items()}
        labels = np.where(minima < .3, 0, np.where(minima < .8, 1, 2))
        for key in DATA_KEYS:
            collected[key].append(selected_arrays[key])
        groups.append(labels)
        observed_minima.append(minima)
        counts += np.bincount(labels, minlength=3)
    if not groups or sum(len(group) for group in groups) < count:
        raise RuntimeError(f"could not obtain {count} endpoint-free, in-domain tasks from {generated} candidates")
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    labels, minima = np.concatenate(groups), np.concatenate(observed_minima)
    selected = np.concatenate([np.flatnonzero(labels == group)[:quota[group]] for group in range(3)])
    if len(selected) < count:
        # A scarce category is reported, never invented or filled from after-run successes.
        available = np.setdiff1d(np.arange(len(labels)), selected)
        selected = np.concatenate((selected, available[:count - len(selected)]))
    selected_labels = labels[selected]
    return ({key: value[selected] for key, value in arrays.items()}, selected_labels,
            {"generated_candidate_pairs": generated, "domain_candidate_pairs": domain_candidates,
             "endpoint_free_pairs": endpoint_free, "in_domain_hull_pairs": hull_free,
             "classification_samples": 17, "classification_is_discrete": True,
             "requested_group_counts": dict(zip(GROUPS, quota.tolist())),
             "available_group_counts": dict(zip(GROUPS, counts.tolist())),
             "selected_group_counts": {name: int((selected_labels == i).sum())
                                       for i, name in enumerate(GROUPS)},
             "all_requested_groups_filled": bool(np.all(counts >= quota)),
             "initial_classification_min_distance_m": minima[selected].tolist()})


def _evaluate(field, q, data, fractions, bounds):
    trajectory = _fit(q, data)
    inside = _inside(trajectory, bounds)
    sample = trajectory.sample(trajectory.duration[:, None] * fractions)
    positions = sample.position
    indices = torch.nonzero(inside, as_tuple=False).flatten()
    obs = q.new_full((len(q),), float("inf"))
    finite_query = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    if len(indices):
        distances = field.query(positions[indices].reshape(-1, 3)).reshape(len(indices), -1)
        if distances.dtype != q.dtype or distances.device != q.device:
            raise TypeError("field.query must preserve query dtype and device")
        finite = torch.isfinite(distances).all(dim=1)
        # Mask nonfinite distances before evaluating the penalty. Such rows are
        # still rejected explicitly, rather than receiving a favourable cost.
        safe = torch.where(torch.isfinite(distances), distances, torch.zeros_like(distances))
        values = softplus((.3 - safe) / .1).square().mean(dim=1)
        values = torch.where(finite, values, torch.full_like(values, float("inf")))
        obs = obs.index_copy(0, indices, values)
        finite_query[indices] = finite
    c = trajectory.position_coefficients
    u = fractions[None, :, None]
    jerk = ((60. * c[:, 5, None] * u + 24. * c[:, 4, None]) * u
            + 6. * c[:, 3, None]) / trajectory.duration[:, None, None].pow(3)
    components = {"obstacle": obs,
                  "position_anchor": WEIGHTS["position_anchor"] * q[:, :3].square().sum(dim=1),
                  "velocity_anchor": WEIGHTS["velocity_anchor"] * q[:, 3:6].square().sum(dim=1),
                  "acceleration_anchor": WEIGHTS["acceleration_anchor"] * q[:, 6:9].square().sum(dim=1),
                  "duration_anchor": WEIGHTS["duration_anchor"] * q[:, 9].square(),
                  "jerk": WEIGHTS["jerk"] * jerk.square().sum(dim=-1).mean(dim=1) / 4.}
    total = sum(components.values())
    finite = inside & finite_query & torch.isfinite(total)
    return {"total": total, "obstacle": obs, "components": components,
            "inside": inside, "finite": finite, "positions": positions}


def _optimize(field, data, bounds, steps, free_parameters):
    device = data["p0"].device
    q = torch.zeros((len(data["p0"]), 10), device=device, dtype=torch.float32)
    fractions = torch.linspace(0., 1., 96, device=device, dtype=q.dtype)
    mask = torch.tensor(free_parameters, device=device, dtype=q.dtype)
    with torch.no_grad():
        initial = _evaluate(field, q, data, fractions, bounds)
    history = {"q": [q.clone()], "cost": [initial["total"].clone()],
               "obstacle": [initial["obstacle"].clone()], "finite": [initial["finite"].clone()]}
    updates = {key: [] for key in ("step_size", "reject_count", "domain_reject_count",
                                   "nonfinite_trial_count", "accepted", "stalled",
                                   "finite_gradient", "gradient_norm")}
    for _ in range(steps):
        current = q.detach().requires_grad_(True)
        value = _evaluate(field, current, data, fractions, bounds)
        if bool(value["finite"].any()):
            gradient, = torch.autograd.grad(value["total"][value["finite"]].sum(), current)
        else:
            gradient = torch.zeros_like(current)
        gradient = gradient * mask
        finite_gradient = torch.isfinite(gradient).all(dim=1)
        usable = value["finite"] & finite_gradient
        gradient = torch.where(usable[:, None], gradient, torch.zeros_like(gradient)).detach()
        active = usable & (gradient.square().sum(dim=1) > 1e-18)
        next_q = current.detach().clone()
        next_cost, next_obs = value["total"].detach().clone(), value["obstacle"].detach().clone()
        next_finite = value["finite"].detach().clone()
        alpha = q.new_full((len(q),), .05)
        accepted = torch.zeros_like(active)
        step_size = torch.zeros_like(alpha)
        rejected = torch.zeros(len(q), dtype=torch.int64, device=device)
        domain_rejected, nonfinite_trials = rejected.clone(), rejected.clone()
        for _backtrack in range(12):
            ids = torch.nonzero(active, as_tuple=False).flatten()
            if not len(ids):
                break
            candidate = ((current.detach()[ids] - alpha[ids, None] * gradient[ids]).clamp(-1., 1.) * mask)
            displacement = candidate - current.detach()[ids]
            slope = (gradient[ids] * displacement).sum(dim=1)
            with torch.no_grad():
                trial = _evaluate(field, candidate, _subset(data, ids), fractions, bounds)
            armijo = (trial["finite"] & (slope < -1e-14)
                      & (trial["total"] <= value["total"].detach()[ids] + 1e-4 * slope)
                      & (trial["total"] < value["total"].detach()[ids]))
            good, bad = ids[armijo], ids[~armijo]
            next_q[good] = candidate[armijo]
            next_cost[good], next_obs[good] = trial["total"][armijo], trial["obstacle"][armijo]
            next_finite[good] = trial["finite"][armijo]
            accepted[good], step_size[good] = True, alpha[good]
            active[good] = False
            rejected[bad] += 1
            domain_rejected[ids[~trial["inside"]]] += 1
            nonfinite_trials[ids[trial["inside"] & ~trial["finite"]]] += 1
            alpha[bad] *= .5
        q = next_q
        for key, item in (("q", q), ("cost", next_cost), ("obstacle", next_obs), ("finite", next_finite)):
            history[key].append(item.detach().clone())
        for key, item in (("step_size", step_size), ("reject_count", rejected),
                          ("domain_reject_count", domain_rejected), ("nonfinite_trial_count", nonfinite_trials),
                          ("accepted", accepted), ("stalled", ~accepted),
                          ("finite_gradient", finite_gradient),
                          ("gradient_norm", torch.linalg.vector_norm(gradient, dim=1))):
            updates[key].append(item.detach().clone())
    with torch.no_grad():
        final = _evaluate(field, q, data, fractions, bounds)
    arrays = {key: torch.stack(value).cpu().numpy() for key, value in history.items()}
    arrays.update({key: torch.stack(value).cpu().numpy() for key, value in updates.items()})
    arrays["initial_components"] = torch.stack(list(initial["components"].values()), dim=1).cpu().numpy()
    arrays["final_components"] = torch.stack(list(final["components"].values()), dim=1).cpu().numpy()
    return q, arrays


def _dense_observation(oracle, q, data):
    data64 = _to(data, device=q.device, dtype=torch.float64)
    # 511 midpoint interiors share no nodes with the 96-point training grid;
    # the two physical endpoints are intentionally checked in both grids.
    fractions = torch.cat((q.new_tensor([0.], dtype=torch.float64),
                           (torch.arange(511, device=q.device, dtype=torch.float64) + .5) / 511.,
                           q.new_tensor([1.], dtype=torch.float64)))
    with torch.no_grad():
        trajectory = _fit(q.double(), data64)
        sample = trajectory.sample(trajectory.duration[:, None] * fractions)
        values = _oracle_values(oracle, sample.position.reshape(-1, 3)).reshape(len(q), 513)
        result = {"fractions": fractions.cpu().numpy(), "distance": values.cpu().numpy(),
                  "duration": trajectory.duration.cpu().numpy()}
        for name in ("position", "velocity", "acceleration"):
            result[name] = getattr(sample, name).cpu().numpy()
    return result


def _statistics(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    subset = values[finite]
    return {"minimum": float(subset.min()) if subset.size else None,
            "median": float(np.median(subset)) if subset.size else None,
            "p95": float(np.percentile(subset, 95)) if subset.size else None,
            "maximum": float(subset.max()) if subset.size else None,
            "nonfinite_count": int((~finite).sum())}


def _outcomes(history, before, after, groups):
    minimum_before, minimum_after = before["distance"].min(axis=1), after["distance"].min(axis=1)
    improvement = minimum_after - minimum_before
    rows = []
    for i in range(len(groups)):
        rows.append({"index": i, "selection_group": GROUPS[int(groups[i])],
                     "cost_before": float(history["cost"][0, i]), "cost_after": float(history["cost"][-1, i]),
                     "obstacle_before": float(history["obstacle"][0, i]), "obstacle_after": float(history["obstacle"][-1, i]),
                     "observed_min_mesh_distance_before_m": float(minimum_before[i]),
                     "observed_min_mesh_distance_after_m": float(minimum_after[i]),
                     "observed_vehicle_clearance_before_m": float(minimum_before[i] - .2),
                     "observed_vehicle_clearance_after_m": float(minimum_after[i] - .2),
                     "required_clearance_surplus_before_m": float(minimum_before[i] - .3),
                     "required_clearance_surplus_after_m": float(minimum_after[i] - .3),
                     "observed_min_distance_change_m": float(improvement[i]),
                     "required_clearance_violating_samples_before": int((before["distance"][i] < .3).sum()),
                     "required_clearance_violating_samples_after": int((after["distance"][i] < .3).sum()),
                     "raw_mesh_penetrating_samples_before": int((before["distance"][i] < 0.).sum()),
                     "raw_mesh_penetrating_samples_after": int((after["distance"][i] < 0.).sum()),
                     "vehicle_radius_violating_samples_before": int((before["distance"][i] < .2).sum()),
                     "vehicle_radius_violating_samples_after": int((after["distance"][i] < .2).sum()),
                     "accepted_steps": int(history["accepted"][:, i].sum()),
                     "stalled_steps": int(history["stalled"][:, i].sum()),
                     "rejected_trials": int(history["reject_count"][:, i].sum()),
                     "domain_rejected_trials": int(history["domain_reject_count"][:, i].sum()),
                     "nonfinite_trials": int(history["nonfinite_trial_count"][:, i].sum()),
                     "all_iterations_finite": bool(history["finite"][:, i].all() & history["finite_gradient"][:, i].all()),
                     "target_displacement_m": float(np.linalg.norm(.5 * history["q"][-1, i, :3])),
                     "duration_ratio": float(after["duration"][i] / before["duration"][i]),
                     "sampled_max_speed_after_m_s": float(np.linalg.norm(after["velocity"][i], axis=1).max()),
                     "sampled_max_acceleration_after_m_s2": float(np.linalg.norm(after["acceleration"][i], axis=1).max())})
    def group_summary(mask):
        return {"count": int(mask.sum()),
                "cost_change": _statistics((history["cost"][-1] - history["cost"][0])[mask]),
                "obstacle_change": _statistics((history["obstacle"][-1] - history["obstacle"][0])[mask]),
                "observed_min_distance_change_m": _statistics(improvement[mask]),
                "observed_vehicle_clearance_after_m": _statistics(minimum_after[mask] - .2),
                "required_clearance_surplus_after_m": _statistics(minimum_after[mask] - .3),
                "clearance_improved_over_1mm": int((mask & (improvement > .001)).sum()),
                "clearance_worsened_over_1mm": int((mask & (improvement < -.001)).sum()),
                "required_clearance_violating_trajectories_before": int((mask & (minimum_before < .3)).sum()),
                "required_clearance_violating_trajectories_after": int((mask & (minimum_after < .3)).sum()),
                "required_clearance_recovered": int((mask & (minimum_before < .3) & (minimum_after >= .3)).sum()),
                "previously_required_clear_became_violating": int((mask & (minimum_before >= .3) & (minimum_after < .3)).sum()),
                "raw_mesh_penetrating_trajectories_before": int((mask & (minimum_before < 0.)).sum()),
                "raw_mesh_penetrating_trajectories_after": int((mask & (minimum_after < 0.)).sum()),
                "vehicle_radius_violating_trajectories_before": int((mask & (minimum_before < .2)).sum()),
                "vehicle_radius_violating_trajectories_after": int((mask & (minimum_after < .2)).sum())}
    return {"per_trajectory": rows, "all": group_summary(np.ones(len(groups), dtype=bool)),
            "groups": {name: group_summary(groups == i) for i, name in enumerate(GROUPS)},
            "monotonic_cost_is_line_search_mechanism": True,
            "stored_cost_increase_count": int((np.diff(history["cost"], axis=0) > 1e-6).sum()),
            "all_iterations_finite": bool(history["finite"].all() & history["finite_gradient"].all()),
            "all_oracle_observations_finite": bool(np.isfinite(before["distance"]).all() & np.isfinite(after["distance"]).all()),
            "accepted_step_counts": _statistics(history["accepted"].sum(axis=0)),
            "total_rejected_trials": int(history["reject_count"].sum()),
            "total_domain_rejected_trials": int(history["domain_reject_count"].sum()),
            "total_nonfinite_trials": int(history["nonfinite_trial_count"].sum()),
            "final_parameters_at_box_limit": int((np.abs(history["q"][-1]) >= 1. - 1e-6).sum()),
            "component_order": ["obstacle", *WEIGHTS],
            "initial_components": history["initial_components"].tolist(),
            "final_components": history["final_components"].tolist()}


def _fd_checks(field, data, q, bounds, indices):
    selected = torch.tensor(indices, device=q.device, dtype=torch.int64)
    data = _to(_subset(data, selected), device=q.device, dtype=torch.float64)
    parameters = q[selected].detach().double().requires_grad_(True)
    fractions = torch.linspace(0., 1., 96, device=q.device, dtype=torch.float64)
    base = _evaluate(field, parameters, data, fractions, bounds)
    finite_base = bool(base["finite"].all())
    if not finite_base:
        return {"passed": False, "finite": False, "indices": indices,
                "failure": "nonfinite base cost or domain hull", "pipeline_gate": False,
                "failure_reasons": ["nonfinite_base_cost_or_domain_hull"],
                "nonfinite_counts": {"base_trajectories": int((~base["finite"]).sum())}}, {}
    obs_gradient = None
    if base["obstacle"].requires_grad:
        obs_gradient, = torch.autograd.grad(base["obstacle"].sum(), parameters,
                                           retain_graph=True, allow_unused=True)
    obs_connected = obs_gradient is not None
    if obs_gradient is None:
        obs_gradient = torch.full_like(parameters, float("nan"))
    total_gradient, = torch.autograd.grad(base["total"].sum(), parameters)
    base_ids = field.leaf_ids(base["positions"].detach().reshape(-1, 3)).reshape(len(indices), 96)
    h = torch.tensor([1e-3, 3e-4, 1e-4], device=q.device, dtype=torch.float64)
    # Layout: trajectory, parameter, step size, minus/plus, parameter channel.
    variants = parameters.detach()[:, None, None, None, :].expand(-1, 10, 3, 2, -1).clone()
    delta = torch.eye(10, device=q.device, dtype=torch.float64)[:, None, None, :] * h[None, :, None, None]
    delta = delta * q.new_tensor([-1., 1.], dtype=torch.float64)[None, None, :, None]
    variants += delta[None]
    repeated_indices = torch.arange(len(indices), device=q.device).repeat_interleave(10 * 3 * 2)
    with torch.no_grad():
        probe = _evaluate(field, variants.reshape(-1, 10), _subset(data, repeated_indices), fractions, bounds)
        ids = field.leaf_ids(probe["positions"].reshape(-1, 3)).reshape(len(indices), 10, 3, 2, 96)
    obs_values = probe["obstacle"].reshape(len(indices), 10, 3, 2)
    total_values = probe["total"].reshape(len(indices), 10, 3, 2)
    fd_obs = (obs_values[..., 1] - obs_values[..., 0]) / (2 * h)
    fd_total = (total_values[..., 1] - total_values[..., 0]) / (2 * h)
    changed_base = (ids != base_ids[:, None, None, None]).sum(dim=-1)
    changed_sides = (ids[:, :, :, 0] != ids[:, :, :, 1]).sum(dim=-1)
    errors_obs = (fd_obs - obs_gradient[:, :, None]).abs()
    errors_total = (fd_total - total_gradient[:, :, None]).abs()
    atol, rtol = 2e-6, 1e-3
    finite_obs = torch.isfinite(fd_obs) & torch.isfinite(obs_gradient[:, :, None])
    finite_total = torch.isfinite(fd_total) & torch.isfinite(total_gradient[:, :, None])
    # Infinities must not pass through the IEEE comparison inf <= inf.
    pass_obs = finite_obs & (errors_obs <= atol + rtol * fd_obs.abs())
    pass_total = finite_total & (errors_total <= atol + rtol * fd_total.abs())
    finite = finite_obs.all() & finite_total.all()
    probe_inside = probe["inside"].reshape(len(indices), 10, 3, 2)
    probe_finite = probe["finite"].reshape(len(indices), 10, 3, 2)
    nonfinite_counts = {
        "probe_trajectories": int((~probe_finite).sum()),
        "in_domain_probe_trajectories": int((probe_inside & ~probe_finite).sum()),
        "obstacle_probe_values": int((~torch.isfinite(obs_values)).sum()),
        "total_probe_values": int((~torch.isfinite(total_values)).sum()),
        "obstacle_fd_values": int((~torch.isfinite(fd_obs)).sum()),
        "total_fd_values": int((~torch.isfinite(fd_total)).sum()),
        "obstacle_autograd_values": int((~torch.isfinite(obs_gradient)).sum()),
        "total_autograd_values": int((~torch.isfinite(total_gradient)).sum()),
    }
    outside_count = int((~probe_inside).sum())
    failure_reasons = []
    if outside_count:
        failure_reasons.append("perturbed_trajectory_hull_outside_domain")
    if nonfinite_counts["in_domain_probe_trajectories"]:
        failure_reasons.append("nonfinite_in_domain_field_cost")
    if not obs_connected:
        failure_reasons.append("obstacle_autograd_disconnected")
    if not bool(finite):
        failure_reasons.append("nonfinite_finite_difference_or_autograd")
    if bool((finite_obs & ~pass_obs).any() | (finite_total & ~pass_total).any()):
        failure_reasons.append("finite_difference_tolerance_exceeded")
    raw = {"q": parameters.detach().cpu().numpy(), "steps": h.cpu().numpy(),
           "obs_autograd": obs_gradient.cpu().numpy(), "total_autograd": total_gradient.cpu().numpy(),
           "obs_minus_plus": obs_values.cpu().numpy(), "total_minus_plus": total_values.cpu().numpy(),
           "obs_fd": fd_obs.cpu().numpy(), "total_fd": fd_total.cpu().numpy(),
           "obs_absolute_error": errors_obs.cpu().numpy(), "total_absolute_error": errors_total.cpu().numpy(),
           "leaf_changes_from_base": changed_base.cpu().numpy(), "leaf_changes_between_sides": changed_sides.cpu().numpy(),
           "probe_inside": probe_inside.cpu().numpy(), "probe_finite": probe_finite.cpu().numpy(),
           "finite": (finite_obs.all(dim=-1) & finite_total.all(dim=-1)).cpu().numpy(),
           "crosses_parameter_box": (variants.abs() > 1.).any(dim=(2, 3, 4)).cpu().numpy(),
           "passed": (pass_obs.all(dim=-1) & pass_total.all(dim=-1)).cpu().numpy()}
    rows = []
    for i, trajectory_index in enumerate(indices):
        for j, name in enumerate(PARAMETER_NAMES):
            rows.append({"trajectory": trajectory_index, "parameter": name,
                         "normalized_parameter": float(raw["q"][i, j]),
                         "obs_autograd": float(raw["obs_autograd"][i, j]), "total_autograd": float(raw["total_autograd"][i, j]),
                         "obs_fd": raw["obs_fd"][i, j].tolist(), "total_fd": raw["total_fd"][i, j].tolist(),
                         "obs_absolute_error": raw["obs_absolute_error"][i, j].tolist(),
                         "total_absolute_error": raw["total_absolute_error"][i, j].tolist(),
                         "leaf_changes_from_base_minus_plus": raw["leaf_changes_from_base"][i, j].tolist(),
                         "leaf_changes_between_sides": raw["leaf_changes_between_sides"][i, j].tolist(),
                         "obs_error_increased_on_last_step": bool(raw["obs_absolute_error"][i, j, -1] > raw["obs_absolute_error"][i, j, -2]),
                         "crosses_parameter_box": bool(raw["crosses_parameter_box"][i, j]),
                         "finite": bool(raw["finite"][i, j]),
                         "outside_domain_probe_count": int((~raw["probe_inside"][i, j]).sum()),
                         "nonfinite_probe_count": int((~raw["probe_finite"][i, j]).sum()),
                         "passed": bool(raw["passed"][i, j])})
    result = {"passed": bool(finite & pass_obs.all() & pass_total.all()), "pipeline_gate": False,
              "indices": indices, "normalized_steps": h.cpu().tolist(), "atol": atol, "rtol": rtol,
              "rows": rows, "finite": bool(finite), "obstacle_autograd_connected": obs_connected,
              "outside_domain_probe_count": outside_count, "nonfinite_counts": nonfinite_counts,
              "failure_reasons": failure_reasons,
              "obs_maximum_absolute_error": float(errors_obs.max()),
              "total_maximum_absolute_error": float(errors_total.max()),
              "note": "centre differences of unclipped normalized parameters; crossing a leaf or parameter limit is recorded, not filtered; nonfinite JSON values become null, while NPZ retains the original inf/nan"}
    return result, raw


def _json_safe(value):
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def run_batch_optimization(field, oracle, output_dir, *, seed=23, count=64,
                           steps=50, device="cuda") -> dict:
    """Optimize real fitted trajectories and save all requested observations.

    `oracle` accepts same-device Nx3 Torch points and returns signed distances.
    It may internally chunk obstacles; this caller chunks at most 4096 points.
    `field.logical_bounds`, when present, takes precedence over padded bounds.
    Reported parameter limits are experiment choices, not aircraft limits.
    `pipeline_passed` measures completion/finiteness; FD and observed collision
    outcomes are separate and are never promoted to a continuous safety claim.
    """
    if not 4 <= count <= 2048 or steps < 1:
        raise ValueError("count must be 4..2048 and steps must be positive")
    device = torch.device(device)
    bounds = tuple(torch.as_tensor(value, device=device, dtype=torch.float64)
                   for value in getattr(field, "logical_bounds", field.bounds))
    if (any(bound.shape != (3,) for bound in bounds)
            or not bool(torch.isfinite(torch.stack(bounds)).all())
            or not bool((bounds[1] > bounds[0]).all())):
        raise ValueError("logical bounds must be finite ordered xyz vectors")
    def stamp():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()
    elapsed = {}
    start = stamp()
    arrays, labels, selection = _sample_tasks(oracle, bounds, device, count, seed)
    data = _to(arrays, device=device, dtype=torch.float32)
    elapsed["task_sampling_with_oracle"] = stamp() - start
    start = stamp()
    optimized, history = _optimize(field, data, bounds, steps, [1.] * 10)
    elapsed["main_optimization"] = stamp() - start
    control_indices = np.linspace(0, count - 1, min(16, count), dtype=np.int64)
    control_ids = torch.tensor(control_indices, device=device, dtype=torch.int64)
    control_data = _subset(data, control_ids)
    start = stamp()
    control_q, control_history = _optimize(field, control_data, bounds, steps, [0.] * 3 + [1.] * 6 + [0.])
    elapsed["fixed_goal_and_duration_control"] = stamp() - start
    start = stamp()
    before = _dense_observation(oracle, torch.zeros_like(optimized), data)
    after = _dense_observation(oracle, optimized, data)
    control_before = {key: value if key == "fractions" else value[control_indices] for key, value in before.items()}
    control_after = _dense_observation(oracle, control_q, control_data)
    elapsed["independent_dense_oracle_validation"] = stamp() - start
    main_result = _outcomes(history, before, after, labels)
    control_result = _outcomes(control_history, control_before, control_after, labels[control_indices])
    fd_indices = [int(np.flatnonzero(labels == group)[0]) for group in range(3) if (labels == group).any()]
    for index in np.argsort(after["distance"].min(axis=1)):
        if int(index) not in fd_indices:
            fd_indices.append(int(index))
        if len(fd_indices) == 4:
            break
    start = stamp()
    fd_before, fd_before_raw = _fd_checks(field, data, torch.zeros_like(optimized), bounds, fd_indices)
    fd_after, fd_after_raw = _fd_checks(field, data, optimized, bounds, fd_indices)
    elapsed["finite_difference_before_and_after"] = stamp() - start
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, json_path = output_dir / "batch-optimization.npz", output_dir / "batch-optimization.json"
    stored_inputs = {key: value.detach().cpu().numpy() for key, value in data.items()}
    raw = {"group": labels, "group_names": np.array(GROUPS), "control_indices": control_indices,
           "fd_indices": np.array(fd_indices), "bounds": torch.stack(bounds).cpu().numpy(),
           "train_fractions": torch.linspace(0., 1., 96, dtype=torch.float32).numpy(),
           "parameter_names": np.array(PARAMETER_NAMES)}
    for prefix, collection in (("input", stored_inputs), ("main", history), ("control", control_history),
                               ("before", before), ("after", after), ("control_before", control_before),
                               ("control_after", control_after), ("fd_before", fd_before_raw), ("fd_after", fd_after_raw)):
        raw.update({f"{prefix}_{key}": value for key, value in collection.items()})
    np.savez_compressed(raw_path, **raw)
    pipeline_passed = (main_result["all_iterations_finite"] and control_result["all_iterations_finite"]
                       and main_result["all_oracle_observations_finite"] and control_result["all_oracle_observations_finite"])
    result = {"pipeline_passed": bool(pipeline_passed), "completed": True,
              "device": str(device), "optimization_dtype": "torch.float32", "fd_dtype": "torch.float64",
              "seed": seed, "count": count, "steps": steps, "selection": selection,
              "configuration": {"train_samples": 96, "oracle_validation_samples": 513,
                                "validation_nodes": "511 midpoint interiors plus endpoints; only endpoints overlap training grid",
                                "logical_bounds": torch.stack(bounds).cpu().tolist(),
                                "length_m": [4., 8.], "nominal_speed_m_s": 2.,
                                "endpoint_minimum_distance_m": .65, "endpoint_domain_margin_m": .8,
                                "vehicle_radius_m": .2, "safety_margin_m": .1, "required_distance_m": .3,
                                "selection_group_distance_thresholds_m": [.3, .8],
                                "parameter_scales": {"position_m": .5, "velocity_m_s": 2., "acceleration_m_s2": 2., "duration_fraction": .2},
                                "normalized_parameter_bounds": [-1., 1.], "jerk_normalization_m_s3": 2.,
                                "loss_weights": WEIGHTS, "obstacle_cost": "mean(softplus((0.3 - d)/0.1)^2)",
                                "initial_step_size": .05, "armijo_coefficient": 1e-4, "maximum_backtracks": 12,
                                "gradient_reduction": "sum of independent per-trajectory costs, not batch mean",
                                "control_free_parameters": "terminal velocity and acceleration only",
                                "full_path_speed_acceleration_constraints": False},
              "main": main_result, "fixed_goal_duration_control": {"indices": control_indices.tolist(), **control_result},
              "fd_before": fd_before, "fd_after": fd_after,
              "wall_seconds": elapsed,
              "timing_scope": "synchronized whole phases, including Python validation and oracle work; not isolated query microbenchmarks",
              "limitations": ["monotonic total cost is enforced by line search, not evidence of global convergence",
                              "oracle results are discrete 513-time observations, not continuous clearance or collision guarantees",
                              "finite differences do not gate pipeline completion; seam failures and step-size dependence remain in results",
                              "parameter limits and loss weights are fixed experiment choices, not a flight controller or validated vehicle dynamics"],
              "artifacts": {"raw": str(raw_path), "summary": str(json_path)}}
    result = _json_safe(result)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
