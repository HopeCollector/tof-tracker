"""Verify differentiable 3D polynomial trajectories using independent polynomials.

Run inside the project Docker container. The oracles evaluate known physical-time
coefficients or elementary constant-acceleration motion, without an old solver.
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch

from devtools.trajectory import PolynomialTrajectory, TrajectoryState, fit_trajectory


FIELDS = ("position", "velocity", "acceleration", "yaw", "yaw_rate")


def close(actual, expected, *, atol=2e-10, rtol=2e-10):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert actual.device == expected.device, (actual.device, expected.device)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def rejected(action, error=(ValueError, TypeError)):
    try:
        action()
    except error:
        return
    raise AssertionError("invalid input was accepted")


def tensor(value, device, dtype=torch.float64):
    return torch.tensor(value, dtype=dtype, device=device)


def polynomial(coefficients, seconds, derivative=0, *, vector=True):
    """Direct monomial sum in seconds; coefficients are supplied independently."""
    power_axis = -2 if vector else -1
    terms = []
    for power in range(derivative, coefficients.shape[power_axis]):
        factor = math.factorial(power) / math.factorial(power - derivative)
        coefficient = coefficients.select(power_axis, power)
        coefficient = coefficient.unsqueeze(-2 if vector else -1)
        argument = seconds.unsqueeze(-1) if vector else seconds
        terms.append(factor * coefficient * argument ** (power - derivative))
    return sum(terms)


def known_state(position_coefficients, yaw_coefficients, seconds):
    return TrajectoryState(
        position=polynomial(position_coefficients, seconds),
        velocity=polynomial(position_coefficients, seconds, 1),
        acceleration=polynomial(position_coefficients, seconds, 2),
        yaw=polynomial(yaw_coefficients, seconds, vector=False),
        yaw_rate=polynomial(yaw_coefficients, seconds, 1, vector=False),
    )


def boundary(position_coefficients, yaw_coefficients, seconds):
    state = known_state(position_coefficients, yaw_coefficients, seconds.unsqueeze(-1))
    return TrajectoryState(**{name: getattr(state, name).squeeze(-2 if name in FIELDS[:3] else -1)
                              for name in FIELDS})


def compare_states(actual, expected, **tolerances):
    for name in FIELDS:
        close(getattr(actual, name), getattr(expected, name), **tolerances)


def stationary(device, dtype=torch.float64, batch=()):
    position = torch.zeros((*batch, 3), dtype=dtype, device=device)
    scalar = torch.zeros(batch, dtype=dtype, device=device)
    return TrajectoryState(position, position.clone(), position.clone(), scalar, scalar.clone())


def analytic_polynomials(device):
    generator = torch.Generator(device=device).manual_seed(20261015)
    coefficients = torch.randn((2, 3, 6, 3), generator=generator, device=device,
                               dtype=torch.float64)
    coefficients = coefficients * tensor([1, .5, .25, .08, .03, .01], device)[None, None, :, None]
    yaw_coefficients = torch.randn((2, 3, 4), generator=generator, device=device,
                                   dtype=torch.float64) * .2
    duration = tensor([[.7, 1.2, 2.3], [1.1, 1.8, 2.8]], device)
    initial = boundary(coefficients, yaw_coefficients, torch.zeros_like(duration))
    terminal = boundary(coefficients, yaw_coefficients, duration)
    trajectory = fit_trajectory(initial, terminal, duration)
    assert trajectory.position_coefficients.shape == (2, 3, 6, 3)
    assert trajectory.yaw_coefficients.shape == (2, 3, 4)
    close(trajectory.duration, duration)
    powers = torch.arange(6, dtype=duration.dtype, device=device)
    close(trajectory.position_coefficients,
          coefficients * duration[..., None, None] ** powers[:, None])
    close(trajectory.yaw_coefficients,
          yaw_coefficients * duration[..., None] ** powers[:4])
    seconds = duration[..., None] * torch.linspace(0, 1, 17, dtype=duration.dtype, device=device)
    compare_states(trajectory.sample(seconds), known_state(coefficients, yaw_coefficients, seconds))
    # Endpoint p/v/a and yaw/rate are checked against all ten supplied fields.
    for index, expected in ((0, initial), (-1, terminal)):
        sampled = trajectory.sample(seconds)
        for name in FIELDS:
            dimension = -2 if name in FIELDS[:3] else -1
            close(getattr(sampled, name).select(dimension, index), getattr(expected, name))
    return {"checks": ["seeded full-3D quintic and cubic physical-time coefficient recovery",
                        "17 per-trajectory times and all ten endpoint conditions"],
            "batch_shape": [2, 3], "samples": 17}


def constant_acceleration(device):
    position = tensor([1.2, -.7, 2.3], device)
    velocity = tensor([.8, -1.1, .6], device)
    acceleration = tensor([.4, .3, -.5], device)
    yaw, yaw_rate, yaw_acceleration = (tensor(value, device) for value in (.7, -.2, .35))
    duration = tensor(2.3, device)
    initial = TrajectoryState(position, velocity, acceleration, yaw, yaw_rate)
    terminal = TrajectoryState(position + velocity * duration + acceleration * duration**2 / 2,
                               velocity + acceleration * duration, acceleration,
                               yaw + yaw_rate * duration + yaw_acceleration * duration**2 / 2,
                               yaw_rate + yaw_acceleration * duration)
    trajectory = fit_trajectory(initial, terminal, duration)
    seconds = tensor([0, .17, .7, 1.1, 2.3], device)
    expected = TrajectoryState(position + seconds[:, None] * velocity
                               + seconds[:, None]**2 * acceleration / 2,
                               velocity + seconds[:, None] * acceleration,
                               acceleration.expand(5, 3),
                               yaw + seconds * yaw_rate + seconds**2 * yaw_acceleration / 2,
                               yaw_rate + seconds * yaw_acceleration)
    compare_states(trajectory.sample(seconds), expected)
    close(trajectory.position_coefficients[3:], torch.zeros((3, 3), device=device, dtype=duration.dtype))
    close(trajectory.yaw_coefficients[3:], torch.zeros((1,), device=device, dtype=duration.dtype))
    return {"checks": ["nonzero initial position, velocity and acceleration on all three axes",
                        "constant linear/yaw acceleration with physical seconds and zero excess coefficients"]}


def yaw_branches(device):
    initial = replace(stationary(device), yaw=tensor(3.05, device), yaw_rate=tensor(.2, device))
    terminal = replace(stationary(device), yaw=tensor(-3.04, device), yaw_rate=tensor(-.1, device))
    duration = tensor(1.7, device)
    seconds = torch.linspace(0, 1.7, 13, dtype=duration.dtype, device=device)
    u = seconds / duration

    def hermite(end_yaw):
        # Independent cubic Hermite basis, expressed using endpoint values/slopes.
        return ((2*u**3 - 3*u**2 + 1) * initial.yaw
                + (u**3 - 2*u**2 + u) * duration * initial.yaw_rate
                + (-2*u**3 + 3*u**2) * end_yaw
                + (u**3 - u**2) * duration * terminal.yaw_rate)

    raw = fit_trajectory(initial, terminal, duration)
    explicit_raw = fit_trajectory(initial, terminal, duration, yaw_mode="unwrapped")
    short = fit_trajectory(initial, terminal, duration, yaw_mode="shortest")
    close(raw.sample(seconds).yaw, hermite(terminal.yaw))
    compare_states(raw.sample(seconds), explicit_raw.sample(seconds))
    continuous_end = terminal.yaw + 2 * math.pi
    close(short.sample(seconds).yaw, hermite(continuous_end))
    assert short.sample(duration).yaw.item() > math.pi, "shortest yaw was wrapped after interpolation"
    close(short.sample(duration).yaw_rate, terminal.yaw_rate.reshape(1))
    multiple_turns = replace(terminal, yaw=initial.yaw + 4*math.pi + .6)
    close(fit_trajectory(initial, multiple_turns, duration).sample(duration).yaw,
          multiple_turns.yaw.reshape(1))
    close(fit_trajectory(initial, multiple_turns, duration, yaw_mode="shortest").sample(duration).yaw,
          (initial.yaw + .6).reshape(1))
    zero = stationary(device)
    half_turn = replace(zero, yaw=tensor(math.pi, device))
    tie = fit_trajectory(zero, half_turn, duration, yaw_mode="shortest")
    close(tie.sample(duration).yaw, tensor([-math.pi], device))
    return {"checks": ["default unwrapped preserves raw endpoints and multiple turns",
                        "shortest crosses pi continuously and retains terminal yaw rate",
                        "exact positive half-turn selects -pi; branch derivative is intentionally not checked"]}


def broadcasting(device):
    initial = TrajectoryState(
        tensor([[[1, 2, 3]], [[-2, .4, 1]]], device),
        tensor([[[.2, -.3, .4], [.1, .7, -.2], [.8, -.5, .3]]], device),
        tensor([.1, -.2, .3], device), tensor([[.2], [-.4]], device), tensor(.15, device))
    terminal = TrajectoryState(
        tensor([[[2, -1, 4], [3, .2, 1], [.4, 2, -3]]], device),
        tensor([[[.7, .2, -.1]], [[-.2, .1, .5]]], device),
        tensor([-.3, .1, .2], device), tensor([[.6, -.2, 1.1]], device),
        tensor([[-.1], [.3]], device))
    duration = tensor([[1.1, 1.7, 2.1], [1.4, 1.2, 2.4]], device)
    trajectory = fit_trajectory(initial, terminal, duration)
    assert trajectory.duration.shape == (2, 3)
    expanded = [{name: torch.broadcast_to(getattr(state, name), (2, 3, 3) if name in FIELDS[:3]
                                           else (2, 3)) for name in FIELDS}
                for state in (initial, terminal)]
    per_batch_seconds = duration[..., None] * tensor([0, .2, .8, 1], device)
    per_batch = trajectory.sample(per_batch_seconds)
    scalar = trajectory.sample(tensor(.3, device))
    shared_seconds = tensor([0, .2, .6, 1], device)
    shared = trajectory.sample(shared_seconds)
    # Four additional time batches cross with every trajectory, not with S.
    extra_seconds = tensor([[0, .1, .3, .7], [.05, .2, .4, .8],
                            [.1, .3, .5, .9], [.15, .4, .6, 1]], device)[:, None, None, :]
    extra = trajectory.sample(extra_seconds)
    assert extra.position.shape == (4, 2, 3, 4, 3)
    assert extra.yaw.shape == (4, 2, 3, 4)
    assert scalar.position.shape == (2, 3, 1, 3)
    assert scalar.yaw.shape == (2, 3, 1)
    for row in range(2):
        for column in range(3):
            states = [TrajectoryState(**{name: values[name][row, column] for name in FIELDS})
                      for values in expanded]
            one = fit_trajectory(*states, duration[row, column])
            for actual, seconds in ((per_batch, per_batch_seconds[row, column]),
                                    (scalar, tensor(.3, device)), (shared, shared_seconds)):
                expected = one.sample(seconds)
                compare_states(TrajectoryState(**{name: getattr(actual, name)[row, column]
                                                   for name in FIELDS}), expected)
            expected_extra = one.sample(extra_seconds[:, 0, 0, :])
            compare_states(TrajectoryState(**{name: getattr(extra, name)[:, row, column]
                                               for name in FIELDS}), expected_extra)

    # Strided inputs remain legal; contiguity is not part of the input contract.
    positions = torch.arange(6, device=device, dtype=torch.float64).reshape(3, 2).t()
    yaws = torch.arange(4, device=device, dtype=torch.float64)[::2]
    state = replace(stationary(device, batch=(2,)), position=positions, yaw=yaws)
    durations = tensor([1.1, 99, 1.4, 99], device)[::2]
    noncontiguous_times = torch.arange(10, device=device, dtype=torch.float64)[::2] / 10
    assert not positions.is_contiguous() and not yaws.is_contiguous()
    strided = fit_trajectory(state, state, durations).sample(noncontiguous_times)
    close(strided.position, positions[:, None, :].expand(2, 5, 3))
    close(strided.yaw, yaws[:, None].expand(2, 5))
    return {"checks": ["all endpoint fields and T broadcast jointly to (2,3)",
                        "scalar, shared and per-batch seconds; extra time batches (4,1,1,S)",
                        "noncontiguous strided states, durations and sample times remain accepted"]}


def time_derivatives(device):
    coefficients = tensor([[[.4, -.8, 1], [.6, .3, -.2], [.2, -.1, .3],
                            [.04, .02, -.03], [-.02, .01, .015], [.003, -.002, .004]],
                           [[-.7, .2, .6], [.1, -.4, .7], [-.1, .3, -.2],
                            [.02, -.03, .01], [.01, .02, -.01], [-.002, .001, .003]]], device)
    yaw_coefficients = tensor([[.3, -.2, .1, -.02], [-.5, .4, -.1, .03]], device)
    duration = tensor([1.6, 2.1], device)
    initial = boundary(coefficients, yaw_coefficients, torch.zeros_like(duration))
    terminal = boundary(coefficients, yaw_coefficients, duration)
    seconds = tensor([[.12, .43, .91, 1.3], [.2, .6, 1.2, 1.8]], device).requires_grad_()
    sampled = fit_trajectory(initial, terminal, duration).sample(seconds)
    for axis in range(3):
        velocity = torch.autograd.grad(sampled.position[..., axis].sum(), seconds,
                                       create_graph=True, retain_graph=True)[0]
        acceleration = torch.autograd.grad(velocity.sum(), seconds, retain_graph=True)[0]
        close(velocity, sampled.velocity[..., axis])
        close(acceleration, sampled.acceleration[..., axis])
    rate = torch.autograd.grad(sampled.yaw.sum(), seconds)[0]
    close(rate, sampled.yaw_rate)
    return {"checks": ["autograd d(position)/dt equals velocity in seconds on every axis",
                        "autograd d2(position)/dt2 equals acceleration; d(yaw)/dt equals yaw_rate"]}


def all_input_gradients(device):
    initial = (tensor([.3, -.4, 1.1], device), tensor([.4, -.2, .3], device),
               tensor([.1, .3, -.2], device), tensor(.4, device), tensor(.25, device))
    terminal = (tensor([1.7, .6, -.2], device), tensor([-.3, .5, .2], device),
                tensor([-.2, .1, .4], device), tensor(-.8, device), tensor(-.15, device))
    inputs = tuple(value.requires_grad_() for value in (*initial, *terminal,
                                                       tensor(1.9, device), tensor([.31, 1.27], device)))

    def evaluate(*values, yaw_mode="unwrapped"):
        trajectory = fit_trajectory(TrajectoryState(*values[:5]), TrajectoryState(*values[5:10]),
                                    values[10], yaw_mode=yaw_mode)
        sampled = trajectory.sample(values[11])
        return tuple(getattr(sampled, name) for name in FIELDS)

    for mode in ("unwrapped", "shortest"):
        assert torch.autograd.gradcheck(lambda *values: evaluate(*values, yaw_mode=mode), inputs,
                                        eps=1e-6, atol=3e-6, rtol=3e-4)
    # A second-order check also catches graph-breaking custom derivative paths.
    assert torch.autograd.gradgradcheck(evaluate, inputs, eps=1e-6, atol=5e-6, rtol=5e-4)
    gradients = torch.autograd.grad(sum((index + 1) * value.square().sum()
                                       for index, value in enumerate(evaluate(*inputs))), inputs)
    for gradient, source in zip(gradients, inputs):
        assert gradient.shape == source.shape and bool(torch.isfinite(gradient).all())
        assert bool((gradient != 0).any()), "an input lost its gradient connection"
    # Distinct prefixes expand to (2,2); numerical Jacobians check that backward
    # sums over expanded axes, including the shared time grid and duration.
    broadcast_initial = (*initial[:3], tensor(3.05, device), initial[4])
    broadcast_terminal = (tensor([[[1.7, .6, -.2]], [[-.3, 1.2, .7]]], device),
                          *terminal[1:3], tensor([[-3.04, -2.9]], device), terminal[4])
    broadcast_inputs = tuple(value.detach().clone().requires_grad_() for value in (
        *broadcast_initial, *broadcast_terminal, tensor([[1.9], [2.2]], device),
        tensor([[[.31, 1.27], [.47, 1.51]]], device)))
    broadcast_outputs = evaluate(*broadcast_inputs, yaw_mode="shortest")
    assert broadcast_outputs[0].shape == (2, 2, 2, 3)
    assert bool((broadcast_outputs[3] > math.pi).any()), "test must exercise a continuous pi crossing"
    assert torch.autograd.gradcheck(lambda *values: evaluate(*values, yaw_mode="shortest"),
                                    broadcast_inputs, eps=1e-6, atol=3e-6, rtol=3e-4)
    return {"checks": ["float64 numerical gradcheck of p/v/a/yaw/rate at both ends, T and time",
                        "unwrapped and shortest away from branch; second-order gradgradcheck",
                        "every input has finite nonzero gradients",
                        "joint (2,2) broadcasting reduces gradients across expanded axes; shortest crosses pi"],
            "input_tensors": 12, "yaw_branch_limit": "no derivative claim at +/-pi"}


def float32_and_small_duration(device):
    initial = stationary(device, torch.float32, batch=(1,))
    initial = replace(initial, position=tensor([[.2, -.3, 1.1]], device, torch.float32),
                      velocity=tensor([[.4, -.2, .3]], device, torch.float32),
                      acceleration=tensor([[.1, .2, -.1]], device, torch.float32),
                      yaw=tensor([.2], device, torch.float32), yaw_rate=tensor([.3], device, torch.float32))
    duration = tensor([1.3], device, torch.float32)
    terminal = TrajectoryState(initial.position + initial.velocity * duration[:, None]
                               + initial.acceleration * duration[:, None]**2 / 2,
                               initial.velocity + initial.acceleration * duration[:, None],
                               initial.acceleration, initial.yaw + initial.yaw_rate * duration,
                               initial.yaw_rate)
    trajectory = fit_trajectory(initial, terminal, duration)
    seconds = tensor(.6, device, torch.float32)
    sampled = trajectory.sample(seconds)
    assert sampled.position.shape == (1, 1, 3) and sampled.yaw.shape == (1, 1)
    expected = TrajectoryState((initial.position + initial.velocity * seconds
                                + initial.acceleration * seconds**2 / 2)[:, None, :],
                               (initial.velocity + initial.acceleration * seconds)[:, None, :],
                               initial.acceleration[:, None, :],
                               (initial.yaw + initial.yaw_rate * seconds)[:, None],
                               initial.yaw_rate[:, None])
    compare_states(sampled, expected, atol=2e-5, rtol=2e-5)
    for name in FIELDS:
        assert getattr(sampled, name).dtype == torch.float32
    scalar = fit_trajectory(stationary(device), stationary(device), tensor(1e-12, device))
    small = scalar.sample(tensor([0, 5e-13, 1e-12], device))
    close(small.position, torch.zeros((3, 3), device=device, dtype=torch.float64))
    assert scalar.duration.shape == ()
    assert scalar.sample(tensor(5e-13, device)).position.shape == (1, 3)
    empty_times = torch.empty((0,), device=device, dtype=torch.float32)
    empty_samples = trajectory.sample(empty_times)
    empty_state = stationary(device, torch.float32, batch=(0,))
    empty_trajectory = fit_trajectory(empty_state, empty_state, empty_times)
    assert empty_trajectory.position_coefficients.shape == (0, 6, 3)
    assert empty_trajectory.yaw_coefficients.shape == (0, 4)
    assert empty_trajectory.duration.shape == (0,)
    empty_batch = empty_trajectory.sample(tensor([0, .3], device, torch.float32))
    both_empty = empty_trajectory.sample(empty_times)
    for name in FIELDS:
        suffix = (3,) if name in FIELDS[:3] else ()
        assert getattr(empty_samples, name).shape == (1, 0, *suffix)
        assert getattr(empty_batch, name).shape == (0, 2, *suffix)
        assert getattr(both_empty, name).shape == (0, 0, *suffix)
    return {"checks": ["float32 dtype and singleton batch/sample axes retained",
                        "unbatched scalar sample retains S=1; positive 1e-12 duration accepted",
                        "empty sample axis, empty trajectory batch and their combination preserve all field shapes"]}


def validation(device):
    state = stationary(device)
    duration = tensor(1, device)
    count = 0

    def reject(action, error=(ValueError, TypeError)):
        nonlocal count
        rejected(action, error)
        count += 1

    def bad_field(name, value):
        reject(lambda: fit_trajectory(replace(state, **{name: value}), state, duration))
        reject(lambda: fit_trajectory(state, replace(state, **{name: value}), duration))

    for name in FIELDS:
        original = getattr(state, name)
        for bad in (None, 0.0, original.tolist()):
            bad_field(name, bad)
        for dtype in (torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64):
            bad_field(name, original.to(dtype=dtype))
        bad_field(name, original.to(dtype=torch.float32))
        for bad in (math.nan, math.inf, -math.inf):
            bad_field(name, torch.full_like(original, bad))
        bad_field(name, original.to_sparse())
    for name in FIELDS[:3]:
        for bad in (tensor(0, device), torch.zeros(2, device=device, dtype=torch.float64),
                    torch.zeros((3, 1), device=device, dtype=torch.float64)):
            bad_field(name, bad)
    reject(lambda: fit_trajectory(None, state, duration))
    reject(lambda: fit_trajectory(state, {}, duration))
    for bad in (None, 1, 1.0, [1.0], tensor(0, device), tensor(-1, device),
                tensor(math.nan, device), tensor(math.inf, device), tensor(-math.inf, device),
                tensor(1, device, torch.float32), tensor(1, device, torch.int64),
                tensor(1, device, torch.float16), tensor(1, device).to_sparse()):
        reject(lambda bad=bad: fit_trajectory(state, state, bad))
    batch_state = stationary(device, batch=(2,))
    reject(lambda: fit_trajectory(batch_state, stationary(device, batch=(3,)), duration))
    reject(lambda: fit_trajectory(batch_state, batch_state, tensor([1, 2, 3], device)))
    reject(lambda: fit_trajectory(replace(batch_state, yaw=tensor([0, 0, 0], device)),
                                  batch_state, duration))
    for mode in ("", "wrapped", None, 1, ["shortest"]):
        reject(lambda mode=mode: fit_trajectory(state, state, duration, yaw_mode=mode))

    trajectory = fit_trajectory(state, state, duration)
    for bad in (None, 0.5, [0.5], tensor([-.001], device), tensor([1.001], device),
                tensor([0, math.nan], device), tensor([0, math.inf], device),
                tensor([0], device, torch.float32), tensor([0], device, torch.int64),
                tensor([0], device, torch.float16), tensor([0], device).to_sparse()):
        reject(lambda bad=bad: trajectory.sample(bad))
    batched = fit_trajectory(batch_state, batch_state, tensor([1, 2], device))
    reject(lambda: batched.sample(tensor([[.5, 1.01], [.5, 1.5]], device)))
    reject(lambda: batched.sample(torch.zeros((3, 4), dtype=torch.float64, device=device)))
    for bad in (tensor([-1e-15], device), tensor([1 + 1e-15], device)):
        reject(lambda bad=bad: trajectory.sample(bad))

    coefficients, yaw_coefficients = trajectory.position_coefficients, trajectory.yaw_coefficients

    def construct(position=coefficients, yaw=yaw_coefficients, seconds=duration):
        return PolynomialTrajectory(position_coefficients=position, yaw_coefficients=yaw,
                                    duration=seconds)

    compare_states(construct().sample(tensor([0, .5, 1], device)),
                   trajectory.sample(tensor([0, .5, 1], device)))
    for bad in (None, coefficients.tolist(), tensor(0, device),
                torch.zeros((5, 3), dtype=torch.float64, device=device),
                torch.zeros((6, 2), dtype=torch.float64, device=device),
                coefficients.float(), coefficients.to_sparse(), coefficients + math.nan):
        reject(lambda bad=bad: construct(position=bad))
    for bad in (None, yaw_coefficients.tolist(), tensor(0, device),
                torch.zeros(3, dtype=torch.float64, device=device),
                yaw_coefficients.float(), yaw_coefficients.to_sparse(), yaw_coefficients + math.inf):
        reject(lambda bad=bad: construct(yaw=bad))
    for bad in (None, 1, tensor(0, device), tensor(-1, device), tensor(math.nan, device),
                tensor(math.inf, device), duration.float(), duration.to_sparse()):
        reject(lambda bad=bad: construct(seconds=bad))
    reject(lambda: construct(position=coefficients.expand(2, 6, 3)))
    reject(lambda: construct(yaw=yaw_coefficients.expand(2, 4)))
    reject(lambda: construct(seconds=duration.expand(2)))
    reject(lambda: PolynomialTrajectory(coefficients.expand(2, 6, 3),
                                        yaw_coefficients.expand(1, 4), duration.expand(2)))
    # Inputs are finite: reject arithmetic overflow rather than returning a
    # trajectory/output containing inf. This is separate from nonfinite input.
    for dtype, magnitude in ((torch.float32, 1e38), (torch.float64, 1e308)):
        finite_initial = stationary(device, dtype)
        finite_terminal = replace(finite_initial,
                                  position=torch.full((3,), magnitude, device=device, dtype=dtype))
        reject(lambda initial=finite_initial, terminal=finite_terminal, dtype=dtype:
               fit_trajectory(initial, terminal, tensor(1, device, dtype)), ValueError)
    finite_coefficients = coefficients.clone()
    finite_coefficients[2, 0] = 1e308
    overflow_at_sample = construct(position=finite_coefficients)
    reject(lambda: overflow_at_sample.sample(tensor(.5, device)), ValueError)
    # Device mismatch is only constructible when CUDA is available.
    device_cases = 0
    if torch.cuda.is_available():
        other = torch.device("cpu" if device.type == "cuda" else "cuda")
        for name in FIELDS:
            bad_field(name, getattr(state, name).to(other))
        reject(lambda: fit_trajectory(state, state, duration.to(other)))
        reject(lambda: trajectory.sample(tensor([.5], other)))
        reject(lambda: construct(position=coefficients.to(other)))
        reject(lambda: construct(yaw=yaw_coefficients.to(other)))
        reject(lambda: construct(seconds=duration.to(other)))
        device_cases = 15
    return {"checks": ["Tensor-only float32/float64 strided finite inputs and common dtype/device",
                        "state vector dimensions, compatible fit/time batches and exact coefficient batches",
                        "strict positive duration; times outside [0,T] rejected without clipping",
                        "invalid yaw mode and direct PolynomialTrajectory construction",
                        "finite inputs causing coefficient/output overflow explicitly raise ValueError"],
            "rejected_cases": count, "device_mismatch_cases": device_cases}


def batch_forward_backward(device):
    size, samples = 4096, 32
    index = torch.arange(size, device=device, dtype=torch.float32)
    positions = torch.stack((index * .001, index * -.002, 1 + index * .0003), dim=-1).requires_grad_()
    zeros = torch.zeros_like(positions)
    scalars = torch.zeros(size, device=device, dtype=torch.float32)
    initial = TrajectoryState(positions, zeros, zeros, scalars, scalars)
    endpoint = (positions.detach() + tensor([1, -.5, .3], device, torch.float32)).requires_grad_()
    end_yaw = torch.full_like(scalars, .4).requires_grad_()
    terminal = TrajectoryState(endpoint, zeros, zeros, end_yaw, scalars)
    duration = (1.1 + index * .0001).requires_grad_()
    fractions = torch.linspace(0, 1, samples, device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    sampled = fit_trajectory(initial, terminal, duration).sample(duration[:, None] * fractions)
    loss = sampled.position.square().mean() + .1*sampled.velocity.square().mean() + sampled.yaw.square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    assert sampled.position.shape == (size, samples, 3)
    assert bool(torch.isfinite(loss))
    close(sampled.position[:, 0], positions.detach(), atol=2e-5, rtol=2e-5)
    close(sampled.position[:, -1], endpoint.detach(), atol=2e-5, rtol=2e-5)
    for value in (positions, endpoint, end_yaw, duration):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all())
        assert bool((value.grad != 0).any())
    return {"checks": ["4096 float32 trajectories x 32 samples in one forward/backward graph",
                        "endpoint recovery and finite nonzero initial/terminal/yaw/duration gradients"],
            "batch_size": size, "samples": samples, "elapsed_seconds": elapsed,
            "timing_scope": "single synchronized correctness run, not a benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    verifiers = [analytic_polynomials, constant_acceleration, yaw_branches, broadcasting,
                 time_derivatives, float32_and_small_duration, validation, batch_forward_backward]
    if device.type == "cpu":
        verifiers.insert(5, all_input_gradients)
    groups = {}
    for verify in verifiers:
        groups[verify.__name__] = {"status": "passed", **verify(device)}
    result = {"status": "passed", "device": str(device), "torch_version": torch.__version__,
              "scope": "development trajectory fitting/sampling",
              "numerical_gradcheck": "CPU float64 only; CUDA checks real batch forward/backward",
              "groups": groups}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"trajectory ({device}): {len(groups)} groups passed; result: {args.results}")


if __name__ == "__main__":
    main()
