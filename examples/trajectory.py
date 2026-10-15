"""Run in Docker: just run python examples/trajectory.py."""

import torch

from devtools.trajectory import TrajectoryState, fit_trajectory


def main():
    dtype = torch.float64
    initial = TrajectoryState(
        position=torch.tensor([0., 0., 0.], dtype=dtype),
        velocity=torch.tensor([1., -0.5, 0.25], dtype=dtype),
        acceleration=torch.tensor([0.2, 0.4, -0.1], dtype=dtype),
        yaw=torch.tensor(0.3, dtype=dtype),
        yaw_rate=torch.tensor(0.2, dtype=dtype),
    )
    # Illustrative algorithm endpoint, not a model or training loop.
    endpoint = torch.tensor([3., 1., -0.5], dtype=dtype, requires_grad=True)
    terminal = TrajectoryState(
        position=endpoint,
        velocity=torch.tensor([0.5, 0., -0.25], dtype=dtype),
        acceleration=torch.zeros(3, dtype=dtype),
        yaw=torch.tensor(0.8, dtype=dtype),
        yaw_rate=torch.tensor(-0.1, dtype=dtype),
    )
    duration = torch.tensor(2., dtype=dtype, requires_grad=True)
    trajectory = fit_trajectory(initial, terminal, duration)
    # Fractions are caller-owned: these sample times also depend on T.
    fractions = torch.tensor([0., 0.5, 1.], dtype=dtype)
    samples = trajectory.sample(duration * fractions)
    print("position at 0, T/2, T (m):", samples.position.tolist())
    print("yaw (rad):", samples.yaw.tolist())
    print("yaw_rate (rad/s):", samples.yaw_rate.tolist())
    probe = samples.position[1].square().sum() + samples.yaw[1].square()
    endpoint_gradient, duration_gradient = torch.autograd.grad(probe, (endpoint, duration))
    print("probe gradient to endpoint:", endpoint_gradient.tolist())
    print("probe gradient to T:", duration_gradient.item())


if __name__ == "__main__":
    main()
