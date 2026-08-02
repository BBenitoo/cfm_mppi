import unittest

import torch

from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import unicycle_dynamics


def _zero_terminal_cost(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    del goal
    return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)


def _single_integrator_dynamics(
    state: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    return state + 0.1 * action


class _ZeroNoise:
    def __init__(self, control_dim: int, dtype: torch.dtype) -> None:
        self.control_dim = control_dim
        self.dtype = dtype

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:
        return torch.zeros((*sample_shape, self.control_dim), dtype=self.dtype)


class _FixedNoise:
    def __init__(self, samples: torch.Tensor) -> None:
        self.samples = samples

    def rsample(self, sample_shape: torch.Size) -> torch.Tensor:
        branch_count, sample_count, horizon = sample_shape
        if self.samples.shape != (sample_count, horizon, 2):
            raise AssertionError(
                f"Unexpected requested sample shape: {tuple(sample_shape)}"
            )
        return self.samples.unsqueeze(0).expand(branch_count, -1, -1, -1).clone()


def _make_solver(stage_cost, *, num_samples: int = 4) -> FlowMPPI:
    solver = FlowMPPI(
        num_samples=num_samples,
        dim_state=2,
        dim_control=2,
        dynamics=_single_integrator_dynamics,
        stage_cost=stage_cost,
        terminal_cost=_zero_terminal_cost,
        u_min=torch.tensor([-2.0, -2.0], dtype=torch.float64),
        u_max=torch.tensor([2.0, 2.0], dtype=torch.float64),
        sigmas=torch.tensor([0.5, 0.5], dtype=torch.float64),
        lambda_=1.0,
        goal=torch.zeros(2, dtype=torch.float64),
        horizon=2,
        dt=0.1,
        device=torch.device("cpu"),
        dtype=torch.float64,
        dynamics_type="singleintegrator",
    )
    solver._noise_distribution = _ZeroNoise(2, torch.float64)
    return solver


class MPPICharacterizationTest(unittest.TestCase):
    def test_score_pairs_first_future_obstacle_with_initial_state(self) -> None:
        def squared_distance_stage(
            state,
            action,
            goal,
            obstacle,
            radius,
            time,
            prev_action,
        ):
            del action, goal, radius, time, prev_action
            return torch.sum((state[:, :2] - obstacle[0, :2]) ** 2, dim=1)

        solver = _make_solver(squared_distance_stage)
        states = torch.tensor(
            [[[0.0, 0.0], [3.0, 0.0], [8.0, 0.0]]],
            dtype=torch.float64,
        )
        controls = torch.zeros((1, 2, 2), dtype=torch.float64)
        future_obstacles = torch.tensor(
            [[[1.0, 0.0], [5.0, 0.0]]],
            dtype=torch.float64,
        )

        cost = solver.score_trajectories(
            states=states,
            controls=controls,
            goal=torch.zeros(2, dtype=torch.float64),
            obstacle_state=future_obstacles,
            rad=0.5,
        )

        # Legacy contract: p[t+1:t+H] is paired with x[t:t+H-1].
        # A future synchronized-time change must update this test explicitly.
        torch.testing.assert_close(cost, torch.tensor([5.0], dtype=torch.float64))

    def test_changing_one_branch_obstacles_does_not_change_other_branch(self) -> None:
        def obstacle_distance_stage(
            state,
            action,
            goal,
            obstacle,
            radius,
            time,
            prev_action,
        ):
            del action, goal, radius, time, prev_action
            differences = state[:, None, :2] - obstacle[None, :, :2]
            return torch.sum(differences**2, dim=(1, 2))

        state = torch.zeros((1, 2), dtype=torch.float64)
        controls = torch.zeros((2, 2, 2), dtype=torch.float64)
        obstacles = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[2.0, 0.0], [2.0, 0.0]]],
            ],
            dtype=torch.float64,
        )
        changed_obstacles = obstacles.clone()
        changed_obstacles[1, :, :, 0] = -2.0

        fixed_noises = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[-1.0, 0.0], [-1.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.5, 0.0]],
            ],
            dtype=torch.float64,
        )

        first_solver = _make_solver(obstacle_distance_stage)
        second_solver = _make_solver(obstacle_distance_stage)
        first_solver._noise_distribution = _FixedNoise(fixed_noises)
        second_solver._noise_distribution = _FixedNoise(fixed_noises)
        first = first_solver.forward_branches(
            state,
            controls,
            torch.zeros((1, 2), dtype=torch.float64),
            obstacles,
            rad=0.5,
        )
        second = second_solver.forward_branches(
            state,
            controls,
            torch.zeros((1, 2), dtype=torch.float64),
            changed_obstacles,
            rad=0.5,
        )

        _, first_controls, first_states, first_costs = first
        _, second_controls, second_states, second_costs = second
        torch.testing.assert_close(first_controls[0], second_controls[0])
        torch.testing.assert_close(first_states[0], second_states[0])
        torch.testing.assert_close(first_costs[0], second_costs[0])
        self.assertFalse(torch.allclose(first_controls[1], second_controls[1]))
        self.assertFalse(torch.isclose(first_costs[1], second_costs[1]).item())

    def test_unicycle_translation_uses_pre_rotation_heading(self) -> None:
        state = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        action = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

        next_state = unicycle_dynamics(state, action, delta_t=0.1)

        torch.testing.assert_close(
            next_state,
            torch.tensor([[0.1, 0.0, 0.1]], dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
