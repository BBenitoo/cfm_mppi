import unittest
from unittest.mock import patch

import torch

from cfm_mppi.evaluation.eval_utils import CFMConfig
from cfm_mppi.evaluation.eval_vrc import (
    constant_velocity_prediction,
    plan_vrc_branches,
)


class _FakeBranchSolver:
    def __init__(self) -> None:
        self.received_predictions = None

    def forward_branches(
        self,
        state,
        branch_controls,
        goal,
        branch_obstacle_states,
        rad,
    ):
        del state, goal, rad
        self.received_predictions = branch_obstacle_states.clone()
        branch_count, horizon, _ = branch_controls.shape
        selected_controls = torch.full((horizon, 2), 0.75)
        optimal_states = torch.zeros((branch_count, horizon + 1, 3))
        optimal_states[1] = 2.0
        branch_costs = torch.tensor([4.0, 1.0])
        return (
            selected_controls,
            branch_controls,
            optimal_states,
            branch_costs,
        )


class VRCPlanningCharacterizationTest(unittest.TestCase):
    def test_constant_velocity_prediction_starts_at_t_plus_one(self) -> None:
        positions = torch.tensor(
            [[[1.0, -2.0], [0.5, 3.0]]],
            dtype=torch.float64,
        )
        velocities = torch.tensor(
            [[[2.0, 1.0], [-1.0, 0.5]]],
            dtype=torch.float64,
        )

        prediction = constant_velocity_prediction(
            positions,
            velocities,
            horizon=3,
            dt=0.1,
        )

        expected = torch.stack(
            [positions + velocities * offset for offset in (0.1, 0.2, 0.3)],
            dim=-1,
        )
        self.assertEqual(prediction.shape, (1, 2, 2, 3))
        torch.testing.assert_close(prediction, expected)
        self.assertFalse(torch.equal(prediction[..., 0], positions))

    def test_disabling_environment_vrc_keeps_branch_predictions(self) -> None:
        horizon = 3
        cfm_controls = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        future_cfm_controls = torch.zeros((4, horizon, 2))
        constant_velocity_obstacles = torch.zeros((2, horizon, 2))
        branch_indices = torch.tensor([3, 1])
        branch_states = torch.zeros((2, horizon + 1, 3))
        branch_controls = torch.zeros((2, horizon, 2))
        branch_predictions = torch.arange(
            2 * 2 * horizon * 2,
            dtype=torch.float32,
        ).reshape(2, 2, horizon, 2)
        solver = _FakeBranchSolver()

        with (
            patch(
                "cfm_mppi.evaluation.eval_vrc.generate_cfm_candidates",
                return_value=(
                    cfm_controls,
                    future_cfm_controls,
                    constant_velocity_obstacles,
                    0,
                ),
            ),
            patch(
                "cfm_mppi.evaluation.eval_vrc.select_cfm_branches",
                return_value=(branch_indices, branch_states, branch_controls),
            ),
            patch(
                "cfm_mppi.evaluation.eval_vrc." "build_branch_pedestrian_predictions",
                return_value=branch_predictions,
            ) as build_predictions,
            patch(
                "cfm_mppi.evaluation.eval_vrc.build_vrc_tube",
            ) as build_scalar_tube,
        ):
            plan = plan_vrc_branches(
                model=object(),
                solver=solver,
                config=CFMConfig(ode_times=[1.0], device="cpu"),
                state=torch.zeros((1, 3)),
                goal=torch.zeros((1, 2)),
                noisy_action_seq=torch.zeros((4, 2, horizon)),
                noise_level=torch.zeros(1),
                current_positions=torch.zeros((1, 2, 2)),
                current_velocities=torch.zeros((1, 2, 2)),
                planning_horizon=horizon,
                histories={},
                num_branches=2,
                build_selected_vrc_tube=False,
            )

        build_predictions.assert_called_once()
        prediction_kwargs = build_predictions.call_args.kwargs
        torch.testing.assert_close(prediction_kwargs["branch_states"], branch_states)
        torch.testing.assert_close(
            prediction_kwargs["branch_controls"], branch_controls
        )
        torch.testing.assert_close(
            prediction_kwargs["current_positions"], torch.zeros((1, 2, 2))
        )
        torch.testing.assert_close(
            prediction_kwargs["current_velocities"], torch.zeros((1, 2, 2))
        )
        self.assertEqual(prediction_kwargs["dt"], 0.1)
        build_scalar_tube.assert_not_called()
        torch.testing.assert_close(solver.received_predictions, branch_predictions)
        torch.testing.assert_close(plan.pedestrian_predictions, branch_predictions)
        torch.testing.assert_close(plan.cfm_branch_controls, branch_controls)
        self.assertEqual(plan.selected_branch, 1)
        torch.testing.assert_close(plan.selected_cfm_controls, cfm_controls[1:2])
        self.assertIsNone(plan.selected_vrc_tube)


if __name__ == "__main__":
    unittest.main()
