import unittest
from unittest.mock import patch

import torch

from cfm_mppi.evaluation.eval_utils import CFMConfig
from cfm_mppi.evaluation.eval_vrc_cv_prediction import (
    plan_with_constant_velocity_prediction,
)


class _FakeBranchSolver:
    def __init__(self) -> None:
        self.branch_obstacle_states = None

    def forward_branches(
        self,
        state,
        branch_controls,
        goal,
        branch_obstacle_states,
        rad,
    ):
        self.branch_obstacle_states = branch_obstacle_states.clone()
        branch_states = torch.zeros((2, 4, 3))
        branch_states[1] = 1.0
        branch_costs = torch.tensor([3.0, 1.0])
        selected_controls = torch.full((3, 2), 7.0)
        return (
            selected_controls,
            branch_controls,
            branch_states,
            branch_costs,
        )


class ConstantVelocityPredictionPlanTest(unittest.TestCase):
    def test_all_mppi_branches_receive_the_same_cv_prediction(self) -> None:
        cfm_controls = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        future_cfm_controls = torch.zeros((3, 3, 2))
        constant_velocity_obstacles = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            ]
        )
        branch_indices = torch.tensor([2, 0])
        branch_states = torch.zeros((2, 4, 3))
        branch_controls = torch.zeros((2, 3, 2))
        solver = _FakeBranchSolver()
        selected_tube = object()

        with (
            patch(
                "cfm_mppi.evaluation.eval_vrc_cv_prediction."
                "generate_cfm_candidates",
                return_value=(
                    cfm_controls,
                    future_cfm_controls,
                    constant_velocity_obstacles,
                    0,
                ),
            ),
            patch(
                "cfm_mppi.evaluation.eval_vrc_cv_prediction.select_cfm_branches",
                return_value=(branch_indices, branch_states, branch_controls),
            ),
            patch(
                "cfm_mppi.evaluation.eval_vrc_cv_prediction.build_vrc_tube",
                return_value=selected_tube,
            ) as build_tube,
        ):
            plan = plan_with_constant_velocity_prediction(
                model=object(),
                solver=solver,
                config=CFMConfig(ode_times=[1.0], device="cpu"),
                state=torch.zeros((1, 3)),
                goal=torch.zeros((1, 2)),
                noisy_action_seq=torch.zeros((3, 2, 3)),
                noise_level=torch.zeros(1),
                current_positions=torch.zeros((1, 2, 2)),
                current_velocities=torch.zeros((1, 2, 2)),
                planning_horizon=3,
                histories={},
                num_branches=2,
            )

        expected_predictions = constant_velocity_obstacles.unsqueeze(0).expand(
            2,
            -1,
            -1,
            -1,
        )
        torch.testing.assert_close(
            solver.branch_obstacle_states,
            expected_predictions,
        )
        torch.testing.assert_close(plan.pedestrian_predictions, expected_predictions)
        self.assertEqual(plan.selected_branch, 1)
        torch.testing.assert_close(plan.selected_cfm_controls, cfm_controls[0:1])
        self.assertIs(plan.selected_vrc_tube, selected_tube)
        build_tube.assert_called_once()
        torch.testing.assert_close(
            build_tube.call_args.kwargs["states"],
            torch.ones((4, 3)),
        )
        torch.testing.assert_close(
            build_tube.call_args.kwargs["controls_uni"],
            torch.full((3, 2), 7.0),
        )


if __name__ == "__main__":
    unittest.main()
