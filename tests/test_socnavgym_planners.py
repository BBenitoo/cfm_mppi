from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from cfm_mppi.evaluation.socnavgym_adapter import (
    HumanState,
    SocNavState,
    SocNavStep,
)
from cfm_mppi.evaluation.socnavgym_planners import (
    SocNavCFMMPPIPlanner,
    SocNavPlannerConfig,
    SocNavVRCMPPIPlanner,
    socnav_diff_drive_dynamics,
)
from cfm_mppi.evaluation.socnavgym_runner import (
    EpisodeContext,
    PlanningBudget,
)


class _CaptureSolver:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _planner_config() -> SocNavPlannerConfig:
    return SocNavPlannerConfig(
        horizon=6,
        max_history=2,
        cfm_candidates=4,
        branches=2,
        mppi_samples_per_branch=3,
        safe_margin_coefs=(0.2, 0.4),
        ode_times_initial=(0.5, 1.0),
        ode_times_warm=(0.8, 1.0),
    )


def _human(
    human_id: int,
    position: tuple[float, float],
    velocity: tuple[float, float],
    *,
    radius: float = 0.36,
) -> HumanState:
    return HumanState(
        id=human_id,
        position=np.asarray(position, dtype=np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        radius=radius,
        orientation=0.0,
        gaze=False,
    )


def _state(
    robot_position: tuple[float, float] = (1.0, -0.5),
    *,
    heading: float = 0.2,
    human_position: tuple[float, float] = (2.0, 1.0),
    human_velocity: tuple[float, float] = (0.1, -0.2),
) -> SocNavState:
    return SocNavState(
        robot_state=np.asarray(
            [robot_position[0], robot_position[1], heading],
            dtype=np.float32,
        ),
        goal=np.asarray([4.0, 3.0], dtype=np.float32),
        robot_body_velocity=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        robot_radius=0.25,
        goal_radius=0.35,
        humans=(
            _human(7, human_position, human_velocity),
        ),
    )


def _context(seed: int = 19) -> EpisodeContext:
    return EpisodeContext(
        env_seed=5,
        planner_seed=seed,
        time_step=0.1,
        episode_length=256,
        max_human_speed=0.73,
        control_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        control_high=np.asarray([1.0, 1.0], dtype=np.float32),
    )


def _branch_plan(kwargs, *, selected_vrc_tube=None):
    history_length = len(kwargs["histories"]["ego_control_sin"])
    total_horizon = kwargs["planning_horizon"]
    future_horizon = total_horizon - history_length
    branches = kwargs["num_branches"]
    device = kwargs["state"].device
    selected_controls = torch.cat(
        (
            torch.full((future_horizon, 1), 0.25, device=device),
            torch.full((future_horizon, 1), -0.1, device=device),
        ),
        dim=1,
    )
    times = torch.arange(
        future_horizon + 1,
        device=device,
        dtype=torch.float32,
    )
    branch_states = kwargs["state"].expand(branches, -1).unsqueeze(1).repeat(
        1, future_horizon + 1, 1
    )
    branch_states[:, :, 0] += 0.025 * times
    branch_states[:, :, 1] -= 0.01 * times
    branch_controls = selected_controls.unsqueeze(0).expand(branches, -1, -1)
    pedestrian_predictions = kwargs["current_positions"].squeeze(0).unsqueeze(
        0
    ).unsqueeze(2).expand(branches, -1, future_horizon, -1).clone()
    pedestrian_predictions[..., 0] += 0.01
    return SimpleNamespace(
        selected_controls=selected_controls,
        selected_cfm_controls=torch.full(
            (1, 2, total_horizon),
            0.5,
            device=device,
        ),
        selected_branch=1,
        cfm_branch_indices=torch.arange(branches, device=device),
        cfm_branch_states=branch_states,
        cfm_branch_controls=branch_controls,
        pedestrian_predictions=pedestrian_predictions,
        mppi_branch_states=branch_states + torch.tensor(
            [0.02, -0.01, 0.0], device=device
        ),
        branch_costs=torch.arange(branches, device=device, dtype=torch.float32),
        selected_vrc_tube=selected_vrc_tube,
    )


class SocNavDynamicsTest(unittest.TestCase):
    def test_rotates_before_translating_and_wraps_heading(self) -> None:
        states = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -2.0, np.pi - 0.05]],
            dtype=torch.float64,
        )
        actions = torch.tensor(
            [[1.0, 1.0], [0.5, 1.0]],
            dtype=torch.float64,
        )

        result = socnav_diff_drive_dynamics(states, actions, delta_t=0.1)

        expected_headings = torch.atan2(
            torch.sin(states[:, 2] + 0.1 * actions[:, 1]),
            torch.cos(states[:, 2] + 0.1 * actions[:, 1]),
        )
        expected = torch.stack(
            (
                states[:, 0] + 0.1 * actions[:, 0] * torch.cos(expected_headings),
                states[:, 1] + 0.1 * actions[:, 0] * torch.sin(expected_headings),
                expected_headings,
            ),
            dim=1,
        )
        torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-12)
        self.assertGreater(float(result[0, 1]), 0.0)
        self.assertLess(float(result[1, 2]), -3.0)


class SharedPlannerContractTest(unittest.TestCase):
    def _make_planners(self):
        config = _planner_config()
        baseline = SocNavCFMMPPIPlanner(
            torch.nn.Identity(),
            config=config,
            device="cpu",
            solver_factory=_CaptureSolver,
        )
        vrc = SocNavVRCMPPIPlanner(
            torch.nn.Identity(),
            config=config,
            device="cpu",
            solver_factory=_CaptureSolver,
        )
        return baseline, vrc

    def test_both_planners_have_identical_explicit_budget_and_dynamics(self) -> None:
        baseline, vrc = self._make_planners()
        initial = _state()
        context = _context()
        global_rng_before = torch.random.get_rng_state().clone()

        baseline.reset_episode(initial, context)
        vrc.reset_episode(initial, context)

        self.assertEqual(baseline.budget, PlanningBudget(4, 6))
        self.assertEqual(vrc.budget, baseline.budget)
        self.assertEqual(
            baseline._solver.kwargs["num_samples"],
            _planner_config().mppi_samples_per_branch,
        )
        self.assertEqual(
            vrc._solver.kwargs["num_samples"],
            _planner_config().mppi_samples_per_branch,
        )
        torch.testing.assert_close(
            baseline._solver.kwargs["u_min"],
            torch.tensor([-1.0, -1.0]),
        )
        torch.testing.assert_close(
            vrc._solver.kwargs["u_max"],
            torch.tensor([1.0, 1.0]),
        )
        robot = torch.tensor([[0.0, 0.0, 0.0]])
        control = torch.tensor([[0.5, 0.7]])
        torch.testing.assert_close(
            baseline._solver.kwargs["dynamics"](robot, control),
            vrc._solver.kwargs["dynamics"](robot, control),
        )
        torch.testing.assert_close(baseline._warm_start, vrc._warm_start)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before)

    def test_cv_and_vrc_route_only_the_pedestrian_prediction_differently(self) -> None:
        baseline, vrc = self._make_planners()
        initial = _state()
        context = _context()
        baseline.reset_episode(initial, context)
        vrc.reset_episode(initial, context)

        with (
            mock.patch(
                "cfm_mppi.evaluation.socnavgym_planners."
                "plan_with_constant_velocity_prediction",
                side_effect=lambda **kwargs: _branch_plan(kwargs),
            ) as cv_plan,
            mock.patch(
                "cfm_mppi.evaluation.socnavgym_planners.plan_vrc_branches",
                side_effect=lambda **kwargs: _branch_plan(kwargs),
            ) as vrc_plan,
        ):
            baseline_command = baseline.plan(initial, 0)
            vrc_command = vrc.plan(initial, 0)

        np.testing.assert_array_equal(
            baseline_command.control,
            vrc_command.control,
        )
        for call in (cv_plan.call_args, vrc_plan.call_args):
            self.assertFalse(call.kwargs["build_selected_vrc_tube"])
            self.assertEqual(call.kwargs["planning_horizon"], 6)
            self.assertEqual(call.kwargs["num_branches"], 2)
            self.assertEqual(call.kwargs["look_ahead_distance"], 0.1)
        self.assertNotIn("vrc_params", cv_plan.call_args.kwargs)
        self.assertIn("vrc_params", vrc_plan.call_args.kwargs)
        self.assertAlmostEqual(
            vrc_plan.call_args.kwargs["vrc_params"].robot_radius,
            0.25,
        )
        self.assertAlmostEqual(
            vrc_plan.call_args.kwargs["vrc_params"].pedestrian_radius,
            0.36,
        )
        self.assertAlmostEqual(
            vrc_plan.call_args.kwargs["prediction_params"].maximum_speed,
            context.max_human_speed,
        )
        self.assertEqual(
            baseline_command.diagnostics["refinement_rollouts"],
            vrc_command.diagnostics["refinement_rollouts"],
        )

    def test_vrc_planner_rejects_any_selected_environment_tube(self) -> None:
        _, planner = self._make_planners()
        initial = _state()
        planner.reset_episode(initial, _context())

        with mock.patch(
            "cfm_mppi.evaluation.socnavgym_planners.plan_vrc_branches",
            side_effect=lambda **kwargs: _branch_plan(
                kwargs,
                selected_vrc_tube=object(),
            ),
        ), self.assertRaisesRegex(RuntimeError, "remain internal"):
            planner.plan(initial, 0)

    def test_visualization_trace_is_opt_in_and_uses_causal_vrc_branch(self) -> None:
        config = _planner_config()
        initial = _state()
        context = _context()
        baseline = SocNavCFMMPPIPlanner(
            torch.nn.Identity(),
            config=config,
            device="cpu",
            solver_factory=_CaptureSolver,
            record_visualization=True,
        )
        vrc = SocNavVRCMPPIPlanner(
            torch.nn.Identity(),
            config=config,
            device="cpu",
            solver_factory=_CaptureSolver,
            record_visualization=True,
        )
        baseline.reset_episode(initial, context)
        vrc.reset_episode(initial, context)

        with (
            mock.patch(
                "cfm_mppi.evaluation.socnavgym_planners."
                "plan_with_constant_velocity_prediction",
                side_effect=lambda **kwargs: _branch_plan(kwargs),
            ),
            mock.patch(
                "cfm_mppi.evaluation.socnavgym_planners.plan_vrc_branches",
                side_effect=lambda **kwargs: _branch_plan(kwargs),
            ),
        ):
            baseline_trace = baseline.plan(initial, 0).diagnostics[
                "visualization"
            ]
            vrc_trace = vrc.plan(initial, 0).diagnostics["visualization"]

        self.assertEqual(baseline_trace["human_ids"], [7])
        self.assertIsNone(baseline_trace["pedestrian_prediction_vrc"])
        self.assertIsNone(baseline_trace["vrc_tube"])
        self.assertEqual(
            tuple(vrc_trace["robot_candidate_trajectories"].shape),
            (2, 7, 3),
        )
        self.assertEqual(tuple(vrc_trace["robot_prediction"].shape), (7, 3))
        self.assertEqual(
            tuple(vrc_trace["pedestrian_prediction_no_vrc"].shape),
            (1, 6, 2),
        )
        self.assertEqual(
            tuple(vrc_trace["pedestrian_prediction_vrc"].shape),
            (1, 6, 2),
        )
        self.assertEqual(tuple(vrc_trace["vrc_forces"].shape), (1, 2))
        self.assertEqual(tuple(vrc_trace["vrc_tube"]["centers"].shape), (7, 2))
        torch.testing.assert_close(
            vrc_trace["robot_conditioning_trajectory"],
            vrc_trace["robot_candidate_trajectories"][1],
        )
        self.assertFalse(
            torch.equal(
                vrc_trace["robot_conditioning_trajectory"],
                vrc_trace["robot_prediction"],
            )
        )


class PlannerHistoryAndWarmStartTest(unittest.TestCase):
    def test_commits_synchronized_environment_truth_and_keeps_fixed_horizon(self):
        config = _planner_config()
        planner = SocNavCFMMPPIPlanner(
            torch.nn.Identity(),
            config=config,
            device="cpu",
            solver_factory=_CaptureSolver,
        )
        context = _context()
        current = _state()
        planner.reset_episode(current, context)
        planning_horizons = []

        def fake_plan(**kwargs):
            planning_horizons.append(kwargs["planning_horizon"])
            return _branch_plan(kwargs)

        with mock.patch(
            "cfm_mppi.evaluation.socnavgym_planners."
            "plan_with_constant_velocity_prediction",
            side_effect=fake_plan,
        ):
            for step_index in range(5):
                command = planner.plan(current, step_index)
                next_state = _state(
                    robot_position=(
                        float(current.robot_position[0] + 0.05),
                        float(current.robot_position[1] + 0.02),
                    ),
                    heading=float(current.robot_heading + 0.01),
                    human_position=(
                        float(current.human_positions[0, 0] + 0.01),
                        float(current.human_positions[0, 1] - 0.02),
                    ),
                    human_velocity=(0.1, -0.2),
                )
                transition = SocNavStep(
                    state=next_state,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={},
                    requested_control=command.control.copy(),
                    applied_control=command.control.copy(),
                    applied_action=np.asarray([0.25, 0.0, -0.1], dtype=np.float32),
                )
                previous = current
                planner.observe_transition(previous, command, transition)
                current = next_state

                self.assertEqual(planner.warm_start_shape, (4, 2, 6))
                snapshot = planner.history_snapshot()
                history_length = planner.history_length
                expected_prefix = (
                    snapshot["ego_control_sin"].expand(4, -1, -1) / 10.0
                )
                torch.testing.assert_close(
                    planner._warm_start[:, :, :history_length],
                    expected_prefix,
                )

                if step_index == 0:
                    torch.testing.assert_close(
                        snapshot["ego_state"][0, :, 0],
                        torch.tensor(previous.robot_state),
                    )
                    torch.testing.assert_close(
                        snapshot["obs_state"][0, 0, :, 0],
                        torch.tensor(next_state.human_positions[0]),
                    )
                    torch.testing.assert_close(
                        snapshot["ego_control_sin"][0, :, 0],
                        torch.tensor([0.5, 0.2]),
                        rtol=0.0,
                        atol=1e-6,
                    )

        self.assertEqual(planning_horizons, [6, 6, 6, 6, 6])
        self.assertEqual(planner.history_length, 2)
        snapshot = planner.history_snapshot()
        # After five transitions a max-two buffer contains q_3,q_4 and p_4,p_5.
        torch.testing.assert_close(
            snapshot["ego_state"][0, :2, 0],
            torch.tensor([1.15, -0.44]),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            snapshot["ego_state"][0, :2, 1],
            torch.tensor([1.20, -0.42]),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            snapshot["obs_state"][0, 0, :, -1],
            torch.tensor([2.05, 0.90]),
            rtol=0.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
