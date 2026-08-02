import random
import json
import unittest

import numpy as np

from cfm_mppi.evaluation.socnavgym_adapter import (
    HumanState,
    SocNavGymAdapter,
    SocNavState,
    SocNavStep,
)
from cfm_mppi.evaluation.socnavgym_runner import (
    PlannerCommand,
    PlanningBudget,
    RunnerContractError,
    assert_matching_budgets,
    run_paired_socnavgym_evaluation,
    run_socnavgym_episode,
)


def _state(robot_x, *, human_id=7, human_x=2.0):
    return SocNavState(
        robot_state=np.asarray([robot_x, 0.0, 0.0], dtype=np.float32),
        goal=np.asarray([20.0, 0.0], dtype=np.float32),
        robot_body_velocity=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        robot_radius=0.25,
        goal_radius=0.3,
        humans=(
            HumanState(
                id=human_id,
                position=np.asarray([human_x, 0.0], dtype=np.float32),
                velocity=np.asarray([0.1, 0.0], dtype=np.float32),
                radius=0.35,
                orientation=0.0,
                gaze=False,
            ),
        ),
    )


class _FakeEnvironment:
    time_step = 0.1
    episode_length = 4
    max_human_speed = 0.8
    physical_control_low = np.asarray([-1.0, -2.0], dtype=np.float32)
    physical_control_high = np.asarray([1.0, 2.0], dtype=np.float32)

    def __init__(
        self,
        *,
        terminate_after=2,
        clipped=False,
        changed_id=False,
        first_step_collision=False,
        minimum_distances=None,
    ):
        self.terminate_after = terminate_after
        self.clipped = clipped
        self.changed_id = changed_id
        self.first_step_collision = first_step_collision
        self.minimum_distances = minimum_distances
        self.controls = []
        self.random_draws = []
        self.step_index = 0

    def reset(self, *, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.step_index = 0
        return _state(0.0), {"reset": seed}

    def step(self, control):
        received = np.asarray(control, dtype=np.float32).copy()
        self.controls.append(received)
        self.random_draws.append((random.random(), float(np.random.random())))
        self.step_index += 1
        terminated = self.step_index >= self.terminate_after
        human_id = 99 if self.changed_id else 7
        requested = received.copy()
        applied = received.copy()
        if self.clipped:
            applied[0] -= 0.25
        info = (
            {
                "SUCCESS": True,
                "PATH_LENGTH": 3.25,
                "MINIMUM_DISTANCE_TO_HUMAN": 0.75,
                "TIME_TO_REACH_GOAL": 0.2,
            }
            if terminated
            else (
                {"COLLISION_WALL": True}
                if self.first_step_collision and self.step_index == 1
                else {}
            )
        )
        if self.minimum_distances is not None:
            info["MINIMUM_DISTANCE_TO_HUMAN"] = self.minimum_distances[
                self.step_index - 1
            ]
        return SocNavStep(
            state=_state(
                10.0 + self.step_index,
                human_id=human_id,
                human_x=2.0 + self.step_index,
            ),
            reward=1.5,
            terminated=terminated,
            truncated=False,
            info=info,
            requested_control=requested,
            applied_control=applied,
            applied_action=np.asarray(
                [applied[0], 0.0, applied[1] / 2.0], dtype=np.float32
            ),
        )


class _FakePlanner:
    name = "fake"
    budget = PlanningBudget(cfm_candidates=20, refinement_rollouts=200)

    def __init__(self, control=(0.5, 0.2)):
        self.control = np.asarray(control, dtype=np.float32)
        self.context = None
        self.planned_robot_x = []
        self.transitions = []
        self.synchronize_calls = 0

    def reset_episode(self, initial_state, context):
        random.random()
        np.random.random()
        self.context = context
        self.initial_state = initial_state

    def plan(self, state, step_index):
        random.random()
        np.random.random()
        self.planned_robot_x.append(float(state.robot_position[0]))
        return PlannerCommand(
            self.control,
            diagnostics={"planner_only": np.asarray([step_index])},
        )

    def observe_transition(self, previous_state, command, transition):
        random.random()
        np.random.random()
        self.transitions.append((previous_state, command, transition))

    def synchronize(self):
        self.synchronize_calls += 1


class SocNavGymRunnerTest(unittest.TestCase):
    def test_uses_environment_truth_and_keeps_diagnostics_out_of_step(self):
        environment = _FakeEnvironment(terminate_after=2)
        planner = _FakePlanner()

        result = run_socnavgym_episode(
            environment,
            planner,
            env_seed=13,
            planner_seed=31,
        )

        self.assertEqual(planner.planned_robot_x, [0.0, 11.0])
        self.assertEqual(len(environment.controls), 2)
        for control in environment.controls:
            self.assertEqual(control.shape, (2,))
            np.testing.assert_allclose(control, [0.5, 0.2])
        self.assertEqual(len(planner.transitions), 1)
        self.assertEqual(result.context.env_seed, 13)
        self.assertEqual(result.context.planner_seed, 31)
        np.testing.assert_array_equal(result.context.control_low, [-1.0, -2.0])
        self.assertFalse(result.runner_limit_reached)
        self.assertEqual(len(result.steps), 2)
        self.assertTrue(result.steps[-1].terminated)

        summary = result.summary()
        self.assertTrue(summary["success"])
        self.assertEqual(summary["environment_path_length"], 3.25)
        self.assertEqual(summary["environment_minimum_distance_to_human"], 0.75)
        self.assertEqual(summary["environment_time_to_reach_goal"], 0.2)
        self.assertEqual(result.to_dict()["steps"][0]["diagnostics"]["planner_only"], [0])
        json.dumps(result.to_dict())

    def test_aggregates_nonterminal_collision_events(self):
        result = run_socnavgym_episode(
            _FakeEnvironment(terminate_after=2, first_step_collision=True),
            _FakePlanner(),
            env_seed=9,
        )

        summary = result.summary()
        self.assertTrue(summary["collision_wall"])
        self.assertTrue(summary["collision_any"])
        self.assertFalse(summary["collision"])

    def test_aggregates_environment_minimum_distance_across_steps(self):
        result = run_socnavgym_episode(
            _FakeEnvironment(
                terminate_after=3,
                minimum_distances=(0.4, 0.9, 0.7),
            ),
            _FakePlanner(),
            env_seed=8,
        )

        self.assertEqual(
            result.summary()["environment_minimum_distance_to_human"],
            0.4,
        )

    def test_snapshots_diagnostics_and_rejects_large_payloads(self):
        reusable = np.asarray([1.0, 2.0], dtype=np.float32)

        class SnapshotPlanner(_FakePlanner):
            def plan(self, state, step_index):
                del state, step_index
                return PlannerCommand([0.0, 0.0], {"values": reusable})

        result = run_socnavgym_episode(
            _FakeEnvironment(terminate_after=1),
            SnapshotPlanner(),
            env_seed=4,
        )
        reusable[:] = 99.0
        self.assertEqual(result.steps[0].diagnostics["values"], [1.0, 2.0])

        class LargeDiagnosticPlanner(_FakePlanner):
            def plan(self, state, step_index):
                del state, step_index
                return PlannerCommand(
                    [0.0, 0.0],
                    {"rollout": np.zeros(4097, dtype=np.float32)},
                )

        with self.assertRaisesRegex(RunnerContractError, "exceeds element limit"):
            run_socnavgym_episode(
                _FakeEnvironment(terminate_after=1),
                LargeDiagnosticPlanner(),
                env_seed=4,
            )

    def test_planner_does_not_perturb_environment_python_or_numpy_rng(self):
        seed = 23
        environment = _FakeEnvironment(terminate_after=2)
        planner = _FakePlanner()

        run_socnavgym_episode(environment, planner, env_seed=seed)

        expected_python = random.Random(seed)
        expected_numpy = np.random.RandomState(seed)
        expected = [
            (expected_python.random(), float(expected_numpy.random_sample()))
            for _ in range(2)
        ]
        np.testing.assert_allclose(environment.random_draws, expected)

    def test_rejects_out_of_bounds_or_clipped_control(self):
        out_of_bounds_environment = _FakeEnvironment()
        with self.assertRaisesRegex(RunnerContractError, "outside the shared bounds"):
            run_socnavgym_episode(
                out_of_bounds_environment,
                _FakePlanner(control=(1.1, 0.0)),
                env_seed=1,
            )
        self.assertEqual(out_of_bounds_environment.controls, [])

        with self.assertRaisesRegex(RunnerContractError, "clipped"):
            run_socnavgym_episode(
                _FakeEnvironment(clipped=True),
                _FakePlanner(),
                env_seed=1,
            )

    def test_rejects_human_identity_change(self):
        with self.assertRaisesRegex(RunnerContractError, "identity/order changed"):
            run_socnavgym_episode(
                _FakeEnvironment(changed_id=True),
                _FakePlanner(),
                env_seed=2,
            )

    def test_explicit_runner_limit_is_distinct_from_environment_truncation(self):
        result = run_socnavgym_episode(
            _FakeEnvironment(terminate_after=4),
            _FakePlanner(),
            env_seed=3,
            max_steps=1,
        )

        self.assertTrue(result.runner_limit_reached)
        self.assertFalse(result.steps[-1].terminated)
        self.assertFalse(result.steps[-1].truncated)

    def test_rejects_invalid_step_limits_and_missing_environment_truncation(self):
        for value in (True, 1.5):
            with self.assertRaises(TypeError):
                run_socnavgym_episode(
                    _FakeEnvironment(),
                    _FakePlanner(),
                    env_seed=1,
                    max_steps=value,
                )
        with self.assertRaises(ValueError):
            run_socnavgym_episode(
                _FakeEnvironment(),
                _FakePlanner(),
                env_seed=1,
                max_steps=4,
            )
        with self.assertRaisesRegex(RunnerContractError, "exhausted"):
            run_socnavgym_episode(
                _FakeEnvironment(terminate_after=5),
                _FakePlanner(),
                env_seed=1,
            )

    def test_requires_matching_planning_budgets(self):
        first = _FakePlanner()
        second = _FakePlanner()
        assert_matching_budgets([first, second])
        second.budget = PlanningBudget(cfm_candidates=20, refinement_rollouts=201)
        with self.assertRaisesRegex(RunnerContractError, "budgets do not match"):
            assert_matching_budgets([first, second])


class PairedSocNavGymEvaluationTest(unittest.TestCase):
    def test_uses_fresh_closed_environment_for_every_method_and_seed(self):
        environments = []
        planner_counts = {"first": 0, "second": 0}

        class EnvironmentAdapter(SocNavGymAdapter):
            def __init__(self):
                self.closed_for_test = False

            @property
            def time_step(self):
                return 0.1

            @property
            def episode_length(self):
                return 4

            @property
            def max_human_speed(self):
                return 0.8

            @property
            def physical_control_low(self):
                return np.asarray([-1.0, -2.0], dtype=np.float32)

            @property
            def physical_control_high(self):
                return np.asarray([1.0, 2.0], dtype=np.float32)

            def reset(self, *, seed):
                self.seed = seed
                self.index = 0
                return _state(float(seed)), {"seed": seed}

            def step(self, control):
                self.index += 1
                control = np.asarray(control, dtype=np.float32)
                return SocNavStep(
                    state=_state(float(self.seed + self.index)),
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    info={"SUCCESS": True},
                    requested_control=control,
                    applied_control=control,
                    applied_action=np.asarray(
                        [control[0], 0.0, control[1] / 2.0], dtype=np.float32
                    ),
                )

            def close(self):
                self.closed_for_test = True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                self.close()
                return False

        def environment_factory():
            environment = EnvironmentAdapter()
            environments.append(environment)
            return environment

        def planner_factory(name):
            def create():
                planner_counts[name] += 1
                planner = _FakePlanner()
                planner.name = name
                return planner

            return create

        result = run_paired_socnavgym_evaluation(
            environment_factory,
            {
                "first": planner_factory("first"),
                "second": planner_factory("second"),
            },
            env_seeds=[3, 5],
            planner_seed_for_env=lambda seed: seed + 100,
            execution_order_offset=1,
        )

        self.assertEqual(len(environments), 4)
        self.assertTrue(all(environment.closed_for_test for environment in environments))
        self.assertEqual(planner_counts, {"first": 2, "second": 2})
        self.assertEqual(len(result.episodes), 4)
        self.assertEqual(result.env_seeds, (3, 5))
        self.assertEqual(
            result.execution_orders,
            (("second", "first"), ("first", "second")),
        )
        self.assertEqual(
            tuple(episode.planner_name for episode in result.episodes),
            ("first", "second", "first", "second"),
        )
        self.assertTrue(
            all(episode.context.planner_seed in (103, 105) for episode in result.episodes)
        )
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
