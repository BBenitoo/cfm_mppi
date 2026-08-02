from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from cfm_mppi.evaluation.socnavgym_adapter import (
    DEFAULT_ENV_ID,
    HUMAN_GOAL_REACHED_POLICY,
    SocNavGymAdapter,
    SocNavGymAdapterError,
    _install_geometric_human_goal_policy,
    make_socnavgym_env,
    physical_to_normalized_action,
)


@dataclass
class _FakeHuman:
    id: int
    x: float
    y: float
    orientation: float
    radius: float
    vx: float
    vy: float
    gaze: bool = False


class _FakeActionSpace:
    def contains(self, action):
        action = np.asarray(action)
        return (
            action.shape == (3,)
            and action.dtype == np.float32
            and np.all(action >= np.asarray([-1.0, 0.0, -1.0]))
            and np.all(action <= np.asarray([1.0, 0.0, 1.0]))
        )


class _FakeWorldFrameEnv:
    MAX_ADVANCE_ROBOT = 1.5
    MAX_ROTATION = 2.0
    MAX_ADVANCE_HUMAN = 0.8
    TIMESTEP = 0.1
    EPISODE_LENGTH = 64
    ROBOT_RADIUS = 0.25
    GOAL_RADIUS = 0.35
    get_padded_observations = False

    def __init__(self, *, replace_human_id=False):
        self.unwrapped = self
        self.action_space = _FakeActionSpace()
        self.robot = SimpleNamespace(
            type="diff-drive",
            goal_x=4.0,
            goal_y=-2.0,
            x=1.0,
            y=2.0,
            orientation=0.25,
            vel_x=0.4,
            vel_y=-0.1,
            vel_a=0.2,
        )
        self._static = _FakeHuman(11, 2.0, 3.0, 0.5, 0.36, 0.2, 0.1, True)
        self._dynamic = _FakeHuman(3, -1.0, 0.5, -0.4, 0.31, -0.3, 0.25)
        self.static_humans = [self._static]
        self.dynamic_humans = [self._dynamic]
        self.moving_interactions = []
        self.static_interactions = []
        self.h_l_interactions = []
        self.replace_human_id = replace_human_id
        self.last_action = None
        self.reset_seed = None
        self.reset_options = None
        self.close_count = 0

    def _robot_observation(self):
        return np.asarray(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                self.robot.goal_x,
                self.robot.goal_y,
                self.robot.x,
                self.robot.y,
                np.sin(self.robot.orientation),
                np.cos(self.robot.orientation),
                self.robot.vel_x,
                self.robot.vel_y,
                self.robot.vel_a,
                self.ROBOT_RADIUS,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _human_observation(human):
        return np.asarray(
            [
                0,
                1,
                0,
                0,
                0,
                0,
                human.x,
                human.y,
                np.sin(human.orientation),
                np.cos(human.orientation),
                human.radius,
                human.vx,
                human.vy,
                float(human.gaze),
            ],
            dtype=np.float32,
        )

    def _observation(self):
        humans = self.static_humans + self.dynamic_humans
        if humans:
            human_observation = np.concatenate(
                [self._human_observation(human) for human in humans]
            )
        else:
            human_observation = np.empty(0, dtype=np.float32)
        return {
            "robot": self._robot_observation(),
            "humans": human_observation,
        }

    def reset(self, *, seed=None, options=None):
        self.reset_seed = seed
        self.reset_options = options
        self.static_humans = [self._static]
        self.dynamic_humans = [self._dynamic]
        return self._observation(), {"reset_seed": seed}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        # Return a deliberately non-integrated state.  The adapter must trust
        # this observation rather than updating its previous robot state.
        self.robot.x = 9.25
        self.robot.y = -7.5
        self.robot.orientation = -0.75
        self.robot.vel_x = float(action[0] * self.MAX_ADVANCE_ROBOT)
        self.robot.vel_y = 0.0
        self.robot.vel_a = float(action[2] * self.MAX_ROTATION)

        # Swap the wrapper traversal order while retaining the same IDs.
        self.static_humans = [self._dynamic]
        self.dynamic_humans = [self._static]
        if self.replace_human_id:
            self.static_humans[0] = _FakeHuman(
                99,
                -1.0,
                0.5,
                -0.4,
                0.31,
                -0.3,
                0.25,
            )
        return self._observation(), 1.25, False, True, {"event": "timeout"}

    def close(self):
        self.close_count += 1


class PhysicalActionTest(unittest.TestCase):
    def test_converts_and_clips_physical_diff_drive_action(self):
        action = physical_to_normalized_action(
            [3.0, -3.0],
            max_linear_speed=1.5,
            max_angular_speed=2.0,
        )

        np.testing.assert_array_equal(
            action,
            np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
        )
        self.assertEqual(action.dtype, np.float32)

    def test_rejects_invalid_physical_actions_and_limits(self):
        with self.assertRaises(ValueError):
            physical_to_normalized_action(
                [1.0],
                max_linear_speed=1.0,
                max_angular_speed=1.0,
            )
        with self.assertRaises(ValueError):
            physical_to_normalized_action(
                [np.nan, 0.0],
                max_linear_speed=1.0,
                max_angular_speed=1.0,
            )
        with self.assertRaises(ValueError):
            physical_to_normalized_action(
                [0.0, 0.0],
                max_linear_speed=0.0,
                max_angular_speed=1.0,
            )


class SocNavGymAdapterTest(unittest.TestCase):
    def test_reset_parses_world_frame_state_and_exposes_limits(self):
        env = _FakeWorldFrameEnv()
        adapter = SocNavGymAdapter(env=env)

        state, info = adapter.reset(seed=123, options={"scenario": "fixed"})

        self.assertEqual(info, {"reset_seed": 123})
        self.assertEqual(env.reset_seed, 123)
        self.assertEqual(env.reset_options, {"scenario": "fixed"})
        np.testing.assert_allclose(state.robot_state, [1.0, 2.0, 0.25])
        np.testing.assert_array_equal(state.goal, [4.0, -2.0])
        np.testing.assert_allclose(state.robot_body_velocity, [0.4, -0.1, 0.2])
        expected_world_velocity = [
            0.4 * np.cos(0.25) + 0.1 * np.sin(0.25),
            0.4 * np.sin(0.25) - 0.1 * np.cos(0.25),
            0.2,
        ]
        np.testing.assert_allclose(state.robot_velocity, expected_world_velocity)
        self.assertEqual(state.robot_radius, 0.25)
        self.assertEqual(state.goal_radius, 0.35)
        self.assertEqual(state.human_ids, (3, 11))
        np.testing.assert_array_equal(
            state.human_positions,
            np.asarray([[-1.0, 0.5], [2.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            state.human_velocities,
            np.asarray([[-0.3, 0.25], [0.2, 0.1]], dtype=np.float32),
        )
        np.testing.assert_allclose(state.human_radii, [0.31, 0.36])
        self.assertAlmostEqual(adapter.time_step, 0.1)
        self.assertAlmostEqual(adapter.max_human_speed, 0.8)
        self.assertEqual(adapter.episode_length, 64)
        np.testing.assert_array_equal(
            adapter.physical_control_low,
            [-1.5, -2.0],
        )
        np.testing.assert_array_equal(
            adapter.physical_control_high,
            [1.5, 2.0],
        )
        self.assertFalse(adapter.physical_control_low.flags.writeable)
        self.assertFalse(state.robot_state.flags.writeable)
        self.assertFalse(state.robot_body_velocity.flags.writeable)
        self.assertFalse(state.humans[0].position.flags.writeable)

    def test_step_uses_returned_truth_and_reports_clipped_control(self):
        env = _FakeWorldFrameEnv()
        adapter = SocNavGymAdapter(env=env)
        adapter.reset(seed=7)

        result = adapter.step([3.0, -3.0])

        np.testing.assert_array_equal(
            env.last_action,
            np.asarray([1.0, 0.0, -1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(result.requested_control, [3.0, -3.0])
        np.testing.assert_array_equal(result.applied_control, [1.5, -2.0])
        np.testing.assert_array_equal(result.applied_action, env.last_action)
        np.testing.assert_array_equal(result.normalized_action, env.last_action)
        np.testing.assert_allclose(
            result.state.robot_state,
            [9.25, -7.5, -0.75],
        )
        self.assertEqual(result.state.human_ids, (3, 11))
        self.assertEqual(result.reward, 1.25)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertTrue(result.done)
        self.assertEqual(result.info, {"event": "timeout"})
        self.assertFalse(result.requested_control.flags.writeable)
        self.assertFalse(result.applied_control.flags.writeable)
        self.assertFalse(result.applied_action.flags.writeable)
        self.assertIs(adapter.state, result.state)
        with self.assertRaises(RuntimeError):
            adapter.step([0.0, 0.0])

    def test_detects_human_identity_change_within_episode(self):
        adapter = SocNavGymAdapter(env=_FakeWorldFrameEnv(replace_human_id=True))
        adapter.reset(seed=5)

        with self.assertRaisesRegex(
            SocNavGymAdapterError,
            "human IDs changed within the episode",
        ):
            adapter.step([0.0, 0.0])

    def test_requires_reset_and_supports_idempotent_context_close(self):
        env = _FakeWorldFrameEnv()
        with SocNavGymAdapter(env=env) as adapter:
            with self.assertRaises(RuntimeError):
                _ = adapter.state
            with self.assertRaises(RuntimeError):
                adapter.step([0.0, 0.0])

        self.assertEqual(env.close_count, 1)
        adapter.close()
        self.assertEqual(env.close_count, 1)
        with self.assertRaises(RuntimeError):
            adapter.reset(seed=1)


class SocNavGymFactoryTest(unittest.TestCase):
    def test_geometric_goal_policy_removes_wall_clock_dependency(self):
        class Human:
            def __init__(self, *, human_type="dynamic"):
                self.width = 0.72
                self.type = human_type
                self.x = 0.0
                self.y = 0.0
                self.goal_x = 2.0
                self.goal_y = 0.0
                self.goal_radius = 0.25
                self.initial_time = -1e12

            def has_reached_goal(self, offset=None):
                del offset
                return True

        module = SimpleNamespace(Human=Human)
        _install_geometric_human_goal_policy(module)
        installed = Human.has_reached_goal
        _install_geometric_human_goal_policy(module)

        self.assertIs(Human.has_reached_goal, installed)
        self.assertEqual(
            installed.__cfm_mppi_goal_policy__, HUMAN_GOAL_REACHED_POLICY
        )
        self.assertFalse(Human().has_reached_goal())
        near = Human()
        near.x = 1.5
        self.assertTrue(near.has_reached_goal())
        self.assertFalse(Human(human_type="static").has_reached_goal(offset=100.0))

    def test_factory_imports_optional_dependencies_only_when_called(self):
        base_env = SimpleNamespace(close=mock.Mock())
        wrapped_env = object()
        gymnasium = SimpleNamespace(make=mock.Mock(return_value=base_env))
        world_wrapper = mock.Mock(return_value=wrapped_env)
        wrappers = SimpleNamespace(WorldFrameObservations=world_wrapper)
        human_module = SimpleNamespace(
            Human=type(
                "Human",
                (),
                {"has_reached_goal": lambda self, offset=None: False},
            )
        )

        def import_optional(name):
            return {
                "gymnasium": gymnasium,
                "socnavgym": SimpleNamespace(),
                "socnavgym.envs.utils.human": human_module,
                "socnavgym.wrappers": wrappers,
            }[name]

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "scenario.yaml"
            config.write_text("env: {}\n", encoding="utf-8")
            with mock.patch(
                "cfm_mppi.evaluation.socnavgym_adapter.importlib.import_module",
                side_effect=import_optional,
            ) as importer:
                result = make_socnavgym_env(config)

        self.assertIs(result, wrapped_env)
        self.assertEqual(
            [call.args[0] for call in importer.call_args_list],
            [
                "gymnasium",
                "socnavgym",
                "socnavgym.envs.utils.human",
                "socnavgym.wrappers",
            ],
        )
        gymnasium.make.assert_called_once_with(
            DEFAULT_ENV_ID,
            config=str(config.resolve()),
        )
        world_wrapper.assert_called_once_with(base_env)
        base_env.close.assert_not_called()

    def test_factory_closes_base_environment_when_wrapping_fails(self):
        base_env = SimpleNamespace(close=mock.Mock())
        gymnasium = SimpleNamespace(make=mock.Mock(return_value=base_env))
        wrappers = SimpleNamespace(
            WorldFrameObservations=mock.Mock(side_effect=RuntimeError("broken"))
        )
        human_module = SimpleNamespace(
            Human=type(
                "Human",
                (),
                {"has_reached_goal": lambda self, offset=None: False},
            )
        )

        def import_optional(name):
            return {
                "gymnasium": gymnasium,
                "socnavgym": SimpleNamespace(),
                "socnavgym.envs.utils.human": human_module,
                "socnavgym.wrappers": wrappers,
            }[name]

        with tempfile.TemporaryDirectory() as temporary_directory:
            config = Path(temporary_directory) / "scenario.yaml"
            config.write_text("env: {}\n", encoding="utf-8")
            with mock.patch(
                "cfm_mppi.evaluation.socnavgym_adapter.importlib.import_module",
                side_effect=import_optional,
            ), self.assertRaisesRegex(RuntimeError, "broken"):
                make_socnavgym_env(config)

        base_env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
