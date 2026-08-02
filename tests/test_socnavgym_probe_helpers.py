from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from cfm_mppi.diagnostics.socnavgym_contract import (
    ContractViolation,
    DEFAULT_MANIFEST,
    EXIT_INTERNAL,
    _zero_action,
    extract_direct_url_commit,
    load_candidate,
    main,
    validate_world_frame_observation,
)


class SocNavGymProbeHelperTest(unittest.TestCase):
    def test_zero_action_accepts_diff_drive_lateral_axis_contract(self) -> None:
        action_space = SimpleNamespace(
            shape=(3,),
            dtype=np.dtype(np.float32),
            low=np.asarray([-1.0, 0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            contains=lambda action: (
                action.dtype == np.float32
                and action.shape == (3,)
                and np.all(action >= np.asarray([-1.0, 0.0, -1.0]))
                and np.all(action <= np.asarray([1.0, 0.0, 1.0]))
            ),
        )
        env = SimpleNamespace(
            action_space=action_space,
            unwrapped=SimpleNamespace(robot=SimpleNamespace(type="diff-drive")),
        )

        action = _zero_action(env)

        np.testing.assert_array_equal(action, np.zeros(3, dtype=np.float32))

    def test_selected_candidate_is_pinned_v1_world_contract(self) -> None:
        candidate = load_candidate(DEFAULT_MANIFEST)

        self.assertEqual(candidate.name, "v1-1ef13ee")
        self.assertEqual(
            candidate.commit,
            "1ef13ee604b71730e9ec7f2d9fd9cb2e8b796549",
        )
        self.assertEqual(candidate.env_id, "SocNavGym-v1")
        self.assertEqual(candidate.python_minor, "3.11")
        self.assertEqual(candidate.gymnasium_version, "0.29.1")
        self.assertEqual(candidate.numpy_version, "1.26.4")
        self.assertEqual(candidate.time_step, 0.1)
        self.assertEqual(candidate.human_count, 2)
        self.assertEqual(candidate.robot_type, "diff-drive")
        self.assertEqual(candidate.human_policy, "orca")
        self.assertEqual(candidate.set_shape, "no-walls")
        self.assertFalse(candidate.padded_observations)
        self.assertEqual(candidate.robot_world_dim, 16)
        self.assertEqual(candidate.human_world_stride, 14)
        self.assertTrue(candidate.config_path.is_file())

    def test_current_main_candidate_is_explicitly_rejected(self) -> None:
        candidate = load_candidate(DEFAULT_MANIFEST, "v2-95fbc9c")

        self.assertEqual(candidate.status, "rejected")
        self.assertIn("socnavenv_v1", candidate.reason)

    def test_extracts_pep610_commit_and_rejects_invalid_metadata(self) -> None:
        direct_url = json.dumps(
            {
                "url": "https://github.com/gnns4hri/SocNavGym.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "abc123",
                },
            }
        )

        self.assertEqual(extract_direct_url_commit(direct_url), "abc123")
        self.assertIsNone(extract_direct_url_commit(None))
        self.assertIsNone(extract_direct_url_commit("not json"))
        self.assertIsNone(extract_direct_url_commit("[]"))
        self.assertIsNone(extract_direct_url_commit("null"))

    def test_world_observation_matches_unwrapped_state(self) -> None:
        robot = SimpleNamespace(
            goal_x=4.0,
            goal_y=-3.0,
            x=1.25,
            y=-0.5,
            orientation=0.4,
            vel_x=0.6,
            vel_y=0.0,
            vel_a=-0.2,
        )
        human = SimpleNamespace(
            id=7,
            x=-1.0,
            y=2.0,
            orientation=-0.3,
            width=0.72,
            speed=0.5,
        )
        base_env = SimpleNamespace(
            robot=robot,
            ROBOT_RADIUS=0.25,
            get_padded_observations=False,
            static_humans=[],
            dynamic_humans=[human],
            moving_interactions=[],
            static_interactions=[],
            h_l_interactions=[],
            HUMAN_GAZE_ANGLE=np.pi,
        )
        robot_obs = np.asarray(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                robot.goal_x,
                robot.goal_y,
                robot.x,
                robot.y,
                np.sin(robot.orientation),
                np.cos(robot.orientation),
                robot.vel_x,
                robot.vel_y,
                robot.vel_a,
                base_env.ROBOT_RADIUS,
            ],
            dtype=np.float32,
        )
        human_obs = np.asarray(
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
                human.width / 2,
                human.speed * np.cos(human.orientation),
                human.speed * np.sin(human.orientation),
                1,
            ],
            dtype=np.float32,
        )

        details = validate_world_frame_observation(
            {"robot": robot_obs, "humans": human_obs},
            base_env,
        )

        self.assertEqual(details["human_count"], 1)
        self.assertEqual(details["human_ids"], [7])
        self.assertFalse(details["human_ids_exposed_by_wrapper"])

    def test_world_observation_rejects_wrong_human_stride(self) -> None:
        base_env = SimpleNamespace(
            robot=SimpleNamespace(
                goal_x=0.0,
                goal_y=0.0,
                x=0.0,
                y=0.0,
                orientation=0.0,
                vel_x=0.0,
                vel_y=0.0,
                vel_a=0.0,
            ),
            ROBOT_RADIUS=0.25,
            get_padded_observations=False,
            static_humans=[],
            dynamic_humans=[],
            moving_interactions=[],
            static_interactions=[],
            h_l_interactions=[],
            HUMAN_GAZE_ANGLE=np.pi,
        )
        observation = {
            "robot": np.zeros(16, dtype=np.float32),
            "humans": np.zeros(13, dtype=np.float32),
        }

        with self.assertRaises(ContractViolation):
            validate_world_frame_observation(observation, base_env)

    def test_output_write_failure_has_stable_internal_exit_and_json(self) -> None:
        report = {
            "schema_version": 1,
            "candidate": "v1-1ef13ee",
            "checks": [],
            "success": True,
            "exit_code": 0,
        }
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent_is_file = Path(temporary_directory) / "not-a-directory"
            parent_is_file.write_text("occupied", encoding="utf-8")
            output_path = parent_is_file / "report.json"
            with mock.patch(
                "cfm_mppi.diagnostics.socnavgym_contract.run_probe",
                return_value=(report, 0),
            ), redirect_stdout(stdout):
                exit_code = main(["--output", str(output_path)])

        printed_report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, EXIT_INTERNAL)
        self.assertFalse(printed_report["success"])
        self.assertEqual(printed_report["exit_code"], EXIT_INTERNAL)
        self.assertEqual(printed_report["checks"][-1]["name"], "output_report")
        self.assertEqual(printed_report["checks"][-1]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
