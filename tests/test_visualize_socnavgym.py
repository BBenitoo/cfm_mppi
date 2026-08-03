import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from cfm_mppi.evaluation.socnavgym_planners import (
    VISUALIZATION_TRACE_SCHEMA,
)
from cfm_mppi.evaluation.visualize_socnavgym import (
    BASELINE_PLANNER,
    VRC_PLANNER,
    VisualizationTraceError,
    choose_key_step,
    load_paired_episodes,
    render_animation_comparison,
    render_static_comparison,
)


MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None


def _state(robot_x: float, human_y: float) -> dict[str, object]:
    return {
        "robot_state": [robot_x, 0.0, 0.0],
        "robot_radius": 0.25,
        "goal": [3.0, 0.0],
        "goal_radius": 0.3,
        "humans": [
            {
                "id": 7,
                "position": [1.0, human_y],
                "velocity": [0.0, 0.2],
                "radius": 0.3,
            }
        ],
    }


def _visualization_trace(
    step_index: int,
    *,
    is_vrc: bool,
) -> dict[str, object]:
    robot_x = 0.2 * step_index
    robot_prediction = [
        [robot_x, 0.0, 0.0],
        [robot_x + 0.4, 0.0, 0.0],
        [robot_x + 0.8, 0.0, 0.0],
    ]
    if is_vrc and step_index == 1:
        robot_prediction = [
            [robot_x, 0.0, 0.0],
            [robot_x + 0.35, -0.2, -0.2],
            [robot_x + 0.7, -0.35, -0.3],
        ]

    no_vrc = [
        [1.0, 0.2 + 0.2 * step_index],
        [1.0, 0.4 + 0.2 * step_index],
        [1.0, 0.6 + 0.2 * step_index],
    ]
    if step_index == 0:
        with_vrc = no_vrc
        force = [0.0, 0.0]
    else:
        with_vrc = [
            [1.25, 0.2 + 0.2 * step_index],
            [1.5, 0.4 + 0.2 * step_index],
            [1.75, 0.6 + 0.2 * step_index],
        ]
        force = [1.5, 0.5]

    return {
        "schema_version": VISUALIZATION_TRACE_SCHEMA,
        "human_ids": [7],
        "robot_candidate_trajectories": [
            robot_prediction,
            [
                [robot_x, 0.0, 0.0],
                [robot_x + 0.3, 0.15, 0.2],
                [robot_x + 0.6, 0.25, 0.3],
            ],
        ],
        "robot_conditioning_trajectory": [
            [robot_x, 0.0, 0.0],
            [robot_x + 0.35, -0.1, -0.1],
            [robot_x + 0.7, -0.2, -0.2],
        ],
        "robot_prediction": robot_prediction,
        "pedestrian_prediction_no_vrc": [no_vrc],
        "pedestrian_prediction_vrc": [with_vrc] if is_vrc else None,
        "vrc_tube": (
            {
                "centers": [
                    [robot_x, 0.0],
                    [robot_x + 0.35, -0.1],
                    [robot_x + 0.7, -0.2],
                ],
                "longitudinal_radii": [0.45, 0.4, 0.35],
                "lateral_radii": [0.3, 0.27, 0.24],
                "theta": [0.0, -0.1, -0.2],
                "influence_cutoff": 2.0,
            }
            if is_vrc
            else None
        ),
        "vrc_forces": [force] if is_vrc else None,
    }


def _episode(planner: str) -> dict[str, object]:
    is_vrc = planner == VRC_PLANNER
    initial = _state(0.0, 0.0)
    middle = _state(0.2, 0.2)
    final = _state(0.4, 0.4)
    return {
        "planner": planner,
        "context": {"env_seed": 17, "time_step": 0.2},
        "initial_state": initial,
        "steps": [
            {
                "state": initial,
                "next_state": middle,
                "diagnostics": {
                    "visualization": _visualization_trace(0, is_vrc=is_vrc)
                },
            },
            {
                "state": middle,
                "next_state": final,
                "diagnostics": {
                    "visualization": _visualization_trace(1, is_vrc=is_vrc)
                },
            },
        ],
        "summary": {"success": True},
    }


def _paired_document() -> dict[str, object]:
    return {
        "episodes": [
            _episode(BASELINE_PLANNER),
            _episode(VRC_PLANNER),
        ]
    }


def _write_document(directory: str, document: dict[str, object]) -> Path:
    path = Path(directory) / "paired.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class SocNavGymVisualizationTest(unittest.TestCase):
    def test_loads_matched_two_step_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = _write_document(directory, _paired_document())
            pair = load_paired_episodes(source)

        self.assertEqual(pair.source, source.resolve())
        self.assertEqual(pair.env_seed, 17)
        self.assertEqual(pair.time_step, 0.2)
        self.assertEqual(pair.baseline["planner"], BASELINE_PLANNER)
        self.assertEqual(pair.vrc["planner"], VRC_PLANNER)

    def test_choose_key_step_prefers_largest_vrc_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pair = load_paired_episodes(
                _write_document(directory, _paired_document())
            )

        self.assertEqual(choose_key_step(pair), 1)

    def test_missing_trace_explains_how_to_record_it(self) -> None:
        document = _paired_document()
        baseline = document["episodes"][0]
        baseline["steps"][0]["diagnostics"] = {}

        with tempfile.TemporaryDirectory() as directory:
            source = _write_document(directory, document)
            with self.assertRaises(VisualizationTraceError) as raised:
                load_paired_episodes(source)

        message = str(raised.exception)
        self.assertIn("no planner visualization trace", message)
        self.assertIn("--record-visualization", message)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is not installed")
    def test_renders_nonempty_static_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pair = load_paired_episodes(
                _write_document(directory, _paired_document())
            )
            output = Path(directory) / "comparison.png"

            rendered = render_static_comparison(
                pair,
                output,
                step_index=1,
                history_steps=0,
                tube_stride=1,
                dpi=40,
            )

            self.assertEqual(rendered, output.resolve())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    @unittest.skipUnless(
        MATPLOTLIB_AVAILABLE and PILLOW_AVAILABLE,
        "matplotlib and Pillow are required for GIF rendering",
    )
    def test_renders_nonempty_two_frame_gif(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            pair = load_paired_episodes(
                _write_document(directory, _paired_document())
            )
            output = Path(directory) / "comparison.gif"

            rendered = render_animation_comparison(
                pair,
                output,
                history_steps=0,
                frame_stride=1,
                tube_stride=1,
                show_candidates=False,
                fps=2.0,
                dpi=30,
            )

            self.assertEqual(rendered, output.resolve())
            self.assertGreater(output.stat().st_size, 0)
            with Image.open(output) as image:
                self.assertEqual(image.format, "GIF")
                self.assertEqual(image.n_frames, 2)


if __name__ == "__main__":
    unittest.main()
