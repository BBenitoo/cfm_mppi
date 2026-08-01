import unittest

import torch

from cfm_mppi.evaluation.eval_utils import (
    EpisodeMetrics,
    FreezingMetrics,
    compute_episode_metrics,
    compute_freezing_metrics,
    summarize_freezing_metrics,
    summarize_metrics,
)


class EpisodeMetricsTest(unittest.TestCase):
    def test_no_collision_and_final_goal_distance(self) -> None:
        states = torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [0.0, 1.0, 4.0],
                [100.0, 100.0, 100.0],
            ],
            dtype=torch.float64,
        )
        obstacle_positions = torch.tensor(
            [
                [
                    [10.0, 10.0, 10.0],
                    [10.0, 10.0, 10.0],
                    [-100.0, -100.0, -100.0],
                ]
            ],
            dtype=torch.float64,
        )
        goal = torch.tensor([0.0, 0.0, -200.0], dtype=torch.float64)

        metrics = compute_episode_metrics(
            states,
            obstacle_positions,
            goal,
            collision_radius=0.5,
        )

        self.assertIsInstance(metrics, EpisodeMetrics)
        self.assertFalse(metrics.collision.item())
        self.assertEqual(metrics.collision.shape, torch.Size([]))
        torch.testing.assert_close(
            metrics.final_goal_distance,
            torch.tensor(5.0, dtype=torch.float64),
        )

    def test_collision_with_any_obstacle_at_any_time(self) -> None:
        states = torch.tensor(
            [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        obstacle_positions = torch.tensor(
            [
                [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
                [[4.0, 1.1, 4.0], [4.0, 0.0, 4.0]],
            ],
            dtype=torch.float64,
        )

        metrics = compute_episode_metrics(
            states,
            obstacle_positions,
            goal=torch.tensor([2.0, 0.0], dtype=torch.float64),
            collision_radius=0.2,
        )

        self.assertTrue(metrics.collision.item())

    def test_collision_threshold_is_strict(self) -> None:
        states = torch.tensor([[0.0], [0.0]], dtype=torch.float64)
        obstacle_positions = torch.tensor(
            [[[3.0], [4.0]]],
            dtype=torch.float64,
        )
        goal = torch.zeros(2, dtype=torch.float64)

        at_threshold = compute_episode_metrics(
            states,
            obstacle_positions,
            goal,
            collision_radius=5.0,
        )
        inside_threshold = compute_episode_metrics(
            states,
            obstacle_positions,
            goal,
            collision_radius=5.0001,
        )

        self.assertFalse(at_threshold.collision.item())
        self.assertTrue(inside_threshold.collision.item())

    def test_empty_obstacle_set_has_no_collision(self) -> None:
        states = torch.tensor(
            [[0.0, 2.0], [0.0, 0.0]],
            dtype=torch.float32,
        )
        obstacle_positions = torch.empty((0, 2, 2), dtype=torch.float32)

        metrics = compute_episode_metrics(
            states,
            obstacle_positions,
            goal=torch.tensor([3.0, 0.0]),
            collision_radius=0.5,
        )

        self.assertFalse(metrics.collision.item())
        torch.testing.assert_close(
            metrics.final_goal_distance,
            torch.tensor(1.0),
        )

    def test_invalid_shapes_and_negative_radius_are_rejected(self) -> None:
        valid_states = torch.zeros((2, 2))
        valid_obstacles = torch.zeros((1, 2, 2))
        valid_goal = torch.zeros(2)
        invalid_inputs = (
            (
                "states rank",
                torch.zeros(2),
                valid_obstacles,
                valid_goal,
                0.5,
            ),
            (
                "states spatial dimension",
                torch.zeros((1, 2)),
                valid_obstacles,
                valid_goal,
                0.5,
            ),
            (
                "empty state trajectory",
                torch.empty((2, 0)),
                torch.empty((1, 2, 0)),
                valid_goal,
                0.5,
            ),
            (
                "obstacle rank",
                valid_states,
                torch.zeros((2, 2)),
                valid_goal,
                0.5,
            ),
            (
                "obstacle spatial dimension",
                valid_states,
                torch.zeros((1, 1, 2)),
                valid_goal,
                0.5,
            ),
            (
                "time dimension mismatch",
                valid_states,
                torch.zeros((1, 2, 3)),
                valid_goal,
                0.5,
            ),
            (
                "goal rank",
                valid_states,
                valid_obstacles,
                torch.zeros((1, 2)),
                0.5,
            ),
            (
                "goal spatial dimension",
                valid_states,
                valid_obstacles,
                torch.zeros(1),
                0.5,
            ),
            (
                "negative radius",
                valid_states,
                valid_obstacles,
                valid_goal,
                -0.1,
            ),
        )

        for name, states, obstacles, goal, radius in invalid_inputs:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    compute_episode_metrics(states, obstacles, goal, radius)


class MetricsSummaryTest(unittest.TestCase):
    def test_summary_uses_percentage_and_population_variance(self) -> None:
        episodes = [
            EpisodeMetrics(
                collision=torch.tensor(False),
                final_goal_distance=torch.tensor(1.0, dtype=torch.float64),
            ),
            EpisodeMetrics(
                collision=torch.tensor(True),
                final_goal_distance=torch.tensor(3.0, dtype=torch.float64),
            ),
        ]

        summary = summarize_metrics(episodes)

        torch.testing.assert_close(
            summary.collision_rate_percent,
            torch.tensor(50.0),
        )
        torch.testing.assert_close(
            summary.mean_final_goal_distance,
            torch.tensor(2.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            summary.variance_final_goal_distance,
            torch.tensor(1.0, dtype=torch.float64),
        )

    def test_single_episode_variance_is_zero_and_finite(self) -> None:
        summary = summarize_metrics(
            [
                EpisodeMetrics(
                    collision=torch.tensor(True),
                    final_goal_distance=torch.tensor(2.5),
                )
            ]
        )

        torch.testing.assert_close(
            summary.collision_rate_percent,
            torch.tensor(100.0),
        )
        torch.testing.assert_close(
            summary.mean_final_goal_distance,
            torch.tensor(2.5),
        )
        torch.testing.assert_close(
            summary.variance_final_goal_distance,
            torch.tensor(0.0),
        )
        self.assertTrue(torch.isfinite(summary.variance_final_goal_distance))

    def test_empty_episode_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_metrics([])


class FreezingMetricsTest(unittest.TestCase):
    def test_counts_each_maximal_qualifying_interval_once(self) -> None:
        states = torch.zeros((3, 25), dtype=torch.float64)
        linear_speeds = torch.tensor(
            [0.0] * 10 + [-0.2] * 2 + [-0.01] * 11 + [0.2] * 2,
            dtype=torch.float64,
        )

        metrics = compute_freezing_metrics(
            states=states,
            linear_speeds=linear_speeds,
            goal=torch.tensor([10.0, 0.0], dtype=torch.float64),
            dt=0.1,
        )

        self.assertIsInstance(metrics, FreezingMetrics)
        self.assertEqual(metrics.event_count.item(), 2)
        self.assertTrue(metrics.occurred.item())

    def test_requires_full_duration_and_strict_speed_threshold(self) -> None:
        states = torch.zeros((2, 10))
        goal = torch.tensor([10.0, 0.0])

        too_short = compute_freezing_metrics(
            states=states,
            linear_speeds=torch.tensor([0.0] * 9 + [0.2]),
            goal=goal,
            dt=0.1,
        )
        at_speed_threshold = compute_freezing_metrics(
            states=states,
            linear_speeds=torch.full((10,), 0.05),
            goal=goal,
            dt=0.1,
        )

        self.assertEqual(too_short.event_count.item(), 0)
        self.assertEqual(at_speed_threshold.event_count.item(), 0)

    def test_excludes_stopping_at_or_inside_goal_threshold(self) -> None:
        states = torch.zeros((2, 10))
        states[0] = 0.5

        at_goal_threshold = compute_freezing_metrics(
            states=states,
            linear_speeds=torch.zeros(10),
            goal=torch.zeros(2),
            dt=0.1,
            goal_distance_threshold=0.5,
        )

        self.assertEqual(at_goal_threshold.event_count.item(), 0)

    def test_summary_is_episode_rate_and_preserves_total_events(self) -> None:
        summary = summarize_freezing_metrics(
            [
                FreezingMetrics(event_count=torch.tensor(0)),
                FreezingMetrics(event_count=torch.tensor(2)),
                FreezingMetrics(event_count=torch.tensor(1)),
            ]
        )

        torch.testing.assert_close(
            summary.freezing_rate_percent,
            torch.tensor(200.0 / 3.0),
        )
        self.assertEqual(summary.total_event_count.item(), 3)

    def test_invalid_inputs_are_rejected(self) -> None:
        states = torch.zeros((2, 10))
        speeds = torch.zeros(10)
        goal = torch.zeros(2)

        invalid_calls = (
            lambda: compute_freezing_metrics(states, speeds[:-1], goal, dt=0.1),
            lambda: compute_freezing_metrics(states, speeds, goal, dt=0.0),
            lambda: compute_freezing_metrics(
                states,
                speeds,
                goal,
                dt=0.1,
                minimum_duration=0.0,
            ),
            lambda: compute_freezing_metrics(
                states,
                speeds,
                goal,
                dt=0.1,
                speed_threshold=-0.1,
            ),
            lambda: compute_freezing_metrics(
                states,
                speeds,
                goal,
                dt=0.1,
                goal_distance_threshold=-0.1,
            ),
            lambda: summarize_freezing_metrics([]),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
