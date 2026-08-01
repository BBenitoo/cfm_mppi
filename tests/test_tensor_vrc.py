import unittest

import numpy as np
import torch

from cfm_mppi.vrc.build_vrc import (
    RobotForceParameters,
    VRCParameters,
    build_tensor_vrc_tube,
    build_vrc_tube,
    force_from_tensor_vrc_tube,
    force_from_vrc_tube,
    predict_pedestrians_with_tensor_vrc,
)


class TensorVRCTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vrc_params = VRCParameters()
        self.force_params = RobotForceParameters()
        self.states = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.1],
                    [0.1, 0.0, 0.2],
                    [0.2, 0.1, 0.3],
                    [0.3, 0.1, 0.4],
                    [0.4, 0.2, 0.5],
                ],
                [
                    [1.0, -0.5, -0.2],
                    [0.9, -0.4, -0.1],
                    [0.8, -0.3, 0.0],
                    [0.7, -0.2, 0.1],
                    [0.6, -0.1, 0.2],
                ],
            ],
            dtype=torch.float64,
        )
        self.controls = torch.tensor(
            [
                [
                    [0.8, 0.2],
                    [1.0, -0.1],
                    [0.6, 0.3],
                    [0.4, -0.2],
                ],
                [
                    [-0.5, 0.1],
                    [-0.7, -0.2],
                    [-0.4, 0.2],
                    [-0.2, -0.1],
                ],
            ],
            dtype=torch.float64,
        )

    def test_tensor_tube_matches_scalar_tubes(self) -> None:
        tensor_tube = build_tensor_vrc_tube(
            self.states,
            self.controls,
            self.vrc_params,
        )

        for branch_index in range(self.states.shape[0]):
            scalar_tube = build_vrc_tube(
                self.states[branch_index],
                self.controls[branch_index],
                self.vrc_params,
            )
            expected_centers = torch.from_numpy(
                np.stack([ellipse.center for ellipse in scalar_tube])
            )
            expected_longitudinal = torch.tensor(
                [ellipse.longitudinal_radius for ellipse in scalar_tube],
                dtype=torch.float64,
            )
            expected_lateral = torch.tensor(
                [ellipse.lateral_radius for ellipse in scalar_tube],
                dtype=torch.float64,
            )

            torch.testing.assert_close(
                tensor_tube.centers[branch_index],
                expected_centers,
                rtol=1e-12,
                atol=1e-12,
            )
            torch.testing.assert_close(
                tensor_tube.longitudinal_radii[branch_index],
                expected_longitudinal,
                rtol=1e-12,
                atol=1e-12,
            )
            torch.testing.assert_close(
                tensor_tube.lateral_radii[branch_index],
                expected_lateral,
                rtol=1e-12,
                atol=1e-12,
            )

        # The terminal state must reuse the final control.
        self.assertEqual(tensor_tube.length, self.controls.shape[1] + 1)

    def test_batched_force_matches_scalar_force_including_tail_preview(self) -> None:
        tensor_tube = build_tensor_vrc_tube(
            self.states,
            self.controls,
            self.vrc_params,
        )
        positions = torch.tensor(
            [
                [[0.3, 0.7], [2.0, 1.0], [-1.0, -0.5]],
                [[0.5, 0.4], [1.5, -1.0], [-0.5, 0.2]],
            ],
            dtype=torch.float64,
        )
        scalar_tubes = [
            build_vrc_tube(
                self.states[branch_index],
                self.controls[branch_index],
                self.vrc_params,
            )
            for branch_index in range(self.states.shape[0])
        ]

        for current_index in (0, self.controls.shape[1] - 1):
            actual = force_from_tensor_vrc_tube(
                pedestrian_positions=positions,
                vrc_tube=tensor_tube,
                current_index=current_index,
                force_params=self.force_params,
                preview_steps=3,
                discount=0.5,
            )
            expected = np.stack(
                [
                    np.stack(
                        [
                            force_from_vrc_tube(
                                pedestrian_position=point.numpy(),
                                vrc_tube=scalar_tubes[branch_index],
                                current_index=current_index,
                                force_params=self.force_params,
                                preview_steps=3,
                                discount=0.5,
                            )
                            for point in positions[branch_index]
                        ]
                    )
                    for branch_index in range(self.states.shape[0])
                ]
            )
            torch.testing.assert_close(
                actual,
                torch.from_numpy(expected),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_batched_prediction_matches_scalar_recurrence(self) -> None:
        positions = torch.tensor(
            [[[0.4, 0.6], [1.4, -0.2], [-0.3, 0.1]]],
            dtype=torch.float64,
        )
        velocities = torch.tensor(
            [[[0.3, -0.1], [-0.2, 0.4], [0.1, 0.2]]],
            dtype=torch.float64,
        )
        horizon = self.controls.shape[1]
        dt = 0.1
        relaxation_time = 0.5
        maximum_speed = 2.0
        preview_steps = 3
        preview_discount = 0.7

        tensor_tube = build_tensor_vrc_tube(
            self.states,
            self.controls,
            self.vrc_params,
        )
        actual = predict_pedestrians_with_tensor_vrc(
            current_positions=positions,
            current_velocities=velocities,
            vrc_tube=tensor_tube,
            horizon=horizon,
            dt=dt,
            force_params=self.force_params,
            relaxation_time=relaxation_time,
            maximum_speed=maximum_speed,
            preview_steps=preview_steps,
            preview_discount=preview_discount,
        )

        expected_branches = []
        for branch_index in range(self.states.shape[0]):
            scalar_tube = build_vrc_tube(
                self.states[branch_index],
                self.controls[branch_index],
                self.vrc_params,
            )
            branch_positions = positions.squeeze(0).numpy().copy()
            branch_velocities = velocities.squeeze(0).numpy().copy()
            baseline_velocities = branch_velocities.copy()
            prediction = np.empty(
                (branch_positions.shape[0], horizon, 2),
                dtype=np.float64,
            )

            for step in range(horizon):
                for pedestrian_index in range(branch_positions.shape[0]):
                    vrc_force = force_from_vrc_tube(
                        pedestrian_position=branch_positions[pedestrian_index],
                        vrc_tube=scalar_tube,
                        current_index=step,
                        force_params=self.force_params,
                        preview_steps=preview_steps,
                        discount=preview_discount,
                    )
                    relaxation_force = (
                        baseline_velocities[pedestrian_index]
                        - branch_velocities[pedestrian_index]
                    ) / relaxation_time
                    branch_velocities[pedestrian_index] += (
                        relaxation_force + vrc_force
                    ) * dt

                    speed = np.linalg.norm(branch_velocities[pedestrian_index])
                    if speed > maximum_speed:
                        branch_velocities[pedestrian_index] *= (
                            maximum_speed / speed
                        )
                    branch_positions[pedestrian_index] += (
                        branch_velocities[pedestrian_index] * dt
                    )
                prediction[:, step, :] = branch_positions
            expected_branches.append(prediction)

        torch.testing.assert_close(
            actual,
            torch.from_numpy(np.stack(expected_branches)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_center_fallback_and_cutoff_boundary(self) -> None:
        states = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        controls = torch.zeros((1, 1, 2), dtype=torch.float64)
        tube = build_tensor_vrc_tube(states, controls, self.vrc_params)

        center = tube.centers[:, :1, :]
        center_force = force_from_tensor_vrc_tube(
            pedestrian_positions=center,
            vrc_tube=tube,
            current_index=0,
            force_params=self.force_params,
            preview_steps=1,
        )
        expected_direction = torch.tensor(
            [[[0.0, 1.0]]],
            dtype=torch.float64,
        )
        expected_magnitude = min(
            self.force_params.strength
            * np.exp(1.0 / self.force_params.decay),
            self.force_params.maximum_force,
        )
        torch.testing.assert_close(
            center_force,
            expected_magnitude * expected_direction,
            rtol=1e-12,
            atol=1e-12,
        )

        cutoff_position = center.clone()
        cutoff_position[..., 0] += (
            self.force_params.influence_cutoff
            * tube.longitudinal_radii[:, :1]
            * tube.headings[:, :1, 0]
        )
        cutoff_position[..., 1] += (
            self.force_params.influence_cutoff
            * tube.longitudinal_radii[:, :1]
            * tube.headings[:, :1, 1]
        )
        cutoff_force = force_from_tensor_vrc_tube(
            pedestrian_positions=cutoff_position,
            vrc_tube=tube,
            current_index=0,
            force_params=self.force_params,
            preview_steps=1,
        )
        torch.testing.assert_close(
            cutoff_force,
            torch.zeros_like(cutoff_force),
            rtol=0.0,
            atol=1e-12,
        )

    def test_inactive_preview_force_still_contributes_weight(self) -> None:
        states = torch.tensor(
            [[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        controls = torch.zeros((1, 1, 2), dtype=torch.float64)
        tube = build_tensor_vrc_tube(states, controls, self.vrc_params)
        position = torch.tensor([[[0.5, 0.0]]], dtype=torch.float64)

        first_ellipse_force = force_from_tensor_vrc_tube(
            pedestrian_positions=position,
            vrc_tube=tube,
            current_index=0,
            force_params=self.force_params,
            preview_steps=1,
            discount=0.5,
        )
        force_with_inactive_preview = force_from_tensor_vrc_tube(
            pedestrian_positions=position,
            vrc_tube=tube,
            current_index=0,
            force_params=self.force_params,
            preview_steps=2,
            discount=0.5,
        )
        torch.testing.assert_close(
            force_with_inactive_preview,
            first_ellipse_force / 1.5,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_singleton_dimensions_and_speed_limit(self) -> None:
        states = torch.zeros((1, 2, 3), dtype=torch.float64)
        controls = torch.zeros((1, 1, 2), dtype=torch.float64)
        tube = build_tensor_vrc_tube(states, controls, self.vrc_params)
        prediction = predict_pedestrians_with_tensor_vrc(
            current_positions=torch.zeros((1, 1, 2), dtype=torch.float64),
            current_velocities=torch.tensor(
                [[[3.0, 0.0]]],
                dtype=torch.float64,
            ),
            vrc_tube=tube,
            horizon=1,
            dt=0.1,
            force_params=RobotForceParameters(strength=0.0),
            relaxation_time=0.5,
            maximum_speed=2.0,
            preview_steps=1,
        )
        self.assertEqual(prediction.shape, (1, 1, 1, 2))
        torch.testing.assert_close(
            prediction[0, 0, 0],
            torch.tensor([0.2, 0.0], dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
