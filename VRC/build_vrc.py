from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class VRCParameters:
    robot_radius: float = 0.35
    pedestrian_radius: float = 0.30
    personal_margin: float = 0.20

    forward_time_headway: float = 0.8
    lateral_time_headway: float = 0.15

    braking_deceleration: float = 1.0
    angular_gain: float = 0.15
    center_shift_ratio: float = 0.5

    min_longitudinal_radius: float = 0.75
    max_longitudinal_radius: float = 3.0

    min_lateral_radius: float = 0.70
    max_lateral_radius: float = 1.50


@dataclass
class VRCEllipse:
    center: NDArray[np.float64]
    sigma: NDArray[np.float64]
    sigma_inv: NDArray[np.float64]

    longitudinal_radius: float
    lateral_radius: float

    theta: float
    time_index: int


def build_vrc_tube(
    robot_trajectory: NDArray[np.float64],
    params: VRCParameters,
) -> list[VRCEllipse]:
    """
    robot_trajectory shape: (K + 1, 5)
    columns: [x, y, theta, v, omega]
    """
    ellipses: list[VRCEllipse] = [] #定义一个名为 ellipses 的变量，它是一个“由 VRCEllipse 对象组成的列表”，并且初始为空列表。

    base_radius = (
        params.robot_radius
        + params.pedestrian_radius
        + params.personal_margin
    )

    for k, state in enumerate(robot_trajectory):
        x, y, theta, v, omega = state

        speed = abs(v)

        braking_distance = (
            speed**2
            / max(2.0 * params.braking_deceleration, 1e-6)
        )

        longitudinal_radius = (
            base_radius
            + params.forward_time_headway * speed
            + braking_distance
        )

        lateral_radius = (
            base_radius
            + params.lateral_time_headway * speed
            + params.angular_gain * abs(omega)
        )

        longitudinal_radius = np.clip(
            longitudinal_radius,
            params.min_longitudinal_radius,
            params.max_longitudinal_radius,
        )

        lateral_radius = np.clip(
            lateral_radius,
            params.min_lateral_radius,
            params.max_lateral_radius,
        )

        heading = np.array(
            [np.cos(theta), np.sin(theta)],
            dtype=np.float64,
        )

        center_shift = (
            params.center_shift_ratio
            * params.forward_time_headway
            * v
        )

        center = np.array([x, y]) + center_shift * heading

        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float64,
        )

        local_covariance = np.diag(
            [
                longitudinal_radius**2,
                lateral_radius**2,
            ]
        )

        sigma = rotation @ local_covariance @ rotation.T
        sigma_inv = np.linalg.inv(sigma)

        ellipses.append(
            VRCEllipse(
                center=center,
                sigma=sigma,
                sigma_inv=sigma_inv,
                longitudinal_radius=float(longitudinal_radius),
                lateral_radius=float(lateral_radius),
                theta=float(theta),
                time_index=k,
            )
        )

    return ellipses


# Calculate the distance from a pedestrian to the VRC ellipse
def elliptical_distance(
    point: NDArray[np.float64],
    ellipse: VRCEllipse,
) -> float:
    displacement = point - ellipse.center

    squared_distance = (
        displacement.T
        @ ellipse.sigma_inv
        @ displacement
    )

    return float(np.sqrt(max(squared_distance, 0.0)))


# Calculate the repulsive force from a VRC ellipse on a pedestrian
@dataclass
class RobotForceParameters:
    strength: float = 1.5
    decay: float = 0.35
    influence_cutoff: float = 2.5
    maximum_force: float = 2.5


def force_from_vrc_ellipse(
    pedestrian_position: NDArray[np.float64],
    ellipse: VRCEllipse,
    params: RobotForceParameters,
) -> NDArray[np.float64]:
    """
    Returns an acceleration-like social force in m/s^2.
    """
    displacement = pedestrian_position - ellipse.center

    rho_squared = (
        displacement.T
        @ ellipse.sigma_inv
        @ displacement
    )

    rho = float(np.sqrt(max(rho_squared, 1e-12)))

    if rho >= params.influence_cutoff:
        return np.zeros(2, dtype=np.float64)

    normal = ellipse.sigma_inv @ displacement
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-8:
        # Very rare case: pedestrian exactly at ellipse center.
        normal = np.array(
            [
                -np.sin(ellipse.theta),
                np.cos(ellipse.theta),
            ],
            dtype=np.float64,
        )
    else:
        normal = normal / normal_norm

    magnitude = params.strength * np.exp(
        (1.0 - rho) / params.decay
    )

    magnitude = min(
        magnitude,
        params.maximum_force,
    )

    return magnitude * normal


# Calculate the repulsive force from a VRC tube on a pedestrian
def force_from_vrc_tube(
    pedestrian_position: NDArray[np.float64],
    vrc_tube: list[VRCEllipse],
    current_index: int,
    force_params: RobotForceParameters,
    preview_steps: int = 8,
    discount: float = 0.85,
) -> NDArray[np.float64]:
    end_index = min(
        current_index + preview_steps,
        len(vrc_tube),
    )

    total_force = np.zeros(2, dtype=np.float64)
    total_weight = 0.0

    for future_index in range(current_index, end_index):
        relative_step = future_index - current_index
        weight = discount**relative_step

        ellipse_force = force_from_vrc_ellipse(
            pedestrian_position,
            vrc_tube[future_index],
            force_params,
        )

        total_force += weight * ellipse_force
        total_weight += weight

    if total_weight < 1e-8:
        return total_force

    return total_force / total_weight

