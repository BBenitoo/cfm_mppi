from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import torch


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


@dataclass
class TensorVRCTube:
    """Batched VRC representation that stays on the Torch device.

    All tensors include a leading branch dimension and a time dimension:
    ``centers`` and ``headings`` have shape ``[B, T, 2]`` while the radii
    have shape ``[B, T]``.
    """

    centers: torch.Tensor
    headings: torch.Tensor
    lateral_headings: torch.Tensor
    longitudinal_radii: torch.Tensor
    lateral_radii: torch.Tensor
    inv_longitudinal_radius_sq: torch.Tensor
    inv_lateral_radius_sq: torch.Tensor

    @property
    def num_branches(self) -> int:
        return self.centers.shape[0]

    @property
    def length(self) -> int:
        return self.centers.shape[1]


def build_vrc_tube(
    #robot_trajectory: NDArray[np.float64],
    states: torch.Tensor,
    controls_uni: torch.Tensor,
    params: VRCParameters,
) -> list[VRCEllipse]:
    """
    states shape: (K + 1, 3)
    controls_uni shape: (K, 2)
    columns: [x, y, theta, v, omega]
    """
    # Convert torch tensors in GPU to numpy arrays
    states = states.detach().cpu().numpy()
    controls_uni = controls_uni.detach().cpu().numpy()

    ellipses: list[VRCEllipse] = [] #定义一个名为 ellipses 的变量，它是一个“由 VRCEllipse 对象组成的列表”，并且初始为空列表。

    base_radius = (
        params.robot_radius
        + params.pedestrian_radius
        + params.personal_margin
    )

    for k, state in enumerate(states):
        x, y, theta = state

        if k < len(controls_uni):
            v, omega = controls_uni[k]
        else:
            v, omega = controls_uni[-1]

        speed = abs(v)

        braking_distance = (
            speed**2
            / max(2.0 * params.braking_deceleration, 1e-2) # take the bigger of the two to avoid division by zero
        )

        longitudinal_radius = (
            base_radius
            + params.forward_time_headway * speed
            + braking_distance
        ) # the major axis of the ellipse

        lateral_radius = (
            base_radius
            + params.lateral_time_headway * speed
            + params.angular_gain * abs(omega)
        ) # the minor axis of the ellipse

        longitudinal_radius = np.clip(
            longitudinal_radius,
            params.min_longitudinal_radius,
            params.max_longitudinal_radius,
        ) # Clip the longitudinal radius to the specified bounds

        lateral_radius = np.clip(
            lateral_radius,
            params.min_lateral_radius,
            params.max_lateral_radius,
        ) # Clip the lateral radius to the specified bounds

        heading = np.array(
            [np.cos(theta), np.sin(theta)],
            dtype=np.float64,
        ) # the heading unit vector of the robot, based on its orientation theta

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


def build_tensor_vrc_tube(
    states: torch.Tensor,
    controls_uni: torch.Tensor,
    params: VRCParameters,
) -> TensorVRCTube:
    """Build VRC tubes for every branch without leaving the Torch device.

    Args:
        states: Robot states with shape ``[B, H + 1, 3]``.
        controls_uni: Unicycle controls with shape ``[B, H, 2]``.
        params: VRC geometry parameters.
    """
    if states.ndim != 3 or states.shape[-1] != 3:
        raise ValueError("states must have shape [B, H + 1, 3]")
    if controls_uni.ndim != 3 or controls_uni.shape[-1] != 2:
        raise ValueError("controls_uni must have shape [B, H, 2]")
    if states.shape[0] != controls_uni.shape[0]:
        raise ValueError("states and controls_uni must have the same branch count")
    if states.shape[1] != controls_uni.shape[1] + 1:
        raise ValueError("states must contain exactly one more step than controls_uni")
    if controls_uni.shape[1] == 0:
        raise ValueError("controls_uni must contain at least one control step")

    # The scalar implementation applies the final control to the final state.
    controls_for_states = torch.cat(
        [controls_uni, controls_uni[:, -1:, :]],
        dim=1,
    )
    linear_velocity = controls_for_states[..., 0]
    angular_velocity = controls_for_states[..., 1]
    speed = torch.abs(linear_velocity)

    base_radius = (
        params.robot_radius
        + params.pedestrian_radius
        + params.personal_margin
    )
    braking_denominator = max(
        2.0 * params.braking_deceleration,
        1e-2,
    )
    longitudinal_radii = (
        base_radius
        + params.forward_time_headway * speed
        + speed.square() / braking_denominator
    ).clamp(
        min=params.min_longitudinal_radius,
        max=params.max_longitudinal_radius,
    )
    lateral_radii = (
        base_radius
        + params.lateral_time_headway * speed
        + params.angular_gain * torch.abs(angular_velocity)
    ).clamp(
        min=params.min_lateral_radius,
        max=params.max_lateral_radius,
    )

    theta = states[..., 2]
    headings = torch.stack(
        [torch.cos(theta), torch.sin(theta)],
        dim=-1,
    )
    lateral_headings = torch.stack(
        [-headings[..., 1], headings[..., 0]],
        dim=-1,
    )
    center_shift = (
        params.center_shift_ratio
        * params.forward_time_headway
        * linear_velocity
    )
    centers = states[..., :2] + center_shift.unsqueeze(-1) * headings

    return TensorVRCTube(
        centers=centers,
        headings=headings,
        lateral_headings=lateral_headings,
        longitudinal_radii=longitudinal_radii,
        lateral_radii=lateral_radii,
        inv_longitudinal_radius_sq=longitudinal_radii.square().reciprocal(),
        inv_lateral_radius_sq=lateral_radii.square().reciprocal(),
    )


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
    strength: float = 1.5 # A
    decay: float = 0.35 # B
    influence_cutoff: float = 2.5 # the maximum distance at which the robot can influence the pedestrian
    maximum_force: float = 2.5


def force_from_tensor_vrc_tube(
    pedestrian_positions: torch.Tensor,
    vrc_tube: TensorVRCTube,
    current_index: int,
    force_params: RobotForceParameters,
    preview_steps: int = 8,
    discount: float = 0.85,
    preview_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate VRC forces for all branches and pedestrians in one batch.

    Args:
        pedestrian_positions: Positions with shape ``[B, P, 2]``.
        vrc_tube: Batched VRC geometry with ``B`` branches.
        current_index: First VRC time index included in the preview.

    Returns:
        Force tensor with shape ``[B, P, 2]``.
    """
    if (
        pedestrian_positions.ndim != 3
        or pedestrian_positions.shape[0] != vrc_tube.num_branches
        or pedestrian_positions.shape[-1] != 2
    ):
        raise ValueError(
            "pedestrian_positions must have shape [B, P, 2] and match "
            "the VRC branch count"
        )
    if current_index < 0:
        raise ValueError("current_index must be non-negative")

    end_index = min(
        current_index + preview_steps,
        vrc_tube.length,
    )
    if current_index >= end_index:
        return torch.zeros_like(pedestrian_positions)

    centers = vrc_tube.centers[:, current_index:end_index, :]
    headings = vrc_tube.headings[:, current_index:end_index, :]
    lateral_headings = vrc_tube.lateral_headings[
        :, current_index:end_index
    ]
    inv_longitudinal_radius_sq = vrc_tube.inv_longitudinal_radius_sq[
        :, current_index:end_index
    ]
    inv_lateral_radius_sq = vrc_tube.inv_lateral_radius_sq[
        :, current_index:end_index
    ]

    # [B, P, W, 2], where W is the available preview length.
    displacement = pedestrian_positions.unsqueeze(2) - centers.unsqueeze(1)
    longitudinal_heading = headings.unsqueeze(1)
    lateral_heading = lateral_headings.unsqueeze(1)

    longitudinal_displacement = torch.sum(
        displacement * longitudinal_heading,
        dim=-1,
    )
    lateral_displacement = torch.sum(
        displacement * lateral_heading,
        dim=-1,
    )
    squared_distance = (
        longitudinal_displacement.square()
        * inv_longitudinal_radius_sq.unsqueeze(1)
        + lateral_displacement.square()
        * inv_lateral_radius_sq.unsqueeze(1)
    )
    rho = torch.sqrt(torch.clamp_min(squared_distance, 0.0))

    longitudinal_normal = (
        longitudinal_displacement
        * inv_longitudinal_radius_sq.unsqueeze(1)
    )
    lateral_normal = (
        lateral_displacement
        * inv_lateral_radius_sq.unsqueeze(1)
    )
    normal = (
        longitudinal_normal.unsqueeze(-1) * longitudinal_heading
        + lateral_normal.unsqueeze(-1) * lateral_heading
    )
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    normalized_normal = normal / normal_norm.clamp_min(1e-8)
    fallback_normal = lateral_heading.expand_as(normal)
    normalized_normal = torch.where(
        normal_norm < 1e-8,
        fallback_normal,
        normalized_normal,
    )

    magnitude = force_params.strength * torch.exp(
        (1.0 - rho) / force_params.decay
    )
    magnitude = torch.clamp(magnitude, max=force_params.maximum_force)
    ellipse_forces = magnitude.unsqueeze(-1) * normalized_normal
    ellipse_forces = torch.where(
        (rho < force_params.influence_cutoff).unsqueeze(-1),
        ellipse_forces,
        torch.zeros_like(ellipse_forces),
    )

    preview_length = end_index - current_index
    if preview_weights is None:
        preview_offsets = torch.arange(
            preview_length,
            device=pedestrian_positions.device,
            dtype=pedestrian_positions.dtype,
        )
        preview_weights = torch.pow(
            torch.as_tensor(
                discount,
                device=pedestrian_positions.device,
                dtype=pedestrian_positions.dtype,
            ),
            preview_offsets,
        )
    else:
        if preview_weights.numel() < preview_length:
            raise ValueError("preview_weights is shorter than the preview window")
        preview_weights = preview_weights.to(
            device=pedestrian_positions.device,
            dtype=pedestrian_positions.dtype,
        )[:preview_length]

    total_force = torch.sum(
        ellipse_forces * preview_weights.view(1, 1, -1, 1),
        dim=2,
    )
    total_weight = torch.sum(preview_weights)
    return torch.where(
        total_weight < 1e-8,
        total_force,
        total_force / total_weight.clamp_min(1e-8),
    )


def predict_pedestrians_with_tensor_vrc(
    current_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    vrc_tube: TensorVRCTube,
    horizon: int,
    dt: float,
    force_params: RobotForceParameters,
    relaxation_time: float = 0.5,
    maximum_speed: float = 2.0,
    preview_steps: int = 8,
    preview_discount: float = 0.85,
) -> torch.Tensor:
    """Roll out all branch-conditioned pedestrian predictions on one device.

    Only the time recurrence remains in Python. Branches, pedestrians, and VRC
    preview ellipses are evaluated by batched Torch operations.

    Returns:
        Predicted positions with shape ``[B, P, horizon, 2]``.
    """
    if current_positions.ndim == 3:
        if current_positions.shape[0] != 1:
            raise ValueError("current_positions batch dimension must be 1")
        current_positions = current_positions.squeeze(0)
    if current_velocities.ndim == 3:
        if current_velocities.shape[0] != 1:
            raise ValueError("current_velocities batch dimension must be 1")
        current_velocities = current_velocities.squeeze(0)
    if (
        current_positions.ndim != 2
        or current_velocities.ndim != 2
        or current_positions.shape != current_velocities.shape
        or current_positions.shape[-1] != 2
    ):
        raise ValueError(
            "current_positions and current_velocities must have shape [1, P, 2] "
            "or [P, 2]"
        )
    if horizon < 0 or horizon > vrc_tube.length:
        raise ValueError("horizon must be between 0 and the VRC tube length")

    device = vrc_tube.centers.device
    dtype = vrc_tube.centers.dtype
    positions = current_positions.to(device=device, dtype=dtype)
    velocities = current_velocities.to(device=device, dtype=dtype)
    positions = positions.unsqueeze(0).expand(
        vrc_tube.num_branches, -1, -1
    ).clone()
    velocities = velocities.unsqueeze(0).expand(
        vrc_tube.num_branches, -1, -1
    ).clone()
    baseline_velocities = velocities.clone()
    prediction = torch.empty(
        vrc_tube.num_branches,
        positions.shape[1],
        horizon,
        2,
        device=device,
        dtype=dtype,
    )

    preview_offsets = torch.arange(
        max(preview_steps, 0),
        device=device,
        dtype=dtype,
    )
    preview_weights = torch.pow(
        torch.as_tensor(preview_discount, device=device, dtype=dtype),
        preview_offsets,
    )
    relaxation_denominator = max(relaxation_time, 1e-6)

    for step in range(horizon):
        vrc_force = force_from_tensor_vrc_tube(
            pedestrian_positions=positions,
            vrc_tube=vrc_tube,
            current_index=step,
            force_params=force_params,
            preview_steps=preview_steps,
            discount=preview_discount,
            preview_weights=preview_weights,
        )
        relaxation_force = (
            baseline_velocities - velocities
        ) / relaxation_denominator
        velocities = velocities + (relaxation_force + vrc_force) * dt

        speed = torch.linalg.vector_norm(
            velocities,
            dim=-1,
            keepdim=True,
        )
        speed_scale = torch.clamp(
            maximum_speed / speed.clamp_min(torch.finfo(dtype).tiny),
            max=1.0,
        )
        velocities = velocities * speed_scale
        positions = positions + velocities * dt
        prediction[:, :, step, :] = positions

    return prediction


def force_from_vrc_ellipse(
    pedestrian_position: NDArray[np.float64],
    ellipse: VRCEllipse,
    params: RobotForceParameters,
) -> NDArray[np.float64]:
    """
    Returns an acceleration-like social force in m/s^2.
    """
    displacement = pedestrian_position - ellipse.center

    rho = elliptical_distance(pedestrian_position, ellipse)

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
