from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from cfm_mppi.evaluation.eval_utils import (
    CFMConfig,
    EpisodeMetrics,
    FreezingMetrics,
    compute_episode_metrics,
    compute_freezing_metrics,
    run_CFM,
    summarize_freezing_metrics,
    summarize_metrics,
)
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import (
    stage_cost,
    terminal_cost,
    unicycle_dynamics,
)
from cfm_mppi.vrc.build_vrc import (
    RobotForceParameters,
    VRCEllipse,
    VRCParameters,
    build_tensor_vrc_tube,
    build_vrc_tube,
    force_from_vrc_tube,
    predict_pedestrians_with_tensor_vrc,
)

if TYPE_CHECKING:
    from cfm_mppi.utils import AgentHistory, HumanAgent


# Evaluation horizon and CFM parameters
SAFE_MARGIN = 0.5
HORIZON = 80
SAFE_COEF = [0.1, 0.3, 0.5, 0.7, 0.9]
GOAL_COEF = 0.1
ODE_TIMES = [0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
ODE_TIMES_WARM = [0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
NOISE_LEVEL_VALUE = 0.8

# MPPI parameters. Each of the N_BRANCHES gets N_MPPI_SAMPLES samples.
N_CFM_SAMPLES = 200
N_BRANCHES = 10
N_MPPI_SAMPLES = 200
MPPI_SIGMA = torch.tensor([0.3, 0.6])
MPPI_LAMBDA = 0.1
U_MIN = torch.tensor([-2.0, -2.0])
U_MAX = torch.tensor([2.0, 2.0])
LOOK_AHEAD_DISTANCE = 0.1

SPACE_SCALE = 10.0
DT = 0.1
MAX_HISTORY_LENGTH = 10
# A freeze is one maximal low-speed interval away from the goal.
FREEZING_MINIMUM_DURATION = 1.0
FREEZING_SPEED_THRESHOLD = 0.05
FREEZING_GOAL_DISTANCE_THRESHOLD = 0.5

VRC_PARAMS = VRCParameters()
ROBOT_FORCE_PARAMS = RobotForceParameters()


@dataclass
class PedestrianPredictionParameters:
    """Parameters for a VRC-conditioned pedestrian rollout."""

    relaxation_time: float = 0.5
    maximum_speed: float = 2.0
    preview_steps: int = 8
    preview_discount: float = 0.85


@dataclass
class BranchPlan:
    """All branch-dependent outputs needed by one closed-loop step."""

    selected_controls: torch.Tensor
    selected_cfm_controls: torch.Tensor
    selected_branch: int
    cfm_branch_indices: torch.Tensor
    cfm_branch_states: torch.Tensor
    cfm_branch_controls: torch.Tensor
    pedestrian_predictions: torch.Tensor
    mppi_branch_states: torch.Tensor
    branch_costs: torch.Tensor
    selected_vrc_tube: list[VRCEllipse] | None


def constant_velocity_prediction(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    horizon: int,
    dt: float,
) -> torch.Tensor:
    """Predict future positions with shape ``[1, num_pedestrians, 2, horizon]``."""
    offsets = (
        torch.arange(
            1,
            horizon + 1,
            device=positions.device,
            dtype=positions.dtype,
        )
        * dt
    )
    return positions.unsqueeze(-1) + velocities.unsqueeze(-1) * offsets


def generate_cfm_candidates(
    model: TransformerModel,
    config: CFMConfig,
    state: torch.Tensor,
    goal: torch.Tensor,
    noisy_action_seq: torch.Tensor,
    noise_level: torch.Tensor,
    current_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    planning_horizon: int,
    histories: dict[str, AgentHistory],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run only CFM and return its future candidate controls.

    The original ``synthesize_control`` immediately invokes MPPI using a single
    constant-velocity obstacle prediction. VRC planning needs to stop between
    these two stages so that every CFM branch can create its own pedestrian
    prediction first.
    """
    state_history = histories["ego_state"].get()
    control_history = histories["ego_control_sin"].get()
    obstacle_state_history = histories["obs_state"].get()
    obstacle_control_history = histories["obs_control"].get()
    history_length = len(histories["ego_state"])
    future_horizon = planning_horizon - history_length
    if future_horizon <= 0:
        raise ValueError("The planning horizon must exceed the history length.")

    future_positions = constant_velocity_prediction(
        current_positions, current_velocities, future_horizon, config.dt
    )
    future_velocities = current_velocities.unsqueeze(-1).expand(
        -1, -1, -1, future_horizon
    )

    if control_history is None:
        obstacle_positions = future_positions
        obstacle_velocities = future_velocities
        origin = state[:, :2]
    else:
        obstacle_positions = torch.cat(
            [obstacle_state_history, future_positions], dim=-1
        )
        obstacle_velocities = torch.cat(
            [obstacle_control_history, future_velocities], dim=-1
        )
        origin = state_history[:, :2, 0]

    goal_cfm = goal - origin
    obstacle_positions_cfm = obstacle_positions - origin.unsqueeze(1).unsqueeze(-1)
    controls_sin = run_CFM(
        model=model,
        config=config,
        noisy_action_seq=noisy_action_seq,
        noise_level=noise_level,
        start_pos=torch.zeros(1, 2, device=config.device),
        goal_pos=goal_cfm,
        obs_positions=obstacle_positions_cfm,
        obs_velocities=obstacle_velocities,
        control_history=control_history,
    ).detach()

    future_controls_sin = controls_sin[:, :, history_length:].transpose(1, 2)
    future_obstacles = (
        obstacle_positions[:, :, :, history_length:].squeeze(0).transpose(1, 2)
    )
    return (
        controls_sin,
        future_controls_sin,
        future_obstacles,
        history_length,
    )


def select_cfm_branches(
    solver: FlowMPPI,
    state: torch.Tensor,
    controls_sin: torch.Tensor,
    goal: torch.Tensor,
    obstacle_prediction: torch.Tensor,
    num_branches: int,
    radius: float,
    look_ahead_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out CFM candidates and retain the lowest-cost nominal branches."""
    candidate_states, candidate_controls = solver.rollout_si_controls(
        state, controls_sin, d=look_ahead_distance
    )
    candidate_costs = solver.score_trajectories(
        candidate_states,
        candidate_controls,
        goal.squeeze(),
        obstacle_prediction,
        radius,
    )
    branch_count = min(num_branches, candidate_controls.shape[0])
    branch_indices = torch.topk(candidate_costs, k=branch_count, largest=False).indices
    return (
        branch_indices,
        candidate_states[branch_indices],
        candidate_controls[branch_indices],
    )


# Retained as a scalar NumPy reference for regression checks and compatibility.
# The planner hot path uses predict_pedestrians_with_tensor_vrc instead.
def predict_pedestrians_with_vrc(
    current_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    vrc_tube,
    horizon: int,
    dt: float,
    force_params: RobotForceParameters,
    prediction_params: PedestrianPredictionParameters,
) -> torch.Tensor:
    """Predict pedestrians under the VRC induced by one robot branch.

    The current measured velocity is the pedestrian's baseline intent. A
    relaxation term prevents a temporary VRC force from permanently increasing
    velocity after the robot is no longer influential.

    Returns:
        Tensor with shape ``[num_pedestrians, horizon, 2]``.
    """
    device = current_positions.device
    dtype = current_positions.dtype
    positions = current_positions.squeeze(0).detach().cpu().numpy().astype(np.float64)
    velocities = current_velocities.squeeze(0).detach().cpu().numpy().astype(np.float64)
    baseline_velocities = velocities.copy()
    prediction = np.empty((positions.shape[0], horizon, 2), dtype=np.float64)

    for step in range(horizon):
        for pedestrian_index in range(positions.shape[0]):
            vrc_force = force_from_vrc_tube(
                pedestrian_position=positions[pedestrian_index],
                vrc_tube=vrc_tube,
                current_index=min(step, len(vrc_tube) - 1),
                force_params=force_params,
                preview_steps=prediction_params.preview_steps,
                discount=prediction_params.preview_discount,
            )
            relaxation_force = (
                baseline_velocities[pedestrian_index] - velocities[pedestrian_index]
            ) / max(prediction_params.relaxation_time, 1e-6)
            velocities[pedestrian_index] += (relaxation_force + vrc_force) * dt

            speed = np.linalg.norm(velocities[pedestrian_index])
            if speed > prediction_params.maximum_speed:
                velocities[pedestrian_index] *= prediction_params.maximum_speed / speed
            positions[pedestrian_index] += velocities[pedestrian_index] * dt

        prediction[:, step, :] = positions

    return torch.as_tensor(prediction, device=device, dtype=dtype)


def build_branch_pedestrian_predictions(
    branch_states: torch.Tensor,
    branch_controls: torch.Tensor,
    current_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    vrc_params: VRCParameters,
    force_params: RobotForceParameters,
    prediction_params: PedestrianPredictionParameters,
    dt: float,
) -> torch.Tensor:
    """Build and roll out all branch-conditioned predictions on one device."""
    horizon = branch_controls.shape[1]
    vrc_tube = build_tensor_vrc_tube(
        states=branch_states,
        controls_uni=branch_controls,
        params=vrc_params,
    )
    return predict_pedestrians_with_tensor_vrc(
        current_positions=current_positions,
        current_velocities=current_velocities,
        vrc_tube=vrc_tube,
        horizon=horizon,
        dt=dt,
        force_params=force_params,
        relaxation_time=prediction_params.relaxation_time,
        maximum_speed=prediction_params.maximum_speed,
        preview_steps=prediction_params.preview_steps,
        preview_discount=prediction_params.preview_discount,
    )


def plan_vrc_branches(
    model: TransformerModel,
    solver: FlowMPPI,
    config: CFMConfig,
    state: torch.Tensor,
    goal: torch.Tensor,
    noisy_action_seq: torch.Tensor,
    noise_level: torch.Tensor,
    current_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    planning_horizon: int,
    histories: dict[str, AgentHistory],
    num_branches: int = N_BRANCHES,
    look_ahead_distance: float = LOOK_AHEAD_DISTANCE,
    vrc_params: VRCParameters = VRC_PARAMS,
    force_params: RobotForceParameters = ROBOT_FORCE_PARAMS,
    prediction_params: PedestrianPredictionParameters | None = None,
    build_selected_vrc_tube: bool = True,
) -> BranchPlan:
    """Execute CFM -> VRC -> pedestrian prediction -> branch MPPI."""
    if prediction_params is None:
        prediction_params = PedestrianPredictionParameters()

    with torch.no_grad():
        (
            cfm_controls,
            future_cfm_controls,
            constant_velocity_obstacles,
            _,
        ) = generate_cfm_candidates(
            model=model,
            config=config,
            state=state,
            goal=goal,
            noisy_action_seq=noisy_action_seq,
            noise_level=noise_level,
            current_positions=current_positions,
            current_velocities=current_velocities,
            planning_horizon=planning_horizon,
            histories=histories,
        )
        (
            branch_indices,
            branch_states,
            branch_controls,
        ) = select_cfm_branches(
            solver=solver,
            state=state,
            controls_sin=future_cfm_controls,
            goal=goal,
            obstacle_prediction=constant_velocity_obstacles,
            num_branches=num_branches,
            radius=config.agent_radius,
            look_ahead_distance=look_ahead_distance,
        )

    with torch.no_grad():
        pedestrian_predictions = build_branch_pedestrian_predictions(
            branch_states=branch_states,
            branch_controls=branch_controls,
            current_positions=current_positions,
            current_velocities=current_velocities,
            vrc_params=vrc_params,
            force_params=force_params,
            prediction_params=prediction_params,
            dt=config.dt,
        )

    with torch.no_grad():
        (
            selected_controls,
            _,
            mppi_branch_states,
            branch_costs,
        ) = solver.forward_branches(
            state=state,
            branch_controls=branch_controls,
            goal=goal,
            branch_obstacle_states=pedestrian_predictions,
            rad=config.agent_radius,
        )

    selected_branch = int(torch.argmin(branch_costs).item())
    selected_cfm_index = int(branch_indices[selected_branch].item())
    selected_vrc_tube = (
        build_vrc_tube(
            states=mppi_branch_states[selected_branch],
            controls_uni=selected_controls,
            params=vrc_params,
        )
        if build_selected_vrc_tube
        else None
    )
    return BranchPlan(
        selected_controls=selected_controls,
        selected_cfm_controls=cfm_controls[selected_cfm_index : selected_cfm_index + 1],
        selected_branch=selected_branch,
        cfm_branch_indices=branch_indices,
        cfm_branch_states=branch_states,
        cfm_branch_controls=branch_controls,
        pedestrian_predictions=pedestrian_predictions,
        mppi_branch_states=mppi_branch_states,
        branch_costs=branch_costs,
        selected_vrc_tube=selected_vrc_tube,
    )


def update_sfm_environment(
    humans: list[HumanAgent],
    positions: torch.Tensor,
    velocities: torch.Tensor,
    state: torch.Tensor,
    histories: dict[str, AgentHistory],
    time_index: int,
    vrc_tube: list[VRCEllipse] | None = None,
    vrc_force_params: RobotForceParameters = ROBOT_FORCE_PARAMS,
    vrc_current_index: int = 0,
) -> None:
    """Advance synthetic pedestrians using human-human and robot-VRC forces."""
    if time_index == 0:
        return

    num_humans = len(humans)
    has_vrc = vrc_tube is not None and len(vrc_tube) > 0
    robot_velocity = None
    if not has_vrc:
        robot_velocity = histories["ego_control_sin"].get()[:, :, -1].cpu()
    for human_index, human in enumerate(humans):
        other_indices = np.r_[0:human_index, human_index + 1 : num_humans]
        other_states = positions[0, other_indices, :, time_index - 1].cpu()
        other_controls = velocities[0, other_indices, :, time_index - 1].cpu()

        # When a VRC is available it replaces the robot's point-agent force,
        # avoiding double-counting the same robot interaction.
        if not has_vrc:
            other_states = torch.cat([other_states, state[:, :2].cpu()], dim=0)
            other_controls = torch.cat([other_controls, robot_velocity], dim=0)

        human.social_force_step(
            other_states.numpy(),
            other_controls.numpy(),
            vrc_tube=vrc_tube,
            vrc_current_index=vrc_current_index,
            vrc_force_params=vrc_force_params,
        )
        positions[0, human_index, :, time_index] = torch.as_tensor(
            human.state, device=positions.device
        )
        velocities[0, human_index, :, time_index] = torch.as_tensor(
            human.control, device=velocities.device
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate closed-loop CFM-VRC-MPPI planning."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="sfm",
        choices=("ucy", "sdd", "sfm"),
    )
    return parser.parse_args()


def main() -> None:
    from cfm_mppi.utils import AgentHistory, HumanAgent

    args = parse_args()
    dataset = args.dataset
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_level = torch.tensor([NOISE_LEVEL_VALUE], device=device)

    checkpoint_path = Path("../output_dir/cfm_transformer/checkpoint.pth")
    model = TransformerModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device=device)

    if dataset in ("ucy", "sdd"):
        batch_ego = torch.load(f"../../dataset/eval80_ego_{dataset}.pt", map_location="cpu")
        with open(f"../../dataset/eval80_obs_{dataset}.pkl", "rb") as file:
            batch_obs = pickle.load(file)
    else:
        batch_ego = torch.zeros(300)

    all_average_times = []
    episodes: list[EpisodeMetrics] = []
    freezing_episodes: list[FreezingMetrics] = []
    state_trajectories = torch.zeros(
        [batch_ego.shape[0], 3, HORIZON + 1], dtype=torch.float32
    )
    control_trajectories = torch.zeros(
        [batch_ego.shape[0], 2, HORIZON], dtype=torch.float32
    )

    for scenario_index in range(batch_ego.shape[0]):
        if dataset in ("ucy", "sdd"):
            state_obs = batch_obs[scenario_index]
            nan_mask = torch.isnan(state_obs).any(dim=(0, 2, 3))
            state_obs = state_obs[:, ~nan_mask]
            pos_obs = state_obs[:, :, :2, :].to(device)
            vel_obs = state_obs[:, :, 2:4, :].to(device)
            goal = batch_ego[scenario_index, :2, -1].to(device)
            humans = []
        else:
            goal = torch.tensor([[6.0, 6.0]], dtype=torch.float32, device=device)
            rng = np.random.RandomState(scenario_index)
            humans = [
                HumanAgent(goal.squeeze().cpu().numpy(), random_generator=rng)
                for _ in range(20)
            ]
            pos_obs = torch.zeros(
                [1, len(humans), 2, HORIZON],
                dtype=torch.float32,
                device=device,
            )
            vel_obs = torch.zeros_like(pos_obs)
            for human_index, human in enumerate(humans):
                pos_obs[0, human_index, :, 0] = torch.as_tensor(
                    human.state, device=device
                )
                vel_obs[0, human_index, :, 0] = torch.as_tensor(
                    human.control, device=device
                )

        solver = FlowMPPI(
            num_samples=N_MPPI_SAMPLES,
            dim_state=3,
            dim_control=2,
            dynamics=unicycle_dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            u_min=U_MIN,
            u_max=U_MAX,
            sigmas=MPPI_SIGMA,
            lambda_=MPPI_LAMBDA,
            goal=goal.squeeze(),
            horizon=HORIZON,
            dt=DT,
            device=device,
            dynamics_type="unicycle",
        )

        state = torch.zeros(1, 3, device=device)
        x_t = torch.randn(
            [N_CFM_SAMPLES, 2, HORIZON],
            dtype=torch.float32,
            device=device,
        )
        state_hist = torch.zeros([3, HORIZON + 1], dtype=torch.float32)
        state_hist[:, 0] = state.cpu()
        control_hist = torch.zeros([2, HORIZON], dtype=torch.float32)
        histories = {
            "ego_state": AgentHistory(max_length=MAX_HISTORY_LENGTH),
            "ego_control_sin": AgentHistory(max_length=MAX_HISTORY_LENGTH),
            "obs_state": AgentHistory(max_length=MAX_HISTORY_LENGTH),
            "obs_control": AgentHistory(max_length=MAX_HISTORY_LENGTH),
        }

        total_time = 0.0
        active_vrc_tube = None
        for step in range(HORIZON):
            if dataset == "sfm":
                update_sfm_environment(
                    humans=humans,
                    positions=pos_obs,
                    velocities=vel_obs,
                    state=state,
                    histories=histories,
                    time_index=step,
                    vrc_tube=active_vrc_tube,
                    vrc_force_params=ROBOT_FORCE_PARAMS,
                    vrc_current_index=0,
                )

            time_start = time.time()
            current_positions = pos_obs[:, :, :, step]
            current_velocities = vel_obs[:, :, :, step]
            if step == 0:
                current_noise_level = torch.tensor([0.0], device=device)
                ode_times = ODE_TIMES
            else:
                current_noise_level = noise_level
                ode_times = ODE_TIMES_WARM

            config = CFMConfig(
                ode_times=ode_times,
                dt=DT,
                agent_radius=SAFE_MARGIN,
                space_scale=SPACE_SCALE,
                safe_margin_coefs=SAFE_COEF,
                goal_margin_coef=GOAL_COEF,
                device=device,
            )
            plan = plan_vrc_branches(
                model=model,
                solver=solver,
                config=config,
                state=state,
                goal=goal,
                noisy_action_seq=x_t,
                noise_level=current_noise_level,
                current_positions=current_positions,
                current_velocities=current_velocities,
                planning_horizon=x_t.shape[-1],
                histories=histories,
                build_selected_vrc_tube=dataset == "sfm",
            )
            active_vrc_tube = plan.selected_vrc_tube

            control_dyn = plan.selected_controls[0].unsqueeze(0)
            state = unicycle_dynamics(state, control_dyn, DT)
            state_hist[:, step + 1] = state.cpu()
            control_hist[:, step] = control_dyn.cpu()
            total_time += time.time() - time_start

            if step == HORIZON - 1:
                break

            selected_cfm_controls = plan.selected_cfm_controls
            history_length = len(histories["ego_control_sin"])
            noise = torch.randn(
                N_CFM_SAMPLES,
                selected_cfm_controls.shape[1],
                selected_cfm_controls.shape[2],
                device=device,
            )
            x_t = (
                noise_level * selected_cfm_controls / SPACE_SCALE
                + (1.0 - noise_level) * noise
            )
            x_t = x_t[:, :, history_length + 1 :]

            histories["ego_control_sin"].update(
                selected_cfm_controls[:, :, history_length]
            )
            histories["ego_state"].update(state)
            histories["obs_state"].update(current_positions)
            histories["obs_control"].update(current_velocities)

            control_history = histories["ego_control_sin"].get()
            x_t = torch.cat(
                [
                    control_history.expand(N_CFM_SAMPLES, -1, -1)
                    / SPACE_SCALE,
                    x_t,
                ],
                dim=-1,
            )

        episode_metrics = compute_episode_metrics(
            states=state_hist[:, 1:],
            obstacle_positions=pos_obs.squeeze(0).detach().cpu(),
            goal=goal.squeeze().detach().cpu(),
            collision_radius=SAFE_MARGIN,
        )
        freezing_metrics = compute_freezing_metrics(
            states=state_hist[:, :-1],
            linear_speeds=control_hist[0],
            goal=goal.squeeze().detach().cpu(),
            dt=DT,
            minimum_duration=FREEZING_MINIMUM_DURATION,
            speed_threshold=FREEZING_SPEED_THRESHOLD,
            goal_distance_threshold=FREEZING_GOAL_DISTANCE_THRESHOLD,
        )
        all_average_times.append(total_time / HORIZON)
        episodes.append(episode_metrics)
        freezing_episodes.append(freezing_metrics)
        state_trajectories[scenario_index] = state_hist
        control_trajectories[scenario_index] = control_hist
        print(scenario_index, flush=True)

    all_average_times = torch.as_tensor(all_average_times)
    mean_time = torch.mean(all_average_times)
    var_time = torch.var(all_average_times)
    summary = summarize_metrics(episodes)
    freezing_summary = summarize_freezing_metrics(freezing_episodes)

    directory_path = Path(f"./results/{dataset}_uni")
    directory_path.mkdir(parents=True, exist_ok=True)
    filename = directory_path / "cfm_vrc_mppi.txt"
    with open(filename, "w") as file:
        file.write("===== SIMULATION RESULTS =====\n\n")
        file.write(f"Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("===== HYPERPARAMETERS =====\n")
        file.write(f"SAFE_MARGIN: {SAFE_MARGIN}\n")
        file.write(f"SAFE_COEF: {SAFE_COEF}\n")
        file.write(f"GOAL_COEF: {GOAL_COEF}\n")
        file.write(
            f"FREEZING_MINIMUM_DURATION: {FREEZING_MINIMUM_DURATION}\n"
        )
        file.write(f"FREEZING_SPEED_THRESHOLD: {FREEZING_SPEED_THRESHOLD}\n")
        file.write(
            "FREEZING_GOAL_DISTANCE_THRESHOLD: "
            f"{FREEZING_GOAL_DISTANCE_THRESHOLD}\n"
        )
        file.write(f"N_BRANCHES: {N_BRANCHES}\n")
        file.write(f"N_MPPI_SAMPLES_PER_BRANCH: {N_MPPI_SAMPLES}\n")
        file.write(f"MPPI_SIGMA: {MPPI_SIGMA}\n")
        file.write(f"MPPI_LAMBDA: {MPPI_LAMBDA}\n")
        file.write(f"VRC_PARAMS: {VRC_PARAMS}\n")
        file.write(f"ROBOT_FORCE_PARAMS: {ROBOT_FORCE_PARAMS}\n")
        file.write("===== SUMMARY STATISTICS =====\n")
        file.write(
            f"Average Time:\n  Mean: {mean_time:.4f}\n"
            f"  Variance: {var_time:.6f}\n\n"
        )
        file.write(
            f"Collision Rate:\n  {summary.collision_rate_percent:.4f}\n\n"
        )
        file.write(
            "Freezing Rate (episodes with at least one event):\n  "
            f"{freezing_summary.freezing_rate_percent:.4f}\n"
            f"Total Freezing Events:\n  {freezing_summary.total_event_count}\n\n"
        )
        file.write(
            f"Distance:\n  Mean: {summary.mean_final_goal_distance:.4f}\n"
            "  Variance: "
            f"{summary.variance_final_goal_distance:.6f}\n\n"
        )
        file.write("===== DETAILED RESULTS =====\n")
        for result_index, episode in enumerate(episodes):
            file.write(f"{result_index + 1}\t" f"{episode.collision:.4f}\n")


if __name__ == "__main__":
    main()
