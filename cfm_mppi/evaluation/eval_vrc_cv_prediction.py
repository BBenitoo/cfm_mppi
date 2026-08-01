"""Evaluate constant-velocity planning with VRC-reactive pedestrians.

This ablation keeps the CFM branch selection and branch-wise MPPI structure from
``eval_vrc.py``, but every MPPI branch scores against the same constant-velocity
pedestrian prediction.  The selected robot plan still creates a VRC tube that is
used by the next real SFM pedestrian update.  Consequently, pedestrian reactions
affect future observations but are not anticipated inside the current plan.

The planning/response coupling interpretation applies directly to the ``sfm``
dataset.  ``ucy`` and ``sdd`` use recorded pedestrian trajectories and therefore
only compare the planner-side prediction models.
"""

from __future__ import annotations

import argparse
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
    summarize_freezing_metrics,
    summarize_metrics,
)
from cfm_mppi.evaluation.eval_vrc import (
    DT,
    FREEZING_GOAL_DISTANCE_THRESHOLD,
    FREEZING_MINIMUM_DURATION,
    FREEZING_SPEED_THRESHOLD,
    GOAL_COEF,
    HORIZON,
    LOOK_AHEAD_DISTANCE,
    MAX_HISTORY_LENGTH,
    MPPI_LAMBDA,
    MPPI_SIGMA,
    NOISE_LEVEL_VALUE,
    N_BRANCHES,
    N_CFM_SAMPLES,
    N_MPPI_SAMPLES,
    ODE_TIMES,
    ODE_TIMES_WARM,
    ROBOT_FORCE_PARAMS,
    SAFE_COEF,
    SAFE_MARGIN,
    SPACE_SCALE,
    U_MAX,
    U_MIN,
    VRC_PARAMS,
    BranchPlan,
    generate_cfm_candidates,
    select_cfm_branches,
    update_sfm_environment,
)
from cfm_mppi.models.transformer import TransformerModel
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import stage_cost, terminal_cost, unicycle_dynamics
from cfm_mppi.vrc.build_vrc import VRCParameters, build_vrc_tube

if TYPE_CHECKING:
    from cfm_mppi.utils import AgentHistory


def plan_with_constant_velocity_prediction(
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
    vrc_params: VRCParameters = VRC_PARAMS,
    build_selected_vrc_tube: bool = True,
) -> BranchPlan:
    """Plan against constant-velocity pedestrians, then build an environment VRC.

    The selected VRC tube is an output only: it is never used to construct the
    pedestrian trajectories passed to MPPI in this planning call.
    """
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
            look_ahead_distance=LOOK_AHEAD_DISTANCE,
        )

        # Every robot branch sees the same CV pedestrian future.  Keeping the
        # branch dimension preserves eval_vrc's MPPI sampling structure while
        # removing branch-conditioned/VRC-conditioned pedestrian prediction.
        branch_obstacle_predictions = (
            constant_velocity_obstacles.unsqueeze(0)
            .expand(branch_controls.shape[0], -1, -1, -1)
            .contiguous()
        )
        (
            selected_controls,
            _,
            mppi_branch_states,
            branch_costs,
        ) = solver.forward_branches(
            state=state,
            branch_controls=branch_controls,
            goal=goal,
            branch_obstacle_states=branch_obstacle_predictions,
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
        selected_cfm_controls=cfm_controls[
            selected_cfm_index : selected_cfm_index + 1
        ],
        selected_branch=selected_branch,
        cfm_branch_indices=branch_indices,
        cfm_branch_states=branch_states,
        pedestrian_predictions=branch_obstacle_predictions,
        mppi_branch_states=mppi_branch_states,
        branch_costs=branch_costs,
        selected_vrc_tube=selected_vrc_tube,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate constant-velocity MPPI prediction with VRC-reactive "
            "pedestrian updates."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="sfm",
        choices=("ucy", "sdd", "sfm"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cfm_mppi.utils import AgentHistory, HumanAgent

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
        batch_ego = torch.load(
            f"../../dataset/eval80_ego_{dataset}.pt",
            map_location="cpu",
        )
        with open(f"../../dataset/eval80_obs_{dataset}.pkl", "rb") as file:
            batch_obs = pickle.load(file)
    else:
        batch_ego = torch.zeros(300)

    all_average_times = []
    episodes: list[EpisodeMetrics] = []
    freezing_episodes: list[FreezingMetrics] = []
    state_trajectories = torch.zeros(
        [batch_ego.shape[0], 3, HORIZON + 1],
        dtype=torch.float32,
    )
    control_trajectories = torch.zeros(
        [batch_ego.shape[0], 2, HORIZON],
        dtype=torch.float32,
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
                    human.state,
                    device=device,
                )
                vel_obs[0, human_index, :, 0] = torch.as_tensor(
                    human.control,
                    device=device,
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
            plan = plan_with_constant_velocity_prediction(
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
    filename = directory_path / "cfm_cv_mppi_vrc_response.txt"
    environment_response = (
        "VRC-conditioned SFM"
        if dataset == "sfm"
        else "recorded trajectory (no online VRC response)"
    )
    with open(filename, "w") as file:
        file.write("===== SIMULATION RESULTS =====\n\n")
        file.write(f"Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("PLANNING_PEDESTRIAN_MODEL: constant_velocity\n")
        file.write(f"ENVIRONMENT_PEDESTRIAN_RESPONSE: {environment_response}\n")
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
