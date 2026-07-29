from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import pickle
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from cfm_mppi.evaluation.eval_utils import CFMConfig, run_CFM
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
    build_vrc_tube,
    force_from_vrc_tube,
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

VRC_PARAMS = VRCParameters()
ROBOT_FORCE_PARAMS = RobotForceParameters()


class SegmentProfiler:
    """Collect synchronized wall-clock timings for mixed CPU/CUDA stages."""

    def __init__(self) -> None:
        self.enabled = False
        self.device: torch.device | None = None
        self.samples: defaultdict[str, list[float]] = defaultdict(list)

    def _synchronize(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return

        # CUDA kernels are asynchronous. Synchronizing at both boundaries keeps
        # work from adjacent stages out of this stage's wall-clock measurement.
        self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.samples[name].append(elapsed_ms)

    def report(self) -> None:
        reference_steps = len(self.samples.get("01_cfm", []))
        if reference_steps == 0:
            print("No profiling samples were collected.", flush=True)
            return

        total_measured = sum(
            sum(stage_samples) for stage_samples in self.samples.values()
        )
        print("\n========== SEGMENT PERFORMANCE ==========", flush=True)
        print(
            f"{'stage':34s}"
            f"{'ms/plan step':>15s}"
            f"{'share':>10s}"
            f"{'p50/call':>12s}"
            f"{'p95/call':>12s}"
            f"{'calls':>8s}",
            flush=True,
        )
        for name in sorted(self.samples):
            values = np.asarray(self.samples[name], dtype=np.float64)
            stage_total = values.sum()
            print(
                f"{name:34s}"
                f"{stage_total / reference_steps:15.3f}"
                f"{100.0 * stage_total / total_measured:9.1f}%"
                f"{np.percentile(values, 50):12.3f}"
                f"{np.percentile(values, 95):12.3f}"
                f"{len(values):8d}",
                flush=True,
            )
        print(f"Profiled planning steps: {reference_steps}", flush=True)
        print("=========================================\n", flush=True)


SEGMENT_PROFILER = SegmentProfiler()


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
    pedestrian_predictions: torch.Tensor
    mppi_branch_states: torch.Tensor
    branch_costs: torch.Tensor
    selected_vrc_tube: list[VRCEllipse]


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
    """Build one VRC and one pedestrian prediction for every CFM branch."""
    horizon = branch_controls.shape[1]
    predictions = []
    for branch_index in range(branch_controls.shape[0]):
        vrc_tube = build_vrc_tube(
            states=branch_states[branch_index],
            controls_uni=branch_controls[branch_index],
            params=vrc_params,
        )
        predictions.append(
            predict_pedestrians_with_vrc(
                current_positions=current_positions,
                current_velocities=current_velocities,
                vrc_tube=vrc_tube,
                horizon=horizon,
                dt=dt,
                force_params=force_params,
                prediction_params=prediction_params,
            )
        )
    return torch.stack(predictions, dim=0)


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
    vrc_params: VRCParameters = VRC_PARAMS,
    force_params: RobotForceParameters = ROBOT_FORCE_PARAMS,
    prediction_params: PedestrianPredictionParameters | None = None,
) -> BranchPlan:
    """Execute CFM -> VRC -> pedestrian prediction -> branch MPPI."""
    if prediction_params is None:
        prediction_params = PedestrianPredictionParameters()

    with torch.no_grad():
        with SEGMENT_PROFILER.track("01_cfm"):
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
        with SEGMENT_PROFILER.track("02_candidate_rollout_and_rank"):
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

    with SEGMENT_PROFILER.track("03_vrc_and_pedestrian_prediction"):
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
        with SEGMENT_PROFILER.track("04_branch_mppi"):
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

    with SEGMENT_PROFILER.track("05_final_selection_and_vrc"):
        selected_branch = int(torch.argmin(branch_costs).item())
        selected_cfm_index = int(branch_indices[selected_branch].item())
        selected_vrc_tube = build_vrc_tube(
            states=mppi_branch_states[selected_branch],
            controls_uni=selected_controls,
            params=vrc_params,
        )
    return BranchPlan(
        selected_controls=selected_controls,
        selected_cfm_controls=cfm_controls[selected_cfm_index : selected_cfm_index + 1],
        selected_branch=selected_branch,
        cfm_branch_indices=branch_indices,
        cfm_branch_states=branch_states,
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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect synchronized per-stage CPU/CUDA timings.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help=(
            "Stop after the profiled scenarios and skip writing evaluation results. "
            "This option also enables profiling."
        ),
    )
    parser.add_argument(
        "--profile-scenarios",
        type=int,
        default=1,
        help="Number of initial scenarios to profile (default: 1).",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=int,
        default=10,
        help="Closed-loop steps to skip before collecting timings (default: 10).",
    )
    args = parser.parse_args()
    if args.profile_scenarios < 1:
        parser.error("--profile-scenarios must be at least 1")
    if not 0 <= args.profile_warmup_steps < HORIZON:
        parser.error(f"--profile-warmup-steps must be in [0, {HORIZON - 1}]")
    return args


def main() -> None:
    from cfm_mppi.utils import AgentHistory, HumanAgent, evaluate

    args = parse_args()
    dataset = args.dataset
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile_enabled = args.profile or args.profile_only
    SEGMENT_PROFILER.device = device
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
    all_collisions = []
    all_distances = []
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
            SEGMENT_PROFILER.enabled = (
                profile_enabled
                and scenario_index < args.profile_scenarios
                and step >= args.profile_warmup_steps
            )
            if (
                SEGMENT_PROFILER.enabled
                and scenario_index == 0
                and step == args.profile_warmup_steps
                and device.type == "cuda"
            ):
                torch.cuda.reset_peak_memory_stats(device)

            if dataset == "sfm":
                with SEGMENT_PROFILER.track("00_sfm_environment"):
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
            )
            active_vrc_tube = plan.selected_vrc_tube

            with SEGMENT_PROFILER.track("06_execute_and_cpu_copy"):
                control_dyn = plan.selected_controls[0].unsqueeze(0)
                state = unicycle_dynamics(state, control_dyn, DT)
                state_hist[:, step + 1] = state.cpu()
                control_hist[:, step] = control_dyn.cpu()
            total_time += time.time() - time_start

            if step == HORIZON - 1:
                break

            with SEGMENT_PROFILER.track("07_warm_start_update"):
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

        collision, distance = evaluate(
            state_hist[:, 1:],
            control_hist,
            pos_obs.squeeze(0).detach().cpu(),
            goal.squeeze().detach().cpu(),
            SAFE_MARGIN,
        )
        all_average_times.append(total_time / HORIZON)
        all_collisions.append(collision)
        all_distances.append(distance)
        state_trajectories[scenario_index] = state_hist
        control_trajectories[scenario_index] = control_hist
        print(scenario_index, flush=True)

        if profile_enabled and scenario_index + 1 == args.profile_scenarios:
            SEGMENT_PROFILER.report()
            if device.type == "cuda":
                print(
                    "Peak allocated CUDA memory: "
                    f"{torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB",
                    flush=True,
                )
                print(
                    "Peak reserved CUDA memory: "
                    f"{torch.cuda.max_memory_reserved(device) / 1024**2:.1f} MiB",
                    flush=True,
                )
            if args.profile_only:
                return

    all_average_times = torch.as_tensor(all_average_times)
    all_collisions = torch.as_tensor(all_collisions).float()
    all_distances = torch.as_tensor(all_distances)
    mean_time = torch.mean(all_average_times)
    var_time = torch.var(all_average_times)
    collision_rate = torch.mean(all_collisions) * 100
    mean_distance = torch.mean(all_distances)
    var_distance = torch.var(all_distances)

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
        file.write(f"Collision Rate:\n  {collision_rate:.4f}\n\n")
        file.write(
            f"Distance:\n  Mean: {mean_distance:.4f}\n"
            f"  Variance: {var_distance:.6f}\n\n"
        )
        file.write("===== DETAILED RESULTS =====\n")
        for result_index in range(all_collisions.shape[0]):
            file.write(f"{result_index + 1}\t" f"{all_collisions[result_index]:.4f}\n")


if __name__ == "__main__":
    main()
