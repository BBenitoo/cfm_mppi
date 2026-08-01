import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from cfm_mppi.reward import single_cbf_reward_fn_pairwise, single_goal_reward_fn

@dataclass
class CFMConfig:
    ode_times: List[float]
    dt: float = 0.1
    agent_radius: float = 0.5
    space_scale: float = 10.0
    safe_margin_coefs: Optional[List[float]] = None
    goal_margin_coef: float = 0.1
    device: str = 'cuda'


@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics computed from one completed evaluation episode."""

    collision: torch.Tensor
    final_goal_distance: torch.Tensor


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate non-timing metrics across evaluation episodes."""

    collision_rate_percent: torch.Tensor
    mean_final_goal_distance: torch.Tensor
    variance_final_goal_distance: torch.Tensor


@dataclass(frozen=True)
class FreezingMetrics:
    """Freezing events detected in one completed episode."""

    event_count: torch.Tensor

    @property
    def occurred(self) -> torch.Tensor:
        """Whether at least one freezing event occurred in the episode."""
        return self.event_count > 0


@dataclass(frozen=True)
class FreezingSummary:
    """Aggregate freezing metrics across evaluation episodes."""

    freezing_rate_percent: torch.Tensor
    total_event_count: torch.Tensor


def compute_episode_metrics(
    states: torch.Tensor,
    obstacle_positions: torch.Tensor,
    goal: torch.Tensor,
    collision_radius: float,
) -> EpisodeMetrics:
    """Compute collision and final-goal-distance metrics for one episode.

    Args:
        states: Robot states with shape ``[state_dim, time]``.
        obstacle_positions: Obstacle states with shape
            ``[num_obstacles, obstacle_dim, time]``.
        goal: Goal position with shape ``[goal_dim]``.
        collision_radius: Strict center-distance collision threshold. A distance
            exactly equal to this value is not counted as a collision.

    Returns:
        Scalar tensors on the same device as the inputs.
    """
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] == 0:
        raise ValueError(
            "states must have shape [state_dim >= 2, time > 0]."
        )
    if obstacle_positions.ndim != 3 or obstacle_positions.shape[1] < 2:
        raise ValueError(
            "obstacle_positions must have shape "
            "[num_obstacles, obstacle_dim >= 2, time]."
        )
    if obstacle_positions.shape[-1] != states.shape[-1]:
        raise ValueError(
            "states and obstacle_positions must have the same time dimension."
        )
    if goal.ndim != 1 or goal.numel() < 2:
        raise ValueError("goal must have shape [goal_dim >= 2].")
    if collision_radius < 0:
        raise ValueError("collision_radius must be non-negative.")

    obstacle_distances = torch.linalg.vector_norm(
        states[:2].unsqueeze(0) - obstacle_positions[:, :2],
        dim=1,
    )
    collision = torch.any(obstacle_distances < collision_radius)
    final_goal_distance = torch.linalg.vector_norm(states[:2, -1] - goal[:2])
    return EpisodeMetrics(
        collision=collision,
        final_goal_distance=final_goal_distance,
    )


def summarize_metrics(episodes: Sequence[EpisodeMetrics]) -> EvaluationSummary:
    """Aggregate completed episode metrics using population variance."""
    if not episodes:
        raise ValueError("At least one episode is required to summarize metrics.")

    collisions = torch.stack([episode.collision for episode in episodes]).float()
    final_goal_distances = torch.stack(
        [episode.final_goal_distance for episode in episodes]
    )
    return EvaluationSummary(
        collision_rate_percent=collisions.mean() * 100.0,
        mean_final_goal_distance=final_goal_distances.mean(),
        variance_final_goal_distance=final_goal_distances.var(correction=0),
    )


def compute_freezing_metrics(
    states: torch.Tensor,
    linear_speeds: torch.Tensor,
    goal: torch.Tensor,
    dt: float,
    minimum_duration: float = 1.0,
    speed_threshold: float = 0.05,
    goal_distance_threshold: float = 0.5,
) -> FreezingMetrics:
    """Count maximal continuous freezing intervals in one episode.

    A freezing event is a maximal interval lasting at least
    ``minimum_duration`` where the robot's absolute linear speed is strictly
    below ``speed_threshold`` and its distance to the goal is strictly above
    ``goal_distance_threshold``. Long intervals count once rather than once per
    overlapping time window.

    Args:
        states: Robot states with shape ``[state_dim, time]``.
        linear_speeds: Signed linear speeds in metres per second with shape
            ``[time]``. The absolute value is used.
        goal: Goal position with shape ``[goal_dim]``.
        dt: Duration represented by each state/control sample in seconds.
        minimum_duration: Minimum continuous low-speed duration in seconds.
        speed_threshold: Strict low-speed threshold in metres per second.
        goal_distance_threshold: Strict distance threshold in metres used to
            exclude normal stopping at the goal.
    """
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] == 0:
        raise ValueError(
            "states must have shape [state_dim >= 2, time > 0]."
        )
    if linear_speeds.ndim != 1:
        raise ValueError("linear_speeds must have shape [time].")
    if linear_speeds.shape[0] != states.shape[-1]:
        raise ValueError("states and linear_speeds must have the same time dimension.")
    if goal.ndim != 1 or goal.numel() < 2:
        raise ValueError("goal must have shape [goal_dim >= 2].")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if minimum_duration <= 0:
        raise ValueError("minimum_duration must be positive.")
    if speed_threshold < 0:
        raise ValueError("speed_threshold must be non-negative.")
    if goal_distance_threshold < 0:
        raise ValueError("goal_distance_threshold must be non-negative.")

    goal_distances = torch.linalg.vector_norm(
        states[:2] - goal[:2].unsqueeze(-1),
        dim=0,
    )
    freezing_samples = (linear_speeds.abs() < speed_threshold) & (
        goal_distances > goal_distance_threshold
    )

    boundary = torch.zeros(1, dtype=torch.bool, device=freezing_samples.device)
    padded_samples = torch.cat([boundary, freezing_samples, boundary])
    transitions = padded_samples[1:].to(torch.int8) - padded_samples[:-1].to(
        torch.int8
    )
    run_starts = torch.nonzero(transitions == 1, as_tuple=False).flatten()
    run_ends = torch.nonzero(transitions == -1, as_tuple=False).flatten()
    minimum_samples = math.ceil(minimum_duration / dt)
    event_count = ((run_ends - run_starts) >= minimum_samples).sum()
    return FreezingMetrics(event_count=event_count)


def summarize_freezing_metrics(
    episodes: Sequence[FreezingMetrics],
) -> FreezingSummary:
    """Return the percentage of episodes containing at least one freeze."""
    if not episodes:
        raise ValueError(
            "At least one episode is required to summarize freezing metrics."
        )

    event_counts = torch.stack([episode.event_count for episode in episodes])
    return FreezingSummary(
        freezing_rate_percent=(event_counts > 0).float().mean() * 100.0,
        total_event_count=event_counts.sum(),
    )


def run_CFM(model, config: CFMConfig, noisy_action_seq, noise_level, start_pos, goal_pos, obs_positions, obs_velocities, control_history=None):
    start_pos = start_pos / config.space_scale
    goal_pos = goal_pos / config.space_scale
    obs_positions = obs_positions / config.space_scale
    obs_velocities = obs_velocities / config.space_scale
    rad = config.agent_radius / config.space_scale

    if control_history is not None:
        control_history = control_history / config.space_scale
        
    start_batch = start_pos.repeat(noisy_action_seq.shape[0], 1)
    goal_batch = goal_pos.repeat(noisy_action_seq.shape[0], 1)
    obs_positions = obs_positions.squeeze(0)
    obs_velocities = obs_velocities.squeeze(0)
    batch_size = noisy_action_seq.shape[0]

    if config.safe_margin_coefs is not None:
        num_coef = len(config.safe_margin_coefs)
        size_coef = batch_size // num_coef
        safe_coef = torch.zeros(batch_size, 1, 1, device=noisy_action_seq.device)
        for i in range(num_coef):
            safe_coef[size_coef*i:size_coef*(i+1), 0, 0] = config.safe_margin_coefs[i]

    single_cbf_grad_fn = torch.func.grad(single_cbf_reward_fn_pairwise)
    single_goal_grad_fn = torch.func.grad(single_goal_reward_fn)

    batched_cbf_grad_fn = torch.vmap(single_cbf_grad_fn, in_dims=(0, None, None, None))
    batched_goal_grad_fn = torch.vmap(single_goal_grad_fn, in_dims=(0, None))


    for j in range(len(config.ode_times)):
        if control_history is not None:
            noisy_action_seq[:,:,:control_history.shape[-1]] = control_history
        t_next = torch.tensor([config.ode_times[j]], device=config.device)
        
        u_t_pred = model(noisy_action_seq, noise_level, start=start_batch, goal=goal_batch)
        x_1_pred = noisy_action_seq + (1-noise_level)*u_t_pred
        
        if control_history is not None:
            x_1_pred[:,:,:control_history.shape[-1]] = control_history

        grad_cbf = batched_cbf_grad_fn(x_1_pred, obs_positions, obs_velocities, rad)
        grad_goal = batched_goal_grad_fn(x_1_pred, goal_pos.squeeze(0))

        if control_history is not None:
            mask = torch.ones_like(grad_cbf, device=config.device)
            mask[:,:,:control_history.shape[-1]] = 0
            grad_cbf = grad_cbf * mask
            grad_goal = grad_goal * mask


        u_norm = torch.norm(u_t_pred, keepdim=True)
        grad_cbf_norm = torch.norm(grad_cbf, keepdim=True)
        grad_goal_norm = torch.norm(grad_goal, keepdim=True)

        normalized_grad_cbf = grad_cbf * u_norm / (grad_cbf_norm+1e-8)
        normalized_grad_goal = grad_goal * u_norm / (grad_goal_norm+1e-8)

        markup = 1.01**torch.arange(0, noisy_action_seq.shape[-1], device=config.device).flip(0).unsqueeze(0).unsqueeze(0)

        u_t_pred_new = u_t_pred + config.goal_margin_coef * normalized_grad_goal + safe_coef * normalized_grad_cbf * markup
        noisy_action_seq = noisy_action_seq + (t_next.reshape(-1,1,1)-noise_level.reshape(-1,1,1))*u_t_pred_new
        noise_level = t_next

    return noisy_action_seq * config.space_scale


def synthesize_control(
    model, 
    mppi_solver, 
    config: CFMConfig, 
    ego_state, 
    goal_pos, 
    noisy_action_seq, 
    noise_level, 
    obs_positions, 
    obs_velocities, 
    planning_horizon, 
    histories: dict = None, 
    **mppi_kwargs
):
    """
    Synthesizes control for the ego vehicle using CFM.
    ego_state: [1, dim_state]
    goal_pos: [1, 2]
    obs_positions: [1, num_obst, 2]
    obs_velocities: [1, num_obst, 2]
    planning_horizon: scalar
    """
    if histories is None:
        histories = {}
        
    state_hist = histories.get('ego_state')
    control_hist_sin = histories.get('ego_control_sin')
    obs_state_hist = histories.get('obs_state')
    obs_control_hist = histories.get('obs_control')
    
    # Extract data tensors from history wrapper objects
    state_history = state_hist.get() if state_hist else None
    control_history_sin = control_hist_sin.get() if control_hist_sin else None
    obstacle_state_history = obs_state_hist.get() if obs_state_hist else None
    obstacle_control_history = obs_control_hist.get() if obs_control_hist else None
    
    history_len = len(state_hist) if state_hist else 0
    
    if control_history_sin is not None:
        vel_obs_seq = obs_velocities.unsqueeze(-1).repeat(1, 1, 1, planning_horizon - history_len)
        pos_obs_seq = obs_positions.unsqueeze(-1) + torch.cumsum(vel_obs_seq * config.dt, dim=3)
        pos_obs_seq = torch.cat([obstacle_state_history, pos_obs_seq], dim=3)
        vel_obs_seq = torch.cat([obstacle_control_history, vel_obs_seq], dim=3)
        goal_cfm = goal_pos - state_history[:, :2, 0]
        pos_obs_seq_cfm = pos_obs_seq - state_history[:, :2, 0].unsqueeze(-1)
    else:
        vel_obs_seq = obs_velocities.unsqueeze(-1).repeat(1, 1, 1, planning_horizon)
        pos_obs_seq = obs_positions.unsqueeze(-1) + torch.cumsum(vel_obs_seq * config.dt, dim=3)
        goal_cfm = goal_pos - ego_state[:, :2]
        pos_obs_seq_cfm = pos_obs_seq - ego_state[:, :2].unsqueeze(1).unsqueeze(-1)
    
    # [n_samples, 2, planning_horizon]
    controls_sin = run_CFM(
        model, 
        config, 
        noisy_action_seq, 
        noise_level, 
        torch.zeros(1, 2, device=config.device), 
        goal_cfm, 
        pos_obs_seq_cfm, 
        vel_obs_seq, 
        control_history_sin
    ).detach()
    
    with torch.no_grad():
        x_dyn, x_sin = mppi_solver.forward(ego_state, controls_sin[:,:,history_len:].transpose(1,2),
                                    planning_horizon-history_len, goal_pos, pos_obs_seq[:,:,:,history_len:].squeeze(0).transpose(1,2), config.agent_radius,
                                    **mppi_kwargs)
        x_dyn = x_dyn.unsqueeze(0).transpose(1,2) #[1, 2, planning_horizon-history_len]
        x_sin = x_sin.unsqueeze(0).transpose(1,2) #[1, 2, planning_horizon-history_len]
        x_sin = torch.cat([control_history_sin, x_sin], dim=2) if control_history_sin is not None else x_sin
        
    return x_dyn, x_sin
