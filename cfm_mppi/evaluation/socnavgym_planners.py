"""CFM/MPPI controllers for the shared SocNavGym evaluation runner.

Both controllers in this module use exactly the same CFM candidate count,
branch count, MPPI samples per branch, physical control bounds, and robot
dynamics.  Their only planning difference is the pedestrian prediction passed
to branch-local MPPI: the baseline broadcasts one constant-velocity prediction,
whereas the VRC controller builds a prediction conditioned on each robot branch.

The controllers never advance the real environment.  History is committed only
after :class:`~cfm_mppi.evaluation.socnavgym_adapter.SocNavStep` supplies the
next simulator state, and the warm-start tensor retains a fixed total horizon
even after the finite history buffer becomes full.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator

import numpy as np
import torch

from cfm_mppi.evaluation.eval_utils import CFMConfig
from cfm_mppi.evaluation.eval_vrc import (
    PedestrianPredictionParameters,
    constant_velocity_prediction,
    plan_vrc_branches,
)
from cfm_mppi.evaluation.eval_vrc_cv_prediction import (
    plan_with_constant_velocity_prediction,
)
from cfm_mppi.evaluation.socnavgym_adapter import SocNavState, SocNavStep
from cfm_mppi.evaluation.socnavgym_runner import (
    EpisodeContext,
    PlannerCommand,
    PlanningBudget,
)
from cfm_mppi.mppi.flowmppi import FlowMPPI
from cfm_mppi.mppi.utils import stage_cost, terminal_cost
from cfm_mppi.vrc.build_vrc import (
    RobotForceParameters,
    VRCParameters,
    build_tensor_vrc_tube,
    force_from_tensor_vrc_tube,
)


VISUALIZATION_TRACE_SCHEMA = "cfm_mppi.socnavgym_visualization_trace.v1"


@dataclass(frozen=True)
class SocNavPlannerConfig:
    """Shared algorithmic and numerical configuration for both planners."""

    horizon: int = 80
    max_history: int = 10
    cfm_candidates: int = 200
    branches: int = 10
    mppi_samples_per_branch: int = 200
    mppi_sigma: tuple[float, float] = (0.3, 0.6)
    mppi_lambda: float = 0.1
    space_scale: float = 10.0
    look_ahead_distance: float = 0.1
    warm_noise_level: float = 0.8
    ode_times_initial: tuple[float, ...] = (
        0.5,
        0.8,
        0.85,
        0.9,
        0.92,
        0.94,
        0.96,
        0.98,
        1.0,
    )
    ode_times_warm: tuple[float, ...] = (
        0.85,
        0.9,
        0.92,
        0.94,
        0.96,
        0.98,
        1.0,
    )
    safe_margin_coefs: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    goal_margin_coef: float = 0.1
    extra_clearance: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.max_history <= 0 or self.max_history >= self.horizon:
            raise ValueError("max_history must be in [1, horizon)")
        if self.cfm_candidates <= 0:
            raise ValueError("cfm_candidates must be positive")
        if self.branches <= 0 or self.branches > self.cfm_candidates:
            raise ValueError("branches must be in [1, cfm_candidates]")
        if self.mppi_samples_per_branch <= 0:
            raise ValueError("mppi_samples_per_branch must be positive")
        if len(self.mppi_sigma) != 2 or any(value <= 0 for value in self.mppi_sigma):
            raise ValueError("mppi_sigma must contain two positive values")
        if self.mppi_lambda <= 0:
            raise ValueError("mppi_lambda must be positive")
        if self.space_scale <= 0:
            raise ValueError("space_scale must be positive")
        if self.look_ahead_distance <= 0:
            raise ValueError("look_ahead_distance must be positive")
        if not 0.0 <= self.warm_noise_level <= 1.0:
            raise ValueError("warm_noise_level must lie in [0, 1]")
        if not self.ode_times_initial or not self.ode_times_warm:
            raise ValueError("both ODE time schedules must be non-empty")
        if not self.safe_margin_coefs:
            raise ValueError("safe_margin_coefs must be non-empty")
        if self.cfm_candidates % len(self.safe_margin_coefs) != 0:
            raise ValueError(
                "cfm_candidates must be divisible by the safe-margin group count"
            )
        if self.extra_clearance < 0:
            raise ValueError("extra_clearance must be non-negative")

    @property
    def refinement_rollouts(self) -> int:
        """Total perturbed MPPI trajectories evaluated per decision."""
        return self.branches * self.mppi_samples_per_branch

    @property
    def budget(self) -> PlanningBudget:
        return PlanningBudget(
            cfm_candidates=self.cfm_candidates,
            refinement_rollouts=self.refinement_rollouts,
        )


def socnav_diff_drive_dynamics(
    state: torch.Tensor,
    action: torch.Tensor,
    delta_t: float,
) -> torch.Tensor:
    """Apply SocNavGym v1's rotate-first differential-drive dynamics.

    SocNavGym updates the heading before translating the robot.  This differs
    from the legacy project dynamics, which translates using the old heading.
    The function is batched and preserves the input device and dtype.
    """
    if state.ndim != 2 or state.shape[-1] != 3:
        raise ValueError("state must have shape [batch, 3]")
    if action.ndim != 2 or action.shape != (state.shape[0], 2):
        raise ValueError("action must have shape [batch, 2]")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")

    next_heading_unwrapped = state[:, 2] + action[:, 1] * delta_t
    next_heading = torch.atan2(
        torch.sin(next_heading_unwrapped),
        torch.cos(next_heading_unwrapped),
    )
    displacement = action[:, 0] * delta_t
    return torch.stack(
        (
            state[:, 0] + displacement * torch.cos(next_heading),
            state[:, 1] + displacement * torch.sin(next_heading),
            next_heading,
        ),
        dim=1,
    )


class _TensorHistory:
    """Minimal trailing-time history implementing the existing planner API."""

    def __init__(self, max_length: int) -> None:
        self.max_length = max_length
        self.data: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        item = value.detach().unsqueeze(-1)
        if self.data is None:
            self.data = item
        elif len(self) < self.max_length:
            self.data = torch.cat((self.data, item), dim=-1)
        else:
            self.data = torch.cat((self.data[..., 1:], item), dim=-1)

    def get(self) -> torch.Tensor | None:
        return self.data

    def __len__(self) -> int:
        return 0 if self.data is None else self.data.shape[-1]


@dataclass
class _PendingPlan:
    command: PlannerCommand
    state: SocNavState
    selected_cfm_controls: torch.Tensor
    history_length: int


class _SocNavCFMPlannerBase:
    """Common closed-loop state and sampling logic for the two planners."""

    name = "socnav-cfm-base"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: SocNavPlannerConfig | None = None,
        device: str | torch.device | None = None,
        solver_factory: Callable[..., Any] = FlowMPPI,
        record_visualization: bool = False,
    ) -> None:
        if not isinstance(record_visualization, bool):
            raise TypeError("record_visualization must be a boolean")
        self.model = model
        self.config = config or SocNavPlannerConfig()
        self.budget = self.config.budget
        self.device = self._resolve_device(device)
        self._solver_factory = solver_factory
        self.record_visualization = record_visualization
        self._solver: Any | None = None
        self._context: EpisodeContext | None = None
        self._histories: dict[str, _TensorHistory] = {}
        self._warm_start: torch.Tensor | None = None
        self._initial_goal: np.ndarray | None = None
        self._human_ids: tuple[int, ...] | None = None
        self._human_radii: np.ndarray | None = None
        self._robot_radius: float | None = None
        self._collision_radius: float | None = None
        self._pending: _PendingPlan | None = None
        self._cpu_rng_state: torch.Tensor | None = None
        self._cuda_rng_state: torch.Tensor | None = None

        move_model = getattr(self.model, "to", None)
        if callable(move_model):
            move_model(device=self.device)
        evaluate_model = getattr(self.model, "eval", None)
        if callable(evaluate_model):
            evaluate_model()

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device)
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for the planner but is unavailable")
            # FlowMPPI currently recognizes the canonical unindexed CUDA device.
            return torch.device("cuda")
        if requested.type != "cpu":
            raise ValueError(f"unsupported planner device: {requested}")
        return requested

    @property
    def history_length(self) -> int:
        history = self._histories.get("ego_control_sin")
        return 0 if history is None else len(history)

    @property
    def warm_start_shape(self) -> tuple[int, ...] | None:
        return None if self._warm_start is None else tuple(self._warm_start.shape)

    def history_snapshot(self) -> dict[str, torch.Tensor | None]:
        """Return detached history copies for diagnostics and contract tests."""
        return {
            name: None if history.get() is None else history.get().detach().clone()
            for name, history in self._histories.items()
        }

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_episode(
        self,
        initial_state: SocNavState,
        context: EpisodeContext,
    ) -> None:
        self._validate_state_arrays(initial_state)
        if not np.isclose(context.time_step, 0.1):
            # The checkpoint and CFM reward integrate controls at 0.1 seconds.
            raise ValueError(
                "the current CFM checkpoint requires SocNavGym time_step=0.1"
            )

        self._context = context
        self._initial_goal = initial_state.goal.copy()
        self._human_ids = initial_state.human_ids
        self._human_radii = initial_state.human_radii.copy()
        self._robot_radius = float(initial_state.robot_radius)
        maximum_human_radius = (
            float(initial_state.human_radii.max())
            if initial_state.human_radii.size
            else 0.0
        )
        self._collision_radius = (
            initial_state.robot_radius
            + maximum_human_radius
            + self.config.extra_clearance
        )
        self._histories = {
            "ego_state": _TensorHistory(self.config.max_history),
            "ego_control_sin": _TensorHistory(self.config.max_history),
            "obs_state": _TensorHistory(self.config.max_history),
            "obs_control": _TensorHistory(self.config.max_history),
        }
        self._pending = None

        state_tensor, goal, _, _ = self._planner_tensors(initial_state)
        del state_tensor
        control_low = torch.tensor(
            context.control_low,
            device=self.device,
            dtype=torch.float32,
        )
        control_high = torch.tensor(
            context.control_high,
            device=self.device,
            dtype=torch.float32,
        )
        sigma = torch.as_tensor(
            self.config.mppi_sigma,
            device=self.device,
            dtype=torch.float32,
        )

        def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return socnav_diff_drive_dynamics(state, action, context.time_step)

        self._solver = self._solver_factory(
            num_samples=self.config.mppi_samples_per_branch,
            dim_state=3,
            dim_control=2,
            dynamics=dynamics,
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            u_min=control_low,
            u_max=control_high,
            sigmas=sigma,
            lambda_=self.config.mppi_lambda,
            goal=goal.squeeze(0),
            horizon=self.config.horizon,
            dt=context.time_step,
            device=self.device,
            seed=context.planner_seed,
            dynamics_type="unicycle",
        )
        self._initialize_episode_rng(context.planner_seed)

    def _initialize_episode_rng(self, seed: int) -> None:
        devices = self._fork_rng_devices()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed(int(seed))
            self._warm_start = torch.randn(
                self.config.cfm_candidates,
                2,
                self.config.horizon,
                device=self.device,
                dtype=torch.float32,
            )
            self._cpu_rng_state = torch.random.get_rng_state().clone()
            if self.device.type == "cuda":
                self._cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()
            else:
                self._cuda_rng_state = None

    def _fork_rng_devices(self) -> list[int]:
        if self.device.type != "cuda":
            return []
        return [
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        ]

    @contextmanager
    def _episode_rng(self) -> Iterator[None]:
        if self._cpu_rng_state is None:
            raise RuntimeError("reset_episode() must initialize the planner RNG")
        with torch.random.fork_rng(devices=self._fork_rng_devices()):
            torch.random.set_rng_state(self._cpu_rng_state)
            if self.device.type == "cuda":
                if self._cuda_rng_state is None:
                    raise RuntimeError("CUDA planner RNG state is missing")
                torch.cuda.set_rng_state(self._cuda_rng_state, self.device)
            try:
                yield
            finally:
                self._cpu_rng_state = torch.random.get_rng_state().clone()
                if self.device.type == "cuda":
                    self._cuda_rng_state = torch.cuda.get_rng_state(
                        self.device
                    ).clone()

    def plan(self, state: SocNavState, step_index: int) -> PlannerCommand:
        self._assert_ready()
        if self._pending is not None:
            raise RuntimeError(
                "observe_transition() must commit the previous command before plan()"
            )
        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        self._validate_episode_state(state)
        if self._warm_start is None or self._solver is None:
            raise RuntimeError("planner episode state is incomplete")

        history_length = self.history_length
        future_horizon = self.config.horizon - history_length
        if future_horizon <= 0:
            raise RuntimeError("history exhausted the planning horizon")
        current_state, goal, human_positions, human_velocities = (
            self._planner_tensors(state)
        )
        if history_length == 0:
            noise_level = torch.zeros(1, device=self.device)
            ode_times = self.config.ode_times_initial
        else:
            noise_level = torch.tensor(
                [self.config.warm_noise_level],
                device=self.device,
                dtype=torch.float32,
            )
            ode_times = self.config.ode_times_warm
        cfm_config = CFMConfig(
            ode_times=list(ode_times),
            dt=self._context.time_step,
            agent_radius=float(self._collision_radius),
            space_scale=self.config.space_scale,
            safe_margin_coefs=list(self.config.safe_margin_coefs),
            goal_margin_coef=self.config.goal_margin_coef,
            device=self.device,
        )

        with self._episode_rng():
            plan = self._plan_branches(
                state=current_state,
                goal=goal,
                noisy_action_seq=self._warm_start,
                noise_level=noise_level,
                current_positions=human_positions,
                current_velocities=human_velocities,
                planning_horizon=self.config.horizon,
                cfm_config=cfm_config,
                initial_state=state,
            )

        if plan.selected_vrc_tube is not None:
            raise RuntimeError(
                "SocNavGym planners must not construct or expose an environment VRC tube"
            )
        self._validate_plan_shapes(plan, future_horizon)
        control = (
            plan.selected_controls[0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
        )
        branch_indices = plan.cfm_branch_indices.detach().cpu().tolist()
        branch_costs = plan.branch_costs.detach().cpu().tolist()
        diagnostics = {
            "planner_kind": self.name,
            "history_length": history_length,
            "future_horizon": future_horizon,
            "cfm_candidates": self.config.cfm_candidates,
            "branches": self.config.branches,
            "mppi_samples_per_branch": self.config.mppi_samples_per_branch,
            "refinement_rollouts": self.config.refinement_rollouts,
            "selected_branch": int(plan.selected_branch),
            "cfm_branch_indices": branch_indices,
            "branch_costs": branch_costs,
            "selected_vrc_tube": False,
        }
        if self.record_visualization:
            diagnostics["visualization"] = self._build_visualization_trace(
                plan=plan,
                state=state,
                human_positions=human_positions,
                human_velocities=human_velocities,
                future_horizon=future_horizon,
            )
        command = PlannerCommand(control=control, diagnostics=diagnostics)
        self._pending = _PendingPlan(
            command=command,
            state=state,
            selected_cfm_controls=plan.selected_cfm_controls.detach(),
            history_length=history_length,
        )
        return command

    def _validate_plan_shapes(self, plan: Any, future_horizon: int) -> None:
        if tuple(plan.selected_controls.shape) != (future_horizon, 2):
            raise RuntimeError(
                "selected dynamic controls must have shape "
                f"[{future_horizon}, 2], got {tuple(plan.selected_controls.shape)}"
            )
        expected_cfm_shape = (1, 2, self.config.horizon)
        if tuple(plan.selected_cfm_controls.shape) != expected_cfm_shape:
            raise RuntimeError(
                "selected CFM controls must have shape "
                f"{expected_cfm_shape}, got "
                f"{tuple(plan.selected_cfm_controls.shape)}"
            )
        if plan.cfm_branch_indices.numel() != self.config.branches:
            raise RuntimeError("planner returned the wrong number of CFM branches")
        if plan.branch_costs.numel() != self.config.branches:
            raise RuntimeError("planner returned the wrong number of branch costs")

    def _build_visualization_trace(
        self,
        *,
        plan: Any,
        state: SocNavState,
        human_positions: torch.Tensor,
        human_velocities: torch.Tensor,
        future_horizon: int,
    ) -> dict[str, Any]:
        """Capture planner-only geometry without changing the environment contract."""
        selected_branch = int(plan.selected_branch)
        no_vrc_prediction = constant_velocity_prediction(
            positions=human_positions,
            velocities=human_velocities,
            horizon=future_horizon,
            dt=self._context.time_step,
        ).squeeze(0).transpose(1, 2)
        trace: dict[str, Any] = {
            "schema_version": VISUALIZATION_TRACE_SCHEMA,
            "human_ids": list(state.human_ids),
            "robot_candidate_trajectories": plan.cfm_branch_states.detach(),
            "robot_conditioning_trajectory": plan.cfm_branch_states[
                selected_branch
            ].detach(),
            "robot_prediction": plan.mppi_branch_states[selected_branch].detach(),
            "pedestrian_prediction_no_vrc": no_vrc_prediction.detach(),
            "pedestrian_prediction_vrc": None,
            "vrc_tube": None,
            "vrc_forces": None,
        }
        trace.update(
            self._build_vrc_visualization_trace(
                plan=plan,
                state=state,
                human_positions=human_positions,
            )
        )
        return trace

    def _build_vrc_visualization_trace(
        self,
        *,
        plan: Any,
        state: SocNavState,
        human_positions: torch.Tensor,
    ) -> dict[str, Any]:
        del plan, state, human_positions
        return {}

    def observe_transition(
        self,
        previous_state: SocNavState,
        command: PlannerCommand,
        transition: SocNavStep,
    ) -> None:
        self._assert_ready()
        pending = self._pending
        if pending is None:
            raise RuntimeError("plan() must be called before observe_transition()")
        if command is not pending.command:
            raise RuntimeError("observe_transition() received a different command")
        self._assert_same_state(previous_state, pending.state)
        self._validate_episode_state(previous_state)
        self._validate_episode_state(transition.state)
        if not np.allclose(
            transition.requested_control,
            command.control,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("environment transition does not match planner command")

        previous_robot = torch.tensor(
            previous_state.robot_state,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        actual_world_control = torch.tensor(
            (
                transition.state.robot_position - previous_state.robot_position
            )
            / self._context.time_step,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        next_human_positions = torch.tensor(
            transition.state.human_positions,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        next_human_velocities = torch.tensor(
            transition.state.human_velocities,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        # These four entries are synchronized around the executed transition:
        # q_t, actual world-frame u_t, pedestrian p_{t+1}, and velocity at p_{t+1}.
        self._histories["ego_state"].update(previous_robot)
        self._histories["ego_control_sin"].update(actual_world_control)
        self._histories["obs_state"].update(next_human_positions)
        self._histories["obs_control"].update(next_human_velocities)
        self._assert_history_lengths()

        with self._episode_rng():
            self._warm_start = self._build_fixed_horizon_warm_start(
                pending.selected_cfm_controls,
                old_history_length=pending.history_length,
            )
        self._pending = None

    def _build_fixed_horizon_warm_start(
        self,
        selected_cfm_controls: torch.Tensor,
        *,
        old_history_length: int,
    ) -> torch.Tensor:
        selected = selected_cfm_controls.to(
            device=self.device,
            dtype=torch.float32,
        ).expand(self.config.cfm_candidates, -1, -1)
        normalized_selected = selected / self.config.space_scale
        noise = torch.randn_like(normalized_selected)
        alpha = self.config.warm_noise_level
        mixed = alpha * normalized_selected + (1.0 - alpha) * noise
        future_tail = mixed[:, :, old_history_length + 1 :]

        control_history = self._histories["ego_control_sin"].get()
        if control_history is None:
            raise RuntimeError("control history was not committed")
        prefix = (
            control_history.expand(self.config.cfm_candidates, -1, -1)
            / self.config.space_scale
        )
        missing = self.config.horizon - prefix.shape[-1] - future_tail.shape[-1]
        if missing < 0:
            raise RuntimeError("warm-start construction exceeded the fixed horizon")
        if missing:
            extension_noise = torch.randn(
                self.config.cfm_candidates,
                2,
                missing,
                device=self.device,
                dtype=torch.float32,
            )
            extension_nominal = normalized_selected[:, :, -1:].expand(
                -1,
                -1,
                missing,
            )
            future_tail = torch.cat(
                (
                    future_tail,
                    alpha * extension_nominal
                    + (1.0 - alpha) * extension_noise,
                ),
                dim=-1,
            )
        warm_start = torch.cat((prefix, future_tail), dim=-1)
        expected = (
            self.config.cfm_candidates,
            2,
            self.config.horizon,
        )
        if tuple(warm_start.shape) != expected:
            raise RuntimeError(
                f"warm start must retain shape {expected}, got {tuple(warm_start.shape)}"
            )
        return warm_start

    def _assert_history_lengths(self) -> None:
        lengths = {name: len(history) for name, history in self._histories.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"planner histories are not time-aligned: {lengths}")

    def _planner_tensors(
        self,
        state: SocNavState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = torch.tensor(
            state.robot_state,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 3)
        goal = torch.tensor(
            state.goal,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 2)
        positions = torch.tensor(
            state.human_positions,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, -1, 2)
        velocities = torch.tensor(
            state.human_velocities,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, -1, 2)
        return robot, goal, positions, velocities

    @staticmethod
    def _validate_state_arrays(state: SocNavState) -> None:
        if state.robot_state.shape != (3,) or not np.isfinite(
            state.robot_state
        ).all():
            raise ValueError("SocNav robot_state must be finite with shape (3,)")
        if state.goal.shape != (2,) or not np.isfinite(state.goal).all():
            raise ValueError("SocNav goal must be finite with shape (2,)")
        if state.human_positions.shape != state.human_velocities.shape:
            raise ValueError("human position and velocity shapes must match")
        if not np.isfinite(state.human_positions).all() or not np.isfinite(
            state.human_velocities
        ).all():
            raise ValueError("human states must be finite")

    def _validate_episode_state(self, state: SocNavState) -> None:
        self._validate_state_arrays(state)
        if state.human_ids != self._human_ids:
            raise RuntimeError("SocNav human identity/order changed within the episode")
        if not np.allclose(state.goal, self._initial_goal, rtol=0.0, atol=1e-6):
            raise RuntimeError("SocNav robot goal changed within the episode")
        if not np.allclose(
            state.human_radii,
            self._human_radii,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("SocNav human radii changed within the episode")
        if not np.isclose(
            state.robot_radius,
            self._robot_radius,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("SocNav robot radius changed within the episode")

    @staticmethod
    def _assert_same_state(actual: SocNavState, expected: SocNavState) -> None:
        if actual.human_ids != expected.human_ids or not np.allclose(
            actual.robot_state,
            expected.robot_state,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError("observe_transition() previous state is stale")

    def _assert_ready(self) -> None:
        if self._context is None or self._solver is None:
            raise RuntimeError("reset_episode() must be called before planning")

    def _plan_branches(
        self,
        *,
        state: torch.Tensor,
        goal: torch.Tensor,
        noisy_action_seq: torch.Tensor,
        noise_level: torch.Tensor,
        current_positions: torch.Tensor,
        current_velocities: torch.Tensor,
        planning_horizon: int,
        cfm_config: CFMConfig,
        initial_state: SocNavState,
    ) -> Any:
        raise NotImplementedError


class SocNavCFMMPPIPlanner(_SocNavCFMPlannerBase):
    """CFM + branch-local MPPI with constant-velocity pedestrians."""

    name = "cfm-mppi-cv"

    def _plan_branches(self, **kwargs: Any) -> Any:
        kwargs.pop("initial_state")
        cfm_config = kwargs.pop("cfm_config")
        return plan_with_constant_velocity_prediction(
            model=self.model,
            solver=self._solver,
            config=cfm_config,
            histories=self._histories,
            num_branches=self.config.branches,
            look_ahead_distance=self.config.look_ahead_distance,
            build_selected_vrc_tube=False,
            **kwargs,
        )


class SocNavVRCMPPIPlanner(_SocNavCFMPlannerBase):
    """CFM + VRC-conditioned pedestrian prediction + branch-local MPPI."""

    name = "vrc-mppi"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: SocNavPlannerConfig | None = None,
        device: str | torch.device | None = None,
        solver_factory: Callable[..., Any] = FlowMPPI,
        vrc_params: VRCParameters | None = None,
        robot_force_params: RobotForceParameters | None = None,
        prediction_params: PedestrianPredictionParameters | None = None,
        record_visualization: bool = False,
    ) -> None:
        super().__init__(
            model,
            config=config,
            device=device,
            solver_factory=solver_factory,
            record_visualization=record_visualization,
        )
        self._vrc_template = vrc_params or VRCParameters()
        self._robot_force_params = robot_force_params or RobotForceParameters()
        self._prediction_template = (
            prediction_params or PedestrianPredictionParameters()
        )

    def _episode_vrc_params(self, state: SocNavState) -> VRCParameters:
        maximum_human_radius = (
            float(state.human_radii.max())
            if state.human_radii.size
            else self._vrc_template.pedestrian_radius
        )
        return replace(
            self._vrc_template,
            robot_radius=float(state.robot_radius),
            pedestrian_radius=maximum_human_radius,
        )

    def _plan_branches(self, **kwargs: Any) -> Any:
        initial_state = kwargs.pop("initial_state")
        cfm_config = kwargs.pop("cfm_config")
        episode_vrc_params = self._episode_vrc_params(initial_state)
        prediction_params = replace(
            self._prediction_template,
            # Keep the internal prediction within the same physical speed
            # contract as the SocNavGym episode under evaluation.
            maximum_speed=self._context.max_human_speed,
        )
        plan = plan_vrc_branches(
            model=self.model,
            solver=self._solver,
            config=cfm_config,
            histories=self._histories,
            num_branches=self.config.branches,
            look_ahead_distance=self.config.look_ahead_distance,
            vrc_params=episode_vrc_params,
            force_params=self._robot_force_params,
            prediction_params=prediction_params,
            build_selected_vrc_tube=False,
            **kwargs,
        )
        if plan.selected_vrc_tube is not None:
            raise RuntimeError(
                "VRC planning must remain internal to the SocNavGym controller"
            )
        return plan

    def _build_vrc_visualization_trace(
        self,
        *,
        plan: Any,
        state: SocNavState,
        human_positions: torch.Tensor,
    ) -> dict[str, Any]:
        """Record the exact pre-MPPI VRC that generated the blue forecast."""
        selected_branch = int(plan.selected_branch)
        conditioning_states = plan.cfm_branch_states[
            selected_branch : selected_branch + 1
        ]
        conditioning_controls = plan.cfm_branch_controls[
            selected_branch : selected_branch + 1
        ]
        tube = build_tensor_vrc_tube(
            states=conditioning_states,
            controls_uni=conditioning_controls,
            params=self._episode_vrc_params(state),
        )
        forces = force_from_tensor_vrc_tube(
            pedestrian_positions=human_positions,
            vrc_tube=tube,
            current_index=0,
            force_params=self._robot_force_params,
            preview_steps=self._prediction_template.preview_steps,
            discount=self._prediction_template.preview_discount,
        )
        return {
            "pedestrian_prediction_vrc": plan.pedestrian_predictions[
                selected_branch
            ].detach(),
            "vrc_tube": {
                "centers": tube.centers[0].detach(),
                "longitudinal_radii": tube.longitudinal_radii[0].detach(),
                "lateral_radii": tube.lateral_radii[0].detach(),
                "theta": conditioning_states[0, :, 2].detach(),
                "influence_cutoff": self._robot_force_params.influence_cutoff,
            },
            "vrc_forces": forces[0].detach(),
        }


__all__ = [
    "SocNavCFMMPPIPlanner",
    "SocNavPlannerConfig",
    "SocNavVRCMPPIPlanner",
    "VISUALIZATION_TRACE_SCHEMA",
    "socnav_diff_drive_dynamics",
]
