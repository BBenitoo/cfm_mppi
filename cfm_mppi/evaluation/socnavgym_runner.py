"""Shared closed-loop evaluation runner for SocNavGym planners.

The runner is deliberately simulator-centric: a planner observes one parsed
SocNavGym state and returns a physical differential-drive control ``[v, omega]``.
Only the adapter advances the environment, and its returned observation is the
sole state used by the next planning step.  Planner diagnostics (including VRC
tubes) are recorded but can never be passed to the simulator.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from cfm_mppi.evaluation.socnavgym_adapter import (
    SocNavGymAdapter,
    SocNavState,
    SocNavStep,
)


class RunnerContractError(RuntimeError):
    """A planner or environment violated the shared evaluation contract."""


@dataclass(frozen=True)
class PlanningBudget:
    """Algorithmic samples spent by one planning decision."""

    cfm_candidates: int
    refinement_rollouts: int

    def __post_init__(self) -> None:
        if self.cfm_candidates <= 0 or self.refinement_rollouts <= 0:
            raise ValueError("planning budget counts must be positive")


@dataclass(frozen=True)
class EpisodeContext:
    """Environment contract passed identically to every compared planner."""

    env_seed: int
    planner_seed: int
    time_step: float
    episode_length: int
    max_human_speed: float
    control_low: np.ndarray
    control_high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.control_low, dtype=np.float32).copy()
        high = np.asarray(self.control_high, dtype=np.float32).copy()
        if low.shape != (2,) or high.shape != (2,):
            raise ValueError("physical control limits must have shape (2,)")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("physical control limits must be finite")
        if np.any(low >= high):
            raise ValueError("every physical control lower bound must be below high")
        if not math.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if not math.isfinite(self.max_human_speed) or self.max_human_speed <= 0:
            raise ValueError("max_human_speed must be finite and positive")
        low.setflags(write=False)
        high.setflags(write=False)
        object.__setattr__(self, "control_low", low)
        object.__setattr__(self, "control_high", high)


@dataclass(frozen=True)
class PlannerCommand:
    """One physical command and planner-only diagnostic data."""

    control: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        control = np.asarray(self.control, dtype=np.float32).copy()
        if control.shape != (2,):
            raise ValueError("planner control must have shape (2,) containing [v, omega]")
        if not np.isfinite(control).all():
            raise ValueError("planner control must contain only finite values")
        control.setflags(write=False)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class ClosedLoopPlanner(Protocol):
    """Stateful controller consumed by :func:`run_socnavgym_episode`."""

    name: str
    budget: PlanningBudget

    def reset_episode(
        self,
        initial_state: SocNavState,
        context: EpisodeContext,
    ) -> None: ...

    def plan(self, state: SocNavState, step_index: int) -> PlannerCommand: ...

    def observe_transition(
        self,
        previous_state: SocNavState,
        command: PlannerCommand,
        transition: SocNavStep,
    ) -> None: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    state: SocNavState
    next_state: SocNavState
    requested_control: np.ndarray
    applied_control: np.ndarray
    normalized_action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    decision_seconds: float
    bookkeeping_seconds: float

    def __post_init__(self) -> None:
        for attribute, shape in (
            ("requested_control", (2,)),
            ("applied_control", (2,)),
            ("normalized_action", (3,)),
        ):
            value = np.asarray(getattr(self, attribute), dtype=np.float32).copy()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{attribute} must be a finite array of shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, attribute, value)
        object.__setattr__(self, "info", _snapshot_mapping(self.info))
        object.__setattr__(
            self,
            "diagnostics",
            _snapshot_mapping(self.diagnostics, max_array_elements=4096),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state": _state_to_dict(self.state),
            "next_state": _state_to_dict(self.next_state),
            "requested_control": self.requested_control.tolist(),
            "applied_control": self.applied_control.tolist(),
            "normalized_action": self.normalized_action.tolist(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": _to_jsonable(self.info),
            "diagnostics": _to_jsonable(self.diagnostics),
            "decision_seconds": self.decision_seconds,
            "bookkeeping_seconds": self.bookkeeping_seconds,
        }


@dataclass(frozen=True)
class EpisodeResult:
    planner_name: str
    budget: PlanningBudget
    context: EpisodeContext
    initial_state: SocNavState
    reset_info: Mapping[str, Any]
    steps: tuple[StepRecord, ...]
    runner_limit_reached: bool

    @property
    def final_state(self) -> SocNavState:
        return self.steps[-1].next_state if self.steps else self.initial_state

    @property
    def final_info(self) -> Mapping[str, Any]:
        return self.steps[-1].info if self.steps else self.reset_info

    def summary(self) -> dict[str, Any]:
        decision_times = np.asarray(
            [record.decision_seconds for record in self.steps], dtype=np.float64
        )
        bookkeeping_times = np.asarray(
            [record.bookkeeping_seconds for record in self.steps], dtype=np.float64
        )
        states = [self.initial_state] + [record.next_state for record in self.steps]
        positions = np.stack([state.robot_position for state in states])
        geometric_path_length = float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        )
        minimum_center_distance, minimum_clearance = _minimum_human_distances(states)
        final_goal_distance = float(
            np.linalg.norm(self.final_state.robot_position - self.final_state.goal)
        )
        final_info = self.final_info

        def any_event(key: str) -> bool:
            return any(bool(record.info.get(key, False)) for record in self.steps)

        collision_human = any_event("COLLISION_HUMAN")
        collision_object = any_event("COLLISION_OBJECT")
        collision_wall = any_event("COLLISION_WALL")
        out_of_map = any_event("OUT_OF_MAP")
        collision = any_event("COLLISION")
        return {
            "planner": self.planner_name,
            "env_seed": self.context.env_seed,
            "planner_seed": self.context.planner_seed,
            "steps": len(self.steps),
            "simulation_seconds": len(self.steps) * self.context.time_step,
            "return": float(sum(record.reward for record in self.steps)),
            "terminated": bool(self.steps and self.steps[-1].terminated),
            "truncated": bool(self.steps and self.steps[-1].truncated),
            "runner_limit_reached": self.runner_limit_reached,
            "success": any_event("SUCCESS"),
            "collision": collision,
            "collision_any": (
                collision
                or collision_human
                or collision_object
                or collision_wall
                or out_of_map
            ),
            "collision_human": collision_human,
            "collision_object": collision_object,
            "collision_wall": collision_wall,
            "out_of_map": out_of_map,
            "timeout": any_event("TIMEOUT"),
            "environment_time_to_reach_goal": _optional_finite_float(
                final_info.get("TIME_TO_REACH_GOAL")
            ),
            "environment_path_length": _optional_finite_float(
                final_info.get("PATH_LENGTH")
            ),
            "environment_minimum_distance_to_human": _minimum_finite_info(
                self.steps,
                "MINIMUM_DISTANCE_TO_HUMAN",
            ),
            "final_goal_distance": final_goal_distance,
            "geometric_path_length": geometric_path_length,
            "minimum_human_center_distance": minimum_center_distance,
            "minimum_human_clearance": minimum_clearance,
            "freezing_events": _count_freezing_events(
                self.steps,
                dt=self.context.time_step,
            ),
            "decision_latency": _latency_summary(decision_times),
            "bookkeeping_latency": _latency_summary(bookkeeping_times),
            "total_compute_latency": _latency_summary(
                decision_times + bookkeeping_times
            ),
            "cold_start_decision_seconds": (
                float(decision_times[0]) if decision_times.size else None
            ),
            "steady_state_decision_latency": _latency_summary(decision_times[1:]),
            "environment_final_info": _to_jsonable(final_info),
        }

    def to_dict(self, *, include_steps: bool = True) -> dict[str, Any]:
        document = {
            "planner": self.planner_name,
            "budget": {
                "cfm_candidates": self.budget.cfm_candidates,
                "refinement_rollouts": self.budget.refinement_rollouts,
            },
            "context": {
                "env_seed": self.context.env_seed,
                "planner_seed": self.context.planner_seed,
                "time_step": self.context.time_step,
                "episode_length": self.context.episode_length,
                "max_human_speed": self.context.max_human_speed,
                "control_low": self.context.control_low.tolist(),
                "control_high": self.context.control_high.tolist(),
            },
            "initial_state": _state_to_dict(self.initial_state),
            "reset_info": _to_jsonable(self.reset_info),
            "summary": self.summary(),
        }
        if include_steps:
            document["steps"] = [record.to_dict() for record in self.steps]
        return document


@dataclass(frozen=True)
class PairedEvaluationResult:
    """Results from matched methods evaluated on the same environment seeds."""

    env_seeds: tuple[int, ...]
    planner_names: tuple[str, ...]
    execution_orders: tuple[tuple[str, ...], ...]
    episodes: tuple[EpisodeResult, ...]

    def to_dict(self, *, include_steps: bool = True) -> dict[str, Any]:
        return {
            "schema_version": "cfm_mppi.socnavgym_evaluation.v1",
            "environment_seeds": list(self.env_seeds),
            "planners": list(self.planner_names),
            "execution_orders": [list(order) for order in self.execution_orders],
            "episodes": [
                episode.to_dict(include_steps=include_steps)
                for episode in self.episodes
            ],
            "summaries": [episode.summary() for episode in self.episodes],
        }


@contextmanager
def _preserve_environment_rng():
    """Prevent planner code from perturbing SocNavGym's Python/NumPy RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _synchronize(planner: ClosedLoopPlanner) -> None:
    synchronize = getattr(planner, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _validate_control(control: np.ndarray, context: EpisodeContext) -> None:
    tolerance = 1e-6
    if np.any(control < context.control_low - tolerance) or np.any(
        control > context.control_high + tolerance
    ):
        raise RunnerContractError(
            "planner emitted a physical control outside the shared bounds: "
            f"{control.tolist()} not in "
            f"[{context.control_low.tolist()}, {context.control_high.tolist()}]"
        )


def run_socnavgym_episode(
    environment: SocNavGymAdapter,
    planner: ClosedLoopPlanner,
    *,
    env_seed: int,
    planner_seed: int | None = None,
    max_steps: int | None = None,
) -> EpisodeResult:
    """Run one headless episode under the shared fairness contract."""
    initial_state, reset_info = environment.reset(seed=env_seed)
    context = EpisodeContext(
        env_seed=int(env_seed),
        planner_seed=int(env_seed if planner_seed is None else planner_seed),
        time_step=environment.time_step,
        episode_length=environment.episode_length,
        max_human_speed=environment.max_human_speed,
        control_low=environment.physical_control_low,
        control_high=environment.physical_control_high,
    )
    if max_steps is None:
        step_limit = context.episode_length
    else:
        if isinstance(max_steps, (bool, np.bool_)) or not isinstance(
            max_steps, (int, np.integer)
        ):
            raise TypeError("max_steps must be an integer or None")
        step_limit = int(max_steps)
        if step_limit <= 0 or step_limit >= context.episode_length:
            raise ValueError(
                "an explicit max_steps must be in "
                "[1, environment.episode_length - 1]"
            )

    with _preserve_environment_rng():
        planner.reset_episode(initial_state, context)

    records: list[StepRecord] = []
    state = initial_state
    expected_human_ids = initial_state.human_ids
    for step_index in range(step_limit):
        if state.human_ids != expected_human_ids:
            raise RunnerContractError("human identity/order changed before planning")

        _synchronize(planner)
        decision_start = time.perf_counter_ns()
        with _preserve_environment_rng():
            command = planner.plan(state, step_index)
        _synchronize(planner)
        decision_seconds = (time.perf_counter_ns() - decision_start) / 1e9
        if not isinstance(command, PlannerCommand):
            raise RunnerContractError("planner.plan() must return PlannerCommand")
        _validate_control(command.control, context)

        # Only the two physical control scalars cross the planner/environment
        # boundary.  In particular, VRC diagnostics cannot influence env.step.
        transition = environment.step(command.control)
        if not isinstance(transition, SocNavStep):
            raise RunnerContractError("environment.step() must return SocNavStep")
        if not np.allclose(
            transition.requested_control,
            command.control,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RunnerContractError(
                "environment did not receive exactly the planner's physical control"
            )
        if not np.allclose(
            transition.applied_control,
            transition.requested_control,
            rtol=0.0,
            atol=1e-6,
        ):
            raise RunnerContractError(
                "the environment clipped the planner control; formal evaluations "
                "must respect the shared physical bounds"
            )
        if transition.state.human_ids != expected_human_ids:
            raise RunnerContractError("human identity/order changed after env.step")

        bookkeeping_start = time.perf_counter_ns()
        diagnostic_snapshot = _snapshot_mapping(
            command.diagnostics,
            max_array_elements=4096,
        )
        info_snapshot = _snapshot_mapping(transition.info)
        observer = getattr(planner, "observe_transition", None)
        has_next_decision = not transition.done and step_index + 1 < step_limit
        if callable(observer) and has_next_decision:
            with _preserve_environment_rng():
                observer(state, command, transition)
            _synchronize(planner)
        bookkeeping_seconds = (time.perf_counter_ns() - bookkeeping_start) / 1e9

        records.append(
            StepRecord(
                step_index=step_index,
                state=state,
                next_state=transition.state,
                requested_control=transition.requested_control.copy(),
                applied_control=transition.applied_control.copy(),
                normalized_action=transition.normalized_action.copy(),
                reward=transition.reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=info_snapshot,
                diagnostics=diagnostic_snapshot,
                decision_seconds=decision_seconds,
                bookkeeping_seconds=bookkeeping_seconds,
            )
        )
        state = transition.state
        if transition.done:
            break

    runner_limit_reached = bool(
        records
        and not records[-1].terminated
        and not records[-1].truncated
        and len(records) == step_limit
    )
    if max_steps is None and runner_limit_reached:
        raise RunnerContractError(
            "SocNavGym exhausted EPISODE_LENGTH without terminated or truncated"
        )
    return EpisodeResult(
        planner_name=str(planner.name),
        budget=planner.budget,
        context=context,
        initial_state=initial_state,
        reset_info=_snapshot_mapping(reset_info),
        steps=tuple(records),
        runner_limit_reached=runner_limit_reached,
    )


def assert_matching_budgets(planners: Sequence[ClosedLoopPlanner]) -> None:
    """Fail before evaluation when compared planners use different budgets."""
    if not planners:
        raise ValueError("at least one planner is required")
    expected = planners[0].budget
    mismatches = {
        str(planner.name): planner.budget
        for planner in planners[1:]
        if planner.budget != expected
    }
    if mismatches:
        raise RunnerContractError(
            f"planner budgets do not match {expected}: {mismatches}"
        )


def run_paired_socnavgym_evaluation(
    environment_factory: Callable[[], SocNavGymAdapter],
    planner_factories: Mapping[str, Callable[[], ClosedLoopPlanner]],
    *,
    env_seeds: Sequence[int],
    planner_seed_for_env: Callable[[int], int] | None = None,
    max_steps: int | None = None,
    execution_order_offset: int = 0,
) -> PairedEvaluationResult:
    """Run fresh, matched environments for every ``(planner, env_seed)`` pair."""
    if not planner_factories:
        raise ValueError("at least one planner factory is required")
    planner_names = tuple(str(name) for name in planner_factories)
    if any(not name for name in planner_names) or len(set(planner_names)) != len(
        planner_names
    ):
        raise ValueError("planner factory names must be non-empty and unique")

    seeds: list[int] = []
    for seed in env_seeds:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise TypeError("environment seeds must be integers")
        converted_seed = int(seed)
        if converted_seed < 0:
            raise ValueError("environment seeds must be non-negative")
        seeds.append(converted_seed)
    if not seeds:
        raise ValueError("at least one environment seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("environment seeds must be unique")
    if isinstance(execution_order_offset, (bool, np.bool_)) or not isinstance(
        execution_order_offset, (int, np.integer)
    ):
        raise TypeError("execution_order_offset must be an integer")
    order_offset = int(execution_order_offset)

    episodes_by_key: dict[tuple[int, str], EpisodeResult] = {}
    execution_orders: list[tuple[str, ...]] = []
    expected_budget: PlanningBudget | None = None
    for seed_index, env_seed in enumerate(seeds):
        planner_seed = (
            env_seed
            if planner_seed_for_env is None
            else planner_seed_for_env(env_seed)
        )
        if isinstance(planner_seed, (bool, np.bool_)) or not isinstance(
            planner_seed, (int, np.integer)
        ):
            raise TypeError("planner_seed_for_env must return an integer")
        planners = {
            name: planner_factories[name]()
            for name in planner_names
        }
        for name, planner in planners.items():
            if str(planner.name) != name:
                raise RunnerContractError(
                    f"planner factory {name!r} returned planner named "
                    f"{planner.name!r}"
                )
        assert_matching_budgets(tuple(planners.values()))
        budget = next(iter(planners.values())).budget
        if expected_budget is None:
            expected_budget = budget
        elif budget != expected_budget:
            raise RunnerContractError(
                "planner budget changed between environment seeds: "
                f"{expected_budget} != {budget}"
            )

        execution_order = (
            planner_names
            if (order_offset + seed_index) % 2 == 0
            else tuple(reversed(planner_names))
        )
        execution_orders.append(execution_order)
        reference: EpisodeResult | None = None
        for planner_name in execution_order:
            planner = planners[planner_name]
            environment = environment_factory()
            if not isinstance(environment, SocNavGymAdapter):
                raise TypeError("environment_factory must return SocNavGymAdapter")
            with environment:
                episode = run_socnavgym_episode(
                    environment,
                    planner,
                    env_seed=env_seed,
                    planner_seed=int(planner_seed),
                    max_steps=max_steps,
                )
            if reference is None:
                reference = episode
            else:
                _assert_matching_episode_contract(reference, episode)
            episodes_by_key[(env_seed, planner_name)] = episode

    return PairedEvaluationResult(
        env_seeds=tuple(seeds),
        planner_names=planner_names,
        execution_orders=tuple(execution_orders),
        episodes=tuple(
            episodes_by_key[(env_seed, planner_name)]
            for env_seed in seeds
            for planner_name in planner_names
        ),
    )


def _assert_matching_episode_contract(
    reference: EpisodeResult,
    candidate: EpisodeResult,
) -> None:
    first = reference.context
    second = candidate.context
    scalar_contract = (
        first.env_seed == second.env_seed
        and first.time_step == second.time_step
        and first.episode_length == second.episode_length
        and first.max_human_speed == second.max_human_speed
        and np.array_equal(first.control_low, second.control_low)
        and np.array_equal(first.control_high, second.control_high)
    )
    if not scalar_contract or not _states_match(
        reference.initial_state,
        candidate.initial_state,
    ):
        raise RunnerContractError(
            "paired planners did not receive an identical SocNavGym contract "
            "and initial state for the same seed"
        )


def _states_match(first: SocNavState, second: SocNavState) -> bool:
    if (
        not np.array_equal(first.robot_state, second.robot_state)
        or not np.array_equal(first.goal, second.goal)
        or not np.array_equal(
            first.robot_body_velocity,
            second.robot_body_velocity,
        )
        or first.robot_radius != second.robot_radius
        or first.goal_radius != second.goal_radius
        or first.human_ids != second.human_ids
    ):
        return False
    return all(
        first_human.radius == second_human.radius
        and first_human.orientation == second_human.orientation
        and first_human.gaze == second_human.gaze
        and np.array_equal(first_human.position, second_human.position)
        and np.array_equal(first_human.velocity, second_human.velocity)
        for first_human, second_human in zip(first.humans, second.humans)
    )


def _state_to_dict(state: SocNavState) -> dict[str, Any]:
    return {
        "robot_state": state.robot_state.tolist(),
        "goal": state.goal.tolist(),
        "robot_body_velocity": state.robot_body_velocity.tolist(),
        "robot_velocity": state.robot_velocity.tolist(),
        "robot_radius": state.robot_radius,
        "goal_radius": state.goal_radius,
        "humans": [
            {
                "id": human.id,
                "position": human.position.tolist(),
                "velocity": human.velocity.tolist(),
                "radius": human.radius,
                "orientation": human.orientation,
                "gaze": human.gaze,
            }
            for human in state.humans
        ],
    }


def _minimum_human_distances(
    states: Sequence[SocNavState],
) -> tuple[float | None, float | None]:
    centers: list[float] = []
    clearances: list[float] = []
    for state in states:
        for human in state.humans:
            distance = float(np.linalg.norm(state.robot_position - human.position))
            centers.append(distance)
            clearances.append(distance - state.robot_radius - human.radius)
    if not centers:
        return None, None
    return min(centers), min(clearances)


def _count_freezing_events(
    records: Sequence[StepRecord],
    *,
    dt: float,
    minimum_duration: float = 1.0,
    speed_threshold: float = 0.05,
    goal_distance_threshold: float = 0.5,
) -> int:
    minimum_samples = math.ceil(minimum_duration / dt)
    count = 0
    run_length = 0
    for record in records:
        distance_to_goal = float(
            np.linalg.norm(record.state.robot_position - record.state.goal)
        )
        frozen = (
            abs(float(record.applied_control[0])) < speed_threshold
            and distance_to_goal > goal_distance_threshold
        )
        if frozen:
            run_length += 1
        elif run_length:
            count += int(run_length >= minimum_samples)
            run_length = 0
    if run_length:
        count += int(run_length >= minimum_samples)
    return count


def _latency_summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _optional_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _minimum_finite_info(
    records: Sequence[StepRecord],
    key: str,
) -> float | None:
    values = [
        value
        for record in records
        if (value := _optional_finite_float(record.info.get(key))) is not None
    ]
    return min(values) if values else None


def _snapshot_mapping(
    value: Mapping[str, Any],
    *,
    max_array_elements: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("snapshot value must be a mapping")
    converted = _to_jsonable(value, max_array_elements=max_array_elements)
    if not isinstance(converted, dict):
        raise TypeError("mapping snapshot did not produce a dictionary")
    return converted


def _to_jsonable(
    value: Any,
    *,
    max_array_elements: int | None = None,
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _to_jsonable(
                item,
                max_array_elements=max_array_elements,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if max_array_elements is not None and len(value) > max_array_elements:
            raise RunnerContractError("diagnostic sequence exceeds element limit")
        return [
            _to_jsonable(item, max_array_elements=max_array_elements)
            for item in value
        ]
    if isinstance(value, np.ndarray):
        if max_array_elements is not None and value.size > max_array_elements:
            raise RunnerContractError("diagnostic array exceeds element limit")
        return _to_jsonable(
            value.tolist(),
            max_array_elements=max_array_elements,
        )
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        numel = getattr(value, "numel", None)
        if callable(numel) and max_array_elements is not None:
            if int(numel()) > max_array_elements:
                raise RunnerContractError("diagnostic tensor exceeds element limit")
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        return _to_jsonable(
            value.tolist(),
            max_array_elements=max_array_elements,
        )
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "ClosedLoopPlanner",
    "EpisodeContext",
    "EpisodeResult",
    "PairedEvaluationResult",
    "PlannerCommand",
    "PlanningBudget",
    "RunnerContractError",
    "StepRecord",
    "assert_matching_budgets",
    "run_socnavgym_episode",
    "run_paired_socnavgym_evaluation",
]
