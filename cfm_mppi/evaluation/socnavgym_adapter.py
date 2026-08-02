"""Adapter between SocNavGym v1 and the CFM/MPPI evaluation code.

SocNavGym is an optional, HPC-only dependency.  In particular, importing this
module must not import Gymnasium, DGL, RVO2, or SocNavGym.  Those imports are
therefore confined to :func:`make_socnavgym_env` and only happen when a real
environment is requested.

The selected SocNavGym v1 ``WorldFrameObservations`` contract encodes the
robot as 16 floats and each human as 14 floats.  The wrapper itself does not
expose human IDs, so this adapter pairs observation rows with the unwrapped
simulator objects and returns the rows in a stable ID order for the duration
of an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_ENV_ID = "SocNavGym-v1"
ROBOT_OBSERVATION_DIM = 16
HUMAN_OBSERVATION_STRIDE = 14


class SocNavGymAdapterError(RuntimeError):
    """The environment does not satisfy the adapter's v1 contract."""


def _readonly_float_vector(
    value: Sequence[float] | np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).copy()
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HumanState:
    """One human's world-frame state.

    Position and velocity are two-element ``float32`` arrays in metres and
    metres per second respectively.
    """

    id: int
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    orientation: float
    gaze: bool

    def __post_init__(self) -> None:
        if isinstance(self.id, (bool, np.bool_)) or not isinstance(
            self.id, (int, np.integer)
        ):
            raise ValueError("human id must be an integer")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("human radius must be finite and non-negative")
        if not np.isfinite(self.orientation):
            raise ValueError("human orientation must be finite")
        object.__setattr__(self, "id", int(self.id))
        object.__setattr__(
            self,
            "position",
            _readonly_float_vector(self.position, shape=(2,), name="human position"),
        )
        object.__setattr__(
            self,
            "velocity",
            _readonly_float_vector(self.velocity, shape=(2,), name="human velocity"),
        )
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "orientation", float(self.orientation))
        object.__setattr__(self, "gaze", bool(self.gaze))


@dataclass(frozen=True)
class SocNavState:
    """Planner-facing state parsed from a world-frame observation."""

    robot_state: np.ndarray
    goal: np.ndarray
    robot_body_velocity: np.ndarray
    robot_radius: float
    goal_radius: float
    humans: tuple[HumanState, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.robot_radius) or self.robot_radius < 0:
            raise ValueError("robot radius must be finite and non-negative")
        if not np.isfinite(self.goal_radius) or self.goal_radius < 0:
            raise ValueError("goal radius must be finite and non-negative")
        humans = tuple(self.humans)
        if any(not isinstance(human, HumanState) for human in humans):
            raise ValueError("humans must contain only HumanState values")
        if len({human.id for human in humans}) != len(humans):
            raise ValueError("human IDs must be unique")
        object.__setattr__(
            self,
            "robot_state",
            _readonly_float_vector(
                self.robot_state, shape=(3,), name="robot state"
            ),
        )
        object.__setattr__(
            self,
            "goal",
            _readonly_float_vector(self.goal, shape=(2,), name="goal"),
        )
        object.__setattr__(
            self,
            "robot_body_velocity",
            _readonly_float_vector(
                self.robot_body_velocity,
                shape=(3,),
                name="robot body velocity",
            ),
        )
        object.__setattr__(self, "robot_radius", float(self.robot_radius))
        object.__setattr__(self, "goal_radius", float(self.goal_radius))
        object.__setattr__(self, "humans", humans)

    @property
    def robot_position(self) -> np.ndarray:
        """Robot ``[x, y]`` position in the world frame."""
        return self.robot_state[:2]

    @property
    def robot_heading(self) -> float:
        """Robot world-frame heading in radians."""
        return float(self.robot_state[2])

    @property
    def robot_velocity(self) -> np.ndarray:
        """Robot world-frame ``[vx, vy, omega]`` velocity.

        SocNavGym stores ``robot[12:15]`` as body-frame forward, lateral and
        angular velocity even inside ``WorldFrameObservations``.  This property
        performs the explicit body-to-world conversion.
        """
        forward, lateral, angular = self.robot_body_velocity
        heading = self.robot_heading
        velocity = np.asarray(
            [
                forward * np.cos(heading) - lateral * np.sin(heading),
                forward * np.sin(heading) + lateral * np.cos(heading),
                angular,
            ],
            dtype=np.float32,
        )
        velocity.setflags(write=False)
        return velocity

    @property
    def human_ids(self) -> tuple[int, ...]:
        return tuple(human.id for human in self.humans)

    @property
    def human_positions(self) -> np.ndarray:
        """Human positions with shape ``[num_humans, 2]``."""
        if not self.humans:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([human.position for human in self.humans]).astype(
            np.float32,
            copy=False,
        )

    @property
    def human_velocities(self) -> np.ndarray:
        """Human world-frame velocities with shape ``[num_humans, 2]``."""
        if not self.humans:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([human.velocity for human in self.humans]).astype(
            np.float32,
            copy=False,
        )

    @property
    def human_radii(self) -> np.ndarray:
        """Human radii with shape ``[num_humans]``."""
        return np.asarray(
            [human.radius for human in self.humans],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class SocNavStep:
    """Result of applying one physical differential-drive control."""

    state: SocNavState
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    requested_control: np.ndarray
    applied_control: np.ndarray
    applied_action: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.state, SocNavState):
            raise ValueError("state must be a SocNavState")
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if not isinstance(self.terminated, (bool, np.bool_)) or not isinstance(
            self.truncated, (bool, np.bool_)
        ):
            raise ValueError("terminated and truncated must be booleans")
        if not isinstance(self.info, Mapping):
            raise ValueError("info must be a mapping")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        object.__setattr__(
            self,
            "requested_control",
            _readonly_float_vector(
                self.requested_control,
                shape=(2,),
                name="requested control",
            ),
        )
        object.__setattr__(
            self,
            "applied_control",
            _readonly_float_vector(
                self.applied_control,
                shape=(2,),
                name="applied control",
            ),
        )
        object.__setattr__(
            self,
            "applied_action",
            _readonly_float_vector(
                self.applied_action,
                shape=(3,),
                name="applied normalized action",
            ),
        )

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def normalized_action(self) -> np.ndarray:
        """Alias documenting that ``applied_action`` is normalized and 3-D."""
        return self.applied_action


def make_socnavgym_env(
    config_path: str | Path,
    *,
    env_id: str = DEFAULT_ENV_ID,
) -> Any:
    """Create ``SocNavGym-v1`` wrapped by ``WorldFrameObservations``.

    All optional simulator imports are deliberately delayed until this
    function is called.  The returned object follows the Gymnasium API.
    """
    resolved_config = Path(config_path).expanduser().resolve()
    if not resolved_config.is_file():
        raise FileNotFoundError(f"SocNavGym config not found: {resolved_config}")

    try:
        gymnasium = importlib.import_module("gymnasium")
        # Importing the package performs its Gymnasium environment
        # registration before gymnasium.make is called.
        importlib.import_module("socnavgym")
        wrappers = importlib.import_module("socnavgym.wrappers")
        wrapper_class = getattr(wrappers, "WorldFrameObservations")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "SocNavGym v1 and its dependencies are required to create the "
            "environment; activate the pinned 'vrc' environment."
        ) from exc

    base_env = gymnasium.make(env_id, config=str(resolved_config))
    try:
        return wrapper_class(base_env)
    except BaseException:
        base_env.close()
        raise


def physical_to_normalized_action(
    control: Sequence[float] | np.ndarray,
    *,
    max_linear_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    """Convert physical ``[v, omega]`` to SocNavGym's normalized action.

    SocNavGym v1 expects differential-drive actions in the form
    ``[v_normalized, 0, omega_normalized]``.  Both axes are clipped to the
    simulator's normalized ``[-1, 1]`` bounds.
    """
    physical = np.asarray(control, dtype=np.float32)
    if physical.shape != (2,):
        raise ValueError("control must have shape (2,) containing [v, omega]")
    if not np.isfinite(physical).all():
        raise ValueError("control must contain only finite values")

    limits = np.asarray(
        [max_linear_speed, max_angular_speed],
        dtype=np.float32,
    )
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("action speed limits must be finite and positive")

    normalized = np.clip(physical / limits, -1.0, 1.0)
    return np.asarray(
        [normalized[0], 0.0, normalized[1]],
        dtype=np.float32,
    )


def _human_objects_in_observation_order(base_env: Any) -> list[Any]:
    """Mirror the v1 world-frame wrapper's human traversal order."""
    humans = list(getattr(base_env, "static_humans", ()))
    humans.extend(getattr(base_env, "dynamic_humans", ()))
    interactions = (
        list(getattr(base_env, "moving_interactions", ()))
        + list(getattr(base_env, "static_interactions", ()))
        + list(getattr(base_env, "h_l_interactions", ()))
    )
    for interaction in interactions:
        name = getattr(interaction, "name", None)
        if name == "human-human-interaction":
            humans.extend(getattr(interaction, "humans", ()))
        elif name == "human-laptop-interaction":
            human = getattr(interaction, "human", None)
            if human is None:
                raise SocNavGymAdapterError(
                    "A human-laptop interaction has no human object"
                )
            humans.append(human)
        else:
            raise SocNavGymAdapterError(
                f"Unsupported SocNavGym interaction type: {name!r}"
            )
    return humans


def _validate_encoding(
    actual: np.ndarray,
    expected: Sequence[float],
    *,
    entity: str,
) -> None:
    if not np.array_equal(actual, np.asarray(expected, dtype=np.float32)):
        raise SocNavGymAdapterError(
            f"{entity} one-hot encoding does not match SocNavGym v1"
        )


def parse_world_frame_observation(
    observation: Mapping[str, Any],
    base_env: Any,
    *,
    expected_human_ids: Sequence[int] | None = None,
) -> SocNavState:
    """Parse a v1 world-frame observation into planner-facing state.

    ``expected_human_ids`` fixes the returned human order and detects identity
    changes within an episode.  When it is omitted, IDs are sorted once to
    establish a deterministic initial order.
    """
    if not isinstance(observation, Mapping):
        raise SocNavGymAdapterError("world-frame observation must be a mapping")
    if "robot" not in observation:
        raise SocNavGymAdapterError("world-frame observation has no robot key")

    robot = np.asarray(observation["robot"])
    if robot.shape != (ROBOT_OBSERVATION_DIM,):
        raise SocNavGymAdapterError(
            "robot observation must have shape "
            f"({ROBOT_OBSERVATION_DIM},), got {robot.shape}"
        )
    if not np.issubdtype(robot.dtype, np.number) or not np.isfinite(robot).all():
        raise SocNavGymAdapterError("robot observation must be finite and numeric")
    robot = robot.astype(np.float32, copy=True)
    _validate_encoding(robot[:6], [1, 0, 0, 0, 0, 0], entity="robot")

    heading_vector_norm = float(np.linalg.norm(robot[10:12]))
    if not np.isclose(heading_vector_norm, 1.0, rtol=1e-5, atol=1e-5):
        raise SocNavGymAdapterError("robot heading sine/cosine is not normalized")
    heading = np.float32(np.arctan2(robot[10], robot[11]))
    robot_state = np.asarray([robot[8], robot[9], heading], dtype=np.float32)
    goal = robot[6:8].copy()
    robot_body_velocity = robot[12:15].copy()
    robot_radius = float(robot[15])
    if robot_radius < 0:
        raise SocNavGymAdapterError("robot radius must be non-negative")
    raw_goal_radius = getattr(base_env, "GOAL_RADIUS", None)
    try:
        goal_radius = float(raw_goal_radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SocNavGymAdapterError(
            "SocNavGym environment has no valid GOAL_RADIUS"
        ) from exc
    if not np.isfinite(goal_radius) or goal_radius < 0:
        raise SocNavGymAdapterError(
            "SocNavGym environment GOAL_RADIUS must be finite and non-negative"
        )

    if bool(getattr(base_env, "get_padded_observations", False)):
        raise SocNavGymAdapterError(
            "stable human IDs require unpadded SocNavGym observations"
        )

    human_objects = _human_objects_in_observation_order(base_env)
    if "humans" not in observation:
        if human_objects:
            raise SocNavGymAdapterError(
                "world-frame observation has no humans key for present humans"
            )
        human_values = np.empty(0, dtype=np.float32)
    else:
        human_values = np.asarray(observation["humans"])
    if human_values.ndim != 1 or human_values.size % HUMAN_OBSERVATION_STRIDE:
        raise SocNavGymAdapterError(
            "human observation must be a flat array with stride "
            f"{HUMAN_OBSERVATION_STRIDE}, got {human_values.shape}"
        )
    if not np.issubdtype(human_values.dtype, np.number) or not np.isfinite(
        human_values
    ).all():
        raise SocNavGymAdapterError("human observation must be finite and numeric")
    rows = human_values.astype(np.float32, copy=False).reshape(
        -1,
        HUMAN_OBSERVATION_STRIDE,
    )
    if rows.shape[0] != len(human_objects):
        raise SocNavGymAdapterError(
            "human observation count does not match the unwrapped simulator "
            f"objects: {rows.shape[0]} != {len(human_objects)}"
        )

    by_id: dict[int, HumanState] = {}
    for row, human_object in zip(rows, human_objects):
        _validate_encoding(row[:6], [0, 1, 0, 0, 0, 0], entity="human")
        raw_id = getattr(human_object, "id", None)
        if raw_id is None:
            raise SocNavGymAdapterError("a SocNavGym human has no stable id")
        if isinstance(raw_id, (bool, np.bool_)) or not isinstance(
            raw_id, (int, np.integer)
        ):
            raise SocNavGymAdapterError(
                f"invalid SocNavGym human id: {raw_id!r}"
            )
        human_id = int(raw_id)
        if human_id in by_id:
            raise SocNavGymAdapterError(
                f"duplicate SocNavGym human id: {human_id}"
            )
        orientation_norm = float(np.linalg.norm(row[8:10]))
        if not np.isclose(orientation_norm, 1.0, rtol=1e-5, atol=1e-5):
            raise SocNavGymAdapterError(
                f"human {human_id} heading sine/cosine is not normalized"
            )
        radius = float(row[10])
        if radius < 0:
            raise SocNavGymAdapterError(
                f"human {human_id} radius must be non-negative"
            )
        by_id[human_id] = HumanState(
            id=human_id,
            position=row[6:8].copy(),
            velocity=row[11:13].copy(),
            radius=radius,
            orientation=float(np.arctan2(row[8], row[9])),
            gaze=bool(row[13] >= 0.5),
        )

    current_ids = set(by_id)
    if expected_human_ids is None:
        stable_ids = tuple(sorted(current_ids))
    else:
        stable_ids = tuple(int(human_id) for human_id in expected_human_ids)
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("expected_human_ids must be unique")
        if current_ids != set(stable_ids):
            raise SocNavGymAdapterError(
                "SocNavGym human IDs changed within the episode: expected "
                f"{stable_ids}, observed {tuple(sorted(current_ids))}"
            )

    return SocNavState(
        robot_state=robot_state,
        goal=goal,
        robot_body_velocity=robot_body_velocity,
        robot_radius=robot_radius,
        goal_radius=goal_radius,
        humans=tuple(by_id[human_id] for human_id in stable_ids),
    )


class SocNavGymAdapter:
    """Stateful adapter for a fixed-human SocNavGym v1 episode."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        env_id: str = DEFAULT_ENV_ID,
        env: Any | None = None,
    ) -> None:
        if env is not None and config_path is not None:
            raise ValueError("pass either config_path or env, not both")
        if env is None:
            if config_path is None:
                raise ValueError("config_path is required when env is not supplied")
            env = make_socnavgym_env(config_path, env_id=env_id)

        self._env = env
        self._closed = False
        self._episode_done = False
        self._state: SocNavState | None = None
        self._human_ids: tuple[int, ...] | None = None

    @property
    def env(self) -> Any:
        """The wrapped Gymnasium environment, primarily for rendering."""
        return self._env

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    @property
    def state(self) -> SocNavState:
        if self._state is None:
            raise RuntimeError("reset() must be called before reading state")
        return self._state

    @property
    def max_linear_speed(self) -> float:
        return self._positive_env_value("MAX_ADVANCE_ROBOT")

    @property
    def max_angular_speed(self) -> float:
        return self._positive_env_value("MAX_ROTATION")

    @property
    def max_human_speed(self) -> float:
        """Maximum physical pedestrian speed configured in SocNavGym."""
        return self._positive_env_value("MAX_ADVANCE_HUMAN")

    @property
    def time_step(self) -> float:
        return self._positive_env_value("TIMESTEP")

    @property
    def episode_length(self) -> int:
        value = getattr(self.unwrapped, "EPISODE_LENGTH", None)
        if isinstance(value, (bool, np.bool_)):
            raise SocNavGymAdapterError(
                "SocNavGym environment EPISODE_LENGTH must be a positive integer"
            )
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SocNavGymAdapterError(
                "SocNavGym environment has no valid EPISODE_LENGTH"
            ) from exc
        if converted <= 0 or converted != value:
            raise SocNavGymAdapterError(
                "SocNavGym environment EPISODE_LENGTH must be a positive integer"
            )
        return converted

    @property
    def physical_control_low(self) -> np.ndarray:
        """Lower physical ``[v, omega]`` limits used by the simulator."""
        limits = np.asarray(
            [-self.max_linear_speed, -self.max_angular_speed],
            dtype=np.float32,
        )
        limits.setflags(write=False)
        return limits

    @property
    def physical_control_high(self) -> np.ndarray:
        """Upper physical ``[v, omega]`` limits used by the simulator."""
        limits = np.asarray(
            [self.max_linear_speed, self.max_angular_speed],
            dtype=np.float32,
        )
        limits.setflags(write=False)
        return limits

    def _positive_env_value(self, name: str) -> float:
        value = getattr(self.unwrapped, name, None)
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SocNavGymAdapterError(
                f"SocNavGym environment has no valid {name}"
            ) from exc
        if not np.isfinite(converted) or converted <= 0:
            raise SocNavGymAdapterError(
                f"SocNavGym environment {name} must be finite and positive"
            )
        return converted

    def normalize_action(
        self,
        control: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        robot_type = getattr(getattr(self.unwrapped, "robot", None), "type", None)
        if robot_type != "diff-drive":
            raise SocNavGymAdapterError(
                f"expected a diff-drive SocNavGym robot, got {robot_type!r}"
            )
        action = physical_to_normalized_action(
            control,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
        )
        action_space = getattr(self._env, "action_space", None)
        contains = getattr(action_space, "contains", None)
        if callable(contains) and not contains(action):
            raise SocNavGymAdapterError(
                "normalized differential-drive action is outside action_space"
            )
        return action

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[SocNavState, dict[str, Any]]:
        self._assert_open()
        # Once the underlying environment is reset, a state from the previous
        # episode is no longer valid even if the new observation is malformed.
        self._state = None
        self._human_ids = None
        self._episode_done = False
        if options is None:
            result = self._env.reset(seed=seed)
        else:
            result = self._env.reset(seed=seed, options=dict(options))
        if not isinstance(result, tuple) or len(result) != 2:
            raise SocNavGymAdapterError(
                "reset() must return (observation, info)"
            )
        observation, info = result
        if not isinstance(info, Mapping):
            raise SocNavGymAdapterError("reset info must be a mapping")

        state = parse_world_frame_observation(observation, self.unwrapped)
        self._state = state
        self._human_ids = state.human_ids
        return state, dict(info)

    def step(
        self,
        control: Sequence[float] | np.ndarray,
    ) -> SocNavStep:
        self._assert_open()
        if self._state is None or self._human_ids is None:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_done:
            raise RuntimeError("reset() must be called after an episode ends")

        requested_control = np.asarray(control, dtype=np.float32)
        normalized_action = self.normalize_action(requested_control)
        result = self._env.step(normalized_action.copy())
        if not isinstance(result, tuple) or len(result) != 5:
            raise SocNavGymAdapterError(
                "step() must return "
                "(observation, reward, terminated, truncated, info)"
            )
        observation, reward, terminated, truncated, info = result
        if not isinstance(info, Mapping):
            raise SocNavGymAdapterError("step info must be a mapping")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated,
            (bool, np.bool_),
        ):
            raise SocNavGymAdapterError(
                "terminated and truncated must be booleans"
            )
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise SocNavGymAdapterError("step reward must be finite")

        # This observation is the sole source of truth after the step.  The
        # adapter deliberately performs no local dynamics integration.
        state = parse_world_frame_observation(
            observation,
            self.unwrapped,
            expected_human_ids=self._human_ids,
        )
        applied_control = np.clip(
            requested_control,
            self.physical_control_low,
            self.physical_control_high,
        ).astype(np.float32, copy=False)
        terminated_value = bool(terminated)
        truncated_value = bool(truncated)
        self._state = state
        self._episode_done = terminated_value or truncated_value
        return SocNavStep(
            state=state,
            reward=reward_value,
            terminated=terminated_value,
            truncated=truncated_value,
            info=dict(info),
            requested_control=requested_control.copy(),
            applied_control=applied_control,
            applied_action=normalized_action,
        )

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("SocNavGymAdapter is closed")

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self) -> "SocNavGymAdapter":
        self._assert_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False


__all__ = [
    "DEFAULT_ENV_ID",
    "HUMAN_OBSERVATION_STRIDE",
    "ROBOT_OBSERVATION_DIM",
    "HumanState",
    "SocNavGymAdapter",
    "SocNavGymAdapterError",
    "SocNavState",
    "SocNavStep",
    "make_socnavgym_env",
    "parse_world_frame_observation",
    "physical_to_normalized_action",
]
