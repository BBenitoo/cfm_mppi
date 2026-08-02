"""Strict, headless contract probe for a pinned SocNavGym installation.

The module deliberately imports no SocNavGym dependencies at import time.  This
keeps the normal CFM-MPPI test suite usable on machines where the simulator is
not installed, while providing a fail-closed acceptance check for the HPC
environment that will run the experiments.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse

# NumPy is part of the contract being checked.  Keep the CLI importable when it
# is absent or has a broken binary installation so the dependency stage can
# return the documented exit code instead of failing before a report exists.
np: Any
try:
    import numpy as _numpy
except Exception:  # pragma: no cover - exercised in a deliberately bare env
    np = None
else:
    np = _numpy


SCHEMA_VERSION = 1
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "configs" / "socnavgym" / "candidates.json"
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_REVISION = 10
EXIT_DEPENDENCY = 11
EXIT_RVO2 = 12
EXIT_GYM_REGISTRATION = 20
EXIT_WORLD_WRAPPER = 21
EXIT_GYM_API = 22
EXIT_WORLD_OBSERVATION = 23
EXIT_DETERMINISM = 24
EXIT_HEADLESS_ROLLOUT = 25
EXIT_TIMEOUT = 30
EXIT_INTERNAL = 70

STAGE_NAMES = (
    "candidate",
    "dependencies",
    "revision",
    "rvo2_native",
    "gym_registration",
    "base_environment",
    "world_wrapper",
    "gym_api",
    "world_observation",
    "determinism",
    "headless_rollout",
)


class ContractViolation(RuntimeError):
    """A dependency imported, but did not satisfy the required contract."""


class ProbeTimeout(RuntimeError):
    """The configured wall-clock deadline expired."""


class StageFailure(RuntimeError):
    """A named probe stage failed with a stable process exit code."""

    def __init__(self, stage: str, exit_code: int, cause: BaseException) -> None:
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.exit_code = exit_code
        self.cause = cause


@dataclass(frozen=True)
class Candidate:
    """One pinned upstream contract candidate."""

    name: str
    commit: str
    distribution_version: str
    env_id: str
    entry_point: str
    config_path: Path
    status: str
    python_minor: str | None = None
    gymnasium_version: str | None = None
    numpy_version: str | None = None
    time_step: float | None = None
    human_count: int | None = None
    robot_type: str | None = None
    human_policy: str | None = None
    set_shape: str | None = None
    padded_observations: bool | None = None
    robot_world_dim: int | None = None
    human_world_stride: int | None = None
    reason: str | None = None


@dataclass
class CheckResult:
    """Serializable result of one probe stage."""

    name: str
    status: str
    duration_seconds: float
    details: dict[str, Any]


def extract_direct_url_commit(direct_url_text: str | None) -> str | None:
    """Extract a PEP 610 VCS commit from ``direct_url.json`` text."""
    if not direct_url_text:
        return None
    try:
        document = json.loads(direct_url_text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    vcs_info = document.get("vcs_info")
    if not isinstance(vcs_info, Mapping):
        return None
    commit = vcs_info.get("commit_id")
    return str(commit) if commit else None


def load_candidate(
    manifest_path: Path,
    candidate_name: str | None = None,
) -> Candidate:
    """Load a candidate and resolve its config relative to the repository."""
    manifest_path = manifest_path.expanduser().resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractViolation(
            f"Candidate manifest not found: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"Invalid candidate manifest: {exc}") from exc

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ContractViolation(
            f"Expected manifest schema {SCHEMA_VERSION}, got "
            f"{document.get('schema_version')!r}"
        )
    selected_name = candidate_name or document.get("selected")
    candidates = document.get("candidates", {})
    if selected_name not in candidates:
        raise ContractViolation(f"Unknown SocNavGym candidate: {selected_name!r}")

    raw = candidates[selected_name]
    repository_root = manifest_path.parents[2]
    config_value = raw.get("config")
    config_path = (
        (repository_root / config_value).resolve() if config_value else repository_root
    )
    return Candidate(
        name=selected_name,
        commit=str(raw["commit"]),
        distribution_version=str(raw["distribution_version"]),
        env_id=str(raw["env_id"]),
        entry_point=str(raw["entry_point"]),
        config_path=config_path,
        status=str(raw["status"]),
        python_minor=raw.get("python_minor"),
        gymnasium_version=raw.get("gymnasium_version"),
        numpy_version=raw.get("numpy_version"),
        time_step=raw.get("time_step"),
        human_count=raw.get("human_count"),
        robot_type=raw.get("robot_type"),
        human_policy=raw.get("human_policy"),
        set_shape=raw.get("set_shape"),
        padded_observations=raw.get("padded_observations"),
        robot_world_dim=raw.get("robot_world_dim"),
        human_world_stride=raw.get("human_world_stride"),
        reason=raw.get("reason"),
    )


def validate_world_frame_observation(
    observation: Mapping[str, Any],
    base_env: Any,
    *,
    robot_dim: int = 16,
    human_stride: int = 14,
) -> dict[str, Any]:
    """Validate the v1 ``WorldFrameObservations`` numerical layout.

    This check intentionally targets the selected v1 contract.  It verifies
    world-frame values against the unwrapped simulator state rather than only
    checking array shapes.
    """
    if not isinstance(observation, Mapping):
        raise ContractViolation("World-frame observation must be a mapping")
    if "robot" not in observation or "humans" not in observation:
        raise ContractViolation("World-frame observation needs robot and humans keys")

    robot_obs = np.asarray(observation["robot"])
    if robot_obs.shape != (robot_dim,):
        raise ContractViolation(
            f"Expected robot observation shape ({robot_dim},), got {robot_obs.shape}"
        )
    if robot_obs.dtype != np.float32:
        raise ContractViolation(
            f"Expected float32 robot observation, got {robot_obs.dtype}"
        )
    if not np.isfinite(robot_obs).all():
        raise ContractViolation("Robot observation contains NaN or infinity")
    expected_robot_encoding = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
    if not np.array_equal(robot_obs[:6], expected_robot_encoding):
        raise ContractViolation("Robot one-hot encoding does not match the v1 contract")

    robot = base_env.robot
    expected_robot_tail = np.asarray(
        [
            robot.goal_x,
            robot.goal_y,
            robot.x,
            robot.y,
            np.sin(robot.orientation),
            np.cos(robot.orientation),
            robot.vel_x,
            robot.vel_y,
            robot.vel_a,
            base_env.ROBOT_RADIUS,
        ],
        dtype=np.float32,
    )
    if not np.allclose(robot_obs[6:], expected_robot_tail, rtol=1e-6, atol=1e-6):
        raise ContractViolation(
            "World-frame robot fields do not match env.unwrapped.robot"
        )

    if getattr(base_env, "get_padded_observations", False):
        raise ContractViolation("The controlled probe requires padding to be disabled")
    interaction_lists = (
        getattr(base_env, "moving_interactions", []),
        getattr(base_env, "static_interactions", []),
        getattr(base_env, "h_l_interactions", []),
    )
    if any(interaction_lists):
        raise ContractViolation(
            "The controlled probe requires interactions to be disabled"
        )

    humans = list(getattr(base_env, "static_humans", [])) + list(
        getattr(base_env, "dynamic_humans", [])
    )
    human_obs = np.asarray(observation["humans"])
    if human_obs.ndim != 1 or human_obs.size % human_stride != 0:
        raise ContractViolation(
            f"Human observation must be flat with stride {human_stride}, "
            f"got shape {human_obs.shape}"
        )
    if human_obs.dtype != np.float32:
        raise ContractViolation(
            f"Expected float32 human observation, got {human_obs.dtype}"
        )
    human_rows = human_obs.reshape(-1, human_stride)
    if human_rows.shape[0] != len(humans):
        raise ContractViolation(
            "Human row count does not match the unwrapped fixed-human lists: "
            f"{human_rows.shape[0]} != {len(humans)}"
        )
    if not np.isfinite(human_rows).all():
        raise ContractViolation("Human observation contains NaN or infinity")

    human_ids: list[int] = []
    for row, human in zip(human_rows, humans):
        expected_human_encoding = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.float32)
        if not np.array_equal(row[:6], expected_human_encoding):
            raise ContractViolation(
                "Human one-hot encoding does not match the v1 contract"
            )
        expected_tail = np.asarray(
            [
                human.x,
                human.y,
                np.sin(human.orientation),
                np.cos(human.orientation),
                human.width / 2,
                human.speed * np.cos(human.orientation),
                human.speed * np.sin(human.orientation),
            ],
            dtype=np.float32,
        )
        if not np.allclose(row[6:13], expected_tail, rtol=1e-5, atol=1e-5):
            raise ContractViolation(
                "World-frame human fields do not match the unwrapped human state"
            )
        relative_bearing = (
            np.arctan2(
                robot.y - human.y,
                robot.x - human.x,
            )
            - human.orientation
        )
        relative_bearing = np.arctan2(
            np.sin(relative_bearing),
            np.cos(relative_bearing),
        )
        gaze_half_angle = base_env.HUMAN_GAZE_ANGLE / 2
        expected_gaze = float(-gaze_half_angle <= relative_bearing <= gaze_half_angle)
        if not np.isclose(row[13], expected_gaze, rtol=0.0, atol=1e-6):
            raise ContractViolation(
                "World-frame human gaze does not match the unwrapped state"
            )
        human_id = getattr(human, "id", None)
        if human_id is None:
            raise ContractViolation("An unwrapped human has no stable id field")
        human_ids.append(int(human_id))

    if len(human_ids) != len(set(human_ids)):
        raise ContractViolation("Human ids are not unique within the episode")
    return {
        "robot_shape": list(robot_obs.shape),
        "human_shape": list(human_obs.shape),
        "human_count": len(humans),
        "human_ids": human_ids,
        "human_ids_exposed_by_wrapper": False,
    }


def _git_commit_near(package_file: str | None) -> str | None:
    if not package_file:
        return None
    package_path = Path(package_file).resolve()
    for directory in (package_path.parent, *package_path.parents):
        if not (directory / ".git").exists():
            continue
        expected_package_file = (directory / "socnavgym" / "__init__.py").resolve()
        if expected_package_file != package_path:
            continue
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    return None


def _assert_distribution_owns_import(
    distribution: metadata.Distribution,
    socnavgym_module: Any,
) -> None:
    """Prove that the inspected dist-info owns the imported package source."""
    module_file = getattr(socnavgym_module, "__file__", None)
    if module_file is None:
        raise ContractViolation("Imported socnavgym has no __file__ provenance")
    imported_init = Path(module_file).resolve()

    distribution_inits: list[Path] = []
    for item in distribution.files or ():
        item_parts = tuple(item.parts)
        if item_parts[-2:] == ("socnavgym", "__init__.py"):
            distribution_inits.append(
                Path(str(distribution.locate_file(item))).resolve()
            )
    if distribution_inits:
        if imported_init not in distribution_inits:
            raise ContractViolation(
                "The imported socnavgym module is not owned by the inspected "
                f"distribution: {imported_init} not in {distribution_inits}"
            )
        return

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text).get("url", "")
        except json.JSONDecodeError:
            direct_url = ""
        parsed_url = urlparse(direct_url)
        if parsed_url.scheme == "file":
            source_root = Path(unquote(parsed_url.path)).resolve()
            expected_init = (source_root / "socnavgym" / "__init__.py").resolve()
            if imported_init == expected_init:
                return

    raise ContractViolation(
        "Cannot prove that the socnavgym dist-info belongs to the imported module"
    )


def _installed_socnavgym_commit(socnavgym_module: Any) -> tuple[str | None, str]:
    try:
        distribution = metadata.distribution("socnavgym")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        _assert_distribution_owns_import(distribution, socnavgym_module)
        direct_url_commit = extract_direct_url_commit(
            distribution.read_text("direct_url.json")
        )
        if direct_url_commit:
            return direct_url_commit, "PEP 610 direct_url.json"
    git_commit = _git_commit_near(getattr(socnavgym_module, "__file__", None))
    if git_commit:
        return git_commit, "git rev-parse"
    return None, "unverifiable"


def _module_version(module: Any) -> str | None:
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _assert_observation_space_contains(env: Any, observation: Any) -> None:
    if not env.observation_space.contains(observation):
        raise ContractViolation("observation_space.contains(observation) is false")


def _assert_observations_equal(left: Any, right: Any, path: str = "obs") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise ContractViolation(f"Determinism mismatch at {path}: keys differ")
        for key in sorted(left):
            _assert_observations_equal(left[key], right[key], f"{path}.{key}")
        return
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or not np.allclose(
        left_array,
        right_array,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ContractViolation(f"Determinism mismatch at {path}")


def _make_wrapped_env(context: dict[str, Any]) -> Any:
    gym = context["modules"]["gymnasium"]
    wrapper_class = context["world_wrapper_class"]
    candidate: Candidate = context["candidate"]
    base = gym.make(candidate.env_id, config=str(candidate.config_path))
    return wrapper_class(base)


def _zero_action(env: Any) -> np.ndarray:
    action_space = env.action_space
    if action_space.shape != (3,):
        raise ContractViolation(
            f"Expected continuous action shape (3,), got {action_space.shape}"
        )
    if action_space.dtype != np.float32:
        raise ContractViolation(
            f"Expected float32 action space, got {action_space.dtype}"
        )
    robot_type = getattr(getattr(env.unwrapped, "robot", None), "type", None)
    if robot_type == "diff-drive":
        expected_low = np.asarray([-1.0, 0.0, -1.0], dtype=np.float32)
        expected_high = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    elif robot_type == "holonomic":
        expected_low = np.full(3, -1.0, dtype=np.float32)
        expected_high = np.full(3, 1.0, dtype=np.float32)
    else:
        raise ContractViolation(f"Unsupported robot action contract: {robot_type!r}")
    if not np.array_equal(action_space.low, expected_low):
        raise ContractViolation(
            f"Expected normalized action lower bounds {expected_low.tolist()}"
        )
    if not np.array_equal(action_space.high, expected_high):
        raise ContractViolation(
            f"Expected normalized action upper bounds {expected_high.tolist()}"
        )
    action = np.zeros(3, dtype=np.float32)
    if not action_space.contains(action):
        raise ContractViolation("Float32 zero action is not in the action space")
    return action


def _collect_rollout(
    context: dict[str, Any],
    *,
    seed: int,
    steps: int,
) -> list[tuple[Any, float | None, bool | None, bool | None]]:
    env = _make_wrapped_env(context)
    trajectory: list[tuple[Any, float | None, bool | None, bool | None]] = []
    candidate: Candidate = context["candidate"]
    expected_human_ids: tuple[int, ...] | None = None

    def validate_observation(observation: Any) -> None:
        nonlocal expected_human_ids
        _assert_observation_space_contains(env, observation)
        if candidate.robot_world_dim is None or candidate.human_world_stride is None:
            raise ContractViolation(
                "Candidate does not declare world-frame observation dimensions"
            )
        details = validate_world_frame_observation(
            observation,
            env.unwrapped,
            robot_dim=candidate.robot_world_dim,
            human_stride=candidate.human_world_stride,
        )
        if (
            candidate.human_count is not None
            and details["human_count"] != candidate.human_count
        ):
            raise ContractViolation(
                f"Expected {candidate.human_count} humans, got "
                f"{details['human_count']}"
            )
        current_human_ids = tuple(details["human_ids"])
        if expected_human_ids is None:
            expected_human_ids = current_human_ids
        elif current_human_ids != expected_human_ids:
            raise ContractViolation(
                "Human id order changed within a fixed-human episode: "
                f"{expected_human_ids} -> {current_human_ids}"
            )

    try:
        reset_result = env.reset(seed=seed)
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise ContractViolation("reset() must return (observation, info)")
        observation, _ = reset_result
        validate_observation(observation)
        trajectory.append((observation, None, None, None))
        action = _zero_action(env)
        for _ in range(steps):
            step_result = env.step(action)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise ContractViolation(
                    "step() must return (observation, reward, terminated, truncated, info)"
                )
            observation, reward, terminated, truncated, _ = step_result
            validate_observation(observation)
            trajectory.append(
                (observation, float(reward), bool(terminated), bool(truncated))
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return trajectory


def _assert_rollouts_equal(
    left: Sequence[tuple[Any, float | None, bool | None, bool | None]],
    right: Sequence[tuple[Any, float | None, bool | None, bool | None]],
) -> None:
    if len(left) != len(right):
        raise ContractViolation("Seed replay produced different trajectory lengths")
    for index, (left_step, right_step) in enumerate(zip(left, right)):
        _assert_observations_equal(left_step[0], right_step[0], f"step[{index}]")
        if left_step[1:] != right_step[1:]:
            raise ContractViolation(
                f"Seed replay reward/termination mismatch at step {index}"
            )


@contextmanager
def _deadline(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise ProbeTimeout(f"Probe exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class ProbeSession:
    """Run ordered checks and retain a stable JSON report."""

    def __init__(self, candidate_name: str | None) -> None:
        self.started_monotonic = time.perf_counter()
        self.context: dict[str, Any] = {}
        self.current_stage: str | None = None
        self.report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "candidate": candidate_name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": {
                "python": sys.version,
                "implementation": platform.python_implementation(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "checks": [],
            "success": False,
            "exit_code": None,
        }

    def stage(
        self,
        name: str,
        exit_code: int,
        operation: Callable[[], dict[str, Any] | None],
    ) -> None:
        self.current_stage = name
        started = time.perf_counter()
        try:
            details = operation() or {}
        except ProbeTimeout:
            raise
        except BaseException as exc:
            duration = time.perf_counter() - started
            self.report["checks"].append(
                asdict(
                    CheckResult(
                        name=name,
                        status="fail",
                        duration_seconds=duration,
                        details={
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
            )
            raise StageFailure(name, exit_code, exc) from exc
        duration = time.perf_counter() - started
        self.report["checks"].append(
            asdict(
                CheckResult(
                    name=name,
                    status="pass",
                    duration_seconds=duration,
                    details=details,
                )
            )
        )
        self.current_stage = None

    def finish(self, exit_code: int) -> dict[str, Any]:
        completed = {check["name"] for check in self.report["checks"]}
        for name in STAGE_NAMES:
            if name not in completed:
                self.report["checks"].append(
                    asdict(
                        CheckResult(
                            name=name,
                            status="skipped_due_to_previous_failure",
                            duration_seconds=0.0,
                            details={},
                        )
                    )
                )
        self.report["success"] = exit_code == EXIT_OK
        self.report["exit_code"] = exit_code
        self.report["duration_seconds"] = time.perf_counter() - self.started_monotonic
        self.report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        return self.report


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Execute the complete probe and return its report and exit code."""
    session = ProbeSession(args.candidate)
    context = session.context
    exit_code = EXIT_OK

    try:
        with _deadline(args.timeout_seconds):

            def load_candidate_stage() -> dict[str, Any]:
                candidate = load_candidate(args.manifest, args.candidate)
                if args.config is not None:
                    candidate = Candidate(
                        **{
                            **asdict(candidate),
                            "config_path": args.config.expanduser().resolve(),
                        }
                    )
                if candidate.status != "probe":
                    raise ContractViolation(
                        candidate.reason
                        or f"Candidate {candidate.name} is not approved for probing"
                    )
                if not candidate.config_path.is_file():
                    raise ContractViolation(
                        f"SocNavGym config not found: {candidate.config_path}"
                    )
                context["candidate"] = candidate
                session.report["candidate"] = candidate.name
                return {
                    "commit": candidate.commit,
                    "distribution_version": candidate.distribution_version,
                    "env_id": candidate.env_id,
                    "config": str(candidate.config_path),
                }

            session.stage("candidate", EXIT_CONFIG, load_candidate_stage)

            def dependency_stage() -> dict[str, Any]:
                candidate: Candidate = context["candidate"]
                modules: dict[str, Any] = {}
                versions: dict[str, str | None] = {}
                failures: dict[str, str] = {}
                actual_python_minor = (
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                )
                if (
                    candidate.python_minor is not None
                    and actual_python_minor != candidate.python_minor
                ):
                    failures["python"] = (
                        f"The pinned probe requires Python {candidate.python_minor}; found "
                        f"{platform.python_version()}"
                    )
                names = (
                    "numpy",
                    "torch",
                    "gymnasium",
                    "yaml",
                    "cv2",
                    "matplotlib",
                    "shapely",
                    "dgl",
                    "rvo2",
                    "socnavgym",
                )
                for name in names:
                    try:
                        module = importlib.import_module(name)
                    except BaseException as exc:
                        failures[name] = f"{type(exc).__name__}: {exc}"
                    else:
                        modules[name] = module
                        versions[name] = _module_version(module)
                expected_versions = {
                    "numpy": candidate.numpy_version,
                    "gymnasium": candidate.gymnasium_version,
                }
                for name, expected in expected_versions.items():
                    actual = versions.get(name)
                    if expected is not None and actual != expected:
                        failures[f"{name}_version"] = (
                            f"Expected {expected}, found {actual}"
                        )
                if failures:
                    raise ContractViolation(
                        "Dependency contract failed: "
                        + json.dumps(failures, sort_keys=True)
                    )
                context["modules"] = modules
                return {"versions": versions}

            session.stage("dependencies", EXIT_DEPENDENCY, dependency_stage)

            def revision_stage() -> dict[str, Any]:
                candidate: Candidate = context["candidate"]
                installed_commit, source = _installed_socnavgym_commit(
                    context["modules"]["socnavgym"]
                )
                if installed_commit is None:
                    raise ContractViolation(
                        "Cannot prove the installed SocNavGym commit using PEP 610 "
                        "metadata or a nearby git checkout"
                    )
                if installed_commit.lower() != candidate.commit.lower():
                    raise ContractViolation(
                        f"Installed SocNavGym commit {installed_commit} does not match "
                        f"the pinned commit {candidate.commit}"
                    )
                try:
                    distribution_version = metadata.version("socnavgym")
                except metadata.PackageNotFoundError:
                    distribution_version = None
                if distribution_version != candidate.distribution_version:
                    raise ContractViolation(
                        f"Installed distribution version {distribution_version!r} does "
                        f"not match {candidate.distribution_version!r}"
                    )
                return {
                    "installed_commit": installed_commit,
                    "commit_source": source,
                    "distribution_version": distribution_version,
                }

            session.stage("revision", EXIT_REVISION, revision_stage)

            def rvo2_stage() -> dict[str, Any]:
                rvo2 = context["modules"]["rvo2"]
                simulator = rvo2.PyRVOSimulator(
                    0.1,
                    5.0,
                    10,
                    5.0,
                    5.0,
                    0.3,
                    1.0,
                )
                first = simulator.addAgent((-1.0, 0.0))
                second = simulator.addAgent((1.0, 0.0))
                before = np.asarray(
                    [
                        simulator.getAgentPosition(first),
                        simulator.getAgentPosition(second),
                    ],
                    dtype=np.float64,
                )
                simulator.setAgentPrefVelocity(first, (0.5, 0.0))
                simulator.setAgentPrefVelocity(second, (-0.5, 0.0))
                simulator.doStep()
                after = np.asarray(
                    [
                        simulator.getAgentPosition(first),
                        simulator.getAgentPosition(second),
                    ],
                    dtype=np.float64,
                )
                if not np.isfinite(after).all():
                    raise ContractViolation("RVO2 returned a non-finite position")
                if np.allclose(before, after):
                    raise ContractViolation("RVO2 doStep() did not move either agent")
                global_time = float(simulator.getGlobalTime())
                if not np.isclose(global_time, 0.1, rtol=0.0, atol=1e-6):
                    raise ContractViolation(
                        f"RVO2 global time did not advance by 0.1: {global_time}"
                    )
                return {
                    "global_time": global_time,
                    "positions_before": before.tolist(),
                    "positions_after": after.tolist(),
                }

            session.stage("rvo2_native", EXIT_RVO2, rvo2_stage)

            def registration_stage() -> dict[str, Any]:
                candidate: Candidate = context["candidate"]
                gym = context["modules"]["gymnasium"]
                specification = gym.spec(candidate.env_id)
                entry_point = str(specification.entry_point)
                if entry_point != candidate.entry_point:
                    raise ContractViolation(
                        f"Expected entry point {candidate.entry_point!r}, got "
                        f"{entry_point!r}"
                    )
                return {"entry_point": entry_point}

            session.stage(
                "gym_registration",
                EXIT_GYM_REGISTRATION,
                registration_stage,
            )

            def base_environment_stage() -> dict[str, Any]:
                gym = context["modules"]["gymnasium"]
                candidate: Candidate = context["candidate"]
                base_env = gym.make(
                    candidate.env_id,
                    config=str(candidate.config_path),
                )
                context["base_env"] = base_env
                unwrapped = base_env.unwrapped
                if candidate.time_step is not None and not np.isclose(
                    float(unwrapped.TIMESTEP),
                    candidate.time_step,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ContractViolation(
                        f"Expected time_step {candidate.time_step}, got "
                        f"{unwrapped.TIMESTEP}"
                    )
                if (
                    candidate.robot_type is not None
                    and unwrapped.robot.type != candidate.robot_type
                ):
                    raise ContractViolation(
                        f"Expected robot_type {candidate.robot_type!r}, got "
                        f"{unwrapped.robot.type!r}"
                    )
                if (
                    candidate.human_policy is not None
                    and unwrapped.HUMAN_POLICY != candidate.human_policy
                ):
                    raise ContractViolation(
                        f"Expected human_policy {candidate.human_policy!r}, got "
                        f"{unwrapped.HUMAN_POLICY!r}"
                    )
                if (
                    candidate.set_shape is not None
                    and unwrapped.set_shape != candidate.set_shape
                ):
                    raise ContractViolation(
                        f"Expected set_shape {candidate.set_shape!r}, got "
                        f"{unwrapped.set_shape!r}"
                    )
                if (
                    candidate.padded_observations is not None
                    and bool(unwrapped.get_padded_observations)
                    != candidate.padded_observations
                ):
                    raise ContractViolation(
                        "The environment did not apply the pinned padding setting"
                    )
                humans = list(unwrapped.static_humans) + list(unwrapped.dynamic_humans)
                if (
                    candidate.human_count is not None
                    and len(humans) != candidate.human_count
                ):
                    raise ContractViolation(
                        f"Expected {candidate.human_count} humans, got {len(humans)}"
                    )
                return {
                    "base_class": type(unwrapped).__qualname__,
                    "time_step": float(unwrapped.TIMESTEP),
                    "robot_type": unwrapped.robot.type,
                    "human_policy": unwrapped.HUMAN_POLICY,
                    "human_count": len(humans),
                    "set_shape": unwrapped.set_shape,
                    "padded_observations": bool(unwrapped.get_padded_observations),
                }

            session.stage(
                "base_environment",
                EXIT_GYM_REGISTRATION,
                base_environment_stage,
            )

            def wrapper_stage() -> dict[str, Any]:
                wrapper_module = importlib.import_module("socnavgym.wrappers")
                wrapper_class = getattr(wrapper_module, "WorldFrameObservations")
                context["world_wrapper_class"] = wrapper_class
                base_env = context["base_env"]
                env = wrapper_class(base_env)
                context.pop("base_env")
                context["env"] = env
                return {
                    "wrapper": (
                        f"{wrapper_class.__module__}.{wrapper_class.__qualname__}"
                    )
                }

            session.stage("world_wrapper", EXIT_WORLD_WRAPPER, wrapper_stage)

            def api_stage() -> dict[str, Any]:
                env = context["env"]
                reset_result = env.reset(seed=args.seed)
                if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                    raise ContractViolation("reset() must return a 2-tuple")
                initial_observation, reset_info = reset_result
                if not isinstance(reset_info, Mapping):
                    raise ContractViolation("reset info must be a mapping")
                _assert_observation_space_contains(env, initial_observation)
                action = _zero_action(env)
                step_result = env.step(action)
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    raise ContractViolation("step() must return a 5-tuple")
                next_observation, reward, terminated, truncated, step_info = step_result
                if not isinstance(step_info, Mapping):
                    raise ContractViolation("step info must be a mapping")
                if not np.isfinite(float(reward)):
                    raise ContractViolation("step reward is not finite")
                if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
                    truncated, (bool, np.bool_)
                ):
                    raise ContractViolation("terminated and truncated must be booleans")
                _assert_observation_space_contains(env, next_observation)
                context["initial_observation"] = initial_observation
                context["next_observation"] = next_observation
                return {
                    "action_shape": list(action.shape),
                    "action_dtype": str(action.dtype),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }

            session.stage("gym_api", EXIT_GYM_API, api_stage)

            def world_observation_stage() -> dict[str, Any]:
                candidate: Candidate = context["candidate"]
                if (
                    candidate.robot_world_dim is None
                    or candidate.human_world_stride is None
                ):
                    raise ContractViolation(
                        "Candidate does not declare world-frame observation dimensions"
                    )
                env = context["env"]
                initial_observation, _ = env.reset(seed=args.seed)
                initial_details = validate_world_frame_observation(
                    initial_observation,
                    env.unwrapped,
                    robot_dim=candidate.robot_world_dim,
                    human_stride=candidate.human_world_stride,
                )
                next_observation, _, _, _, _ = env.step(_zero_action(env))
                next_details = validate_world_frame_observation(
                    next_observation,
                    env.unwrapped,
                    robot_dim=candidate.robot_world_dim,
                    human_stride=candidate.human_world_stride,
                )
                if initial_details["human_ids"] != next_details["human_ids"]:
                    raise ContractViolation(
                        "Human id order changed across the first environment step"
                    )
                if (
                    candidate.human_count is not None
                    and next_details["human_count"] != candidate.human_count
                ):
                    raise ContractViolation(
                        f"Expected {candidate.human_count} humans, got "
                        f"{next_details['human_count']}"
                    )
                return {
                    **next_details,
                    "ids_stable_across_first_step": True,
                }

            session.stage(
                "world_observation",
                EXIT_WORLD_OBSERVATION,
                world_observation_stage,
            )

            def determinism_stage() -> dict[str, Any]:
                first = _collect_rollout(
                    context,
                    seed=args.seed,
                    steps=args.determinism_steps,
                )
                second = _collect_rollout(
                    context,
                    seed=args.seed,
                    steps=args.determinism_steps,
                )
                _assert_rollouts_equal(first, second)
                return {"compared_transitions": len(first) - 1}

            session.stage("determinism", EXIT_DETERMINISM, determinism_stage)

            def headless_rollout_stage() -> dict[str, Any]:
                completed_steps = 0
                episodes = 0
                while completed_steps < args.steps:
                    remaining = args.steps - completed_steps
                    trajectory = _collect_rollout(
                        context,
                        seed=args.seed + episodes,
                        steps=remaining,
                    )
                    transitions = len(trajectory) - 1
                    if transitions <= 0:
                        raise ContractViolation(
                            "Headless rollout completed no environment transitions"
                        )
                    completed_steps += transitions
                    episodes += 1
                return {
                    "completed_steps": completed_steps,
                    "episodes": episodes,
                    "render_called": False,
                }

            session.stage(
                "headless_rollout",
                EXIT_HEADLESS_ROLLOUT,
                headless_rollout_stage,
            )
    except ProbeTimeout as exc:
        exit_code = EXIT_TIMEOUT
        session.report["checks"].append(
            asdict(
                CheckResult(
                    name=session.current_stage or "timeout",
                    status="fail",
                    duration_seconds=0.0,
                    details={
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            )
        )
    except StageFailure as exc:
        exit_code = exc.exit_code
    except BaseException as exc:
        exit_code = EXIT_INTERNAL
        session.report["checks"].append(
            asdict(
                CheckResult(
                    name=session.current_stage or "internal",
                    status="fail",
                    duration_seconds=0.0,
                    details={
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        )
    finally:
        env = context.get("env")
        close_target = env if env is not None else context.get("base_env")
        if close_target is not None:
            try:
                close_target.close()
            except BaseException as exc:
                if exit_code == EXIT_OK:
                    exit_code = EXIT_HEADLESS_ROLLOUT
                    session.report["checks"].append(
                        asdict(
                            CheckResult(
                                name="close",
                                status="fail",
                                duration_seconds=0.0,
                                details={
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                },
                            )
                        )
                    )

    return session.finish(exit_code), exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the pinned SocNavGym/RVO2 contract on an HPC node."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Candidate manifest (default: repository manifest).",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Candidate key; defaults to the manifest's selected candidate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override the candidate's controlled environment config.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--determinism-steps", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional authoritative JSON report path; the report is also printed "
            "to stdout, where dependency imports may have emitted other text."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.determinism_steps <= 0 or args.timeout_seconds <= 0:
        parser.error(
            "--steps and --determinism-steps must be positive; "
            "--timeout-seconds must be positive"
        )

    report, exit_code = run_probe(args)
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        try:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text + "\n", encoding="utf-8")
        except OSError as exc:
            exit_code = EXIT_INTERNAL
            report["success"] = False
            report["exit_code"] = exit_code
            report["checks"].append(
                asdict(
                    CheckResult(
                        name="output_report",
                        status="fail",
                        duration_seconds=0.0,
                        details={
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
            )
            report_text = json.dumps(report, indent=2, sort_keys=True)
    print(report_text, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
