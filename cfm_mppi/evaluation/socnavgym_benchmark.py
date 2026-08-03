"""Locked SocNavGym benchmark manifest, job mapping, and result validation.

The formal benchmark is intentionally data driven.  A manifest fixes the
scenario configurations, environment seeds, planning budget, and shard size;
the suite command is not allowed to override any of those experimental
choices.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_SUITE = (
    REPOSITORY_ROOT / "configs" / "socnavgym" / "benchmark_v1" / "suite.json"
)
CANONICAL_PLANNERS = ("cfm-mppi-cv", "vrc-mppi")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkContractError(ValueError):
    """The locked benchmark contract or one of its results is invalid."""


@dataclass(frozen=True)
class BenchmarkScenario:
    """One fixed SocNavGym scenario in the benchmark matrix."""

    id: str
    role: str
    config_path: Path
    config_sha256: str
    human_policy: str
    dynamic_humans: int
    static_humans: int
    map_size_metres: tuple[int, int]
    robot_start: tuple[float, float] | None
    robot_goal: tuple[float, float] | None


@dataclass(frozen=True)
class BenchmarkJob:
    """One independently rerunnable paired scenario-seed benchmark shard."""

    index: int
    scenario_index: int
    shard_index: int
    scenario: BenchmarkScenario
    seed_indices: tuple[int, ...]
    env_seeds: tuple[int, ...]

    @property
    def relative_output_path(self) -> Path:
        seed_label = "-".join(str(seed) for seed in self.env_seeds)
        return Path("shards") / self.scenario.id / f"seeds-{seed_label}.json"


@dataclass(frozen=True)
class BenchmarkSuite:
    """Fully resolved and validated benchmark definition."""

    manifest_path: Path
    manifest_sha256: str
    suite_id: str
    description: str
    environment_id: str
    observation_wrapper: str
    socnavgym_commit: str
    human_goal_reached_policy: str
    seed_path: Path
    seed_sha256: str
    checkpoint_sha256: str
    rvo2_module_sha256: str
    dgl_backend: str
    device: str
    cuda_device_contains: str
    development_seeds: tuple[int, ...]
    benchmark_seeds: tuple[int, ...]
    planner_seed_offset: int
    seeds_per_job: int
    output_subdirectory: Path
    planner: Mapping[str, int | str]
    scenarios: tuple[BenchmarkScenario, ...]
    jobs: tuple[BenchmarkJob, ...]
    expected_scenario_seed_pairs: int

    def job(self, index: int) -> BenchmarkJob:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("job index must be an integer")
        if index < 0 or index >= len(self.jobs):
            raise IndexError(
                f"job index {index} is outside the locked range 0..{len(self.jobs) - 1}"
            )
        return self.jobs[index]

    def default_output_root(self) -> Path:
        return REPOSITORY_ROOT / "output_dir" / self.output_subdirectory

    def output_path(self, output_root: Path, job: BenchmarkJob) -> Path:
        return Path(output_root).resolve() / job.relative_output_path


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_benchmark_suite(
    manifest_path: Path = DEFAULT_BENCHMARK_SUITE,
) -> BenchmarkSuite:
    """Load the locked manifest and reject any contract drift."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json_mapping(manifest_path, "benchmark manifest")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "suite_id",
            "description",
            "locked",
            "environment_id",
            "observation_wrapper",
            "socnavgym_commit",
            "human_goal_reached_policy",
            "seed_file",
            "seed_file_sha256",
            "checkpoint_sha256",
            "rvo2_module_sha256",
            "dgl_backend",
            "device",
            "cuda_device_contains",
            "planner_seed_offset",
            "seeds_per_job",
            "expected_jobs",
            "expected_scenario_seed_pairs",
            "output_subdirectory",
            "planner",
            "scenarios",
        },
        "benchmark manifest",
    )
    if manifest["schema_version"] != 1 or manifest["locked"] is not True:
        raise BenchmarkContractError(
            "benchmark manifest must use schema_version=1 and locked=true"
        )

    suite_id = _nonempty_string(manifest["suite_id"], "suite_id")
    description = _nonempty_string(manifest["description"], "description")
    environment_id = _nonempty_string(manifest["environment_id"], "environment_id")
    observation_wrapper = _nonempty_string(
        manifest["observation_wrapper"], "observation_wrapper"
    )
    if environment_id != "SocNavGym-v1" or observation_wrapper != "WorldFrameObservations":
        raise BenchmarkContractError(
            "formal benchmark must use SocNavGym-v1 + WorldFrameObservations"
        )
    socnavgym_commit = _nonempty_string(
        manifest["socnavgym_commit"], "socnavgym_commit"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", socnavgym_commit):
        raise BenchmarkContractError("socnavgym_commit must be a full Git commit hash")
    human_goal_reached_policy = _nonempty_string(
        manifest["human_goal_reached_policy"], "human_goal_reached_policy"
    )
    if human_goal_reached_policy != "geometric-only-v1":
        raise BenchmarkContractError(
            "formal benchmark must disable SocNavGym's wall-clock human goal fallback"
        )

    seed_path = _resolve_relative_file(
        manifest_path.parent, manifest["seed_file"], "seed_file"
    )
    declared_seed_sha = _declared_sha256(
        manifest["seed_file_sha256"], "seed_file_sha256"
    )
    actual_seed_sha = sha256_file(seed_path)
    if actual_seed_sha != declared_seed_sha:
        raise BenchmarkContractError(
            f"seed file hash mismatch: expected {declared_seed_sha}, got {actual_seed_sha}"
        )
    seed_document = _read_json_mapping(seed_path, "benchmark seed file")
    _require_exact_keys(
        seed_document,
        {
            "schema_version",
            "suite_id",
            "locked",
            "selection_protocol",
            "development_seeds",
            "benchmark_seeds",
        },
        "benchmark seed file",
    )
    if (
        seed_document["schema_version"] != 1
        or seed_document["locked"] is not True
        or seed_document["suite_id"] != suite_id
    ):
        raise BenchmarkContractError(
            "seed file schema, lock, and suite_id must match the manifest"
        )
    _nonempty_string(seed_document["selection_protocol"], "selection_protocol")
    development_seeds = _seed_tuple(
        seed_document["development_seeds"], "development_seeds"
    )
    benchmark_seeds = _seed_tuple(
        seed_document["benchmark_seeds"], "benchmark_seeds"
    )
    if set(development_seeds) & set(benchmark_seeds):
        raise BenchmarkContractError(
            "development and benchmark seed sets must be disjoint"
        )

    checkpoint_sha256 = _declared_sha256(
        manifest["checkpoint_sha256"], "checkpoint_sha256"
    )
    rvo2_module_sha256 = _declared_sha256(
        manifest["rvo2_module_sha256"], "rvo2_module_sha256"
    )
    dgl_backend = _nonempty_string(manifest["dgl_backend"], "dgl_backend")
    if dgl_backend != "pytorch":
        raise BenchmarkContractError("formal benchmark requires DGLBACKEND=pytorch")
    device = _nonempty_string(manifest["device"], "device")
    cuda_device_contains = _nonempty_string(
        manifest["cuda_device_contains"], "cuda_device_contains"
    )
    if device != "cuda" or cuda_device_contains != "H100":
        raise BenchmarkContractError("benchmark_v1 is locked to an NVIDIA H100 GPU")

    planner = _planner_contract(manifest["planner"])
    planner_seed_offset = _nonnegative_int(
        manifest["planner_seed_offset"], "planner_seed_offset"
    )
    seeds_per_job = _positive_int(manifest["seeds_per_job"], "seeds_per_job")
    if seeds_per_job != 1:
        raise BenchmarkContractError(
            "benchmark_v1 requires one paired scenario-seed result per job"
        )
    if len(benchmark_seeds) % seeds_per_job:
        raise BenchmarkContractError(
            "benchmark seed count must be divisible by seeds_per_job"
        )

    output_subdirectory = _safe_relative_path(
        manifest["output_subdirectory"], "output_subdirectory"
    )
    raw_scenarios = manifest["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise BenchmarkContractError("scenarios must be a non-empty list")
    scenarios = tuple(
        _load_scenario(manifest_path.parent, raw, index)
        for index, raw in enumerate(raw_scenarios)
    )
    scenario_ids = [scenario.id for scenario in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise BenchmarkContractError("scenario ids must be unique")
    if len({scenario.id.casefold() for scenario in scenarios}) != len(scenarios):
        raise BenchmarkContractError("scenario ids must also be unique case-insensitively")
    config_paths = [scenario.config_path for scenario in scenarios]
    if len(set(config_paths)) != len(config_paths):
        raise BenchmarkContractError("every scenario must have its own config file")

    jobs: list[BenchmarkJob] = []
    shards_per_scenario = len(benchmark_seeds) // seeds_per_job
    for scenario_index, scenario in enumerate(scenarios):
        for shard_index in range(shards_per_scenario):
            start = shard_index * seeds_per_job
            stop = start + seeds_per_job
            jobs.append(
                BenchmarkJob(
                    index=len(jobs),
                    scenario_index=scenario_index,
                    shard_index=shard_index,
                    scenario=scenario,
                    seed_indices=tuple(range(start, stop)),
                    env_seeds=benchmark_seeds[start:stop],
                )
            )
    expected_pairs = len(scenarios) * len(benchmark_seeds)
    if _positive_int(manifest["expected_jobs"], "expected_jobs") != len(jobs):
        raise BenchmarkContractError("expected_jobs does not match the derived job matrix")
    if (
        _positive_int(
            manifest["expected_scenario_seed_pairs"],
            "expected_scenario_seed_pairs",
        )
        != expected_pairs
    ):
        raise BenchmarkContractError(
            "expected_scenario_seed_pairs does not match scenarios x seeds"
        )

    return BenchmarkSuite(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        suite_id=suite_id,
        description=description,
        environment_id=environment_id,
        observation_wrapper=observation_wrapper,
        socnavgym_commit=socnavgym_commit,
        human_goal_reached_policy=human_goal_reached_policy,
        seed_path=seed_path,
        seed_sha256=actual_seed_sha,
        checkpoint_sha256=checkpoint_sha256,
        rvo2_module_sha256=rvo2_module_sha256,
        dgl_backend=dgl_backend,
        device=device,
        cuda_device_contains=cuda_device_contains,
        development_seeds=development_seeds,
        benchmark_seeds=benchmark_seeds,
        planner_seed_offset=planner_seed_offset,
        seeds_per_job=seeds_per_job,
        output_subdirectory=output_subdirectory,
        planner=planner,
        scenarios=scenarios,
        jobs=tuple(jobs),
        expected_scenario_seed_pairs=expected_pairs,
    )


def benchmark_metadata(suite: BenchmarkSuite, job: BenchmarkJob) -> dict[str, Any]:
    """Build immutable experiment-identifying metadata for one shard."""
    metadata = {
        "schema_version": "cfm_mppi.socnavgym_benchmark_shard.v1",
        "suite_id": suite.suite_id,
        "manifest": str(suite.manifest_path),
        "manifest_sha256": suite.manifest_sha256,
        "seed_file": str(suite.seed_path),
        "seed_file_sha256": suite.seed_sha256,
        "socnavgym_commit": suite.socnavgym_commit,
        "job_index": job.index,
        "expected_jobs": len(suite.jobs),
        "expected_scenario_seed_pairs": suite.expected_scenario_seed_pairs,
        "scenario_id": job.scenario.id,
        "scenario_index": job.scenario_index,
        "scenario_role": job.scenario.role,
        "scenario_config_sha256": job.scenario.config_sha256,
        "human_policy": job.scenario.human_policy,
        "human_goal_reached_policy": suite.human_goal_reached_policy,
        "dynamic_humans": job.scenario.dynamic_humans,
        "static_humans": job.scenario.static_humans,
        "map_size_metres": list(job.scenario.map_size_metres),
        "shard_index": job.shard_index,
        "seed_indices": list(job.seed_indices),
        "environment_seeds": list(job.env_seeds),
        "planner_seeds": [
            seed + suite.planner_seed_offset for seed in job.env_seeds
        ],
        "execution_order_offset": job.index % 2,
        "device": suite.device,
        "cuda_device_contains": suite.cuda_device_contains,
    }
    if job.scenario.robot_start is not None:
        metadata["robot_start"] = list(job.scenario.robot_start)
        metadata["robot_goal"] = list(job.scenario.robot_goal)
    return metadata


def validate_shard_document(
    document: Mapping[str, Any],
    suite: BenchmarkSuite,
    job: BenchmarkJob,
    *,
    require_full_steps: bool = True,
) -> None:
    """Reject incomplete, smoke-only, or contract-mismatched shard output."""
    if not isinstance(document, Mapping):
        raise BenchmarkContractError("benchmark shard must be a JSON object")
    if document.get("schema_version") != "cfm_mppi.socnavgym_evaluation.v1":
        raise BenchmarkContractError("unexpected evaluation result schema")
    if document.get("environment_seeds") != list(job.env_seeds):
        raise BenchmarkContractError("shard environment seeds do not match its job")
    if document.get("planners") != list(CANONICAL_PLANNERS):
        raise BenchmarkContractError("formal shard must contain both canonical planners")
    expected_orders = [
        (
            list(CANONICAL_PLANNERS)
            if (job.index + seed_index) % 2 == 0
            else list(reversed(CANONICAL_PLANNERS))
        )
        for seed_index in range(len(job.env_seeds))
    ]
    if document.get("execution_orders") != expected_orders:
        raise BenchmarkContractError(
            "shard planner execution order differs from its counterbalance schedule"
        )

    metadata = document.get("metadata")
    benchmark = document.get("benchmark")
    if not isinstance(metadata, Mapping) or not isinstance(benchmark, Mapping):
        raise BenchmarkContractError("shard is missing metadata or benchmark contract")
    if metadata.get("random_model_smoke_only") is not False:
        raise BenchmarkContractError("random-model smoke output is not a benchmark result")
    if metadata.get("repository_dirty") is not False:
        raise BenchmarkContractError("formal shard must come from a clean repository")
    if metadata.get("checkpoint_sha256") != suite.checkpoint_sha256:
        raise BenchmarkContractError("checkpoint hash differs from the manifest")
    if metadata.get("device") != suite.device:
        raise BenchmarkContractError("formal benchmark must run on CUDA")
    cuda_device = metadata.get("cuda_device")
    if not isinstance(cuda_device, str) or suite.cuda_device_contains not in cuda_device:
        raise BenchmarkContractError("formal benchmark must run on an NVIDIA H100")
    if metadata.get("socnavgym_commit") != suite.socnavgym_commit:
        raise BenchmarkContractError("SocNavGym commit differs from the manifest")
    if metadata.get("environment_config_sha256") != job.scenario.config_sha256:
        raise BenchmarkContractError("environment config hash differs from the manifest")
    if metadata.get("planner_seed_offset") != suite.planner_seed_offset:
        raise BenchmarkContractError("planner seed offset differs from the manifest")
    if metadata.get("execution_order_offset") != job.index % 2:
        raise BenchmarkContractError("execution-order offset differs from its job")
    if (
        metadata.get("socnavgym_human_goal_policy")
        != suite.human_goal_reached_policy
    ):
        raise BenchmarkContractError("SocNavGym human goal policy differs from manifest")
    rvo2_module = metadata.get("rvo2_module")
    if (
        not isinstance(rvo2_module, Mapping)
        or rvo2_module.get("sha256") != suite.rvo2_module_sha256
    ):
        raise BenchmarkContractError("RVO2 binary hash differs from the manifest")
    recorded_planner = metadata.get("planner_config")
    if not isinstance(recorded_planner, Mapping):
        raise BenchmarkContractError("shard is missing recorded planner config")
    for key in (
        "horizon",
        "max_history",
        "cfm_candidates",
        "branches",
        "mppi_samples_per_branch",
    ):
        if recorded_planner.get(key) != suite.planner[key]:
            raise BenchmarkContractError("planner config differs from the manifest")
    expected_benchmark = benchmark_metadata(suite, job)
    if dict(benchmark) != expected_benchmark:
        raise BenchmarkContractError("embedded benchmark metadata differs from its job")

    summaries = document.get("summaries")
    episodes = document.get("episodes")
    expected_episode_keys = [
        (seed, planner) for seed in job.env_seeds for planner in CANONICAL_PLANNERS
    ]
    if not isinstance(summaries, list) or len(summaries) != len(expected_episode_keys):
        raise BenchmarkContractError("shard summary count is incomplete")
    if not isinstance(episodes, list) or len(episodes) != len(expected_episode_keys):
        raise BenchmarkContractError("shard episode count is incomplete")
    for expected, summary, episode in zip(expected_episode_keys, summaries, episodes):
        if not isinstance(summary, Mapping) or not isinstance(episode, Mapping):
            raise BenchmarkContractError("episode and summary entries must be objects")
        key = (summary.get("env_seed"), summary.get("planner"))
        if key != expected:
            raise BenchmarkContractError("summary order or identity is invalid")
        if episode.get("planner") != expected[1]:
            raise BenchmarkContractError("episode planner order is invalid")
        if summary.get("planner_seed") != expected[0] + suite.planner_seed_offset:
            raise BenchmarkContractError("summary planner seed is invalid")
        if episode.get("summary") != summary:
            raise BenchmarkContractError(
                "top-level and episode-local summaries are inconsistent"
            )
        expected_budget = {
            "cfm_candidates": suite.planner["cfm_candidates"],
            "refinement_rollouts": (
                suite.planner["branches"]
                * suite.planner["mppi_samples_per_branch"]
            ),
        }
        if episode.get("budget") != expected_budget:
            raise BenchmarkContractError("episode planning budget is invalid")
        context = episode.get("context")
        if not isinstance(context, Mapping) or context.get("env_seed") != expected[0]:
            raise BenchmarkContractError("episode context seed is invalid")
        if context.get("planner_seed") != expected[0] + suite.planner_seed_offset:
            raise BenchmarkContractError("episode planner seed is invalid")
        _validate_episode_summary(
            summary,
            episode,
            require_full_steps=require_full_steps,
            expected_human_count=(
                job.scenario.dynamic_humans + job.scenario.static_humans
            ),
            expected_robot_start=job.scenario.robot_start,
            expected_robot_goal=job.scenario.robot_goal,
        )

    for seed_offset in range(len(job.env_seeds)):
        first = episodes[seed_offset * len(CANONICAL_PLANNERS)]
        second = episodes[seed_offset * len(CANONICAL_PLANNERS) + 1]
        if first.get("initial_state") != second.get("initial_state"):
            raise BenchmarkContractError(
                "paired planners do not record the same initial environment state"
            )


def _validate_episode_summary(
    summary: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    require_full_steps: bool,
    expected_human_count: int,
    expected_robot_start: tuple[float, float] | None,
    expected_robot_goal: tuple[float, float] | None,
) -> None:
    import numpy as np

    expected_episode_keys = {
        "planner",
        "budget",
        "context",
        "initial_state",
        "reset_info",
        "summary",
    }
    if require_full_steps:
        expected_episode_keys.add("steps")
    if set(episode) != expected_episode_keys:
        missing = sorted(expected_episode_keys - set(episode))
        extra = sorted(set(episode) - expected_episode_keys)
        raise BenchmarkContractError(
            f"formal episode schema differs; missing={missing}, extra={extra}"
        )
    expected_keys = {
        "planner",
        "env_seed",
        "planner_seed",
        "steps",
        "simulation_seconds",
        "return",
        "terminated",
        "truncated",
        "runner_limit_reached",
        "success",
        "collision",
        "collision_any",
        "collision_human",
        "collision_object",
        "collision_wall",
        "out_of_map",
        "timeout",
        "environment_time_to_reach_goal",
        "environment_path_length",
        "environment_minimum_distance_to_human",
        "final_goal_distance",
        "geometric_path_length",
        "minimum_human_center_distance",
        "minimum_human_clearance",
        "freezing_events",
        "decision_latency",
        "bookkeeping_latency",
        "total_compute_latency",
        "cold_start_decision_seconds",
        "steady_state_decision_latency",
        "environment_final_info",
    }
    if set(summary) != expected_keys:
        missing = sorted(expected_keys - set(summary))
        extra = sorted(set(summary) - expected_keys)
        raise BenchmarkContractError(
            f"formal summary schema differs; missing={missing}, extra={extra}"
        )
    steps_count = summary["steps"]
    if isinstance(steps_count, bool) or not isinstance(steps_count, int) or steps_count <= 0:
        raise BenchmarkContractError("formal episode must contain at least one step")
    freezing_events = summary["freezing_events"]
    if (
        isinstance(freezing_events, bool)
        or not isinstance(freezing_events, int)
        or freezing_events < 0
    ):
        raise BenchmarkContractError("freezing_events must be a non-negative integer")
    boolean_fields = (
        "terminated",
        "truncated",
        "runner_limit_reached",
        "success",
        "collision",
        "collision_any",
        "collision_human",
        "collision_object",
        "collision_wall",
        "out_of_map",
        "timeout",
    )
    if any(not isinstance(summary[field], bool) for field in boolean_fields):
        raise BenchmarkContractError("formal summary event fields must be booleans")
    if summary["runner_limit_reached"] is not False:
        raise BenchmarkContractError("formal episode ended at a debug runner limit")
    if not (summary["terminated"] or summary["truncated"]):
        raise BenchmarkContractError("formal episode did not reach an environment end")
    for field in (
        "simulation_seconds",
        "return",
        "final_goal_distance",
        "geometric_path_length",
        "minimum_human_center_distance",
        "minimum_human_clearance",
        "cold_start_decision_seconds",
    ):
        if not _is_finite_number(summary[field]):
            raise BenchmarkContractError(f"formal summary field {field} must be finite")
    for field in (
        "simulation_seconds",
        "final_goal_distance",
        "geometric_path_length",
        "minimum_human_center_distance",
        "cold_start_decision_seconds",
    ):
        if summary[field] < 0:
            raise BenchmarkContractError(
                f"formal summary field {field} must be non-negative"
            )
    for field in (
        "environment_time_to_reach_goal",
        "environment_path_length",
        "environment_minimum_distance_to_human",
    ):
        value = summary[field]
        if value is not None and (not _is_finite_number(value) or value < 0):
            raise BenchmarkContractError(
                f"formal summary field {field} must be non-negative or null"
            )
    for field in (
        "decision_latency",
        "bookkeeping_latency",
        "total_compute_latency",
        "steady_state_decision_latency",
    ):
        latency = summary[field]
        if not isinstance(latency, Mapping) or set(latency) != {
            "mean",
            "p50",
            "p95",
            "max",
        }:
            raise BenchmarkContractError(f"formal summary latency {field} is invalid")
        values = tuple(latency.values())
        allow_empty = field == "steady_state_decision_latency" and steps_count == 1
        if allow_empty:
            if any(value is not None for value in values):
                raise BenchmarkContractError("one-step steady-state latency must be empty")
        elif any(
            not _is_finite_number(value) or value < 0 for value in values
        ):
            raise BenchmarkContractError(
                f"formal summary latency {field} must be finite and non-negative"
            )
    if not isinstance(summary["environment_final_info"], Mapping):
        raise BenchmarkContractError("environment_final_info must be an object")
    if not isinstance(episode.get("initial_state"), Mapping):
        raise BenchmarkContractError("formal episode is missing its initial state")
    if not isinstance(episode.get("reset_info"), Mapping):
        raise BenchmarkContractError("formal episode is missing reset info")
    initial_state = _state_from_document(
        episode["initial_state"], "episode initial_state"
    )
    if len(initial_state.humans) != expected_human_count:
        raise BenchmarkContractError(
            "formal episode human count differs from its scenario"
        )
    if expected_robot_start is not None:
        expected_start = np.asarray(expected_robot_start, dtype=np.float32)
        expected_goal = np.asarray(expected_robot_goal, dtype=np.float32)
        if not np.array_equal(initial_state.robot_position, expected_start):
            raise BenchmarkContractError(
                "formal episode robot start differs from its scenario"
            )
        if not np.array_equal(initial_state.goal, expected_goal):
            raise BenchmarkContractError(
                "formal episode robot goal differs from its scenario"
            )
    if not require_full_steps:
        return
    steps = episode.get("steps")
    if not isinstance(steps, list) or len(steps) != steps_count:
        raise BenchmarkContractError("episode step records and summary disagree")
    canonical = _canonical_summary_from_steps(
        episode,
        expected_human_count=expected_human_count,
    )
    differences = [
        key
        for key in expected_keys
        if not _json_values_equal(summary[key], canonical[key])
    ]
    if differences:
        raise BenchmarkContractError(
            "summary fields disagree with canonical step-derived values: "
            f"{sorted(differences)}"
        )


def _canonical_summary_from_steps(
    episode: Mapping[str, Any],
    *,
    expected_human_count: int,
) -> dict[str, Any]:
    """Reconstruct an episode and recompute every summary metric from steps."""
    import numpy as np

    from cfm_mppi.evaluation.socnavgym_runner import (
        EpisodeContext,
        EpisodeResult,
        PlanningBudget,
        StepRecord,
        _states_match,
    )

    context_document = episode.get("context")
    expected_context_keys = {
        "env_seed",
        "planner_seed",
        "time_step",
        "episode_length",
        "max_human_speed",
        "control_low",
        "control_high",
    }
    if not isinstance(context_document, Mapping) or set(
        context_document
    ) != expected_context_keys:
        raise BenchmarkContractError("formal episode context schema is invalid")
    for field in ("env_seed", "planner_seed", "episode_length"):
        if not _is_int(context_document[field]):
            raise BenchmarkContractError(f"episode context {field} must be an integer")
    for field in ("time_step", "max_human_speed"):
        if not _is_finite_number(context_document[field]):
            raise BenchmarkContractError(f"episode context {field} must be finite")
    control_low = _finite_vector(context_document["control_low"], 2, "control_low")
    control_high = _finite_vector(
        context_document["control_high"], 2, "control_high"
    )
    try:
        context = EpisodeContext(
            env_seed=context_document["env_seed"],
            planner_seed=context_document["planner_seed"],
            time_step=context_document["time_step"],
            episode_length=context_document["episode_length"],
            max_human_speed=context_document["max_human_speed"],
            control_low=control_low,
            control_high=control_high,
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError("formal episode context is invalid") from exc
    if (
        context.time_step != 0.1
        or context.episode_length != 256
        or context.max_human_speed != 0.8
        or not np.array_equal(context.control_low, np.asarray([-1.0, -1.0]))
        or not np.array_equal(context.control_high, np.asarray([1.0, 1.0]))
    ):
        raise BenchmarkContractError("formal episode context differs from the suite")

    initial_state = _state_from_document(
        episode["initial_state"], "episode initial_state"
    )
    if len(initial_state.humans) != expected_human_count:
        raise BenchmarkContractError(
            "formal episode human count differs from its scenario"
        )
    expected_human_ids = initial_state.human_ids
    reset_info = episode.get("reset_info")
    if not isinstance(reset_info, Mapping):
        raise BenchmarkContractError("formal episode reset_info must be an object")

    raw_steps = episode["steps"]
    records = []
    previous_state = initial_state
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise BenchmarkContractError("formal step records must be objects")
        expected_step_keys = {
            "step_index",
            "state",
            "next_state",
            "requested_control",
            "applied_control",
            "normalized_action",
            "reward",
            "terminated",
            "truncated",
            "info",
            "diagnostics",
            "decision_seconds",
            "bookkeeping_seconds",
        }
        if set(raw_step) != expected_step_keys:
            missing = sorted(expected_step_keys - set(raw_step))
            extra = sorted(set(raw_step) - expected_step_keys)
            raise BenchmarkContractError(
                f"formal step schema differs; missing={missing}, extra={extra}"
            )
        if not _is_int(raw_step["step_index"]) or raw_step["step_index"] != index:
            raise BenchmarkContractError("formal step indices are not contiguous")
        for field in ("terminated", "truncated"):
            if not isinstance(raw_step[field], bool):
                raise BenchmarkContractError(f"formal step {field} must be boolean")
        if index + 1 < len(raw_steps) and (
            raw_step["terminated"] or raw_step["truncated"]
        ):
            raise BenchmarkContractError(
                "formal episode contains steps after environment termination"
            )
        if index + 1 == len(raw_steps) and not (
            raw_step["terminated"] or raw_step["truncated"]
        ):
            raise BenchmarkContractError("formal final step is not terminal")
        for field in ("reward", "decision_seconds", "bookkeeping_seconds"):
            if not _is_finite_number(raw_step[field]):
                raise BenchmarkContractError(f"formal step {field} must be finite")
        for field in ("decision_seconds", "bookkeeping_seconds"):
            if raw_step[field] < 0:
                raise BenchmarkContractError(
                    f"formal step {field} must be non-negative"
                )
        info = raw_step["info"]
        diagnostics = raw_step["diagnostics"]
        if not isinstance(info, Mapping) or not isinstance(diagnostics, Mapping):
            raise BenchmarkContractError(
                "formal step info and diagnostics must be objects"
            )
        for event_key in (
            "SUCCESS",
            "COLLISION",
            "COLLISION_HUMAN",
            "COLLISION_OBJECT",
            "COLLISION_WALL",
            "OUT_OF_MAP",
            "TIMEOUT",
        ):
            if not isinstance(info.get(event_key), bool):
                raise BenchmarkContractError(
                    f"formal step info {event_key} must be boolean"
                )

        state = _state_from_document(raw_step["state"], f"step {index} state")
        next_state = _state_from_document(
            raw_step["next_state"], f"step {index} next_state"
        )
        if not _states_match(previous_state, state):
            raise BenchmarkContractError("formal step state chain is discontinuous")
        if state.human_ids != expected_human_ids or next_state.human_ids != (
            expected_human_ids
        ):
            raise BenchmarkContractError("formal episode human identity/order changed")

        requested = _finite_vector(
            raw_step["requested_control"], 2, "requested_control"
        )
        applied = _finite_vector(raw_step["applied_control"], 2, "applied_control")
        normalized = _finite_vector(
            raw_step["normalized_action"], 3, "normalized_action"
        )
        if not np.allclose(requested, applied, rtol=0.0, atol=1e-6):
            raise BenchmarkContractError("formal environment clipped planner control")
        if np.any(applied < context.control_low - 1e-6) or np.any(
            applied > context.control_high + 1e-6
        ):
            raise BenchmarkContractError("formal applied control is out of bounds")
        expected_action = np.asarray(
            [applied[0], 0.0, applied[1]], dtype=np.float32
        )
        if not np.allclose(normalized, expected_action, rtol=0.0, atol=1e-6):
            raise BenchmarkContractError(
                "formal normalized action disagrees with applied control"
            )
        try:
            records.append(
                StepRecord(
                    step_index=index,
                    state=state,
                    next_state=next_state,
                    requested_control=requested,
                    applied_control=applied,
                    normalized_action=normalized,
                    reward=raw_step["reward"],
                    terminated=raw_step["terminated"],
                    truncated=raw_step["truncated"],
                    info=info,
                    diagnostics=diagnostics,
                    decision_seconds=raw_step["decision_seconds"],
                    bookkeeping_seconds=raw_step["bookkeeping_seconds"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError("formal step record is invalid") from exc
        previous_state = next_state

    budget = episode["budget"]
    try:
        result = EpisodeResult(
            planner_name=episode["planner"],
            budget=PlanningBudget(
                cfm_candidates=budget["cfm_candidates"],
                refinement_rollouts=budget["refinement_rollouts"],
            ),
            context=context,
            initial_state=initial_state,
            reset_info=reset_info,
            steps=tuple(records),
            runner_limit_reached=False,
        )
        return result.summary()
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkContractError(
            "could not derive canonical summary from formal steps"
        ) from exc


def _state_from_document(raw: Any, label: str):
    import numpy as np

    from cfm_mppi.evaluation.socnavgym_adapter import HumanState, SocNavState

    expected_keys = {
        "robot_state",
        "goal",
        "robot_body_velocity",
        "robot_velocity",
        "robot_radius",
        "goal_radius",
        "humans",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise BenchmarkContractError(f"{label} schema is invalid")
    robot_state = _finite_vector(raw["robot_state"], 3, f"{label}.robot_state")
    goal = _finite_vector(raw["goal"], 2, f"{label}.goal")
    body_velocity = _finite_vector(
        raw["robot_body_velocity"], 3, f"{label}.robot_body_velocity"
    )
    recorded_velocity = _finite_vector(
        raw["robot_velocity"], 3, f"{label}.robot_velocity"
    )
    for field in ("robot_radius", "goal_radius"):
        if not _is_finite_number(raw[field]) or raw[field] < 0:
            raise BenchmarkContractError(f"{label}.{field} must be non-negative")
    raw_humans = raw["humans"]
    if not isinstance(raw_humans, list):
        raise BenchmarkContractError(f"{label}.humans must be a list")
    humans = []
    human_ids = []
    for index, raw_human in enumerate(raw_humans):
        human_label = f"{label}.humans[{index}]"
        human_keys = {
            "id",
            "position",
            "velocity",
            "radius",
            "orientation",
            "gaze",
        }
        if not isinstance(raw_human, Mapping) or set(raw_human) != human_keys:
            raise BenchmarkContractError(f"{human_label} schema is invalid")
        if not _is_int(raw_human["id"]):
            raise BenchmarkContractError(f"{human_label}.id must be an integer")
        if not _is_finite_number(raw_human["radius"]) or raw_human["radius"] < 0:
            raise BenchmarkContractError(f"{human_label}.radius must be non-negative")
        if not _is_finite_number(raw_human["orientation"]):
            raise BenchmarkContractError(f"{human_label}.orientation must be finite")
        if not isinstance(raw_human["gaze"], bool):
            raise BenchmarkContractError(f"{human_label}.gaze must be boolean")
        try:
            human = HumanState(
                id=raw_human["id"],
                position=_finite_vector(
                    raw_human["position"], 2, f"{human_label}.position"
                ),
                velocity=_finite_vector(
                    raw_human["velocity"], 2, f"{human_label}.velocity"
                ),
                radius=raw_human["radius"],
                orientation=raw_human["orientation"],
                gaze=raw_human["gaze"],
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError(f"{human_label} is invalid") from exc
        humans.append(human)
        human_ids.append(human.id)
    if human_ids != sorted(human_ids) or len(set(human_ids)) != len(human_ids):
        raise BenchmarkContractError(f"{label} human IDs must be unique and sorted")
    try:
        state = SocNavState(
            robot_state=robot_state,
            goal=goal,
            robot_body_velocity=body_velocity,
            robot_radius=raw["robot_radius"],
            goal_radius=raw["goal_radius"],
            humans=tuple(humans),
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkContractError(f"{label} is invalid") from exc
    if not np.array_equal(recorded_velocity, state.robot_velocity):
        raise BenchmarkContractError(
            f"{label}.robot_velocity disagrees with body velocity and heading"
        )
    return state


def _finite_vector(value: Any, length: int, label: str):
    import numpy as np

    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not _is_finite_number(item) for item in value)
    ):
        raise BenchmarkContractError(
            f"{label} must be a finite numeric list of length {length}"
        )
    return np.asarray(value, dtype=np.float32)


def _json_values_equal(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first is second
    if isinstance(first, Real) or isinstance(second, Real):
        return (
            isinstance(first, Real)
            and isinstance(second, Real)
            and first == second
        )
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        return (
            isinstance(first, Mapping)
            and isinstance(second, Mapping)
            and set(first) == set(second)
            and all(_json_values_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, list) or isinstance(second, list):
        return (
            isinstance(first, list)
            and isinstance(second, list)
            and len(first) == len(second)
            and all(
                _json_values_equal(first_item, second_item)
                for first_item, second_item in zip(first, second)
            )
        )
    return type(first) is type(second) and first == second


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def read_and_validate_shard(
    path: Path,
    suite: BenchmarkSuite,
    job: BenchmarkJob,
    *,
    require_full_steps: bool = True,
) -> dict[str, Any]:
    document = _read_json_mapping(Path(path), f"benchmark shard {job.index}")
    validate_shard_document(
        document, suite, job, require_full_steps=require_full_steps
    )
    return document


def _load_scenario(
    base: Path, raw: Any, index: int
) -> BenchmarkScenario:
    label = f"scenario[{index}]"
    if not isinstance(raw, Mapping):
        raise BenchmarkContractError(f"{label} must be an object")
    _require_exact_keys(
        raw,
        {
            "id",
            "role",
            "config",
            "config_sha256",
            "human_policy",
            "dynamic_humans",
            "static_humans",
            "map_size_metres",
        }
        | (
            {"robot_start", "robot_goal"}
            if "robot_start" in raw or "robot_goal" in raw
            else set()
        ),
        label,
    )
    scenario_id = _nonempty_string(raw["id"], f"{label}.id")
    if not _SAFE_ID.fullmatch(scenario_id):
        raise BenchmarkContractError(f"{label}.id is not a safe lowercase identifier")
    role = _nonempty_string(raw["role"], f"{label}.role")
    if role not in ("primary", "sensitivity"):
        raise BenchmarkContractError(f"{label}.role must be primary or sensitivity")
    human_policy = _nonempty_string(raw["human_policy"], f"{label}.human_policy")
    if human_policy not in ("orca", "sfm"):
        raise BenchmarkContractError(f"{label}.human_policy must be orca or sfm")
    if (human_policy == "orca") != (role == "primary"):
        raise BenchmarkContractError(
            f"{label}: ORCA scenarios are primary and SFM scenarios are sensitivity"
        )
    dynamic_humans = _positive_int(raw["dynamic_humans"], f"{label}.dynamic_humans")
    static_humans = _nonnegative_int(raw["static_humans"], f"{label}.static_humans")
    raw_map = raw["map_size_metres"]
    if not isinstance(raw_map, list) or len(raw_map) != 2:
        raise BenchmarkContractError(f"{label}.map_size_metres must contain x and y")
    map_size = tuple(
        _positive_int(value, f"{label}.map_size_metres") for value in raw_map
    )
    robot_start: tuple[float, float] | None = None
    robot_goal: tuple[float, float] | None = None
    if "robot_start" in raw or "robot_goal" in raw:
        if "robot_start" not in raw or "robot_goal" not in raw:
            raise BenchmarkContractError(
                f"{label} must define robot_start and robot_goal together"
            )
        robot_start = _finite_coordinate_pair(
            raw["robot_start"], f"{label}.robot_start"
        )
        robot_goal = _finite_coordinate_pair(raw["robot_goal"], f"{label}.robot_goal")
        limits = (map_size[0] / 2.0 - 0.5, map_size[1] / 2.0 - 0.5)
        for field, point in (("robot_start", robot_start), ("robot_goal", robot_goal)):
            if abs(point[0]) > limits[0] or abs(point[1]) > limits[1]:
                raise BenchmarkContractError(
                    f"{label}.{field} must lie inside the map's 0.5 m spawn margin"
                )
        if robot_start == robot_goal:
            raise BenchmarkContractError(
                f"{label}.robot_start and robot_goal must differ"
            )
    config_path = _resolve_relative_file(base, raw["config"], f"{label}.config")
    declared_sha = _declared_sha256(raw["config_sha256"], f"{label}.config_sha256")
    actual_sha = sha256_file(config_path)
    if actual_sha != declared_sha:
        raise BenchmarkContractError(
            f"{label} config hash mismatch: expected {declared_sha}, got {actual_sha}"
        )
    config = _read_yaml_mapping(config_path, label)
    _validate_socnavgym_config(
        config,
        label=label,
        human_policy=human_policy,
        dynamic_humans=dynamic_humans,
        static_humans=static_humans,
        map_size=map_size,
    )
    return BenchmarkScenario(
        id=scenario_id,
        role=role,
        config_path=config_path,
        config_sha256=actual_sha,
        human_policy=human_policy,
        dynamic_humans=dynamic_humans,
        static_humans=static_humans,
        map_size_metres=(map_size[0], map_size[1]),
        robot_start=robot_start,
        robot_goal=robot_goal,
    )


def _validate_socnavgym_config(
    config: Mapping[str, Any],
    *,
    label: str,
    human_policy: str,
    dynamic_humans: int,
    static_humans: int,
    map_size: Sequence[int],
) -> None:
    expected = {
        ("episode", "episode_length"): 256,
        ("episode", "time_step"): 0.1,
        ("episode", "end_with_collision"): True,
        ("robot", "robot_radius"): 0.25,
        ("robot", "goal_radius"): 0.35,
        ("robot", "robot_type"): "diff-drive",
        ("robot", "min_goal_orientation_threshold"): 2 * math.pi,
        ("robot", "max_goal_orientation_threshold"): 2 * math.pi,
        ("human", "human_diameter"): 0.72,
        ("human", "human_policy"): human_policy,
        ("human", "prob_to_avoid_robot"): 1.0,
        ("env", "max_advance_human"): 0.8,
        ("env", "max_advance_robot"): 1.0,
        ("env", "max_rotation"): 1.0,
        ("env", "margin"): 0.5,
        ("env", "min_static_humans"): static_humans,
        ("env", "max_static_humans"): static_humans,
        ("env", "min_dynamic_humans"): dynamic_humans,
        ("env", "max_dynamic_humans"): dynamic_humans,
        ("env", "get_padded_observations"): False,
        ("env", "set_shape"): "no-walls",
        ("env", "add_corridors"): False,
        ("env", "min_map_x"): map_size[0],
        ("env", "max_map_x"): map_size[0],
        ("env", "min_map_y"): map_size[1],
        ("env", "max_map_y"): map_size[1],
        ("env", "crowd_dispersal_probability"): 0.0,
        ("env", "human_laptop_dispersal_probability"): 0.0,
        ("env", "crowd_formation_probability"): 0.0,
        ("env", "human_laptop_formation_probability"): 0.0,
    }
    for path, expected_value in expected.items():
        value = _nested(config, path, label)
        if not _same_value(value, expected_value):
            dotted = ".".join(path)
            raise BenchmarkContractError(
                f"{label} requires {dotted}={expected_value!r}, got {value!r}"
            )

    zero_env_fields = (
        "min_tables",
        "max_tables",
        "min_plants",
        "max_plants",
        "min_laptops",
        "max_laptops",
        "min_h_h_dynamic_interactions",
        "max_h_h_dynamic_interactions",
        "min_h_h_dynamic_interactions_non_dispersing",
        "max_h_h_dynamic_interactions_non_dispersing",
        "min_h_h_static_interactions",
        "max_h_h_static_interactions",
        "min_h_h_static_interactions_non_dispersing",
        "max_h_h_static_interactions_non_dispersing",
        "min_human_in_h_h_interactions",
        "max_human_in_h_h_interactions",
        "min_h_l_interactions",
        "max_h_l_interactions",
        "min_h_l_interactions_non_dispersing",
        "max_h_l_interactions_non_dispersing",
    )
    for field in zero_env_fields:
        value = _nested(config, ("env", field), label)
        if not _same_value(value, 0):
            raise BenchmarkContractError(
                f"{label} requires env.{field}=0 to keep the benchmark human-only"
            )


def _planner_contract(raw: Any) -> dict[str, int | str]:
    if not isinstance(raw, Mapping):
        raise BenchmarkContractError("planner must be an object")
    keys = {
        "selection",
        "horizon",
        "max_history",
        "cfm_candidates",
        "branches",
        "mppi_samples_per_branch",
    }
    _require_exact_keys(raw, keys, "planner")
    if raw["selection"] != "both":
        raise BenchmarkContractError("formal benchmark must select both planners")
    result: dict[str, int | str] = {"selection": "both"}
    for key in keys - {"selection"}:
        result[key] = _positive_int(raw[key], f"planner.{key}")
    return result


def _finite_coordinate_pair(value: Any, label: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not _is_finite_number(component) for component in value)
    ):
        raise BenchmarkContractError(f"{label} must be a finite [x, y] list")
    return (float(value[0]), float(value[1]))


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkContractError(f"invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{label} must contain a JSON object")
    return value


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BenchmarkContractError(f"invalid YAML in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{label} config must contain a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BenchmarkContractError(
            f"{label} fields differ from schema; missing={missing}, extra={extra}"
        )


def _resolve_relative_file(base: Path, raw: Any, label: str) -> Path:
    relative = _safe_relative_path(raw, label)
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise BenchmarkContractError(f"{label} escapes the manifest directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _safe_relative_path(raw: Any, label: str) -> Path:
    text = _nonempty_string(raw, label)
    path = Path(text)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BenchmarkContractError(f"{label} must be a safe relative path")
    return path


def _seed_tuple(raw: Any, label: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise BenchmarkContractError(f"{label} must be a non-empty explicit list")
    seeds = tuple(_nonnegative_int(value, label) for value in raw)
    if len(set(seeds)) != len(seeds):
        raise BenchmarkContractError(f"{label} contains duplicate seeds")
    return seeds


def _nonempty_string(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise BenchmarkContractError(f"{label} must be a non-empty string")
    return raw


def _nonnegative_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise BenchmarkContractError(f"{label} must be a non-negative integer")
    return raw


def _positive_int(raw: Any, label: str) -> int:
    value = _nonnegative_int(raw, label)
    if value == 0:
        raise BenchmarkContractError(f"{label} must be positive")
    return value


def _declared_sha256(raw: Any, label: str) -> str:
    value = _nonempty_string(raw, label)
    if not _SHA256.fullmatch(value):
        raise BenchmarkContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nested(mapping: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    value: Any = mapping
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise BenchmarkContractError(
                f"{label} is missing config field {'.'.join(path)}"
            )
        value = value[component]
    return value


def _same_value(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return value is expected
    if isinstance(expected, (int, float)) and not isinstance(value, bool):
        try:
            return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError, OverflowError):
            return False
    return value == expected


__all__ = [
    "BenchmarkContractError",
    "BenchmarkJob",
    "BenchmarkScenario",
    "BenchmarkSuite",
    "CANONICAL_PLANNERS",
    "DEFAULT_BENCHMARK_SUITE",
    "benchmark_metadata",
    "load_benchmark_suite",
    "read_and_validate_shard",
    "sha256_file",
    "validate_shard_document",
]
