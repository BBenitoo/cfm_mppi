"""Execute and aggregate the locked SocNavGym benchmark suite."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from cfm_mppi.evaluation.socnavgym_benchmark import (
    BenchmarkContractError,
    BenchmarkJob,
    BenchmarkSuite,
    CANONICAL_PLANNERS,
    DEFAULT_BENCHMARK_SUITE,
    benchmark_metadata,
    load_benchmark_suite,
    read_and_validate_shard,
    sha256_file,
    validate_shard_document,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_CANDIDATES = (
    REPOSITORY_ROOT / "output_dir" / "cfm_transformer" / "checkpoint.pth",
    REPOSITORY_ROOT
    / "cfm_mppi"
    / "output_dir"
    / "cfm_transformer"
    / "checkpoint.pth",
)
DEFAULT_CHECKPOINT = next(
    (path for path in _CHECKPOINT_CANDIDATES if path.is_file()),
    _CHECKPOINT_CANDIDATES[0],
)
_RUNTIME_METADATA_FIELDS = (
    "checkpoint_sha256",
    "implementation_sha256",
    "repository_commit",
    "repository_dirty",
    "python",
    "platform",
    "torch",
    "torch_cuda_runtime",
    "cudnn",
    "gymnasium",
    "socnavgym",
    "socnavgym_commit",
    "socnavgym_human_goal_policy",
    "numpy",
    "dgl",
    "pyrvo2",
    "rvo2_module",
    "device",
    "cuda_device",
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one immutable shard of the locked SocNavGym benchmark, or "
            "validate/list/aggregate the complete suite."
        )
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_BENCHMARK_SUITE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--job-index",
        type=int,
        help="locked job index; defaults to SLURM_ARRAY_TASK_ID",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-jobs", action="store_true")
    modes.add_argument("--validate-configs", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected immutable job without evaluating planners",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write an explicitly incomplete aggregate while jobs are still running",
    )
    return parser


def _output_root(args: argparse.Namespace, suite: BenchmarkSuite) -> Path:
    if args.output_root is None:
        root = suite.default_output_root().resolve()
    else:
        root = Path(args.output_root).expanduser().resolve()
    allowed_root = (REPOSITORY_ROOT / "output_dir").resolve()
    try:
        root.relative_to(allowed_root)
    except ValueError as exc:
        raise BenchmarkContractError(
            "benchmark output root must be inside the repository's ignored "
            f"output_dir: {allowed_root}"
        ) from exc
    return root


def _job_index(args: argparse.Namespace) -> int:
    if args.job_index is not None:
        return args.job_index
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise BenchmarkContractError(
            "select --job-index, or run inside a Slurm array with "
            "SLURM_ARRAY_TASK_ID set"
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise BenchmarkContractError("SLURM_ARRAY_TASK_ID must be an integer") from exc


def _job_description(
    suite: BenchmarkSuite,
    job: BenchmarkJob,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "suite_id": suite.suite_id,
        "job_index": job.index,
        "scenario_id": job.scenario.id,
        "scenario_role": job.scenario.role,
        "human_policy": job.scenario.human_policy,
        "dynamic_humans": job.scenario.dynamic_humans,
        "config": str(job.scenario.config_path),
        "config_sha256": job.scenario.config_sha256,
        "environment_seeds": list(job.env_seeds),
        "planner_seeds": [
            seed + suite.planner_seed_offset for seed in job.env_seeds
        ],
        "planner": dict(suite.planner),
        "device": suite.device,
        "cuda_device_contains": suite.cuda_device_contains,
        "execution_order": (
            list(CANONICAL_PLANNERS)
            if job.index % 2 == 0
            else list(reversed(CANONICAL_PLANNERS))
        ),
        "output": str(suite.output_path(output_root, job)),
    }


def list_jobs(suite: BenchmarkSuite, output_root: Path) -> None:
    print("job\tscenario\trole\tpolicy\thumans\tseeds\toutput")
    for job in suite.jobs:
        seeds = ",".join(str(seed) for seed in job.env_seeds)
        print(
            f"{job.index}\t{job.scenario.id}\t{job.scenario.role}\t"
            f"{job.scenario.human_policy}\t{job.scenario.dynamic_humans}\t"
            f"{seeds}\t{suite.output_path(output_root, job)}"
        )
    print(
        f"# {len(suite.scenarios)} scenarios x {len(suite.benchmark_seeds)} seeds "
        f"= {suite.expected_scenario_seed_pairs} scenario-seed pairs; "
        f"{len(suite.jobs)} paired single-seed jobs"
    )


def validate_environment_matrix(suite: BenchmarkSuite) -> dict[str, Any]:
    """Reset every locked scenario/seed pair without running either planner."""
    from cfm_mppi.evaluation.socnavgym_adapter import SocNavGymAdapter

    validated = 0
    scenario_counts: dict[str, int] = {}
    for scenario in suite.scenarios:
        print(
            f"validating {scenario.id}: {len(suite.benchmark_seeds)} seeds",
            flush=True,
        )
        with _deadline(30.0, f"constructing {scenario.id}"):
            environment = SocNavGymAdapter(scenario.config_path)
        try:
            if environment.time_step != 0.1 or environment.episode_length != 256:
                raise BenchmarkContractError(
                    f"{scenario.id}: runtime timestep/episode length drifted"
                )
            for env_seed in suite.benchmark_seeds:
                with _deadline(
                    30.0, f"resetting {scenario.id} with seed {env_seed}"
                ):
                    state, _ = environment.reset(seed=env_seed)
                expected_humans = scenario.dynamic_humans + scenario.static_humans
                if len(state.humans) != expected_humans:
                    raise BenchmarkContractError(
                        f"{scenario.id} seed {env_seed}: expected {expected_humans} "
                        f"humans, got {len(state.humans)}"
                    )
                if tuple(sorted(state.human_ids)) != state.human_ids:
                    raise BenchmarkContractError(
                        f"{scenario.id} seed {env_seed}: human IDs are not stable/sorted"
                    )
                validated += 1
        finally:
            environment.close()
        scenario_counts[scenario.id] = len(suite.benchmark_seeds)
    return {
        "suite_id": suite.suite_id,
        "validated_scenario_seed_pairs": validated,
        "expected_scenario_seed_pairs": suite.expected_scenario_seed_pairs,
        "scenarios": scenario_counts,
    }


@contextmanager
def _deadline(seconds: float, label: str):
    """Bound known SocNavGym placement retry loops on POSIX login nodes."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def timed_out(signum, frame):
        del signum, frame
        raise TimeoutError(f"SocNavGym timed out after {seconds:g}s while {label}")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _assert_checkpoint(suite: BenchmarkSuite, checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"formal checkpoint not found: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != suite.checkpoint_sha256:
        raise BenchmarkContractError(
            f"checkpoint hash differs from the locked suite: expected "
            f"{suite.checkpoint_sha256}, got {actual}"
        )
    return checkpoint


def _assert_clean_repository() -> dict[str, Any]:
    from cfm_mppi.evaluation.eval_socnavgym import _git_metadata

    git = _git_metadata()
    if git.get("repository_commit") is None:
        raise BenchmarkContractError("formal benchmark requires a Git checkout")
    if git.get("repository_dirty") is not False:
        raise BenchmarkContractError(
            "formal benchmark refuses a dirty repository; commit the benchmark "
            "implementation before submitting H100 jobs"
        )
    return git


def _assert_socnavgym_revision(suite: BenchmarkSuite) -> None:
    from cfm_mppi.evaluation.eval_socnavgym import _distribution_commit

    actual = _distribution_commit("socnavgym")
    if actual != suite.socnavgym_commit:
        raise BenchmarkContractError(
            "installed SocNavGym commit differs from the locked suite: "
            f"expected {suite.socnavgym_commit}, got {actual}"
        )


def _assert_h100(suite: BenchmarkSuite) -> None:
    import torch

    if not torch.cuda.is_available():
        raise BenchmarkContractError("formal benchmark requires an allocated CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if suite.cuda_device_contains not in device_name:
        raise BenchmarkContractError(
            f"formal benchmark requires an NVIDIA H100, got {device_name!r}"
        )


def _current_run_contract(
    suite: BenchmarkSuite,
    checkpoint: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from cfm_mppi.evaluation.eval_socnavgym import (
        _distribution_commit,
        _implementation_sha256,
        _module_binary_metadata,
        _package_version,
    )
    from cfm_mppi.evaluation.socnavgym_adapter import HUMAN_GOAL_REACHED_POLICY

    if os.environ.get("DGLBACKEND") != suite.dgl_backend:
        raise BenchmarkContractError(
            f"formal benchmark requires DGLBACKEND={suite.dgl_backend}"
        )
    runtime_metadata = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "implementation_sha256": _implementation_sha256(),
        "repository_commit": git["repository_commit"],
        "repository_dirty": git["repository_dirty"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gymnasium": _package_version("gymnasium"),
        "socnavgym": _package_version("socnavgym"),
        "socnavgym_commit": _distribution_commit("socnavgym"),
        "socnavgym_human_goal_policy": HUMAN_GOAL_REACHED_POLICY,
        "numpy": _package_version("numpy"),
        "dgl": _package_version("dgl"),
        "pyrvo2": _package_version("pyrvo2"),
        "rvo2_module": _module_binary_metadata("rvo2"),
        "device": suite.device,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    return {
        "schema_version": "cfm_mppi.socnavgym_run_contract.v1",
        "suite_id": suite.suite_id,
        "manifest_sha256": suite.manifest_sha256,
        "seed_file_sha256": suite.seed_sha256,
        "scenario_config_sha256": {
            scenario.id: scenario.config_sha256 for scenario in suite.scenarios
        },
        "checkpoint_sha256": suite.checkpoint_sha256,
        "rvo2_module_sha256": suite.rvo2_module_sha256,
        "dgl_backend": suite.dgl_backend,
        "planner_seed_offset": suite.planner_seed_offset,
        "planner": dict(suite.planner),
        "device": suite.device,
        "cuda_device_contains": suite.cuda_device_contains,
        "runtime_metadata": runtime_metadata,
    }


def _validate_run_contract_static(
    contract: Mapping[str, Any], suite: BenchmarkSuite
) -> None:
    expected = {
        "schema_version": "cfm_mppi.socnavgym_run_contract.v1",
        "suite_id": suite.suite_id,
        "manifest_sha256": suite.manifest_sha256,
        "seed_file_sha256": suite.seed_sha256,
        "scenario_config_sha256": {
            scenario.id: scenario.config_sha256 for scenario in suite.scenarios
        },
        "checkpoint_sha256": suite.checkpoint_sha256,
        "rvo2_module_sha256": suite.rvo2_module_sha256,
        "dgl_backend": suite.dgl_backend,
        "planner_seed_offset": suite.planner_seed_offset,
        "planner": dict(suite.planner),
        "device": suite.device,
        "cuda_device_contains": suite.cuda_device_contains,
    }
    if set(contract) != set(expected) | {"runtime_metadata"}:
        raise BenchmarkContractError("run contract fields differ from its schema")
    for key, value in expected.items():
        if contract.get(key) != value:
            raise BenchmarkContractError(f"run contract field {key} differs from suite")
    runtime = contract.get("runtime_metadata")
    if not isinstance(runtime, Mapping) or set(runtime) != set(
        _RUNTIME_METADATA_FIELDS
    ):
        raise BenchmarkContractError("run contract runtime metadata is invalid")
    if runtime.get("checkpoint_sha256") != suite.checkpoint_sha256:
        raise BenchmarkContractError("run contract checkpoint is invalid")
    if runtime.get("repository_dirty") is not False:
        raise BenchmarkContractError("run contract repository must be clean")
    if runtime.get("socnavgym_commit") != suite.socnavgym_commit:
        raise BenchmarkContractError("run contract SocNavGym commit is invalid")
    if runtime.get("socnavgym_human_goal_policy") != suite.human_goal_reached_policy:
        raise BenchmarkContractError("run contract human-goal policy is invalid")
    rvo2_module = runtime.get("rvo2_module")
    if (
        not isinstance(rvo2_module, Mapping)
        or rvo2_module.get("sha256") != suite.rvo2_module_sha256
    ):
        raise BenchmarkContractError("run contract RVO2 binary is invalid")
    if runtime.get("device") != suite.device or suite.cuda_device_contains not in str(
        runtime.get("cuda_device")
    ):
        raise BenchmarkContractError("run contract H100 device is invalid")


def _read_run_contract(path: Path, suite: BenchmarkSuite) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"invalid run contract: {path}") from exc
    if not isinstance(document, dict):
        raise BenchmarkContractError("run contract must be a JSON object")
    _validate_run_contract_static(document, suite)
    return document


def _ensure_run_contract(
    output_root: Path,
    suite: BenchmarkSuite,
    checkpoint: Path,
    git: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _current_run_contract(suite, checkpoint, git)
    # Validate before publishing so a broken first worker cannot permanently
    # poison an otherwise empty immutable output root.
    _validate_run_contract_static(expected, suite)
    path = output_root / "run_contract.json"
    if not path.exists():
        try:
            _write_new_json_atomic(path, expected)
        except BenchmarkContractError:
            if not path.exists():
                raise
    actual = _read_run_contract(path, suite)
    if actual != expected:
        raise BenchmarkContractError(
            "output root belongs to a different code, dependency, Git, model, "
            "or H100 runtime contract"
        )
    return actual


def _assert_document_matches_run_contract(
    document: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    metadata = document.get("metadata")
    expected = contract.get("runtime_metadata")
    if not isinstance(metadata, Mapping) or not isinstance(expected, Mapping):
        raise BenchmarkContractError("result or run contract lacks runtime metadata")
    actual = {field: metadata.get(field) for field in _RUNTIME_METADATA_FIELDS}
    if actual != dict(expected):
        differences = [
            field
            for field in _RUNTIME_METADATA_FIELDS
            if actual.get(field) != expected.get(field)
        ]
        raise BenchmarkContractError(
            f"runtime changed while the job was executing: {differences}"
        )


def _existing_shard_matches_current_code(
    document: Mapping[str, Any],
    git: Mapping[str, Any],
) -> None:
    from cfm_mppi.evaluation.eval_socnavgym import _implementation_sha256

    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise BenchmarkContractError("existing shard has no metadata")
    if metadata.get("implementation_sha256") != _implementation_sha256():
        raise BenchmarkContractError(
            "existing shard was produced by different evaluation code; use a new "
            "output root for a new benchmark revision"
        )
    if metadata.get("repository_commit") != git.get("repository_commit"):
        raise BenchmarkContractError(
            "existing shard was produced by a different Git commit; use a new "
            "output root for a new benchmark revision"
        )


def run_job(
    args: argparse.Namespace,
    suite: BenchmarkSuite,
    job: BenchmarkJob,
    output_root: Path,
) -> str:
    from cfm_mppi.evaluation.eval_socnavgym import run_evaluation

    checkpoint = _assert_checkpoint(suite, args.checkpoint)
    output_path = suite.output_path(output_root, job)
    if args.dry_run:
        print(json.dumps(_job_description(suite, job, output_root), indent=2))
        return "dry-run"

    git = _assert_clean_repository()
    _assert_socnavgym_revision(suite)
    _assert_h100(suite)
    run_contract = _ensure_run_contract(output_root, suite, checkpoint, git)
    if output_path.exists():
        existing = read_and_validate_shard(output_path, suite, job)
        _assert_document_matches_run_contract(existing, run_contract)
        _existing_shard_matches_current_code(existing, git)
        print(f"validated existing shard; skipping job {job.index}: {output_path}")
        return "skipped"

    planner = suite.planner
    evaluation_args = argparse.Namespace(
        config=job.scenario.config_path,
        checkpoint=checkpoint,
        planner="both",
        seeds=job.env_seeds,
        planner_seed_offset=suite.planner_seed_offset,
        execution_order_offset=job.index % 2,
        max_steps=None,
        device=suite.device,
        output=output_path,
        summary_only=False,
        allow_random_model=False,
        horizon=planner["horizon"],
        max_history=planner["max_history"],
        cfm_candidates=planner["cfm_candidates"],
        branches=planner["branches"],
        mppi_samples_per_branch=planner["mppi_samples_per_branch"],
    )
    document = run_evaluation(evaluation_args)
    document["benchmark"] = benchmark_metadata(suite, job)
    validate_shard_document(document, suite, job)
    _assert_document_matches_run_contract(document, run_contract)
    _write_new_json_atomic(output_path, document)
    print(json.dumps(document["summaries"], indent=2, sort_keys=True))
    print(f"wrote immutable benchmark shard {output_path}")
    return "written"


def aggregate_suite(
    suite: BenchmarkSuite,
    output_root: Path,
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    run_contract_path = output_root / "run_contract.json"
    run_contract = (
        _read_run_contract(run_contract_path, suite)
        if run_contract_path.is_file()
        else None
    )
    shards: list[tuple[BenchmarkJob, Path, dict[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    for job in suite.jobs:
        path = suite.output_path(output_root, job)
        if not path.is_file():
            missing.append(
                {
                    "job_index": job.index,
                    "scenario_id": job.scenario.id,
                    "environment_seeds": list(job.env_seeds),
                    "path": str(path),
                }
            )
            continue
        document = read_and_validate_shard(path, suite, job)
        if run_contract is None:
            raise BenchmarkContractError(
                "benchmark shards exist without the required run_contract.json"
            )
        _assert_document_matches_run_contract(document, run_contract)
        shards.append((job, path, document))
    if missing and not allow_incomplete:
        raise BenchmarkContractError(
            f"benchmark is incomplete: {len(missing)} of {len(suite.jobs)} jobs "
            "are missing; pass --allow-incomplete only for a progress snapshot"
        )

    common_contract = _common_runtime_contract(shards)
    if (
        run_contract is not None
        and common_contract is not None
        and common_contract != run_contract["runtime_metadata"]
    ):
        raise BenchmarkContractError("shards differ from output-root run contract")
    summaries_by_scenario: dict[str, list[Mapping[str, Any]]] = {
        scenario.id: [] for scenario in suite.scenarios
    }
    shard_index: list[dict[str, Any]] = []
    for job, path, document in shards:
        summaries = document["summaries"]
        summaries_by_scenario[job.scenario.id].extend(summaries)
        shard_index.append(
            {
                "job_index": job.index,
                "scenario_id": job.scenario.id,
                "environment_seeds": list(job.env_seeds),
                "path": str(path.relative_to(output_root)),
                "sha256": sha256_file(path),
            }
        )

    scenario_results = []
    for scenario in suite.scenarios:
        summaries = summaries_by_scenario[scenario.id]
        completed_environment_seeds = sorted(
            {
                int(summary["env_seed"])
                for summary in summaries
                if isinstance(summary, Mapping)
            }
        )
        missing_environment_seeds = sorted(
            set(suite.benchmark_seeds) - set(completed_environment_seeds)
        )
        scenario_results.append(
            {
                "scenario_id": scenario.id,
                "role": scenario.role,
                "human_policy": scenario.human_policy,
                "dynamic_humans": scenario.dynamic_humans,
                "complete": not missing_environment_seeds,
                "completed_environment_seeds": completed_environment_seeds,
                "missing_environment_seeds": missing_environment_seeds,
                "planners": {
                    planner: _aggregate_metrics(
                        summary
                        for summary in summaries
                        if summary.get("planner") == planner
                    )
                    for planner in CANONICAL_PLANNERS
                },
            }
        )

    all_summaries = [
        summary
        for summaries in summaries_by_scenario.values()
        for summary in summaries
    ]
    role_results: dict[str, Any] = {}
    for role in ("primary", "sensitivity"):
        ids = {scenario.id for scenario in suite.scenarios if scenario.role == role}
        role_summaries = [
            summary
            for scenario_id, summaries in summaries_by_scenario.items()
            if scenario_id in ids
            for summary in summaries
        ]
        role_results[role] = {
            planner: _aggregate_metrics(
                summary
                for summary in role_summaries
                if summary.get("planner") == planner
            )
            for planner in CANONICAL_PLANNERS
        }

    return {
        "schema_version": "cfm_mppi.socnavgym_benchmark_index.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite.suite_id,
        "manifest": str(suite.manifest_path),
        "manifest_sha256": suite.manifest_sha256,
        "seed_file_sha256": suite.seed_sha256,
        "run_contract": (
            {
                "path": "run_contract.json",
                "sha256": sha256_file(run_contract_path),
            }
            if run_contract is not None
            else None
        ),
        "complete": not missing,
        "completed_jobs": len(shards),
        "expected_jobs": len(suite.jobs),
        "completed_scenario_seed_pairs": len(shards) * suite.seeds_per_job,
        "expected_scenario_seed_pairs": suite.expected_scenario_seed_pairs,
        "missing_jobs": missing,
        "runtime_contract": common_contract,
        "shards": shard_index,
        "scenario_results": scenario_results,
        "role_results": role_results,
        "overall": {
            planner: _aggregate_metrics(
                summary
                for summary in all_summaries
                if summary.get("planner") == planner
            )
            for planner in CANONICAL_PLANNERS
        },
    }


def _common_runtime_contract(
    shards: Sequence[tuple[BenchmarkJob, Path, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    if not shards:
        return None
    fields = _RUNTIME_METADATA_FIELDS
    reference_metadata = shards[0][2]["metadata"]
    contract = {field: reference_metadata.get(field) for field in fields}
    for job, _, document in shards[1:]:
        metadata = document["metadata"]
        candidate = {field: metadata.get(field) for field in fields}
        if candidate != contract:
            differences = [
                field for field in fields if candidate[field] != contract[field]
            ]
            raise BenchmarkContractError(
                f"job {job.index} runtime differs from other shards: {differences}"
            )
    return contract


def _aggregate_metrics(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(summaries)
    count = len(items)

    def count_true(key: str) -> int:
        return sum(bool(item.get(key, False)) for item in items)

    def mean(key: str) -> float | None:
        values = [_finite_float(item.get(key)) for item in items]
        finite = [value for value in values if value is not None]
        return sum(finite) / len(finite) if finite else None

    successes = count_true("success")
    decision_numerator = 0.0
    decision_steps = 0
    steady_numerator = 0.0
    steady_steps = 0
    for item in items:
        steps = int(item.get("steps", 0))
        decision = item.get("decision_latency")
        if isinstance(decision, Mapping):
            latency = _finite_float(decision.get("mean"))
            if latency is not None and steps > 0:
                decision_numerator += latency * steps
                decision_steps += steps
        steady = item.get("steady_state_decision_latency")
        if isinstance(steady, Mapping):
            latency = _finite_float(steady.get("mean"))
            if latency is not None and steps > 1:
                steady_numerator += latency * (steps - 1)
                steady_steps += steps - 1

    successful_times = [
        _finite_float(item.get("environment_time_to_reach_goal"))
        for item in items
        if bool(item.get("success", False))
    ]
    successful_times = [value for value in successful_times if value is not None]
    freezing_episodes = sum(int(item.get("freezing_events", 0)) > 0 for item in items)
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count if count else None,
        "collisions": count_true("collision_any"),
        "collision_rate": count_true("collision_any") / count if count else None,
        "timeouts": count_true("timeout"),
        "timeout_rate": count_true("timeout") / count if count else None,
        "freezing_episodes": freezing_episodes,
        "freezing_episode_rate": freezing_episodes / count if count else None,
        "mean_freezing_events": mean("freezing_events"),
        "mean_steps": mean("steps"),
        "mean_return": mean("return"),
        "mean_final_goal_distance": mean("final_goal_distance"),
        "mean_geometric_path_length": mean("geometric_path_length"),
        "mean_minimum_human_clearance": mean("minimum_human_clearance"),
        "mean_success_time_to_goal": (
            sum(successful_times) / len(successful_times)
            if successful_times
            else None
        ),
        "decision_latency_weighted_mean_seconds": (
            decision_numerator / decision_steps if decision_steps else None
        ),
        "steady_state_decision_latency_weighted_mean_seconds": (
            steady_numerator / steady_steps if steady_steps else None
        ),
        "planning_decisions": decision_steps,
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _write_new_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically publish a shard without ever replacing an existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BenchmarkContractError(
                f"refusing to overwrite existing benchmark shard: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_replace_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically refresh the derived aggregate index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    suite = load_benchmark_suite(args.suite)
    output_root = _output_root(args, suite)

    if args.list_jobs:
        if args.dry_run or args.allow_incomplete:
            raise BenchmarkContractError(
                "--dry-run/--allow-incomplete do not apply to --list-jobs"
            )
        list_jobs(suite, output_root)
        return 0
    if args.validate_configs:
        if args.dry_run or args.allow_incomplete:
            raise BenchmarkContractError(
                "--dry-run/--allow-incomplete do not apply to --validate-configs"
            )
        print(json.dumps(validate_environment_matrix(suite), indent=2, sort_keys=True))
        return 0
    if args.aggregate:
        if args.dry_run:
            raise BenchmarkContractError("--dry-run does not apply to --aggregate")
        document = aggregate_suite(
            suite, output_root, allow_incomplete=args.allow_incomplete
        )
        scenario_files = []
        for result in document["scenario_results"]:
            scenario_id = result["scenario_id"]
            relative_path = Path("scenarios") / f"{scenario_id}.json"
            scenario_document = {
                "schema_version": "cfm_mppi.socnavgym_scenario_index.v1",
                "created_at_utc": document["created_at_utc"],
                "suite_id": suite.suite_id,
                "manifest_sha256": suite.manifest_sha256,
                "complete": result["complete"],
                "run_contract": document["run_contract"],
                "scenario": result,
                "shards": [
                    shard
                    for shard in document["shards"]
                    if shard["scenario_id"] == scenario_id
                ],
            }
            _write_replace_json_atomic(
                output_root / relative_path, scenario_document
            )
            scenario_files.append(str(relative_path))
        document["scenario_files"] = scenario_files
        index_path = output_root / "benchmark_index.json"
        _write_replace_json_atomic(index_path, document)
        print(
            f"wrote {'complete' if document['complete'] else 'incomplete'} "
            f"benchmark index {index_path}"
        )
        return 0
    if args.allow_incomplete:
        raise BenchmarkContractError("--allow-incomplete requires --aggregate")

    job = suite.job(_job_index(args))
    run_job(args, suite, job, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
