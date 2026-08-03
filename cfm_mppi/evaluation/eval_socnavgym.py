"""Command-line entry point for matched SocNavGym closed-loop evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import torch

from cfm_mppi.evaluation.socnavgym_adapter import (
    HUMAN_GOAL_REACHED_POLICY,
    SocNavGymAdapter,
)
from cfm_mppi.evaluation.socnavgym_planners import (
    SocNavCFMMPPIPlanner,
    SocNavPlannerConfig,
    SocNavVRCMPPIPlanner,
)
from cfm_mppi.evaluation.socnavgym_runner import (
    PairedEvaluationResult,
    run_paired_socnavgym_evaluation,
)
from cfm_mppi.models.transformer import TransformerModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "socnavgym" / "probe_v1_world.yaml"
CHECKPOINT_CANDIDATES = (
    REPOSITORY_ROOT / "output_dir" / "cfm_transformer" / "checkpoint.pth",
    REPOSITORY_ROOT
    / "cfm_mppi"
    / "output_dir"
    / "cfm_transformer"
    / "checkpoint.pth",
)
DEFAULT_CHECKPOINT = next(
    (path for path in CHECKPOINT_CANDIDATES if path.is_file()),
    CHECKPOINT_CANDIDATES[0],
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "output_dir" / "socnavgym" / "evaluation.json"
)


def parse_seed_spec(specification: str) -> tuple[int, ...]:
    """Parse comma-separated integers and Python-style ``start:stop[:step]``."""
    seeds: list[int] = []
    for raw_item in specification.split(","):
        item = raw_item.strip()
        if not item:
            raise argparse.ArgumentTypeError("seed specification contains an empty item")
        if ":" not in item:
            try:
                seeds.append(int(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid seed: {item!r}") from exc
            continue
        components = item.split(":")
        if len(components) not in (2, 3) or any(part == "" for part in components):
            raise argparse.ArgumentTypeError(f"invalid seed range: {item!r}")
        try:
            values = [int(part) for part in components]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed range: {item!r}") from exc
        start, stop = values[:2]
        step = values[2] if len(values) == 3 else 1
        if step == 0:
            raise argparse.ArgumentTypeError("seed range step must not be zero")
        seeds.extend(range(start, stop, step))
    if not seeds:
        raise argparse.ArgumentTypeError("seed specification produced no seeds")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("environment seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seed specification contains duplicates")
    return tuple(seeds)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(device.type)


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    allow_random_model: bool,
) -> tuple[TransformerModel, bool]:
    checkpoint_loaded = checkpoint_path.is_file()
    if not checkpoint_loaded and not allow_random_model:
        raise FileNotFoundError(
            f"CFM checkpoint not found: {checkpoint_path}. Download the pretrained "
            "checkpoint or use --allow-random-model only for plumbing smoke tests."
        )
    # A fixed initialization makes explicit smoke-only random runs repeatable.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TransformerModel()
    if checkpoint_loaded:
        # Project training checkpoints contain an argparse.Namespace alongside
        # tensor state. Allow only that known legacy metadata type while
        # retaining PyTorch's weights-only unpickler.
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("checkpoint must be a mapping containing the 'model' key")
        model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    model.eval()
    return model, checkpoint_loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    paths = (
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "socnavgym_adapter.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "socnavgym_runner.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "socnavgym_planners.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "eval_socnavgym.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "socnavgym_benchmark.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "eval_socnavgym_suite.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "eval_vrc.py",
        REPOSITORY_ROOT / "cfm_mppi" / "evaluation" / "eval_vrc_cv_prediction.py",
        REPOSITORY_ROOT / "cfm_mppi" / "mppi" / "flowmppi.py",
        REPOSITORY_ROOT / "cfm_mppi" / "vrc" / "build_vrc.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _distribution_commit(distribution: str) -> str | None:
    try:
        direct_url_text = importlib.metadata.distribution(distribution).read_text(
            "direct_url.json"
        )
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    return str(commit) if commit else None


def _module_binary_metadata(module_name: str) -> dict[str, str | None]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return {"path": None, "sha256": None}
    path = Path(spec.origin).resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _git_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"repository_commit": None, "repository_dirty": None}
    return {
        "repository_commit": revision or None,
        "repository_dirty": bool(status),
    }


def _planner_factories(
    selection: str,
    *,
    model: TransformerModel,
    config: SocNavPlannerConfig,
    device: torch.device,
):
    available = {
        SocNavCFMMPPIPlanner.name: lambda: SocNavCFMMPPIPlanner(
            model,
            config=config,
            device=device,
        ),
        SocNavVRCMPPIPlanner.name: lambda: SocNavVRCMPPIPlanner(
            model,
            config=config,
            device=device,
        ),
    }
    if selection == "both":
        return available
    selected_name = {
        "cfm": SocNavCFMMPPIPlanner.name,
        "vrc": SocNavVRCMPPIPlanner.name,
    }[selection]
    return {selected_name: available[selected_name]}


def _result_document(
    result: PairedEvaluationResult,
    *,
    args: argparse.Namespace,
    planner_config: SocNavPlannerConfig,
    device: torch.device,
    checkpoint_loaded: bool,
) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    document = result.to_dict(include_steps=not args.summary_only)
    document["metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment_config": str(config_path),
        "environment_config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path) if checkpoint_loaded else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_loaded else None,
        "random_model_smoke_only": not checkpoint_loaded,
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
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
        "implementation_sha256": _implementation_sha256(),
        "planner_config": asdict(planner_config),
        "planner_seed_offset": args.planner_seed_offset,
        "execution_order_offset": getattr(args, "execution_order_offset", 0),
        **_git_metadata(),
    }
    return document


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched CFM+MPPI and VRC+MPPI controllers in fresh "
            "SocNavGym-v1 environments."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--planner", choices=("both", "cfm", "vrc"), default="both")
    parser.add_argument("--seeds", type=parse_seed_spec, default=parse_seed_spec("0"))
    parser.add_argument("--planner-seed-offset", type=int, default=0)
    parser.add_argument("--execution-order-offset", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--allow-random-model",
        action="store_true",
        help="allow an untrained model strictly for integration smoke tests",
    )
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--cfm-candidates", type=int, default=200)
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--mppi-samples-per-branch", type=int, default=200)
    parser.add_argument("--robot-start", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--robot-goal", type=float, nargs=2, metavar=("X", "Y"))
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Run one configured evaluation and return its unwritten JSON document.

    Keeping computation separate from output lets the locked benchmark suite
    attach its job contract before publishing an immutable shard.
    """
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SocNavGym config not found: {config_path}")
    device = _resolve_device(args.device)
    planner_config = SocNavPlannerConfig(
        horizon=args.horizon,
        max_history=args.max_history,
        cfm_candidates=args.cfm_candidates,
        branches=args.branches,
        mppi_samples_per_branch=args.mppi_samples_per_branch,
    )
    model, checkpoint_loaded = _load_model(
        checkpoint_path,
        device=device,
        allow_random_model=args.allow_random_model,
    )
    factories = _planner_factories(
        args.planner,
        model=model,
        config=planner_config,
        device=device,
    )
    robot_start = getattr(args, "robot_start", None)
    robot_goal = getattr(args, "robot_goal", None)
    if (robot_start is None) != (robot_goal is None):
        raise ValueError("robot_start and robot_goal must be specified together")
    paired_result = run_paired_socnavgym_evaluation(
        lambda: SocNavGymAdapter(
            config_path,
            fixed_robot_start=robot_start,
            fixed_robot_goal=robot_goal,
        ),
        factories,
        env_seeds=args.seeds,
        planner_seed_for_env=lambda env_seed: env_seed + args.planner_seed_offset,
        max_steps=args.max_steps,
        execution_order_offset=getattr(args, "execution_order_offset", 0),
    )
    document = _result_document(
        paired_result,
        args=args,
        planner_config=planner_config,
        device=device,
        checkpoint_loaded=checkpoint_loaded,
    )
    return document


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output_path = Path(args.output).expanduser().resolve()
    document = run_evaluation(args)
    _write_json(output_path, document)
    print(json.dumps(document["summaries"], indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
