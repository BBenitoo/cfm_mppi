"""Side-by-side static and animated views of matched SocNavGym planners.

The renderer consumes the opt-in trace produced by ``eval_socnavgym
--record-visualization``.  It deliberately keeps planner-internal predictions
separate from the simulator's current pedestrian state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cfm_mppi.evaluation.socnavgym_planners import VISUALIZATION_TRACE_SCHEMA


BASELINE_PLANNER = "cfm-mppi-cv"
VRC_PLANNER = "vrc-mppi"

COLOR_NO_VRC = "#6B7280"
COLOR_VRC = "#0072B2"
COLOR_ROBOT = "#E69F00"
COLOR_ROBOT_DARK = "#A85D00"
COLOR_TUBE = "#56B4E9"
COLOR_FORCE = "#CC79A7"
COLOR_HISTORY = "#374151"
COLOR_GOAL = "#009E73"
COLOR_GRID = "#E5E7EB"

PANEL_WSPACE = 0.03
LEGEND_FONTSIZE = 9
FIGURE_SIZE = (9.6, 5.8)


class VisualizationTraceError(ValueError):
    """The paired evaluation does not contain a usable visualization trace."""


@dataclass(frozen=True)
class PairedEpisodes:
    source: Path
    env_seed: int
    baseline: Mapping[str, Any]
    vrc: Mapping[str, Any]

    @property
    def time_step(self) -> float:
        return float(self.baseline["context"]["time_step"])


def _episode_seed(episode: Mapping[str, Any]) -> int:
    try:
        return int(episode["context"]["env_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualizationTraceError("episode is missing context.env_seed") from exc


def _steps(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise VisualizationTraceError(
            "visualization requires non-empty episode steps; do not use "
            "--summary-only when recording the trace"
        )
    if any(not isinstance(step, Mapping) for step in steps):
        raise VisualizationTraceError("episode steps must be JSON objects")
    return steps


def _trace(step: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = step.get("diagnostics")
    trace = diagnostics.get("visualization") if isinstance(diagnostics, Mapping) else None
    if not isinstance(trace, Mapping):
        raise VisualizationTraceError(
            "input has no planner visualization trace; rerun "
            "cfm_mppi.evaluation.eval_socnavgym with --record-visualization. "
            "Existing immutable benchmark shards cannot reconstruct VRC forecasts."
        )
    if trace.get("schema_version") != VISUALIZATION_TRACE_SCHEMA:
        raise VisualizationTraceError(
            "unsupported visualization trace schema: "
            f"{trace.get('schema_version')!r}"
        )
    return trace


def load_paired_episodes(
    path: str | Path,
    *,
    env_seed: int | None = None,
) -> PairedEpisodes:
    """Load one matched baseline/VRC episode pair from evaluation JSON."""
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise VisualizationTraceError(f"invalid evaluation JSON: {source}") from exc
    if not isinstance(document, Mapping):
        raise VisualizationTraceError("evaluation root must be a JSON object")
    episodes = document.get("episodes")
    if not isinstance(episodes, list):
        raise VisualizationTraceError("evaluation JSON has no episodes list")

    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise VisualizationTraceError("episode entries must be JSON objects")
        planner = episode.get("planner")
        if planner not in (BASELINE_PLANNER, VRC_PLANNER):
            continue
        key = (_episode_seed(episode), str(planner))
        if key in by_key:
            raise VisualizationTraceError(f"duplicate episode for seed/planner {key}")
        by_key[key] = episode

    paired_seeds = sorted(
        seed
        for seed, planner in by_key
        if planner == BASELINE_PLANNER and (seed, VRC_PLANNER) in by_key
    )
    if not paired_seeds:
        raise VisualizationTraceError(
            f"input must contain both {BASELINE_PLANNER!r} and {VRC_PLANNER!r} "
            "for the same environment seed"
        )
    if env_seed is None:
        if len(paired_seeds) != 1:
            raise VisualizationTraceError(
                "input contains multiple paired seeds; select one with --seed: "
                + ", ".join(str(seed) for seed in paired_seeds)
            )
        selected_seed = paired_seeds[0]
    else:
        selected_seed = int(env_seed)
        if selected_seed not in paired_seeds:
            raise VisualizationTraceError(
                f"seed {selected_seed} is not a complete pair; available paired "
                f"seeds: {paired_seeds}"
            )

    baseline = by_key[(selected_seed, BASELINE_PLANNER)]
    vrc = by_key[(selected_seed, VRC_PLANNER)]
    if baseline.get("initial_state") != vrc.get("initial_state"):
        raise VisualizationTraceError(
            "paired planners do not have an identical initial state"
        )
    baseline_steps = _steps(baseline)
    vrc_steps = _steps(vrc)
    for step in baseline_steps:
        _trace(step)
    for step in vrc_steps:
        vrc_trace = _trace(step)
        if vrc_trace.get("pedestrian_prediction_vrc") is None:
            raise VisualizationTraceError("VRC episode is missing its blue forecast")
        if vrc_trace.get("vrc_tube") is None:
            raise VisualizationTraceError("VRC episode is missing its causal tube")
        if vrc_trace.get("vrc_forces") is None:
            raise VisualizationTraceError("VRC episode is missing current forces")

    baseline_dt = float(baseline["context"]["time_step"])
    vrc_dt = float(vrc["context"]["time_step"])
    if not math.isclose(baseline_dt, vrc_dt, rel_tol=0.0, abs_tol=1e-12):
        raise VisualizationTraceError("paired episodes use different time steps")
    return PairedEpisodes(
        source=source,
        env_seed=selected_seed,
        baseline=baseline,
        vrc=vrc,
    )


def _finite_array(value: Any, *, name: str, last_dim: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VisualizationTraceError(f"{name} is not a numeric array") from exc
    if last_dim is not None and (array.ndim == 0 or array.shape[-1] != last_dim):
        raise VisualizationTraceError(
            f"{name} must end in dimension {last_dim}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise VisualizationTraceError(f"{name} contains non-finite values")
    return array


def choose_key_step(pair: PairedEpisodes) -> int:
    """Select the shared step with the clearest current/predicted VRC effect."""
    baseline_steps = _steps(pair.baseline)
    vrc_steps = _steps(pair.vrc)
    shared_length = min(len(baseline_steps), len(vrc_steps))
    if shared_length <= 0:
        raise VisualizationTraceError("paired episodes have no shared decision step")

    best_step = 0
    best_score = -math.inf
    for step_index in range(shared_length):
        trace = _trace(vrc_steps[step_index])
        no_vrc = _finite_array(
            trace["pedestrian_prediction_no_vrc"],
            name="no-VRC pedestrian prediction",
            last_dim=2,
        )
        with_vrc = _finite_array(
            trace["pedestrian_prediction_vrc"],
            name="VRC pedestrian prediction",
            last_dim=2,
        )
        forces = _finite_array(
            trace["vrc_forces"],
            name="VRC force",
            last_dim=2,
        )
        if no_vrc.shape != with_vrc.shape:
            raise VisualizationTraceError(
                "no-VRC and VRC pedestrian predictions have different shapes"
            )
        prediction_separation = float(
            np.linalg.norm(with_vrc - no_vrc, axis=-1).max(initial=0.0)
        )
        current_force = float(np.linalg.norm(forces, axis=-1).max(initial=0.0))

        baseline_robot = _finite_array(
            _trace(baseline_steps[step_index])["robot_prediction"],
            name="baseline robot prediction",
            last_dim=3,
        )
        vrc_robot = _finite_array(
            trace["robot_prediction"],
            name="VRC robot prediction",
            last_dim=3,
        )
        common = min(len(baseline_robot), len(vrc_robot))
        robot_separation = (
            float(
                np.linalg.norm(
                    baseline_robot[:common, :2] - vrc_robot[:common, :2],
                    axis=-1,
                ).max(initial=0.0)
            )
            if common
            else 0.0
        )
        score = prediction_separation + current_force + 0.25 * robot_separation
        if score > best_score:
            best_score = score
            best_step = step_index
    return best_step


def _state_sequence(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    initial = episode.get("initial_state")
    if not isinstance(initial, Mapping):
        raise VisualizationTraceError("episode is missing initial_state")
    sequence = [initial]
    for step in _steps(episode):
        next_state = step.get("next_state")
        if not isinstance(next_state, Mapping):
            raise VisualizationTraceError("step is missing next_state")
        sequence.append(next_state)
    return sequence


def _robot_history_states(
    episode: Mapping[str, Any],
    step_index: int,
) -> list[Mapping[str, Any]]:
    states = _state_sequence(episode)
    current = min(step_index, len(states) - 2)
    return states[: current + 1]


def _frame_step(episode: Mapping[str, Any], frame_index: int) -> int:
    return min(frame_index, len(_steps(episode)) - 1)


def _collect_state_points(states: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for state in states:
        robot_state = _finite_array(
            state.get("robot_state"), name="robot state", last_dim=3
        )
        points.append(robot_state[:2].reshape(1, 2))
        humans = state.get("humans", [])
        if not isinstance(humans, list):
            raise VisualizationTraceError("state humans must be a list")
        if humans:
            points.append(
                np.stack(
                    [
                        _finite_array(
                            human.get("position"),
                            name="human position",
                            last_dim=2,
                        )
                        for human in humans
                    ]
                )
            )
        goal = _finite_array(state.get("goal"), name="goal", last_dim=2)
        points.append(goal.reshape(1, 2))
    return points


def _collect_frame_points(
    episode: Mapping[str, Any],
    frame_index: int,
    *,
    is_vrc: bool,
    show_candidates: bool,
) -> list[np.ndarray]:
    step_index = _frame_step(episode, frame_index)
    step = _steps(episode)[step_index]
    state = step.get("state")
    if not isinstance(state, Mapping):
        raise VisualizationTraceError("step is missing its decision-time state")
    trace = _trace(step)
    history_states = _robot_history_states(episode, step_index)
    robot_history = np.stack(
        [
            _finite_array(
                history_state.get("robot_state"),
                name="robot state",
                last_dim=3,
            )[:2]
            for history_state in history_states
        ]
    )
    points = _collect_state_points([state])
    points.append(robot_history)
    trace_geometry = [("robot_prediction", 3)]
    if show_candidates:
        trace_geometry.append(("robot_candidate_trajectories", 3))
    if is_vrc:
        trace_geometry.extend(
            [
                ("robot_conditioning_trajectory", 3),
                ("pedestrian_prediction_vrc", 2),
            ]
        )
    else:
        trace_geometry.append(("pedestrian_prediction_no_vrc", 2))
    for key, last_dim in trace_geometry:
        value = trace.get(key)
        if value is None:
            continue
        array = _finite_array(value, name=key, last_dim=last_dim)
        points.append(array[..., :2].reshape(-1, 2))
    tube = trace.get("vrc_tube")
    if is_vrc and isinstance(tube, Mapping):
        centers = _finite_array(tube.get("centers"), name="VRC centers", last_dim=2)
        longitudinal = _finite_array(
            tube.get("longitudinal_radii"), name="VRC longitudinal radii"
        ).reshape(-1)
        lateral = _finite_array(
            tube.get("lateral_radii"), name="VRC lateral radii"
        ).reshape(-1)
        if len(centers) != len(longitudinal) or len(centers) != len(lateral):
            raise VisualizationTraceError("VRC tube arrays have different lengths")
        radii = np.maximum(longitudinal, lateral).reshape(-1, 1)
        points.extend((centers - radii, centers + radii))
    return points


def _shared_bounds(
    pair: PairedEpisodes,
    frame_indices: Sequence[int],
    *,
    show_candidates: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    points: list[np.ndarray] = []
    for frame_index in frame_indices:
        points.extend(
            _collect_frame_points(
                pair.baseline,
                frame_index,
                is_vrc=False,
                show_candidates=show_candidates,
            )
        )
        points.extend(
            _collect_frame_points(
                pair.vrc,
                frame_index,
                is_vrc=True,
                show_candidates=show_candidates,
            )
        )
    if not points:
        raise VisualizationTraceError("no finite geometry is available to plot")
    stacked = np.concatenate([point for point in points if point.size], axis=0)
    minimum = stacked.min(axis=0)
    maximum = stacked.max(axis=0)
    center = 0.5 * (minimum + maximum)
    span = max(float((maximum - minimum).max()), 1.0)
    half_span = 0.5 * span + max(0.6, 0.08 * span)
    return (
        (float(center[0] - half_span), float(center[0] + half_span)),
        (float(center[1] - half_span), float(center[1] + half_span)),
    )


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for SocNavGym visualization; activate the "
            "project's vrc environment or install matplotlib>=3.8"
        ) from exc
    return plt, animation, Line2D, Circle, Ellipse, FancyArrowPatch, Patch


def _draw_vrc_tube(ax: Any, trace: Mapping[str, Any], Ellipse: Any, stride: int) -> None:
    tube = trace.get("vrc_tube")
    if not isinstance(tube, Mapping):
        return
    centers = _finite_array(tube.get("centers"), name="VRC centers", last_dim=2)
    longitudinal = _finite_array(
        tube.get("longitudinal_radii"), name="VRC longitudinal radii"
    ).reshape(-1)
    lateral = _finite_array(
        tube.get("lateral_radii"), name="VRC lateral radii"
    ).reshape(-1)
    theta = _finite_array(tube.get("theta"), name="VRC theta").reshape(-1)
    length = len(centers)
    if not (length == len(longitudinal) == len(lateral) == len(theta)):
        raise VisualizationTraceError("VRC tube arrays have different lengths")
    indices = list(range(0, length, stride))
    if indices and indices[-1] != length - 1:
        indices.append(length - 1)
    for index in reversed(indices):
        progress = index / max(length - 1, 1)
        alpha = 0.025 + 0.14 * (1.0 - progress)
        ax.add_patch(
            Ellipse(
                xy=centers[index],
                width=2.0 * longitudinal[index],
                height=2.0 * lateral[index],
                angle=math.degrees(theta[index]),
                facecolor=COLOR_TUBE,
                edgecolor=COLOR_VRC,
                linewidth=0.7,
                alpha=alpha,
                zorder=1,
            )
        )


def _draw_robot(
    ax: Any,
    state: Mapping[str, Any],
    Circle: Any,
    FancyArrowPatch: Any,
) -> None:
    robot = _finite_array(state.get("robot_state"), name="robot state", last_dim=3)
    radius = float(state.get("robot_radius", 0.25))
    position = robot[:2]
    heading = float(robot[2])
    ax.add_patch(
        Circle(
            position,
            radius=radius,
            facecolor=COLOR_ROBOT,
            edgecolor=COLOR_ROBOT_DARK,
            linewidth=1.4,
            zorder=9,
        )
    )
    direction = np.asarray([math.cos(heading), math.sin(heading)])
    arrow_end = position + max(0.42, 1.8 * radius) * direction
    ax.add_patch(
        FancyArrowPatch(
            posA=position,
            posB=arrow_end,
            arrowstyle="-|>",
            mutation_scale=11,
            color=COLOR_ROBOT_DARK,
            linewidth=1.5,
            zorder=10,
        )
    )
    ax.text(
        position[0],
        position[1],
        "R",
        ha="center",
        va="center",
        fontsize=7,
        color="white",
        fontweight="bold",
        zorder=11,
    )


def _draw_humans(
    ax: Any,
    state: Mapping[str, Any],
    Circle: Any,
) -> None:
    for human in state.get("humans", []):
        position = _finite_array(
            human.get("position"), name="human position", last_dim=2
        )
        radius = float(human.get("radius", 0.3))
        ax.add_patch(
            Circle(
                position,
                radius=radius,
                facecolor="white",
                edgecolor=COLOR_HISTORY,
                linewidth=1.2,
                zorder=8,
            )
        )


def _draw_robot_start(ax: Any, episode: Mapping[str, Any]) -> None:
    initial_state = episode.get("initial_state")
    if not isinstance(initial_state, Mapping):
        raise VisualizationTraceError("episode is missing initial_state")
    robot = _finite_array(
        initial_state.get("robot_state"), name="initial robot state", last_dim=3
    )
    ax.plot(
        [robot[0]],
        [robot[1]],
        linestyle="none",
        marker="x",
        markersize=7.5,
        markerfacecolor=COLOR_ROBOT_DARK,
        markeredgecolor=COLOR_ROBOT_DARK,
        markeredgewidth=0.7,
        zorder=8,
    )


def _draw_predictions(
    ax: Any,
    state: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    show_no_vrc: bool,
    show_vrc: bool,
) -> None:
    humans = state.get("humans", [])
    current_positions = (
        np.stack(
            [
                _finite_array(human.get("position"), name="human position", last_dim=2)
                for human in humans
            ]
        )
        if humans
        else np.empty((0, 2), dtype=np.float64)
    )
    if show_no_vrc:
        no_vrc = _finite_array(
            trace.get("pedestrian_prediction_no_vrc"),
            name="no-VRC pedestrian prediction",
            last_dim=2,
        )
        if no_vrc.shape[0] != len(current_positions):
            raise VisualizationTraceError(
                "no-VRC pedestrian prediction does not match current humans"
            )
        for current, prediction in zip(current_positions, no_vrc):
            path = np.vstack((current, prediction))
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=COLOR_NO_VRC,
                linestyle=(0, (4, 3)),
                linewidth=1.35,
                alpha=0.85,
                zorder=5,
            )

    if not show_vrc:
        return
    with_vrc = _finite_array(
        trace.get("pedestrian_prediction_vrc"),
        name="VRC pedestrian prediction",
        last_dim=2,
    )
    if with_vrc.shape[0] != len(current_positions):
        raise VisualizationTraceError(
            "VRC pedestrian prediction does not match current humans"
        )
    for current, prediction in zip(current_positions, with_vrc):
        path = np.vstack((current, prediction))
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=COLOR_VRC,
            linewidth=1.65,
            alpha=0.95,
            zorder=6,
        )


def _draw_force_arrows(
    ax: Any,
    state: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    force_scale: float,
) -> None:
    humans = state.get("humans", [])
    if not humans:
        return
    positions = np.stack(
        [
            _finite_array(human.get("position"), name="human position", last_dim=2)
            for human in humans
        ]
    )
    forces = _finite_array(trace.get("vrc_forces"), name="VRC forces", last_dim=2)
    if forces.shape != positions.shape:
        raise VisualizationTraceError("VRC forces do not match current humans")
    magnitudes = np.linalg.norm(forces, axis=1)
    active = magnitudes > 1e-6
    if not active.any():
        return
    directions = forces[active] / magnitudes[active, np.newaxis]
    display_lengths = np.minimum(
        0.9,
        0.25 + magnitudes[active] * force_scale,
    )
    display_vectors = directions * display_lengths[:, np.newaxis]
    ax.quiver(
        positions[active, 0],
        positions[active, 1],
        display_vectors[:, 0],
        display_vectors[:, 1],
        color=COLOR_FORCE,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.006,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.5,
        zorder=12,
    )


def _draw_goal(ax: Any, state: Mapping[str, Any], Circle: Any) -> None:
    goal = _finite_array(state.get("goal"), name="goal", last_dim=2)
    goal_radius = float(state.get("goal_radius", 0.35))
    ax.add_patch(
        Circle(
            goal,
            radius=goal_radius,
            facecolor=COLOR_GOAL,
            edgecolor=COLOR_GOAL,
            linewidth=1.2,
            alpha=0.16,
            zorder=2,
        )
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        s=70,
        color=COLOR_GOAL,
        edgecolors="white",
        linewidths=0.5,
        zorder=7,
    )


def _draw_robot_predictions(
    ax: Any,
    trace: Mapping[str, Any],
    *,
    show_candidates: bool,
    show_conditioning: bool,
) -> None:
    if show_candidates:
        candidates = _finite_array(
            trace.get("robot_candidate_trajectories"),
            name="robot candidate trajectories",
            last_dim=3,
        )
        for candidate in candidates:
            ax.plot(
                candidate[:, 0],
                candidate[:, 1],
                color=COLOR_ROBOT,
                linewidth=0.85,
                alpha=0.16,
                zorder=2,
            )
    if show_conditioning:
        conditioning = _finite_array(
            trace.get("robot_conditioning_trajectory"),
            name="VRC conditioning trajectory",
            last_dim=3,
        )
        ax.plot(
            conditioning[:, 0],
            conditioning[:, 1],
            color=COLOR_ROBOT_DARK,
            linestyle=(0, (1.5, 2.2)),
            linewidth=1.25,
            alpha=0.8,
            zorder=3,
        )
    prediction = _finite_array(
        trace.get("robot_prediction"), name="robot prediction", last_dim=3
    )
    ax.plot(
        prediction[:, 0],
        prediction[:, 1],
        color=COLOR_ROBOT,
        linewidth=2.35,
        zorder=7,
    )


def _draw_panel(
    ax: Any,
    episode: Mapping[str, Any],
    frame_index: int,
    *,
    is_vrc: bool,
    tube_stride: int,
    force_scale: float,
    show_candidates: bool,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    Circle: Any,
    Ellipse: Any,
    FancyArrowPatch: Any,
) -> None:
    steps = _steps(episode)
    step_index = _frame_step(episode, frame_index)
    step = steps[step_index]
    state = step.get("state")
    if not isinstance(state, Mapping):
        raise VisualizationTraceError("step is missing its decision-time state")
    trace = _trace(step)
    history = _robot_history_states(episode, step_index)

    ax.set_facecolor("white")
    ax.grid(True, color=COLOR_GRID, linewidth=0.7, zorder=0)
    _draw_vrc_tube(ax, trace, Ellipse, tube_stride)
    _draw_goal(ax, state, Circle)
    _draw_robot_predictions(
        ax,
        trace,
        show_candidates=show_candidates,
        show_conditioning=is_vrc,
    )
    _draw_humans(ax, state, Circle)
    _draw_predictions(
        ax,
        state,
        trace,
        show_no_vrc=not is_vrc,
        show_vrc=is_vrc,
    )
    if is_vrc:
        _draw_force_arrows(ax, state, trace, force_scale=force_scale)

    robot_history = np.stack(
        [
            _finite_array(item.get("robot_state"), name="robot state", last_dim=3)[:2]
            for item in history
        ]
    )
    ax.plot(
        robot_history[:, 0],
        robot_history[:, 1],
        color=COLOR_ROBOT_DARK,
        linewidth=1.35,
        alpha=0.55,
        zorder=4,
    )
    _draw_robot_start(ax, episode)
    _draw_robot(ax, state, Circle, FancyArrowPatch)

    x_limits, y_limits = bounds
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    if not is_vrc:
        ax.set_ylabel("y [m]")
    ax.tick_params(labelsize=8)
    label = "VRC" if is_vrc else "Baseline"
    time_seconds = step_index * float(episode["context"]["time_step"])
    ax.set_title(
        f"{label}\nt = {time_seconds:.1f} s",
        fontsize=10,
        fontweight="bold",
    )


def _legend_handles(Line2D: Any, Patch: Any) -> list[Any]:
    return [
        Line2D(
            [0], [0], color=COLOR_ROBOT, linewidth=2.35, label="Current robot plan"
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_ROBOT_DARK,
            linewidth=1.25,
            linestyle=(0, (1.5, 2.2)),
            label="VRC conditioning branch",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_NO_VRC,
            linewidth=1.35,
            linestyle=(0, (4, 3)),
            label="CV forecast",
        ),
        Line2D(
            [0], [0], color=COLOR_VRC, linewidth=1.65, label="VRC forecast"
        ),
        Patch(
            facecolor=COLOR_TUBE,
            edgecolor=COLOR_VRC,
            alpha=0.22,
            label="Temporal VRC ellipses",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_HISTORY,
            linewidth=0,
            marker="o",
            markerfacecolor="white",
            label="Current pedestrians",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_ROBOT_DARK,
            linewidth=1.35,
            alpha=0.55,
            label="Robot history",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_ROBOT_DARK,
            linewidth=0,
            marker="x",
            markeredgecolor=COLOR_ROBOT_DARK,
            label="Robot start",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_FORCE,
            marker=r"$\rightarrow$",
            markersize=12,
            linewidth=0,
            label="Current VRC force",
        ),
    ]


def _new_figure(plt: Any):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        sharex=True,
        sharey=True,
    )
    figure.patch.set_facecolor("white")
    return figure, axes


def _configure_figure(
    figure: Any,
    pair: PairedEpisodes,
    Line2D: Any,
    Patch: Any,
) -> None:
    figure.suptitle(
        f"Environment seed {pair.env_seed}",
        fontsize=12,
        fontweight="bold",
        y=0.975,
    )
    figure.legend(
        handles=_legend_handles(Line2D, Patch),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.86,
        bottom=0.19,
        wspace=PANEL_WSPACE,
    )


def render_static_comparison(
    pair: PairedEpisodes,
    output_path: str | Path,
    *,
    step_index: int | None = None,
    history_steps: int = 30,
    tube_stride: int = 5,
    force_scale: float = 0.25,
    show_candidates: bool = True,
    dpi: int = 220,
) -> Path:
    """Render one aligned baseline/VRC decision as a publication-ready figure."""
    if history_steps < 0:
        raise ValueError("history_steps must be non-negative")
    if tube_stride <= 0 or dpi <= 0 or force_scale <= 0:
        raise ValueError("tube_stride, force_scale, and dpi must be positive")
    shared_length = min(len(_steps(pair.baseline)), len(_steps(pair.vrc)))
    selected_step = choose_key_step(pair) if step_index is None else int(step_index)
    if selected_step < 0 or selected_step >= shared_length:
        raise ValueError(
            f"static step must be in [0, {shared_length - 1}], got {selected_step}"
        )

    plt, _, Line2D, Circle, Ellipse, FancyArrowPatch, Patch = _matplotlib()
    bounds = _shared_bounds(
        pair,
        [selected_step],
        show_candidates=show_candidates,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "text.color": "#111827",
        }
    ):
        figure, axes = _new_figure(plt)
        _draw_panel(
            axes[0],
            pair.baseline,
            selected_step,
            is_vrc=False,
            tube_stride=tube_stride,
            force_scale=force_scale,
            show_candidates=show_candidates,
            bounds=bounds,
            Circle=Circle,
            Ellipse=Ellipse,
            FancyArrowPatch=FancyArrowPatch,
        )
        _draw_panel(
            axes[1],
            pair.vrc,
            selected_step,
            is_vrc=True,
            tube_stride=tube_stride,
            force_scale=force_scale,
            show_candidates=show_candidates,
            bounds=bounds,
            Circle=Circle,
            Ellipse=Ellipse,
            FancyArrowPatch=FancyArrowPatch,
        )
        _configure_figure(figure, pair, Line2D, Patch)
        figure.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
        plt.close(figure)
    return output


def render_animation_comparison(
    pair: PairedEpisodes,
    output_path: str | Path,
    *,
    history_steps: int = 30,
    frame_stride: int = 1,
    tube_stride: int = 5,
    force_scale: float = 0.25,
    show_candidates: bool = True,
    fps: float = 8.0,
    dpi: int = 110,
) -> Path:
    """Render the paired decisions as GIF or MP4 with a fixed shared frame."""
    if history_steps < 0:
        raise ValueError("history_steps must be non-negative")
    if any(value <= 0 for value in (frame_stride, tube_stride, force_scale, fps, dpi)):
        raise ValueError("animation stride, scale, fps, and dpi must be positive")
    max_length = max(len(_steps(pair.baseline)), len(_steps(pair.vrc)))
    frame_indices = list(range(0, max_length, int(frame_stride)))
    if frame_indices[-1] != max_length - 1:
        frame_indices.append(max_length - 1)

    plt, mpl_animation, Line2D, Circle, Ellipse, FancyArrowPatch, Patch = _matplotlib()
    bounds = _shared_bounds(
        pair,
        frame_indices,
        show_candidates=show_candidates,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in (".gif", ".mp4"):
        raise ValueError("animation output must end in .gif or .mp4")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#9CA3AF",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "text.color": "#111827",
        }
    ):
        figure, axes = _new_figure(plt)
        _configure_figure(figure, pair, Line2D, Patch)

        def update(frame_index: int):
            for axis in axes:
                axis.clear()
            _draw_panel(
                axes[0],
                pair.baseline,
                frame_index,
                is_vrc=False,
                tube_stride=tube_stride,
                force_scale=force_scale,
                show_candidates=show_candidates,
                bounds=bounds,
                Circle=Circle,
                Ellipse=Ellipse,
                FancyArrowPatch=FancyArrowPatch,
            )
            _draw_panel(
                axes[1],
                pair.vrc,
                frame_index,
                is_vrc=True,
                tube_stride=tube_stride,
                force_scale=force_scale,
                show_candidates=show_candidates,
                bounds=bounds,
                Circle=Circle,
                Ellipse=Ellipse,
                FancyArrowPatch=FancyArrowPatch,
            )
            return tuple(axes)

        movie = mpl_animation.FuncAnimation(
            figure,
            update,
            frames=frame_indices,
            interval=1000.0 / fps,
            repeat=False,
            blit=False,
        )
        try:
            if suffix == ".gif":
                writer = mpl_animation.PillowWriter(fps=fps)
            else:
                if not mpl_animation.writers.is_available("ffmpeg"):
                    raise RuntimeError(
                        "MP4 export requires an ffmpeg executable visible to "
                        "Matplotlib; use --animation-format gif or install ffmpeg"
                    )
                writer = mpl_animation.FFMpegWriter(
                    fps=fps,
                    codec="h264",
                    extra_args=["-pix_fmt", "yuv420p"],
                )
            movie.save(output, writer=writer, dpi=dpi)
        finally:
            plt.close(figure)
    return output


def _parse_step(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        step = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("step must be 'auto' or a non-negative integer") from exc
    if step < 0:
        raise argparse.ArgumentTypeError("step must be non-negative")
    return step


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render matched baseline/VRC SocNavGym planner traces as a static "
            "comparison and an aligned animation."
        )
    )
    parser.add_argument("input", type=Path, help="paired evaluation JSON with trace")
    parser.add_argument("--seed", type=int, help="environment seed when input has several")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--step", type=_parse_step, default=None, metavar="auto|N")
    parser.add_argument(
        "--history-steps",
        type=int,
        default=30,
        help=(
            "deprecated compatibility option; robot history is always shown "
            "from the start and pedestrian history is hidden"
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tube-stride", type=int, default=5)
    parser.add_argument("--force-scale", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--static-dpi", type=int, default=220)
    parser.add_argument("--animation-dpi", type=int, default=110)
    parser.add_argument(
        "--static-format", choices=("png", "pdf", "svg"), default="png"
    )
    parser.add_argument(
        "--animation-format", choices=("gif", "mp4", "none"), default="gif"
    )
    parser.add_argument("--hide-candidates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.history_steps < 0:
        raise ValueError("history_steps must be non-negative")
    pair = load_paired_episodes(args.input, env_seed=args.seed)
    selected_step = choose_key_step(pair) if args.step is None else args.step
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else pair.source.parent / f"visualization-seed-{pair.env_seed}"
    )
    static_path = output_dir / (
        f"seed-{pair.env_seed}-step-{selected_step:03d}.{args.static_format}"
    )
    render_static_comparison(
        pair,
        static_path,
        step_index=selected_step,
        history_steps=args.history_steps,
        tube_stride=args.tube_stride,
        force_scale=args.force_scale,
        show_candidates=not args.hide_candidates,
        dpi=args.static_dpi,
    )
    print(f"wrote {static_path}")
    if args.animation_format != "none":
        animation_path = output_dir / (
            f"seed-{pair.env_seed}.{args.animation_format}"
        )
        render_animation_comparison(
            pair,
            animation_path,
            history_steps=args.history_steps,
            frame_stride=args.frame_stride,
            tube_stride=args.tube_stride,
            force_scale=args.force_scale,
            show_candidates=not args.hide_candidates,
            fps=args.fps,
            dpi=args.animation_dpi,
        )
        print(f"wrote {animation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PairedEpisodes",
    "VisualizationTraceError",
    "choose_key_step",
    "load_paired_episodes",
    "render_animation_comparison",
    "render_static_comparison",
]
