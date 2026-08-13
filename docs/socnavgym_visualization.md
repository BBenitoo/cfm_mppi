# SocNavGym baseline/VRC visualization

The visualization workflow renders a matched environment seed as two aligned
panels:

- **Baseline · no VRC** shows the CFM+MPPI controller planned against a
  constant-velocity pedestrian forecast.
- **VRC · conditioned forecast** shows VRC+MPPI, including the selected
  branch's VRC-conditioned pedestrian forecast, temporal VRC ellipses, and
  current VRC force.

One invocation produces a publication-ready static comparison and, by default,
an aligned GIF of the complete closed-loop episodes. The two panels share axes
and decision indices. They have the same seeded initial state, but their
simulator states can diverge after the planners choose different controls.

## 1. Record a standalone trace

Visualization data is opt-in because the candidate trajectories and
time-indexed forecasts make the evaluation JSON substantially larger. Generate
a dedicated trace with `eval_socnavgym --record-visualization`; do not overwrite
an immutable benchmark shard. Existing shards without this flag cannot be used
to reconstruct the planner-internal VRC forecast or tube.

Trace capture also performs extra diagnostic geometry work and device-to-host
serialization. Treat the resulting latency fields as visualization-run
diagnostics, not as formal benchmark timings; use the original locked shards
for performance results.

Activate the same environment used for evaluation and configure headless
rendering:

```bash
conda activate vrc
export DGLBACKEND=pytorch
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cfm-mppi-matplotlib"
export SDL_VIDEODRIVER=dummy
mkdir -p "$MPLCONFIGDIR"
```

For example, this records the benchmark-v3 SFM sensitivity scenario at
environment seed `1022`, using the v3 planner seed and execution-order
contract:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym \
  --config configs/socnavgym/benchmark_v3/sfm_fixed_10.yaml \
  --checkpoint cfm_mppi/output_dir/cfm_transformer/checkpoint.pth \
  --planner both \
  --seeds 1022 \
  --planner-seed-offset 1000000 \
  --execution-order-offset 0 \
  --device cuda \
  --record-visualization \
  --output output_dir/socnavgym/visualization/v3-sfm-seed-1022.json
```

Seed `1022` is job `322` for this SFM scenario. It is an even job, hence
`--execution-order-offset 0` reproduces its CFM-first ordering. Use
`configs/socnavgym/benchmark_v3/orca_fixed_10.yaml` instead when the simulator
human policy should be ORCA.

The side-by-side input requires both planners and non-summary per-step records.
Keep `--planner both`, and do not combine `--record-visualization` with
`--summary-only`. Omit `--max-steps` when the animation should cover the full
episode.

## 2. Export the static figure and GIF

Pass the standalone evaluation JSON to `visualize_socnavgym`:

```bash
python -m cfm_mppi.evaluation.visualize_socnavgym \
  output_dir/socnavgym/visualization/v3-sfm-seed-1022.json \
  --seed 1022 \
  --output-dir output_dir/socnavgym/visualization/v3-sfm-seed-1022
```

This writes:

- `seed-1022-step-NNN.png`, the static side-by-side comparison;
- `seed-1022.gif`, the aligned animation.

`--seed` may be omitted when the input contains exactly one complete baseline
and VRC pair. If `--output-dir` is omitted, outputs go next to the trace in a
`visualization-seed-1022` directory.

The static step defaults to `--step auto`. Auto selection considers only
decision steps available in both episodes and chooses the step with the largest
combined visible VRC effect:

```text
max pedestrian forecast separation
+ max current VRC-force magnitude
+ 0.25 * max baseline/VRC robot-plan separation
```

Pass `--step N` to select a specific zero-based decision step. Useful rendering
options include:

```bash
python -m cfm_mppi.evaluation.visualize_socnavgym TRACE.json \
  --seed 1022 \
  --step 40 \
  --tube-stride 2 \
  --force-scale 0.25 \
  --frame-stride 2 \
  --fps 8 \
  --static-format pdf \
  --animation-format gif
```

- The executed robot path is always shown from its starting position. Pedestrian
  history trails and pedestrian ID labels are intentionally hidden.
- `--history-steps` remains accepted for compatibility with older commands but
  no longer clips the robot path.
- `--tube-stride 5` draws every fifth temporal ellipse plus the last ellipse.
- `--force-scale` changes arrow length for display only, not the recorded force.
  Nonzero arrows have a small minimum display length so their direction remains
  legible; read exact magnitudes from the trace rather than the drawn length.
- `--frame-stride` subsamples animation decisions. If one episode ends first,
  its last decision remains frozen while the other finishes.
- `--hide-candidates` suppresses the faint robot candidates. Use
  `--animation-format none` for a static-only export, or `mp4` when `ffmpeg` is
  available.

## Reading the figure

Both panels show their decision-time robot pose, complete executed robot path,
current pedestrian positions, and orange robot plans. The left baseline panel
shows gray CV pedestrian forecasts. The right-hand VRC panel instead shows the
conditioning branch, blue VRC forecast, temporal tube, and force arrows.

| Appearance | Meaning |
| --- | --- |
| Orange robot disc and dark-orange arrow | Current robot position and heading |
| Small dark-orange triangle | Robot starting position |
| Dark-orange trail | Executed robot path from the starting position |
| Faint orange paths | Candidate CFM robot trajectories |
| Orange solid path | Final selected MPPI robot prediction |
| Dark-orange short-dashed path | Selected CFM conditioning branch used to build the VRC tube |
| Gray dashed paths in the left panel | No-VRC constant-velocity (`CV`) pedestrian forecasts |
| Blue solid paths | VRC-conditioned pedestrian forecasts for the selected branch |
| Translucent light-blue ellipses | The selected branch's temporal VRC influence tube |
| Unnumbered dark-gray-outlined circles | Current simulator-observed pedestrian positions |
| Magenta arrows | Current nonzero planner-internal VRC force on each pedestrian |
| Green star/region | Robot goal and goal radius |

The baseline label must not be interpreted as an SFM predictor. Its planning
forecast is constant velocity in both benchmark scenarios. In the
`sfm_fixed_10` scenario, SFM is the **SocNavGym simulator's pedestrian policy**,
whereas the gray dashed line is still the planner's CV forecast. The blue line
is the VRC-conditioned forecast produced inside VRC+MPPI.

Likewise, the VRC tube and force arrows are planner diagnostics, not additional
simulator dynamics. They are built from the selected pre-MPPI conditioning
branch and used to form the VRC pedestrian prediction. The evaluator still
passes only the physical robot command `[v, omega]` (mapped to SocNavGym's
action format) to the environment; it never applies the recorded VRC force or
tube directly to SocNavGym pedestrians. The displayed current pedestrian
positions come from `env.step()` and therefore represent simulator truth, not
an internal rollout.
