# SocNavGym benchmark v1

This benchmark is the formal matched comparison of CFM+MPPI and VRC+MPPI in
an interactive pedestrian simulator that is independent of the project's
legacy VRC-SFM loop.  VRC affects only the planner's internal pedestrian
forecast; SocNavGym remains the source of closed-loop truth for both methods.

## Locked experiment matrix

The manifest is
`configs/socnavgym/benchmark_v1/suite.json`, and the explicit seed list is
`configs/socnavgym/benchmark_v1/seeds.json`.

| Scenario | Report group | SocNavGym pedestrian policy | Dynamic humans |
| --- | --- | --- | ---: |
| `orca_sparse_2` | primary | ORCA | 2 |
| `orca_medium_5` | primary | ORCA | 5 |
| `orca_dense_8` | primary | ORCA | 8 |
| `sfm_sparse_2` | sensitivity | SocNavGym SFM | 2 |
| `sfm_medium_5` | sensitivity | SocNavGym SFM | 5 |
| `sfm_dense_8` | sensitivity | SocNavGym SFM | 8 |

All six configs use the same 10 m by 10 m open map, differential-drive robot,
physical control limits, 0.1 s timestep, 256-step episode limit, and no static
objects, walls, formations, or interaction groups.  This avoids static
obstacles that the current planner does not observe.  The ORCA scenarios are
the primary benchmark.  SocNavGym's own SFM scenarios are a separately
reported policy-sensitivity check, not the legacy VRC-SFM simulator.

SocNavGym v1 normally lets 15 seconds of real wall-clock time force a human to
"reach" its goal. That would couple pedestrian dynamics to planner latency.
The adapter therefore installs the locked `geometric-only-v1` goal condition:
humans select a new goal only after physically entering the goal region. The
policy name is recorded in every shard and checked by the aggregator. Results
should therefore be described as using the patched SocNavGym-v1
`geometric-only-v1` protocol, not unmodified upstream goal timing.

The formal environment seeds are the frozen explicit list `1000..1029`.
Development seeds `17` and `29` are disjoint and must not be substituted into
formal results.  Every scenario uses every formal seed:

- 6 scenarios x 30 seeds = 180 matched scenario-seed pairs;
- 2 planners per pair = 360 full episodes;
- 1 paired scenario-seed result per H100 task = 180 Slurm array tasks.

A seed matches CFM and VRC within one exact scenario. It does not imply the
same initial geometry across different densities or pedestrian policies,
because those configs consume different random streams.

Each task keeps both planners for one seed together as the smallest recoverable
fairness unit. Job parity alternates CFM-first and VRC-first, so process/model/
CUDA cold-start order is counterbalanced across the 180 tasks. Episodes remain
stored in canonical planner order regardless of execution order.

The fixed per-decision planner settings are horizon 80, history 10, 200 CFM
candidates, 10 branches, and 200 MPPI samples per branch.  Formal suite jobs do
not expose CLI overrides for planners, seeds, budgets, or episode length.
The manifest also locks `device=cuda` and requires an H100 device name; a CPU
or another GPU class cannot produce a valid formal shard.

## Preflight on the HPC login node

Use the dedicated environment and project checkout:

```bash
cd /users/3149157r/MSc/cfm_mppi
conda activate vrc
export DGLBACKEND=pytorch
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cfm-mppi-benchmark-matplotlib"
export SDL_VIDEODRIVER=dummy
mkdir -p "$MPLCONFIGDIR"

python -m cfm_mppi.evaluation.eval_socnavgym_suite --list-jobs
python -m cfm_mppi.evaluation.eval_socnavgym_suite --job-index 0 --dry-run
python -m cfm_mppi.evaluation.eval_socnavgym_suite --validate-configs
```

`--validate-configs` constructs the six real SocNavGym environments and resets
all 180 scenario-seed pairs.  It verifies spawn feasibility, exact human
counts, stable sorted IDs, timestep, and episode length without running either
planner. Each construction/reset has a 30-second deadline because the upstream
placement loop can otherwise retry an impossible seeded layout indefinitely.

Before a formal job, `git status --short` must be empty.  The suite refuses to
produce formal output from a dirty checkout.  It also verifies that the model
matches the checkpoint hash frozen in the manifest; the detected migrated HPC
path is currently
`cfm_mppi/output_dir/cfm_transformer/checkpoint.pth`.

Create the Slurm log directory before submission (Slurm opens its output file
before the script starts):

```bash
mkdir -p /users/3149157r/MSc/cfm_mppi/cfm_mppi/evaluation/logs
```

## Pilot, then full H100 benchmark

First submit one representative shard of every scenario as a full-budget,
full-episode timing pilot. The selected IDs also exercise both process-start
planner orders. Keep pilot output separate from formal results:

```bash
sbatch \
  --array=0,31,60,91,120,151%5 \
  --export=ALL,SOCNAV_BENCHMARK_OUTPUT_ROOT=/users/3149157r/MSc/cfm_mppi/output_dir/socnavgym/benchmark_v1_pilot \
  scripts/run_socnavgym_benchmark_h100.sbatch
```

Inspect it with `squeue`, `sacct`, and the corresponding files under
`cfm_mppi/evaluation/logs/`.  Once the worst-case runtime and memory are
acceptable, freeze the clean commit and submit the complete locked array into
the fresh default formal directory:

```bash
sbatch scripts/run_socnavgym_benchmark_h100.sbatch
```

The script requests one H100, 8 CPU cores, 32 GB RAM, and one hour per task,
with at most five tasks running concurrently.  Task IDs are locked to `0-179`.
The Python command has a 55-minute process deadline so an upstream placement
retry cannot consume an H100 forever.
Pilot shards are deliberately not mixed into formal latency results.

The first task atomically creates `run_contract.json` in its output root.
Every other task checks it before computing and again before publishing, so a
changed commit, implementation, model, Python/Torch/DGL/RVO2 stack, or GPU
contract fails early instead of contaminating the directory.

By default, immutable shards are written below:

```text
output_dir/socnavgym/benchmark_v1/
  run_contract.json
  shards/<scenario-id>/seeds-<seed>.json
  benchmark_index.json
  scenarios/<scenario-id>.json
```

To use another run directory, choose a subdirectory of the repository's
ignored `output_dir/`, export the same absolute directory to every array task,
and pass the same path to aggregation:

```bash
CUSTOM_ROOT=/users/3149157r/MSc/cfm_mppi/output_dir/socnavgym/benchmark_v1_run2
sbatch \
  --export=ALL,SOCNAV_BENCHMARK_OUTPUT_ROOT="$CUSTOM_ROOT" \
  scripts/run_socnavgym_benchmark_h100.sbatch

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --aggregate --output-root "$CUSTOM_ROOT"
```

Valid shards are never overwritten.  Failed/missing task IDs can be resubmitted
with Slurm's `--array` option; valid outputs will be checked and skipped.  A
contract-mismatched existing file is a hard error.  Preserve and move such a
file aside for diagnosis rather than combining it with the current run.

Do not edit, pull, switch branches, or commit in this checkout from submission
until every array task has finished. Later-starting tasks read the live
checkout; changing it either fails the clean-tree guard or creates shards that
the aggregator correctly refuses to mix. A task killed by the 55-minute
process deadline normally appears as a nonzero/failed Slurm task with no final
shard; resubmit that task only after diagnosing its log. Atomic publication
prevents a partial target JSON (an interrupted hidden temporary file may be
left for inspection).

## Aggregate and inspect results

After all 180 tasks finish, run:

```bash
conda activate vrc
python -m cfm_mppi.evaluation.eval_socnavgym_suite --aggregate
```

This validates every shard and writes
`output_dir/socnavgym/benchmark_v1/benchmark_index.json` plus one derived file
per scenario under `scenarios/`. It reports per scenario,
primary/sensitivity group, and overall metrics for both planners.
It refuses an incomplete benchmark.  For a clearly marked progress snapshot:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --aggregate --allow-incomplete
```

The aggregator rejects random-model runs, debug step limits, missing per-step
truth, dirty repositories, config/checkpoint/seed mismatches, and heterogeneous
code, dependency, Git, or GPU contracts across shards. It reconstructs every
episode from its step records and recomputes return, path/clearance, freezing,
event, and latency metrics before accepting them into an aggregate.

## Which files run what

- Formal H100 experiment: `scripts/run_socnavgym_benchmark_h100.sbatch`
- Locked suite runner and aggregator:
  `cfm_mppi/evaluation/eval_socnavgym_suite.py`
- Manifest, seed list, and six scenario configs:
  `configs/socnavgym/benchmark_v1/`
- Ad hoc single-scenario smoke/debug evaluation:
  `cfm_mppi/evaluation/eval_socnavgym.py`
- Shared simulator adapter and evaluation loop:
  `cfm_mppi/evaluation/socnavgym_adapter.py` and
  `cfm_mppi/evaluation/socnavgym_runner.py`
