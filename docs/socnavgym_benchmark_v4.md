# SocNavGym benchmark v4

Benchmark v4 preserves the benchmark-v3 planner, simulator, policy, runtime,
and fairness contracts while enlarging the open map, fixing the robot route,
and evaluating three crowd densities.

## Locked matrix

| Scenario | Report group | Human policy | Dynamic humans |
| --- | --- | --- | ---: |
| `orca_sparse_10` | primary | ORCA | 10 |
| `orca_medium_15` | primary | ORCA | 15 |
| `orca_dense_20` | primary | ORCA | 20 |
| `sfm_sparse_10` | sensitivity | SFM | 10 |
| `sfm_medium_15` | sensitivity | SFM | 15 |
| `sfm_dense_20` | sensitivity | SFM | 20 |

Every scenario uses a 20 m x 20 m open map. The robot starts at
`(8 m, 8 m)` and has the fixed goal `(-8 m, -8 m)`. Its initial heading
remains seeded and random, as in earlier suites. Pedestrian positions,
headings, speeds, and goals also remain seeded and random. During reset the
fixed robot start and goal regions are reserved so a pedestrian cannot spawn
on either location; the reservation is removed before simulation begins.

The formal environment seeds are the frozen consecutive list `1000..1099`:

- 6 scenarios x 100 seeds = 600 matched scenario-seed pairs;
- 2 planners per pair = 1,200 full episodes;
- 1 paired result per H100 task = 600 Slurm array tasks;
- each scenario has 50 CFM-first and 50 VRC-first tasks.

All other settings remain identical to benchmark v3: 0.1 s timestep,
256-step episode limit, no static objects, walls, or interaction groups, the
`geometric-only-v1` human-goal protocol, horizon 80, history 10, 200 CFM
candidates, 10 branches, 200 MPPI samples per branch, the same locked
checkpoint, and the same H100 runtime contract.

## Validate and run

```bash
conda activate vrc
export DGLBACKEND=pytorch

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v4/suite.json \
  --list-jobs

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v4/suite.json \
  --validate-configs

sbatch scripts/run_socnavgym_benchmark_v4_h100.sbatch
```

For a six-scenario pilot that alternates execution order, use:

```bash
sbatch \
  --array=0,101,200,301,400,501%5 \
  --export=ALL,SOCNAV_BENCHMARK_OUTPUT_ROOT=/users/3149157r/MSc/cfm_mppi/output_dir/socnavgym/benchmark_v4_pilot \
  scripts/run_socnavgym_benchmark_v4_h100.sbatch
```

Aggregate only after all 600 tasks complete:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v4/suite.json \
  --aggregate \
  --output-root output_dir/socnavgym/benchmark_v4
```
