# SocNavGym benchmark v2

Benchmark v2 preserves the benchmark-v1 planner, runtime, map, simulator, and
fairness contracts while changing only the crowd-density levels and the formal
seed count. Benchmark v1 and its completed outputs remain unchanged.

## Locked matrix

| Scenario | Report group | Human policy | Dynamic humans |
| --- | --- | --- | ---: |
| `orca_sparse_5` | primary | ORCA | 5 |
| `orca_medium_10` | primary | ORCA | 10 |
| `orca_dense_15` | primary | ORCA | 15 |
| `sfm_sparse_5` | sensitivity | SFM | 5 |
| `sfm_medium_10` | sensitivity | SFM | 10 |
| `sfm_dense_15` | sensitivity | SFM | 15 |

The formal environment seeds are the frozen consecutive list `1000..1049`:

- 6 scenarios x 50 seeds = 300 matched scenario-seed pairs;
- 2 planners per pair = 600 full episodes;
- 1 paired result per H100 task = 300 Slurm array tasks;
- each scenario has 25 CFM-first and 25 VRC-first tasks.

All other settings remain identical to benchmark v1: 10 m x 10 m open map,
0.1 s timestep, 256-step episode limit, no static objects or interaction
groups, the `geometric-only-v1` human-goal protocol, horizon 80, history 10,
200 CFM candidates, 10 branches, 200 MPPI samples per branch, the same locked
checkpoint, and the same H100 runtime contract.

## Validate and run

```bash
conda activate vrc
export DGLBACKEND=pytorch

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v2/suite.json \
  --list-jobs

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v2/suite.json \
  --validate-configs

sbatch scripts/run_socnavgym_benchmark_v2_h100.sbatch
```

For a six-scenario pilot that also alternates execution order, use:

```bash
sbatch \
  --array=0,51,100,151,200,251%5 \
  --export=ALL,SOCNAV_BENCHMARK_OUTPUT_ROOT=/users/3149157r/MSc/cfm_mppi/output_dir/socnavgym/benchmark_v2_pilot \
  scripts/run_socnavgym_benchmark_v2_h100.sbatch
```

Aggregate only after all 300 tasks complete:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v2/suite.json \
  --aggregate \
  --output-root output_dir/socnavgym/benchmark_v2
```
