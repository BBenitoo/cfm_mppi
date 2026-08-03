# SocNavGym benchmark v3

Benchmark v3 preserves the benchmark-v2 planner, runtime, map, simulator, and
fairness contracts while fixing every scenario at 10 dynamic humans and
increasing the formal seed count to 300. Earlier benchmark suites and their
completed outputs remain unchanged.

## Locked matrix

| Scenario | Report group | Human policy | Dynamic humans |
| --- | --- | --- | ---: |
| `orca_fixed_10` | primary | ORCA | 10 |
| `sfm_fixed_10` | sensitivity | SFM | 10 |

The formal environment seeds are the frozen consecutive list `1000..1299`:

- 2 scenarios x 300 seeds = 600 matched scenario-seed pairs;
- 2 planners per pair = 1,200 full episodes;
- 1 paired result per H100 task = 600 Slurm array tasks;
- each scenario has 150 CFM-first and 150 VRC-first tasks.

All other settings remain identical to benchmark v2: 10 m x 10 m open map,
0.1 s timestep, 256-step episode limit, no static objects or interaction
groups, the `geometric-only-v1` human-goal protocol, horizon 80, history 10,
200 CFM candidates, 10 branches, 200 MPPI samples per branch, the same locked
checkpoint, and the same H100 runtime contract.

## Validate and run

```bash
conda activate vrc
export DGLBACKEND=pytorch

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v3/suite.json \
  --list-jobs

python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v3/suite.json \
  --validate-configs

sbatch scripts/run_socnavgym_benchmark_v3_h100.sbatch
```

For a two-scenario pilot that covers both execution orders, use:

```bash
sbatch \
  --array=0,1,300,301%5 \
  --export=ALL,SOCNAV_BENCHMARK_OUTPUT_ROOT=/users/3149157r/MSc/cfm_mppi/output_dir/socnavgym/benchmark_v3_pilot \
  scripts/run_socnavgym_benchmark_v3_h100.sbatch
```

Aggregate only after all 600 tasks complete:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite configs/socnavgym/benchmark_v3/suite.json \
  --aggregate \
  --output-root output_dir/socnavgym/benchmark_v3
```
