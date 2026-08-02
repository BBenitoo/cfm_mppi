# SocNavGym shared evaluation

The SocNavGym evaluation path compares the constant-velocity CFM+MPPI baseline
and VRC+MPPI without changing the simulator's pedestrian dynamics.  Both
controllers use `SocNavGym-v1` with `WorldFrameObservations` through the same
adapter and closed-loop runner.

## Fairness contract

- Every `(planner, environment seed)` pair gets a newly constructed environment.
- Method execution order alternates across seeds to counterbalance shared
  model/CUDA cache and thermal-order effects. Use at least two seeds for timing
  comparisons; cold first-decision and steady-state latency are reported
  separately.
- Paired planners must receive exactly the same initial robot, goal, human IDs,
  human states, timestep, episode length, and physical control bounds.
- Both planners use the same default budget: 200 CFM candidates, 10 branches,
  and 200 MPPI samples per branch (2,000 refinement rollouts).
- The baseline broadcasts one constant-velocity pedestrian forecast to every
  branch. VRC+MPPI uses one branch-conditioned VRC pedestrian forecast. This is
  their only interaction-prediction difference.
- VRC is planner-internal. Both planners call their branch planner with
  `build_selected_vrc_tube=False`; only physical `[v, omega]` reaches the
  environment.
- The adapter maps physical controls to SocNavGym's normalized
  `[v_normalized, 0, omega_normalized]` action. Formal runs fail if a requested
  control would be clipped.
- The state returned by `env.step()` is the only closed-loop truth. The runner
  never advances a second local robot or pedestrian simulation.
- Python and NumPy environment RNG state is isolated from planner work. Each
  planner has an episode-local Torch RNG initialized from the recorded planner
  seed.
- `terminated` and `truncated` are handled separately. An explicit `--max-steps`
  is a smoke/debug runner limit and is recorded as such.

## Run a paired evaluation

Activate the dedicated environment and configure headless dependencies:

```bash
conda activate vrc
export DGLBACKEND=pytorch
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cfm-mppi-matplotlib"
export SDL_VIDEODRIVER=dummy
mkdir -p "$MPLCONFIGDIR"
```

Place the pretrained model at `output_dir/cfm_transformer/checkpoint.pth`. The
CLI also detects the migrated HPC location
`cfm_mppi/output_dir/cfm_transformer/checkpoint.pth`. Then run both methods with
the same seed schedule:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym \
  --planner both \
  --seeds 0:20 \
  --device cuda \
  --output output_dir/socnavgym/evaluation.json
```

The checkpoint is mandatory for a formal run. `--allow-random-model` exists
only for plumbing smoke tests and marks the output
`random_model_smoke_only=true`; such output is not a performance result.

The JSON output records the config and checkpoint SHA-256 hashes, dependency
versions, device, planner parameters, seeds, per-step environment truth,
planner diagnostics, environment metrics, geometric metrics, and synchronized
planning latency. Use `--summary-only` to omit trajectories. Do not set
`--max-steps` for a full episode.

The current default config, `configs/socnavgym/probe_v1_world.yaml`, remains the
controlled two-ORCA-human integration scenario. The locked six-scenario formal
benchmark, fixed seeds, H100 array entry point, and aggregation workflow are
documented in [`socnavgym_benchmark.md`](socnavgym_benchmark.md).

## Verification

Normal tests use fake environments and do not require SocNavGym. On the pinned
HPC environment, additionally run the real contract test:

```bash
python -m unittest discover -s tests -v
RUN_SOCNAVGYM_CONTRACT=1 python -m unittest \
  tests.test_socnavgym_contract_integration -v
```

For a minimal end-to-end smoke without a checkpoint, reduce all budgets and
limit execution to one step. This checks integration only:

```bash
python -m cfm_mppi.evaluation.eval_socnavgym \
  --planner both --seeds 17 --max-steps 1 --device cuda \
  --allow-random-model --summary-only \
  --horizon 12 --max-history 2 \
  --cfm-candidates 10 --branches 2 --mppi-samples-per-branch 2 \
  --output output_dir/socnavgym/smoke.json
```
