# SocNavGym HPC contract probe

This repository does not make SocNavGym a mandatory dependency.  The selected
candidate is pinned in `configs/socnavgym/candidates.json` and must pass the
strict probe on the actual HPC software stack before simulator integration is
enabled.

## Selected upstream contract

- SocNavGym commit: `1ef13ee604b71730e9ec7f2d9fd9cb2e8b796549`
- Environment: `SocNavGym-v1`
- Wrapper: official `WorldFrameObservations`
- Gymnasium: `0.29.1`
- Controlled config: `configs/socnavgym/probe_v1_world.yaml`

The inspected current-main candidate
`95fbc9cebe4357201145a7d0a28c7736c0efe6cf` is recorded as rejected.  At that
revision the wrapper imports the deleted `socnavenv_v1` module and still uses
the v1 observation layout while the environment exports v2.

## HPC installation outline

Create a separate Python 3.11 environment and load the cluster's compiler,
CMake, CUDA, and Torch modules (or install the site-recommended Torch wheel)
first.  Confirm that Torch imports before installing the common probe
dependencies.  Install this repository without dependency resolution so its
unconstrained `torch` requirement cannot replace the cluster's ABI-matched
build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -r requirements/socnavgym-hpc.txt
python -m pip install -e . --no-deps
```

Install a DGL build compatible with the environment's exact Torch and CUDA ABI.
Also install a Python-RVO2 build which exposes the module `rvo2` and the class
`PyRVOSimulator`.  SocNavGym imports both DGL and RVO2 unconditionally, even
when the DSRNN reward is selected.  The historical RVO2 source recommended by
SocNavGym is:

```text
https://github.com/sybrenstuvel/Python-RVO2.git
commit c2c46ba8d59556aa10faf03479293236efea154d
```

That binding predates Python 3.11, so it is a build candidate rather than an
assumed dependency.  Prefer a cluster-provided compatible wheel when one is
available; the probe performs a native two-agent simulation and fails if the
installed binding is not operational.

Finally install the selected SocNavGym source without allowing its incomplete
dependency list to replace the cluster-specific packages:

```bash
python -m pip install --no-deps \
  "socnavgym @ git+https://github.com/gnns4hri/SocNavGym.git@1ef13ee604b71730e9ec7f2d9fd9cb2e8b796549"
```

## Run the acceptance probe

```bash
python scripts/probe_socnavgym_hpc.py \
  --candidate v1-1ef13ee \
  --seed 17 \
  --steps 256 \
  --timeout-seconds 180 \
  --output socnavgym_probe_v1.json
```

The probe does not render.  It checks the installed commit using PEP 610
metadata (or a nearby Git checkout), runs native RVO2, validates Gym reset/step
and spaces, compares wrapper world coordinates with `env.unwrapped`, replays a
fixed seed twice, and performs a headless rollout.

Stable exit codes are:

| Code | Meaning |
| ---: | --- |
| 0 | all contracts passed |
| 2 | candidate manifest or config error |
| 10 | installed SocNavGym revision/version mismatch or unverifiable |
| 11 | dependency import failed |
| 12 | native RVO2 smoke test failed |
| 20 | Gym registration mismatch or base environment creation failed |
| 21 | `WorldFrameObservations` import/construction failed |
| 22 | reset/step/action/space contract failed |
| 23 | world-frame shape/value contract failed |
| 24 | fixed-seed replay differed |
| 25 | headless rollout or close failed |
| 30 | probe timed out |
| 70 | unexpected internal failure |

The file passed with `--output` is the authoritative machine-readable JSON
report.  The same report is printed to stdout for convenience, but native
dependencies such as DGL may print their own initialization messages there.
The report records the platform, dependency versions, exact commit, every check
result, exception traceback, and total outcome.  A missing dependency is a
failure on HPC, not a skip.  An output-file creation or write failure returns
exit code 70 and leaves the failure report on stdout.

## Optional integration test

Normal local test discovery skips the real SocNavGym check.  On the configured
HPC node, explicitly enable it:

```bash
RUN_SOCNAVGYM_CONTRACT=1 python -m unittest \
  tests.test_socnavgym_contract_integration -v
```
