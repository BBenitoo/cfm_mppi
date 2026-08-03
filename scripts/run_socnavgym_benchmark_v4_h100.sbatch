#!/bin/bash -l

#SBATCH --job-name=socnav_benchmark_v4
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-599%5
#SBATCH --output=/users/3149157r/MSc/cfm_mppi/cfm_mppi/evaluation/logs/%x-%A_%a.out
#SBATCH --error=/users/3149157r/MSc/cfm_mppi/cfm_mppi/evaluation/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/users/3149157r/MSc/cfm_mppi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vrc
cd "$PROJECT_ROOT"

export DGLBACKEND=pytorch
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cfm-mppi-matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export SDL_AUDIODRIVER=dummy
export SDL_VIDEODRIVER=dummy
mkdir -p "$MPLCONFIGDIR"

hostname
python --version
command -v python
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
git rev-parse HEAD
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

BENCHMARK_SUITE="$PROJECT_ROOT/configs/socnavgym/benchmark_v4/suite.json"
BENCHMARK_OUTPUT_ROOT="${SOCNAV_BENCHMARK_OUTPUT_ROOT:-$PROJECT_ROOT/output_dir/socnavgym/benchmark_v4}"

timeout --signal=TERM --kill-after=30s 55m \
  python -m cfm_mppi.evaluation.eval_socnavgym_suite \
  --suite "$BENCHMARK_SUITE" \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$BENCHMARK_OUTPUT_ROOT"
