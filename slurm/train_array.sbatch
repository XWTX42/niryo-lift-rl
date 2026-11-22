#!/bin/bash
# Run every (config, seed) pair as one task of a Slurm job array on the LMU IfI student pool.
# Cluster notes: https://www.rz.ifi.lmu.de/infos/slurm_de.html
#   - jobs run on the student workstation machines; a job is paused while someone is logged in
#     on that machine, so submitting in the evening is best
#   - at most 15 running / 30 submitted jobs per user; shorter jobs get higher priority
#   - CPU only: our networks are small, no GPU (and no abakus access) needed
#
# Submit from the project root (niryo-lift-rl/). The logs/ directory must exist first:
#     mkdir -p logs
#     sbatch --array=0 slurm/train_array.sbatch     # test with ONE task first
#     sbatch slurm/train_array.sbatch               # then all 10 tasks
# 2 configs x 5 seeds = 10 tasks -> --array=0-9

#SBATCH --job-name=niryo-lift
#SBATCH --array=0-9
##SBATCH --partition=NAME                   # leave commented to use the default partition (see `sinfo`)
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00                     # ~30 min on a laptop; short limit = higher priority
#SBATCH --output=logs/%x_%A_%a.out          # %x job name, %A array job id, %a task index

set -euo pipefail

CONFIGS=(
    configs/niryo_lift_fixed_sac_her.yaml
    configs/niryo_lift_fixed_sac_her_demos.yaml
)
SEEDS=(0 1 2 3 4)

# Task index -> (config, seed): tasks 0-4 = first config, 5-9 = second config.
CONFIG=${CONFIGS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}

# Use exactly the CPUs Slurm gave us.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date): $CONFIG seed $SEED"
# The environment was created beforehand with `uv sync`; call its Python directly so the job
# does not try to download anything.
.venv/bin/python scripts/train.py "$CONFIG" --seed "$SEED"
echo "finished at $(date)"
