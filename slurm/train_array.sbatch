#!/bin/bash
#SBATCH --job-name=niryo-lift
#SBATCH --array=0-9
#SBATCH --partition=AMD                   
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00                     
#SBATCH --output=logs/%x_%A_%a.out          

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
