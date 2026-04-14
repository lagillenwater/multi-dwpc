#!/bin/bash
#SBATCH --job-name=lv-random
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/lv/%x_%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"
REPL_ID=$((SLURM_ARRAY_TASK_ID + 1))

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

$PYTHON_EXE scripts/lv_generate_control_replicates.py \
  --output-dir "$OUTPUT_DIR" \
  --control random \
  --replicate-id "$REPL_ID"
