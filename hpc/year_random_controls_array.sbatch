#!/bin/bash
#SBATCH --job-name=year-random
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%A_%a.out

set -euo pipefail

# Example:
#   sbatch --array=0-19 hpc/year_random_controls_array.sbatch

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_EXE="${PYTHON_EXE:-python}"
REPL_ID=$((SLURM_ARRAY_TASK_ID + 1))

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

RANDOM_SAMPLE_IDS="$REPL_ID" N_RANDOM_SAMPLES=1 $PYTHON_EXE scripts/random_null_datasets.py
