#!/bin/bash
#SBATCH --job-name=random-path-l5
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON_EXE="${PYTHON_EXE:-python}"

OUTPUT_DIR="${RANDOM_PATH_OUTPUT_DIR:-output/random_path_count_sampling_len5}"
DATA_DIR="${RANDOM_PATH_DATA_DIR:-data}"
NUM_PAIRS="${RANDOM_PATH_NUM_PAIRS:-20}"
MIN_LENGTH="${RANDOM_PATH_MIN_LENGTH:-2}"
MAX_LENGTH="${RANDOM_PATH_MAX_LENGTH:-5}"
SEED="${RANDOM_PATH_SEED:-11}"
DEGREE_CAP="${RANDOM_PATH_DEGREE_CAP:-50}"
WORKERS="${RANDOM_PATH_WORKERS:-$SLURM_CPUS_PER_TASK}"

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

$PYTHON_EXE scripts/plot_random_path_counts.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-pairs "$NUM_PAIRS" \
  --min-length "$MIN_LENGTH" \
  --max-length "$MAX_LENGTH" \
  --seed "$SEED" \
  --degree-percentile-cap "$DEGREE_CAP" \
  --workers "$WORKERS"
