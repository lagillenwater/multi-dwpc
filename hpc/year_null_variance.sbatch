#!/bin/bash
#SBATCH --job-name=year-null-var
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="${YEAR_NULL_VAR_RESULTS_DIR:-output/dwpc_direct/all_GO_positive_growth/results}"
OUTPUT_DIR="${YEAR_NULL_VAR_OUTPUT_DIR:-output/year_null_variance_exp}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

$PYTHON_EXE scripts/year_null_variance_experiment.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR"
