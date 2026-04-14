#!/bin/bash
#SBATCH --job-name=year-dwpc-warm
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/year/%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${YEAR_DWPC_OUTPUT_DIR:-output/dwpc_direct/all_GO_positive_growth}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

mkdir -p "$OUTPUT_DIR"

$PYTHON_EXE scripts/compute_dwpc_direct.py \
  --output-dir "$OUTPUT_DIR" \
  --warmup-cache \
  --n-workers "${YEAR_DWPC_WARMUP_N_WORKERS:-2}"
