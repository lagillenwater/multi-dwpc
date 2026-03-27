#!/bin/bash
#SBATCH --job-name=lv-precompute-final
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

$PYTHON_EXE scripts/lv_prepare_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --metapath-limit-per-target "${LV_METAPATH_LIMIT:-2}" \
  ${LV_ALL_METAPATHS:+--all-metapaths} \
  --finalize-precompute \
  --resume
