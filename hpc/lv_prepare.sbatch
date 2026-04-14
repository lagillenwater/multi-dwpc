#!/bin/bash
#SBATCH --job-name=lv-prepare
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/lv/%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"
LV_LOADINGS_PATH="${LV_LOADINGS_PATH:-data/lv_loadings/multiplier_model_z.tsv.gz}"
LV_LVS="${LV_LVS:-LV603,LV246,LV57}"

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

$PYTHON_EXE scripts/lv_prepare_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --lv-loadings "$LV_LOADINGS_PATH" \
  --lvs "$LV_LVS" \
  --metapath-limit-per-target "${LV_METAPATH_LIMIT:-2}" \
  ${LV_ALL_METAPATHS:+--all-metapaths} \
  --prepare-metadata-only \
  --resume
