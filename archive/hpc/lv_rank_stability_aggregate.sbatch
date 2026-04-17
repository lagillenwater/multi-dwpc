#!/bin/bash
#SBATCH --job-name=lv-rank-stab-agg
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# Note: --output is set by submit script to organize logs by job name

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
# Path to LV experiment output containing replicate_manifest.csv and summaries
# Set LV_OUTPUT_DIR to override (should contain 100 permutation + random replicates)
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

RANK_METRIC="${LV_RANK_STAB_RANK_METRIC:-effect_size_d}"

$PYTHON_EXE scripts/lv_rank_stability_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --b-values "${LV_RANK_STAB_B_VALUES:-2,4,6,8,10,20}" \
  --seeds "${LV_RANK_STAB_SEEDS:-11,22,33,44,55}" \
  --top-k-metapaths "${LV_RANK_STAB_TOP_K:-5,10}" \
  --rank-metric "$RANK_METRIC"
