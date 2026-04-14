#!/bin/bash
#SBATCH --job-name=lv-group-qc
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
# Note: --output is set by submit script to organize logs by job name

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
ANALYSIS_DIR="${LV_RANK_STAB_ANALYSIS_DIR:-$OUTPUT_DIR/lv_rank_stability_experiment}"
QC_OUTPUT_DIR="${LV_GROUP_QC_OUTPUT_DIR:-$OUTPUT_DIR/lv_group_qc_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

$PYTHON_EXE scripts/lv_group_qc_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --analysis-dir "$ANALYSIS_DIR" \
  --qc-output-dir "$QC_OUTPUT_DIR" \
  --gap-b "${LV_GROUP_QC_GAP_B:-5}" \
  --score-gap-max-k "${LV_GROUP_QC_SCORE_GAP_MAX_K:-20}" \
  --diagnostic-b "${LV_GROUP_QC_DIAGNOSTIC_B:-5}" \
  --random-promiscuity-tolerance "${LV_RANDOM_PROM_TOL:-2}" \
  --near-tie-threshold "${LV_GROUP_QC_NEAR_TIE_THRESHOLD:-0.10}"
