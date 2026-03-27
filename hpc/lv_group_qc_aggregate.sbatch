#!/bin/bash
#SBATCH --job-name=lv-group-qc
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
ANALYSIS_DIR="${LV_RANK_STAB_ANALYSIS_DIR:-$OUTPUT_DIR/lv_rank_stability_experiment}"
QC_OUTPUT_DIR="${LV_GROUP_QC_OUTPUT_DIR:-$OUTPUT_DIR/lv_group_qc_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

$PYTHON_EXE scripts/lv_group_qc_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --analysis-dir "$ANALYSIS_DIR" \
  --qc-output-dir "$QC_OUTPUT_DIR" \
  --gap-b "${LV_GROUP_QC_GAP_B:-5}" \
  --tier1-b "${LV_GROUP_QC_TIER1_B:-5}" \
  --tier2-b "${LV_GROUP_QC_TIER2_B:-10}" \
  --random-promiscuity-tolerance "${LV_RANDOM_PROM_TOL:-2}" \
  --rho-threshold "${LV_GROUP_QC_RHO_THRESHOLD:-0.90}" \
  --top5-threshold "${LV_GROUP_QC_TOP5_THRESHOLD:-0.80}" \
  --top10-threshold "${LV_GROUP_QC_TOP10_THRESHOLD:-0.85}" \
  --random-mae-threshold "${LV_GROUP_QC_RANDOM_MAE_THRESHOLD:-1.0}" \
  --random-within-tol-threshold "${LV_GROUP_QC_RANDOM_WITHIN_TOL_THRESHOLD:-0.90}" \
  --perm-overlap-threshold "${LV_GROUP_QC_PERM_OVERLAP_THRESHOLD:-0.70}"
