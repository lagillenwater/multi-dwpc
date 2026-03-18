#!/bin/bash
#SBATCH --job-name=lv-rank-stab-agg
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

$PYTHON_EXE scripts/lv_rank_stability_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --b-values "${LV_RANK_STAB_B_VALUES:-1,2,5,10,20}" \
  --seeds "${LV_RANK_STAB_SEEDS:-11,22,33,44,55}" \
  --top-k-metapaths "${LV_RANK_STAB_TOP_K:-10}"
