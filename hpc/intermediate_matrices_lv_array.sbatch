#!/bin/bash
#SBATCH --job-name=int-matrix-lv
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/int-matrix-lv_%A_%a.out

set -euo pipefail

# REPO_ROOT should be passed via --export, fallback to detection
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

LV_LIST="${INT_MATRIX_LV_LIST:-$REPO_ROOT/output/intermediate_matrices/lv_list.txt}"
OUTPUT_DIR="${INT_MATRIX_OUTPUT_DIR:-$REPO_ROOT/output/intermediate_matrices}"
DWPC_THRESHOLD="${INT_MATRIX_DWPC_THRESHOLD:-p75}"
MAX_RANK="${INT_MATRIX_MAX_RANK:-5}"

# Get LV info for this array task (tab-separated: lv_output_dir, lv_id, target_set_id)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LV_LIST")

if [[ -z "$LINE" ]]; then
  echo "No LV at index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

LV_OUTPUT_DIR_REL=$(echo "$LINE" | cut -f1)
LV_ID=$(echo "$LINE" | cut -f2)
TARGET_SET_ID=$(echo "$LINE" | cut -f3)

# Make LV output dir absolute
LV_OUTPUT_DIR="$REPO_ROOT/$LV_OUTPUT_DIR_REL"

echo "Task $SLURM_ARRAY_TASK_ID: $LV_ID -> $TARGET_SET_ID"
echo "LV output dir: $LV_OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "DWPC threshold: $DWPC_THRESHOLD"

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

python3 scripts/intermediate_matrices.py \
  --mode lv \
  --lv-id "$LV_ID" \
  --target-set-id "$TARGET_SET_ID" \
  --lv-output-dir "$LV_OUTPUT_DIR" \
  --max-metapath-rank "$MAX_RANK" \
  --dwpc-threshold "$DWPC_THRESHOLD" \
  --output-dir "$OUTPUT_DIR"

echo "Done: $LV_ID -> $TARGET_SET_ID"
