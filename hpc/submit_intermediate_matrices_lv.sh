#!/bin/bash
#
# Submit LV intermediate matrices analysis.
#
# Generates gene x intermediate matrices for LV-phenotype pairs.
#
# Usage:
#   bash hpc/submit_intermediate_matrices_lv.sh
#
# Environment variables:
#   INT_MATRIX_LV_OUTPUT_DIRS - Space-separated list of LV output directories
#   INT_MATRIX_OUTPUT_DIR     - Output directory for matrices
#   INT_MATRIX_DWPC_THRESHOLD - DWPC threshold (default: p75)
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${INT_MATRIX_OUTPUT_DIR:-output/intermediate_matrices}"
LV_OUTPUT_DIRS="${INT_MATRIX_LV_OUTPUT_DIRS:-output/lv_experiment_more_B output/lv_experiment_lv603_se}"
DWPC_THRESHOLD="${INT_MATRIX_DWPC_THRESHOLD:-p75}"
MAX_RANK="${INT_MATRIX_MAX_RANK:-5}"

mkdir -p "$OUTPUT_DIR" hpc/logs

# Build list of (lv_output_dir, lv_id, target_set_id) combinations
LV_LIST="$OUTPUT_DIR/lv_list.txt"
> "$LV_LIST"

for LV_DIR in $LV_OUTPUT_DIRS; do
  if [[ ! -f "$LV_DIR/feature_manifest.csv" ]] || [[ ! -f "$LV_DIR/lv_top_genes.csv" ]]; then
    echo "Warning: Missing required files in $LV_DIR"
    continue
  fi

  python3 -c "
import pandas as pd
# lv_id comes from lv_top_genes.csv, target_set_id from feature_manifest.csv
top_genes = pd.read_csv('$LV_DIR/lv_top_genes.csv')
manifest = pd.read_csv('$LV_DIR/feature_manifest.csv')
lv_ids = top_genes['lv_id'].unique()
target_set_ids = manifest['target_set_id'].unique()
for lv_id in lv_ids:
    for target_set_id in target_set_ids:
        print(f'$LV_DIR\t{lv_id}\t{target_set_id}')
" >> "$LV_LIST"
done

N_LV=$(wc -l < "$LV_LIST")
if [[ "$N_LV" -le 0 ]]; then
  echo "No LV-target pairs found."
  exit 1
fi

echo "Repo root:         $REPO_ROOT"
echo "Output dir:        $OUTPUT_DIR"
echo "LV output dirs:    $LV_OUTPUT_DIRS"
echo "DWPC threshold:    $DWPC_THRESHOLD"
echo "Max rank:          $MAX_RANK"
echo "LV-target pairs:   $N_LV"
echo

# Submit array job
ARRAY_JOB=$(sbatch \
  --parsable \
  --export=ALL,INT_MATRIX_LV_LIST="$LV_LIST",INT_MATRIX_OUTPUT_DIR="$OUTPUT_DIR",INT_MATRIX_DWPC_THRESHOLD="$DWPC_THRESHOLD",INT_MATRIX_MAX_RANK="$MAX_RANK" \
  --array="0-$((N_LV - 1))" \
  hpc/intermediate_matrices_lv_array.sbatch)

echo "Submitted array job: $ARRAY_JOB (${N_LV} tasks)"
echo "Monitor with: squeue -u \$USER"
