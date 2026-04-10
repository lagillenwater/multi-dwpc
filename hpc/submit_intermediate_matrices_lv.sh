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
# Use absolute path so sbatch jobs can find it
LV_LIST="$REPO_ROOT/$OUTPUT_DIR/lv_list.txt"
> "$LV_LIST"

for LV_DIR in $LV_OUTPUT_DIRS; do
  if [[ ! -f "$LV_DIR/feature_manifest.csv" ]] || [[ ! -f "$LV_DIR/lv_top_genes.csv" ]]; then
    echo "Warning: Missing required files in $LV_DIR"
    continue
  fi

  python3 -c "
import pandas as pd
from pathlib import Path

lv_dir = Path('$LV_DIR')

# Try lv_target_map.csv first (best source of LV-target pairs)
target_map_path = lv_dir / 'lv_target_map.csv'
if target_map_path.exists():
    pairs = pd.read_csv(target_map_path)
    for _, row in pairs.iterrows():
        print(f'$LV_DIR\t{row[\"lv_id\"]}\t{row[\"target_set_id\"]}')
else:
    # Fallback to lv_metapath_results.csv
    results_path = lv_dir / 'lv_metapath_results.csv'
    if results_path.exists():
        results = pd.read_csv(results_path)
        pairs = results[['lv_id', 'target_set_id']].drop_duplicates()
        for _, row in pairs.iterrows():
            print(f'$LV_DIR\t{row[\"lv_id\"]}\t{row[\"target_set_id\"]}')
    else:
        print(f'Warning: No lv_target_map.csv or lv_metapath_results.csv in $LV_DIR', file=__import__('sys').stderr)
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

# Submit array job (use absolute paths)
ARRAY_JOB=$(sbatch \
  --parsable \
  --export=ALL,REPO_ROOT="$REPO_ROOT",INT_MATRIX_LV_LIST="$LV_LIST",INT_MATRIX_OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR",INT_MATRIX_DWPC_THRESHOLD="$DWPC_THRESHOLD",INT_MATRIX_MAX_RANK="$MAX_RANK" \
  --array="0-$((N_LV - 1))" \
  hpc/intermediate_matrices_lv_array.sbatch)

echo "Submitted array job: $ARRAY_JOB (${N_LV} tasks)"
echo "Monitor with: squeue -u \$USER"
