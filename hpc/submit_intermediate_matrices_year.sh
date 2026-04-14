#!/bin/bash
#
# Submit year intermediate matrices analysis as an array job.
#
# Generates gene x intermediate matrices for selected GO terms.
#
# Usage:
#   bash hpc/submit_intermediate_matrices_year.sh
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${INT_MATRIX_OUTPUT_DIR:-output/intermediate_matrices}"
SUPPORT_PATH="${INT_MATRIX_SUPPORT:-output/year_direct_go_term_support_b5.csv}"
DIRECT_RESULTS_DIR="${INT_MATRIX_DIRECT_RESULTS:-output/dwpc_direct/all_GO_positive_growth/results}"
SELECTION_COL="${INT_MATRIX_SELECTION_COL:-selected_by_effective_n_all}"
MAX_RANK="${INT_MATRIX_MAX_RANK:-5}"
DWPC_THRESHOLD="${INT_MATRIX_DWPC_THRESHOLD:-p75}"

# Create log directory for this job type
LOG_DIR="hpc/logs/year/int-matrix"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Extract unique GO terms with selected metapaths
# Use absolute path so sbatch jobs can find it
GO_LIST="$REPO_ROOT/$OUTPUT_DIR/go_list_year.txt"
python3 -c "
import pandas as pd

support = pd.read_csv('$SUPPORT_PATH')
sel_col = '$SELECTION_COL'
support[sel_col] = support[sel_col].astype(str).str.strip().str.lower().isin({'1','true','t','yes'})

# Get GO terms with selected metapaths in 2024
selected = support[(support['year'] == 2024) & support[sel_col]]
go_ids = sorted(selected['go_id'].unique())

with open('$GO_LIST', 'w') as f:
    for g in go_ids:
        f.write(g + '\n')
print(f'Wrote {len(go_ids)} GO terms to $GO_LIST')
"

N_GO=$(wc -l < "$GO_LIST")
if [[ "$N_GO" -le 0 ]]; then
  echo "No GO terms found. Check support table and selection column."
  exit 1
fi

echo "Repo root:         $REPO_ROOT"
echo "Output dir:        $OUTPUT_DIR"
echo "Support path:      $SUPPORT_PATH"
echo "Direct results:    $DIRECT_RESULTS_DIR"
echo "Max rank:          $MAX_RANK"
echo "DWPC threshold:    $DWPC_THRESHOLD"
echo "GO terms:          $N_GO"
echo

# Submit array job (use absolute paths)
ARRAY_JOB=$(sbatch \
  --parsable \
  --export=ALL,REPO_ROOT="$REPO_ROOT",INT_MATRIX_GO_LIST="$GO_LIST",INT_MATRIX_OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR",INT_MATRIX_SUPPORT="$REPO_ROOT/$SUPPORT_PATH",INT_MATRIX_DIRECT_RESULTS="$REPO_ROOT/$DIRECT_RESULTS_DIR",INT_MATRIX_MAX_RANK="$MAX_RANK",INT_MATRIX_DWPC_THRESHOLD="$DWPC_THRESHOLD",INT_MATRIX_SELECTION_COL="$SELECTION_COL" \
  --array="0-$((N_GO - 1))%50" \
  --output="$LOG_DIR/%A/%a.out" \
  hpc/intermediate_matrices_year_array.sbatch)

# Create array job log subdirectory
mkdir -p "$LOG_DIR/$ARRAY_JOB"

echo "Submitted array job: $ARRAY_JOB (${N_GO} tasks)"
echo "Array logs: $LOG_DIR/$ARRAY_JOB/"
echo "Monitor with: squeue -u \$USER"
