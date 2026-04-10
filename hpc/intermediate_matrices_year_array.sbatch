#!/bin/bash
#SBATCH --job-name=int-matrix-year
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/int-matrix-year_%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

GO_LIST="${INT_MATRIX_GO_LIST:-output/intermediate_matrices/go_list_year.txt}"
OUTPUT_DIR="${INT_MATRIX_OUTPUT_DIR:-output/intermediate_matrices}"
SUPPORT_PATH="${INT_MATRIX_SUPPORT:-output/year_direct_go_term_support_b5.csv}"
DIRECT_RESULTS_DIR="${INT_MATRIX_DIRECT_RESULTS:-output/dwpc_direct/all_GO_positive_growth/results}"
MAX_RANK="${INT_MATRIX_MAX_RANK:-5}"
DWPC_THRESHOLD="${INT_MATRIX_DWPC_THRESHOLD:-p75}"
SELECTION_COL="${INT_MATRIX_SELECTION_COL:-selected_by_effective_n_all}"

# Get GO term for this array task
GO_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GO_LIST")

if [[ -z "$GO_ID" ]]; then
  echo "No GO term at index $SLURM_ARRAY_TASK_ID"
  exit 0
fi

echo "Task $SLURM_ARRAY_TASK_ID: GO term $GO_ID"
echo "Output dir: $OUTPUT_DIR"
echo "DWPC threshold: $DWPC_THRESHOLD"

module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

python3 scripts/intermediate_matrices.py \
  --mode year \
  --go-id "$GO_ID" \
  --support-path "$SUPPORT_PATH" \
  --direct-results-dir "$DIRECT_RESULTS_DIR" \
  --selection-col "$SELECTION_COL" \
  --max-metapath-rank "$MAX_RANK" \
  --dwpc-threshold "$DWPC_THRESHOLD" \
  --output-dir "$OUTPUT_DIR"

echo "Done: $GO_ID"
