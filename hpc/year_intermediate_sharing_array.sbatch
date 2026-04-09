#!/bin/bash
#SBATCH --job-name=year-int-share
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/year-int-share_%A_%a.out
#SBATCH --qos=normal

set -euo pipefail

module load anaconda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

GO_LIST="${YEAR_INT_SHARE_GO_LIST:-output/year_intermediate_sharing/go_list.txt}"
OUTPUT_DIR="${YEAR_INT_SHARE_OUTPUT_DIR:-output/year_intermediate_sharing}"
SUPPORT_PATH="${YEAR_INT_SHARE_SUPPORT:-output/year_direct_go_term_support_b5.csv}"
ADDED_PATH="${YEAR_INT_SHARE_ADDED:-output/intermediate/upd_go_bp_2024_added.csv}"
DIRECT_DIR="${YEAR_INT_SHARE_DIRECT_DIR:-output/dwpc_direct/all_GO_positive_growth/results}"
MAX_RANK="${YEAR_INT_SHARE_MAX_RANK:-5}"

GO_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GO_LIST")

if [[ -z "$GO_ID" ]]; then
  echo "No GO term at index $SLURM_ARRAY_TASK_ID in $GO_LIST"
  exit 0
fi

echo "Processing GO term: $GO_ID (array index $SLURM_ARRAY_TASK_ID)"

python3 scripts/year_intermediate_sharing_2016_2024.py \
  --direct-results-dir "$DIRECT_DIR" \
  --support-path "$SUPPORT_PATH" \
  --added-pairs-path "$ADDED_PATH" \
  --max-metapath-rank "$MAX_RANK" \
  --go-id "$GO_ID" \
  --output-dir "$OUTPUT_DIR"

echo "Done: $GO_ID"
