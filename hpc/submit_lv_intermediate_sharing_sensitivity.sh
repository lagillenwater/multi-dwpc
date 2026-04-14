#!/bin/bash
#
# Submit LV intermediate sharing sensitivity analysis.
#
# Tests multiple path_top_k values to assess stability of intermediate sharing metrics.
#
# Usage:
#   bash hpc/submit_lv_intermediate_sharing_sensitivity.sh
#
# Environment variables:
#   LV_INT_SHARE_OUTPUT_DIRS - Space-separated list of LV experiment output dirs
#   LV_INT_SHARE_OUTPUT_DIR  - Output directory for results
#   LV_INT_SHARE_B           - B value for metapath selection (default: 10)
#   LV_INT_SHARE_K_VALUES    - Comma-separated path_top_k values (default: 10,50,100,500,1000)
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${LV_INT_SHARE_OUTPUT_DIR:-output/lv_intermediate_sharing_sensitivity}"
LV_OUTPUT_DIRS="${LV_INT_SHARE_OUTPUT_DIRS:-output/lv_single_target_refactor}"
B_VALUE="${LV_INT_SHARE_B:-10}"
DWPC_THRESHOLD="${LV_INT_SHARE_DWPC_THRESHOLD:-p75}"
K_VALUES="${LV_INT_SHARE_K_VALUES:-10,50,100,500,1000}"

# Create log directory for this job type
LOG_DIR="hpc/logs/lv/lv-int-share-sensitivity"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Repo root:        $REPO_ROOT"
echo "Output dir:       $OUTPUT_DIR"
echo "LV output dirs:   $LV_OUTPUT_DIRS"
echo "B value:          $B_VALUE"
echo "DWPC threshold:   $DWPC_THRESHOLD"
echo "K values:         $K_VALUES"
echo

# Single job - runs all k values sequentially
JOB_CMD="cd \"$REPO_ROOT\" && module load anaconda && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate multi_dwpc && python3 scripts/lv_intermediate_sharing_sensitivity.py --lv-output-dirs $LV_OUTPUT_DIRS --b $B_VALUE --dwpc-threshold $DWPC_THRESHOLD --path-top-k-values $K_VALUES --output-dir \"$OUTPUT_DIR\""

JOB_ID=$(sbatch \
  --parsable \
  --export=ALL \
  --job-name="lv-int-share-sens" \
  --partition=amilan \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=32G \
  --time=08:00:00 \
  --output="$LOG_DIR/%j.out" \
  --wrap="bash -c '$JOB_CMD'")

echo "Submitted job: $JOB_ID"
echo "Monitor with: tail -f $LOG_DIR/${JOB_ID}.out"
