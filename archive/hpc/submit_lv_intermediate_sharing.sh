#!/bin/bash
#
# Submit LV intermediate sharing analysis.
#
# Computes % of genes sharing intermediates for metapaths with
# effect size d > threshold (per web_tool_discussion.md).
#
# Usage:
#   bash hpc/submit_lv_intermediate_sharing.sh
#
# Environment variables:
#   LV_INT_SHARE_OUTPUT_DIRS      - Space-separated list of LV experiment output dirs
#   LV_INT_SHARE_OUTPUT_DIR       - Output directory for results
#   LV_INT_SHARE_B                - B value for effect size computation (default: 10)
#   LV_INT_SHARE_EFFECT_THRESHOLD - Min effect size d for metapath selection (default: 0.2)
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${LV_INT_SHARE_OUTPUT_DIR:-output/lv_intermediate_sharing}"
LV_OUTPUT_DIRS="${LV_INT_SHARE_OUTPUT_DIRS:-output/lv_single_target_refactor}"
B_VALUE="${LV_INT_SHARE_B:-10}"
EFFECT_THRESHOLD="${LV_INT_SHARE_EFFECT_THRESHOLD:-0.2}"

# Create log directory for this job type
LOG_DIR="hpc/logs/lv/lv-int-share"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Repo root:            $REPO_ROOT"
echo "Output dir:           $OUTPUT_DIR"
echo "LV output dirs:       $LV_OUTPUT_DIRS"
echo "B value:              $B_VALUE"
echo "Effect size threshold: $EFFECT_THRESHOLD (Cohen's d: large>=0.8, medium>=0.5, small>=0.2)"
echo

# Single job (small analysis, only 2-3 LVs)
# Runs intermediate sharing analysis followed by visualization
JOB_CMD="cd \"$REPO_ROOT\" && module load anaconda && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate multi_dwpc && python3 scripts/lv_intermediate_sharing.py --lv-output-dirs $LV_OUTPUT_DIRS --b $B_VALUE --effect-size-threshold $EFFECT_THRESHOLD --output-dir \"$OUTPUT_DIR\" && python3 scripts/plot_lv_intermediate_sharing.py --input-dir \"$OUTPUT_DIR\""

JOB_ID=$(sbatch \
  --parsable \
  --export=ALL \
  --job-name="lv-int-share" \
  --partition=amilan \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --output="$LOG_DIR/%j.out" \
  --wrap="bash -c '$JOB_CMD'")

echo "Submitted job: $JOB_ID"
echo "Monitor with: tail -f $LOG_DIR/${JOB_ID}.out"
