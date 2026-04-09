#!/bin/bash
#
# Submit LV intermediate sharing analysis.
#
# Computes % of genes sharing intermediates for supported metapaths
# using effective number selection at b=10.
#
# Usage:
#   bash hpc/submit_lv_intermediate_sharing.sh
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${LV_INT_SHARE_OUTPUT_DIR:-output/lv_intermediate_sharing}"
LV_RESULTS="${LV_INT_SHARE_RESULTS:-output/lv_multidwpc/lv_metapath_results.csv}"
LV_PAIR_DWPC="${LV_INT_SHARE_PAIR_DWPC:-output/lv_multidwpc/lv_pair_dwpc_top_selected.csv}"
LV_TOP_GENES="${LV_INT_SHARE_TOP_GENES:-output/lv_multidwpc/lv_top_genes.csv}"

mkdir -p "$OUTPUT_DIR" hpc/logs

echo "Repo root:      $REPO_ROOT"
echo "Output dir:     $OUTPUT_DIR"
echo "LV results:     $LV_RESULTS"
echo "LV pair DWPC:   $LV_PAIR_DWPC"
echo "LV top genes:   $LV_TOP_GENES"
echo

# Single job (small analysis, only 2-3 LVs)
JOB_CMD="cd \"$REPO_ROOT\" && module load anaconda && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate multi_dwpc && python3 scripts/lv_intermediate_sharing.py --lv-results-path \"$LV_RESULTS\" --lv-pair-dwpc-path \"$LV_PAIR_DWPC\" --lv-top-genes-path \"$LV_TOP_GENES\" --output-dir \"$OUTPUT_DIR\""

JOB_ID=$(sbatch \
  --parsable \
  --export=ALL \
  --job-name="lv-int-share" \
  --partition=amilan \
  --qos=normal \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --output="hpc/logs/lv-int-share_%j.out" \
  --wrap="bash -c '$JOB_CMD'")

echo "Submitted job: $JOB_ID"
echo "Monitor with: tail -f hpc/logs/lv-int-share_${JOB_ID}.out"
