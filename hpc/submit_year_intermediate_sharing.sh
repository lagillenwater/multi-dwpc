#!/bin/bash
#
# Submit year intermediate sharing analysis as an array job.
#
# Computes % of 2024-added genes sharing intermediates with 2016 genes
# for consensus-selected metapaths (selected in both years).
#
# Usage:
#   bash hpc/submit_year_intermediate_sharing.sh
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${YEAR_INT_SHARE_OUTPUT_DIR:-output/year_intermediate_sharing}"
SUPPORT_PATH="${YEAR_INT_SHARE_SUPPORT:-output/year_direct_go_term_support_b5.csv}"
ADDED_PATH="${YEAR_INT_SHARE_ADDED:-output/intermediate/upd_go_bp_2024_added.csv}"
SELECTION_COL="${YEAR_INT_SHARE_SELECTION_COL:-selected_by_effective_n_all}"
MAX_RANK="${YEAR_INT_SHARE_MAX_RANK:-5}"

mkdir -p "$OUTPUT_DIR" hpc/logs

# Extract unique GO terms that have consensus metapaths (selected in both 2016 and 2024)
GO_LIST="$OUTPUT_DIR/go_list.txt"
python3 -c "
import pandas as pd

support = pd.read_csv('$SUPPORT_PATH')
sel_col = '$SELECTION_COL'
support[sel_col] = support[sel_col].astype(str).str.strip().str.lower().isin({'1','true','t','yes'})

# Get GO terms with selected metapaths in 2016
sel_2016 = support[(support['year'] == 2016) & support[sel_col]][['go_id', 'metapath']].drop_duplicates()
# Get GO terms with selected metapaths in 2024
sel_2024 = support[(support['year'] == 2024) & support[sel_col]][['go_id', 'metapath']].drop_duplicates()

# Consensus: selected in both years
consensus = sel_2016.merge(sel_2024, on=['go_id', 'metapath'], how='inner')
go_ids = sorted(consensus['go_id'].unique())

with open('$GO_LIST', 'w') as f:
    for g in go_ids:
        f.write(g + '\n')
print(f'Wrote {len(go_ids)} GO terms (with consensus metapaths) to $GO_LIST')
"

N_GO=$(wc -l < "$GO_LIST")
if [[ "$N_GO" -le 0 ]]; then
  echo "No GO terms found. Check support table and selection column."
  exit 1
fi

echo "Repo root:    $REPO_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "Support path: $SUPPORT_PATH"
echo "Added pairs:  $ADDED_PATH"
echo "Max rank:     $MAX_RANK"
echo "GO terms:     $N_GO"
echo

# Submit array job
ARRAY_JOB=$(sbatch \
  --parsable \
  --export=ALL,YEAR_INT_SHARE_GO_LIST="$GO_LIST",YEAR_INT_SHARE_OUTPUT_DIR="$OUTPUT_DIR",YEAR_INT_SHARE_SUPPORT="$SUPPORT_PATH",YEAR_INT_SHARE_ADDED="$ADDED_PATH",YEAR_INT_SHARE_MAX_RANK="$MAX_RANK" \
  --array="0-$((N_GO - 1))" \
  hpc/year_intermediate_sharing_array.sbatch)

echo "Submitted array job: $ARRAY_JOB (${N_GO} tasks)"

# Submit aggregation job after array completes (just concatenate small summary files)
AGG_CMD="cd \"$REPO_ROOT\" && head -1 \"\$(ls $OUTPUT_DIR/intermediate_sharing_summary_GO:*.csv | head -1)\" > \"$OUTPUT_DIR/intermediate_sharing_summary_all.csv\" && tail -n +2 -q $OUTPUT_DIR/intermediate_sharing_summary_GO:*.csv >> \"$OUTPUT_DIR/intermediate_sharing_summary_all.csv\" && echo 'Aggregated summary to $OUTPUT_DIR/intermediate_sharing_summary_all.csv'"

AGG_JOB=$(sbatch \
  --parsable \
  --export=ALL \
  --dependency="afterok:${ARRAY_JOB}" \
  --job-name="year-int-share-agg" \
  --partition=amilan \
  --qos=normal \
  --cpus-per-task=1 \
  --mem=1G \
  --time=00:10:00 \
  --output="hpc/logs/year-int-share-agg_%j.out" \
  --wrap="bash -c '$AGG_CMD'")

echo "Submitted aggregation job: $AGG_JOB (depends on $ARRAY_JOB)"
