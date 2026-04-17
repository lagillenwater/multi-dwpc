#!/bin/bash
#
# Submit year subgraph intermediate analysis as an array job.
#
# Usage:
#   bash hpc/submit_year_subgraph_intermediates.sh
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

OUTPUT_DIR="${YEAR_SGI_OUTPUT_DIR:-output/year_subgraph_intermediates}"
SUPPORT_PATH="${YEAR_SGI_SUPPORT:-output/year_direct_go_term_support_b5.csv}"
SELECTION_COL="${YEAR_SGI_SELECTION_COL:-selected_by_effective_n_all}"
MAX_RANK="${YEAR_SGI_MAX_RANK:-5}"

mkdir -p "$OUTPUT_DIR" hpc/logs/year

# Extract unique GO terms from 2016-selected metapaths in the support table
GO_LIST="$OUTPUT_DIR/go_list.txt"
python3 -c "
import pandas as pd
support = pd.read_csv('$SUPPORT_PATH')
sel_col = '$SELECTION_COL'
support[sel_col] = support[sel_col].astype(str).str.strip().str.lower().isin({'1','true','t','yes'})
go_ids = sorted(support[(support['year'] == 2016) & support[sel_col]]['go_id'].unique())
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

echo "Repo root:    $REPO_ROOT"
echo "Output dir:   $OUTPUT_DIR"
echo "Support path: $SUPPORT_PATH"
echo "Max rank:     $MAX_RANK"
echo "GO terms:     $N_GO"
echo

# Submit array job
ARRAY_JOB=$(sbatch \
  --parsable \
  --export=ALL,YEAR_SGI_GO_LIST="$GO_LIST",YEAR_SGI_OUTPUT_DIR="$OUTPUT_DIR",YEAR_SGI_SUPPORT="$SUPPORT_PATH",YEAR_SGI_MAX_RANK="$MAX_RANK" \
  --array="0-$((N_GO - 1))" \
  hpc/year_subgraph_intermediates_array.sbatch)

echo "Submitted array job: $ARRAY_JOB (${N_GO} tasks)"

# Submit aggregation job after array completes
AGG_CMD="cd \"$REPO_ROOT\" && module load anaconda && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate multi_dwpc && python3 scripts/aggregate_subgraph_intermediates.py --input-dir \"$OUTPUT_DIR\" --output-dir \"$OUTPUT_DIR\""

AGG_JOB=$(sbatch \
  --parsable \
  --export=ALL \
  --dependency="afterok:${ARRAY_JOB}" \
  --job-name="year-sgi-aggregate" \
  --partition=amilan \
  --qos=normal \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --output="hpc/logs/year/year-sgi-aggregate_%j.out" \
  --wrap="bash -c '$AGG_CMD'")

echo "Submitted aggregation job: $AGG_JOB (depends on $ARRAY_JOB)"
