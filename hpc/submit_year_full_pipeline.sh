#!/bin/bash
#
# Full Year Analysis Pipeline - End-to-end HPC orchestration
#
# Compares 2016 baseline genes vs 2024-added genes across GO terms.
#
# Stages:
# 1. Select optimal B using elbow detection (if not already done)
# 2. Run intermediate sharing at chosen B (or all B values)
# 3. Select top GO terms by effect size
# 4. Generate global summaries
# 5. Generate consumable outputs (gene tables, subgraphs)
#
# Usage:
#   bash hpc/submit_year_full_pipeline.sh
#
# Environment variables:
#   YEAR_OUTPUT_DIR      - Year experiment output directory
#   YEAR_PIPELINE_OUTPUT - Output directory for pipeline results
#   YEAR_B_VALUES        - Comma-separated B values (default: "2,5,10,20,30")
#   YEAR_TOP_GO_TERMS    - Number of top GO terms to analyze (default: 10)
#   YEAR_SKIP_B_SELECT   - Set to "1" to skip B selection
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Configuration
OUTPUT_DIR="${YEAR_PIPELINE_OUTPUT:-output/year_full_analysis}"
YEAR_OUTPUT_DIR="${YEAR_OUTPUT_DIR:-output/year_experiment}"
ADDED_PAIRS_PATH="${YEAR_ADDED_PAIRS:-output/intermediate/upd_go_bp_2024_added.csv}"
B_VALUES="${YEAR_B_VALUES:-2,5,10,20,30}"
TOP_GO_TERMS="${YEAR_TOP_GO_TERMS:-10}"
SKIP_B_SELECT="${YEAR_SKIP_B_SELECT:-0}"
EFFECT_THRESHOLD="${YEAR_EFFECT_THRESHOLD:-0.2}"
DWPC_PERCENTILE="${YEAR_DWPC_PERCENTILE:-75}"

# Create directories
LOG_DIR="hpc/logs/year/full_pipeline"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================================="
echo "Year Full Analysis Pipeline"
echo "=============================================="
echo "Repo root:          $REPO_ROOT"
echo "Output dir:         $OUTPUT_DIR"
echo "Year output dir:    $YEAR_OUTPUT_DIR"
echo "Added pairs:        $ADDED_PAIRS_PATH"
echo "B values:           $B_VALUES"
echo "Top GO terms:       $TOP_GO_TERMS"
echo "Effect threshold:   $EFFECT_THRESHOLD"
echo "DWPC percentile:    $DWPC_PERCENTILE"
echo "Skip B selection:   $SKIP_B_SELECT"
echo "Log dir:            $LOG_DIR"
echo

# ============================================
# Stage 1: B Selection (optional)
# ============================================
echo "Stage 1: B Selection"
echo "--------------------"

B_SELECT_JOB_ID=""
CHOSEN_B_PATH="$OUTPUT_DIR/b_selection/chosen_b.json"

if [[ "$SKIP_B_SELECT" == "1" ]] && [[ -f "$CHOSEN_B_PATH" ]]; then
    echo "Skipping B selection - using existing $CHOSEN_B_PATH"
    CHOSEN_B=$(python3 -c "import json; print(json.load(open('$CHOSEN_B_PATH'))['chosen_b'])")
    echo "Chosen B = $CHOSEN_B"
else
    VARIANCE_DIR="$YEAR_OUTPUT_DIR/year_null_variance_experiment"
    RANK_DIR="$YEAR_OUTPUT_DIR/year_rank_stability_experiment"

    if [[ ! -d "$VARIANCE_DIR" ]] || [[ ! -d "$RANK_DIR" ]]; then
        echo "Warning: Variance or rank stability directories not found"
        echo "Using default B = 10"
        CHOSEN_B=10
        mkdir -p "$OUTPUT_DIR/b_selection"
        echo "{\"chosen_b\": 10, \"note\": \"default - experiments not found\"}" > "$CHOSEN_B_PATH"
    else
        B_SELECT_CMD="python3 scripts/select_optimal_b.py \\
            --analysis-type year \\
            --variance-dir \"$VARIANCE_DIR\" \\
            --rank-dir \"$RANK_DIR\" \\
            --output-dir \"$OUTPUT_DIR/b_selection\" \\
            --aggregation median"

        B_SELECT_JOB_ID=$(sbatch \
            --parsable \
            --export=ALL \
            --job-name="year-b-select" \
            --partition=amilan \
            --qos=normal \
            --cpus-per-task=2 \
            --mem="4G" \
            --time="00:15:00" \
            --output="$LOG_DIR/b_select_%j.out" \
            --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $B_SELECT_CMD'")

        echo "Submitted B selection: $B_SELECT_JOB_ID"
    fi
fi

# ============================================
# Stage 2: Intermediate Sharing (all B values)
# ============================================
echo ""
echo "Stage 2: Intermediate Sharing"
echo "-----------------------------"

INT_SHARE_CMD="python3 scripts/year_intermediate_sharing.py \\
    --year-output-dir \"$YEAR_OUTPUT_DIR\" \\
    --added-pairs-path \"$ADDED_PAIRS_PATH\" \\
    --b-values $B_VALUES \\
    --effect-size-threshold $EFFECT_THRESHOLD \\
    --dwpc-percentile $DWPC_PERCENTILE \\
    --output-dir \"$OUTPUT_DIR/intermediate_sharing\""

if [[ -n "$B_SELECT_JOB_ID" ]]; then
    DEP_FLAG="--dependency=afterok:$B_SELECT_JOB_ID"
else
    DEP_FLAG=""
fi

INT_SHARE_JOB_ID=$(sbatch \
    --parsable \
    $DEP_FLAG \
    --export=ALL \
    --job-name="year-int-share" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=4 \
    --mem="32G" \
    --time="08:00:00" \
    --output="$LOG_DIR/int_share_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $INT_SHARE_CMD'")

echo "Submitted intermediate sharing: $INT_SHARE_JOB_ID"

# ============================================
# Stage 3: Select Top GO Terms
# ============================================
echo ""
echo "Stage 3: Select Top GO Terms"
echo "-----------------------------"

# This stage identifies the top GO terms by median effect size
TOP_GO_CMD="
CHOSEN_B=\$(python3 -c \"import json; print(json.load(open('$OUTPUT_DIR/b_selection/chosen_b.json'))['chosen_b'])\")
echo \"Using chosen B = \$CHOSEN_B\"

# Determine input directory
if [[ -d \"$OUTPUT_DIR/intermediate_sharing/b\$CHOSEN_B\" ]]; then
    INPUT_DIR=\"$OUTPUT_DIR/intermediate_sharing/b\$CHOSEN_B\"
else
    INPUT_DIR=\"$OUTPUT_DIR/intermediate_sharing\"
fi

# Extract top GO terms by median effect size
python3 -c \"
import pandas as pd
import json

df = pd.read_csv('\$INPUT_DIR/intermediate_sharing_by_metapath.csv')

# Group by GO term and compute median effect size
go_summary = df.groupby('go_id').agg(
    n_metapaths=('metapath', 'count'),
    median_effect_size_d=('effect_size_d', 'median'),
    max_effect_size_d=('effect_size_d', 'max'),
    n_genes_2016=('n_genes_2016', 'first'),
    n_genes_2024_added=('n_genes_2024_added', 'first'),
).reset_index()

# Sort by median effect size and take top N
top_go = go_summary.nlargest($TOP_GO_TERMS, 'median_effect_size_d')
top_go.to_csv('$OUTPUT_DIR/top_go_terms.csv', index=False)
print(f'Selected {len(top_go)} top GO terms')

# Also save as list for downstream
top_go_ids = top_go['go_id'].tolist()
with open('$OUTPUT_DIR/top_go_ids.json', 'w') as f:
    json.dump(top_go_ids, f)
print(f'Top GO terms: {top_go_ids}')
\"
"

TOP_GO_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$INT_SHARE_JOB_ID \
    --export=ALL \
    --job-name="year-top-go" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=1 \
    --mem="4G" \
    --time="00:10:00" \
    --output="$LOG_DIR/top_go_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $TOP_GO_CMD'")

echo "Submitted top GO selection: $TOP_GO_JOB_ID"

# ============================================
# Stage 4: Global Summary
# ============================================
echo ""
echo "Stage 4: Global Summary"
echo "-----------------------"

SUMMARY_CMD="python3 scripts/generate_global_summary.py \\
    --analysis-type year \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
    --b-values $B_VALUES \\
    --chosen-b-json \"$OUTPUT_DIR/b_selection/chosen_b.json\" \\
    --output-dir \"$OUTPUT_DIR/global_summary\""

SUMMARY_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$INT_SHARE_JOB_ID \
    --export=ALL \
    --job-name="year-summary" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="8G" \
    --time="00:30:00" \
    --output="$LOG_DIR/summary_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $SUMMARY_CMD'")

echo "Submitted global summary: $SUMMARY_JOB_ID"

# ============================================
# Stage 5: Gene Connectivity Table (for top GO terms)
# ============================================
echo ""
echo "Stage 5: Gene Connectivity Table"
echo "---------------------------------"

GENE_CMD="
CHOSEN_B=\$(python3 -c \"import json; print(json.load(open('$OUTPUT_DIR/b_selection/chosen_b.json'))['chosen_b'])\")
echo \"Using chosen B = \$CHOSEN_B for gene table\"

python3 scripts/generate_gene_table.py \\
    --analysis-type year \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
    --lv-output-dirs \"$YEAR_OUTPUT_DIR\" \\
    --b \$CHOSEN_B \\
    --output-dir \"$OUTPUT_DIR/consumable\"
"

GENE_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$TOP_GO_JOB_ID \
    --export=ALL \
    --job-name="year-gene" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="16G" \
    --time="01:00:00" \
    --output="$LOG_DIR/gene_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $GENE_CMD'")

echo "Submitted gene table: $GENE_JOB_ID"

# ============================================
# Stage 6: Subgraph Visualization (for top GO terms)
# ============================================
echo ""
echo "Stage 6: Subgraph Visualization"
echo "--------------------------------"

VIZ_CMD="
CHOSEN_B=\$(python3 -c \"import json; print(json.load(open('$OUTPUT_DIR/b_selection/chosen_b.json'))['chosen_b'])\")
TOP_GO_IDS=\$(python3 -c \"import json; print(' '.join(json.load(open('$OUTPUT_DIR/top_go_ids.json'))))\")
echo \"Using chosen B = \$CHOSEN_B\"
echo \"Top GO terms: \$TOP_GO_IDS\"

# Generate subgraphs for each top GO term
for go_id in \$TOP_GO_IDS; do
    echo \"Generating subgraphs for \$go_id\"
    python3 scripts/plot_metapath_subgraphs.py \\
        --analysis-type year \\
        --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
        --gene-table \"$OUTPUT_DIR/consumable/gene_connectivity_table.csv\" \\
        --b \$CHOSEN_B \\
        --gene-set-id \"\$go_id\" \\
        --output-dir \"$OUTPUT_DIR/consumable/subgraphs/\$go_id\" \\
        --top-k 3
done
"

VIZ_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$GENE_JOB_ID \
    --export=ALL \
    --job-name="year-viz" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="8G" \
    --time="01:00:00" \
    --output="$LOG_DIR/viz_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $VIZ_CMD'")

echo "Submitted visualization: $VIZ_JOB_ID"

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Year Pipeline Jobs Submitted"
echo "=============================================="
echo ""
echo "Job chain:"
[[ -n "${B_SELECT_JOB_ID:-}" ]] && echo "  1. B Selection:          $B_SELECT_JOB_ID"
echo "  2. Intermediate Sharing: $INT_SHARE_JOB_ID"
echo "  3. Top GO Selection:     $TOP_GO_JOB_ID"
echo "  4. Global Summary:       $SUMMARY_JOB_ID"
echo "  5. Gene Table:           $GENE_JOB_ID"
echo "  6. Visualization:        $VIZ_JOB_ID"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  ls -la $LOG_DIR/"
echo ""
echo "Expected outputs:"
echo "  $OUTPUT_DIR/b_selection/chosen_b.json"
echo "  $OUTPUT_DIR/top_go_terms.csv"
echo "  $OUTPUT_DIR/intermediate_sharing/b*/intermediate_sharing_by_metapath.csv"
echo "  $OUTPUT_DIR/global_summary/global_summary.csv"
echo "  $OUTPUT_DIR/consumable/gene_connectivity_table.csv"
echo "  $OUTPUT_DIR/consumable/subgraphs/<go_id>/*.png"
