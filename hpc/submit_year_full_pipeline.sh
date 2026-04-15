#!/bin/bash
#
# Full Year Analysis Pipeline - End-to-end HPC orchestration
#
# Compares 2016 baseline genes vs 2024-added genes across GO terms. Mirrors
# the LV pipeline structure: chosen B drives a single-B intermediate-sharing
# pass, then downstream consumables (gene table, subgraphs) use that same B.
#
# Stages:
# 1. Select optimal B via variance stabilization (diff_var)
# 2. Intermediate sharing at the chosen B only
# 3. Select top N GO terms by median effect size
# 4. Global summary + LV/GO selection diagnostic plots
# 5. Gene connectivity table (restricted to top GO terms)
# 6. Subgraph visualization per top GO term
#
# Usage:
#   bash hpc/submit_year_full_pipeline.sh
#
# Environment variables:
#   YEAR_OUTPUT_DIR       - Upstream year experiment dir (default layout root)
#   YEAR_VARIANCE_DIR     - Override: path to the null-variance experiment dir
#   YEAR_RANK_DIR         - Override: path to the rank-stability experiment dir
#   YEAR_PIPELINE_OUTPUT  - Where all pipeline outputs go
#   YEAR_B_VALUES         - Default intermediate-sharing grid if ever needed
#   YEAR_TOP_GO_TERMS     - Top N GO terms for consumables (default: 10)
#   YEAR_SKIP_B_SELECT    - "1" to reuse existing chosen_b.json
#   YEAR_STOP_AFTER_B     - "1" to submit only Stage 1 and exit
#   YEAR_EFFECT_THRESHOLD - Cohen's d cutoff for metapath selection (default: 0.5)
#   YEAR_DWPC_PERCENTILE  - Percentile filter for DWPC (default: 75)
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Configuration
OUTPUT_DIR="${YEAR_PIPELINE_OUTPUT:-output/year_full_analysis}"
YEAR_OUTPUT_DIR="${YEAR_OUTPUT_DIR:-output/year_experiment}"
ADDED_PAIRS_PATH="${YEAR_ADDED_PAIRS:-output/intermediate/upd_go_bp_2024_added.csv}"
B_VALUES="${YEAR_B_VALUES:-2,5,10,20,30,40}"
TOP_GO_TERMS="${YEAR_TOP_GO_TERMS:-10}"
SKIP_B_SELECT="${YEAR_SKIP_B_SELECT:-0}"
STOP_AFTER_B="${YEAR_STOP_AFTER_B:-0}"
EFFECT_THRESHOLD="${YEAR_EFFECT_THRESHOLD:-0.5}"
DWPC_PERCENTILE="${YEAR_DWPC_PERCENTILE:-75}"

# SBATCH resource knobs (year data is substantially larger than LV -- longer
# walltimes + more memory than the LV pipeline).
INT_SHARE_CPUS="${YEAR_INT_SHARE_CPUS:-8}"
INT_SHARE_MEM="${YEAR_INT_SHARE_MEM:-64G}"
INT_SHARE_TIME="${YEAR_INT_SHARE_TIME:-24:00:00}"
GENE_CPUS="${YEAR_GENE_CPUS:-4}"
GENE_MEM="${YEAR_GENE_MEM:-32G}"
GENE_TIME="${YEAR_GENE_TIME:-04:00:00}"
VIZ_CPUS="${YEAR_VIZ_CPUS:-2}"
VIZ_MEM="${YEAR_VIZ_MEM:-16G}"
VIZ_TIME="${YEAR_VIZ_TIME:-04:00:00}"
SUMMARY_MEM="${YEAR_SUMMARY_MEM:-16G}"
SUMMARY_TIME="${YEAR_SUMMARY_TIME:-01:00:00}"

LOG_DIR="hpc/logs/year/full_pipeline"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================================="
echo "Year Full Analysis Pipeline"
echo "=============================================="
echo "Repo root:          $REPO_ROOT"
echo "Output dir:         $OUTPUT_DIR"
echo "Year output dir:    $YEAR_OUTPUT_DIR"
echo "Added pairs:        $ADDED_PAIRS_PATH"
echo "B values (legacy):  $B_VALUES"
echo "Top GO terms:       $TOP_GO_TERMS"
echo "Effect threshold:   $EFFECT_THRESHOLD"
echo "DWPC percentile:    $DWPC_PERCENTILE"
echo "Skip B selection:   $SKIP_B_SELECT"
echo "Stop after B:       $STOP_AFTER_B"
echo "Int-share resources: ${INT_SHARE_CPUS}c ${INT_SHARE_MEM} ${INT_SHARE_TIME}"
echo "Log dir:            $LOG_DIR"
echo

# ============================================
# Stage 1: B Selection
# ============================================
echo "Stage 1: B Selection"
echo "--------------------"

B_SELECT_JOB_ID=""
CHOSEN_B_PATH="$OUTPUT_DIR/b_selection/chosen_b.json"

if [[ "$SKIP_B_SELECT" == "1" ]] && [[ -f "$CHOSEN_B_PATH" ]]; then
    echo "Skipping B selection - using existing $CHOSEN_B_PATH"
    CHOSEN_B=$(python3 scripts/read_json_value.py "$CHOSEN_B_PATH" chosen_b)
    echo "Chosen B = $CHOSEN_B"
else
    # Explicit overrides win; otherwise fall back to the legacy single-root layout.
    VARIANCE_DIR="${YEAR_VARIANCE_DIR:-$YEAR_OUTPUT_DIR/year_null_variance_experiment}"
    RANK_DIR="${YEAR_RANK_DIR:-$YEAR_OUTPUT_DIR/year_rank_stability_experiment}"

    if [[ ! -d "$VARIANCE_DIR" ]] || [[ ! -d "$RANK_DIR" ]]; then
        echo "Warning: Variance or rank stability directories not found at"
        echo "         $VARIANCE_DIR"
        echo "         $RANK_DIR"
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
            --time="00:30:00" \
            --output="$LOG_DIR/b_select_%j.out" \
            --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $B_SELECT_CMD'")

        echo "Submitted B selection: $B_SELECT_JOB_ID"
    fi
fi

if [[ "$STOP_AFTER_B" == "1" ]]; then
    echo ""
    echo "YEAR_STOP_AFTER_B=1 set -- exiting after Stage 1."
    echo "Inspect $OUTPUT_DIR/b_selection/ once the job finishes, then rerun"
    echo "with YEAR_SKIP_B_SELECT=1 to continue."
    exit 0
fi

# ============================================
# Stage 2: Intermediate Sharing (at chosen B only)
# ============================================
echo ""
echo "Stage 2: Intermediate Sharing"
echo "-----------------------------"

INT_SHARE_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
echo \"Using chosen B = \$CHOSEN_B for intermediate sharing\"

# Remove stale per-B subdirectories from prior runs so only the chosen B remains.
shopt -s nullglob
for stale_dir in \"$OUTPUT_DIR/intermediate_sharing\"/b*/; do
    base=\$(basename \"\$stale_dir\")
    if [[ \"\$base\" != \"b\$CHOSEN_B\" ]]; then
        echo \"Removing stale \$stale_dir\"
        rm -rf \"\$stale_dir\"
    fi
done
shopt -u nullglob

python3 scripts/year_intermediate_sharing.py \\
    --year-output-dir \"$YEAR_OUTPUT_DIR\" \\
    --added-pairs-path \"$ADDED_PAIRS_PATH\" \\
    --b \$CHOSEN_B \\
    --effect-size-threshold $EFFECT_THRESHOLD \\
    --dwpc-percentile $DWPC_PERCENTILE \\
    --output-dir \"$OUTPUT_DIR/intermediate_sharing\"
"

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
    --cpus-per-task=$INT_SHARE_CPUS \
    --mem="$INT_SHARE_MEM" \
    --time="$INT_SHARE_TIME" \
    --output="$LOG_DIR/int_share_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $INT_SHARE_CMD'")

echo "Submitted intermediate sharing: $INT_SHARE_JOB_ID"

# ============================================
# Stage 3: Select Top GO Terms
# ============================================
echo ""
echo "Stage 3: Select Top GO Terms"
echo "-----------------------------"

TOP_GO_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
echo \"Using chosen B = \$CHOSEN_B for top GO selection\"

python3 scripts/select_top_go_terms.py \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing/b\$CHOSEN_B\" \\
    --top-n $TOP_GO_TERMS \\
    --output-dir \"$OUTPUT_DIR\"
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

SUMMARY_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
echo \"Using chosen B = \$CHOSEN_B for global summary\"

python3 scripts/generate_global_summary.py \\
    --analysis-type year \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
    --b-values \$CHOSEN_B \\
    --chosen-b-json \"$OUTPUT_DIR/b_selection/chosen_b.json\" \\
    --effect-size-threshold $EFFECT_THRESHOLD \\
    --output-dir \"$OUTPUT_DIR/global_summary\"
"

SUMMARY_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$INT_SHARE_JOB_ID \
    --export=ALL \
    --job-name="year-summary" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="$SUMMARY_MEM" \
    --time="$SUMMARY_TIME" \
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
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
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
    --cpus-per-task=$GENE_CPUS \
    --mem="$GENE_MEM" \
    --time="$GENE_TIME" \
    --output="$LOG_DIR/gene_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $GENE_CMD'")

echo "Submitted gene table: $GENE_JOB_ID"

# ============================================
# Stage 6: Subgraph Visualization (top GO terms only)
# ============================================
echo ""
echo "Stage 6: Subgraph Visualization"
echo "--------------------------------"

VIZ_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
TOP_GO_IDS=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/top_go_ids.json\")
echo \"Using chosen B = \$CHOSEN_B\"
echo \"Top GO terms: \$TOP_GO_IDS\"

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
    --cpus-per-task=$VIZ_CPUS \
    --mem="$VIZ_MEM" \
    --time="$VIZ_TIME" \
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
echo "  $OUTPUT_DIR/top_go_terms.csv  +  top_go_ids.json"
echo "  $OUTPUT_DIR/intermediate_sharing/b<B>/intermediate_sharing_by_metapath.csv"
echo "  $OUTPUT_DIR/intermediate_sharing/b<B>/gene_paths.csv"
echo "  $OUTPUT_DIR/global_summary/global_summary.csv (+ plots/)"
echo "  $OUTPUT_DIR/consumable/gene_connectivity_table.csv"
echo "  $OUTPUT_DIR/consumable/subgraphs/<go_id>/*.{pdf,png}"
