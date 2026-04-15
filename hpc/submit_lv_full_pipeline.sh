#!/bin/bash
#
# Full LV Analysis Pipeline - End-to-end HPC orchestration
#
# Stages:
# 1. Select optimal B using elbow detection (if not already done)
# 2. Run intermediate sharing at chosen B (or all B values)
# 3. Generate global summaries
# 4. Generate consumable outputs (gene tables, subgraphs)
#
# Usage:
#   bash hpc/submit_lv_full_pipeline.sh
#
# Environment variables:
#   LV_OUTPUT_DIRS     - Space-separated LV experiment output directories
#   LV_PIPELINE_OUTPUT - Output directory for pipeline results
#   LV_B_VALUES        - Comma-separated B values (default: "2,5,10,20,30")
#   LV_SKIP_B_SELECT   - Set to "1" to skip B selection (use existing chosen_b.json)
#   LV_STOP_AFTER_B    - Set to "1" to submit only Stage 1 (B selection) and exit
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Configuration
OUTPUT_DIR="${LV_PIPELINE_OUTPUT:-output/lv_full_analysis}"
LV_OUTPUT_DIRS="${LV_OUTPUT_DIRS:-output/lv_single_target_refactor}"
B_VALUES="${LV_B_VALUES:-2,5,10,20,30}"
SKIP_B_SELECT="${LV_SKIP_B_SELECT:-0}"
STOP_AFTER_B="${LV_STOP_AFTER_B:-0}"
EFFECT_THRESHOLD="${LV_EFFECT_THRESHOLD:-0.2}"
DWPC_PERCENTILE="${LV_DWPC_PERCENTILE:-75}"

# Create directories
LOG_DIR="hpc/logs/lv/full_pipeline"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=============================================="
echo "LV Full Analysis Pipeline"
echo "=============================================="
echo "Repo root:          $REPO_ROOT"
echo "Output dir:         $OUTPUT_DIR"
echo "LV output dirs:     $LV_OUTPUT_DIRS"
echo "B values:           $B_VALUES"
echo "Effect threshold:   $EFFECT_THRESHOLD"
echo "DWPC percentile:    $DWPC_PERCENTILE"
echo "Skip B selection:   $SKIP_B_SELECT"
echo "Stop after B:       $STOP_AFTER_B"
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
    # Determine variance and rank stability directories
    VARIANCE_DIR=""
    RANK_DIR=""

    for lv_dir in $LV_OUTPUT_DIRS; do
        if [[ -d "$lv_dir/lv_null_variance_experiment" ]]; then
            VARIANCE_DIR="$lv_dir/lv_null_variance_experiment"
        fi
        if [[ -d "$lv_dir/lv_rank_stability_experiment" ]]; then
            RANK_DIR="$lv_dir/lv_rank_stability_experiment"
        fi
    done

    if [[ -z "$VARIANCE_DIR" ]] || [[ -z "$RANK_DIR" ]]; then
        echo "Warning: Variance or rank stability directories not found"
        echo "Using default B = 10"
        CHOSEN_B=10
        mkdir -p "$OUTPUT_DIR/b_selection"
        echo "{\"chosen_b\": 10, \"note\": \"default - experiments not found\"}" > "$CHOSEN_B_PATH"
    else
        B_SELECT_CMD="python3 scripts/select_optimal_b.py \\
            --analysis-type lv \\
            --variance-dir \"$VARIANCE_DIR\" \\
            --rank-dir \"$RANK_DIR\" \\
            --output-dir \"$OUTPUT_DIR/b_selection\" \\
            --aggregation median"

        B_SELECT_JOB_ID=$(sbatch \
            --parsable \
            --export=ALL \
            --job-name="lv-b-select" \
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

if [[ "$STOP_AFTER_B" == "1" ]]; then
    echo ""
    echo "LV_STOP_AFTER_B=1 set -- exiting after Stage 1."
    echo "Inspect $OUTPUT_DIR/b_selection/ once the job finishes, then rerun with LV_SKIP_B_SELECT=1 to continue."
    exit 0
fi

# ============================================
# Stage 2: Intermediate Sharing (all B values)
# ============================================
echo ""
echo "Stage 2: Intermediate Sharing"
echo "-----------------------------"

INT_SHARE_CMD="python3 scripts/lv_intermediate_sharing.py \\
    --lv-output-dirs $LV_OUTPUT_DIRS \\
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
    --job-name="lv-int-share" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=4 \
    --mem="32G" \
    --time="04:00:00" \
    --output="$LOG_DIR/int_share_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $INT_SHARE_CMD'")

echo "Submitted intermediate sharing: $INT_SHARE_JOB_ID"

# ============================================
# Stage 3: Global Summary
# ============================================
echo ""
echo "Stage 3: Global Summary"
echo "-----------------------"

# Summary at all B values
SUMMARY_CMD="python3 scripts/generate_global_summary.py \\
    --analysis-type lv \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
    --b-values $B_VALUES \\
    --chosen-b-json \"$OUTPUT_DIR/b_selection/chosen_b.json\" \\
    --output-dir \"$OUTPUT_DIR/global_summary\""

SUMMARY_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$INT_SHARE_JOB_ID \
    --export=ALL \
    --job-name="lv-summary" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="8G" \
    --time="00:30:00" \
    --output="$LOG_DIR/summary_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $SUMMARY_CMD'")

echo "Submitted global summary: $SUMMARY_JOB_ID"

# ============================================
# Stage 4: Gene Connectivity Table
# ============================================
echo ""
echo "Stage 4: Gene Connectivity Table"
echo "---------------------------------"

# Use chosen B for consumable outputs (read from JSON after B selection completes)
GENE_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
echo \"Using chosen B = \$CHOSEN_B for gene table\"

python3 scripts/generate_gene_table.py \\
    --analysis-type lv \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
    --lv-output-dirs $LV_OUTPUT_DIRS \\
    --b \$CHOSEN_B \\
    --output-dir \"$OUTPUT_DIR/consumable\"
"

GENE_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$INT_SHARE_JOB_ID \
    --export=ALL \
    --job-name="lv-gene" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="16G" \
    --time="01:00:00" \
    --output="$LOG_DIR/gene_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $GENE_CMD'")

echo "Submitted gene table: $GENE_JOB_ID"

# ============================================
# Stage 5: Subgraph Visualization
# ============================================
echo ""
echo "Stage 5: Subgraph Visualization"
echo "--------------------------------"

VIZ_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
echo \"Using chosen B = \$CHOSEN_B for visualization\"

python3 scripts/plot_metapath_subgraphs.py \\
    --analysis-type lv \\
    --input-dir \"$OUTPUT_DIR/intermediate_sharing\" \\
    --gene-table \"$OUTPUT_DIR/consumable/gene_connectivity_table.csv\" \\
    --b \$CHOSEN_B \\
    --output-dir \"$OUTPUT_DIR/consumable/subgraphs\" \\
    --top-k 3
"

VIZ_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$GENE_JOB_ID \
    --export=ALL \
    --job-name="lv-viz" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="8G" \
    --time="00:30:00" \
    --output="$LOG_DIR/viz_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $VIZ_CMD'")

echo "Submitted visualization: $VIZ_JOB_ID"

# ============================================
# Stage 6: Plotting (intermediate sharing plots)
# ============================================
echo ""
echo "Stage 6: Intermediate Sharing Plots"
echo "------------------------------------"

PLOT_CMD="
CHOSEN_B=\$(python3 scripts/read_json_value.py \"$OUTPUT_DIR/b_selection/chosen_b.json\" chosen_b)
echo \"Using chosen B = \$CHOSEN_B for plots\"

# Check if subdirectory exists
if [[ -d \"$OUTPUT_DIR/intermediate_sharing/b\$CHOSEN_B\" ]]; then
    INPUT_DIR=\"$OUTPUT_DIR/intermediate_sharing/b\$CHOSEN_B\"
else
    INPUT_DIR=\"$OUTPUT_DIR/intermediate_sharing\"
fi

python3 scripts/plot_lv_intermediate_sharing.py --input-dir \"\$INPUT_DIR\"
"

PLOT_JOB_ID=$(sbatch \
    --parsable \
    --dependency=afterok:$INT_SHARE_JOB_ID \
    --export=ALL \
    --job-name="lv-plot" \
    --partition=amilan \
    --qos=normal \
    --cpus-per-task=2 \
    --mem="4G" \
    --time="00:15:00" \
    --output="$LOG_DIR/plot_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $PLOT_CMD'")

echo "Submitted plots: $PLOT_JOB_ID"

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "LV Pipeline Jobs Submitted"
echo "=============================================="
echo ""
echo "Job chain:"
[[ -n "${B_SELECT_JOB_ID:-}" ]] && echo "  1. B Selection:          $B_SELECT_JOB_ID"
echo "  2. Intermediate Sharing: $INT_SHARE_JOB_ID"
echo "  3. Global Summary:       $SUMMARY_JOB_ID"
echo "  4. Gene Table:           $GENE_JOB_ID"
echo "  5. Visualization:        $VIZ_JOB_ID"
echo "  6. Plots:                $PLOT_JOB_ID"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  ls -la $LOG_DIR/"
echo ""
echo "Expected outputs:"
echo "  $OUTPUT_DIR/b_selection/chosen_b.json"
echo "  $OUTPUT_DIR/intermediate_sharing/b*/intermediate_sharing_by_metapath.csv"
echo "  $OUTPUT_DIR/global_summary/global_summary.csv"
echo "  $OUTPUT_DIR/consumable/gene_connectivity_table.csv"
echo "  $OUTPUT_DIR/consumable/subgraphs/*.png"
