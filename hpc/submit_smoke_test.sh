#!/bin/bash
#
# HPC Smoke Test - Validate and profile before full deployment
#
# Runs minimal test jobs to:
# 1. Validate end-to-end pipeline works
# 2. Measure memory and time for resource estimation
# 3. Check all expected output files are created
#
# Usage:
#   bash hpc/submit_smoke_test.sh
#
# After completion, check logs for resource usage to extrapolate for full runs.
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Configuration
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-output/smoke_test}"
B_VALUE="${SMOKE_B:-10}"  # Single B value for smoke test
LV_ID="${SMOKE_LV_ID:-LV246}"  # Single LV for testing
GO_ID="${SMOKE_GO_ID:-}"  # Single GO term for year testing (optional)

# Create directories
LOG_DIR="hpc/logs/smoke_test"
mkdir -p "$SMOKE_OUTPUT_DIR" "$LOG_DIR"

echo "=============================================="
echo "HPC Smoke Test"
echo "=============================================="
echo "Repo root:    $REPO_ROOT"
echo "Output dir:   $SMOKE_OUTPUT_DIR"
echo "B value:      $B_VALUE"
echo "LV ID:        $LV_ID"
echo "GO ID:        ${GO_ID:-'(auto-select first available)'}"
echo "Log dir:      $LOG_DIR"
echo

# Helper function to submit job (matches working pattern from submit_lv_intermediate_sharing.sh)
submit_timed_job() {
    local job_name="$1"
    local mem="$2"
    local time_limit="$3"
    local cmd="$4"

    # Match the exact pattern from working scripts
    local job_cmd="cd \"$REPO_ROOT\" && module load anaconda && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate multi_dwpc && $cmd"

    local job_id
    job_id=$(sbatch \
        --parsable \
        --export=ALL \
        --job-name="smoke-$job_name" \
        --partition=amilan \
        --qos=normal \
        --cpus-per-task=4 \
        --mem="$mem" \
        --time="$time_limit" \
        --output="$LOG_DIR/${job_name}_%j.out" \
        --wrap="bash -c '$job_cmd'")

    echo "$job_id"
}

# ============================================
# Stage 1: LV Intermediate Sharing (single LV, single B)
# ============================================
echo "Stage 1: LV Intermediate Sharing"
echo "--------------------------------"

LV_OUTPUT_DIR="output/lv_single_target_refactor"  # Existing LV experiment output

# Check if source data exists
if [[ ! -d "$LV_OUTPUT_DIR" ]]; then
    echo "Warning: LV output directory not found: $LV_OUTPUT_DIR"
    echo "Skipping LV smoke test. Run LV experiments first."
    LV_JOB_ID=""
else
    LV_CMD="python3 scripts/pipeline/lv_intermediate_sharing.py --lv-output-dirs $LV_OUTPUT_DIR --b $B_VALUE --effect-size-threshold 0.2 --dwpc-percentile 75 --output-dir \"$SMOKE_OUTPUT_DIR/lv_intermediate_sharing\""

    LV_JOB_ID=$(submit_timed_job "lv-int" "16G" "01:00:00" "$LV_CMD")
    echo "Submitted LV intermediate sharing: $LV_JOB_ID"
fi

# ============================================
# Stage 2: Global Summary (depends on Stage 1)
# ============================================
echo ""
echo "Stage 2: Global Summary"
echo "-----------------------"

if [[ -n "${LV_JOB_ID:-}" ]]; then
    SUMMARY_CMD="python3 scripts/pipeline/generate_global_summary.py --analysis-type lv --input-dir \"$SMOKE_OUTPUT_DIR/lv_intermediate_sharing\" --output-dir \"$SMOKE_OUTPUT_DIR/lv_global_summary\""

    SUMMARY_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterok:$LV_JOB_ID \
        --export=ALL \
        --job-name="smoke-summary" \
        --partition=amilan \
        --qos=normal \
        --cpus-per-task=2 \
        --mem="4G" \
        --time="00:15:00" \
        --output="$LOG_DIR/summary_%j.out" \
        --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $SUMMARY_CMD'")

    echo "Submitted global summary: $SUMMARY_JOB_ID (depends on $LV_JOB_ID)"
else
    SUMMARY_JOB_ID=""
fi

# ============================================
# Stage 3: Gene Table (depends on Stage 1)
# ============================================
echo ""
echo "Stage 3: Gene Connectivity Table"
echo "---------------------------------"

if [[ -n "${LV_JOB_ID:-}" ]]; then
    GENE_CMD="python3 scripts/pipeline/generate_gene_table.py --analysis-type lv --input-dir \"$SMOKE_OUTPUT_DIR/lv_intermediate_sharing\" --lv-output-dirs $LV_OUTPUT_DIR --output-dir \"$SMOKE_OUTPUT_DIR/lv_consumable\""

    GENE_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterok:$LV_JOB_ID \
        --export=ALL \
        --job-name="smoke-gene" \
        --partition=amilan \
        --qos=normal \
        --cpus-per-task=2 \
        --mem="8G" \
        --time="00:30:00" \
        --output="$LOG_DIR/gene_%j.out" \
        --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $GENE_CMD'")

    echo "Submitted gene table: $GENE_JOB_ID (depends on $LV_JOB_ID)"
else
    GENE_JOB_ID=""
fi

# ============================================
# Stage 4: Subgraph Visualization (depends on Stage 3)
# ============================================
echo ""
echo "Stage 4: Subgraph Visualization"
echo "--------------------------------"

if [[ -n "${GENE_JOB_ID:-}" ]]; then
    VIZ_CMD="python3 scripts/visualization/plot_metapath_subgraphs.py --analysis-type lv --input-dir \"$SMOKE_OUTPUT_DIR/lv_intermediate_sharing\" --gene-table \"$SMOKE_OUTPUT_DIR/lv_consumable/gene_connectivity_table.csv\" --output-dir \"$SMOKE_OUTPUT_DIR/lv_consumable/subgraphs\" --top-k 2"

    VIZ_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterok:$GENE_JOB_ID \
        --export=ALL \
        --job-name="smoke-viz" \
        --partition=amilan \
        --qos=normal \
        --cpus-per-task=2 \
        --mem="4G" \
        --time="00:15:00" \
        --output="$LOG_DIR/viz_%j.out" \
        --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $VIZ_CMD'")

    echo "Submitted visualization: $VIZ_JOB_ID (depends on $GENE_JOB_ID)"
else
    VIZ_JOB_ID=""
fi

# ============================================
# Stage 5: Year Intermediate Sharing (single GO term)
# ============================================
echo ""
echo "Stage 5: Year Intermediate Sharing"
echo "-----------------------------------"

YEAR_OUTPUT_DIR="${YEAR_OUTPUT_DIR:-output/year_experiment}"
ADDED_PAIRS_PATH="${ADDED_PAIRS_PATH:-output/intermediate/upd_go_bp_2024_added.csv}"
YEAR_JOB_ID=""

if [[ -d "$YEAR_OUTPUT_DIR" ]] && [[ -f "$ADDED_PAIRS_PATH" ]]; then
    # If no GO_ID specified, auto-select first available from rank stability results
    if [[ -z "${GO_ID:-}" ]]; then
        RUNS_PATH="$YEAR_OUTPUT_DIR/year_rank_stability_experiment/all_runs_long.csv"
        if [[ -f "$RUNS_PATH" ]]; then
            GO_ID=$(head -2 "$RUNS_PATH" | tail -1 | cut -d',' -f1 | tr -d '"')
            echo "Auto-selected GO term: $GO_ID"
        fi
    fi

    if [[ -n "${GO_ID:-}" ]]; then
        YEAR_CMD="python3 scripts/pipeline/year_intermediate_sharing.py --year-output-dir \"$YEAR_OUTPUT_DIR\" --added-pairs-path \"$ADDED_PAIRS_PATH\" --b $B_VALUE --effect-size-threshold 0.2 --dwpc-percentile 75 --go-id \"$GO_ID\" --output-dir \"$SMOKE_OUTPUT_DIR/year_intermediate_sharing\""

        YEAR_JOB_ID=$(sbatch \
            --parsable \
            --export=ALL \
            --job-name="smoke-year" \
            --partition=amilan \
            --qos=normal \
            --cpus-per-task=4 \
            --mem="16G" \
            --time="01:00:00" \
            --output="$LOG_DIR/year_%j.out" \
            --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && $YEAR_CMD'")

        echo "Submitted Year intermediate sharing: $YEAR_JOB_ID (GO term: $GO_ID)"
    else
        echo "Warning: Could not determine GO term for Year smoke test"
    fi
else
    echo "Warning: Year output directory or added pairs not found, skipping Year smoke test"
    echo "  Year output dir: $YEAR_OUTPUT_DIR"
    echo "  Added pairs: $ADDED_PAIRS_PATH"
fi

# ============================================
# Stage 6: Validation Job (depends on all previous)
# ============================================
echo ""
echo "Stage 6: Validation"
echo "-------------------"

# Build dependency string
DEPS=""
[[ -n "${SUMMARY_JOB_ID:-}" ]] && DEPS="${DEPS}:${SUMMARY_JOB_ID}"
[[ -n "${VIZ_JOB_ID:-}" ]] && DEPS="${DEPS}:${VIZ_JOB_ID}"
[[ -n "${YEAR_JOB_ID:-}" ]] && DEPS="${DEPS}:${YEAR_JOB_ID}"
DEPS="${DEPS#:}"  # Remove leading colon

if [[ -n "$DEPS" ]]; then
    VALIDATE_CMD="
echo '=============================================='
echo 'Smoke Test Validation'
echo '=============================================='

# Check LV output files
echo ''
echo '--- LV Analysis ---'
lv_files=(
    '$SMOKE_OUTPUT_DIR/lv_intermediate_sharing/intermediate_sharing_by_metapath.csv'
    '$SMOKE_OUTPUT_DIR/lv_intermediate_sharing/intermediate_sharing_summary.csv'
    '$SMOKE_OUTPUT_DIR/lv_global_summary/global_summary.csv'
    '$SMOKE_OUTPUT_DIR/lv_consumable/gene_connectivity_table.csv'
)

all_ok=true
for f in \"\${lv_files[@]}\"; do
    if [[ -f \"\$f\" ]]; then
        size=\$(wc -c < \"\$f\")
        lines=\$(wc -l < \"\$f\")
        echo \"[OK] \$f (\$lines lines, \$size bytes)\"
    else
        echo \"[MISSING] \$f\"
        all_ok=false
    fi
done

# Check for LV subgraph images
n_lv_images=\$(find '$SMOKE_OUTPUT_DIR/lv_consumable/subgraphs' -name '*.png' 2>/dev/null | wc -l)
echo \"LV subgraph images: \$n_lv_images\"

# Check Year output files
echo ''
echo '--- Year Analysis ---'
year_files=(
    '$SMOKE_OUTPUT_DIR/year_intermediate_sharing/intermediate_sharing_by_metapath.csv'
    '$SMOKE_OUTPUT_DIR/year_intermediate_sharing/intermediate_sharing_summary.csv'
)

for f in \"\${year_files[@]}\"; do
    if [[ -f \"\$f\" ]]; then
        size=\$(wc -c < \"\$f\")
        lines=\$(wc -l < \"\$f\")
        echo \"[OK] \$f (\$lines lines, \$size bytes)\"
    else
        echo \"[SKIPPED] \$f (Year analysis may not have run)\"
    fi
done

echo ''
if \$all_ok; then
    echo '=============================================='
    echo 'SMOKE TEST PASSED'
    echo '=============================================='
else
    echo '=============================================='
    echo 'SMOKE TEST FAILED - Missing LV files'
    echo '=============================================='
    exit 1
fi

# Collect timing info from logs
echo ''
echo 'Resource Usage Summary:'
echo '-----------------------'
for log in $LOG_DIR/*.out; do
    if [[ -f \"\$log\" ]]; then
        name=\$(basename \"\$log\" .out)
        runtime=\$(grep 'Runtime:' \"\$log\" 2>/dev/null | tail -1 || echo 'N/A')
        echo \"\$name: \$runtime\"
    fi
done
"

    VALIDATE_JOB_ID=$(sbatch \
        --parsable \
        --dependency=afterok:$DEPS \
        --export=ALL \
        --job-name="smoke-validate" \
        --partition=amilan \
        --qos=normal \
        --cpus-per-task=1 \
        --mem="1G" \
        --time="00:05:00" \
        --output="$LOG_DIR/validate_%j.out" \
        --wrap="bash -c '$VALIDATE_CMD'")

    echo "Submitted validation: $VALIDATE_JOB_ID (depends on $DEPS)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Smoke Test Jobs Submitted"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  ls -la $LOG_DIR/"
echo ""
echo "After completion, check validation log:"
[[ -n "${VALIDATE_JOB_ID:-}" ]] && echo "  cat $LOG_DIR/validate_${VALIDATE_JOB_ID}.out"
echo ""
echo "Resource estimates for full runs will be in the validation log."
