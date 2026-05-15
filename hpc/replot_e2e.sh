#!/bin/bash
#
# replot_e2e.sh -- Regenerate every PDF in an existing e2e run directory using
# --plot-only modes. Reads pre-existing summary CSVs; performs no path
# enumeration, no DWPC compute, no replicate-bank aggregation, no filter
# recomputation.
#
# All PDFs emitted use TrueType (Type 42) fonts because the repo-root
# matplotlibrc is auto-loaded from $REPO_ROOT.
#
# Usage:
#   bash hpc/replot_e2e.sh output/e2e_full_20260422_211647 [B_VALUES] [SEEDS]
#
# Defaults match the --full mode in submit_end_to_end.sh.
#

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 RUN_DIR [B_VALUES] [SEEDS]" >&2
    exit 2
fi

RUN="$1"
B="${2:-2,4,6,8,10,20,30}"
S="${3:-11,22,33,44,55}"

# Always run from repo root so matplotlibrc applies and relative paths line up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Force matplotlib to use the repo-root matplotlibrc (TrueType / Type 42 fonts)
# regardless of CWD or any user-level matplotlibrc on the HPC node.
# Matplotlib accepts either a file path or a directory containing matplotlibrc.
export MATPLOTLIBRC="$REPO_ROOT/matplotlibrc"

if [[ ! -d "$RUN" ]]; then
    echo "ERROR: run directory $RUN not found" >&2
    exit 1
fi

DATA_PREP_FIG_DIR="$RUN/figures/data_prep"
mkdir -p "$DATA_PREP_FIG_DIR"

echo "============================================================"
echo "Replotting (TrueType, --plot-only) from $RUN"
echo "  B = $B"
echo "  Seeds = $S"
echo "============================================================"

run_step () {
    echo
    echo "--- $1 ---"
    shift
    "$@"
}

# 1. Year + LV experiment plots ---------------------------------------------
run_step "year null variance (plot-only)" \
    python3 scripts/experiments/null_variance_experiment.py --analysis-type year --plot-only \
        --workspace-dir "$RUN/year_experiment" \
        --output-dir "$RUN/year_experiment" \
        --b-values "$B" --seeds "$S"

run_step "year rank stability (plot-only)" \
    python3 scripts/experiments/rank_stability_experiment.py --analysis-type year --plot-only \
        --workspace-dir "$RUN/year_experiment" \
        --output-dir "$RUN/year_experiment" \
        --b-values "$B" --seeds "$S"

run_step "lv null variance (plot-only)" \
    python3 scripts/experiments/null_variance_experiment.py --analysis-type lv --plot-only \
        --output-dir "$RUN/lv_experiment" \
        --b-values "$B" --seeds "$S"

run_step "lv rank stability (plot-only)" \
    python3 scripts/experiments/rank_stability_experiment.py --analysis-type lv --plot-only \
        --output-dir "$RUN/lv_experiment" \
        --b-values "$B" --seeds "$S"

# 2. B-selection diagnostic plots (Stage 1 of the pipeline)
# The orchestrators below skip Stage 1 via --skip-b-select, so we have to
# refresh stabilization_curves_*.{pdf,png} explicitly. --plot-only reads
# stabilization_summary.csv, leaves chosen_b.json untouched.
run_step "year B-selection plots (plot-only)" \
    python3 scripts/pipeline/select_optimal_b.py --plot-only \
        --analysis-type year \
        --variance-dir "$RUN/year_experiment/year_null_variance_experiment" \
        --rank-dir     "$RUN/year_experiment/year_rank_stability_experiment" \
        --output-dir   "$RUN/year_full_analysis/b_selection"

run_step "lv B-selection plots (plot-only)" \
    python3 scripts/pipeline/select_optimal_b.py --plot-only \
        --analysis-type lv \
        --variance-dir "$RUN/lv_experiment/lv_null_variance_experiment" \
        --rank-dir     "$RUN/lv_experiment/lv_rank_stability_experiment" \
        --output-dir   "$RUN/lv_full_analysis/b_selection"

# 3. Pipeline downstream plots (year + LV) ----------------------------------
# Stages 1 (B select, refreshed above) and 2 (intermediate sharing) are
# skipped via --skip-b-select and --start-from-stage 3.  --plot-only is
# forwarded to subprocess calls that support it (year_go_term_support,
# year_snapshot_rank_similarity).

run_step "year pipeline stages 3-8 (plot-only forwarded)" \
    python3 scripts/pipeline/run_year_pipeline.py \
        --output-dir "$RUN/year_full_analysis" \
        --year-output-dir "$RUN/year_experiment" \
        --added-pairs-path output/intermediate/upd_go_bp_2024_added.csv \
        --skip-b-select --start-from-stage 3 \
        --plot-only

run_step "lv pipeline stages 3-6 (plot-only forwarded)" \
    python3 scripts/pipeline/run_lv_pipeline.py \
        --output-dir "$RUN/lv_full_analysis" \
        --lv-output-dirs "$RUN/lv_experiment" \
        --skip-b-select --start-from-stage 3 \
        --plot-only

# 4. Data-prep figures into the run dir -------------------------------------
run_step "data-prep: gene classification" \
    python3 scripts/visualization/plot_gene_classification.py \
        --output-dir "$DATA_PREP_FIG_DIR"

run_step "data-prep: jaccard filtering" \
    python3 scripts/visualization/plot_jaccard_filtering.py \
        --output-dir "$DATA_PREP_FIG_DIR"

run_step "data-prep: GO hierarchy" \
    python3 scripts/visualization/plot_go_hierarchy.py \
        --output-dir "$DATA_PREP_FIG_DIR"

echo
echo "============================================================"
echo "Replot complete. PDFs (TrueType) refreshed under:"
echo "  $RUN/year_experiment/year_*_experiment/"
echo "  $RUN/lv_experiment/lv_*_experiment/"
echo "  $RUN/year_full_analysis/"
echo "  $RUN/lv_full_analysis/"
echo "  $DATA_PREP_FIG_DIR"
echo "============================================================"
