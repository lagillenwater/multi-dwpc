#!/bin/bash
#
# End-to-End Pipeline: Data -> Analysis -> Consumables
#
# Complete workflow from raw data download through final outputs.
# Includes all today's updates: effect_size_z rename, z_is_nan tracking,
# convergence diagnostics, n_genes_filtered_by_dwpc.
#
# Modes:
#   --smoke   Quick validation, fewer B values and seeds (30-60 min)
#   --full    Production run with all B values and seeds (hours/weekend)
#
# Analysis types:
#   --analysis-type year   Year (2016 vs 2024) analysis only
#   --analysis-type lv     LV analysis only
#   --analysis-type both   Both (default)
#
# Usage:
#   bash hpc/submit_end_to_end.sh --smoke
#   bash hpc/submit_end_to_end.sh --full --analysis-type year
#   bash hpc/submit_end_to_end.sh --full --analysis-type both
#

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# ============================================
# Parse arguments
# ============================================
MODE="smoke"
ANALYSIS_TYPE="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) MODE="smoke"; shift ;;
        --full) MODE="full"; shift ;;
        --analysis-type) ANALYSIS_TYPE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ============================================
# Configuration
# ============================================
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="output/e2e_${MODE}_${RUN_ID}"
LOG_DIR="hpc/logs/e2e_${MODE}_${RUN_ID}"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR/year" "$LOG_DIR/lv" "$LOG_DIR/shared"

# LV config
LV_LOADINGS_PATH="${LV_LOADINGS_PATH:-data/lv_loadings/multiplier_model_z.tsv.gz}"
LV_OUTPUT_DIR="$OUTPUT_ROOT/lv_experiment"
LV_PIPELINE_OUTPUT="$OUTPUT_ROOT/lv_full_analysis"

# Year config
YEAR_OUTPUT_DIR="$OUTPUT_ROOT/year_experiment"
YEAR_PIPELINE_OUTPUT="$OUTPUT_ROOT/year_full_analysis"
ADDED_PAIRS_PATH="output/intermediate/upd_go_bp_2024_added.csv"

# Smoke vs full
if [[ "$MODE" == "smoke" ]]; then
    B_VALUES="2,3,5,8"
    SEEDS="11,22"
    N_REPLICATES=10
    LV_METAPATH_LIMIT=2
    LV_USE_ALL_METAPATHS=0
    LV_B_MIN=20
    LV_B_MAX=50
    LV_B_BATCH=10
    YEAR_DWPC_WORKERS=2
    EXP_MEM="8G"
    EXP_TIME="01:00:00"
    DWPC_MEM="32G"
    DWPC_TIME="04:00:00"
    YEAR_DWPC_TASK_MEM="16G"
    YEAR_DWPC_TASK_TIME="02:00:00"
    PIPELINE_MEM="32G"
    PIPELINE_TIME="04:00:00"
else
    B_VALUES="2,4,6,8,10,20,30"
    SEEDS="11,22,33,44,55"
    N_REPLICATES=100
    LV_METAPATH_LIMIT=2
    LV_USE_ALL_METAPATHS=1
    LV_B_MIN=200
    LV_B_MAX=1000
    LV_B_BATCH=100
    YEAR_DWPC_WORKERS=4
    EXP_MEM="16G"
    EXP_TIME="04:00:00"
    DWPC_MEM="64G"
    DWPC_TIME="08:00:00"
    YEAR_DWPC_TASK_MEM="16G"
    YEAR_DWPC_TASK_TIME="02:00:00"
    PIPELINE_MEM="64G"
    PIPELINE_TIME="24:00:00"
fi

if [[ "${LV_USE_ALL_METAPATHS:-0}" == "1" ]]; then
    # lv_prepare_experiment.py uses an explicit --all-metapaths flag;
    # lv_multidwpc_analysis.py defaults to unlimited when --metapath-limit-per-target is omitted.
    LV_PREP_METAPATH_ARG="--all-metapaths"
    LV_DWPC_METAPATH_ARG=""
else
    LV_PREP_METAPATH_ARG="--metapath-limit-per-target $LV_METAPATH_LIMIT"
    LV_DWPC_METAPATH_ARG="--metapath-limit-per-target $LV_METAPATH_LIMIT"
fi

echo "=============================================="
echo "End-to-End Pipeline ($MODE)"
echo "=============================================="
echo "Repo root:       $REPO_ROOT"
echo "Output root:     $OUTPUT_ROOT"
echo "Analysis type:   $ANALYSIS_TYPE"
echo "B values:        $B_VALUES"
echo "Seeds:           $SEEDS"
echo "N replicates:    $N_REPLICATES"
echo "Log dir:         $LOG_DIR"
echo

submit() {
    local name="$1" mem="$2" time="$3" cmd="$4" dep="${5:-}" cpus="${6:-4}" array_spec="${7:-}"
    local subdir="shared"
    case "$name" in
        year-*) subdir="year" ;;
        lv-*)   subdir="lv" ;;
    esac
    local dep_flag=""
    [[ -n "$dep" ]] && dep_flag="--dependency=afterok:$dep"
    local array_flag=""
    local out_pattern="%j"
    if [[ -n "$array_spec" ]]; then
        array_flag="--array=$array_spec"
        out_pattern="%A_%a"
    fi
    local activate="module load anaconda && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate multi_dwpc"
    sbatch --parsable $dep_flag $array_flag --export=ALL \
        --job-name="e2e-$name" --partition=amilan --qos=normal \
        --cpus-per-task="$cpus" --mem="$mem" --time="$time" \
        --output="$LOG_DIR/${subdir}/${name}_${out_pattern}.out" \
        --wrap="bash -c 'cd \"$REPO_ROOT\" && $activate && set -e && $cmd'"
}

# ============================================
# Phase 0: Validate prerequisites
# ============================================
# Data download requires internet access which HPC compute nodes may not
# have.  Run these from a login node BEFORE submitting this script:
#
#   conda activate multi_dwpc
#   python scripts/data_prep/load_data.py
#   python scripts/data_prep/download_lv_loadings.py --output-dir data/lv_loadings  # if running LV
#
# The script checks for required files and exits if missing.

echo "Phase 0: Checking prerequisites"
echo "==============================="

REQUIRED_FILES=(
    "data/edges/GpBP.sparse.npz"
    "data/nodes/Gene.tsv"
    "data/metagraph.json"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  MISSING: $f"
        MISSING=1
    fi
done

if [[ "$MISSING" == "1" ]]; then
    echo ""
    echo "Required data files not found. Run on a login node first:"
    echo "  conda activate multi_dwpc"
    echo "  python scripts/data_prep/load_data.py"
    exit 1
fi
echo "  All prerequisite files found."

if [[ "$ANALYSIS_TYPE" == "lv" ]] || [[ "$ANALYSIS_TYPE" == "both" ]]; then
    if [[ ! -f "$LV_LOADINGS_PATH" ]]; then
        echo ""
        echo "  MISSING: $LV_LOADINGS_PATH"
        echo "  Run on a login node: python scripts/data_prep/download_lv_loadings.py --output-dir data/lv_loadings"
        exit 1
    fi
    echo "  LV loadings found."
fi

echo ""

# ============================================
# Phase 1: Data Filtering + Null Generation (year-specific)
# ============================================
# These outputs feed the year chain only; LV has its own null generation
# downstream via lv_generate_control_replicates.py.
DATA_JOB=""
NULL_JOB=""

if [[ "$ANALYSIS_TYPE" == "year" ]] || [[ "$ANALYSIS_TYPE" == "both" ]]; then
    echo "Phase 1: Data Filtering + Null Generation"
    echo "=========================================="

    # 1a: Filter (single job, fast)
    FILTER_CMD="python3 scripts/data_prep/percent_change_and_filtering.py && \
python3 scripts/data_prep/jaccard_similarity_and_filtering.py && \
python3 scripts/data_prep/build_year_top_genes.py --output-dir $YEAR_OUTPUT_DIR"
    DATA_JOB=$(submit "data-filter" "8G" "00:30:00" "$FILTER_CMD" "" 2)
    echo "  Data filtering: $DATA_JOB"

    # 1b: Year permutation nulls (array: 1 task per replicate)
    YEAR_PERM_CMD="export PERMUTATION_IDS=\$SLURM_ARRAY_TASK_ID && \
python3 scripts/data_prep/permutation_null_datasets.py"
    YEAR_PERM_JOB=$(submit "year-perm" "4G" "00:20:00" "$YEAR_PERM_CMD" "$DATA_JOB" 2 "1-${N_REPLICATES}%10")
    echo "  Year permutations: $YEAR_PERM_JOB (array 1-${N_REPLICATES}%10)"

    # 1c: Year random nulls (array: 1 task per replicate)
    YEAR_RAND_CMD="export RANDOM_SAMPLE_IDS=\$SLURM_ARRAY_TASK_ID && \
python3 scripts/data_prep/random_null_datasets.py"
    YEAR_RAND_JOB=$(submit "year-rand" "4G" "00:20:00" "$YEAR_RAND_CMD" "$DATA_JOB" 2 "1-${N_REPLICATES}%10")
    echo "  Year random nulls: $YEAR_RAND_JOB (array 1-${N_REPLICATES}%10)"

    NULL_JOB="${YEAR_PERM_JOB}:${YEAR_RAND_JOB}"
fi

# ============================================
# Year-specific chain
# ============================================
YEAR_FINAL_JOB=""

if [[ "$ANALYSIS_TYPE" == "year" ]] || [[ "$ANALYSIS_TYPE" == "both" ]]; then
    echo ""
    echo "=== Year Analysis Chain ==="

    # Phase 2b-i: Warm the per-metapath DWPC disk cache (single job, parallel
    # within the job but with bounded memory — matrices are streamed to disk
    # and dropped from RAM by cache_in_memory=False).
    YEAR_DWPC_WARMUP_CMD="python3 scripts/data_prep/compute_dwpc_direct.py \
        --output-dir $YEAR_OUTPUT_DIR --warmup-cache --n-workers $YEAR_DWPC_WORKERS"
    YEAR_DWPC_WARMUP_JOB=$(submit "year-dwpc-warmup" "$DWPC_MEM" "$DWPC_TIME" \
        "$YEAR_DWPC_WARMUP_CMD" "$NULL_JOB")
    echo "  Year DWPC warmup: $YEAR_DWPC_WARMUP_JOB"

    # Phase 2b-ii: Per-dataset DWPC (array: 1 task per dataset).
    # Datasets = 2 real + N permuted * 2 years + N random * 2 years = 2 + 4*N.
    # Each task reads from the warmed disk cache, holds at most one metapath
    # matrix in RAM at a time.
    YEAR_N_DATASETS=$(( 2 + 4 * N_REPLICATES ))
    YEAR_DWPC_CMD="DATASET_NAME=\$(python3 scripts/data_prep/compute_dwpc_direct.py \
        --output-dir $YEAR_OUTPUT_DIR --list-datasets | sed -n \"\${SLURM_ARRAY_TASK_ID}p\") && \
    echo \"[task \$SLURM_ARRAY_TASK_ID] dataset=\$DATASET_NAME\" && \
    python3 scripts/data_prep/compute_dwpc_direct.py \
        --output-dir $YEAR_OUTPUT_DIR --dataset-name \"\$DATASET_NAME\" --read-only-cache"
    YEAR_DWPC_JOB=$(submit "year-dwpc" "$YEAR_DWPC_TASK_MEM" "$YEAR_DWPC_TASK_TIME" "$YEAR_DWPC_CMD" \
        "$YEAR_DWPC_WARMUP_JOB" 2 "1-${YEAR_N_DATASETS}%20")
    echo "  Year DWPC: $YEAR_DWPC_JOB (array 1-${YEAR_N_DATASETS}%20)"

    # Phase 3: Experiments (depend on DWPC)
    YEAR_VAR_CMD="python3 scripts/experiments/year_null_variance_experiment.py \
        --workspace-dir $YEAR_OUTPUT_DIR \
        --output-dir $YEAR_OUTPUT_DIR \
        --b-values $B_VALUES --seeds $SEEDS"
    YEAR_VAR_JOB=$(submit "year-var" "$EXP_MEM" "$EXP_TIME" "$YEAR_VAR_CMD" "$YEAR_DWPC_JOB")
    echo "  Year null variance: $YEAR_VAR_JOB"

    YEAR_RANK_CMD="python3 scripts/experiments/year_rank_stability_experiment.py \
        --workspace-dir $YEAR_OUTPUT_DIR \
        --output-dir $YEAR_OUTPUT_DIR \
        --b-values $B_VALUES --seeds $SEEDS"
    YEAR_RANK_JOB=$(submit "year-rank" "$EXP_MEM" "$EXP_TIME" "$YEAR_RANK_CMD" "$YEAR_DWPC_JOB")
    echo "  Year rank stability: $YEAR_RANK_JOB"

    # Phase 4: Full pipeline (depends on both experiments)
    YEAR_PIPE_CMD="python3 scripts/pipeline/run_year_pipeline.py \
        --output-dir $YEAR_PIPELINE_OUTPUT \
        --year-output-dir $YEAR_OUTPUT_DIR \
        --added-pairs-path $ADDED_PAIRS_PATH"
    YEAR_FINAL_JOB=$(submit "year-pipe" "$PIPELINE_MEM" "$PIPELINE_TIME" \
        "$YEAR_PIPE_CMD" "${YEAR_VAR_JOB}:${YEAR_RANK_JOB}")
    echo "  Year pipeline: $YEAR_FINAL_JOB (depends on experiments)"
fi

# ============================================
# LV-specific chain
# ============================================
LV_FINAL_JOB=""

if [[ "$ANALYSIS_TYPE" == "lv" ]] || [[ "$ANALYSIS_TYPE" == "both" ]]; then
    echo ""
    echo "=== LV Analysis Chain ==="

    # Phase 2b-i: Prep LV metadata only (fast; writes lv_top_genes, lv_targets,
    # feature_manifest, lv_real artifact, replicate_manifest). No DWPC compute.
    LV_META_CMD="python3 scripts/experiments/lv_prepare_experiment.py \
        --output-dir $LV_OUTPUT_DIR \
        --lv-loadings $LV_LOADINGS_PATH \
        --prepare-metadata-only \
        $LV_PREP_METAPATH_ARG"
    LV_META_JOB=$(submit "lv-prep-meta" "8G" "00:30:00" "$LV_META_CMD" "")
    echo "  LV prep (metadata): $LV_META_JOB"

    # Phase 2b-ii: Warm LV DWPC disk cache as an array (1 task per metapath).
    # Array is oversized; tasks whose index exceeds the actual metapath count
    # exit 0 (no-op) so afterok is satisfied for the finalize job.
    LV_WARMUP_ARRAY_SIZE=300
    LV_WARMUP_CMD="METAPATH_LIST=\$(python3 scripts/experiments/lv_prepare_experiment.py \
        --output-dir $LV_OUTPUT_DIR --list-metapaths) && \
    TOTAL=\$(echo \"\$METAPATH_LIST\" | grep -c .) && \
    if [[ \$SLURM_ARRAY_TASK_ID -gt \$TOTAL ]]; then \
        echo \"[task \$SLURM_ARRAY_TASK_ID] no-op (only \$TOTAL metapaths)\"; \
        exit 0; \
    fi && \
    METAPATH=\$(echo \"\$METAPATH_LIST\" | sed -n \"\${SLURM_ARRAY_TASK_ID}p\") && \
    echo \"[task \$SLURM_ARRAY_TASK_ID] metapath=\$METAPATH\" && \
    python3 scripts/experiments/lv_prepare_experiment.py \
        --output-dir $LV_OUTPUT_DIR --warmup-metapath \"\$METAPATH\""
    LV_WARMUP_JOB=$(submit "lv-warmup" "16G" "01:00:00" "$LV_WARMUP_CMD" \
        "$LV_META_JOB" 2 "1-${LV_WARMUP_ARRAY_SIZE}%10")
    echo "  LV warmup: $LV_WARMUP_JOB (array 1-${LV_WARMUP_ARRAY_SIZE}%10)"

    # Phase 2b-iii: Finalize precompute (reads all cached matrices, fills the
    # gene_feature_scores memmap, writes precompute outputs). Fast because all
    # DWPC matrices are now on disk.
    LV_FINAL_PREP_CMD="python3 scripts/experiments/lv_prepare_experiment.py \
        --output-dir $LV_OUTPUT_DIR \
        --lv-loadings $LV_LOADINGS_PATH \
        --finalize-precompute \
        $LV_PREP_METAPATH_ARG"
    LV_PREP_JOB=$(submit "lv-prep-finalize" "16G" "01:00:00" \
        "$LV_FINAL_PREP_CMD" "$LV_WARMUP_JOB")
    echo "  LV prep (finalize): $LV_PREP_JOB"

    # Phase 2c: LV multi-DWPC pipeline (precompute scores + nulls + stats).
    # With --resume it skips stages already completed by the prep chain
    # (top-genes, targets, precompute-scores) and runs nulls + stats.
    LV_DWPC_CMD="python3 scripts/experiments/lv_multidwpc_analysis.py \
        --stage pipeline-fast --resume \
        --output-dir $LV_OUTPUT_DIR \
        --lv-loadings $LV_LOADINGS_PATH \
        $LV_DWPC_METAPATH_ARG \
        --b-min $LV_B_MIN --b-max $LV_B_MAX --b-batch $LV_B_BATCH"
    LV_DWPC_JOB=$(submit "lv-dwpc" "$DWPC_MEM" "$DWPC_TIME" "$LV_DWPC_CMD" "$LV_PREP_JOB")
    echo "  LV multi-DWPC: $LV_DWPC_JOB"

    # Phase 2d: Generate control replicates (arrays, 1 task per replicate).
    LV_PERM_CMD="python3 scripts/experiments/lv_generate_control_replicates.py \
        --output-dir $LV_OUTPUT_DIR --control permuted --replicate-id \$SLURM_ARRAY_TASK_ID"
    LV_RAND_CMD="python3 scripts/experiments/lv_generate_control_replicates.py \
        --output-dir $LV_OUTPUT_DIR --control random --replicate-id \$SLURM_ARRAY_TASK_ID"
    LV_PERM_JOB=$(submit "lv-perm" "4G" "00:20:00" "$LV_PERM_CMD" "$LV_PREP_JOB" 2 "1-${N_REPLICATES}%10")
    LV_RAND_JOB=$(submit "lv-rand" "4G" "00:20:00" "$LV_RAND_CMD" "$LV_PREP_JOB" 2 "1-${N_REPLICATES}%10")
    echo "  LV permutations: $LV_PERM_JOB (array 1-${N_REPLICATES}%10)"
    echo "  LV random nulls: $LV_RAND_JOB (array 1-${N_REPLICATES}%10)"

    # Phase 2e: Compute replicate summaries (depends on replicates + DWPC)
    LV_SUM_CMD="python3 scripts/experiments/lv_compute_replicate_summaries.py \
        --output-dir $LV_OUTPUT_DIR"
    LV_SUM_JOB=$(submit "lv-sum" "8G" "01:00:00" "$LV_SUM_CMD" \
        "${LV_PERM_JOB}:${LV_RAND_JOB}:${LV_DWPC_JOB}")
    echo "  LV summaries: $LV_SUM_JOB"

    # Phase 3: Experiments (depend on summaries)
    LV_VAR_CMD="python3 scripts/experiments/lv_null_variance_experiment.py \
        --output-dir $LV_OUTPUT_DIR \
        --b-values $B_VALUES --seeds $SEEDS"
    LV_VAR_JOB=$(submit "lv-var" "$EXP_MEM" "$EXP_TIME" "$LV_VAR_CMD" "$LV_SUM_JOB")
    echo "  LV null variance: $LV_VAR_JOB"

    LV_RANK_CMD="python3 scripts/experiments/lv_rank_stability_experiment.py \
        --output-dir $LV_OUTPUT_DIR \
        --b-values $B_VALUES --seeds $SEEDS"
    LV_RANK_JOB=$(submit "lv-rank" "$EXP_MEM" "$EXP_TIME" "$LV_RANK_CMD" "$LV_SUM_JOB")
    echo "  LV rank stability: $LV_RANK_JOB"

    # Phase 4: Full pipeline (depends on both experiments)
    LV_PIPE_CMD="python3 scripts/pipeline/run_lv_pipeline.py \
        --output-dir $LV_PIPELINE_OUTPUT \
        --lv-output-dirs $LV_OUTPUT_DIR"
    LV_FINAL_JOB=$(submit "lv-pipe" "$PIPELINE_MEM" "$PIPELINE_TIME" \
        "$LV_PIPE_CMD" "${LV_VAR_JOB}:${LV_RANK_JOB}")
    echo "  LV pipeline: $LV_FINAL_JOB (depends on experiments)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "End-to-End Jobs Submitted ($MODE)"
echo "=============================================="
echo ""
echo "Dependency chain:"
if [[ -n "${YEAR_FINAL_JOB:-}" ]]; then
    echo "  Data filter -> [year-perm array, year-rand array] -> DWPC warmup -> DWPC array (1 task/dataset) -> Experiments (var + rank) -> Pipeline"
fi
if [[ -n "${LV_FINAL_JOB:-}" ]]; then
    echo "  LV: Prep -> [DWPC, Perm array, Rand array] -> Summaries -> Experiments -> Pipeline"
fi
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ls $LOG_DIR/"
echo ""
echo "Expected outputs:"
echo "  $OUTPUT_ROOT/"
if [[ -n "${YEAR_FINAL_JOB:-}" ]]; then
    echo "    year_full_analysis/b_selection/chosen_b.json  (includes convergence_summary)"
    echo "    year_full_analysis/intermediate_sharing/      (includes n_genes_filtered_by_dwpc)"
    echo "    year_full_analysis/consumable/"
fi
if [[ -n "${LV_FINAL_JOB:-}" ]]; then
    echo "    lv_full_analysis/b_selection/chosen_b.json    (includes convergence_summary)"
    echo "    lv_full_analysis/intermediate_sharing/        (includes n_genes_filtered_by_dwpc)"
    echo "    lv_full_analysis/consumable/"
fi
echo ""
echo "New columns from today's updates:"
echo "  - all_runs_long.csv: effect_size_z (renamed from effect_size_d), z_is_nan"
echo "  - chosen_b.json: convergence_summary (last_segment_slope, pct_converged)"
echo "  - intermediate_sharing_by_metapath.csv: n_genes_filtered_by_dwpc"
