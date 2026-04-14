#!/bin/bash
# Submit the full LV pipeline with single-target refactor
# Results saved to output/lv_single_target_refactor/
#
# Usage:
#   bash hpc/submit_lv_single_target_refactor.sh
#
# This script:
# 1. Runs lv_prepare.sbatch (top-genes + targets + feature manifest)
# 2. Runs DWPC cache warmup array jobs
# 3. Runs lv_precompute_finalize.sbatch
# 4. Runs permutation and random control array jobs (100 replicates each)
# 5. Runs summary array jobs
# 6. Runs rank stability aggregation and plotting

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Output directory for refactored single-target results
export LV_OUTPUT_DIR="output/lv_single_target_refactor"
export LV_WORKSPACE_DIR="$LV_OUTPUT_DIR"

# LV configuration - focus on LV603 with neutrophilia target
export LV_LVS="LV603,LV246,LV57"
export LV_LOADINGS_PATH="data/lv_loadings/multiplier_model_z.tsv.gz"

# Replicate and analysis parameters
export LV_N_REPLICATES=100
export LV_METAPATH_LIMIT=2
export LV_ALL_METAPATHS=0

# Rank stability parameters
export LV_RANK_STAB_B_VALUES="2,4,6,8,10,20"
export LV_RANK_STAB_SEEDS="11,22,33,44,55"
export LV_RANK_STAB_TOP_K="5,10"

# Disable optional analyses for initial run
export INCLUDE_TRACKING=1
export INCLUDE_TOP_PATHS=0
export INCLUDE_LV_QC=0
export INCLUDE_LV_DESCRIPTOR_ANALYSIS=0

# Reference seed for tracking
export TRACK_REFERENCE_SEED=11
export TRACK_TOP_N_VALUES="5,10"

echo "=============================================="
echo "LV Single-Target Refactor Pipeline Submission"
echo "=============================================="
echo "Output directory: $LV_OUTPUT_DIR"
echo "LVs: $LV_LVS"
echo "N replicates: $LV_N_REPLICATES"
echo "B values: $LV_RANK_STAB_B_VALUES"
echo "Seeds: $LV_RANK_STAB_SEEDS"
echo ""

# Create output directory and log directory
mkdir -p "$LV_OUTPUT_DIR"
mkdir -p hpc/logs/lv

# Run the full rebuild script
cd "$REPO_ROOT"
bash hpc/submit_lv_full_rebuild_rerun.sh submit

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Results will be in: $LV_OUTPUT_DIR"
