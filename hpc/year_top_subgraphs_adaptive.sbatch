#!/bin/bash
#SBATCH --job-name=year-top-adapt
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/year-top-adapt_%j.out
#SBATCH --qos=normal

set -euo pipefail

module load anaconda || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

YEAR_WORKSPACE_DIR="${YEAR_WORKSPACE_DIR:-output/dwpc_direct/all_GO_positive_growth}"
YEAR_GO_SUPPORT_PATH="${YEAR_GO_SUPPORT_PATH:-output/year_direct_go_term_support.csv}"
YEAR_GLOBAL_SUPPORT_PATH="${YEAR_GLOBAL_SUPPORT_PATH:-output/year_direct_global_metapath_support.csv}"
YEAR_GO_SUPPORT_RUNS_PATH="${YEAR_GO_SUPPORT_RUNS_PATH:-}"
YEAR_SUPPORT_B="${YEAR_SUPPORT_B:-}"
YEAR_TOP_PATHS_OUTPUT_DIR="${YEAR_TOP_PATHS_OUTPUT_DIR:-output/metapath_analysis/top_paths_adaptive}"
YEAR_TOP_YEARS="${YEAR_TOP_YEARS:-2016 2024}"

YEAR_GO_EFFECTIVE_MIN_N="${YEAR_GO_EFFECTIVE_MIN_N:-1}"
YEAR_PAIR_MIN_N="${YEAR_PAIR_MIN_N:-1}"
YEAR_PATH_MIN_COUNT="${YEAR_PATH_MIN_COUNT:-1}"
YEAR_PATH_ENUMERATION_TOP_K="${YEAR_PATH_ENUMERATION_TOP_K:-5000}"
YEAR_TOP_CHUNKSIZE="${YEAR_TOP_CHUNKSIZE:-200000}"
YEAR_DEGREE_D="${YEAR_DEGREE_D:-0.5}"

if [[ -n "$YEAR_GO_SUPPORT_RUNS_PATH" ]]; then
  support_cmd=(
    python3 scripts/year_go_term_support.py
    --runs-path "$YEAR_GO_SUPPORT_RUNS_PATH"
    --go-support-output "$YEAR_GO_SUPPORT_PATH"
    --global-support-output "$YEAR_GLOBAL_SUPPORT_PATH"
  )
  if [[ -n "$YEAR_SUPPORT_B" ]]; then
    support_cmd+=(--b "$YEAR_SUPPORT_B")
  fi
else
  support_cmd=(
    python3 scripts/year_go_term_support.py
    --workspace-dir "$YEAR_WORKSPACE_DIR"
    --go-support-output "$YEAR_GO_SUPPORT_PATH"
    --global-support-output "$YEAR_GLOBAL_SUPPORT_PATH"
  )
fi

"${support_cmd[@]}"

python3 scripts/identify_year_top_paths.py \
  --years ${YEAR_TOP_YEARS} \
  --output-dir "$YEAR_TOP_PATHS_OUTPUT_DIR" \
  --support-path "$YEAR_GO_SUPPORT_PATH" \
  --support-sort-metric "${YEAR_SUPPORT_SORT_METRIC:-consensus_score}" \
  --go-selection-method effective_number \
  --go-effective-min-n "$YEAR_GO_EFFECTIVE_MIN_N" \
  --pair-top-n "${YEAR_PAIR_TOP_N:-20}" \
  --pair-selection-method effective_number \
  --pair-min-n "$YEAR_PAIR_MIN_N" \
  --pair-max-n "${YEAR_PAIR_MAX_N:-5}" \
  --pair-effective-alpha "${YEAR_PAIR_EFFECTIVE_ALPHA:-2.0}" \
  --top-paths "${YEAR_TOP_PATHS:-5}" \
  --path-enumeration-top-k "$YEAR_PATH_ENUMERATION_TOP_K" \
  --path-selection-method effective_number \
  --path-min-count "$YEAR_PATH_MIN_COUNT" \
  --degree-d "$YEAR_DEGREE_D" \
  --chunksize "$YEAR_TOP_CHUNKSIZE"

echo "Saved adaptive year top-subgraph outputs under: $YEAR_TOP_PATHS_OUTPUT_DIR"
