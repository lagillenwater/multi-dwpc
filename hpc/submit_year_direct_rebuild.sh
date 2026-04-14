#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
MODE="${1:-submit}"

SBATCH_QOS="${SBATCH_QOS:-normal}"
SBATCH_PARTITION="${SBATCH_PARTITION:-amilan}"
PLOT_CPUS="${PLOT_CPUS:-2}"
PLOT_MEM="${PLOT_MEM:-8G}"
PLOT_TIME="${PLOT_TIME:-01:00:00}"
POST_CPUS="${POST_CPUS:-4}"
POST_MEM="${POST_MEM:-24G}"
POST_TIME="${POST_TIME:-04:00:00}"

: "${YEAR_N_REPLICATES:=20}"
: "${YEAR_NULL_VAR_B_VALUES:=1,2,5,10,20}"
: "${YEAR_NULL_VAR_SEEDS:=11,22,33,44,55}"
: "${YEAR_RANK_STAB_B_VALUES:=1,2,5,10,20}"
: "${YEAR_RANK_STAB_SEEDS:=11,22,33,44,55}"
: "${YEAR_RANK_STAB_TOP_K:=5,10}"
: "${TRACK_REFERENCE_SEED:=11}"
: "${TRACK_TOP_N_VALUES:=5,10,all}"
: "${INCLUDE_TRACKING:=1}"

export YEAR_N_REPLICATES
export YEAR_NULL_VAR_B_VALUES YEAR_NULL_VAR_SEEDS
export YEAR_RANK_STAB_B_VALUES YEAR_RANK_STAB_SEEDS YEAR_RANK_STAB_TOP_K
export TRACK_REFERENCE_SEED TRACK_TOP_N_VALUES INCLUDE_TRACKING

YEAR_DWPC_OUTPUT_DIR="${YEAR_DWPC_OUTPUT_DIR:-output/dwpc_direct/all_GO_positive_growth}"
YEAR_NULL_ANALYSIS_DIR="${YEAR_NULL_ANALYSIS_DIR:-${YEAR_NULL_VAR_OUTPUT_DIR:-output/year_null_variance_exp}/year_null_variance_experiment}"
YEAR_RANK_ANALYSIS_DIR="${YEAR_RANK_ANALYSIS_DIR:-${YEAR_RANK_STAB_OUTPUT_DIR:-output/year_rank_stability_exp}/year_rank_stability_experiment}"

export YEAR_DWPC_OUTPUT_DIR YEAR_NULL_ANALYSIS_DIR YEAR_RANK_ANALYSIS_DIR

submit_sbatch() {
  local dependency="$1"
  shift

  if [[ -n "$dependency" ]]; then
    sbatch --parsable --export=ALL --dependency="afterok:${dependency}" "$@"
  else
    sbatch --parsable --export=ALL "$@"
  fi
}

submit_wrap() {
  local dependency="$1"
  local job_name="$2"
  local cpus="$3"
  local mem="$4"
  local time_limit="$5"
  local command="$6"
  local wrapped

  printf -v wrapped 'cd "%s" && module load anaconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate multi_dwpc && export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" && mkdir -p "$MPLCONFIGDIR" && %s' \
    "$REPO_ROOT" "$command"

  if [[ -n "$dependency" ]]; then
    sbatch \
      --parsable \
      --export=ALL \
      --dependency="afterok:${dependency}" \
      --job-name="$job_name" \
      --qos="$SBATCH_QOS" \
      --partition="$SBATCH_PARTITION" \
      --cpus-per-task="$cpus" \
      --mem="$mem" \
      --time="$time_limit" \
      --output="hpc/logs/year/%x_%j.out" \
      --wrap="bash -c '$wrapped'"
  else
    sbatch \
      --parsable \
      --export=ALL \
      --job-name="$job_name" \
      --qos="$SBATCH_QOS" \
      --partition="$SBATCH_PARTITION" \
      --cpus-per-task="$cpus" \
      --mem="$mem" \
      --time="$time_limit" \
      --output="hpc/logs/year/%x_%j.out" \
      --wrap="bash -c '$wrapped'"
  fi
}

print_job() {
  local label="$1"
  local job_id="$2"
  printf '%-32s %s\n' "$label" "$job_id"
}

require_year_inputs() {
  local required=(
    "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv"
    "output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv"
  )
  local rel
  for rel in "${required[@]}"; do
    if [[ ! -f "$REPO_ROOT/$rel" ]]; then
      echo "Missing required year input: $rel" >&2
      exit 1
    fi
  done
}

controller_dwpc() {
  cd "$REPO_ROOT"
  mkdir -p "$YEAR_DWPC_OUTPUT_DIR"

  local dataset_manifest="$YEAR_DWPC_OUTPUT_DIR/dataset_manifest.txt"
  "$PYTHON_EXE" scripts/compute_dwpc_direct.py --list-datasets > "$dataset_manifest"

  local n_datasets
  n_datasets=$(wc -l < "$dataset_manifest")
  if [[ "$n_datasets" -le 0 ]]; then
    echo "No year datasets found in $dataset_manifest" >&2
    exit 1
  fi

  local dwpc_job analysis_controller_job
  dwpc_job="$(submit_sbatch "$SLURM_JOB_ID" --array=0-$((n_datasets - 1)) hpc/year_dwpc_array.sbatch)"
  analysis_controller_job="$(submit_wrap "$dwpc_job" "year-analysis-controller" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "bash hpc/submit_year_direct_rebuild.sh __controller_analysis__")"

  echo "Year DWPC controller submitted downstream jobs:"
  print_job "year dwpc array" "$dwpc_job"
  print_job "year analysis controller" "$analysis_controller_job"
}

controller_analysis() {
  cd "$REPO_ROOT"

  local year_null_job year_null_plot_job year_rank_job year_rank_plot_job year_seed_plot_job
  year_null_job="$(submit_sbatch "$SLURM_JOB_ID" hpc/year_null_variance.sbatch)"
  year_null_plot_job="$(submit_wrap "$year_null_job" "year-null-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_year_null_variance_results.py --analysis-dir \"$YEAR_NULL_ANALYSIS_DIR\"")"
  year_rank_job="$(submit_sbatch "$SLURM_JOB_ID" hpc/year_rank_stability.sbatch)"
  year_rank_plot_job="$(submit_wrap "$year_rank_job" "year-rank-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_year_rank_stability_results.py --analysis-dir \"$YEAR_RANK_ANALYSIS_DIR\"")"
  year_seed_plot_job="$(submit_wrap "$year_rank_plot_job" "year-seed-scatter" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_year_rank_seed_comparisons.py --analysis-dir \"$YEAR_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\"")"

  echo "Year analysis controller submitted downstream jobs:"
  print_job "year null aggregate" "$year_null_job"
  print_job "year null plots" "$year_null_plot_job"
  print_job "year rank aggregate" "$year_rank_job"
  print_job "year rank plots" "$year_rank_plot_job"
  print_job "year seed scatters" "$year_seed_plot_job"

  if [[ "$INCLUDE_TRACKING" == "1" ]]; then
    local track_n year_track_job
    IFS=',' read -r -a track_values <<< "$TRACK_TOP_N_VALUES"
    for track_n in "${track_values[@]}"; do
      track_n="${track_n//[[:space:]]/}"
      [[ -z "$track_n" ]] && continue
      year_track_job="$(submit_wrap "$year_seed_plot_job" "year-top-tracking-${track_n}" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
        "$PYTHON_EXE scripts/track_year_top_metapaths.py --analysis-dir \"$YEAR_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\" --top-n \"$track_n\"")"
      print_job "year top-${track_n} tracking" "$year_track_job"
    done
  fi
}

submit_main() {
  cd "$REPO_ROOT"
  mkdir -p hpc/logs/year "$YEAR_DWPC_OUTPUT_DIR"
  require_year_inputs

  local metapath_manifest="$YEAR_DWPC_OUTPUT_DIR/metapath_manifest.txt"
  "$PYTHON_EXE" scripts/compute_dwpc_direct.py --list-metapaths > "$metapath_manifest"

  local n_metapaths
  n_metapaths=$(wc -l < "$metapath_manifest")
  if [[ "$n_metapaths" -le 0 ]]; then
    echo "No year metapaths found in $metapath_manifest" >&2
    exit 1
  fi

  echo "Repo root: $REPO_ROOT"
  echo "Year direct workspace: $YEAR_DWPC_OUTPUT_DIR"
  echo "Year replicate count: $YEAR_N_REPLICATES"
  echo "Year null B values: $YEAR_NULL_VAR_B_VALUES"
  echo "Year rank B values: $YEAR_RANK_STAB_B_VALUES"
  echo "Year rank top-k: $YEAR_RANK_STAB_TOP_K"
  echo "Include tracking: $INCLUDE_TRACKING"
  echo "Tracking sets: $TRACK_TOP_N_VALUES"
  echo

  local warmup_job perm_job random_job dwpc_controller_job
  warmup_job="$(submit_sbatch "" --array=0-$((n_metapaths - 1)) hpc/year_dwpc_cache_warmup_array.sbatch)"
  perm_job="$(submit_sbatch "" --array=0-$((YEAR_N_REPLICATES - 1)) hpc/year_permutations_array.sbatch)"
  random_job="$(submit_sbatch "" --array=0-$((YEAR_N_REPLICATES - 1)) hpc/year_random_controls_array.sbatch)"
  dwpc_controller_job="$(submit_wrap "$warmup_job:$perm_job:$random_job" "year-dwpc-controller" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "bash hpc/submit_year_direct_rebuild.sh __controller_dwpc__")"

  print_job "year cache warmup array" "$warmup_job"
  print_job "year permutations array" "$perm_job"
  print_job "year random array" "$random_job"
  print_job "year dwpc controller" "$dwpc_controller_job"
  echo
  echo "Submission complete."
}

case "$MODE" in
  __controller_dwpc__)
    controller_dwpc
    ;;
  __controller_analysis__)
    controller_analysis
    ;;
  submit)
    submit_main
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
