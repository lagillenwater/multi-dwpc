#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"

RUN_YEAR="${RUN_YEAR:-1}"
RUN_LV="${RUN_LV:-1}"
INCLUDE_TRACKING="${INCLUDE_TRACKING:-1}"
INCLUDE_TOP_PATHS="${INCLUDE_TOP_PATHS:-0}"

SBATCH_QOS="${SBATCH_QOS:-normal}"
SBATCH_PARTITION="${SBATCH_PARTITION:-amilan}"
PLOT_CPUS="${PLOT_CPUS:-2}"
PLOT_MEM="${PLOT_MEM:-8G}"
PLOT_TIME="${PLOT_TIME:-01:00:00}"
POST_CPUS="${POST_CPUS:-4}"
POST_MEM="${POST_MEM:-24G}"
POST_TIME="${POST_TIME:-04:00:00}"

: "${YEAR_NULL_VAR_B_VALUES:=1,2,5,10,20}"
: "${YEAR_NULL_VAR_SEEDS:=11,22,33,44,55}"
: "${YEAR_RANK_STAB_B_VALUES:=1,2,5,10,20}"
: "${YEAR_RANK_STAB_SEEDS:=11,22,33,44,55}"
: "${YEAR_RANK_STAB_TOP_K:=5,10}"
: "${LV_NULL_VAR_B_VALUES:=2,5,10,20,30,40}"
: "${LV_NULL_VAR_SEEDS:=11,22,33,44,55}"
: "${LV_RANK_STAB_B_VALUES:=2,5,10,20,30,40}"
: "${LV_RANK_STAB_SEEDS:=11,22,33,44,55}"
: "${LV_RANK_STAB_TOP_K:=5,10}"
: "${LV_RANK_TOP_METAPATHS:=5}"
: "${TRACK_REFERENCE_SEED:=11}"
: "${TRACK_TOP_N:=5}"
: "${TRACK_TOP_N_VALUES:=5,10,all}"

export YEAR_NULL_VAR_B_VALUES YEAR_NULL_VAR_SEEDS
export YEAR_RANK_STAB_B_VALUES YEAR_RANK_STAB_SEEDS YEAR_RANK_STAB_TOP_K
export LV_NULL_VAR_B_VALUES LV_NULL_VAR_SEEDS
export LV_RANK_STAB_B_VALUES LV_RANK_STAB_SEEDS LV_RANK_STAB_TOP_K LV_RANK_TOP_METAPATHS
export TRACK_REFERENCE_SEED TRACK_TOP_N TRACK_TOP_N_VALUES

cd "$REPO_ROOT"

YEAR_NULL_ANALYSIS_DIR="${YEAR_NULL_ANALYSIS_DIR:-output/year_null_variance_exp/year_null_variance_experiment}"
YEAR_RANK_ANALYSIS_DIR="${YEAR_RANK_ANALYSIS_DIR:-output/year_rank_stability_exp/year_rank_stability_experiment}"
LV_NULL_ANALYSIS_DIR="${LV_NULL_ANALYSIS_DIR:-output/lv_experiment/lv_null_variance_experiment}"
LV_RANK_ANALYSIS_DIR="${LV_RANK_ANALYSIS_DIR:-output/lv_experiment/lv_rank_stability_experiment}"
LV_WORKSPACE_DIR="${LV_WORKSPACE_DIR:-output/lv_experiment}"

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

  printf -v wrapped 'cd "%s" && module load anaconda && conda activate multi_dwpc && export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}" && mkdir -p "$MPLCONFIGDIR" && %s' \
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
      --output="slurm_%x_%j.out" \
      --wrap="bash -lc '$wrapped'"
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
      --output="slurm_%x_%j.out" \
      --wrap="bash -lc '$wrapped'"
  fi
}

print_job() {
  local label="$1"
  local job_id="$2"
  printf '%-28s %s\n' "$label" "$job_id"
}

echo "Repo root: $REPO_ROOT"
echo "Run year chain: $RUN_YEAR"
echo "Run LV chain: $RUN_LV"
echo "Include tracking: $INCLUDE_TRACKING"
echo "Include top paths: $INCLUDE_TOP_PATHS"
echo

if [[ "$RUN_YEAR" == "1" ]]; then
  year_null_job="$(submit_sbatch "" hpc/year_null_variance.sbatch)"
  year_null_plot_job="$(submit_wrap "$year_null_job" "year-null-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_year_null_variance_results.py --analysis-dir \"$YEAR_NULL_ANALYSIS_DIR\"")"

  year_rank_job="$(submit_sbatch "" hpc/year_rank_stability.sbatch)"
  year_rank_plot_job="$(submit_wrap "$year_rank_job" "year-rank-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_year_rank_stability_results.py --analysis-dir \"$YEAR_RANK_ANALYSIS_DIR\"")"
  year_seed_plot_job="$(submit_wrap "$year_rank_plot_job" "year-seed-scatter" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_year_rank_seed_comparisons.py --analysis-dir \"$YEAR_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\"")"

  print_job "year null aggregate" "$year_null_job"
  print_job "year null plots" "$year_null_plot_job"
  print_job "year rank aggregate" "$year_rank_job"
  print_job "year rank plots" "$year_rank_plot_job"
  print_job "year seed scatters" "$year_seed_plot_job"

  if [[ "$INCLUDE_TRACKING" == "1" ]]; then
    IFS=',' read -r -a year_track_values <<< "$TRACK_TOP_N_VALUES"
    for track_n in "${year_track_values[@]}"; do
      track_n="${track_n//[[:space:]]/}"
      [[ -z "$track_n" ]] && continue
      year_track_job="$(submit_wrap "$year_seed_plot_job" "year-top-tracking-${track_n}" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
        "$PYTHON_EXE scripts/track_year_top_metapaths.py --analysis-dir \"$YEAR_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\" --top-n \"$track_n\"")"
      print_job "year top-${track_n} tracking" "$year_track_job"
    done
  fi

  if [[ "$INCLUDE_TOP_PATHS" == "1" ]]; then
    year_paths_job="$(submit_wrap "$year_seed_plot_job" "year-top-paths" "$POST_CPUS" "$POST_MEM" "$POST_TIME" \
      "$PYTHON_EXE scripts/identify_year_top_paths.py --years 2016 2024 --top-n \"$TRACK_TOP_N\" --top-metapaths \"$TRACK_TOP_N\" --pair-top-n 10 --top-paths 5 --plot-path-instances")"
    print_job "year top paths" "$year_paths_job"
  fi

  echo
fi

if [[ "$RUN_LV" == "1" ]]; then
  lv_null_job="$(submit_sbatch "" hpc/lv_null_variance_aggregate.sbatch)"
  lv_null_plot_job="$(submit_wrap "$lv_null_job" "lv-null-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_lv_null_variance_results.py --analysis-dir \"$LV_NULL_ANALYSIS_DIR\"")"

  lv_rank_job="$(submit_sbatch "" hpc/lv_rank_stability_aggregate.sbatch)"
  lv_rank_plot_job="$(submit_wrap "$lv_rank_job" "lv-rank-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_lv_rank_stability_results.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\"")"
  lv_seed_plot_job="$(submit_wrap "$lv_rank_plot_job" "lv-seed-scatter" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_lv_rank_seed_comparisons.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\"")"

  print_job "LV null aggregate" "$lv_null_job"
  print_job "LV null plots" "$lv_null_plot_job"
  print_job "LV rank aggregate" "$lv_rank_job"
  print_job "LV rank plots" "$lv_rank_plot_job"
  print_job "LV seed scatters" "$lv_seed_plot_job"

  if [[ "$INCLUDE_TRACKING" == "1" ]]; then
    IFS=',' read -r -a lv_track_values <<< "$TRACK_TOP_N_VALUES"
    for track_n in "${lv_track_values[@]}"; do
      track_n="${track_n//[[:space:]]/}"
      [[ -z "$track_n" ]] && continue
      lv_track_job="$(submit_wrap "$lv_seed_plot_job" "lv-top-tracking-${track_n}" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
        "$PYTHON_EXE scripts/track_lv_top_metapaths.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\" --top-n \"$track_n\"")"
      print_job "LV top-${track_n} tracking" "$lv_track_job"
    done
  fi

  if [[ "$INCLUDE_TOP_PATHS" == "1" ]]; then
    lv_top_paths_job="$(submit_wrap "$lv_null_plot_job" "lv-top-paths" "$POST_CPUS" "$POST_MEM" "$POST_TIME" \
      "$PYTHON_EXE scripts/identify_lv_top_paths.py --output-dir \"$LV_WORKSPACE_DIR\" --top-metapaths \"$TRACK_TOP_N\" --top-pairs 10 --top-paths 5")"
    lv_rank_paths_job="$(submit_wrap "$lv_seed_plot_job" "lv-rank-top-paths" "$POST_CPUS" "$POST_MEM" "$POST_TIME" \
      "$PYTHON_EXE scripts/identify_lv_rank_top_paths.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --workspace-dir \"$LV_WORKSPACE_DIR\" --top-metapaths-per-run \"$TRACK_TOP_N\" --top-pairs 10 --top-paths 5")"
    print_job "LV top paths" "$lv_top_paths_job"
    print_job "LV rank top paths" "$lv_rank_paths_job"
  fi

  echo
fi

echo "Submission complete."
