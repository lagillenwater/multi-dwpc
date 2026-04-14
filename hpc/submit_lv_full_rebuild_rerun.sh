#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python}"
MODE="${1:-submit}"
LV_OUTPUT_BASENAME="${LV_OUTPUT_BASENAME:-}"

SBATCH_QOS="${SBATCH_QOS:-normal}"
SBATCH_PARTITION="${SBATCH_PARTITION:-amilan}"
PLOT_CPUS="${PLOT_CPUS:-2}"
PLOT_MEM="${PLOT_MEM:-8G}"
PLOT_TIME="${PLOT_TIME:-01:00:00}"
POST_CPUS="${POST_CPUS:-4}"
POST_MEM="${POST_MEM:-24G}"
POST_TIME="${POST_TIME:-04:00:00}"

: "${LV_N_REPLICATES:=100}"
: "${LV_NULL_VAR_B_VALUES:=1,2,5,10,20}"
: "${LV_NULL_VAR_SEEDS:=11,22,33,44,55}"
: "${LV_RANK_STAB_B_VALUES:=1,2,5,10,20}"
: "${LV_RANK_STAB_SEEDS:=11,22,33,44,55}"
: "${LV_RANK_STAB_TOP_K:=5,10}"
: "${LV_RANK_TOP_METAPATHS:=5}"
: "${TRACK_REFERENCE_SEED:=11}"
: "${TRACK_TOP_N:=5}"
: "${TRACK_TOP_N_VALUES:=5,10,all}"
: "${INCLUDE_TRACKING:=1}"
: "${INCLUDE_TOP_PATHS:=0}"
: "${INCLUDE_LV_QC:=1}"
: "${INCLUDE_LV_DESCRIPTOR_ANALYSIS:=1}"

export LV_N_REPLICATES
export LV_NULL_VAR_B_VALUES LV_NULL_VAR_SEEDS
export LV_RANK_STAB_B_VALUES LV_RANK_STAB_SEEDS LV_RANK_STAB_TOP_K LV_RANK_TOP_METAPATHS
export TRACK_REFERENCE_SEED TRACK_TOP_N TRACK_TOP_N_VALUES
export INCLUDE_TRACKING INCLUDE_TOP_PATHS INCLUDE_LV_QC INCLUDE_LV_DESCRIPTOR_ANALYSIS

if [[ -n "${LV_WORKSPACE_DIR:-}" && -n "${LV_OUTPUT_DIR:-}" && "$LV_WORKSPACE_DIR" != "$LV_OUTPUT_DIR" ]]; then
  echo "LV_WORKSPACE_DIR and LV_OUTPUT_DIR disagree; set only one or make them match." >&2
  exit 1
fi

if [[ -n "${LV_OUTPUT_DIR:-}" ]]; then
  LV_WORKSPACE_DIR="$LV_OUTPUT_DIR"
elif [[ -n "${LV_WORKSPACE_DIR:-}" ]]; then
  LV_OUTPUT_DIR="$LV_WORKSPACE_DIR"
elif [[ -n "$LV_OUTPUT_BASENAME" ]]; then
  LV_OUTPUT_DIR="output/$LV_OUTPUT_BASENAME"
  LV_WORKSPACE_DIR="$LV_OUTPUT_DIR"
elif [[ "${LV_ALL_METAPATHS:-0}" == "1" ]]; then
  LV_OUTPUT_DIR="output/lv_experiment_all_metapaths"
  LV_WORKSPACE_DIR="$LV_OUTPUT_DIR"
else
  LV_OUTPUT_DIR="output/lv_experiment"
  LV_WORKSPACE_DIR="$LV_OUTPUT_DIR"
fi

export LV_OUTPUT_DIR LV_WORKSPACE_DIR

LV_NULL_ANALYSIS_DIR="${LV_NULL_ANALYSIS_DIR:-${LV_NULL_VAR_ANALYSIS_DIR:-$LV_OUTPUT_DIR/lv_null_variance_experiment}}"
LV_RANK_ANALYSIS_DIR="${LV_RANK_ANALYSIS_DIR:-${LV_RANK_STAB_ANALYSIS_DIR:-$LV_OUTPUT_DIR/lv_rank_stability_experiment}}"
LV_GROUP_QC_OUTPUT_DIR="${LV_GROUP_QC_OUTPUT_DIR:-$LV_OUTPUT_DIR/lv_group_qc_experiment}"

export LV_NULL_ANALYSIS_DIR LV_RANK_ANALYSIS_DIR LV_GROUP_QC_OUTPUT_DIR

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
      --output="hpc/logs/lv/%x_%j.out" \
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
      --output="hpc/logs/lv/%x_%j.out" \
      --wrap="bash -lc '$wrapped'"
  fi
}

print_job() {
  local label="$1"
  local job_id="$2"
  printf '%-32s %s\n' "$label" "$job_id"
}

controller_warmup() {
  cd "$REPO_ROOT"
  mkdir -p "$LV_WORKSPACE_DIR"

  local metapath_manifest="$LV_WORKSPACE_DIR/metapath_manifest.txt"
  "$PYTHON_EXE" scripts/lv_prepare_experiment.py --output-dir "$LV_WORKSPACE_DIR" --list-metapaths > "$metapath_manifest"

  local n_metapaths
  n_metapaths=$(wc -l < "$metapath_manifest")
  if [[ "$n_metapaths" -le 0 ]]; then
    echo "No LV metapaths found in $metapath_manifest" >&2
    exit 1
  fi

  local warmup_job finalize_job perm_job random_job summary_controller_job
  warmup_job="$(submit_sbatch "$SLURM_JOB_ID" --array=0-$((n_metapaths - 1)) hpc/lv_dwpc_cache_warmup_array.sbatch)"
  finalize_job="$(submit_sbatch "$warmup_job" hpc/lv_precompute_finalize.sbatch)"
  perm_job="$(submit_sbatch "$finalize_job" --array=0-$((LV_N_REPLICATES - 1)) hpc/lv_permutations_array.sbatch)"
  random_job="$(submit_sbatch "$finalize_job" --array=0-$((LV_N_REPLICATES - 1)) hpc/lv_random_controls_array.sbatch)"
  summary_controller_job="$(submit_wrap "$perm_job:$random_job" "lv-summary-controller" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "bash hpc/submit_lv_full_rebuild_rerun.sh __controller_summary__")"

  echo "LV full rebuild controller submitted downstream jobs:"
  print_job "lv cache warmup array" "$warmup_job"
  print_job "lv precompute finalize" "$finalize_job"
  print_job "lv permutations array" "$perm_job"
  print_job "lv random array" "$random_job"
  print_job "lv summary controller" "$summary_controller_job"
}

controller_summary() {
  cd "$REPO_ROOT"
  mkdir -p "$LV_WORKSPACE_DIR"

  local artifact_manifest="$LV_WORKSPACE_DIR/artifact_manifest.txt"
  "$PYTHON_EXE" scripts/lv_compute_replicate_summaries.py --output-dir "$LV_WORKSPACE_DIR" --list-artifacts > "$artifact_manifest"

  local n_artifacts
  n_artifacts=$(wc -l < "$artifact_manifest")
  if [[ "$n_artifacts" -le 0 ]]; then
    echo "No LV artifacts found in $artifact_manifest" >&2
    exit 1
  fi

  local summary_job lv_null_job lv_null_plot_job lv_rank_job lv_rank_plot_job lv_seed_plot_job
  summary_job="$(submit_sbatch "$SLURM_JOB_ID" --array=0-$((n_artifacts - 1)) hpc/lv_summary_array.sbatch)"
  lv_null_job="$(submit_sbatch "$summary_job" hpc/lv_null_variance_aggregate.sbatch)"
  lv_null_plot_job="$(submit_wrap "$lv_null_job" "lv-null-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_lv_null_variance_results.py --analysis-dir \"$LV_NULL_ANALYSIS_DIR\"")"
  lv_rank_job="$(submit_sbatch "$summary_job" hpc/lv_rank_stability_aggregate.sbatch)"
  lv_rank_plot_job="$(submit_wrap "$lv_rank_job" "lv-rank-plots" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_lv_rank_stability_results.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\"")"
  lv_seed_plot_job="$(submit_wrap "$lv_rank_plot_job" "lv-seed-scatter" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "$PYTHON_EXE scripts/plot_lv_rank_seed_comparisons.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\"")"

  echo "LV summary controller submitted downstream jobs:"
  print_job "lv summary array" "$summary_job"
  print_job "lv null aggregate" "$lv_null_job"
  print_job "lv null plots" "$lv_null_plot_job"
  print_job "lv rank aggregate" "$lv_rank_job"
  print_job "lv rank plots" "$lv_rank_plot_job"
  print_job "lv seed scatters" "$lv_seed_plot_job"

  if [[ "$INCLUDE_LV_QC" == "1" ]]; then
    local lv_qc_job lv_qc_plot_job lv_score_gap_plot_job lv_descriptor_job
    lv_qc_job="$(submit_sbatch "$lv_rank_job" hpc/lv_group_qc_aggregate.sbatch)"
    lv_qc_plot_job="$(submit_wrap "$lv_qc_job" "lvqc-dashboard" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
      "$PYTHON_EXE scripts/plot_lv_group_qc_results.py --qc-dir \"$LV_GROUP_QC_OUTPUT_DIR\" --max-b 20 --diagnostic-b \"${LV_GROUP_QC_DIAGNOSTIC_B:-5}\"")"
    lv_score_gap_plot_job="$(submit_wrap "$lv_qc_job" "lvqc-score-separation" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
      "$PYTHON_EXE scripts/plot_lv_score_separation.py --qc-dir \"$LV_GROUP_QC_OUTPUT_DIR\" --max-rank \"${LV_GROUP_QC_SCORE_GAP_MAX_K:-20}\"")"
    print_job "lv group qc aggregate" "$lv_qc_job"
    print_job "lvqc dashboard" "$lv_qc_plot_job"
    print_job "lvqc score separation" "$lv_score_gap_plot_job"

    if [[ "$INCLUDE_LV_DESCRIPTOR_ANALYSIS" == "1" ]]; then
      lv_descriptor_job="$(submit_wrap "$lv_qc_job" "lvqc-descriptor-analysis" "$POST_CPUS" "$POST_MEM" "$POST_TIME" \
        "$PYTHON_EXE scripts/lv_descriptor_outcome_analysis.py --qc-dir \"$LV_GROUP_QC_OUTPUT_DIR\" --workspace-dir \"$LV_WORKSPACE_DIR\" --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --snapshot-b \"${LV_GROUP_QC_GAP_B:-5}\"")"
      print_job "lvqc predictor analysis" "$lv_descriptor_job"
    fi
  fi

  if [[ "$INCLUDE_TRACKING" == "1" ]]; then
    local track_n lv_track_job
    IFS=',' read -r -a track_values <<< "$TRACK_TOP_N_VALUES"
    for track_n in "${track_values[@]}"; do
      track_n="${track_n//[[:space:]]/}"
      [[ -z "$track_n" ]] && continue
      lv_track_job="$(submit_wrap "$lv_seed_plot_job" "lv-top-tracking-${track_n}" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
        "$PYTHON_EXE scripts/track_lv_top_metapaths.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --reference-seed \"$TRACK_REFERENCE_SEED\" --top-n \"$track_n\"")"
      print_job "lv top-${track_n} tracking" "$lv_track_job"
    done
  fi

  if [[ "$INCLUDE_TOP_PATHS" == "1" ]]; then
    local lv_top_paths_job lv_rank_paths_job
    lv_top_paths_job="$(submit_wrap "$lv_null_plot_job" "lv-top-paths" "$POST_CPUS" "$POST_MEM" "$POST_TIME" \
      "$PYTHON_EXE scripts/identify_lv_top_paths.py --output-dir \"$LV_WORKSPACE_DIR\" --top-metapaths \"$TRACK_TOP_N\" --top-pairs 10 --top-paths 5")"
    lv_rank_paths_job="$(submit_wrap "$lv_seed_plot_job" "lv-rank-top-paths" "$POST_CPUS" "$POST_MEM" "$POST_TIME" \
      "$PYTHON_EXE scripts/identify_lv_rank_top_paths.py --analysis-dir \"$LV_RANK_ANALYSIS_DIR\" --workspace-dir \"$LV_WORKSPACE_DIR\" --top-metapaths-per-run \"$TRACK_TOP_N\" --top-pairs 10 --top-paths 5")"
    print_job "lv top paths" "$lv_top_paths_job"
    print_job "lv rank top paths" "$lv_rank_paths_job"
  fi
}

submit_main() {
  cd "$REPO_ROOT"
  mkdir -p hpc/logs/lv "$LV_WORKSPACE_DIR"

  echo "Repo root: $REPO_ROOT"
  echo "LV workspace: $LV_WORKSPACE_DIR"
  echo "LV all metapaths: ${LV_ALL_METAPATHS:-0}"
  echo "LV metapath limit: ${LV_METAPATH_LIMIT:-2}"
  echo "LV replicate count: $LV_N_REPLICATES"
  echo "Include LVQC: $INCLUDE_LV_QC"
  echo "Include LV descriptor analysis: $INCLUDE_LV_DESCRIPTOR_ANALYSIS"
  echo "Include tracking: $INCLUDE_TRACKING"
  echo "Tracking sets: $TRACK_TOP_N_VALUES"
  echo "Include top paths: $INCLUDE_TOP_PATHS"
  echo

  local prepare_job warmup_controller_job
  prepare_job="$(submit_sbatch "" hpc/lv_prepare.sbatch)"
  warmup_controller_job="$(submit_wrap "$prepare_job" "lv-warmup-controller" "$PLOT_CPUS" "$PLOT_MEM" "$PLOT_TIME" \
    "bash hpc/submit_lv_full_rebuild_rerun.sh __controller_warmup__")"

  print_job "lv prepare metadata" "$prepare_job"
  print_job "lv warmup controller" "$warmup_controller_job"
  echo
  echo "Submission complete."
}

case "$MODE" in
  __controller_warmup__)
    controller_warmup
    ;;
  __controller_summary__)
    controller_summary
    ;;
  submit)
    submit_main
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac
