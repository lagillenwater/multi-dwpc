#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LV_ALL_METAPATHS="${LV_ALL_METAPATHS:-1}"
export LV_OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment_all_metapaths}"
export LV_N_REPLICATES="${LV_N_REPLICATES:-100}"
export LV_NULL_VAR_B_VALUES="${LV_NULL_VAR_B_VALUES:-1,2,5,10,20}"
export LV_RANK_STAB_B_VALUES="${LV_RANK_STAB_B_VALUES:-1,2,5,10,20}"
export LV_RANK_STAB_TOP_K="${LV_RANK_STAB_TOP_K:-5,10}"
export LV_GROUP_QC_GAP_B="${LV_GROUP_QC_GAP_B:-5}"
export LV_GROUP_QC_SCORE_GAP_MAX_K="${LV_GROUP_QC_SCORE_GAP_MAX_K:-20}"
export LV_GROUP_QC_DIAGNOSTIC_B="${LV_GROUP_QC_DIAGNOSTIC_B:-5}"
export INCLUDE_LV_QC="${INCLUDE_LV_QC:-1}"
export INCLUDE_LV_DESCRIPTOR_ANALYSIS="${INCLUDE_LV_DESCRIPTOR_ANALYSIS:-1}"

cd "$REPO_ROOT"
bash hpc/submit_lv_full_rebuild_rerun.sh submit "$@"
