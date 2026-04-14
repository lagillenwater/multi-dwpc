#!/bin/bash
#SBATCH --job-name=year-rank-stab
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%j.out

set -euo pipefail

# Run after year DWPC result files already exist, for example:
#   sbatch hpc/year_rank_stability.sbatch

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="${YEAR_RANK_STAB_RESULTS_DIR:-output/dwpc_direct/all_GO_positive_growth/results}"
OUTPUT_DIR="${YEAR_RANK_STAB_OUTPUT_DIR:-output/year_rank_stability_exp}"
# Null models are located in:
#   - output/permutations (permutation null)
#   - output/random_samples (random null)
# Set YEAR_RANK_STAB_WORKSPACE_DIR to parent directory containing these
WORKSPACE_DIR="${YEAR_RANK_STAB_WORKSPACE_DIR:-output}"
TOP_K="${YEAR_RANK_STAB_TOP_K:-5,10}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

RANK_METRIC="${YEAR_RANK_STAB_RANK_METRIC:-effect_size_d}"

$PYTHON_EXE scripts/year_rank_stability_experiment.py \
  --results-dir "$RESULTS_DIR" \
  --workspace-dir "$WORKSPACE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --b-values "${YEAR_RANK_STAB_B_VALUES:-2,4,6,8,10,20}" \
  --seeds "${YEAR_RANK_STAB_SEEDS:-11,22,33,44,55}" \
  --top-k-metapaths "$TOP_K" \
  --rank-metric "$RANK_METRIC"
