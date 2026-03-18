#!/bin/bash
#SBATCH --job-name=year-rank-stab
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

# Run after year DWPC result files already exist, for example:
#   sbatch hpc/year_rank_stability.sbatch

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RESULTS_DIR="${YEAR_RANK_STAB_RESULTS_DIR:-output/dwpc_direct/all_GO_positive_growth/results}"
OUTPUT_DIR="${YEAR_RANK_STAB_OUTPUT_DIR:-output/year_rank_stability_exp}"
TOP_K="${YEAR_RANK_STAB_TOP_K:-10}"
PYTHON_EXE="${PYTHON_EXE:-python}"

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

$PYTHON_EXE scripts/year_rank_stability_experiment.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --b-values "${YEAR_RANK_STAB_B_VALUES:-1,2,5,10,20}" \
  --seeds "${YEAR_RANK_STAB_SEEDS:-11,22,33,44,55}" \
  --top-k-metapaths "$TOP_K"
