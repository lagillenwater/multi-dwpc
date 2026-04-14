#!/bin/bash
#SBATCH --job-name=year-dwpc-warm-mp
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/year/%x_%A_%a.out

set -euo pipefail

# Build the metapath manifest once, then submit the array:
#   mkdir -p output/dwpc_direct/all_GO_positive_growth
#   python scripts/compute_dwpc_direct.py --list-metapaths > output/dwpc_direct/all_GO_positive_growth/metapath_manifest.txt
#   sbatch --array=0-$(($(wc -l < output/dwpc_direct/all_GO_positive_growth/metapath_manifest.txt)-1)) hpc/year_dwpc_cache_warmup_array.sbatch

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${YEAR_DWPC_OUTPUT_DIR:-output/dwpc_direct/all_GO_positive_growth}"
MANIFEST="${METAPATH_MANIFEST:-$OUTPUT_DIR/metapath_manifest.txt}"
PYTHON_EXE="${PYTHON_EXE:-python}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

if [[ ! -f "$MANIFEST" ]]; then
  echo "Metapath manifest not found: $MANIFEST" >&2
  exit 1
fi

METAPATH=$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$METAPATH" ]]; then
  echo "No metapath entry for task id ${TASK_ID}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

$PYTHON_EXE scripts/compute_dwpc_direct.py \
  --output-dir "$OUTPUT_DIR" \
  --warmup-metapath "$METAPATH" \
  --n-workers "${YEAR_DWPC_WARMUP_N_WORKERS:-2}"
