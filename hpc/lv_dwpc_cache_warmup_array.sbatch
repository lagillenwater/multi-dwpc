#!/bin/bash
#SBATCH --job-name=lv-dwpc-warm-mp
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%A_%a.out

set -euo pipefail

# Build the metapath manifest once, then submit the array:
#   python scripts/lv_prepare_experiment.py --output-dir output/lv_experiment --list-metapaths > output/lv_experiment/metapath_manifest.txt
#   sbatch --array=0-$(($(wc -l < output/lv_experiment/metapath_manifest.txt)-1)) hpc/lv_dwpc_cache_warmup_array.sbatch

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
MANIFEST="${LV_METAPATH_MANIFEST:-$OUTPUT_DIR/metapath_manifest.txt}"
PYTHON_EXE="${PYTHON_EXE:-python}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

cd "$REPO_ROOT"
module load anaconda
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

$PYTHON_EXE scripts/lv_prepare_experiment.py \
  --output-dir "$OUTPUT_DIR" \
  --warmup-metapath "$METAPATH"
