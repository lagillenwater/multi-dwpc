#!/bin/bash
#SBATCH --job-name=lv-summary
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# Note: --output is set by submit script to organize logs by array job ID

set -euo pipefail

# Create manifest before submission:
#   python scripts/lv_compute_replicate_summaries.py --output-dir output/lv_experiment --list-artifacts > output/lv_experiment/artifact_manifest.txt
#   sbatch --array=0-$(($(wc -l < output/lv_experiment/artifact_manifest.txt)-1)) hpc/lv_summary_array.sbatch

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${LV_OUTPUT_DIR:-output/lv_experiment}"
MANIFEST="${LV_ARTIFACT_MANIFEST:-$OUTPUT_DIR/artifact_manifest.txt}"
PYTHON_EXE="${PYTHON_EXE:-python}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

cd "$REPO_ROOT"
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

if [[ ! -f "$MANIFEST" ]]; then
  echo "Artifact manifest not found: $MANIFEST" >&2
  exit 1
fi

ARTIFACT_NAME=$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$ARTIFACT_NAME" ]]; then
  echo "No artifact entry for task id ${TASK_ID}" >&2
  exit 1
fi

$PYTHON_EXE scripts/lv_compute_replicate_summaries.py \
  --output-dir "$OUTPUT_DIR" \
  --artifact-name "$ARTIFACT_NAME"
