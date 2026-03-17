#!/bin/bash
#SBATCH --job-name=year-dwpc
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%A_%a.out

set -euo pipefail

# Warm the shared cache once across metapaths, then create the dataset manifest and submit the array:
#   mkdir -p output/dwpc_direct/all_GO_positive_growth
#   python scripts/compute_dwpc_direct.py --list-metapaths > output/dwpc_direct/all_GO_positive_growth/metapath_manifest.txt
#   sbatch --array=0-$(($(wc -l < output/dwpc_direct/all_GO_positive_growth/metapath_manifest.txt)-1)) hpc/year_dwpc_cache_warmup_array.sbatch
#   python3 scripts/compute_dwpc_direct.py --list-datasets > output/dwpc_direct/all_GO_positive_growth/dataset_manifest.txt
#   sbatch --array=0-$(($(wc -l < output/dwpc_direct/all_GO_positive_growth/dataset_manifest.txt)-1)) hpc/year_dwpc_array.sbatch
# Optional: export AUTO_AGGREGATE=1 to submit year-null-variance-exp after the array completes.
# This array assumes the shared DWPC cache is already warm.

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${YEAR_DWPC_OUTPUT_DIR:-output/dwpc_direct/all_GO_positive_growth}"
MANIFEST="${DATASET_MANIFEST:-$OUTPUT_DIR/dataset_manifest.txt}"
PYTHON_EXE="${PYTHON_EXE:-python}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

cd "$REPO_ROOT"
module load anaconda
conda activate multi_dwpc

if [[ ! -f "$MANIFEST" ]]; then
  echo "Dataset manifest not found: $MANIFEST" >&2
  exit 1
fi

DATASET_NAME=$(sed -n "$((TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$DATASET_NAME" ]]; then
  echo "No dataset entry for task id ${TASK_ID}" >&2
  exit 1
fi

if [[ "${AUTO_AGGREGATE:-0}" == "1" && "$TASK_ID" == "0" ]]; then
  sbatch \
    --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --job-name="year-null-var-agg" \
    --output="slurm_year-null-var-agg_%j.out" \
    --wrap="bash -lc 'cd \"$REPO_ROOT\" && module load anaconda && conda activate multi_dwpc && YEAR_NULL_VAR_RESULTS_DIR=\"$OUTPUT_DIR/results\" ${PYTHON_EXE} scripts/year_null_variance_experiment.py --results-dir \"$OUTPUT_DIR/results\" --output-dir \"${YEAR_NULL_VAR_OUTPUT_DIR:-output/year_null_variance_exp}\"'"
fi

$PYTHON_EXE scripts/compute_dwpc_direct.py \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --read-only-cache \
  --n-workers "${YEAR_DWPC_N_WORKERS:-2}"
