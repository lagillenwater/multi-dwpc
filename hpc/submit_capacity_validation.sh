#!/bin/bash
# Two-stage Alpine campaign for the capacity-hurdle-adaptive-null
# validation (docs/tasks/capacity-hurdle-adaptive-null/). Stage 1 prewarms
# the DWPC cache for exactly the metapaths the stratified subsample uses;
# stage 2 (afterok) runs the four-strategy sweep, both controls, the
# runtime benchmark, and the comparison report.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
ACCOUNT="${ACCOUNT:-amc-general}"
PARTITION="${PARTITION:-acpu}"
QOS="${QOS:-cpu-normal}"
SUBSTRATE="${SUBSTRATE:-output/end_to_end_2026_4_23/lv_experiment (1)}"
OUT="output/tier0_capacity_hurdle"
ACTIVATE='source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh && conda activate multi_dwpc'
mkdir -p hpc/logs "$OUT"

# Stage 1: derive the subsample's metapath list, prewarm only those.
PREWARM_CMD="$ACTIVATE && python scripts/experiments/tier0_list_subsample_metapaths.py \
    --substrate-dir \"$SUBSTRATE\" --out $OUT/prewarm_metapaths.txt \
  && while read -r mp; do
       python scripts/prewarm_dwpc_cache.py --single-metapath \"\$mp\"
     done < $OUT/prewarm_metapaths.txt"

PREWARM_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
  --job-name=cap-prewarm --time=04:00:00 --mem=32G --cpus-per-task=4 \
  --output=hpc/logs/cap-prewarm-%j.out \
  --wrap="bash -c 'cd \"$REPO_ROOT\" && set -e && $PREWARM_CMD'")
echo "prewarm job: $PREWARM_ID"

# Stage 2: the validation proper.
VALIDATE_CMD="$ACTIVATE && set -e
for s in promiscuity metaedge_degree capacity_hurdle_adaptive metaedge_degree_hurdle_adaptive; do
  python scripts/experiments/tier0_b_convergence.py \
    --substrate-dir \"$SUBSTRATE\" --strategy \"\$s\" --output-dir $OUT/b_convergence
done
for s in capacity_hurdle_adaptive metaedge_degree_hurdle_adaptive; do
  python scripts/experiments/tier0_capacity_controls.py \
    --substrate-dir \"$SUBSTRATE\" --strategy \"\$s\" --output-dir $OUT
done
python scripts/experiments/tier0_runtime_benchmark.py \
  --substrate-dir \"$SUBSTRATE\" --strategy capacity_hurdle_adaptive --output-dir $OUT
python scripts/experiments/tier0_b_comparison.py \
  --convergence-dir $OUT/b_convergence --output-dir $OUT"

VALIDATE_ID=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
  --job-name=cap-validate --time=02:00:00 --mem=32G --cpus-per-task=2 \
  --dependency=afterok:"$PREWARM_ID" \
  --output=hpc/logs/cap-validate-%j.out \
  --wrap="bash -c 'cd \"$REPO_ROOT\" && $VALIDATE_CMD'")
echo "validation job: $VALIDATE_ID (afterok:$PREWARM_ID)"
