# Variance And Rank Experiment Runbook

This runbook covers the unified null-replicate workflow used by both the LV and year experiments.

## Shared experiment shape

Both domains now follow the same pattern:

1. prepare shared inputs
2. generate explicit null replicate artifacts
3. compute one metapath-summary artifact per replicate
4. run variance analysis
5. run rank-stability analysis

The difference is the underlying bipartite graph:

- LV: `LV x selected gene`
- year: `GO term x gene`

## Null hypotheses

### Permuted control

`permuted` is the primary null in both domains.

It uses degree-preserving bipartite permutations, so it preserves:

- source degree exactly
- target degree exactly
- total edge count exactly
- binary edge structure with no duplicate source-target edges

Interpretation:

- conditional on the observed degree sequences, the specific edge configuration is arbitrary with respect to downstream metapath signal

### Random control

`random` is the matched synthetic control in both domains.

It preserves:

- source set size exactly
- target exclusion from the real source set
- approximate target promiscuity matching

Promiscuity means:

- year: number of GO terms containing the gene
- LV: number of LV top-gene sets containing the gene

## Prerequisites

### Environment

```bash
conda activate multi_dwpc
```

HPC batch scripts already run:

```bash
module load anaconda
conda activate multi_dwpc
```

### Year inputs

```bash
poe load-data
poe filter-change
poe filter-jaccard
```

### LV inputs

Download the LV loading matrix once:

```bash
poe download-lv-loadings
export LV_LOADINGS_PATH=data/lv_loadings/multiplier_model_z.tsv.gz
```

## Local workflow

### Year

#### Step 1: generate explicit null replicate artifacts

```bash
N_PERMUTATIONS=20 poe gen-permutation
N_RANDOM_SAMPLES=20 poe gen-random
```

This writes:

- `output/permutations/all_GO_positive_growth_2016/perm_###.csv`
- `output/permutations/all_GO_positive_growth_2024/perm_###.csv`
- `output/random_samples/all_GO_positive_growth_2016/random_###.csv`
- `output/random_samples/all_GO_positive_growth_2024/random_###.csv`

#### Step 2: warm the shared DWPC cache

```bash
poe warmup-dwpc-cache
```

#### Step 3: compute per-replicate DWPC outputs and metapath summaries

```bash
poe compute-dwpc-direct
```

This now writes both pair-level DWPC files and per-replicate summary files:

- `output/dwpc_direct/all_GO_positive_growth/results/dwpc_*.csv`
- `output/dwpc_direct/all_GO_positive_growth/replicate_summaries/summary_*.csv`
- `output/dwpc_direct/all_GO_positive_growth/replicate_manifest.csv`

#### Step 4: variance

```bash
poe year-null-variance-exp
```

Optional post-processing plots from the saved variance CSVs:

```bash
poe year-null-variance-plots
```

#### Step 5: rank stability

```bash
poe year-rank-stability-exp
```

Optional post-processing plots from the saved rank-stability CSVs:

```bash
poe year-rank-stability-plots
poe identify-year-top-paths
```

Use `20` null replicates by default for both year and LV so the workflows stay parallel. Increase that explicitly for final production runs if you need tighter stability estimates.

### LV

#### Step 1: prepare

```bash
poe lv-prepare-exp
```

This creates the shared LV workspace in `output/lv_experiment/`, including:

- `lv_top_genes.csv`
- `lv_target_map.csv`
- `feature_manifest.csv`
- `gene_feature_scores.npy`
- `real_feature_scores.csv`
- `replicate_artifacts/lv_real.csv`
- `replicate_manifest.csv`

For all metapaths:

```bash
LV_OUTPUT_DIR=output/lv_experiment_all_metapaths poe lv-prepare-exp-all-metapaths
```

#### Step 2: generate null replicate artifacts

```bash
LV_N_REPLICATES=100 poe lv-gen-permutation
LV_N_REPLICATES=100 poe lv-gen-random
```

This writes explicit LV null edge lists under:

- `output/lv_experiment/replicate_artifacts/lv_permuted_###.csv`
- `output/lv_experiment/replicate_artifacts/lv_random_###.csv`

#### Step 3: compute per-replicate metapath summaries

```bash
poe lv-compute-replicate-summaries
```

This writes one summary file per artifact under:

- `output/lv_experiment/replicate_summaries/summary_lv_real.csv`
- `output/lv_experiment/replicate_summaries/summary_lv_permuted_###.csv`
- `output/lv_experiment/replicate_summaries/summary_lv_random_###.csv`

#### Step 4: variance

```bash
poe lv-null-variance-exp
```

Optional post-processing plots from the saved variance CSVs:

```bash
poe lv-null-variance-plots
```

This keeps the original LV analysis semantics:

- for each `B`
- for each seed
- choose `B` explicit null replicate summaries without replacement
- average them to form the null mean
- compute `diff = real - null_mean`
- estimate variance across seeds at fixed `B`

#### Step 5: rank stability

```bash
poe lv-rank-stability-exp
```

Optional post-processing plots and rank/path exports:

```bash
poe lv-rank-stability-plots
poe lv-rank-seed-comparisons
poe lv-rank-top-paths
poe identify-lv-top-paths
```

This ranks metapaths by `diff` for each `(control, B, seed, LV, target set)` and compares rankings across seeds.

## HPC workflow

### Year

#### Step 1: generate controls

```bash
sbatch --array=0-19 hpc/year_permutations_array.sbatch
sbatch --array=0-19 hpc/year_random_controls_array.sbatch
```

#### Step 2: warm the shared DWPC cache across metapaths

```bash
mkdir -p output/dwpc_direct/all_GO_positive_growth
python scripts/compute_dwpc_direct.py --list-metapaths > output/dwpc_direct/all_GO_positive_growth/metapath_manifest.txt
sbatch --array=0-$(($(wc -l < output/dwpc_direct/all_GO_positive_growth/metapath_manifest.txt)-1)) hpc/year_dwpc_cache_warmup_array.sbatch
```

#### Step 3: compute per-replicate DWPC and summary artifacts

```bash
mkdir -p output/dwpc_direct/all_GO_positive_growth
python scripts/compute_dwpc_direct.py --list-datasets > output/dwpc_direct/all_GO_positive_growth/dataset_manifest.txt
sbatch --array=0-$(($(wc -l < output/dwpc_direct/all_GO_positive_growth/dataset_manifest.txt)-1)) hpc/year_dwpc_array.sbatch
```

Notes:

- the metapath warmup array builds the shared `data/dwpc_cache/` one metapath at a time
- the dataset array assumes that cache already exists
- array tasks use read-only cache access and conservative worker defaults

#### Step 4: aggregate variance and rank stability

```bash
sbatch hpc/year_null_variance.sbatch
sbatch hpc/year_rank_stability.sbatch
```

With explicit B/seed settings and top-5 focus:

```bash
export YEAR_NULL_VAR_B_VALUES=1,2,5,10,20
export YEAR_NULL_VAR_SEEDS=11,22,33,44,55
export YEAR_RANK_STAB_B_VALUES=1,2,5,10,20
export YEAR_RANK_STAB_SEEDS=11,22,33,44,55
export YEAR_RANK_STAB_TOP_K=5
sbatch hpc/year_null_variance.sbatch
sbatch hpc/year_rank_stability.sbatch
```

Current year direct aggregate semantics:

- per-replicate summaries use `mean_score`
- for the direct year DWPC path, `mean_score` is mean DWPC over `(GO term, metapath)`
- variance and rank-stability analyses consume that shared summary contract directly

#### Automated year direct rebuild

```bash
bash hpc/submit_year_direct_rebuild.sh
```

This wrapper runs the year direct chain end-to-end with dependencies:

- shared DWPC cache warmup by metapath
- explicit permutation and random null generation
- per-dataset direct DWPC and summary generation
- aggregate year null-variance analysis
- aggregate year rank-stability analysis
- downstream year plots and optional tracking jobs

Useful overrides:

```bash
export YEAR_N_REPLICATES=20
export YEAR_NULL_VAR_B_VALUES=1,2,5,10,20
export YEAR_RANK_STAB_B_VALUES=1,2,5,10,20
export YEAR_RANK_STAB_TOP_K=5,10
export INCLUDE_TRACKING=1
bash hpc/submit_year_direct_rebuild.sh
```

#### Adaptive year top-subgraph extraction

This runs on top of an existing year direct-DWPC workspace. It does not rerun the DWPC arrays.

```bash
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multi_dwpc

export YEAR_WORKSPACE_DIR=output/dwpc_direct/all_GO_positive_growth
export YEAR_GO_SUPPORT_PATH=output/year_direct_go_term_support.csv
export YEAR_GLOBAL_SUPPORT_PATH=output/year_direct_global_metapath_support.csv
export YEAR_TOP_PATHS_OUTPUT_DIR=output/metapath_analysis/top_paths_adaptive
export YEAR_GO_EFFECTIVE_MIN_N=1
export YEAR_PAIR_MIN_N=1
export YEAR_PATH_MIN_COUNT=1
export YEAR_PATH_ENUMERATION_TOP_K=5000

sbatch hpc/year_top_subgraphs_adaptive.sbatch
```

This workflow:

- builds a GO-term-level support table from the existing null replicate summaries
- applies FDR across metapaths within each GO term
- derives a global metapath support table from the GO-term support results with a second metapath-level FDR pass
- extracts year top subgraphs from GO-term-supported metapaths rather than from global metapath averages
- uses an effective-number rule to retain metapaths within each GO term, then gene-GO pairs within each retained `(GO term, metapath)`, then explicit path instances within each retained pair

Outputs are written under `output/metapath_analysis/top_paths_adaptive/` by default.

Important note on path enumeration:

- `YEAR_PATH_ENUMERATION_TOP_K` is a computational search-budget cap, not an inferential threshold
- the effective-number rule determines how many path instances are retained from the enumerated candidates
- the default value `5000` is an engineering safeguard rather than a statistically justified cutoff
- if path-level conclusions matter, validate stability by increasing the cap, for example `1000`, `5000`, and `10000`

### LV

#### Step 1: prepare metadata

```bash
export LV_LOADINGS_PATH=data/lv_loadings/multiplier_model_z.tsv.gz
sbatch hpc/lv_prepare.sbatch
```

#### Step 2: warm the shared DWPC cache across metapaths

```bash
mkdir -p output/lv_experiment
python scripts/lv_prepare_experiment.py --output-dir output/lv_experiment --list-metapaths > output/lv_experiment/metapath_manifest.txt
sbatch --array=0-$(($(wc -l < output/lv_experiment/metapath_manifest.txt)-1)) hpc/lv_dwpc_cache_warmup_array.sbatch
```

#### Step 3: finalize LV precompute outputs

```bash
sbatch hpc/lv_precompute_finalize.sbatch
```

Notes:

- the metapath warmup array builds the shared `data/dwpc_cache/` one metapath at a time for the LV feature set
- the finalize step computes `gene_feature_scores.npy` and `real_feature_scores.csv` using the warmed cache

#### Step 4: generate controls

```bash
sbatch --array=0-99 hpc/lv_permutations_array.sbatch
sbatch --array=0-99 hpc/lv_random_controls_array.sbatch
```

#### Step 5: summarize artifacts

Build the artifact manifest once:

```bash
python scripts/lv_compute_replicate_summaries.py --output-dir output/lv_experiment --list-artifacts > output/lv_experiment/artifact_manifest.txt
```

Then submit the summary array:

```bash
sbatch --array=0-$(($(wc -l < output/lv_experiment/artifact_manifest.txt)-1)) hpc/lv_summary_array.sbatch
```

#### Step 6: aggregate variance and rank stability

```bash
sbatch hpc/lv_null_variance_aggregate.sbatch
sbatch hpc/lv_rank_stability_aggregate.sbatch
```

With top-5 rank-stability focus:

```bash
export LV_RANK_STAB_TOP_K=5
sbatch hpc/lv_null_variance_aggregate.sbatch
sbatch hpc/lv_rank_stability_aggregate.sbatch
```

#### Step 7: build exploratory LVQC packets

After the LV rank-stability aggregate finishes, build the dual-null QC packet:

```bash
sbatch hpc/lv_group_qc_aggregate.sbatch
```

The QC packet is descriptive only. It summarizes within-null seed stability, random-vs-permuted agreement, score separation, and downstream exploratory diagnostics.

LV Slurm logs are written under:

- `hpc/logs/`

Useful overrides:

```bash
export LV_GROUP_QC_GAP_B=5
export LV_GROUP_QC_SCORE_GAP_MAX_K=20
export LV_GROUP_QC_DIAGNOSTIC_B=5
export LV_RANDOM_PROM_TOL=2
sbatch hpc/lv_group_qc_aggregate.sbatch
```

#### Automated LV homeostasis rebuild

```bash
bash hpc/submit_lv_homeostasis_rebuild.sh
```

This wrapper runs the full LV chain end-to-end with:

- `LV603 -> GO:0001780 (neutrophil homeostasis)`
- `LV_N_REPLICATES=100`
- `LV_NULL_VAR_B_VALUES=1,2,5,10,20`
- `LV_RANK_STAB_B_VALUES=1,2,5,10,20`
- outputs written under `output/lv_experiment_all_metapaths/`
- exploratory LVQC outputs
- score-separation plots
- descriptor/outcome analysis and predictor-vs-QC plots

Post-processing after aggregates finish:

```bash
poe year-null-variance-plots
poe year-rank-stability-plots
poe lv-null-variance-plots
poe lv-rank-stability-plots
poe lv-rank-seed-comparisons
poe lv-group-qc-plots
poe lv-score-separation-plots
poe lv-descriptor-outcome-analysis
```

Track the same top 5 metapaths across seeds and B using seed 11 at max B:

```bash
poe track-year-top-metapaths
poe track-lv-top-metapaths
```

## Output locations

### LV shared workspace

- `output/lv_experiment/replicate_manifest.csv`
- `output/lv_experiment/replicate_artifacts/`
- `output/lv_experiment/replicate_summaries/`

### LV variance

- `output/lv_experiment/lv_null_variance_experiment/all_runs_wide.csv`
- `output/lv_experiment/lv_null_variance_experiment/feature_variance_summary.csv`
- `output/lv_experiment/lv_null_variance_experiment/overall_variance_summary.csv`
- `output/lv_experiment/lv_null_variance_experiment/variance_overall_by_group.png`
- `output/lv_experiment/lv_null_variance_experiment/sd_overall_by_group.png`
- `output/lv_experiment/lv_null_variance_experiment/variance_points_with_mean_trend_by_b.png`
- `output/lv_experiment/lv_null_variance_experiment/sd_points_with_mean_trend_by_b.png`

### LV rank stability

- `output/lv_experiment/lv_rank_stability_experiment/all_runs_long.csv`
- `output/lv_experiment/lv_rank_stability_experiment/metapath_rank_table.csv`
- `output/lv_experiment/lv_rank_stability_experiment/pairwise_metrics.csv`
- `output/lv_experiment/lv_rank_stability_experiment/lv_stability_summary.csv`
- `output/lv_experiment/lv_rank_stability_experiment/overall_stability_summary.csv`
- `output/lv_experiment/lv_rank_stability_experiment/spearman_overall_by_group.png`
- `output/lv_experiment/lv_rank_stability_experiment/topk_jaccard_overall_by_group.png`
- `output/lv_experiment/lv_rank_stability_experiment/rho_points_with_mean_trend_by_b.png`
- `output/lv_experiment/lv_rank_stability_experiment/rank_scatter_ref_seed_11/`
- `output/lv_experiment/lv_rank_stability_experiment/top_pairs_runs.csv`
- `output/lv_experiment/lv_rank_stability_experiment/top_paths_runs.csv`

### LV group QC

- `output/lv_experiment/lv_group_qc_experiment/descriptor_panel.csv`
- `output/lv_experiment/lv_group_qc_experiment/gap_summary.csv`
- `output/lv_experiment/lv_group_qc_experiment/random_match_qc.csv`
- `output/lv_experiment/lv_group_qc_experiment/permuted_null_qc.csv`
- `output/lv_experiment/lv_group_qc_experiment/calibration_envelope.csv`
- `output/lv_experiment/lv_group_qc_experiment/descriptor_deviation.csv`
- `output/lv_experiment/lv_group_qc_experiment/within_null_stability_summary.csv`
- `output/lv_experiment/lv_group_qc_experiment/between_null_agreement.csv`
- `output/lv_experiment/lv_group_qc_experiment/score_separation_table.csv`
- `output/lv_experiment/lv_group_qc_experiment/lv_qc_metric_snapshot.csv`
- `output/lv_experiment/lv_group_qc_experiment/lv_qc_summary.csv`
- `output/lv_experiment/lv_group_qc_experiment/lv_group_qc_dashboard.png`
- `output/lv_experiment/lv_group_qc_experiment/lv_score_separation_raw.png`
- `output/lv_experiment/lv_group_qc_experiment/lv_score_separation_standardized.png`
- `output/lv_experiment/lv_group_qc_experiment/lv_descriptor_outcome_table.csv`
- `output/lv_experiment/lv_group_qc_experiment/lv_qc_vs_predictors.png`
- `output/lv_experiment/lv_group_qc_experiment/lv_qc_predictor_correlations.csv`
- `output/lv_experiment/lv_group_qc_experiment/lv_qc_outliers.csv`
- `output/lv_experiment/lv_group_qc_experiment/lv_ranked_score_curves.png`
- `output/lv_experiment/lv_group_qc_experiment/lv_metapath_length_disagreement_summary.csv`

### Year shared workspace

- `output/dwpc_direct/all_GO_positive_growth/replicate_manifest.csv`
- `output/dwpc_direct/all_GO_positive_growth/replicate_summaries/`

### Year variance

- `output/year_null_variance_exp/year_null_variance_experiment/all_runs_long.csv`
- `output/year_null_variance_exp/year_null_variance_experiment/feature_variance_summary.csv`
- `output/year_null_variance_exp/year_null_variance_experiment/overall_variance_summary.csv`
- `output/year_null_variance_exp/year_null_variance_experiment/variance_overall_by_group.png`
- `output/year_null_variance_exp/year_null_variance_experiment/sd_overall_by_group.png`
- `output/year_null_variance_exp/year_null_variance_experiment/variance_points_with_mean_trend_by_b.png`
- `output/year_null_variance_exp/year_null_variance_experiment/sd_points_with_mean_trend_by_b.png`

### Year rank stability

- `output/year_rank_stability_exp/year_rank_stability_experiment/all_runs_long.csv`
- `output/year_rank_stability_exp/year_rank_stability_experiment/metapath_rank_table.csv`
- `output/year_rank_stability_exp/year_rank_stability_experiment/pairwise_metrics.csv`
- `output/year_rank_stability_exp/year_rank_stability_experiment/go_term_stability_summary.csv`
- `output/year_rank_stability_exp/year_rank_stability_experiment/overall_stability_summary.csv`
- `output/year_rank_stability_exp/year_rank_stability_experiment/spearman_overall_by_group.png`
- `output/year_rank_stability_exp/year_rank_stability_experiment/topk_jaccard_overall_by_group.png`
- `output/year_rank_stability_exp/year_rank_stability_experiment/rho_points_with_mean_trend_by_b.png`
