# Multi-DWPC Project Notes

## Environment

- **Local environment**: Does not have scipy installed
- **HPC environment**: Use `conda activate multi_dwpc` which has scipy and all dependencies
- Scripts requiring scipy (path enumeration, sparse matrices) must be run on HPC or in the conda environment

## Key Scripts

Year and LV pipelines are unified behind `--analysis-type {year,lv}` entrypoints
under `scripts/`:

- `scripts/pipeline/intermediate_sharing.py --analysis-type {year,lv}` - Computes intermediate sharing (year: 2016-vs-2024 cohort split; lv: per-LV gene sets)
- `scripts/pipeline/run_year_pipeline.py` / `run_lv_pipeline.py` - End-to-end orchestrators that chain B-selection -> intermediate sharing -> generate global summary -> gene table -> visualization
- `scripts/experiments/null_variance_experiment.py --analysis-type {year,lv}` - Null variance across B/seed
- `scripts/experiments/rank_stability_experiment.py --analysis-type {year,lv}` - Rank stability across B/seed
- `scripts/experiments/lv_multidwpc_analysis.py` - LV multi-DWPC staged orchestrator (LV-only)
- `scripts/data_prep/build_null_datasets.py --method {permutation,random}` - Year null datasets

Layout: `scripts/{data_prep,pipeline,experiments,visualization}/`. Live HPC
submit scripts in `hpc/`.

## HPC Submission

- Submit scripts are in `hpc/`
- Array jobs use GO term lists for parallelization
- Aggregation jobs depend on array completion
