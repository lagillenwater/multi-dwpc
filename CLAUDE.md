# Multi-DWPC Project Notes

## Working style (always)

- Respond in as few words as possible.
- Code in as few lines as possible.
- Touch as little as possible when making edits. 

## Environment

- **Local environment**: Does not have scipy installed
- **HPC environment**: Use `conda activate multi_dwpc` which has scipy and all dependencies
- Scripts requiring scipy (path enumeration, sparse matrices) must be run on HPC or in the conda environment

## Key Scripts

- `scripts/year_intermediate_sharing_2016_2024.py` - Computes intermediate sharing between 2016 and 2024-added genes
- `scripts/lv_intermediate_sharing.py` - Computes intermediate sharing for LV gene sets with b=10 metapath selection
- `scripts/lv_multidwpc_analysis.py` - Main LV analysis orchestrator

## HPC Submission

- Submit scripts are in `hpc/`
- Array jobs use GO term lists for parallelization
- Aggregation jobs depend on array completion
