# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Promiscuity-Controlled Random Gene Sampling
#
# Generate random gene samples controlling for gene promiscuity (number of GO term annotations).
#
# ## Inputs
# - `output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv` (2016 filtered)
# - `output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv` (2024 filtered)
# - `output/intermediate/hetio_bppg_2016_stable.csv` (all 2016 stable genes for promiscuity)
# - `output/intermediate/upd_go_bp_2024_added.csv` (all 2024 added genes for promiscuity)
#
# ## Outputs
# - `output/random_samples/all_GO_positive_growth_2016/random_001.csv`
# - `output/random_samples/all_GO_positive_growth_2024/random_001.csv`
#
# ## Description
# This notebook generates 1 random gene sample per year that controls for gene promiscuity.
# For each real gene in a GO term, we sample a random gene from OTHER GO terms
# that has a similar number of GO annotations (promiscuity).
#
# This approach ensures:
# 1. Same number of genes per GO term
# 2. Genes are sampled from other GO terms (not unannotated)
# 3. Promiscuity distribution is matched (within tolerance)
# 4. Multiple samples for robust statistical testing (matching permutation approach)
#
# **Note:** Neo4j ID mapping is deferred to notebook 2 when Docker stack is running.

# %%
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Setup repo root for consistent paths
# Works whether notebook is run from repo root or notebooks/ subdirectory
if Path.cwd().name == "notebooks":
    repo_root = Path("..").resolve()
else:
    repo_root = Path.cwd()

sys.path.insert(0, str(repo_root))
from src.random_sampling import (
    generate_promiscuity_controlled_samples,
    calculate_gene_promiscuity
)

(repo_root / 'output/random_samples/all_GO_positive_growth_2016').mkdir(
    parents=True, exist_ok=True
)
(repo_root / 'output/random_samples/all_GO_positive_growth_2024').mkdir(
    parents=True, exist_ok=True
)

print(f'Repo root: {repo_root}')
print('Environment setup complete')

# %% [markdown]
# ## Load Data
#
# Load filtered datasets and original GO annotations for promiscuity calculation.

# %%
# Load filtered datasets from notebook 1.3
real_2016 = pd.read_csv(
    repo_root / 'output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv'
)
real_2024 = pd.read_csv(
    repo_root / 'output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv'
)

print('Filtered GO-Gene Associations Loaded')
print('=' * 80)
print(f'2016: {len(real_2016):,} GO-gene pairs')
print(f'      {real_2016["go_id"].nunique()} unique GO terms')
print(f'      {real_2016["entrez_gene_id"].nunique()} unique genes')

print(f'\n2024: {len(real_2024):,} GO-gene pairs')
print(f'      {real_2024["go_id"].nunique()} unique GO terms')
print(f'      {real_2024["entrez_gene_id"].nunique()} unique genes')

# %%
# Load ALL BP-Gene associations for promiscuity calculation
# We need the full dataset to accurately count how many terms each gene belongs to

# For 2016: Use the stable genes from notebook 1.1
all_bpg_2016 = pd.read_csv(repo_root / 'output/intermediate/hetio_bppg_2016_stable.csv')

# Keep only needed columns
all_bpg_2016 = all_bpg_2016[['go_id', 'entrez_gene_id']].drop_duplicates()

print(f'All 2016 BP-Gene associations (stable genes): {len(all_bpg_2016):,}')
print(f'  {all_bpg_2016["go_id"].nunique()} GO terms')
print(f'  {all_bpg_2016["entrez_gene_id"].nunique()} genes')

# %%
# For 2024: Use the added genes from notebook 1.1
all_bpg_2024 = pd.read_csv(repo_root / 'output/intermediate/upd_go_bp_2024_added.csv')

# Keep only needed columns
all_bpg_2024 = all_bpg_2024[['go_id', 'entrez_gene_id']].drop_duplicates()

print(f'All 2024 BP-Gene associations (added genes): {len(all_bpg_2024):,}')
print(f'  {all_bpg_2024["go_id"].nunique()} GO terms')
print(f'  {all_bpg_2024["entrez_gene_id"].nunique()} genes')

# %% [markdown]
# ## Calculate Gene Promiscuity
#
# Count how many GO terms each gene belongs to across the full dataset.

# %%
# Calculate promiscuity for 2016
promiscuity_2016 = calculate_gene_promiscuity(
    all_bpg_2016,
    go_id_col='go_id',
    gene_id_col='entrez_gene_id'
)

print('2016 Gene Promiscuity Statistics')
print('=' * 80)
print(promiscuity_2016['promiscuity'].describe())

print(f'\nMost promiscuous genes (2016):')
top_2016 = promiscuity_2016.nlargest(10, 'promiscuity')
for _, row in top_2016.iterrows():
    print(f"  Gene {row['entrez_gene_id']}: "
          f"{row['promiscuity']} GO terms")

# %%
# Calculate promiscuity for 2024
promiscuity_2024 = calculate_gene_promiscuity(
    all_bpg_2024,
    go_id_col='go_id',
    gene_id_col='entrez_gene_id'
)

print('2024 Gene Promiscuity Statistics')
print('=' * 80)
print(promiscuity_2024['promiscuity'].describe())

print(f'\nMost promiscuous genes (2024):')
top_2024 = promiscuity_2024.nlargest(10, 'promiscuity')
for _, row in top_2024.iterrows():
    print(f"  Gene {row['entrez_gene_id']}: "
          f"{row['promiscuity']} GO terms")

# %% [markdown]
# ## Generate Random Samples for 2016
#
# Generate 1 independent random sample with a fixed random seed.

# %%
print('Generating 1 promiscuity-controlled random sample for 2016...')
print('=' * 80)

random_samples_2016 = []

for i in range(1, 2):
    print(f'\nGenerating random sample {i}/1...')
    
    random_sample = generate_promiscuity_controlled_samples(
        go_gene_df=real_2016,
        all_go_annotations=all_bpg_2016,
        go_id_col='go_id',
        gene_id_col='entrez_gene_id',
        promiscuity_tolerance=2,
        random_state=42 + i
    )
    
    random_samples_2016.append(random_sample)
    
    print(f'  Generated {len(random_sample):,} pairs, '
          f'{random_sample["go_id"].nunique()} GO terms, '
          f'{random_sample["pseudo_gene_id"].nunique()} unique genes')
    print(f'  Real promiscuity: mean={random_sample["real_promiscuity"].mean():.2f}')
    print(f'  Random promiscuity: mean={random_sample["sampled_promiscuity"].mean():.2f}')

print('\n2016 random sample complete')

# %% [markdown]
# ## Generate Random Samples for 2024
#
# Generate 1 independent random sample for 2024 data.

# %%
print('Generating 1 promiscuity-controlled random sample for 2024...')
print('=' * 80)

random_samples_2024 = []

for i in range(1, 2):
    print(f'\nGenerating random sample {i}/1...')
    
    random_sample = generate_promiscuity_controlled_samples(
        go_gene_df=real_2024,
        all_go_annotations=all_bpg_2024,
        go_id_col='go_id',
        gene_id_col='entrez_gene_id',
        promiscuity_tolerance=2,
        random_state=42 + i
    )
    
    random_samples_2024.append(random_sample)
    
    print(f'  Generated {len(random_sample):,} pairs, '
          f'{random_sample["go_id"].nunique()} GO terms, '
          f'{random_sample["pseudo_gene_id"].nunique()} unique genes')
    print(f'  Real promiscuity: mean={random_sample["real_promiscuity"].mean():.2f}')
    print(f'  Random promiscuity: mean={random_sample["sampled_promiscuity"].mean():.2f}')

print('\n2024 random sample complete')

# %% [markdown]
# ## Validation
#
# Verify that random samples have expected properties.

# %%
print('\n' + '=' * 80)
print('VALIDATION: Sample Sizes')
print('=' * 80)

# Validate all 2016 samples
for i, random_2016 in enumerate(random_samples_2016, start=1):
    real_2016_sizes = real_2016.groupby('go_id').size()
    random_2016_sizes = random_2016.groupby('go_id').size()
    
    if (real_2016_sizes == random_2016_sizes).all():
        print(f'2016 Sample {i}: Sample sizes match (PASS)')
    else:
        print(f'2016 Sample {i}: Sample sizes differ (FAIL)')

# Validate all 2024 samples
for i, random_2024 in enumerate(random_samples_2024, start=1):
    real_2024_sizes = real_2024.groupby('go_id').size()
    random_2024_sizes = random_2024.groupby('go_id').size()
    
    if (real_2024_sizes == random_2024_sizes).all():
        print(f'2024 Sample {i}: Sample sizes match (PASS)')
    else:
        print(f'2024 Sample {i}: Sample sizes differ (FAIL)')

# %%
print('\n' + '=' * 80)
print('VALIDATION: No Overlap with Real Genes per GO Term')
print('=' * 80)

# Check all 2016 samples
for i, random_2016 in enumerate(random_samples_2016, start=1):
    overlap_count = 0
    for go_id in real_2016['go_id'].unique():
        real_genes = set(
            real_2016[real_2016['go_id'] == go_id]['entrez_gene_id']
        )
        random_genes = set(
            random_2016[random_2016['go_id'] == go_id]['pseudo_gene_id']
        )
        overlap = len(real_genes & random_genes)
        overlap_count += overlap
    
    if overlap_count == 0:
        print(f'2016 Sample {i}: No overlap (PASS)')
    else:
        print(f'2016 Sample {i}: Found {overlap_count} overlaps (FAIL)')

# Check all 2024 samples
for i, random_2024 in enumerate(random_samples_2024, start=1):
    overlap_count = 0
    for go_id in real_2024['go_id'].unique():
        real_genes = set(
            real_2024[real_2024['go_id'] == go_id]['entrez_gene_id']
        )
        random_genes = set(
            random_2024[random_2024['go_id'] == go_id]['pseudo_gene_id']
        )
        overlap = len(real_genes & random_genes)
        overlap_count += overlap
    
    if overlap_count == 0:
        print(f'2024 Sample {i}: No overlap (PASS)')
    else:
        print(f'2024 Sample {i}: Found {overlap_count} overlaps (FAIL)')

# %%
print('\n' + '=' * 80)
print('VALIDATION: All Random Genes Are Annotated')
print('=' * 80)

all_annotated_2016 = set(all_bpg_2016['entrez_gene_id'])
all_annotated_2024 = set(all_bpg_2024['entrez_gene_id'])

# Check all 2016 samples
for i, random_2016 in enumerate(random_samples_2016, start=1):
    random_genes = set(random_2016['pseudo_gene_id'])
    unannotated = random_genes - all_annotated_2016
    
    if len(unannotated) == 0:
        print(f'2016 Sample {i}: All genes annotated (PASS)')
    else:
        print(f'2016 Sample {i}: Found {len(unannotated)} unannotated (FAIL)')

# Check all 2024 samples
for i, random_2024 in enumerate(random_samples_2024, start=1):
    random_genes = set(random_2024['pseudo_gene_id'])
    unannotated = random_genes - all_annotated_2024
    
    if len(unannotated) == 0:
        print(f'2024 Sample {i}: All genes annotated (PASS)')
    else:
        print(f'2024 Sample {i}: Found {len(unannotated)} unannotated (FAIL)')

# %% [markdown]
# ## Save Outputs

# %%
print('\nSaving random samples...')

random_dir_2016 = repo_root / 'output/random_samples/all_GO_positive_growth_2016'
random_dir_2024 = repo_root / 'output/random_samples/all_GO_positive_growth_2024'

# Save 2016 samples
for i, random_sample in enumerate(random_samples_2016, start=1):
    output_path = random_dir_2016 / f'random_{i:03d}.csv'
    random_out = random_sample.copy()
    random_out['entrez_gene_id'] = random_out['pseudo_gene_id']
    random_out.to_csv(output_path, index=False)
    print(f'  Saved random_{i:03d}.csv: {len(random_sample):,} pairs')

# Save 2024 samples
for i, random_sample in enumerate(random_samples_2024, start=1):
    output_path = random_dir_2024 / f'random_{i:03d}.csv'
    random_out = random_sample.copy()
    random_out['entrez_gene_id'] = random_out['pseudo_gene_id']
    random_out.to_csv(output_path, index=False)
    print(f'  Saved random_{i:03d}.csv: {len(random_sample):,} pairs')

print('\n' + '=' * 80)
print('NOTEBOOK 1.5 COMPLETE')
print('=' * 80)

print('\nGenerated Control Datasets:')
print('  Permuted datasets (1.4): 1 permutation per year (shuffle GO labels)')
print('  Random datasets (1.5): 1 random sample per year (promiscuity-controlled)')
print('  Total: 4 control datasets + 2 real datasets = 6 datasets')

print('\nOutput Files:')
print('  output/permutations/all_GO_positive_growth_2016/perm_001.csv')
print('  output/permutations/all_GO_positive_growth_2024/perm_001.csv')
print('  output/random_samples/all_GO_positive_growth_2016/random_001.csv')
print('  output/random_samples/all_GO_positive_growth_2024/random_001.csv')

print('\nNext Steps:')
print('  Run script 2 to compute DWPC for all datasets')
print('  poe compute-dwpc-direct')
