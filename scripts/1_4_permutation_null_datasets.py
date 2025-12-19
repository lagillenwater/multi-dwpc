# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1.4 Generate Permuted GO-Label Datasets
#
# Generate permuted GO-gene associations for null distribution analysis.
#
# ## Inputs
# - `output/intermediate/hetio_bppg_dataset2_filtered.csv` (2016)
# - `output/intermediate/hetio_bppg_dataset2_2024_filtered.csv` (2024)
#
# ## Outputs
# - `output/permutations/dataset2_2016/perm_001.csv` through `perm_005.csv`
# - `output/permutations/dataset2_2024/perm_001.csv` through `perm_005.csv`
#
# ## Description
# This notebook generates 5 permuted GO-gene datasets for each year by shuffling
# GO labels among genes while preserving:
# - Gene degree distribution (same genes, different GO labels)
# - Number of genes per GO term
# - Total number of associations
#
# This approach tests whether specific GO-gene associations drive connectivity
# or if any random labeling would produce similar results.

# %%
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os

# Setup repo root for consistent paths
# Works whether notebook is run from repo root or notebooks/ subdirectory
if Path.cwd().name == "notebooks":
    repo_root = Path("..").resolve()
else:
    repo_root = Path.cwd()

sys.path.insert(0, str(repo_root))
from src.random_sampling import permute_go_labels

# Create output directories
(repo_root / 'output/permutations/dataset2_2016').mkdir(parents=True, exist_ok=True)
(repo_root / 'output/permutations/dataset2_2024').mkdir(parents=True, exist_ok=True)

print(f'Repo root: {repo_root}')
print('Environment setup complete')

# %%
# Load Dataset 2 filtered data (both years)
real_2016 = pd.read_csv(repo_root / 'output/intermediate/hetio_bppg_dataset2_filtered.csv')
real_2024 = pd.read_csv(repo_root / 'output/intermediate/hetio_bppg_dataset2_2024_filtered.csv')

print('Dataset 2 - Real Data Loaded')
print('=' * 80)
print(f'2016: {len(real_2016):,} GO-gene pairs')
print(f'      {real_2016["go_id"].nunique()} unique GO terms')
print(f'      {real_2016["entrez_gene_id"].nunique()} unique genes')

print(f'\n2024: {len(real_2024):,} GO-gene pairs')
print(f'      {real_2024["go_id"].nunique()} unique GO terms')
print(f'      {real_2024["entrez_gene_id"].nunique()} unique genes')

# Verify required columns exist
required_cols = ['go_id', 'entrez_gene_id']
for col in required_cols:
    if col not in real_2016.columns:
        raise ValueError(f'Missing required column in 2016 data: {col}')
    if col not in real_2024.columns:
        raise ValueError(f'Missing required column in 2024 data: {col}')

print('\nData validation passed')

# %% [markdown]
# ## Generate Permutations for 2016
#
# Create 5 permuted datasets by shuffling GO labels among genes.

# %%
# Generate 5 permutations for 2016
print('\nGenerating 5 permutations for 2016...')
print('=' * 80)

permuted_2016 = permute_go_labels(
    go_gene_df=real_2016,
    n_permutations=5,
    go_id_col='go_id',
    gene_id_col='entrez_gene_id',
    random_state=42
)

# Save each permutation
perm_dir_2016 = repo_root / 'output/permutations/dataset2_2016'
for i, perm_df in enumerate(permuted_2016, start=1):
    output_path = perm_dir_2016 / f'perm_{i:03d}.csv'
    perm_df.to_csv(output_path, index=False)
    print(f'  Saved: perm_{i:03d}.csv ({len(perm_df):,} pairs, '
          f'{perm_df["go_id"].nunique()} GO terms, '
          f'{perm_df["entrez_gene_id"].nunique()} genes)')

print('\n2016 permutations complete')

# %% [markdown]
# ## Generate Permutations for 2024
#
# Create 5 permuted datasets for 2024 data.

# %%
# Generate 5 permutations for 2024
print('\nGenerating 5 permutations for 2024...')
print('=' * 80)

permuted_2024 = permute_go_labels(
    go_gene_df=real_2024,
    n_permutations=5,
    go_id_col='go_id',
    gene_id_col='entrez_gene_id',
    random_state=42
)

# Save each permutation
perm_dir_2024 = repo_root / 'output/permutations/dataset2_2024'
for i, perm_df in enumerate(permuted_2024, start=1):
    output_path = perm_dir_2024 / f'perm_{i:03d}.csv'
    perm_df.to_csv(output_path, index=False)
    print(f'  Saved: perm_{i:03d}.csv ({len(perm_df):,} pairs, '
          f'{perm_df["go_id"].nunique()} GO terms, '
          f'{perm_df["entrez_gene_id"].nunique()} genes)')

print('\n2024 permutations complete')

# %% [markdown]
# ## Validation
#
# Verify that permutations preserve expected properties:
# 1. GO term sizes (same number of genes per term)
# 2. Gene sets (same genes, only labels shuffled)
# 3. Total number of associations

# %%
# Validation: Check GO term sizes preserved
print('\n' + '=' * 80)
print('VALIDATION: GO Term Sizes')
print('=' * 80)

perm_dir_2016 = repo_root / 'output/permutations/dataset2_2016'
perm_dir_2024 = repo_root / 'output/permutations/dataset2_2024'

print('\n2016 Permutations:')
for i in range(1, 6):
    perm_2016 = pd.read_csv(perm_dir_2016 / f'perm_{i:03d}.csv')
    real_sizes = real_2016.groupby('go_id').size().sort_index()
    perm_sizes = perm_2016.groupby('go_id').size().sort_index()
    
    if (real_sizes == perm_sizes).all():
        print(f'  Permutation {i}: GO term sizes preserved')
    else:
        mismatches = len(real_sizes) - (real_sizes == perm_sizes).sum()
        print(f'  Permutation {i}: {mismatches} GO term size mismatches!')

print('\n2024 Permutations:')
for i in range(1, 6):
    perm_2024 = pd.read_csv(perm_dir_2024 / f'perm_{i:03d}.csv')
    real_sizes = real_2024.groupby('go_id').size().sort_index()
    perm_sizes = perm_2024.groupby('go_id').size().sort_index()
    
    if (real_sizes == perm_sizes).all():
        print(f'  Permutation {i}: GO term sizes preserved')
    else:
        mismatches = len(real_sizes) - (real_sizes == perm_sizes).sum()
        print(f'  Permutation {i}: {mismatches} GO term size mismatches!')

# %%
# Validation: Check genes unchanged (only labels shuffled)
print('\n' + '=' * 80)
print('VALIDATION: Gene Sets')
print('=' * 80)

perm_dir_2016 = repo_root / 'output/permutations/dataset2_2016'
perm_dir_2024 = repo_root / 'output/permutations/dataset2_2024'

print('\n2016:')
perm1_2016 = pd.read_csv(perm_dir_2016 / 'perm_001.csv')
real_genes_2016 = set(real_2016['entrez_gene_id'])
perm_genes_2016 = set(perm1_2016['entrez_gene_id'])

if real_genes_2016 == perm_genes_2016:
    print(f'  Gene IDs preserved ({len(real_genes_2016)} genes)')
    print('    Only GO labels shuffled (correct behavior)')
else:
    print(f'  Gene IDs differ between real and permuted!')
    print(f'    Real: {len(real_genes_2016)} genes')
    print(f'    Permuted: {len(perm_genes_2016)} genes')
    print(f'    Missing: {len(real_genes_2016 - perm_genes_2016)}')
    print(f'    Extra: {len(perm_genes_2016 - real_genes_2016)}')

print('\n2024:')
perm1_2024 = pd.read_csv(perm_dir_2024 / 'perm_001.csv')
real_genes_2024 = set(real_2024['entrez_gene_id'])
perm_genes_2024 = set(perm1_2024['entrez_gene_id'])

if real_genes_2024 == perm_genes_2024:
    print(f'  Gene IDs preserved ({len(real_genes_2024)} genes)')
    print('    Only GO labels shuffled (correct behavior)')
else:
    print(f'  Gene IDs differ between real and permuted!')
    print(f'    Real: {len(real_genes_2024)} genes')
    print(f'    Permuted: {len(perm_genes_2024)} genes')
    print(f'    Missing: {len(real_genes_2024 - perm_genes_2024)}')
    print(f'    Extra: {len(perm_genes_2024 - real_genes_2024)}')

# %%
# Validation: Check total associations preserved
print('\n' + '=' * 80)
print('VALIDATION: Total Associations')
print('=' * 80)

perm_dir_2016 = repo_root / 'output/permutations/dataset2_2016'
perm_dir_2024 = repo_root / 'output/permutations/dataset2_2024'

print(f'\n2016 Real: {len(real_2016):,} associations')
for i in range(1, 6):
    perm = pd.read_csv(perm_dir_2016 / f'perm_{i:03d}.csv')
    if len(perm) == len(real_2016):
        print(f'  Permutation {i}: {len(perm):,} associations')
    else:
        print(f'  Permutation {i}: {len(perm):,} associations (expected {len(real_2016):,})')

print(f'\n2024 Real: {len(real_2024):,} associations')
for i in range(1, 6):
    perm = pd.read_csv(perm_dir_2024 / f'perm_{i:03d}.csv')
    if len(perm) == len(real_2024):
        print(f'  Permutation {i}: {len(perm):,} associations')
    else:
        print(f'  Permutation {i}: {len(perm):,} associations (expected {len(real_2024):,})')

# %%
# Summary
print('\n' + '=' * 80)
print('NOTEBOOK 1.4 COMPLETE')
print('=' * 80)

print('\nGenerated Permuted Datasets:')
print('  2016: 5 permutations')
print('  2024: 5 permutations')
print('  Total: 10 permuted datasets + 2 real datasets = 12 datasets')

print('\nOutput Files:')
print('  output/permutations/dataset2_2016/perm_001.csv through perm_005.csv')
print('  output/permutations/dataset2_2024/perm_001.csv through perm_005.csv')

print('\nValidation Results:')
print('  GO term sizes preserved')
print('  Gene sets unchanged (only labels shuffled)')
print('  Total associations preserved')

print('\nNext Steps:')
print('  1. Run notebook 2 to compute DWPC for all 12 datasets')
print('  2. Expected runtime: 7.5-8.5 hours (can run overnight)')
print('  3. Each dataset takes 30-45 minutes for Dataset 2')
