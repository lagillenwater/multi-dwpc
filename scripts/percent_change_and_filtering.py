# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: multi_dwpc
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.005024, "end_time": "2025-12-05T22:40:19.722463", "exception": false, "start_time": "2025-12-05T22:40:19.717439", "status": "completed"}
# # Percent Change and IQR Filtering
#
# ## Description
# Calculates percent change in gene counts from 2016 to 2024,
# classifies genes as stable vs added, and filters GO terms
# using IQR thresholds.
#
# ## Inputs
# - `output/intermediate/common_go_terms.csv`
# - `output/intermediate/hetio_bppg_2016.csv`
# - `output/intermediate/upd_go_bp_2024.csv`
#
# ## Outputs
# - `output/intermediate/hetio_bppg_2016_stable.csv` (2016 GO-gene pairs, stable genes only)
# - `output/intermediate/upd_go_bp_2024_added.csv` (2024 GO-gene pairs, added genes only)
# - `output/intermediate/go_gene_classification_summary.csv` (classification statistics per GO term)
# - `output/intermediate/all_GO_positive_growth.csv`

# %% papermill={"duration": 1.589916, "end_time": "2025-12-05T22:40:21.318934", "exception": false, "start_time": "2025-12-05T22:40:19.729018", "status": "completed"}
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Setup repo root for consistent paths
# Works whether notebook is run from repo root or notebooks/ subdirectory
if Path.cwd().name == "notebooks":
    repo_root = Path("..").resolve()
else:
    repo_root = Path.cwd()

sys.path.insert(0, str(repo_root))
from src.go_processing import (
    calculate_percent_change,
    classify_genes_stable_added
)
from src.filtering import (
    calculate_iqr_thresholds,
    filter_go_terms_iqr
)

print(f"Repo root: {repo_root}")

# %% papermill={"duration": 0.017992, "end_time": "2025-12-05T22:40:21.339007", "exception": false, "start_time": "2025-12-05T22:40:21.321015", "status": "completed"}
com_go_terms_hetio_upd_go_bp_2024 = pd.read_csv(
    repo_root / 'output/intermediate/common_go_terms.csv'
)

print(f'Loaded {len(com_go_terms_hetio_upd_go_bp_2024)} GO terms')

# %% [markdown] papermill={"duration": 0.005007, "end_time": "2025-12-05T22:40:21.345842", "exception": false, "start_time": "2025-12-05T22:40:21.340835", "status": "completed"}
# ### Gene Classification: Stable vs Added
#
# Classification is per GO-term-gene pair (a gene may be stable for one GO term but added for another):
# - **Stable**: GO-gene pair present in both 2016 AND 2024 annotations
# - **Added**: GO-gene pair only in 2024 (gene existed in Hetionet 2016, but was not annotated to this GO term)
#
# This creates separate datasets:
# - 2016 stable genes (baseline - genes with consistent annotations)
# - 2024 added genes (genes newly annotated to GO terms between 2016-2024)
#
# **Research question**: Do newly annotated GO-gene pairs show similar connectivity patterns to stable pairs, while both differ from random permutations?

# %% papermill={"duration": 0.235744, "end_time": "2025-12-05T22:40:21.586692", "exception": false, "start_time": "2025-12-05T22:40:21.350948", "status": "completed"}
print("Loading gene-level data for classification...")
print("=" * 60)

genes_2016 = pd.read_csv(repo_root / 'output/intermediate/hetio_bppg_2016.csv')
genes_2024 = pd.read_csv(repo_root / 'output/intermediate/upd_go_bp_2024.csv')

print(f"2016 GO-gene pairs: {len(genes_2016):,}")
print(f"2024 GO-gene pairs: {len(genes_2024):,}")
print(f"2016 unique GO terms: {genes_2016['go_id'].nunique()}")
print(f"2024 unique GO terms: {genes_2024['go_id'].nunique()}")
print(f"2016 unique genes: {genes_2016['entrez_gene_id'].nunique()}")
print(f"2024 unique genes: {genes_2024['entrez_gene_id'].nunique()}")

# %% papermill={"duration": 0.74867, "end_time": "2025-12-05T22:40:22.337482", "exception": false, "start_time": "2025-12-05T22:40:21.588812", "status": "completed"}
print("Classifying genes...")
genes_2016_stable, genes_2024_added, classification_summary = \
    classify_genes_stable_added(genes_2016, genes_2024)

print("\nGene Classification Results")
print("=" * 60)
print(f"2016 stable pairs: {len(genes_2016_stable):,}")
print(f"2024 added pairs: {len(genes_2024_added):,}")
print(f"GO terms analyzed: {len(classification_summary)}")
print(f"\nStable genes (unique): {genes_2016_stable['entrez_gene_id'].nunique()}")
print(f"Added genes (unique): {genes_2024_added['entrez_gene_id'].nunique()}")

# %% papermill={"duration": 0.060955, "end_time": "2025-12-05T22:40:22.400367", "exception": false, "start_time": "2025-12-05T22:40:22.339412", "status": "completed"}
print("Classification Summary Statistics")
print("=" * 60)
print(f"\nMean genes per GO term:")
print(f"  2016: {classification_summary['n_genes_2016'].mean():.1f}")
print(f"  2024: {classification_summary['n_genes_2024'].mean():.1f}")
print(f"  Stable: {classification_summary['n_stable'].mean():.1f}")
print(f"  Added: {classification_summary['n_added'].mean():.1f}")

print(f"\nMedian percent of genes:")
print(f"  Stable: {classification_summary['pct_stable'].median():.1f}%")
print(f"  Added: {classification_summary['pct_added'].median():.1f}%")

# Create GO term name lookup from 2024 data
go_name_lookup = genes_2024[['go_id', 'go_name']].drop_duplicates()

print(f"\nTop 10 GO terms by number of added genes:")
top_added = classification_summary.nlargest(10, 'n_added')[
    ['go_id', 'n_genes_2024', 'n_stable', 'n_added', 'pct_added']
].merge(go_name_lookup, on='go_id', how='left')
top_added = top_added[['go_id', 'go_name', 'n_genes_2024', 'n_stable', 'n_added', 'pct_added']]
print(top_added)

# %% papermill={"duration": 0.602779, "end_time": "2025-12-05T22:40:23.005138", "exception": false, "start_time": "2025-12-05T22:40:22.402359", "status": "completed"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

axes[0].hist(classification_summary['n_stable'], bins=30,
             alpha=0.7, label='Stable genes', edgecolor='black')
axes[0].hist(classification_summary['n_added'], bins=30,
             alpha=0.7, label='Added genes', edgecolor='black', color='orange')
axes[0].set_xlabel('Number of Genes per GO Term', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution: Stable vs Added Genes', fontsize=12)
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].scatter(classification_summary['n_stable'],
                classification_summary['n_added'],
                alpha=0.4, s=30)
axes[1].set_xlabel('Number of Stable Genes', fontsize=11)
axes[1].set_ylabel('Number of Added Genes', fontsize=11)
axes[1].set_title('Stable vs Added Genes per GO Term', fontsize=12)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
output_images = repo_root / 'output/images'
output_images.mkdir(parents=True, exist_ok=True)
plt.savefig(output_images / 'gene_classification_stable_vs_added.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig(output_images / 'gene_classification_stable_vs_added.jpeg',
            dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved to {output_images / 'gene_classification_stable_vs_added.pdf'}")

# %% papermill={"duration": 0.319067, "end_time": "2025-12-05T22:40:23.330481", "exception": false, "start_time": "2025-12-05T22:40:23.011414", "status": "completed"}
print("Saving classified gene datasets...")
print("=" * 60)

output_intermediate = repo_root / 'output/intermediate'
output_intermediate.mkdir(parents=True, exist_ok=True)

genes_2016_stable.to_csv(
    output_intermediate / 'hetio_bppg_2016_stable.csv',
    index=False
)
genes_2024_added.to_csv(
    output_intermediate / 'upd_go_bp_2024_added.csv',
    index=False
)
classification_summary.to_csv(
    output_intermediate / 'go_gene_classification_summary.csv',
    index=False
)

print(f"Saved: hetio_bppg_2016_stable.csv ({len(genes_2016_stable):,} pairs)")
print(f"Saved: upd_go_bp_2024_added.csv ({len(genes_2024_added):,} pairs)")
print(f"Saved: go_gene_classification_summary.csv "
      f"({len(classification_summary)} GO terms)")
print("\nGene classification complete!")

# %% papermill={"duration": 0.017933, "end_time": "2025-12-05T22:40:23.359083", "exception": false, "start_time": "2025-12-05T22:40:23.341150", "status": "completed"}
com_go_terms_hetio_upd_go_bp_2024 = com_go_terms_hetio_upd_go_bp_2024.merge(
    classification_summary[['go_id', 'n_stable', 'n_added']],
    on='go_id',
    how='left'
)

print("Added classification counts to common_go_terms dataframe")
print(f"New columns: n_stable, n_added")
print(com_go_terms_hetio_upd_go_bp_2024.head())

# %% papermill={"duration": 0.016548, "end_time": "2025-12-05T22:40:23.386363", "exception": false, "start_time": "2025-12-05T22:40:23.369815", "status": "completed"}
# Calculate percent change using src function
com_go_terms_hetio_upd_go_bp_2024 = calculate_percent_change(
    com_go_terms_hetio_upd_go_bp_2024,
    count_col_2016='no_of_genes_in_hetio_GO_2016',
    count_col_2024='no_of_genes_in_GO_2024'
)

print(f"Mean change: "
      f"{com_go_terms_hetio_upd_go_bp_2024['pct_change_genes'].mean():.2f}%")
print(com_go_terms_hetio_upd_go_bp_2024.head())

# %% [markdown] papermill={"duration": 0.015159, "end_time": "2025-12-05T22:40:24.114326", "exception": false, "start_time": "2025-12-05T22:40:24.099167", "status": "completed"}
# ### Growth-Focused IQR Filter Thresholds
#
# This section implements a two-stage filtering approach:
# 1. Filter to GO terms with positive annotation growth only
# 2. Calculate IQR thresholds from growth-only terms
# 3. Apply minimum gene count for statistical power
#
# This approach focuses on GO terms that gained annotations between 2016 and 2024.

# %% papermill={"duration": 0.019978, "end_time": "2025-12-05T22:40:24.148411", "exception": false, "start_time": "2025-12-05T22:40:24.128433", "status": "completed"}
positive_growth_terms = com_go_terms_hetio_upd_go_bp_2024[
    com_go_terms_hetio_upd_go_bp_2024['pct_change_genes'] > 0
].copy()

print("Filter to Positive Growth Only")
print("=" * 60)
print(f"Total GO terms: {len(com_go_terms_hetio_upd_go_bp_2024)}")
print(f"Terms with positive growth: {len(positive_growth_terms)}")
print(f"Retention rate: {len(positive_growth_terms) / len(com_go_terms_hetio_upd_go_bp_2024):.1%}")
print(f"\nPercent change range:")
print(f"  Min: {positive_growth_terms['pct_change_genes'].min():.2f}%")
print(f"  Median: {positive_growth_terms['pct_change_genes'].median():.2f}%")
print(f"  Max: {positive_growth_terms['pct_change_genes'].max():.2f}%")


# %% papermill={"duration": 0.074909, "end_time": "2025-12-05T22:40:24.238912", "exception": false, "start_time": "2025-12-05T22:40:24.164003", "status": "completed"}
# Calculate IQR thresholds using src.filtering function
thresholds_growth = calculate_iqr_thresholds(positive_growth_terms)

print("IQR Thresholds (Growth-Only Terms)")
print("=" * 60)
print(f"Total terms analyzed: {thresholds_growth['total_terms']}")
print(f"\nGene Count Thresholds:")
print(f"  Min (Q25 - 1.5*IQR): {thresholds_growth['min_genes']:.0f}")
print(f"  Max (Q75 + 1.5*IQR): {thresholds_growth['max_genes']:.0f}")
print(f"  Median: {thresholds_growth['median_genes']:.0f}")
print(f"\nPercent Change Thresholds:")
print(f"  Min (Q25 - 1.5*IQR): {thresholds_growth['min_pct_change']:.2f}%")
print(f"  Max (Q75 + 1.5*IQR): {thresholds_growth['max_pct_change']:.2f}%")

# %% [markdown] papermill={"duration": 0.014474, "end_time": "2025-12-05T22:40:24.494871", "exception": false, "start_time": "2025-12-05T22:40:24.480397", "status": "completed"}
# ### Create Filtered Dataset
#
# All GO terms with positive growth, within IQR, and at least 10 genes.

# %% papermill={"duration": 0.024061, "end_time": "2025-12-05T22:40:24.535739", "exception": false, "start_time": "2025-12-05T22:40:24.511678", "status": "completed"}
# Create filtered dataset using src.filtering function
all_GO_positive_growth = filter_go_terms_iqr(
    positive_growth_terms,
    thresholds_growth,
    min_genes=10
)

print("All GO Positive Growth Terms (Growth + IQR + Min Genes)")
print("=" * 60)
print(f"Filtered GO terms: {len(all_GO_positive_growth)}")
print(f"Mean gene count (2016): "
      f"{all_GO_positive_growth['no_of_genes_in_hetio_GO_2016'].mean():.1f}")
print(f"Median gene count (2016): "
      f"{all_GO_positive_growth['no_of_genes_in_hetio_GO_2016'].median():.0f}")

# %% papermill={"duration": 0.382315, "end_time": "2025-12-05T22:40:25.133759", "exception": false, "start_time": "2025-12-05T22:40:24.751444", "status": "completed"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

axes[0].hist(all_GO_positive_growth['n_stable'], bins=30, 
             alpha=0.7, label='Stable genes', edgecolor='black')
axes[0].hist(all_GO_positive_growth['n_added'], bins=30, 
             alpha=0.7, label='Added genes', edgecolor='black', color='orange')
axes[0].set_xlabel('Number of Genes per GO Term', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Filtered Dataset: Stable vs Added Genes', fontsize=12)
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].scatter(all_GO_positive_growth['n_stable'],
                all_GO_positive_growth['n_added'],
                alpha=0.4, s=30)
axes[1].set_xlabel('Number of Stable Genes', fontsize=11)
axes[1].set_ylabel('Number of Added Genes', fontsize=11)
axes[1].set_title('Filtered Dataset: Stable vs Added per GO Term', fontsize=12)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
output_images = repo_root / 'output/images'
plt.savefig(output_images / 'filtered_dataset_stable_vs_added.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig(output_images / 'filtered_dataset_stable_vs_added.jpeg', 
            dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved to {output_images / 'filtered_dataset_stable_vs_added.pdf'}")

# %% [markdown] papermill={"duration": 0.016073, "end_time": "2025-12-05T22:40:25.153486", "exception": false, "start_time": "2025-12-05T22:40:25.137413", "status": "completed"}
# ### Summary
#
# One dataset created using growth-focused IQR filtering:
#
# 1. All GO positive growth: IQR filtering and at least 10 genes per term
#
# Rationale:
# - Positive growth filter: Focuses on GO terms gaining annotations between 2016 and 2024
# - IQR on growth-only: Removes outliers among growing terms specifically
# - Min 10 genes: Ensures adequate statistical power

# %% papermill={"duration": 0.026153, "end_time": "2025-12-05T22:40:25.396720", "exception": false, "start_time": "2025-12-05T22:40:25.370567", "status": "completed"}
# Save outputs
output_intermediate = repo_root / 'output/intermediate'
all_GO_positive_growth.to_csv(
    output_intermediate / 'all_GO_positive_growth.csv',
    index=False
)

print(f'Saved all_GO_positive_growth: {len(all_GO_positive_growth)} GO terms')
print('\n Percent change filtering complete!')
