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

# %% [markdown] papermill={"duration": 0.002571, "end_time": "2025-12-09T20:04:05.880135", "exception": false, "start_time": "2025-12-09T20:04:05.877564", "status": "completed"}
# # 1.3 Jaccard Similarity Analysis and Filtering
#
# Calculate Jaccard similarity between GO terms and filter overlapping terms.
#
# ## Inputs
# - `output/intermediate/hetio_bppg_2016_stable.csv` (2016 stable genes)
# - `output/intermediate/upd_go_bp_2024_added.csv` (2024 added genes)
# - `output/intermediate/dataset1_all_growth.csv`
# - `output/intermediate/dataset2_parents.csv`
#
# ## Outputs
# - `output/intermediate/dataset1_filtered.csv`
# - `output/intermediate/dataset2_filtered.csv`
# - `output/intermediate/hetio_bppg_dataset1_filtered.csv` (2016 stable, Jaccard filtered)
# - `output/intermediate/hetio_bppg_dataset2_filtered.csv` (2016 stable, Jaccard filtered)
# - `output/intermediate/hetio_bppg_dataset1_2024_filtered.csv` (2024 added, Jaccard filtered)
# - `output/intermediate/hetio_bppg_dataset2_2024_filtered.csv` (2024 added, Jaccard filtered)
# - `output/jaccard_similarity/` (cached matrices)
#
# ## Description
# This notebook calculates Jaccard similarity between GO terms based on
# gene overlap. Terms with Jaccard > 0.1 are filtered using greedy pairwise
# removal, keeping the term with greater percent change in gene count.
#
# **Gene Classification:**
# - 2016 datasets contain only stable genes (present in both 2016 and 2024)
# - 2024 datasets contain only added genes (only in 2024, not in 2016)
#
# **Note:** Neo4j ID mapping is deferred to notebook 2 when the Docker stack is running.

# %% papermill={"duration": 1.744295, "end_time": "2025-12-09T20:04:07.632470", "exception": false, "start_time": "2025-12-09T20:04:05.888175", "status": "completed"}
# Standard library
import os
import sys
from pathlib import Path

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# Setup repo root for consistent paths
if Path.cwd().name == "notebooks":
    repo_root = Path("..").resolve()
else:
    repo_root = Path.cwd()

sys.path.insert(0, str(repo_root))

# Project modules
from src.similarity import (
    load_or_calculate_jaccard,
    get_upper_triangle,
    find_similar_pairs,
    count_pairs_below_threshold
)
from src.filtering import filter_overlapping_go_terms, DEFAULT_JACCARD_THRESHOLD
from src.visualization import (
    create_diagonal_mask,
    plot_clustered_heatmap,
    plot_similarity_distribution,
    plot_gene_counts_comparison
)

# Create output directories
(repo_root / 'output/jaccard_similarity').mkdir(parents=True, exist_ok=True)
(repo_root / 'output/intermediate').mkdir(parents=True, exist_ok=True)
(repo_root / 'output/images').mkdir(parents=True, exist_ok=True)

print(f"Repo root: {repo_root}")

# %% papermill={"duration": 0.010929, "end_time": "2025-12-09T20:04:07.645242", "exception": false, "start_time": "2025-12-09T20:04:07.634313", "status": "completed"} tags=["parameters"]
# Parameters (for papermill)
FORCE_RECOMPUTE = False

# %% papermill={"duration": 0.110589, "end_time": "2025-12-09T20:04:07.764373", "exception": false, "start_time": "2025-12-09T20:04:07.653784", "status": "completed"}
# Load data from previous notebooks
hetio_BPpG_2016_stable = pd.read_csv(
    repo_root / 'output/intermediate/hetio_bppg_2016_stable.csv'
)
upd_go_bp_2024_added = pd.read_csv(
    repo_root / 'output/intermediate/upd_go_bp_2024_added.csv'
)
dataset1_all_growth = pd.read_csv(
    repo_root / 'output/intermediate/dataset1_all_growth.csv'
)
dataset2_parents = pd.read_csv(
    repo_root / 'output/intermediate/dataset2_parents.csv'
)

print('Loaded classified gene datasets:')
print(f'  hetio_BPpG_2016_stable: {len(hetio_BPpG_2016_stable)} rows (stable genes)')
print(f'  upd_go_bp_2024_added: {len(upd_go_bp_2024_added)} rows (added genes)')
print(f'  dataset1_all_growth: {len(dataset1_all_growth)} GO terms')
print(f'  dataset2_parents: {len(dataset2_parents)} GO terms')

# %% papermill={"duration": 0.489584, "end_time": "2025-12-09T20:04:08.258893", "exception": false, "start_time": "2025-12-09T20:04:07.769309", "status": "completed"}
# Ensure output directory exists
output_dir = repo_root / "output/images"
output_dir.mkdir(parents=True, exist_ok=True)

# Sort both datasets
dataset1_sorted = dataset1_all_growth.sort_values(
    by="no_of_genes_in_hetio_GO_2016", ascending=True
).reset_index(drop=True)
dataset2_sorted = dataset2_parents.sort_values(
    by="no_of_genes_in_hetio_GO_2016", ascending=True
).reset_index(drop=True)

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# Dataset 1: All Growth Terms
plot_gene_counts_comparison(
    dataset1_sorted, axes[0],
    "Dataset 1: All Growth Terms", len(dataset1_sorted)
)

# Dataset 2: Parent Terms
plot_gene_counts_comparison(
    dataset2_sorted, axes[1],
    "Dataset 2: Parent Terms", len(dataset2_sorted)
)

plt.tight_layout()

# Save figure
plt.savefig(output_dir / "genes_per_go_2016_vs_2024_both_datasets.pdf",
            format="pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "genes_per_go_2016_vs_2024_both_datasets.jpeg",
            format="jpeg", dpi=300, bbox_inches='tight')
plt.show()

print(f"Figure saved to {output_dir}/genes_per_go_2016_vs_2024_both_datasets.pdf")

# %% papermill={"duration": 0.021982, "end_time": "2025-12-09T20:04:08.290075", "exception": false, "start_time": "2025-12-09T20:04:08.268093", "status": "completed"}
# Jaccard similarity functions are imported from src/similarity.py
# See cell-1 for imports: load_or_calculate_jaccard, which uses
# calculate_jaccard_similarity_optimized internally

print("Using Jaccard similarity functions from src/similarity module")

# %% papermill={"duration": 41.45052, "end_time": "2025-12-09T20:04:49.758202", "exception": false, "start_time": "2025-12-09T20:04:08.307682", "status": "completed"}
# Prepare gene sets for each GO term in both datasets

# Define cache directory first
jaccard_cache_dir = repo_root / "output/jaccard_similarity"
jaccard_cache_dir.mkdir(parents=True, exist_ok=True)

# Get GO term sets for filtering
go_terms_dataset1 = set(dataset1_all_growth['go_id'])
go_terms_dataset2 = set(dataset2_parents['go_id'])

# Dataset 1: 2016 data
dataset1_2016_go_genes = {}
for go_id in dataset1_all_growth['go_id']:
    genes = set(hetio_BPpG_2016_stable[hetio_BPpG_2016_stable['go_id'] == go_id]['entrez_gene_id'])
    if len(genes) > 0:
        dataset1_2016_go_genes[go_id] = genes

# Dataset 1: 2024 data
dataset1_2024_go_genes = {}
for go_id in dataset1_all_growth['go_id']:
    genes = set(upd_go_bp_2024_added[upd_go_bp_2024_added['go_id'] == go_id]['entrez_gene_id'])
    if len(genes) > 0:
        dataset1_2024_go_genes[go_id] = genes

# Dataset 2: 2016 data
dataset2_2016_go_genes = {}
for go_id in dataset2_parents['go_id']:
    genes = set(hetio_BPpG_2016_stable[hetio_BPpG_2016_stable['go_id'] == go_id]['entrez_gene_id'])
    if len(genes) > 0:
        dataset2_2016_go_genes[go_id] = genes

# Dataset 2: 2024 data
dataset2_2024_go_genes = {}
for go_id in dataset2_parents['go_id']:
    genes = set(upd_go_bp_2024_added[upd_go_bp_2024_added['go_id'] == go_id]['entrez_gene_id'])
    if len(genes) > 0:
        dataset2_2024_go_genes[go_id] = genes

print("Gene sets prepared for Jaccard similarity calculation")
print("=" * 70)
print("Dataset 1 (All Growth Terms):")
print(f"  2016: {len(dataset1_2016_go_genes)} GO terms")
print(f"  2024: {len(dataset1_2024_go_genes)} GO terms")
print(f"\nDataset 2 (Parent Terms):")
print(f"  2016: {len(dataset2_2016_go_genes)} GO terms")
print(f"  2024: {len(dataset2_2024_go_genes)} GO terms")
print("=" * 70)

# %% papermill={"duration": 0.015933, "end_time": "2025-12-09T20:04:49.789015", "exception": false, "start_time": "2025-12-09T20:04:49.773082", "status": "completed"}
# Check if Jaccard similarity cache files exist
jaccard_cache_dir = repo_root / "output/jaccard_similarity"
cache_files = {
    'dataset1_2016': jaccard_cache_dir / "jaccard_similarity_dataset1_2016.csv",
    'dataset1_2024': jaccard_cache_dir / "jaccard_similarity_dataset1_2024.csv",
    'dataset2_2016': jaccard_cache_dir / "jaccard_similarity_dataset2_2016.csv",
    'dataset2_2024': jaccard_cache_dir / "jaccard_similarity_dataset2_2024.csv"
}

print("Checking for cached Jaccard similarity matrices...")
print("=" * 70)
all_cached = True
for name, path in cache_files.items():
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"FOUND: {name:20s} ({size_kb:.1f} KB)")
    else:
        print(f"NOT FOUND: {name:20s} - will need to compute")
        all_cached = False

if all_cached:
    print("\nAll cache files exist. Loading should be instant.")
else:
    print("\nSome cache files missing. First computation may take 30-60 seconds.")
    print("Subsequent runs will be instant once cached.")

print(f"\nTo force recalculation, delete the cache directory:")
print(f"  rm -rf {jaccard_cache_dir}")
print("=" * 70)

# %% papermill={"duration": 0.618931, "end_time": "2025-12-09T20:04:50.422608", "exception": false, "start_time": "2025-12-09T20:04:49.803677", "status": "completed"}
# Load or calculate Jaccard similarity matrices with progress tracking

print("\n" + "=" * 70)
print("JACCARD SIMILARITY CALCULATION")
if FORCE_RECOMPUTE:
    print("(FORCE_RECOMPUTE enabled - ignoring cache)")
print("=" * 70)

# Delete cache files if forcing recompute
if FORCE_RECOMPUTE:
    for name in ['dataset1_2016', 'dataset1_2024', 'dataset2_2016', 'dataset2_2024']:
        cache_file = jaccard_cache_dir / f"jaccard_similarity_{name}.csv"
        if cache_file.exists():
            cache_file.unlink()
            print(f"Deleted cache: {cache_file.name}")

# Track whether all matrices were loaded from cache
all_from_cache = True

# Calculate for Dataset 1 - 2016
print("\n[1/4] Dataset 1 - 2016...")
jaccard_dataset1_2016, from_cache = load_or_calculate_jaccard(
    dataset1_2016_go_genes, 
    str(jaccard_cache_dir / "jaccard_similarity_dataset1_2016.csv")
)
all_from_cache = all_from_cache and from_cache

# Calculate for Dataset 1 - 2024
print("\n[2/4] Dataset 1 - 2024...")
jaccard_dataset1_2024, from_cache = load_or_calculate_jaccard(
    dataset1_2024_go_genes,
    str(jaccard_cache_dir / "jaccard_similarity_dataset1_2024.csv")
)
all_from_cache = all_from_cache and from_cache

# Calculate for Dataset 2 - 2016
print("\n[3/4] Dataset 2 - 2016...")
jaccard_dataset2_2016, from_cache = load_or_calculate_jaccard(
    dataset2_2016_go_genes,
    str(jaccard_cache_dir / "jaccard_similarity_dataset2_2016.csv")
)
all_from_cache = all_from_cache and from_cache

# Calculate for Dataset 2 - 2024
print("\n[4/4] Dataset 2 - 2024...")
jaccard_dataset2_2024, from_cache = load_or_calculate_jaccard(
    dataset2_2024_go_genes,
    str(jaccard_cache_dir / "jaccard_similarity_dataset2_2024.csv")
)
all_from_cache = all_from_cache and from_cache

print("\n" + "=" * 70)
print("ALL JACCARD MATRICES READY")
if all_from_cache:
    print("(All loaded from cache)")
print("=" * 70)

# Summary statistics
print("\nDataset 1 - 2016 Jaccard Similarity:")
print(f"  Mean: {jaccard_dataset1_2016.values[np.triu_indices_from(jaccard_dataset1_2016.values, k=1)].mean():.4f}")
print(f"  Median: {np.median(jaccard_dataset1_2016.values[np.triu_indices_from(jaccard_dataset1_2016.values, k=1)]):.4f}")
print(f"  Max (off-diagonal): {jaccard_dataset1_2016.values[np.triu_indices_from(jaccard_dataset1_2016.values, k=1)].max():.4f}")

print("\nDataset 1 - 2024 Jaccard Similarity:")
print(f"  Mean: {jaccard_dataset1_2024.values[np.triu_indices_from(jaccard_dataset1_2024.values, k=1)].mean():.4f}")
print(f"  Median: {np.median(jaccard_dataset1_2024.values[np.triu_indices_from(jaccard_dataset1_2024.values, k=1)]):.4f}")
print(f"  Max (off-diagonal): {jaccard_dataset1_2024.values[np.triu_indices_from(jaccard_dataset1_2024.values, k=1)].max():.4f}")

print("\nDataset 2 - 2016 Jaccard Similarity:")
print(f"  Mean: {jaccard_dataset2_2016.values[np.triu_indices_from(jaccard_dataset2_2016.values, k=1)].mean():.4f}")
print(f"  Median: {np.median(jaccard_dataset2_2016.values[np.triu_indices_from(jaccard_dataset2_2016.values, k=1)]):.4f}")
print(f"  Max (off-diagonal): {jaccard_dataset2_2016.values[np.triu_indices_from(jaccard_dataset2_2016.values, k=1)].max():.4f}")

print("\nDataset 2 - 2024 Jaccard Similarity:")
print(f"  Mean: {jaccard_dataset2_2024.values[np.triu_indices_from(jaccard_dataset2_2024.values, k=1)].mean():.4f}")
print(f"  Median: {np.median(jaccard_dataset2_2024.values[np.triu_indices_from(jaccard_dataset2_2024.values, k=1)]):.4f}")
print(f"  Max (off-diagonal): {jaccard_dataset2_2024.values[np.triu_indices_from(jaccard_dataset2_2024.values, k=1)].max():.4f}")

# %% papermill={"duration": 0.482952, "end_time": "2025-12-09T20:04:50.909052", "exception": false, "start_time": "2025-12-09T20:04:50.426100", "status": "completed"}
# Identify highly similar GO term pairs (Jaccard > 0.5)

print("Highly Similar GO Term Pairs (Jaccard Similarity > 0.5)")
print("=" * 80)

print("\nDataset 1 - 2016:")
similar_d1_2016 = find_similar_pairs(jaccard_dataset1_2016, threshold=0.5, top_n=10)
if len(similar_d1_2016) > 0:
    print(f"Found {len(similar_d1_2016)} pairs with similarity > 0.5")
    display(similar_d1_2016)
else:
    print("No pairs with similarity > 0.5")

print("\nDataset 1 - 2024:")
similar_d1_2024 = find_similar_pairs(jaccard_dataset1_2024, threshold=0.5, top_n=10)
if len(similar_d1_2024) > 0:
    print(f"Found {len(similar_d1_2024)} pairs with similarity > 0.5")
    display(similar_d1_2024)
else:
    print("No pairs with similarity > 0.5")

print("\nDataset 2 - 2016:")
similar_d2_2016 = find_similar_pairs(jaccard_dataset2_2016, threshold=0.5, top_n=10)
if len(similar_d2_2016) > 0:
    print(f"Found {len(similar_d2_2016)} pairs with similarity > 0.5")
    display(similar_d2_2016)
else:
    print("No pairs with similarity > 0.5")

print("\nDataset 2 - 2024:")
similar_d2_2024 = find_similar_pairs(jaccard_dataset2_2024, threshold=0.5, top_n=10)
if len(similar_d2_2024) > 0:
    print(f"Found {len(similar_d2_2024)} pairs with similarity > 0.5")
    display(similar_d2_2024)
else:
    print("No pairs with similarity > 0.5")

# Count pairs at different thresholds
print("\n" + "=" * 80)
print("Summary: Number of GO term pairs exceeding similarity thresholds")
print("=" * 80)

thresholds = [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
summary_data = []

for threshold in thresholds:
    summary_data.append({
        'Threshold': threshold,
        'Dataset1_2016': len(find_similar_pairs(jaccard_dataset1_2016, threshold=threshold, top_n=1000)),
        'Dataset1_2024': len(find_similar_pairs(jaccard_dataset1_2024, threshold=threshold, top_n=1000)),
        'Dataset2_2016': len(find_similar_pairs(jaccard_dataset2_2016, threshold=threshold, top_n=1000)),
        'Dataset2_2024': len(find_similar_pairs(jaccard_dataset2_2024, threshold=threshold, top_n=1000))
    })

summary_df = pd.DataFrame(summary_data)
display(summary_df)

# %% [markdown] papermill={"duration": 0.021308, "end_time": "2025-12-09T20:04:50.933426", "exception": false, "start_time": "2025-12-09T20:04:50.912118", "status": "completed"}
# ## Filter Overlapping GO Terms
#
# Remove redundant GO terms with Jaccard similarity > 0.1 (configurable threshold).
#
# Approach:
# - Use greedy pairwise filtering to remove redundant terms
# - For each pair above threshold, remove the GO term with lower absolute percent change in gene count
# - Process pairs in order of highest similarity first
# - Filter based on 2016 Jaccard matrices (baseline year)
# - This ensures GO terms used in downstream analysis are reasonably independent
#
# The threshold of 0.1 corresponds to approximately the 96th percentile of non-zero similarities in both datasets.

# %% papermill={"duration": 0.017124, "end_time": "2025-12-09T20:04:50.961878", "exception": false, "start_time": "2025-12-09T20:04:50.944754", "status": "completed"}
# Filter function is imported from src/filtering.py (see imports cell)
# Uses greedy pairwise filtering:
# - For each pair with Jaccard > threshold (default 0.1)
# - Remove the GO term with lower absolute percent change
# - Returns: filtered_dataset, removed_terms, removal_df

print(f"Using filter_overlapping_go_terms from src/filtering module")
print(f"Default Jaccard threshold: {DEFAULT_JACCARD_THRESHOLD}")

# %% papermill={"duration": 0.481774, "end_time": "2025-12-09T20:04:51.457455", "exception": false, "start_time": "2025-12-09T20:04:50.975681", "status": "completed"}
# Filter Dataset 1 based on 2016 Jaccard matrix
print('Filtering Dataset 1 (all growth terms)...')
print('=' * 80)

dataset1_filtered, removed_d1, removal_details_d1 = filter_overlapping_go_terms(
    jaccard_dataset1_2016, 
    dataset1_all_growth,
    threshold=0.1
)

# Statistics
print(f'\nDataset 1 Statistics:')
print(f'  GO terms before filtering: {len(dataset1_all_growth)}')
print(f'  GO terms after filtering:  {len(dataset1_filtered)}')
print(f'  GO terms removed:          {len(removed_d1)}')
print(f'  Reduction:                 {len(removed_d1)/len(dataset1_all_growth)*100:.1f}%')

# Show removed terms (if any)
if len(removal_details_d1) > 0:
    print(f'\nRemoved Terms Details (Dataset 1, first 10):')
    print(removal_details_d1[['removed_go_id', 'kept_go_id', 'jaccard_similarity', 
                               'removed_pct_change', 'kept_pct_change']].head(10).to_string(index=False))
    print(f'\n... and {len(removal_details_d1) - 10} more pairs removed') if len(removal_details_d1) > 10 else None
else:
    print(f'\nNo overlapping GO terms found (all Jaccard < 0.1)')

# %% papermill={"duration": 0.224732, "end_time": "2025-12-09T20:04:51.685489", "exception": false, "start_time": "2025-12-09T20:04:51.460757", "status": "completed"}
# Filter Dataset 2 based on 2016 Jaccard matrix
print('Filtering Dataset 2 (parent terms)...')
print('=' * 80)

dataset2_filtered, removed_d2, removal_details_d2 = filter_overlapping_go_terms(
    jaccard_dataset2_2016,
    dataset2_parents,
    threshold=0.1
)

# Statistics
print(f'\nDataset 2 Statistics:')
print(f'  GO terms before filtering: {len(dataset2_parents)}')
print(f'  GO terms after filtering:  {len(dataset2_filtered)}')
print(f'  GO terms removed:          {len(removed_d2)}')
print(f'  Reduction:                 {len(removed_d2)/len(dataset2_parents)*100:.1f}%')

# Show removed terms (if any)
if len(removal_details_d2) > 0:
    print(f'\nRemoved Terms Details (Dataset 2, first 10):')
    print(removal_details_d2[['removed_go_id', 'kept_go_id', 'jaccard_similarity',
                               'removed_pct_change', 'kept_pct_change']].head(10).to_string(index=False))
    if len(removal_details_d2) > 10:
        print(f'\n... and {len(removal_details_d2) - 10} more pairs removed')
else:
    print(f'\nNo overlapping GO terms found (all Jaccard < 0.1)')

# %% papermill={"duration": 0.933961, "end_time": "2025-12-09T20:04:52.622796", "exception": false, "start_time": "2025-12-09T20:04:51.688835", "status": "completed"}
# Visualize filtering effect: before vs after Jaccard filtering
# Show 50x50 hierarchical heatmaps of most similar terms before and after filtering

heatmap_pdf = output_dir / "jaccard_filtering_effect_comparison.pdf"
heatmap_jpeg = output_dir / "jaccard_filtering_effect_comparison.jpeg"
heatmaps_exist = heatmap_pdf.exists() and heatmap_jpeg.exists()

skip_heatmaps = all_from_cache and heatmaps_exist and not FORCE_RECOMPUTE

if skip_heatmaps:
    print("Skipping heatmap generation (using cached data and heatmaps exist)")
    print(f"  - {heatmap_pdf.name}")
    print(f"  - {heatmap_jpeg.name}")
else:
    print("Generating before/after filtering comparison heatmaps...")
    
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    n_sample = 50
    
    def get_top_similar_terms(jaccard_matrix, n=50):
        """Get n terms with highest mean similarity to other terms."""
        mean_sim = (jaccard_matrix.sum(axis=1) - 1) / (len(jaccard_matrix) - 1)
        top_terms = mean_sim.nlargest(n).index.tolist()
        return top_terms
    
    # Before filtering: top 50 from full dataset
    top_terms_d1_before = get_top_similar_terms(jaccard_dataset1_2016, n_sample)
    top_terms_d2_before = get_top_similar_terms(jaccard_dataset2_2016, n_sample)
    
    # After filtering: top 50 from filtered dataset only
    filtered_go_d1 = dataset1_filtered['go_id'].tolist()
    filtered_go_d2 = dataset2_filtered['go_id'].tolist()
    jaccard_d1_filtered = jaccard_dataset1_2016.loc[filtered_go_d1, filtered_go_d1]
    jaccard_d2_filtered = jaccard_dataset2_2016.loc[filtered_go_d2, filtered_go_d2]
    top_terms_d1_after = get_top_similar_terms(jaccard_d1_filtered, n_sample)
    top_terms_d2_after = get_top_similar_terms(jaccard_d2_filtered, n_sample)
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    
    # Common color scale
    vmin, vmax = 0, 0.5
    cmap = "YlOrRd"
    
    def plot_hierarchical_heatmap(similarity_matrix, terms, ax, title):
        """Plot hierarchical clustered heatmap for given terms."""
        subset = similarity_matrix.loc[terms, terms]
        
        if len(terms) > 1:
            distance_matrix = 1 - subset.values
            np.fill_diagonal(distance_matrix, 0)
            condensed = squareform(distance_matrix, checks=False)
            Z = linkage(condensed, method='average')
            dend = dendrogram(Z, no_plot=True)
            order = dend['leaves']
            subset = subset.iloc[order, order]
        
        sns.heatmap(
            subset, cmap=cmap, vmin=vmin, vmax=vmax,
            xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Jaccard Similarity'},
            ax=ax, square=True
        )
        ax.set_title(title, fontsize=11)
    
    # Dataset 1 - Before filtering
    plot_hierarchical_heatmap(
        jaccard_dataset1_2016, top_terms_d1_before, axes[0, 0],
        f'Dataset 1 BEFORE Filtering\n(Top {n_sample} most similar of {len(jaccard_dataset1_2016)} terms)'
    )
    
    # Dataset 1 - After filtering
    plot_hierarchical_heatmap(
        jaccard_d1_filtered, top_terms_d1_after, axes[0, 1],
        f'Dataset 1 AFTER Filtering\n(Top {n_sample} most similar of {len(filtered_go_d1)} terms)'
    )
    
    # Dataset 2 - Before filtering
    plot_hierarchical_heatmap(
        jaccard_dataset2_2016, top_terms_d2_before, axes[1, 0],
        f'Dataset 2 BEFORE Filtering\n(Top {n_sample} most similar of {len(jaccard_dataset2_2016)} terms)'
    )
    
    # Dataset 2 - After filtering
    plot_hierarchical_heatmap(
        jaccard_d2_filtered, top_terms_d2_after, axes[1, 1],
        f'Dataset 2 AFTER Filtering\n(Top {n_sample} most similar of {len(filtered_go_d2)} terms)'
    )
    
    plt.suptitle('Effect of Jaccard Filtering: Top 50 Most Similar GO Terms\n(Same color scale: 0-0.5)', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(heatmap_pdf, format="pdf", dpi=300, bbox_inches='tight')
    plt.savefig(heatmap_jpeg, format="jpeg", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nHeatmaps saved to {output_dir}/")

# %% papermill={"duration": 0.051924, "end_time": "2025-12-09T20:04:52.678198", "exception": false, "start_time": "2025-12-09T20:04:52.626274", "status": "completed"}
# Combined summary table
print('\nCombined Summary: Before and After Filtering')
print('=' * 80)

# Calculate unique genes (filter to dataset GO terms first)
bp_d1 = hetio_BPpG_2016_stable[hetio_BPpG_2016_stable['go_id'].isin(go_terms_dataset1)]
bp_d2 = hetio_BPpG_2016_stable[hetio_BPpG_2016_stable['go_id'].isin(go_terms_dataset2)]

genes_d1_before = bp_d1['entrez_gene_id'].nunique()
genes_d1_after = bp_d1[bp_d1['go_id'].isin(dataset1_filtered['go_id'])]['entrez_gene_id'].nunique()

genes_d2_before = bp_d2['entrez_gene_id'].nunique()
genes_d2_after = bp_d2[bp_d2['go_id'].isin(dataset2_filtered['go_id'])]['entrez_gene_id'].nunique()

# GO-gene pairs
pairs_d1_before = len(bp_d1)
pairs_d1_after = len(bp_d1[bp_d1['go_id'].isin(dataset1_filtered['go_id'])])

pairs_d2_before = len(bp_d2)
pairs_d2_after = len(bp_d2[bp_d2['go_id'].isin(dataset2_filtered['go_id'])])

summary_data = {
    'Metric': ['GO terms', 'Unique genes', 'GO-gene pairs', 'Terms removed'],
    'Dataset 1 Before': [
        len(dataset1_all_growth),
        genes_d1_before,
        pairs_d1_before,
        '-'
    ],
    'Dataset 1 After': [
        len(dataset1_filtered),
        genes_d1_after,
        pairs_d1_after,
        len(removed_d1)
    ],
    'Dataset 2 Before': [
        len(dataset2_parents),
        genes_d2_before,
        pairs_d2_before,
        '-'
    ],
    'Dataset 2 After': [
        len(dataset2_filtered),
        genes_d2_after,
        pairs_d2_after,
        len(removed_d2)
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f'\nFiltering complete. Use dataset1_filtered and dataset2_filtered for downstream analyses.')

# %% papermill={"duration": 0.03246, "end_time": "2025-12-09T20:04:52.713985", "exception": false, "start_time": "2025-12-09T20:04:52.681525", "status": "completed"}
# Create GO-gene pair dataframes with filtered GO terms
print('Creating GO-gene pair dataframes with filtered GO terms...')
print('=' * 80)

# Filter Dataset 1 BPpG to filtered GO terms
hetio_BPpG_dataset1_filtered = hetio_BPpG_2016_stable[
    hetio_BPpG_2016_stable['go_id'].isin(dataset1_filtered['go_id'])
].copy()

print(f'Dataset 1 GO-gene pairs:')
print(f'  Before filtering: {pairs_d1_before:,}')
print(f'  After filtering:  {len(hetio_BPpG_dataset1_filtered):,}')
print(f'  Removed:          {pairs_d1_before - len(hetio_BPpG_dataset1_filtered):,}')

# Filter Dataset 2 BPpG to filtered GO terms
hetio_BPpG_dataset2_filtered = hetio_BPpG_2016_stable[
    hetio_BPpG_2016_stable['go_id'].isin(dataset2_filtered['go_id'])
].copy()

print(f'\nDataset 2 GO-gene pairs:')
print(f'  Before filtering: {pairs_d2_before:,}')
print(f'  After filtering:  {len(hetio_BPpG_dataset2_filtered):,}')
print(f'  Removed:          {pairs_d2_before - len(hetio_BPpG_dataset2_filtered):,}')

print(f'\nFiltered dataframes created:')
print(f'  - hetio_BPpG_dataset1_filtered')
print(f'  - hetio_BPpG_dataset2_filtered')
print(f'\nUse these for downstream DWPC calculations to ensure GO term independence.')

# %% [markdown] papermill={"duration": 0.013913, "end_time": "2025-12-09T20:04:52.741800", "exception": false, "start_time": "2025-12-09T20:04:52.727887", "status": "completed"}
# ## Visualize Filtered Datasets
#
# Compare gene counts (2016 vs 2024) for filtered GO term sets.

# %% papermill={"duration": 0.132644, "end_time": "2025-12-09T20:04:52.887730", "exception": false, "start_time": "2025-12-09T20:04:52.755086", "status": "completed"}
# Visualize filtered datasets

# Sort filtered datasets
dataset1_filt_sorted = dataset1_filtered.sort_values(
    by='no_of_genes_in_hetio_GO_2016', 
    ascending=True
).reset_index(drop=True)

dataset2_filt_sorted = dataset2_filtered.sort_values(
    by='no_of_genes_in_hetio_GO_2016',
    ascending=True
).reset_index(drop=True)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# Dataset 1
axes[0].scatter(
    dataset1_filt_sorted.index,
    dataset1_filt_sorted['no_of_genes_in_hetio_GO_2016'],
    alpha=0.6,
    label='2016',
    s=20
)
axes[0].scatter(
    dataset1_filt_sorted.index,
    dataset1_filt_sorted['no_of_genes_in_GO_2024'],
    alpha=0.6,
    label='2024',
    s=20
)
axes[0].set_xlabel('GO Term Index (sorted by 2016 gene count)')
axes[0].set_ylabel('Number of Genes')
axes[0].set_title(f'Dataset 1: All Growth Terms (Filtered)\nn={len(dataset1_filtered)} GO terms')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Dataset 2
axes[1].scatter(
    dataset2_filt_sorted.index,
    dataset2_filt_sorted['no_of_genes_in_hetio_GO_2016'],
    alpha=0.6,
    label='2016',
    s=20
)
axes[1].scatter(
    dataset2_filt_sorted.index,
    dataset2_filt_sorted['no_of_genes_in_GO_2024'],
    alpha=0.6,
    label='2024',
    s=20
)
axes[1].set_xlabel('GO Term Index (sorted by 2016 gene count)')
axes[1].set_ylabel('Number of Genes')
axes[1].set_title(f'Dataset 2: Parent Terms (Filtered)\nn={len(dataset2_filtered)} GO terms')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('Visualization complete')

# %% papermill={"duration": 0.023333, "end_time": "2025-12-09T20:04:52.915070", "exception": false, "start_time": "2025-12-09T20:04:52.891737", "status": "completed"}
# Note: Jaccard similarity matrices are automatically cached during calculation
# Cache location: output/jaccard_similarity/
# Files:
#   - jaccard_similarity_dataset1_2016.csv
#   - jaccard_similarity_dataset1_2024.csv
#   - jaccard_similarity_dataset2_2016.csv
#   - jaccard_similarity_dataset2_2024.csv
#
# To force recalculation, delete the cache files and re-run the calculation cell

print("Jaccard similarity matrices are cached at:")
print(f"  {jaccard_cache_dir}/")
print("\nCached files:")
for dataset in ['dataset1_2016', 'dataset1_2024', 'dataset2_2016', 'dataset2_2024']:
    cache_file = jaccard_cache_dir / f"jaccard_similarity_{dataset}.csv"
    if cache_file.exists():
        file_size = cache_file.stat().st_size / 1024
        print(f"  jaccard_similarity_{dataset}.csv ({file_size:.1f} KB)")
    else:
        print(f"  jaccard_similarity_{dataset}.csv (not found)")

# %% [markdown] papermill={"duration": 0.022533, "end_time": "2025-12-09T20:04:52.976586", "exception": false, "start_time": "2025-12-09T20:04:52.954053", "status": "completed"}
# ### Summary: Jaccard Similarity Analysis
#
# This analysis computed pairwise Jaccard similarity between GO terms based on gene overlap for both datasets and years. Key insights:
#
# **Jaccard Similarity Formula:**
# - J(A,B) = |A ∩ B| / |A ∪ B|
# - Ranges from 0 (no overlap) to 1 (complete overlap)
#
# **Implementation:**
# - Uses sklearn.metrics.pairwise_distances for optimized computation (10-50x faster than nested loops)
# - Results are automatically cached to disk for instant loading on subsequent runs
# - Cache location: output/jaccard_similarity/
#
# **Analysis Components:**
# 1. Computed similarity matrices for all GO term pairs in both datasets
# 2. Generated clustered heatmaps showing pairwise similarity patterns with hierarchical clustering
# 3. Diagonal removed from heatmaps to focus on GO term relationships
# 4. Analyzed distribution of similarity scores
# 5. Identified highly similar GO term pairs that may represent functional redundancy or hierarchical relationships
#
# **Outputs:**
# - Similarity matrices cached in output/jaccard_similarity/
# - Clustered heatmaps and distributions saved to output/images/
#
# **Applications:**
# - Identify redundant or highly overlapping GO terms
# - Understand functional relationships between biological processes
# - Inform feature selection for machine learning (remove highly correlated features)
# - Validate that filtered datasets have appropriate diversity
# - Clustering reveals functional modules and related biological processes

# %% [markdown] papermill={"duration": 0.017294, "end_time": "2025-12-09T20:04:53.011774", "exception": false, "start_time": "2025-12-09T20:04:52.994480", "status": "completed"}
# ### Gene Overlap Analysis: Jaccard Similarity Between GO Terms
#
# Calculate pairwise Jaccard similarity between GO terms based on their annotated genes. This measures how much gene sets overlap between different GO terms, which can reveal functional relationships and redundancy in the ontology.

# %% papermill={"duration": 0.036593, "end_time": "2025-12-09T20:04:53.065796", "exception": false, "start_time": "2025-12-09T20:04:53.029203", "status": "completed"}
# Save filtered datasets - 2016 data
print('Saving filtered datasets (2016)...')

output_intermediate = repo_root / 'output/intermediate'

dataset1_filtered.to_csv(
    output_intermediate / 'dataset1_filtered.csv',
    index=False
)
print(f'Saved dataset1_filtered.csv: '
      f'{len(dataset1_filtered)} GO terms')

dataset2_filtered.to_csv(
    output_intermediate / 'dataset2_filtered.csv',
    index=False
)
print(f'Saved dataset2_filtered.csv: '
      f'{len(dataset2_filtered)} GO terms')

hetio_BPpG_dataset1_filtered.to_csv(
    output_intermediate / 'hetio_bppg_dataset1_filtered.csv',
    index=False
)
print(f'Saved hetio_bppg_dataset1_filtered.csv: '
      f'{len(hetio_BPpG_dataset1_filtered)} rows')

hetio_BPpG_dataset2_filtered.to_csv(
    output_intermediate / 'hetio_bppg_dataset2_filtered.csv',
    index=False
)
print(f'Saved hetio_bppg_dataset2_filtered.csv: '
      f'{len(hetio_BPpG_dataset2_filtered)} rows')

# %% papermill={"duration": 0.096786, "end_time": "2025-12-09T20:04:53.180121", "exception": false, "start_time": "2025-12-09T20:04:53.083335", "status": "completed"}
# Filter and save 2024 datasets with Jaccard-filtered GO terms
print('\nFiltering and saving 2024 datasets...')
print('=' * 80)

# Filter 2024 Dataset 1 to only Jaccard-filtered GO terms
upd_go_bp_2024_dataset1_filtered = upd_go_bp_2024_added[
    upd_go_bp_2024_added['go_id'].isin(dataset1_filtered['go_id'])
].copy()

print(f'Dataset 1 (2024):')
print(f'  GO-gene pairs: {len(upd_go_bp_2024_dataset1_filtered):,}')
print(f'  Unique GO terms: {upd_go_bp_2024_dataset1_filtered["go_id"].nunique()}')
print(f'  Unique genes: {upd_go_bp_2024_dataset1_filtered["entrez_gene_id"].nunique()}')

# Filter 2024 Dataset 2 to only Jaccard-filtered GO terms
upd_go_bp_2024_dataset2_filtered = upd_go_bp_2024_added[
    upd_go_bp_2024_added['go_id'].isin(dataset2_filtered['go_id'])
].copy()

print(f'\nDataset 2 (2024):')
print(f'  GO-gene pairs: {len(upd_go_bp_2024_dataset2_filtered):,}')
print(f'  Unique GO terms: {upd_go_bp_2024_dataset2_filtered["go_id"].nunique()}')
print(f'  Unique genes: {upd_go_bp_2024_dataset2_filtered["entrez_gene_id"].nunique()}')

# Save 2024 filtered datasets
output_intermediate = repo_root / 'output/intermediate'
upd_go_bp_2024_dataset1_filtered.to_csv(
    output_intermediate / 'hetio_bppg_dataset1_2024_filtered.csv',
    index=False
)
print(f'\nSaved hetio_bppg_dataset1_2024_filtered.csv: {len(upd_go_bp_2024_dataset1_filtered):,} rows')

upd_go_bp_2024_dataset2_filtered.to_csv(
    output_intermediate / 'hetio_bppg_dataset2_2024_filtered.csv',
    index=False
)
print(f'Saved hetio_bppg_dataset2_2024_filtered.csv: {len(upd_go_bp_2024_dataset2_filtered):,} rows')

# %% papermill={"duration": 0.022869, "end_time": "2025-12-09T20:04:53.207168", "exception": false, "start_time": "2025-12-09T20:04:53.184299", "status": "completed"}
print('\n' + '=' * 80)
print('NOTEBOOK 1.3 COMPLETE - ALL DATASETS SAVED')
print('=' * 80)

print('\nOutput Summary:')
print('  GO Term Lists (year-agnostic):')
print(f'    - dataset1_filtered.csv ({len(dataset1_filtered)} GO terms)')
print(f'    - dataset2_filtered.csv ({len(dataset2_filtered)} GO terms)')

print('\n  2016 BP-Gene Associations:')
print(f'    - hetio_bppg_dataset1_filtered.csv ({len(hetio_BPpG_dataset1_filtered):,} pairs)')
print(f'    - hetio_bppg_dataset2_filtered.csv ({len(hetio_BPpG_dataset2_filtered):,} pairs)')

print('\n  2024 BP-Gene Associations:')
print(f'    - hetio_bppg_dataset1_2024_filtered.csv ({len(upd_go_bp_2024_dataset1_filtered):,} pairs)')
print(f'    - hetio_bppg_dataset2_2024_filtered.csv ({len(upd_go_bp_2024_dataset2_filtered):,} pairs)')

print('\nDataset Comparison:')
print(f'  Dataset 1: {len(hetio_BPpG_dataset1_filtered):,} (2016) -> {len(upd_go_bp_2024_dataset1_filtered):,} (2024) pairs')
print(f'  Dataset 2: {len(hetio_BPpG_dataset2_filtered):,} (2016) -> {len(upd_go_bp_2024_dataset2_filtered):,} (2024) pairs')

