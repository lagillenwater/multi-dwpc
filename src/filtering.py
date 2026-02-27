"""
GO term filtering functions.

This module provides functions for filtering GO terms based on IQR thresholds
and removing redundant terms with high Jaccard similarity.
"""

import pandas as pd
import numpy as np

# Default Jaccard similarity threshold for filtering overlapping GO terms.
# 0.1 corresponds to approximately the 96th percentile of non-zero
# similarities in typical GO term datasets.
DEFAULT_JACCARD_THRESHOLD = 0.1


def calculate_iqr_thresholds(df, gene_col='no_of_genes_in_hetio_GO_2016',
                             pct_col='pct_change_genes'):
    """
    Calculate IQR-based thresholds for filtering GO terms.

    Uses interquartile range to identify reasonable bounds for gene counts
    and percent change values.

    Parameters
    ----------
    df : pd.DataFrame
        GO term data with gene counts and percent changes
    gene_col : str
        Column name for gene counts
    pct_col : str
        Column name for percent change values

    Returns
    -------
    dict
        Dictionary with keys: min_genes, max_genes, median_genes,
        min_pct_change, max_pct_change, median_pct_change, total_terms
    """
    q1_genes = df[gene_col].quantile(0.25)
    q3_genes = df[gene_col].quantile(0.75)
    iqr_genes = q3_genes - q1_genes
    median_genes = df[gene_col].median()

    q1_pct = df[pct_col].quantile(0.25)
    q3_pct = df[pct_col].quantile(0.75)
    iqr_pct = q3_pct - q1_pct
    median_pct = df[pct_col].median()

    thresholds = {
        'min_genes': q1_genes - 1.5 * iqr_genes,
        'max_genes': q3_genes + 1.5 * iqr_genes,
        'median_genes': median_genes,
        'min_pct_change': q1_pct - 1.5 * iqr_pct,
        'max_pct_change': q3_pct + 1.5 * iqr_pct,
        'median_pct_change': median_pct,
        'total_terms': len(df)
    }

    return thresholds


def filter_go_terms_iqr(df, thresholds, min_genes=10,
                       gene_col='no_of_genes_in_hetio_GO_2016',
                       pct_col='pct_change_genes'):
    """
    Filter GO terms using IQR thresholds and minimum gene count.

    Parameters
    ----------
    df : pd.DataFrame
        GO term comparison data
    thresholds : dict
        Dictionary with min/max values from calculate_iqr_thresholds
    min_genes : int
        Minimum number of genes required (default 10)
    gene_col : str
        Column name for gene counts
    pct_col : str
        Column name for percent change values

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only GO terms within IQR bounds
        and above minimum gene count
    """
    mask = (
        (df[gene_col] >= min_genes) &
        (df[gene_col] >= thresholds['min_genes']) &
        (df[gene_col] <= thresholds['max_genes']) &
        (df[pct_col] >= thresholds['min_pct_change']) &
        (df[pct_col] <= thresholds['max_pct_change'])
    )
    return df.loc[mask].reset_index(drop=True)


def filter_overlapping_go_terms(jaccard_matrix, dataset,
                                threshold=DEFAULT_JACCARD_THRESHOLD):
    """
    Remove redundant GO terms with Jaccard similarity > threshold.

    Uses greedy pairwise filtering: for each pair above threshold,
    removes the GO term with lower absolute percent change in gene count.

    Parameters
    ----------
    jaccard_matrix : pd.DataFrame
        Symmetric matrix of Jaccard similarities (GO terms x GO terms)
    dataset : pd.DataFrame
        Dataset with columns: go_id, no_of_genes_in_hetio_GO_2016,
        no_of_genes_in_GO_2024
    threshold : float
        Jaccard similarity threshold for redundancy (default 0.1)

    Returns
    -------
    filtered_dataset : pd.DataFrame
        Dataset containing only non-redundant GO terms
    removed_terms : dict
        Mapping {removed_go_id: kept_go_id}
    removal_df : pd.DataFrame
        Detailed information about removed pairs
    """
    dataset_with_pct = dataset.copy()
    dataset_with_pct['pct_change'] = np.abs(
        (dataset_with_pct['no_of_genes_in_GO_2024'] -
         dataset_with_pct['no_of_genes_in_hetio_GO_2016']) /
        dataset_with_pct['no_of_genes_in_hetio_GO_2016'] * 100
    )

    pct_change_lookup = dict(zip(dataset_with_pct['go_id'],
                                 dataset_with_pct['pct_change']))

    # Find all pairs above threshold using vectorized upper triangle extraction
    go_terms = jaccard_matrix.index.tolist()
    n = len(go_terms)
    rows, cols = np.triu_indices(n, k=1)
    similarities = jaccard_matrix.values[rows, cols]

    above_threshold = similarities > threshold
    pair_indices = np.where(above_threshold)[0]

    pairs_to_filter = []
    for idx in pair_indices:
        i, j = rows[idx], cols[idx]
        go_id_i = go_terms[i]
        go_id_j = go_terms[j]
        similarity = similarities[idx]

        pct_i = pct_change_lookup.get(go_id_i, 0)
        pct_j = pct_change_lookup.get(go_id_j, 0)

        if pct_i >= pct_j:
            keep_id, remove_id = go_id_i, go_id_j
            keep_pct, remove_pct = pct_i, pct_j
        else:
            keep_id, remove_id = go_id_j, go_id_i
            keep_pct, remove_pct = pct_j, pct_i

        pairs_to_filter.append({
            'keep_id': keep_id,
            'remove_id': remove_id,
            'jaccard_similarity': similarity,
            'keep_pct': keep_pct,
            'remove_pct': remove_pct
        })

    # Greedy removal: process pairs by similarity (highest first)
    pairs_to_filter.sort(key=lambda x: x['jaccard_similarity'], reverse=True)

    terms_to_remove = set()
    removed_terms = {}
    removal_details = []

    for pair in pairs_to_filter:
        keep_id = pair['keep_id']
        remove_id = pair['remove_id']

        if keep_id in terms_to_remove or remove_id in terms_to_remove:
            continue

        terms_to_remove.add(remove_id)
        removed_terms[remove_id] = keep_id

        keep_row = dataset_with_pct[dataset_with_pct['go_id'] == keep_id].iloc[0]
        remove_row = dataset_with_pct[
            dataset_with_pct['go_id'] == remove_id
        ].iloc[0]

        removal_details.append({
            'removed_go_id': remove_id,
            'kept_go_id': keep_id,
            'jaccard_similarity': pair['jaccard_similarity'],
            'removed_genes_2016': remove_row['no_of_genes_in_hetio_GO_2016'],
            'removed_genes_2024': remove_row['no_of_genes_in_GO_2024'],
            'removed_pct_change': remove_row['pct_change'],
            'kept_genes_2016': keep_row['no_of_genes_in_hetio_GO_2016'],
            'kept_genes_2024': keep_row['no_of_genes_in_GO_2024'],
            'kept_pct_change': keep_row['pct_change']
        })

    filtered_dataset = dataset[~dataset['go_id'].isin(terms_to_remove)].copy()
    removal_df = pd.DataFrame(removal_details) if removal_details else pd.DataFrame()

    return filtered_dataset, removed_terms, removal_df
