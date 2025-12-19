"""
Similarity calculations for GO terms.

This module provides functions for calculating Jaccard similarity between GO
terms based on their gene annotations, as well as analysis utilities for
exploring similarity distributions.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os


def get_upper_triangle(similarity_matrix):
    """
    Extract upper triangle values from a similarity matrix.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square symmetric similarity matrix

    Returns
    -------
    np.ndarray
        Upper triangle values (excluding diagonal)
    """
    mask = np.triu_indices_from(similarity_matrix.values, k=1)
    return similarity_matrix.values[mask]


def find_similar_pairs(jaccard_matrix, threshold=0.5, top_n=10):
    """
    Find GO term pairs with high Jaccard similarity.

    Parameters
    ----------
    jaccard_matrix : pd.DataFrame
        Jaccard similarity matrix
    threshold : float
        Minimum Jaccard similarity to report
    top_n : int
        Number of top pairs to return

    Returns
    -------
    pd.DataFrame
        Top similar GO term pairs with their similarity scores
    """
    n = len(jaccard_matrix)
    rows, cols = np.triu_indices(n, k=1)
    similarities = jaccard_matrix.values[rows, cols]

    above_threshold = similarities >= threshold
    filtered_rows = rows[above_threshold]
    filtered_cols = cols[above_threshold]
    filtered_sims = similarities[above_threshold]

    pairs = pd.DataFrame({
        'GO_term_1': jaccard_matrix.index[filtered_rows],
        'GO_term_2': jaccard_matrix.columns[filtered_cols],
        'jaccard_similarity': filtered_sims
    })

    if len(pairs) > 0:
        pairs = pairs.sort_values(
            'jaccard_similarity', ascending=False
        ).head(top_n)

    return pairs


def count_pairs_below_threshold(jaccard_matrix, thresholds):
    """
    Count GO term pairs below various similarity thresholds.

    Parameters
    ----------
    jaccard_matrix : pd.DataFrame
        Jaccard similarity matrix
    thresholds : list
        List of similarity thresholds to check

    Returns
    -------
    results : dict
        Dictionary mapping threshold to count/percentage of pairs below
    total_pairs : int
        Total number of GO term pairs evaluated
    """
    values = get_upper_triangle(jaccard_matrix)
    total_pairs = len(values)

    results = {}
    for threshold in thresholds:
        count = np.sum(values < threshold)
        percentage = (count / total_pairs) * 100
        results[threshold] = {'count': count, 'percentage': percentage}

    return results, total_pairs


def calculate_jaccard_similarity_optimized(go_term_genes_dict):
    """
    Calculate Jaccard similarity matrix for GO terms.

    Uses scipy's pdist for efficient computation of pairwise similarities.

    Parameters
    ----------
    go_term_genes_dict : dict
        Dictionary mapping GO ID to set of gene IDs

    Returns
    -------
    pd.DataFrame
        Symmetric matrix with Jaccard similarities between all GO term pairs
    """
    go_terms = sorted(go_term_genes_dict.keys())
    all_genes = sorted(set().union(*go_term_genes_dict.values()))

    n_terms = len(go_terms)
    n_genes = len(all_genes)

    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

    feature_matrix = np.zeros((n_terms, n_genes), dtype=np.bool_)
    for i, go_id in enumerate(go_terms):
        genes = go_term_genes_dict[go_id]
        for gene in genes:
            feature_matrix[i, gene_to_idx[gene]] = True

    condensed_distances = pdist(feature_matrix, metric='jaccard')
    distance_matrix = squareform(condensed_distances)
    similarities = 1 - distance_matrix

    return pd.DataFrame(similarities, index=go_terms, columns=go_terms)


def load_or_calculate_jaccard(go_term_genes_dict, cache_file):
    """
    Load cached Jaccard matrix or calculate and cache if not exists.

    Parameters
    ----------
    go_term_genes_dict : dict
        Dictionary mapping GO ID to set of gene IDs
    cache_file : str or Path
        Path to cache file for storing/loading Jaccard matrix

    Returns
    -------
    tuple
        (pd.DataFrame, bool) - Jaccard similarity matrix and whether it was
        loaded from cache (True) or computed fresh (False)
    """
    if os.path.exists(cache_file):
        print(f"Loading cached Jaccard matrix from {cache_file}")
        return pd.read_csv(cache_file, index_col=0), True
    else:
        print(f"Computing Jaccard similarity matrix...")
        similarity_matrix = calculate_jaccard_similarity_optimized(
            go_term_genes_dict
        )
        print(f"Saving to {cache_file}")
        similarity_matrix.to_csv(cache_file)
        return similarity_matrix, False
