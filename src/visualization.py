"""
Visualization utilities for GO term analysis.

This module provides functions for visualizing Jaccard similarity matrices
and related GO term analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def create_diagonal_mask(size):
    """
    Create a boolean mask for the diagonal of a square matrix.

    Parameters
    ----------
    size : int
        Size of the square matrix

    Returns
    -------
    np.ndarray
        Boolean mask with True on diagonal, False elsewhere
    """
    mask = np.zeros((size, size), dtype=bool)
    np.fill_diagonal(mask, True)
    return mask


def plot_clustered_heatmap(similarity_matrix, ax, title, vmax=0.8, cmap="YlOrRd"):
    """
    Plot heatmap with hierarchical clustering and no diagonal.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Square symmetric similarity matrix
    ax : matplotlib.axes.Axes
        Axes object to plot on
    title : str
        Title for the heatmap
    vmax : float
        Maximum value for color scale (default 0.8)
    cmap : str
        Colormap name (default "YlOrRd")
    """
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix.values

    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Get dendrogram order
    dend = dendrogram(linkage_matrix, no_plot=True)
    order = dend['leaves']

    # Reorder matrix by clustering
    clustered_matrix = similarity_matrix.iloc[order, order]

    # Create mask for diagonal
    mask = create_diagonal_mask(len(clustered_matrix))

    # Plot heatmap with adjusted scale for sparse data
    sns.heatmap(
        clustered_matrix,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        mask=mask,
        square=True,
        cbar_kws={'label': 'Jaccard Similarity'},
        ax=ax,
        xticklabels=False,
        yticklabels=False
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("GO Terms (clustered)", fontsize=11)
    ax.set_ylabel("GO Terms (clustered)", fontsize=11)


def plot_similarity_distribution(similarity_values, ax, title, color=None):
    """
    Plot histogram of similarity values with mean and median lines.

    Parameters
    ----------
    similarity_values : np.ndarray
        Array of similarity values
    ax : matplotlib.axes.Axes
        Axes object to plot on
    title : str
        Title for the histogram
    color : str, optional
        Color for the histogram bars
    """
    kwargs = {'bins': 50, 'edgecolor': 'black', 'alpha': 0.7}
    if color:
        kwargs['color'] = color

    ax.hist(similarity_values, **kwargs)
    ax.axvline(
        x=similarity_values.mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean = {similarity_values.mean():.3f}'
    )
    ax.axvline(
        x=np.median(similarity_values),
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Median = {np.median(similarity_values):.3f}'
    )
    ax.set_xlabel('Jaccard Similarity', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_gene_counts_comparison(dataset_sorted, ax, title, n_terms):
    """
    Plot gene counts comparison between 2016 and 2024.

    Parameters
    ----------
    dataset_sorted : pd.DataFrame
        Dataset sorted by 2016 gene count, with columns
        'no_of_genes_in_hetio_GO_2016' and 'no_of_genes_in_GO_2024'
    ax : matplotlib.axes.Axes
        Axes object to plot on
    title : str
        Title for the plot
    n_terms : int
        Number of GO terms (for x-axis label)
    """
    ax.plot(
        range(len(dataset_sorted)),
        dataset_sorted["no_of_genes_in_hetio_GO_2016"],
        marker="o",
        linewidth=0.5,
        markersize=1,
        label="2016",
        color="steelblue"
    )
    ax.plot(
        range(len(dataset_sorted)),
        dataset_sorted["no_of_genes_in_GO_2024"],
        marker="s",
        linewidth=0.5,
        markersize=1,
        label="2024",
        color="darkorange"
    )
    ax.set_xlabel(f"GO Terms (n = {n_terms})", fontsize=11)
    ax.set_ylabel("Number of Genes", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Year", fontsize=10, title_fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
