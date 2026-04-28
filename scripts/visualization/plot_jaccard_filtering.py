#!/usr/bin/env python3
"""Plot-only sibling of scripts/data_prep/jaccard_similarity_and_filtering.py figures.

Reads pre-computed unfiltered + filtered GO summary CSVs from --input-dir and,
optionally, the cached Jaccard matrices from --cache-dir. Re-renders the three
PDFs without re-running gene-set extraction, Jaccard computation, or greedy
pairwise filtering.

If the Jaccard cache is missing, the before/after heatmap PDF is skipped with
a warning -- the bar/line gene-count plots are always produced.

Outputs (under --output-dir):
    genes_per_go_2016_vs_2024_both_datasets.{pdf,jpeg}
    genes_per_go_2016_vs_2024_all_go_filtered.{pdf,jpeg}
    jaccard_filtering_effect_comparison.{pdf,jpeg}   (only if cache present)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def plot_gene_counts_comparison(dataset_sorted: pd.DataFrame, ax: plt.Axes,
                                title: str, n_terms: int) -> None:
    """Inline copy of src.visualization.plot_gene_counts_comparison.

    Inlined so this script can run without importing scipy (src.visualization
    pulls scipy at module-load time).
    """
    ax.plot(
        range(len(dataset_sorted)),
        dataset_sorted["no_of_genes_in_hetio_GO_2016"],
        marker="o", linewidth=0.5, markersize=1,
        label="2016", color="steelblue",
    )
    ax.plot(
        range(len(dataset_sorted)),
        dataset_sorted["no_of_genes_in_GO_2024"],
        marker="s", linewidth=0.5, markersize=1,
        label="2024", color="coral",
    )
    ax.set_xlabel(f"GO Term Rank (sorted by 2016 gene count, n={n_terms})", fontsize=11)
    ax.set_ylabel("Number of Genes", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_dual(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.jpeg", format="jpeg", dpi=300, bbox_inches="tight")


def _gene_counts_panel(unfiltered_df: pd.DataFrame, parents_df: pd.DataFrame | None,
                       output_dir: Path) -> None:
    all_sorted = unfiltered_df.sort_values(
        by="no_of_genes_in_hetio_GO_2016", ascending=True
    ).reset_index(drop=True)

    if parents_df is not None and not parents_df.empty:
        parents_sorted = parents_df.sort_values(
            by="no_of_genes_in_hetio_GO_2016", ascending=True
        ).reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        plot_gene_counts_comparison(
            all_sorted, axes[0],
            "All GO Positive Growth Terms", len(all_sorted),
        )
        plot_gene_counts_comparison(
            parents_sorted, axes[1],
            "Parent Terms (One Level Up)", len(parents_sorted),
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
        plot_gene_counts_comparison(
            all_sorted, ax,
            "All GO Positive Growth Terms", len(all_sorted),
        )

    fig.tight_layout()
    _save_dual(fig, output_dir, "genes_per_go_2016_vs_2024_both_datasets")
    plt.close(fig)


def _filtered_gene_counts_panel(filtered_df: pd.DataFrame, output_dir: Path) -> None:
    sorted_df = filtered_df.sort_values(
        by="no_of_genes_in_hetio_GO_2016", ascending=True
    ).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
    ax.scatter(
        sorted_df.index, sorted_df["no_of_genes_in_hetio_GO_2016"],
        alpha=0.6, label="2016", s=20,
    )
    ax.scatter(
        sorted_df.index, sorted_df["no_of_genes_in_GO_2024"],
        alpha=0.6, label="2024", s=20,
    )
    ax.set_xlabel("GO Term Index (sorted by 2016 gene count)")
    ax.set_ylabel("Number of Genes")
    ax.set_title(
        "All GO Positive Growth: All Growth Terms (Filtered)\n"
        f"n={len(sorted_df)} GO terms"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_dual(fig, output_dir, "genes_per_go_2016_vs_2024_all_go_filtered")
    plt.close(fig)


def _heatmap_panel(jaccard_2016_path: Path, filtered_go_ids: list[str],
                   output_dir: Path,
                   parents_jaccard_path: Path | None = None,
                   parents_filtered_go_ids: list[str] | None = None,
                   n_sample: int = 50) -> None:
    """Render the before/after Jaccard heatmap.

    Mirrors the layout in jaccard_similarity_and_filtering.py: 2x2 if parents
    are provided, otherwise 1x2 (before vs after for the all-positive-growth set).
    """
    if not jaccard_2016_path.exists():
        print(f"WARNING: {jaccard_2016_path} not found -- skipping heatmap.")
        return

    try:
        import seaborn as sns  # local import: heavy dep, only needed for heatmap
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        print(f"WARNING: heatmap requires scipy + seaborn ({exc}). Skipping heatmap.")
        return

    jaccard_all = pd.read_csv(jaccard_2016_path, index_col=0)
    parents_jaccard = None
    if parents_jaccard_path and parents_jaccard_path.exists() and parents_filtered_go_ids:
        parents_jaccard = pd.read_csv(parents_jaccard_path, index_col=0)

    def _top_similar(matrix: pd.DataFrame, n: int) -> list[str]:
        mean_sim = (matrix.sum(axis=1) - 1) / (len(matrix) - 1)
        return mean_sim.nlargest(n).index.tolist()

    top_before = _top_similar(jaccard_all, n_sample)
    jaccard_filtered = jaccard_all.loc[filtered_go_ids, filtered_go_ids]
    top_after = _top_similar(jaccard_filtered, n_sample)

    if parents_jaccard is not None and parents_filtered_go_ids:
        parents_top_before = _top_similar(parents_jaccard, n_sample)
        parents_filtered = parents_jaccard.loc[parents_filtered_go_ids, parents_filtered_go_ids]
        parents_top_after = _top_similar(parents_filtered, n_sample)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        axes = np.array([axes])  # one row

    vmin, vmax = 0, 0.5
    cmap = "YlOrRd"

    def _plot(matrix: pd.DataFrame, terms: list[str], ax, title: str) -> None:
        subset = matrix.loc[terms, terms]
        if len(terms) > 1:
            distance_matrix = 1 - subset.values
            np.fill_diagonal(distance_matrix, 0)
            condensed = squareform(distance_matrix, checks=False)
            Z = linkage(condensed, method="average")
            dend = dendrogram(Z, no_plot=True)
            order = dend["leaves"]
            subset = subset.iloc[order, order]
        sns.heatmap(
            subset, cmap=cmap, vmin=vmin, vmax=vmax,
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Jaccard Similarity"},
            ax=ax, square=True,
        )
        ax.set_title(title, fontsize=11)

    _plot(jaccard_all, top_before, axes[0, 0],
          f"All GO Positive Growth BEFORE Filtering\n(Top {n_sample} most similar of {len(jaccard_all)} terms)")
    _plot(jaccard_filtered, top_after, axes[0, 1],
          f"All GO Positive Growth AFTER Filtering\n(Top {n_sample} most similar of {len(filtered_go_ids)} terms)")

    if parents_jaccard is not None and parents_filtered_go_ids:
        _plot(parents_jaccard, parents_top_before, axes[1, 0],
              f"Parents GO Positive Growth BEFORE Filtering\n(Top {n_sample} most similar of {len(parents_jaccard)} terms)")
        _plot(parents_filtered, parents_top_after, axes[1, 1],
              f"Parents GO Positive Growth AFTER Filtering\n(Top {n_sample} most similar of {len(parents_filtered_go_ids)} terms)")

    fig.suptitle(
        "Effect of Jaccard Filtering: Top 50 Most Similar GO Terms\n(Same color scale: 0-0.5)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _save_dual(fig, output_dir, "jaccard_filtering_effect_comparison")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default="output/intermediate",
        help="Directory holding all_GO_positive_growth.csv, all_GO_positive_growth_filtered.csv, parents_GO_postive_growth_filtered.csv (default: output/intermediate).",
    )
    p.add_argument(
        "--cache-dir",
        default="output/jaccard_similarity",
        help="Directory holding cached jaccard matrices (default: output/jaccard_similarity).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the jaccard-filtering PDFs/JPEGs.",
    )
    p.add_argument(
        "--include-parents",
        action="store_true",
        help="Also render parent-term panels (requires parents_GO_postive_growth* + parents jaccard cache).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unfiltered_path = input_dir / "all_GO_positive_growth.csv"
    filtered_path = input_dir / "all_GO_positive_growth_filtered.csv"
    for p in (unfiltered_path, filtered_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input {p} is missing. Run scripts/data_prep/jaccard_similarity_and_filtering.py first."
            )

    unfiltered_df = pd.read_csv(unfiltered_path)
    filtered_df = pd.read_csv(filtered_path)

    parents_df = None
    parents_filtered_ids = None
    if args.include_parents:
        parents_filtered_path = input_dir / "parents_GO_postive_growth_filtered.csv"
        parents_unfiltered_path = input_dir / "parents_GO_postive_growth.csv"
        if parents_unfiltered_path.exists():
            parents_df = pd.read_csv(parents_unfiltered_path)
        if parents_filtered_path.exists():
            parents_filtered_ids = pd.read_csv(parents_filtered_path)["go_id"].tolist()

    _gene_counts_panel(unfiltered_df, parents_df, output_dir)
    _filtered_gene_counts_panel(filtered_df, output_dir)

    _heatmap_panel(
        jaccard_2016_path=cache_dir / "jaccard_similarity_all_GO_positive_growth_2016.csv",
        filtered_go_ids=filtered_df["go_id"].tolist(),
        output_dir=output_dir,
        parents_jaccard_path=(
            cache_dir / "jaccard_similarity_parents_GO_postive_growth_2016.csv"
            if args.include_parents else None
        ),
        parents_filtered_go_ids=parents_filtered_ids,
    )

    print(f"Saved figures under {output_dir}")


if __name__ == "__main__":
    main()
