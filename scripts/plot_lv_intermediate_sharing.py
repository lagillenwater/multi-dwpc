#!/usr/bin/env python3
"""
Visualize LV intermediate sharing results.

Usage:
    python scripts/plot_lv_intermediate_sharing.py --input-dir output/lv_intermediate_sharing
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_figure(fig: plt.Figure, fig_dir: Path, name: str, dpi: int = 150) -> None:
    """Save figure as both PNG and PDF."""
    fig.savefig(fig_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary_comparison(
    summary_df: pd.DataFrame, fig_dir: Path, colors: list[str]
) -> None:
    """Figure 1: Summary comparison across LV-target pairs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    labels = [
        f"{row['lv_id']}\n{row['target_name']}" for _, row in summary_df.iterrows()
    ]
    n_bars = len(summary_df)
    bar_colors = colors[:n_bars]

    # Median % sharing
    ax = axes[0]
    ax.bar(range(n_bars), summary_df["median_pct_sharing"], color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median % Genes Sharing Intermediates")
    ax.set_title("Intermediate Sharing Rate")
    ax.set_ylim(0, 105)
    for i, v in enumerate(summary_df["median_pct_sharing"]):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)

    # Number of metapaths
    ax = axes[1]
    ax.bar(range(n_bars), summary_df["n_metapaths"], color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Selected Metapaths")
    ax.set_title("Metapath Coverage")
    for i, v in enumerate(summary_df["n_metapaths"]):
        ax.text(i, v + 1, str(v), ha="center", fontsize=10)

    # Median Jaccard
    ax = axes[2]
    ax.bar(range(n_bars), summary_df["median_jaccard"], color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median Jaccard Similarity")
    ax.set_title("Within-Group Similarity")
    for i, v in enumerate(summary_df["median_jaccard"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    save_figure(fig, fig_dir, "summary_comparison")


def plot_sharing_by_rank(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    colors: list[str],
) -> None:
    """Figure 2: Intermediate sharing by metapath rank for each LV-target."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )
    n_panels = len(lv_targets)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        ax = axes[idx]
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()
        subset = subset.sort_values("metapath_rank")

        has_paths = subset["n_genes_with_paths"] > 0

        ax.scatter(
            subset.loc[has_paths, "metapath_rank"],
            subset.loc[has_paths, "pct_genes_sharing"],
            c=colors[idx % len(colors)],
            alpha=0.7,
            s=50,
        )

        ax.plot(
            subset.loc[has_paths, "metapath_rank"],
            subset.loc[has_paths, "pct_genes_sharing"],
            c=colors[idx % len(colors)],
            alpha=0.3,
            linewidth=1,
        )

        ax.set_xlabel("Metapath Rank (by effect size)")
        ax.set_ylabel("% Genes Sharing Intermediates")
        ax.set_title(f"{lv_id}: {target_name}")
        ax.set_ylim(-5, 105)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure(fig, fig_dir, "sharing_by_rank")


def plot_effect_size_vs_sharing(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Figure 3: Effect size vs intermediate sharing."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )
    n_panels = len(lv_targets)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        ax = axes[idx]
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()

        has_paths = subset["n_genes_with_paths"] > 0
        subset_valid = subset[has_paths]

        if len(subset_valid) > 0:
            sc = ax.scatter(
                subset_valid["effect_size_d"],
                subset_valid["pct_genes_sharing"],
                c=subset_valid["metapath_rank"],
                cmap="viridis_r",
                s=60,
                alpha=0.7,
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Metapath Rank")

        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_ylabel("% Genes Sharing Intermediates")
        ax.set_title(f"{lv_id}: {target_name}")
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_figure(fig, fig_dir, "effect_size_vs_sharing")


def plot_jaccard_vs_intermediates(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Figure 4: Jaccard similarity vs number of intermediates."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )
    n_panels = len(lv_targets)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        ax = axes[idx]
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()

        valid = (subset["n_genes_with_paths"] > 0) & (
            subset["n_unique_intermediates"] > 0
        )
        subset_valid = subset[valid]

        if len(subset_valid) > 0:
            sc = ax.scatter(
                subset_valid["n_unique_intermediates"],
                subset_valid["median_jaccard_to_group"],
                c=subset_valid["pct_genes_sharing"],
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
                s=60,
                alpha=0.7,
            )
            ax.set_xscale("log")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("% Sharing")

        ax.set_xlabel("Number of Unique Intermediates (log scale)")
        ax.set_ylabel("Median Jaccard Similarity")
        ax.set_title(f"{lv_id}: {target_name}")

    plt.tight_layout()
    save_figure(fig, fig_dir, "jaccard_vs_intermediates")


def plot_top_metapaths(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    top_n: int = 15,
) -> None:
    """Figure 5: Top metapaths for each LV-target (horizontal bar chart)."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )
    n_panels = len(lv_targets)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        ax = axes[idx]
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()

        top_metapaths = subset.sort_values("metapath_rank").head(top_n)
        y_pos = np.arange(len(top_metapaths))

        bar_colors = plt.cm.RdYlGn(top_metapaths["pct_genes_sharing"].fillna(0) / 100)

        ax.barh(y_pos, top_metapaths["effect_size_d"], color=bar_colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_metapaths["metapath"], fontsize=8)
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title(f"{lv_id}: {target_name}\nTop {top_n} Metapaths")
        ax.invert_yaxis()

        for i, (_, row) in enumerate(top_metapaths.iterrows()):
            pct = row["pct_genes_sharing"]
            if pd.notna(pct):
                ax.annotate(
                    f"{pct:.0f}%",
                    xy=(row["effect_size_d"], i),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=7,
                    va="center",
                )

    plt.tight_layout()
    save_figure(fig, fig_dir, "top_metapaths")


def plot_sharing_distribution(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    colors: list[str],
) -> None:
    """Figure 6: Distribution of sharing across metapaths."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()

        valid = subset["n_genes_with_paths"] > 0
        values = subset.loc[valid, "pct_genes_sharing"].dropna()

        if len(values) > 0:
            ax.hist(
                values,
                bins=np.arange(0, 105, 10),
                alpha=0.5,
                label=f"{lv_id}: {target_name}",
                color=colors[idx % len(colors)],
                edgecolor="black",
            )

    ax.set_xlabel("% Genes Sharing Intermediates")
    ax.set_ylabel("Number of Metapaths")
    ax.set_title("Distribution of Intermediate Sharing Across Metapaths")
    ax.legend()
    ax.set_xlim(0, 100)

    plt.tight_layout()
    save_figure(fig, fig_dir, "sharing_distribution")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LV intermediate sharing results."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing intermediate sharing CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for figures. Defaults to input-dir/figures.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    fig_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metapath_path = input_dir / "intermediate_sharing_by_metapath.csv"
    summary_path = input_dir / "intermediate_sharing_summary.csv"

    if not metapath_path.exists() or not summary_path.exists():
        print(f"Required CSV files not found in {input_dir}")
        print("Expected: intermediate_sharing_by_metapath.csv, intermediate_sharing_summary.csv")
        return

    metapath_df = pd.read_csv(metapath_path)
    summary_df = pd.read_csv(summary_path)

    print(f"Loaded {len(metapath_df)} metapath rows, {len(summary_df)} LV-target pairs")

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Color palette
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

    # Generate all figures
    print("Generating figures...")

    plot_summary_comparison(summary_df, fig_dir, colors)
    print("  - summary_comparison")

    plot_sharing_by_rank(metapath_df, summary_df, fig_dir, colors)
    print("  - sharing_by_rank")

    plot_effect_size_vs_sharing(metapath_df, summary_df, fig_dir)
    print("  - effect_size_vs_sharing")

    plot_jaccard_vs_intermediates(metapath_df, summary_df, fig_dir)
    print("  - jaccard_vs_intermediates")

    plot_top_metapaths(metapath_df, summary_df, fig_dir)
    print("  - top_metapaths")

    plot_sharing_distribution(metapath_df, summary_df, fig_dir, colors)
    print("  - sharing_distribution")

    print(f"\nFigures saved to: {fig_dir}")
    print("Generated files:")
    for f in sorted(fig_dir.iterdir()):
        if f.suffix in (".png", ".pdf"):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
