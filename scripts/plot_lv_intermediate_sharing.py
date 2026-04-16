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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    labels = [
        f"{row['lv_id']}\n{row['target_name']}" for _, row in summary_df.iterrows()
    ]
    n_bars = len(summary_df)
    bar_colors = colors[:n_bars]

    # Top-1 intermediate coverage (most important: does the best intermediate cover most genes?)
    ax = axes[0]
    values = summary_df.get("median_top1_coverage", pd.Series([0] * n_bars))
    ax.bar(range(n_bars), values.fillna(0), color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median % Genes Covered")
    ax.set_title("Top-1 Intermediate Coverage")
    ax.set_ylim(0, 105)
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)

    # % of intermediates used by >25% of genes
    ax = axes[1]
    values = summary_df.get("median_pct_intermediates_shared_quarter", pd.Series([0] * n_bars))
    ax.bar(range(n_bars), values.fillna(0), color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median % Intermediates")
    ax.set_title("% Intermediates Used by >25% of Genes")
    ax.set_ylim(0, 105)
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

    # % of intermediates used by majority (>50%) of genes
    ax = axes[2]
    values = summary_df.get("median_pct_intermediates_shared_majority", pd.Series([0] * n_bars))
    ax.bar(range(n_bars), values.fillna(0), color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median % Intermediates")
    ax.set_title("% Intermediates Used by >50% of Genes")
    ax.set_ylim(0, 105)
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

    # % of intermediates used by ALL genes
    ax = axes[3]
    values = summary_df.get("median_pct_intermediates_shared_all", pd.Series([0] * n_bars))
    ax.bar(range(n_bars), values.fillna(0), color=bar_colors)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median % Intermediates")
    ax.set_title("% Intermediates Used by ALL Genes")
    ax.set_ylim(0, 105)
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

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

        ax.set_xlabel("Metapath Rank (by z)")
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

        ax.set_xlabel("z")
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
        ax.set_xlabel("z")
        ax.set_title(f"{lv_id}: {target_name}\nTop {top_n} Metapaths")
        ax.invert_yaxis()

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


def plot_coverage_metrics(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    colors: list[str],
) -> None:
    """Figure 7: Intermediate coverage metrics by metapath."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )
    n_panels = len(lv_targets)

    # Check if coverage columns exist
    if "top1_intermediate_coverage" not in metapath_df.columns:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        ax = axes[idx]
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()

        valid = subset["n_genes_with_paths"] > 0
        subset_valid = subset[valid].sort_values("metapath_rank")

        if len(subset_valid) > 0:
            x = subset_valid["metapath_rank"]
            # Only add labels on first panel to avoid duplicate legend warning
            label_top1 = "Top-1" if idx == 0 else None
            label_top5 = "Top-5" if idx == 0 else None
            ax.scatter(
                x,
                subset_valid["top1_intermediate_coverage"],
                c="#e74c3c",  # Red for top-1
                alpha=0.8,
                s=50,
                label=label_top1,
            )
            if "top5_intermediate_coverage" in subset_valid.columns:
                ax.scatter(
                    x,
                    subset_valid["top5_intermediate_coverage"],
                    c="#3498db",  # Blue for top-5
                    alpha=0.8,
                    s=40,
                    marker="s",
                    label=label_top5,
                )

        ax.set_xlabel("Metapath Rank")
        ax.set_ylabel("% Genes Covered")
        ax.set_title(f"{lv_id}: {target_name}")
        ax.set_ylim(-5, 105)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    save_figure(fig, fig_dir, "coverage_by_rank")


def plot_intermediates_shared_distribution(
    metapath_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    colors: list[str],
) -> None:
    """Figure 8: Distribution of top-1 intermediate coverage across metapaths."""
    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )

    if "top1_intermediate_coverage" not in metapath_df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        subset = metapath_df[
            (metapath_df["lv_id"] == lv_id)
            & (metapath_df["target_name"] == target_name)
        ].copy()

        valid = subset["n_genes_with_paths"] > 0
        values = subset.loc[valid, "top1_intermediate_coverage"].dropna()

        if len(values) > 0:
            ax.hist(
                values,
                bins=np.arange(0, 105, 5),
                alpha=0.5,
                label=f"{lv_id}: {target_name}",
                color=colors[idx % len(colors)],
                edgecolor="black",
            )

    ax.set_xlabel("Top-1 Intermediate Coverage (% of genes)")
    ax.set_ylabel("Number of Metapaths")
    ax.set_title("Distribution of Top-1 Intermediate Coverage Across Metapaths")
    ax.legend()
    ax.set_xlim(0, 100)

    plt.tight_layout()
    save_figure(fig, fig_dir, "top1_coverage_distribution")


def _get_display_label(row: pd.Series, max_len: int = 30) -> str:
    """Get display label for an intermediate, preferring name over ID."""
    name = row.get("intermediate_name")
    int_id = row.get("intermediate_id", "")

    if pd.notna(name) and name:
        label = str(name)
    else:
        # Fall back to ID, but strip the type prefix for readability
        if ":" in int_id:
            label = int_id.split(":", 1)[1]
        else:
            label = int_id

    # Truncate if too long
    if len(label) > max_len:
        label = label[:max_len - 3] + "..."

    return label


def plot_top_intermediates(
    top_int_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    colors: list[str],
    top_n_metapaths: int = 5,
) -> None:
    """Figure 9: Top intermediates for top metapaths."""
    if top_int_df is None or top_int_df.empty:
        return

    lv_targets = list(
        summary_df[["lv_id", "target_name"]].itertuples(index=False, name=None)
    )

    for idx, (lv_id, target_name) in enumerate(lv_targets):
        # Filter to this LV and top metapaths
        subset = top_int_df[
            (top_int_df["lv_id"] == lv_id)
            & (top_int_df["metapath_rank"] <= top_n_metapaths)
        ].copy()

        if subset.empty:
            continue

        # Get unique metapaths
        metapaths = subset.sort_values("metapath_rank")["metapath"].unique()
        n_metapaths = len(metapaths)

        fig, axes = plt.subplots(1, n_metapaths, figsize=(4 * n_metapaths, 6))
        if n_metapaths == 1:
            axes = [axes]

        for mp_idx, metapath in enumerate(metapaths):
            ax = axes[mp_idx]
            mp_data = subset[subset["metapath"] == metapath].sort_values(
                "intermediate_rank"
            ).head(10)

            y_pos = np.arange(len(mp_data))
            ax.barh(
                y_pos,
                mp_data["pct_genes_using"],
                color=colors[idx % len(colors)],
                alpha=0.8,
            )
            ax.set_yticks(y_pos)
            # Use name if available, otherwise truncate ID
            labels = [_get_display_label(row) for _, row in mp_data.iterrows()]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("% Genes Using")
            ax.set_title(f"{metapath}", fontsize=10)
            ax.set_xlim(0, 105)
            ax.invert_yaxis()

        fig.suptitle(f"{lv_id}: {target_name} - Top Intermediates", fontsize=12)
        plt.tight_layout()
        save_figure(fig, fig_dir, f"top_intermediates_{lv_id}")


# Node type colors for consistent styling
NODE_TYPE_COLORS = {
    "G": "#6baed6",   # Gene - blue
    "A": "#fd8d3c",   # Anatomy - orange
    "BP": "#74c476",  # Biological Process - green
    "CC": "#9e9ac8",  # Cellular Component - purple
    "C": "#31a354",   # Compound - darker green
    "D": "#fdae6b",   # Disease - light orange
    "MF": "#bcbddc",  # Molecular Function - light purple
    "PC": "#a1d99b",  # Pharmacologic Class - light green
    "PW": "#e6550d",  # Pathway - red-orange
    "SE": "#756bb1",  # Side Effect - purple
    "S": "#d9d9d9",   # Symptom - gray
}

NODE_TYPE_NAMES = {
    "G": "Gene",
    "A": "Anatomy",
    "BP": "Biological Process",
    "CC": "Cellular Component",
    "C": "Compound",
    "D": "Disease",
    "MF": "Molecular Function",
    "PC": "Pharmacologic Class",
    "PW": "Pathway",
    "SE": "Side Effect",
    "S": "Symptom",
}


def plot_top_shared_intermediates_aggregated(
    top_int_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    top_n: int = 15,
) -> None:
    """Plot top shared intermediates by % of metapaths where they appear as top-ranked.

    Shows intermediates ranked by how consistently they appear across metapaths,
    colored by node type. X-axis shows % of metapaths where this intermediate
    is among the top-ranked (rank <= 5).
    """
    if top_int_df is None or top_int_df.empty:
        return

    lv_targets = list(
        summary_df[["lv_id", "target_id", "target_name"]].itertuples(index=False, name=None)
    )

    for lv_id, target_id, target_name in lv_targets:
        # Filter to this LV
        subset = top_int_df[top_int_df["lv_id"] == lv_id].copy()

        if subset.empty:
            continue

        # Count total metapaths for this LV
        n_metapaths = subset["metapath"].nunique()

        # Filter to top-ranked intermediates per metapath (rank <= 5)
        top_ranked = subset[subset["intermediate_rank"] <= 5].copy()

        if top_ranked.empty:
            continue

        # Count how many metapaths each intermediate appears in (as top-ranked)
        agg = top_ranked.groupby(["intermediate_id", "intermediate_name"]).agg(
            n_metapaths_top_ranked=("metapath", "nunique"),
            median_pct_genes=("pct_genes_using", "median"),
        ).reset_index()

        # Compute % of metapaths where this intermediate is top-ranked
        agg["pct_metapaths_top_ranked"] = agg["n_metapaths_top_ranked"] / n_metapaths * 100

        # Sort by % metapaths descending and take top N
        agg = agg.sort_values("pct_metapaths_top_ranked", ascending=False).head(top_n)

        if agg.empty:
            continue

        # Extract node type from intermediate_id (format: "TYPE:ID")
        agg["node_type"] = agg["intermediate_id"].apply(
            lambda x: x.split(":")[0] if ":" in x else "?"
        )

        # Get display labels
        agg["label"] = agg.apply(
            lambda row: row["intermediate_name"] if pd.notna(row["intermediate_name"])
            else row["intermediate_id"].split(":", 1)[1] if ":" in row["intermediate_id"]
            else row["intermediate_id"],
            axis=1
        )

        # Truncate long labels
        agg["label"] = agg["label"].apply(
            lambda x: x[:40] + "..." if len(str(x)) > 40 else x
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(agg) * 0.4)))

        # Get colors based on node type
        bar_colors = [NODE_TYPE_COLORS.get(nt, "#999999") for nt in agg["node_type"]]

        y_pos = np.arange(len(agg))
        bars = ax.barh(y_pos, agg["pct_metapaths_top_ranked"], color=bar_colors, alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(agg["label"], fontsize=9)
        ax.set_xlabel("% of metapaths where top-5 ranked")
        ax.set_xlim(0, 105)
        ax.set_title(f"Most consistent intermediate nodes (n={n_metapaths} metapaths)\n{target_name}")
        ax.invert_yaxis()

        # Add legend for node types present in the data
        present_types = agg["node_type"].unique()
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=NODE_TYPE_COLORS.get(nt, "#999999"), alpha=0.85)
            for nt in sorted(present_types)
        ]
        legend_labels = [nt for nt in sorted(present_types)]
        ax.legend(
            legend_handles, legend_labels,
            title="Node type",
            loc="lower right",
            fontsize=8,
        )

        plt.tight_layout()
        save_figure(fig, fig_dir, f"top_shared_intermediates_{lv_id}")


def plot_metapath_intermediate_heatmap(
    top_int_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fig_dir: Path,
    top_n_intermediates: int = 20,
) -> None:
    """Plot heatmap of metapath x intermediate connectivity.

    Shows which intermediates are used by which metapaths, with color intensity
    representing the number of genes using that intermediate for that metapath.
    """
    if top_int_df is None or top_int_df.empty:
        return

    lv_targets = list(
        summary_df[["lv_id", "target_id", "target_name"]].itertuples(index=False, name=None)
    )

    for lv_id, target_id, target_name in lv_targets:
        # Filter to this LV
        subset = top_int_df[top_int_df["lv_id"] == lv_id].copy()

        if subset.empty:
            continue

        # Get top intermediates by total genes using across all metapaths
        int_totals = subset.groupby("intermediate_id")["n_genes_using"].sum()
        top_intermediates = int_totals.nlargest(top_n_intermediates).index.tolist()

        # Filter to top intermediates
        subset = subset[subset["intermediate_id"].isin(top_intermediates)]

        if subset.empty:
            continue

        # Create pivot table: metapath x intermediate
        # Get display labels for intermediates
        int_labels = {}
        for _, row in subset.drop_duplicates("intermediate_id").iterrows():
            int_id = row["intermediate_id"]
            if pd.notna(row.get("intermediate_name")) and row["intermediate_name"]:
                label = str(row["intermediate_name"])
            else:
                label = int_id.split(":", 1)[1] if ":" in int_id else int_id
            # Truncate long labels
            if len(label) > 25:
                label = label[:22] + "..."
            int_labels[int_id] = label

        subset["int_label"] = subset["intermediate_id"].map(int_labels)

        pivot = subset.pivot_table(
            index="metapath",
            columns="int_label",
            values="n_genes_using",
            aggfunc="max",
            fill_value=0,
        )

        if pivot.empty:
            continue

        # Sort metapaths by metapath_rank
        metapath_order = (
            subset.drop_duplicates("metapath")
            .sort_values("metapath_rank")["metapath"]
            .tolist()
        )
        pivot = pivot.reindex([mp for mp in metapath_order if mp in pivot.index])

        # Sort intermediates by total genes using
        int_order = (
            subset.groupby("int_label")["n_genes_using"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        pivot = pivot[[col for col in int_order if col in pivot.columns]]

        # Create figure
        fig_width = max(10, len(pivot.columns) * 0.5)
        fig_height = max(4, len(pivot) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(pivot.index, fontsize=8)

        ax.set_xlabel("")
        ax.set_ylabel("Metapath")
        ax.set_title(f"Metapath x Intermediate connectivity\n{target_name}")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Number of genes")

        plt.tight_layout()
        save_figure(fig, fig_dir, f"metapath_intermediate_heatmap_{lv_id}")


def main() -> None:
    import json as _json
    parser = argparse.ArgumentParser(
        description="Visualize LV or year intermediate sharing results."
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
    parser.add_argument(
        "--analysis-type",
        choices=["lv", "year"],
        default="lv",
        help="Dataset type. LV uses lv_id; year uses go_id (renamed internally).",
    )
    parser.add_argument(
        "--top-ids-json",
        help=(
            "Optional JSON file holding a list of ids to include (e.g., "
            "top_go_ids.json). If set, plots are restricted to those entities -- "
            "essential for year runs where there may be hundreds of GO terms."
        ),
    )
    parser.add_argument(
        "--nodes-dir",
        default="data/nodes",
        help="Dir with node TSVs for resolving target names in year mode.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    fig_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    metapath_path = input_dir / "intermediate_sharing_by_metapath.csv"
    summary_path = input_dir / "intermediate_sharing_summary.csv"
    top_int_path = input_dir / "top_intermediates_by_metapath.csv"

    if not metapath_path.exists() or not summary_path.exists():
        print(f"Required CSV files not found in {input_dir}")
        print("Expected: intermediate_sharing_by_metapath.csv, intermediate_sharing_summary.csv")
        return

    metapath_df = pd.read_csv(metapath_path)
    summary_df = pd.read_csv(summary_path)
    top_int_df = pd.read_csv(top_int_path) if top_int_path.exists() else None

    # Year data should use scripts/plot_year_intermediate_sharing.py, which
    # preserves the 2016-vs-2024 cohort split. This LV-only script's column
    # assumptions (single n_genes_with_paths, single pct_genes_sharing) don't
    # map to year semantics without pooling cohorts, which destroys the
    # comparison. Bail out with a clear message rather than render misleading
    # numbers.
    if args.analysis_type == "year":
        raise SystemExit(
            "This script is LV-only. For year data use "
            "scripts/plot_year_intermediate_sharing.py (preserves the "
            "2016-vs-2024 cohort split)."
        )

    # Optional filter to a provided list of ids (top GOs for year, etc.).
    if args.top_ids_json:
        with open(args.top_ids_json) as fh:
            top_ids = [str(x) for x in _json.load(fh)]
        before = len(summary_df)
        summary_df = summary_df[summary_df["lv_id"].astype(str).isin(top_ids)].copy()
        metapath_df = metapath_df[metapath_df["lv_id"].astype(str).isin(top_ids)].copy()
        if top_int_df is not None and not top_int_df.empty:
            top_int_df = top_int_df[top_int_df["lv_id"].astype(str).isin(top_ids)].copy()
        print(f"Filtered to {len(summary_df)} of {before} ids from {args.top_ids_json}")

    if summary_df.empty:
        print("Nothing to plot after filtering -- exiting.")
        return

    print(f"Loaded {len(metapath_df)} metapath rows, {len(summary_df)} gene-set pairs")
    if top_int_df is not None:
        print(f"Loaded {len(top_int_df)} top intermediate rows")

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

    plot_coverage_metrics(metapath_df, summary_df, fig_dir, colors)
    print("  - coverage_by_rank")

    plot_intermediates_shared_distribution(metapath_df, summary_df, fig_dir, colors)
    print("  - intermediates_shared_distribution")

    plot_top_intermediates(top_int_df, summary_df, fig_dir, colors)
    print("  - top_intermediates_*")

    plot_top_shared_intermediates_aggregated(top_int_df, summary_df, fig_dir)
    print("  - top_shared_intermediates_*")

    plot_metapath_intermediate_heatmap(top_int_df, summary_df, fig_dir)
    print("  - metapath_intermediate_heatmap_*")

    print(f"\nFigures saved to: {fig_dir}")
    print("Generated files:")
    for f in sorted(fig_dir.iterdir()):
        if f.suffix in (".png", ".pdf"):
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
