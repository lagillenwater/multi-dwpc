#!/usr/bin/env python3
"""
Plot overall LV-by-metapath performance from lv_metapath_results.csv.

Outputs summary CSVs and overview plots similar in spirit to existing metapath
signature/divergence analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {
    "lv_id",
    "metapath",
    "target_set_id",
    "min_d",
    "min_diff",
    "d_perm",
    "d_rand",
    "p_perm_fdr",
    "p_rand_fdr",
    "supported",
}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "min_d",
        "min_diff",
        "d_perm",
        "d_rand",
        "p_perm_fdr",
        "p_rand_fdr",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["supported"] = (
        out["supported"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    return out


def _build_metapath_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["lv_id", "metapath"], as_index=False)
        .agg(
            n_target_sets=("target_set_id", "nunique"),
            min_d=("min_d", "max"),
            min_diff=("min_diff", "max"),
            d_perm=("d_perm", "max"),
            d_rand=("d_rand", "max"),
            p_perm_fdr=("p_perm_fdr", "min"),
            p_rand_fdr=("p_rand_fdr", "min"),
            supported=("supported", "max"),
        )
        .reset_index(drop=True)
    )
    summary["combined_fdr"] = np.maximum(summary["p_perm_fdr"], summary["p_rand_fdr"])
    summary["neglog10_combined_fdr"] = -np.log10(
        np.clip(summary["combined_fdr"], 1e-300, 1.0)
    )
    return summary


def _build_lv_summary(results: pd.DataFrame) -> pd.DataFrame:
    lv_summary = (
        results.groupby("lv_id", as_index=False)
        .agg(
            n_features=("metapath", "size"),
            n_supported=("supported", "sum"),
            median_min_d=("min_d", "median"),
            max_min_d=("min_d", "max"),
            median_min_diff=("min_diff", "median"),
            max_min_diff=("min_diff", "max"),
        )
        .reset_index(drop=True)
    )
    lv_summary["support_fraction"] = lv_summary["n_supported"] / lv_summary["n_features"]
    return lv_summary


def _plot_support_rate(lv_summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = lv_summary.sort_values("support_fraction", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ordered, x="lv_id", y="support_fraction", ax=ax, color="#3b82f6")
    ax.set_ylim(0, 1)
    ax.set_xlabel("LV")
    ax.set_ylabel("Supported fraction")
    ax.set_title("Supported Metapath Fraction by LV")
    for idx, row in ordered.reset_index(drop=True).iterrows():
        ax.text(
            idx,
            min(row["support_fraction"] + 0.02, 0.98),
            f"{int(row['n_supported'])}/{int(row['n_features'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _save_figure(fig, output_dir / "support_fraction_by_lv.png")


def _plot_effect_heatmap(
    metapath_summary: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    pivot = metapath_summary.pivot(index="metapath", columns="lv_id", values="min_d")
    support = metapath_summary.pivot(index="metapath", columns="lv_id", values="supported")
    if pivot.empty:
        return

    top_index = pivot.max(axis=1).sort_values(ascending=False).head(top_n).index
    heat = pivot.loc[top_index]
    support = support.reindex(index=heat.index, columns=heat.columns).fillna(False)

    fig_h = max(5, 0.28 * len(heat))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="coolwarm",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "min_d (stronger positive signal is larger)"},
    )
    for i in range(support.shape[0]):
        for j in range(support.shape[1]):
            if bool(support.iloc[i, j]):
                ax.scatter(j + 0.5, i + 0.5, s=10, c="black")
    ax.set_title("LV-by-Metapath Effect Heatmap (top metapaths)")
    ax.set_xlabel("LV")
    ax.set_ylabel("Metapath")
    _save_figure(fig, output_dir / "effect_size_heatmap.png")


def _plot_significance_heatmap(
    metapath_summary: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    pivot = metapath_summary.pivot(
        index="metapath", columns="lv_id", values="neglog10_combined_fdr"
    )
    if pivot.empty:
        return

    top_index = pivot.max(axis=1).sort_values(ascending=False).head(top_n).index
    heat = pivot.loc[top_index]

    fig_h = max(5, 0.28 * len(heat))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="mako",
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "-log10(max(FDR_perm, FDR_rand))"},
    )
    ax.set_title("LV-by-Metapath Significance Heatmap (top metapaths)")
    ax.set_xlabel("LV")
    ax.set_ylabel("Metapath")
    _save_figure(fig, output_dir / "significance_heatmap.png")


def _plot_top_metapaths(
    metapath_summary: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    lvs = sorted(metapath_summary["lv_id"].unique())
    if not lvs:
        return

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(lvs),
        figsize=(6 * len(lvs), max(5, 0.32 * top_n + 2)),
        squeeze=False,
    )
    axes = axes.ravel()

    for idx, lv_id in enumerate(lvs):
        ax = axes[idx]
        subset = metapath_summary[metapath_summary["lv_id"] == lv_id].copy()
        if subset.empty:
            ax.axis("off")
            continue
        top = subset.sort_values("min_d", ascending=False).head(top_n)
        top = top.sort_values("min_d", ascending=True)
        colors = np.where(top["supported"], "#059669", "#9ca3af")
        ax.barh(top["metapath"], top["min_d"], color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{lv_id}: Top metapaths by min_d")
        ax.set_xlabel("min_d")
        ax.set_ylabel("Metapath")

    _save_figure(fig, output_dir / "top_metapaths_by_lv.png")


def _plot_perm_vs_random_scatter(results: pd.DataFrame, output_dir: Path) -> None:
    lvs = sorted(results["lv_id"].unique())
    if not lvs:
        return

    n_cols = min(3, max(1, len(lvs)))
    n_rows = int(np.ceil(len(lvs) / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx, lv_id in enumerate(lvs):
        ax = axes_flat[idx]
        subset = results[results["lv_id"] == lv_id].copy()
        sns.scatterplot(
            data=subset,
            x="d_perm",
            y="d_rand",
            hue="supported",
            palette={True: "#059669", False: "#9ca3af"},
            alpha=0.8,
            s=30,
            ax=ax,
            legend=False,
        )
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        lim = np.nanmax(np.abs(subset[["d_perm", "d_rand"]].to_numpy()))
        if np.isfinite(lim) and lim > 0:
            lim *= 1.05
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.3)
        ax.set_title(f"{lv_id}: d_perm vs d_rand")
        ax.set_xlabel("d_perm")
        ax.set_ylabel("d_rand")

    for idx in range(len(lvs), len(axes_flat)):
        axes_flat[idx].axis("off")

    _save_figure(fig, output_dir / "perm_vs_random_effect_scatter.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate overall LV metapath-performance plots and summaries."
    )
    parser.add_argument(
        "--results",
        default="output/lv_multidwpc/lv_metapath_results.csv",
        help="Path to lv_metapath_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="output/lv_multidwpc/performance_plots",
        help="Directory where plots and summary tables are written.",
    )
    parser.add_argument(
        "--top-n-heatmap",
        type=int,
        default=60,
        help="Number of metapaths to show in overview heatmaps.",
    )
    parser.add_argument(
        "--top-n-bars",
        type=int,
        default=20,
        help="Number of metapaths per LV in top-metapath bar plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    sns.set_theme(style="whitegrid", context="talk")
    results = pd.read_csv(results_path)
    missing = sorted(REQUIRED_COLUMNS - set(results.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {results_path}: {missing}"
        )

    results = _coerce_columns(results)
    metapath_summary = _build_metapath_summary(results)
    lv_summary = _build_lv_summary(results)

    metapath_summary = metapath_summary.sort_values(
        ["lv_id", "supported", "min_d", "min_diff"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    lv_summary = lv_summary.sort_values("lv_id").reset_index(drop=True)

    metapath_summary_path = output_dir / "lv_metapath_performance_summary.csv"
    lv_summary_path = output_dir / "lv_overall_performance_summary.csv"
    metapath_summary.to_csv(metapath_summary_path, index=False)
    lv_summary.to_csv(lv_summary_path, index=False)

    _plot_support_rate(lv_summary=lv_summary, output_dir=output_dir)
    _plot_effect_heatmap(
        metapath_summary=metapath_summary,
        output_dir=output_dir,
        top_n=max(1, int(args.top_n_heatmap)),
    )
    _plot_significance_heatmap(
        metapath_summary=metapath_summary,
        output_dir=output_dir,
        top_n=max(1, int(args.top_n_heatmap)),
    )
    _plot_top_metapaths(
        metapath_summary=metapath_summary,
        output_dir=output_dir,
        top_n=max(1, int(args.top_n_bars)),
    )
    _plot_perm_vs_random_scatter(results=results, output_dir=output_dir)

    print("LV performance analysis complete.")
    print(f"  Input: {results_path}")
    print(f"  Plots + summaries: {output_dir}")
    print(f"  LVs: {results['lv_id'].nunique()}")
    print(f"  Rows: {len(results):,}")


if __name__ == "__main__":
    main()
