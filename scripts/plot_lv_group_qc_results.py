#!/usr/bin/env python3
"""Create a single dashboard figure for LV group QC results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

matplotlib.use("Agg")


GROUP_COLORS = {
    "LV246 | adipose_tissue": "#1f77b4",
    "LV57 | hypothyroidism": "#ff7f0e",
    "LV603 | neutrophil_bp": "#2ca02c",
}

NULL_STYLES = {
    "permuted": ("-", "o"),
    "random": ("--", "s"),
}


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _group_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_label"] = out["lv_id"].astype(str) + " | " + out["target_set_id"].astype(str)
    return out


def _ordered_group_labels(group_qc_df: pd.DataFrame) -> list[str]:
    df = _group_label(group_qc_df)
    return df["group_label"].astype(str).tolist()


def _group_color(label: str) -> str:
    return GROUP_COLORS.get(label, "#444444")


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value)).strip("_")


def _metric_specs(df: pd.DataFrame) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = [("mean_spearman_rho", "Spearman rho")]
    if "mean_rbo" in df.columns:
        specs.append(("mean_rbo", "Rank-biased overlap"))
    return specs


def _plot_within_null_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    group_order: list[str],
) -> None:
    metric_vals: list[float] = []
    for group_label in group_order:
        gdf = df[df["group_label"].astype(str) == group_label].copy()
        color = _group_color(group_label)
        for control in sorted(gdf["control"].astype(str).unique().tolist()):
            cdf = gdf[gdf["control"].astype(str) == control].copy().sort_values("b")
            if cdf.empty or metric_col not in cdf.columns:
                continue
            metric_vals.extend(cdf[metric_col].astype(float).tolist())
            linestyle, marker = NULL_STYLES.get(control, ("-", "o"))
            ax.plot(
                cdf["b"].astype(float),
                cdf[metric_col].astype(float),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5.5,
                color=color,
                alpha=0.95,
                label=f"{group_label} [{control}]",
            )
    ax.set_title(metric_label)
    ax.set_xlabel("B")
    ax.set_ylabel("Within-null seed stability")
    if metric_vals:
        ymin = float(np.nanmin(metric_vals))
        ymax = float(np.nanmax(metric_vals))
        span = max(0.02, ymax - ymin)
        lower = max(0.0, ymin - 0.15 * span)
        upper = min(1.02, ymax + 0.10 * span)
        ax.set_ylim(lower, upper)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_between_null_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    group_order: list[str],
) -> None:
    metric_vals: list[float] = []
    for group_label in group_order:
        gdf = df[df["group_label"].astype(str) == group_label].copy().sort_values("b")
        if gdf.empty or metric_col not in gdf.columns:
            continue
        color = _group_color(group_label)
        metric_vals.extend(gdf[metric_col].astype(float).tolist())
        ax.plot(
            gdf["b"].astype(float),
            gdf[metric_col].astype(float),
            linestyle="-",
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
            alpha=0.95,
            label=group_label,
        )
    ax.set_title(metric_label)
    ax.set_xlabel("B")
    ax.set_ylabel("Random vs permuted agreement")
    if metric_vals:
        ymin = float(np.nanmin(metric_vals))
        ymax = float(np.nanmax(metric_vals))
        span = max(0.02, ymax - ymin)
        lower = max(0.0, ymin - 0.15 * span)
        upper = min(1.02, ymax + 0.10 * span)
        ax.set_ylim(lower, upper)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_pairwise_rbo_heatmap(
    ax: plt.Axes,
    pairwise_df: pd.DataFrame,
    seeds: list[int],
) -> None:
    matrix = pd.DataFrame(np.nan, index=seeds, columns=seeds, dtype=float)
    for seed in seeds:
        matrix.loc[seed, seed] = 1.0
    for row in pairwise_df.itertuples(index=False):
        matrix.loc[int(row.seed_a), int(row.seed_b)] = float(row.rbo)
        matrix.loc[int(row.seed_b), int(row.seed_a)] = float(row.rbo)
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", norm=Normalize(vmin=0.0, vmax=1.0))
    ax.set_xticks(range(len(seeds)), labels=[str(seed) for seed in seeds])
    ax.set_yticks(range(len(seeds)), labels=[str(seed) for seed in seeds])
    ax.set_title("Pairwise seed RBO")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    for i, seed_a in enumerate(seeds):
        for j, seed_b in enumerate(seeds):
            value = matrix.loc[seed_a, seed_b]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_prefix_overlap(
    ax: plt.Axes,
    prefix_df: pd.DataFrame,
) -> None:
    max_k = 0
    metric_vals: list[float] = []
    for row_key, group in prefix_df.groupby(["seed_a", "seed_b"], sort=True):
        seed_a, seed_b = row_key
        ordered = group.sort_values("k")
        max_k = max(max_k, int(ordered["k"].max()))
        vals = ordered["prefix_jaccard"].astype(float).tolist()
        metric_vals.extend(vals)
        ax.plot(
            ordered["k"].astype(int),
            ordered["prefix_jaccard"].astype(float),
            linewidth=1.8,
            alpha=0.9,
            label=f"{seed_a} vs {seed_b}",
        )
    ax.set_title("Prefix-overlap curve")
    ax.set_xlabel("Top-k prefix")
    ax.set_ylabel("Jaccard")
    if metric_vals:
        ymin = float(np.nanmin(metric_vals))
        ymax = float(np.nanmax(metric_vals))
        span = max(0.02, ymax - ymin)
        ax.set_ylim(max(0.0, ymin - 0.10 * span), min(1.02, ymax + 0.10 * span))
    if max_k:
        ax.set_xlim(1, max_k)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, loc="lower right", ncol=2)


def _plot_between_null_scatter(
    ax: plt.Axes,
    scatter_df: pd.DataFrame,
    group_label: str,
    b: int,
) -> None:
    if scatter_df.empty:
        ax.axis("off")
        return
    color = _group_color(group_label)
    ax.scatter(
        scatter_df["mean_rank_permuted"].astype(float),
        scatter_df["mean_rank_random"].astype(float),
        s=38,
        alpha=0.78,
        color=color,
    )
    max_rank = float(
        scatter_df[["mean_rank_permuted", "mean_rank_random"]].astype(float).to_numpy().max()
    )
    ax.plot([0.5, max_rank + 0.5], [0.5, max_rank + 0.5], linestyle="--", color="#999999", linewidth=1.4)
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.set_title(f"Random vs permuted scatter (B={b})")
    ax.set_xlabel("Permuted mean rank")
    ax.set_ylabel("Random mean rank")
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _write_diagnostic_plots(
    qc_dir: Path,
    pairwise_df: pd.DataFrame,
    prefix_df: pd.DataFrame,
    scatter_df: pd.DataFrame,
    group_order: list[str],
    diagnostic_b: int,
    diagnostic_control: str,
) -> None:
    diag_dir = qc_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for group_label in group_order:
        lv_id, target_set_id = [part.strip() for part in group_label.split("|", maxsplit=1)]
        group_pairwise = pairwise_df[
            (pairwise_df["lv_id"].astype(str) == lv_id)
            & (pairwise_df["target_set_id"].astype(str) == target_set_id)
            & (pairwise_df["control"].astype(str) == str(diagnostic_control))
            & (pairwise_df["b"].astype(int) == int(diagnostic_b))
        ].copy()
        group_prefix = prefix_df[
            (prefix_df["lv_id"].astype(str) == lv_id)
            & (prefix_df["target_set_id"].astype(str) == target_set_id)
            & (prefix_df["control"].astype(str) == str(diagnostic_control))
            & (prefix_df["b"].astype(int) == int(diagnostic_b))
        ].copy()
        group_scatter = scatter_df[
            (scatter_df["lv_id"].astype(str) == lv_id)
            & (scatter_df["target_set_id"].astype(str) == target_set_id)
            & (scatter_df["b"].astype(int) == int(diagnostic_b))
        ].copy()
        if group_pairwise.empty and group_prefix.empty and group_scatter.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        seeds = sorted(
            set(group_pairwise["seed_a"].astype(int).tolist()) | set(group_pairwise["seed_b"].astype(int).tolist())
        )
        if group_pairwise.empty or not seeds:
            axes[0].axis("off")
        else:
            _plot_pairwise_rbo_heatmap(axes[0], group_pairwise, seeds)
        if group_prefix.empty:
            axes[1].axis("off")
        else:
            _plot_prefix_overlap(axes[1], group_prefix)
        _plot_between_null_scatter(axes[2], group_scatter, group_label, diagnostic_b)
        fig.suptitle(
            f"LVQC diagnostics: {group_label} | control={diagnostic_control} | B={diagnostic_b}",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = diag_dir / f"{_sanitize(group_label)}_{_sanitize(diagnostic_control)}_b{int(diagnostic_b)}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-dir",
        default="output/lv_experiment_all_metapaths/lv_group_qc_experiment",
        help="Directory containing LV group QC CSV outputs",
    )
    parser.add_argument("--diagnostic-b", type=int, default=5)
    parser.add_argument("--diagnostic-control", default="permuted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc_dir = Path(args.qc_dir)

    group_qc_df = _load_csv(
        qc_dir / "group_qc_summary.csv",
        ["lv_id", "target_set_id", "tier", "recommended_b", "descriptor_status"],
    )
    within_df = _load_csv(
        qc_dir / "within_null_stability_summary.csv",
        ["control", "b", "lv_id", "target_set_id", "mean_spearman_rho"],
    )
    between_df = _load_csv(
        qc_dir / "between_null_agreement.csv",
        ["b", "lv_id", "target_set_id", "mean_spearman_rho"],
    )
    pairwise_df = _load_csv(
        qc_dir / "within_null_pairwise_diagnostics.csv",
        ["control", "b", "lv_id", "target_set_id", "seed_a", "seed_b", "rbo"],
    )
    prefix_df = _load_csv(
        qc_dir / "within_null_prefix_overlap.csv",
        ["control", "b", "lv_id", "target_set_id", "seed_a", "seed_b", "k", "prefix_jaccard"],
    )
    scatter_df = _load_csv(
        qc_dir / "between_null_rank_scatter.csv",
        ["b", "lv_id", "target_set_id", "metapath", "mean_rank_permuted", "mean_rank_random"],
    )

    group_qc_df = _group_label(group_qc_df)
    within_df = _group_label(within_df)
    between_df = _group_label(between_df)
    pairwise_df = _group_label(pairwise_df)
    prefix_df = _group_label(prefix_df)
    scatter_df = _group_label(scatter_df)
    group_order = _ordered_group_labels(group_qc_df)

    within_specs = _metric_specs(within_df)
    between_specs = _metric_specs(between_df)

    fig = plt.figure(figsize=(16, 8.2))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.25,
    )

    within_grid = gs[0, :].subgridspec(1, 2, hspace=0.35, wspace=0.25)
    within_axes = [fig.add_subplot(within_grid[0, j]) for j in range(2)]
    for ax, (metric_col, metric_label) in zip(within_axes, within_specs):
        _plot_within_null_metric(ax, within_df, metric_col, metric_label, group_order)
    for ax in within_axes[len(within_specs) :]:
        ax.axis("off")
    if within_axes:
        handles, labels = within_axes[0].get_legend_handles_labels()
        if handles:
            within_axes[0].legend(handles, labels, fontsize=8, loc="lower right", ncol=2)

    between_grid = gs[1, :].subgridspec(1, 2, hspace=0.35, wspace=0.25)
    between_axes = [fig.add_subplot(between_grid[0, j]) for j in range(2)]
    for ax, (metric_col, metric_label) in zip(between_axes, between_specs):
        _plot_between_null_metric(ax, between_df, metric_col, metric_label, group_order)
    for ax in between_axes[len(between_specs) :]:
        ax.axis("off")
    if between_axes:
        handles, labels = between_axes[0].get_legend_handles_labels()
        if handles:
            between_axes[0].legend(handles, labels, fontsize=8, loc="lower right")

    fig.suptitle(
        "LVQC",
        fontsize=18,
        y=0.995,
    )
    out_path = qc_dir / "lv_group_qc_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _write_diagnostic_plots(
        qc_dir=qc_dir,
        pairwise_df=pairwise_df,
        prefix_df=prefix_df,
        scatter_df=scatter_df,
        group_order=group_order,
        diagnostic_b=int(args.diagnostic_b),
        diagnostic_control=str(args.diagnostic_control),
    )
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
