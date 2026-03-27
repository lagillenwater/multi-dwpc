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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-dir",
        default="output/lv_experiment_all_metapaths/lv_group_qc_experiment",
        help="Directory containing LV group QC CSV outputs",
    )
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

    group_qc_df = _group_label(group_qc_df)
    within_df = _group_label(within_df)
    between_df = _group_label(between_df)
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
        "LV QC\n"
        "Top: within-null seed stability | Bottom: random-vs-permuted agreement\n"
        "Spearman rho = whole-ranking agreement | RBO = top-weighted agreement",
        fontsize=18,
        y=0.98,
    )
    out_path = qc_dir / "lv_group_qc_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
