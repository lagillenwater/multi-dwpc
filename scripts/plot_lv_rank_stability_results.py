#!/usr/bin/env python3
"""Plot LV rank-stability rho summaries from existing aggregate CSV outputs."""

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


LV_COLORS = {
    "LV246": "#1f77b4",
    "LV57": "#ff7f0e",
    "LV603": "#2ca02c",
}


def _top_k_labels_from_columns(df: pd.DataFrame) -> list[str]:
    labels = []
    for col in df.columns:
        prefix = "mean_topk_jaccard_"
        if col.startswith(prefix):
            labels.append(col[len(prefix):])
    return sorted(set(labels), key=int)


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _color_map(ids: list[str]) -> dict[str, str]:
    fallback = plt.get_cmap("tab10")
    colors = {}
    for idx, entity_id in enumerate(ids):
        colors[entity_id] = LV_COLORS.get(entity_id, fallback(idx % 10))
    return colors


def _load_entity_df(analysis_dir: Path) -> pd.DataFrame:
    summary_path = analysis_dir / "lv_stability_summary.csv"
    if summary_path.exists():
        return _load_csv(
            summary_path,
            required_columns=["control", "b", "lv_id", "target_set_id", "mean_spearman_rho"],
        )

    legacy_path = analysis_dir / "metapath_pairwise_metrics.csv"
    legacy_df = _load_csv(
        legacy_path,
        required_columns=["b", "lv_id", "target_set_id", "spearman_rho"],
    )
    legacy_df = (
        legacy_df.groupby(["b", "lv_id", "target_set_id"], as_index=False)
        .agg(
            n_pairs=("spearman_rho", "size"),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
        )
        .sort_values(["lv_id", "target_set_id", "b"])
        .reset_index(drop=True)
    )
    legacy_df.insert(0, "control", "combined")
    return legacy_df


def _mean_by_lv(entity_df: pd.DataFrame) -> pd.DataFrame:
    return (
        entity_df.groupby(["control", "b", "lv_id"], as_index=False)["mean_spearman_rho"]
        .mean()
        .rename(columns={"mean_spearman_rho": "mean_rho_by_lv"})
        .sort_values(["control", "lv_id", "b"])
    )


def _mean_metric_by_lv(plot_df: pd.DataFrame) -> pd.DataFrame:
    return (
        plot_df.groupby(["metric_key", "control", "b", "lv_id"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "mean_metric_value"})
    )


def _plot_rho(entity_df: pd.DataFrame, output_path: Path) -> None:
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    lv_ids = sorted(entity_df["lv_id"].dropna().astype(str).unique().tolist())
    if not controls or not lv_ids:
        raise ValueError("LV stability summary must contain control and lv_id values")

    mean_df = _mean_by_lv(entity_df)
    colors = _color_map(lv_ids)
    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, control in zip(axes, controls):
        control_points = entity_df[entity_df["control"].astype(str) == control].copy()
        control_mean = mean_df[mean_df["control"].astype(str) == control].copy()
        for lv_id in lv_ids:
            points = control_points[control_points["lv_id"].astype(str) == lv_id].copy()
            if points.empty:
                continue
            b_values = points["b"].astype(float).to_numpy()
            jitter = rng.uniform(-0.14, 0.14, size=len(points))
            ax.scatter(
                b_values + jitter,
                points["mean_spearman_rho"].astype(float),
                s=28,
                alpha=0.30,
                color=colors[lv_id],
                edgecolors="none",
            )
            line = control_mean[control_mean["lv_id"].astype(str) == lv_id].copy().sort_values("b")
            ax.plot(
                line["b"].astype(float),
                line["mean_rho_by_lv"].astype(float),
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                color=colors[lv_id],
                label=lv_id,
            )
        ax.set_xlabel("Null replicate count (B)")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean Spearman rho across seed pairs")
    axes[-1].legend(title="LV", loc="best")
    fig.suptitle("LV rank-stability rho by B")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overlap_and_rank(entity_df: pd.DataFrame, output_path: Path) -> None:
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    lv_ids = sorted(entity_df["lv_id"].dropna().astype(str).unique().tolist())
    top_k_labels = _top_k_labels_from_columns(entity_df)
    if not controls or not lv_ids:
        raise ValueError("LV stability summary must contain control and lv_id values")

    metric_specs: list[tuple[str, str, str]] = []
    if "5" in top_k_labels:
        metric_specs.append(("top5", "mean_topk_jaccard_5", "Mean top-5 Jaccard across seeds"))
    if "10" in top_k_labels:
        metric_specs.append(("top10", "mean_topk_jaccard_10", "Mean top-10 Jaccard across seeds"))
    metric_specs.append(("all", "mean_spearman_rho", "Mean Spearman rho across seeds"))

    plot_rows = []
    for metric_key, metric_col, metric_label in metric_specs:
        subset = entity_df[["control", "b", "lv_id", "target_set_id", metric_col]].copy()
        subset = subset.rename(columns={metric_col: "metric_value"})
        subset["metric_key"] = metric_key
        subset["metric_label"] = metric_label
        plot_rows.append(subset)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    mean_df = _mean_metric_by_lv(plot_df)
    colors = _color_map(lv_ids)
    fig, axes = plt.subplots(
        len(metric_specs),
        len(controls),
        figsize=(7.0 * len(controls), 4.8 * len(metric_specs)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        if len(metric_specs) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    rng = np.random.default_rng(42)
    for row_idx, (metric_key, _, metric_label) in enumerate(metric_specs):
        for col_idx, control in enumerate(controls):
            ax = axes[row_idx, col_idx]
            control_points = plot_df[
                (plot_df["metric_key"].astype(str) == metric_key)
                & (plot_df["control"].astype(str) == control)
            ].copy()
            control_mean = mean_df[
                (mean_df["metric_key"].astype(str) == metric_key)
                & (mean_df["control"].astype(str) == control)
            ].copy()
            for lv_id in lv_ids:
                points = control_points[control_points["lv_id"].astype(str) == lv_id].copy()
                if points.empty:
                    continue
                b_values = points["b"].astype(float).to_numpy()
                jitter = rng.uniform(-0.14, 0.14, size=len(points))
                ax.scatter(
                    b_values + jitter,
                    points["metric_value"].astype(float),
                    s=28,
                    alpha=0.30,
                    color=colors[lv_id],
                    edgecolors="none",
                )
                line = control_mean[control_mean["lv_id"].astype(str) == lv_id].copy().sort_values("b")
                ax.plot(
                    line["b"].astype(float),
                    line["mean_metric_value"].astype(float),
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                    color=colors[lv_id],
                    label=lv_id,
                )
            if row_idx == 0:
                ax.set_title(control)
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(metric_specs) - 1:
                ax.set_xlabel("B")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
    axes[0, -1].legend(title="LV", loc="best")
    fig.suptitle("LV overlap and rank stability by B")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/lv_experiment/lv_rank_stability_experiment",
        help="Directory containing lv_stability_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    entity_df = _load_entity_df(analysis_dir)
    rho_path = analysis_dir / "rho_points_with_mean_trend_by_b.png"
    _plot_rho(entity_df, rho_path)
    print(f"Saved plot: {rho_path}")
    topk_path = analysis_dir / "topk_jaccard_overall_by_group.png"
    _plot_overlap_and_rank(entity_df, topk_path)
    print(f"Saved plot: {topk_path}")


if __name__ == "__main__":
    main()
