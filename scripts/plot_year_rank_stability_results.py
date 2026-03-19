#!/usr/bin/env python3
"""Plot year rank-stability rho summaries from existing aggregate CSV outputs."""

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


YEAR_COLORS = {
    "2016": "#1f77b4",
    "2024": "#ff7f0e",
}


def _top_k_labels_from_columns(df: pd.DataFrame) -> list[str]:
    labels = []
    for col in df.columns:
        prefix = "mean_topk_jaccard_"
        if col.startswith(prefix):
            labels.append(col[len(prefix):])
    def _sort_key(label: str) -> tuple[int, int | str]:
        if label == "all":
            return (1, label)
        return (0, int(label))
    return sorted(set(labels), key=_sort_key)


def _format_top_k_label(label: str) -> str:
    return "all" if str(label) == "all" else str(int(label))


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _mean_by_year(entity_df: pd.DataFrame) -> pd.DataFrame:
    return (
        entity_df.groupby(["control", "b", "year"], as_index=False)["mean_spearman_rho"]
        .mean()
        .rename(columns={"mean_spearman_rho": "mean_rho_by_year"})
        .sort_values(["control", "year", "b"])
    )


def _mean_topk_by_year(entity_df: pd.DataFrame) -> pd.DataFrame:
    top_k_labels = _top_k_labels_from_columns(entity_df)
    rows = []
    for label in top_k_labels:
        metric_col = f"mean_topk_jaccard_{label}"
        subset = (
            entity_df.groupby(["control", "b", "year"], as_index=False)[metric_col]
            .mean()
            .rename(columns={metric_col: "mean_topk_by_year"})
        )
        subset["top_k"] = label
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_rho(entity_df: pd.DataFrame, output_path: Path) -> None:
    entity_df = entity_df.copy()
    entity_df["year"] = entity_df["year"].astype(int)
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(entity_df["year"].dropna().astype(int).unique().tolist())
    if not controls or not years:
        raise ValueError("Year stability summary must contain control and year values")

    mean_df = _mean_by_year(entity_df)
    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, control in zip(axes, controls):
        control_points = entity_df[entity_df["control"].astype(str) == control].copy()
        control_mean = mean_df[mean_df["control"].astype(str) == control].copy()
        for year in years:
            color = YEAR_COLORS.get(str(year), "#333333")
            points = control_points[control_points["year"].astype(int) == int(year)].copy()
            if points.empty:
                continue
            b_values = points["b"].astype(float).to_numpy()
            jitter = rng.uniform(-0.14, 0.14, size=len(points))
            ax.scatter(
                b_values + jitter,
                points["mean_spearman_rho"].astype(float),
                s=28,
                alpha=0.30,
                color=color,
                edgecolors="none",
            )
            line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
            ax.plot(
                line["b"].astype(float),
                line["mean_rho_by_year"].astype(float),
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                color=color,
                label=str(year),
            )
        ax.set_xlabel("Null replicate count (B)")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean Spearman rho across seed pairs")
    axes[-1].legend(title="Year", loc="best")
    fig.suptitle("Year rank-stability rho by B")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topk(entity_df: pd.DataFrame, output_path: Path) -> None:
    entity_df = entity_df.copy()
    entity_df["year"] = entity_df["year"].astype(int)
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(entity_df["year"].dropna().astype(int).unique().tolist())
    top_k_labels = _top_k_labels_from_columns(entity_df)
    if not controls or not years or not top_k_labels:
        return

    mean_df = _mean_topk_by_year(entity_df)
    fig, axes = plt.subplots(
        len(top_k_labels),
        len(controls),
        figsize=(7.0 * len(controls), 4.8 * len(top_k_labels)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        if len(top_k_labels) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    rng = np.random.default_rng(42)
    for row_idx, top_k_label in enumerate(top_k_labels):
        metric_col = f"mean_topk_jaccard_{top_k_label}"
        for col_idx, control in enumerate(controls):
            ax = axes[row_idx, col_idx]
            control_points = entity_df[entity_df["control"].astype(str) == control].copy()
            control_mean = mean_df[
                (mean_df["control"].astype(str) == control)
                & (mean_df["top_k"].astype(str) == str(top_k_label))
            ].copy()
            for year in years:
                color = YEAR_COLORS.get(str(year), "#333333")
                points = control_points[control_points["year"].astype(int) == int(year)].copy()
                if points.empty:
                    continue
                b_values = points["b"].astype(float).to_numpy()
                jitter = rng.uniform(-0.14, 0.14, size=len(points))
                ax.scatter(
                    b_values + jitter,
                    points[metric_col].astype(float),
                    s=28,
                    alpha=0.30,
                    color=color,
                    edgecolors="none",
                )
                line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
                ax.plot(
                    line["b"].astype(float),
                    line["mean_topk_by_year"].astype(float),
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                    color=color,
                    label=str(year),
                )
            if row_idx == 0:
                ax.set_title(control)
            if col_idx == 0:
                ax.set_ylabel(f"Mean top-{_format_top_k_label(top_k_label)} Jaccard across seeds")
            if row_idx == len(top_k_labels) - 1:
                ax.set_xlabel("B")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
    axes[0, -1].legend(title="Year", loc="best")
    fig.suptitle("Year top-k overlap stability by B")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/year_rank_stability_exp/year_rank_stability_experiment",
        help="Directory containing go_term_stability_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    entity_df = _load_csv(
        analysis_dir / "go_term_stability_summary.csv",
        required_columns=["control", "year", "b", "go_id", "mean_spearman_rho"],
    )
    rho_path = analysis_dir / "rho_points_with_mean_trend_by_b.png"
    _plot_rho(entity_df, rho_path)
    print(f"Saved plot: {rho_path}")
    topk_path = analysis_dir / "topk_jaccard_overall_by_group.png"
    _plot_topk(entity_df, topk_path)
    print(f"Saved plot: {topk_path}")


if __name__ == "__main__":
    main()
