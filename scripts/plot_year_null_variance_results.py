#!/usr/bin/env python3
"""Plot year null-variance summaries from existing aggregate CSV outputs."""

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


CONTROL_COLORS = {"permuted": "#1f77b4", "random": "#d62728"}


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _normalize_year(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    work = df.copy()
    work[year_col] = work[year_col].astype(int)
    return work


def _plot_metric(
    feature_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    point_metric: str,
    mean_metric: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    controls = sorted(feature_df["control"].dropna().astype(str).unique().tolist())
    if not controls:
        raise ValueError("No control values found in year feature summary")

    year_order = sorted(feature_df["year"].dropna().astype(int).unique().tolist())
    if not year_order:
        raise ValueError("No year values found in year feature summary")
    year_positions = {year: idx for idx, year in enumerate(year_order)}

    fig, axes = plt.subplots(1, len(controls), figsize=(6.8 * len(controls), 5.0), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, control in zip(axes, controls):
        color = CONTROL_COLORS.get(control, "#333333")
        feature_subset = (
            feature_df[feature_df["control"].astype(str) == control]
            .copy()
            .sort_values("year")
        )
        mean_subset = (
            overall_df[overall_df["control"].astype(str) == control]
            .copy()
            .sort_values("year")
        )
        x_center = feature_subset["year"].astype(int).map(year_positions).astype(float).to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(feature_subset))
        ax.scatter(
            x_center + jitter,
            feature_subset[point_metric].astype(float),
            s=34,
            alpha=0.35,
            color=color,
            edgecolors="none",
        )
        ax.plot(
            mean_subset["year"].astype(int).map(year_positions).astype(float),
            mean_subset[mean_metric].astype(float),
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=color,
        )
        ax.set_xticks(list(year_positions.values()))
        ax.set_xticklabels([str(year) for year in year_order])
        ax.set_xlabel("Year")
        ax.set_title(control)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/year_null_variance_exp/year_null_variance_experiment",
        help="Directory containing feature_variance_summary.csv and overall_variance_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)

    feature_df = _normalize_year(
        _load_csv(
            analysis_dir / "feature_variance_summary.csv",
            required_columns=["year", "control", "diff_std", "diff_var"],
        )
    )
    overall_df = _normalize_year(
        _load_csv(
            analysis_dir / "overall_variance_summary.csv",
            required_columns=["year", "control", "mean_diff_std", "mean_diff_var"],
        )
    )

    _plot_metric(
        feature_df=feature_df,
        overall_df=overall_df,
        point_metric="diff_std",
        mean_metric="mean_diff_std",
        y_label="SD(diff) across replicates",
        title="Year per-feature SD by control and year",
        output_path=analysis_dir / "sd_points_with_mean_trend_by_year.png",
    )
    _plot_metric(
        feature_df=feature_df,
        overall_df=overall_df,
        point_metric="diff_var",
        mean_metric="mean_diff_var",
        y_label="Variance(diff) across replicates",
        title="Year per-feature variance by control and year",
        output_path=analysis_dir / "variance_points_with_mean_trend_by_year.png",
    )

    print(f"Saved plot: {analysis_dir / 'sd_points_with_mean_trend_by_year.png'}")
    print(f"Saved plot: {analysis_dir / 'variance_points_with_mean_trend_by_year.png'}")


if __name__ == "__main__":
    main()
