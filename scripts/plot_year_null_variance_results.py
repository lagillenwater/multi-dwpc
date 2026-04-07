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


YEAR_COLORS = {
    "2016": "#1f77b4",
    "2024": "#ff7f0e",
}


def _save_dual(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _entity_mean(feature_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    return (
        feature_df.groupby(["control", "b", "year"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: f"mean_{metric_col}"})
        .sort_values(["control", "year", "b"])
    )


def _plot_metric(
    feature_df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    feature_df = feature_df.copy()
    feature_df["year"] = feature_df["year"].astype(int)
    controls = sorted(feature_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(feature_df["year"].dropna().astype(int).unique().tolist())
    if not controls or not years:
        raise ValueError("Year feature summary must contain control and year values")

    mean_df = _entity_mean(feature_df, metric_col)
    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, control in zip(axes, controls):
        control_points = feature_df[feature_df["control"].astype(str) == control].copy()
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
                points[metric_col].astype(float),
                s=28,
                alpha=0.30,
                color=color,
                edgecolors="none",
            )
            line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
            ax.plot(
                line["b"].astype(float),
                line[f"mean_{metric_col}"].astype(float),
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                color=color,
                label=str(year),
            )
        ax.set_xlabel("Null replicate count (B)")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(title="Year", loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    _save_dual(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/year_null_variance_exp/year_null_variance_experiment",
        help="Directory containing feature_variance_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    feature_path = analysis_dir / "feature_variance_summary.csv"
    if not feature_path.exists():
        alt = analysis_dir / "feature_variance_summary_supported_only.csv"
        if alt.exists():
            feature_path = alt

    feature_df = _load_csv(
        feature_path,
        required_columns=["control", "year", "b", "diff_std", "diff_var"],
    )

    _plot_metric(
        feature_df=feature_df,
        metric_col="diff_std",
        y_label="SD(diff) across seeds",
        title="Year null SD by B",
        output_path=analysis_dir / "sd_points_with_mean_trend_by_b.pdf",
    )
    _plot_metric(
        feature_df=feature_df,
        metric_col="diff_var",
        y_label="Variance(diff) across seeds",
        title="Year null variance by B",
        output_path=analysis_dir / "variance_points_with_mean_trend_by_b.pdf",
    )

    print(f"Saved plot: {analysis_dir / 'sd_points_with_mean_trend_by_b.pdf'}")
    print(f"Saved plot: {analysis_dir / 'sd_points_with_mean_trend_by_b.png'}")
    print(f"Saved plot: {analysis_dir / 'variance_points_with_mean_trend_by_b.pdf'}")
    print(f"Saved plot: {analysis_dir / 'variance_points_with_mean_trend_by_b.png'}")


if __name__ == "__main__":
    main()
