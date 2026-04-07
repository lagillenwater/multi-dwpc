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


def _compute_elbow(mean_df: pd.DataFrame, metric_col: str, increasing: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for (control, year), group in mean_df.groupby(["control", "year"], sort=True):
        curve = group.sort_values("b").copy()
        if len(curve) < 3:
            continue
        x = curve["b"].astype(float).to_numpy()
        y = curve[f"mean_{metric_col}"].astype(float).to_numpy()
        x_log = np.log10(x)
        x_norm = (x_log - x_log.min()) / (x_log.max() - x_log.min()) if x_log.max() > x_log.min() else np.zeros_like(x_log)
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if y_max > y_min:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = np.zeros_like(y)
        progress = y_norm if increasing else 1.0 - y_norm
        line = np.linspace(progress[0], progress[-1], len(progress))
        distance = progress - line
        geometric_idx = int(np.argmax(distance))

        target_fraction = 0.90
        if increasing:
            target_value = y_min + (target_fraction * (y_max - y_min))
            target_candidates = np.where(y >= target_value)[0]
        else:
            target_value = y_max - (target_fraction * (y_max - y_min))
            target_candidates = np.where(y <= target_value)[0]
        target_idx = int(target_candidates[0]) if len(target_candidates) else len(curve) - 1
        idx = max(geometric_idx, target_idx)
        rows.append(
            {
                "control": str(control),
                "year": int(year),
                "metric": str(metric_col),
                "elbow_b": int(curve["b"].iloc[idx]),
                "elbow_mean_value": float(curve[f"mean_{metric_col}"].iloc[idx]),
                "elbow_distance": float(distance[idx]),
                "geometric_elbow_b": int(curve["b"].iloc[geometric_idx]),
                "geometric_distance": float(distance[geometric_idx]),
                "target_fraction": float(target_fraction),
                "target_b": int(curve["b"].iloc[target_idx]),
                "elbow_method": "max_geometric_and_target",
            }
        )
    return pd.DataFrame(rows)


def _plot_metric(
    feature_df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    feature_df = feature_df.copy()
    feature_df["year"] = feature_df["year"].astype(int)

    # Canonical x-axis bin for plotting/ticks
    feature_df["b_plot"] = feature_df["b"].astype(float).round().astype(int)

    controls = sorted(feature_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(feature_df["year"].dropna().astype(int).unique().tolist())
    if not controls or not years:
        raise ValueError("Year feature summary must contain control and year values")

    mean_df = _entity_mean(feature_df, metric_col)
    mean_df = mean_df.copy()
    mean_df["b_plot"] = mean_df["b"].astype(float).round().astype(int)

    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    for ax, control in zip(axes, controls):
        control_points = feature_df[feature_df["control"].astype(str) == control].copy()
        control_mean = mean_df[mean_df["control"].astype(str) == control].copy()

        width = 0.32 if len(years) > 1 else 0.55
        offsets = {
            int(y): ((idx - (len(years) - 1) / 2.0) * width)
            for idx, y in enumerate(years)
        }

        all_b = sorted(control_points["b_plot"].dropna().astype(int).unique().tolist())

        for year in years:
            color = YEAR_COLORS.get(str(year), "#333333")
            points = control_points[control_points["year"].astype(int) == int(year)].copy()
            if points.empty:
                continue

            b_vals = sorted(points["b_plot"].astype(int).unique().tolist())
            positions = [b + offsets[int(year)] for b in b_vals]

            box_data = [
                points.loc[points["b_plot"].astype(int) == b, metric_col].astype(float).to_numpy()
                for b in b_vals
            ]

            ax.boxplot(
                box_data,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.5, "edgecolor": color},
                medianprops={"color": color, "linewidth": 1.4},
                whiskerprops={"color": color, "alpha": 0.5},
                capprops={"color": color, "alpha": 0.5},
                showfliers=False,
            )

            line = (
                control_mean[control_mean["year"].astype(int) == int(year)]
                .groupby("b_plot", as_index=False)[f"mean_{metric_col}"]
                .mean()
                .sort_values("b_plot")
            )

            ax.plot(
                line["b_plot"].astype(float),
                line[f"mean_{metric_col}"].astype(float),
                marker="o",
                linewidth=2.2,
                markersize=1,
                color=color,
                alpha=0.4,
                label=str(year),
            )

        ax.set_xlabel("Null replicate count (B)")
        ax.set_xticks(all_b)
        ax.set_xticklabels([str(b) for b in all_b])
        ax.tick_params(axis="x", labelrotation=0)
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

    mean_std_df = _entity_mean(feature_df, "diff_std")
    mean_var_df = _entity_mean(feature_df, "diff_var")
    elbow_df = pd.concat(
        [
            _compute_elbow(mean_std_df, "diff_std", increasing=False),
            _compute_elbow(mean_var_df, "diff_var", increasing=False),
        ],
        ignore_index=True,
    )
    if not elbow_df.empty:
        elbow_df.to_csv(analysis_dir / "elbow_summary.csv", index=False)

    _plot_metric(
        feature_df=feature_df,
        metric_col="diff_std",
        y_label="SD(diff) across seeds",
        title="Year null SD by B",
        output_path=analysis_dir / "sd_boxplots_with_mean_trend_by_b.pdf",
    )
    _plot_metric(
        feature_df=feature_df,
        metric_col="diff_var",
        y_label="Variance(diff) across seeds",
        title="Year null variance by B",
        output_path=analysis_dir / "variance_boxplots_with_mean_trend_by_b.pdf",
    )

    if not elbow_df.empty:
        print(f"Saved summary: {analysis_dir / 'elbow_summary.csv'}")
    print(f"Saved plot: {analysis_dir / 'sd_boxplots_with_mean_trend_by_b.pdf'}")
    print(f"Saved plot: {analysis_dir / 'sd_boxplots_with_mean_trend_by_b.png'}")
    print(f"Saved plot: {analysis_dir / 'variance_boxplots_with_mean_trend_by_b.pdf'}")
    print(f"Saved plot: {analysis_dir / 'variance_boxplots_with_mean_trend_by_b.png'}")


if __name__ == "__main__":
    main()
