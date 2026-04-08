#!/usr/bin/env python3
"""Plot LV null-variance summaries from existing aggregate CSV outputs."""

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


def _entity_mean(feature_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    return (
        feature_df.groupby(["control", "b", "lv_id"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: f"mean_{metric_col}"})
        .sort_values(["control", "lv_id", "b"])
    )


def _compute_elbow(mean_df: pd.DataFrame, metric_col: str, increasing: bool) -> pd.DataFrame:
    rows: list[dict] = []
    value_col = f"mean_{metric_col}"
    for (control, lv_id), group in mean_df.groupby(["control", "lv_id"], sort=True):
        curve = group.sort_values("b").copy()
        if len(curve) < 3:
            continue
        x = curve["b"].astype(float).to_numpy()
        y = curve[value_col].astype(float).to_numpy()
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if y_max > y_min:
            y_norm = (y - y_min) / (y_max - y_min)
        else:
            y_norm = np.zeros_like(y)
        if not increasing:
            y_norm = 1.0 - y_norm
        line = np.linspace(y_norm[0], y_norm[-1], len(y_norm))
        distance = y_norm - line
        idx = int(np.argmax(distance))
        rows.append(
            {
                "control": str(control),
                "lv_id": str(lv_id),
                "metric": str(metric_col),
                "elbow_b": int(curve["b"].iloc[idx]),
                "elbow_mean_value": float(curve[value_col].iloc[idx]),
                "elbow_distance": float(distance[idx]),
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
    controls = sorted(feature_df["control"].dropna().astype(str).unique().tolist())
    lv_ids = sorted(feature_df["lv_id"].dropna().astype(str).unique().tolist())
    if not controls or not lv_ids:
        raise ValueError("LV feature summary must contain control and lv_id values")

    mean_df = _entity_mean(feature_df, metric_col)
    colors = _color_map(lv_ids)
    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, control in zip(axes, controls):
        control_points = feature_df[feature_df["control"].astype(str) == control].copy()
        control_mean = mean_df[mean_df["control"].astype(str) == control].copy()
        for lv_id in lv_ids:
            points = control_points[control_points["lv_id"].astype(str) == lv_id].copy()
            if points.empty:
                continue
            b_values = points["b"].astype(float).to_numpy()
            jitter = rng.uniform(-0.14, 0.14, size=len(points))
            ax.scatter(
                b_values + jitter,
                points[metric_col].astype(float),
                s=28,
                alpha=0.30,
                color=colors[lv_id],
                edgecolors="none",
            )
            line = control_mean[control_mean["lv_id"].astype(str) == lv_id].copy().sort_values("b")
            ax.plot(
                line["b"].astype(float),
                line[f"mean_{metric_col}"].astype(float),
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                color=colors[lv_id],
                label=lv_id,
            )
        ax.set_xlabel("Null replicate count (B)")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(title="LV", loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/lv_experiment/lv_null_variance_experiment",
        help="Directory containing feature_variance_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)

    feature_df = _load_csv(
        analysis_dir / "feature_variance_summary.csv",
        required_columns=["control", "lv_id", "b", "diff_std", "diff_var"],
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
        title="LV null SD by B",
        output_path=analysis_dir / "sd_points_with_mean_trend_by_b.png",
    )
    _plot_metric(
        feature_df=feature_df,
        metric_col="diff_var",
        y_label="Variance(diff) across seeds",
        title="LV null variance by B",
        output_path=analysis_dir / "variance_points_with_mean_trend_by_b.png",
    )

    print(f"Saved plot: {analysis_dir / 'sd_points_with_mean_trend_by_b.png'}")
    print(f"Saved plot: {analysis_dir / 'variance_points_with_mean_trend_by_b.png'}")
    if not elbow_df.empty:
        print(f"Saved summary: {analysis_dir / 'elbow_summary.csv'}")


if __name__ == "__main__":
    main()
