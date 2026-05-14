#!/usr/bin/env python3
"""Plot null-variance summaries from existing aggregate CSV outputs.

Replaces `plot_year_null_variance_results.py` and
`plot_lv_null_variance_results.py`. Dispatches on `--analysis-type {year,lv}`.

Year: boxplots-with-mean-overlay per (control, year), boxes offset within B.
LV:   scatter-with-jitter per (control, lv_id), mean line per LV.
"""

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

LV_COLORS = {
    "LV246": "#1f77b4",
    "LV57": "#ff7f0e",
    "LV603": "#2ca02c",
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


def _lv_color_map(lv_ids: list[str]) -> dict:
    fallback = plt.get_cmap("tab10")
    return {lv_id: LV_COLORS.get(lv_id, fallback(idx % 10)) for idx, lv_id in enumerate(lv_ids)}


def _entity_mean(feature_df: pd.DataFrame, metric_col: str, group_keys: list[str]) -> pd.DataFrame:
    return (
        feature_df.groupby(group_keys, as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: f"mean_{metric_col}"})
        .sort_values(group_keys)
    )


def _compute_elbow(
    mean_df: pd.DataFrame, metric_col: str, increasing: bool, group_keys: list[str], entity_col: str
) -> pd.DataFrame:
    """Generic elbow detection on per-entity mean curves vs B."""
    rows: list[dict] = []
    value_col = f"mean_{metric_col}"
    for key_tuple, group in mean_df.groupby(group_keys, sort=True):
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        entity_value = key_tuple[group_keys.index(entity_col)]
        control = key_tuple[group_keys.index("control")]
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
        row = {
            "control": str(control),
            entity_col: int(entity_value) if entity_col == "year" else str(entity_value),
            "metric": str(metric_col),
            "elbow_b": int(curve["b"].iloc[idx]),
            "elbow_mean_value": float(curve[value_col].iloc[idx]),
            "elbow_distance": float(distance[idx]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Year-specific plot (boxplots with year offsets + mean line)
# ---------------------------------------------------------------------------


def _plot_metric_year(
    feature_df: pd.DataFrame, metric_col: str, y_label: str, title: str, output_path: Path
) -> None:
    feature_df = feature_df.copy()
    feature_df["year"] = feature_df["year"].astype(int)
    feature_df["b_plot"] = feature_df["b"].astype(float).round().astype(int)

    controls = sorted(feature_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(feature_df["year"].dropna().astype(int).unique().tolist())
    if not controls or not years:
        raise ValueError("Year feature summary must contain control and year values")

    mean_df = _entity_mean(feature_df, metric_col, ["control", "b", "year"]).copy()
    mean_df["b_plot"] = mean_df["b"].astype(float).round().astype(int)

    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    for ax, control in zip(axes, controls):
        control_points = feature_df[feature_df["control"].astype(str) == control].copy()
        control_mean = mean_df[mean_df["control"].astype(str) == control].copy()
        width = 0.32 if len(years) > 1 else 0.55
        offsets = {int(y): ((idx - (len(years) - 1) / 2.0) * width) for idx, y in enumerate(years)}
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


# ---------------------------------------------------------------------------
# LV-specific plot (scatter with jitter + mean line per LV)
# ---------------------------------------------------------------------------


def _plot_metric_lv(
    feature_df: pd.DataFrame, metric_col: str, y_label: str, title: str, output_path: Path
) -> None:
    controls = sorted(feature_df["control"].dropna().astype(str).unique().tolist())
    lv_ids = sorted(feature_df["lv_id"].dropna().astype(str).unique().tolist())
    if not controls or not lv_ids:
        raise ValueError("LV feature summary must contain control and lv_id values")

    mean_df = _entity_mean(feature_df, metric_col, ["control", "b", "lv_id"])
    colors = _lv_color_map(lv_ids)
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
    _save_dual(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-type", required=True, choices=["year", "lv"],
        help="year (boxplots overlaying years) or lv (scatter per LV).",
    )
    parser.add_argument(
        "--analysis-dir", required=True,
        help="Directory containing feature_variance_summary.csv (and optionally _supported_only fallback for year).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = args.analysis_type
    analysis_dir = Path(args.analysis_dir)

    feature_path = analysis_dir / "feature_variance_summary.csv"
    if domain == "year" and not feature_path.exists():
        alt = analysis_dir / "feature_variance_summary_supported_only.csv"
        if alt.exists():
            feature_path = alt

    entity_col = "year" if domain == "year" else "lv_id"
    feature_df = _load_csv(
        feature_path,
        required_columns=["control", entity_col, "b", "diff_std", "diff_var"],
    )

    group_keys = ["control", "b", entity_col]
    mean_std_df = _entity_mean(feature_df, "diff_std", group_keys)
    mean_var_df = _entity_mean(feature_df, "diff_var", group_keys)
    elbow_df = pd.concat(
        [
            _compute_elbow(mean_std_df, "diff_std", increasing=False, group_keys=["control", entity_col], entity_col=entity_col),
            _compute_elbow(mean_var_df, "diff_var", increasing=False, group_keys=["control", entity_col], entity_col=entity_col),
        ],
        ignore_index=True,
    )
    if not elbow_df.empty:
        elbow_df.to_csv(analysis_dir / "elbow_summary.csv", index=False)

    if domain == "year":
        plot_fn = _plot_metric_year
        sd_name = "sd_boxplots_with_mean_trend_by_b.pdf"
        var_name = "variance_boxplots_with_mean_trend_by_b.pdf"
        title_label = "Year"
    else:
        plot_fn = _plot_metric_lv
        sd_name = "sd_points_with_mean_trend_by_b.pdf"
        var_name = "variance_points_with_mean_trend_by_b.pdf"
        title_label = "LV"

    plot_fn(
        feature_df=feature_df,
        metric_col="diff_std",
        y_label="SD(diff) across seeds",
        title=f"{title_label} null SD by B",
        output_path=analysis_dir / sd_name,
    )
    plot_fn(
        feature_df=feature_df,
        metric_col="diff_var",
        y_label="Variance(diff) across seeds",
        title=f"{title_label} null variance by B",
        output_path=analysis_dir / var_name,
    )

    if not elbow_df.empty:
        print(f"Saved summary: {analysis_dir / 'elbow_summary.csv'}")
    print(f"Saved plot: {analysis_dir / sd_name}")
    print(f"Saved plot: {(analysis_dir / sd_name).with_suffix('.png')}")
    print(f"Saved plot: {analysis_dir / var_name}")
    print(f"Saved plot: {(analysis_dir / var_name).with_suffix('.png')}")


if __name__ == "__main__":
    main()
