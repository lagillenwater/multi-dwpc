#!/usr/bin/env python3
"""Plot rank-stability summaries from existing aggregate CSV outputs.

Replaces `plot_year_rank_stability_results.py` and
`plot_lv_rank_stability_results.py`. Dispatches on `--analysis-type {year,lv}`.

Year: boxplots with year offsets within B; mean line per year.
LV:   scatter with jitter per LV; mean line per LV.
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


def _top_k_labels_from_columns(df: pd.DataFrame) -> list[str]:
    """Sort top-k labels numerically, with 'all' (if present) placed at the end."""
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


def _compute_elbow(
    entity_df: pd.DataFrame, metric_col: str, increasing: bool,
    group_keys: list[str], entity_col: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for key_tuple, group in entity_df.groupby(group_keys, sort=True):
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        entity_value = key_tuple[group_keys.index(entity_col)]
        control = key_tuple[group_keys.index("control")]
        curve = (
            group.groupby("b", as_index=False)[metric_col]
            .mean()
            .sort_values("b")
            .reset_index(drop=True)
        )
        if len(curve) < 3:
            continue
        x = curve["b"].astype(float).to_numpy()
        y = curve[metric_col].astype(float).to_numpy()
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
            "elbow_mean_value": float(curve[metric_col].iloc[idx]),
            "elbow_distance": float(distance[idx]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Year-specific plots (boxplots)
# ---------------------------------------------------------------------------


def _mean_by_year(entity_df: pd.DataFrame) -> pd.DataFrame:
    return (
        entity_df.groupby(["control", "b", "year"], as_index=False)["mean_spearman_rho"]
        .mean()
        .rename(columns={"mean_spearman_rho": "mean_rho_by_year"})
        .sort_values(["control", "year", "b"])
    )


def _mean_metric_by_year(plot_df: pd.DataFrame) -> pd.DataFrame:
    return (
        plot_df.groupby(["metric_key", "control", "b", "year"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "mean_metric_value"})
    )


def _plot_rho_year(entity_df: pd.DataFrame, output_path: Path) -> None:
    entity_df = entity_df.copy()
    entity_df["year"] = entity_df["year"].astype(int)
    entity_df["b_plot"] = entity_df["b"].astype(float).round().astype(int)

    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(entity_df["year"].dropna().astype(int).unique().tolist())
    if not controls or not years:
        raise ValueError("Year stability summary must contain control and year values")

    mean_df = _mean_by_year(entity_df).copy()
    mean_df["b_plot"] = mean_df["b"].astype(float).round().astype(int)
    fig, axes = plt.subplots(1, len(controls), figsize=(7.0 * len(controls), 5.2), sharey=True)
    if len(controls) == 1:
        axes = [axes]

    for ax, control in zip(axes, controls):
        control_points = entity_df[entity_df["control"].astype(str) == control].copy()
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
                points[points["b"].astype(int) == int(b)]["mean_spearman_rho"].astype(float).to_numpy()
                for b in sorted(points["b"].astype(int).unique().tolist())
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
            line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
            ax.plot(
                line["b"].astype(float),
                line["mean_rho_by_year"].astype(float),
                marker="o",
                linewidth=2.2,
                markersize=1,
                color=color,
                alpha=0.55,
                label=str(year),
            )
        # Per-subplot axis cosmetics — INSIDE the loop (lv had this right; year had it
        # outside the loop, so only the last subplot got labeled — fixed in unification).
        ax.set_xlabel("Null replicate count (B)")
        ax.set_xticks(all_b)
        ax.set_xticklabels([str(b) for b in all_b])
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean Spearman rho across seed pairs")
    axes[-1].legend(title="Year", loc="best")
    fig.suptitle("Year rank-stability rho by B")
    fig.tight_layout()
    _save_dual(fig, output_path)
    plt.close(fig)


def _plot_overlap_and_rank_year(entity_df: pd.DataFrame, output_path: Path) -> None:
    entity_df = entity_df.copy()
    entity_df["year"] = entity_df["year"].astype(int)
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(entity_df["year"].dropna().astype(int).unique().tolist())
    top_k_labels = _top_k_labels_from_columns(entity_df)
    if not controls or not years:
        raise ValueError("Year stability summary must contain control and year values")

    metric_specs: list[tuple[str, str, str]] = []
    if "5" in top_k_labels:
        metric_specs.append(("top5", "mean_topk_jaccard_5", "Mean top-5 Jaccard across seeds"))
    if "10" in top_k_labels:
        metric_specs.append(("top10", "mean_topk_jaccard_10", "Mean top-10 Jaccard across seeds"))
    if "all" in top_k_labels:
        metric_specs.append(("topall", "mean_topk_jaccard_all", "Mean top-all Jaccard across seeds"))
    metric_specs.append(("rho", "mean_spearman_rho", "Mean Spearman rho across seeds"))

    plot_rows = []
    for metric_key, metric_col, metric_label in metric_specs:
        subset = entity_df[["control", "b", "year", "go_id", metric_col]].copy()
        subset = subset.rename(columns={metric_col: "metric_value"})
        subset["metric_key"] = metric_key
        subset["metric_label"] = metric_label
        plot_rows.append(subset)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    mean_df = _mean_metric_by_year(plot_df)
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
            width = 0.32 if len(years) > 1 else 0.55
            offsets = {int(y): ((idx - (len(years) - 1) / 2.0) * width) for idx, y in enumerate(years)}
            for year in years:
                color = YEAR_COLORS.get(str(year), "#333333")
                points = control_points[control_points["year"].astype(int) == int(year)].copy()
                if points.empty:
                    continue
                positions = [
                    float(b) + float(offsets[int(year)])
                    for b in sorted(points["b"].astype(int).unique().tolist())
                ]
                box_data = [
                    points[points["b"].astype(int) == int(b)]["metric_value"].astype(float).to_numpy()
                    for b in sorted(points["b"].astype(int).unique().tolist())
                ]
                ax.boxplot(
                    box_data,
                    positions=positions,
                    widths=width * 0.85,
                    patch_artist=True,
                    boxprops={"facecolor": color, "alpha": 0.22, "edgecolor": color},
                    medianprops={"color": color, "linewidth": 1.4},
                    whiskerprops={"color": color, "alpha": 0.5},
                    capprops={"color": color, "alpha": 0.5},
                    flierprops={
                        "marker": "o", "markersize": 2.5, "markerfacecolor": color,
                        "markeredgecolor": "none", "alpha": 0.20,
                    },
                )
                line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
                ax.plot(
                    line["b"].astype(float),
                    line["mean_metric_value"].astype(float),
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                    color=color,
                    alpha=0.55,
                    label=str(year),
                )
            if row_idx == 0:
                ax.set_title(control)
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(metric_specs) - 1:
                ax.set_xlabel("B")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
    axes[0, -1].legend(title="Year", loc="best")
    fig.suptitle("Year overlap and rank stability by B")
    fig.tight_layout()
    _save_dual(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LV-specific plots (scatter + jitter)
# ---------------------------------------------------------------------------


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


def _plot_rho_lv(entity_df: pd.DataFrame, output_path: Path) -> None:
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    lv_ids = sorted(entity_df["lv_id"].dropna().astype(str).unique().tolist())
    if not controls or not lv_ids:
        raise ValueError("LV stability summary must contain control and lv_id values")

    mean_df = _mean_by_lv(entity_df)
    colors = _lv_color_map(lv_ids)
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
    _save_dual(fig, output_path)
    plt.close(fig)


def _plot_overlap_and_rank_lv(entity_df: pd.DataFrame, output_path: Path) -> None:
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
        subset = entity_df[["control", "b", "lv_id", metric_col]].copy()
        subset = subset.rename(columns={metric_col: "metric_value"})
        subset["metric_key"] = metric_key
        subset["metric_label"] = metric_label
        plot_rows.append(subset)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    mean_df = _mean_metric_by_lv(plot_df)
    colors = _lv_color_map(lv_ids)
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
    _save_dual(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Domain-aware entity-summary loader (lv has a legacy fallback path)
# ---------------------------------------------------------------------------


def _load_entity_df(analysis_dir: Path, domain: str) -> pd.DataFrame:
    if domain == "year":
        entity_path = analysis_dir / "go_term_stability_summary.csv"
        if not entity_path.exists():
            alt = analysis_dir / "go_term_stability_summary_supported_only.csv"
            if alt.exists():
                entity_path = alt
        return _load_csv(
            entity_path,
            required_columns=["control", "year", "b", "go_id", "mean_spearman_rho"],
        )

    summary_path = analysis_dir / "lv_stability_summary.csv"
    if summary_path.exists():
        return _load_csv(
            summary_path,
            required_columns=["control", "b", "lv_id", "mean_spearman_rho"],
        )
    legacy_path = analysis_dir / "metapath_pairwise_metrics.csv"
    legacy_df = _load_csv(legacy_path, required_columns=["b", "lv_id", "spearman_rho"])
    legacy_df = (
        legacy_df.groupby(["b", "lv_id"], as_index=False)
        .agg(
            n_pairs=("spearman_rho", "size"),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
        )
        .sort_values(["lv_id", "b"])
        .reset_index(drop=True)
    )
    legacy_df.insert(0, "control", "combined")
    return legacy_df


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
        help="Directory containing {go_term,lv}_stability_summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain = args.analysis_type
    analysis_dir = Path(args.analysis_dir)
    entity_col = "year" if domain == "year" else "lv_id"

    entity_df = _load_entity_df(analysis_dir, domain)

    elbow_frames = [
        _compute_elbow(entity_df, "mean_spearman_rho", increasing=True,
                       group_keys=["control", entity_col], entity_col=entity_col)
    ]
    for col in entity_df.columns:
        if col.startswith("mean_topk_jaccard_"):
            elbow_frames.append(
                _compute_elbow(entity_df, col, increasing=True,
                               group_keys=["control", entity_col], entity_col=entity_col)
            )
    elbow_df = pd.concat(elbow_frames, ignore_index=True)
    if not elbow_df.empty:
        elbow_df.to_csv(analysis_dir / "elbow_summary.csv", index=False)

    if domain == "year":
        rho_path = analysis_dir / "rho_boxplots_with_mean_trend_by_b.pdf"
        topk_path = analysis_dir / "topk_jaccard_boxplots_with_mean_trend_by_b.pdf"
        _plot_rho_year(entity_df, rho_path)
        _plot_overlap_and_rank_year(entity_df, topk_path)
    else:
        rho_path = analysis_dir / "rho_points_with_mean_trend_by_b.pdf"
        topk_path = analysis_dir / "topk_jaccard_points_with_mean_trend_by_b.pdf"
        _plot_rho_lv(entity_df, rho_path)
        _plot_overlap_and_rank_lv(entity_df, topk_path)

    print(f"Saved plot: {rho_path}")
    print(f"Saved plot: {rho_path.with_suffix('.png')}")
    print(f"Saved plot: {topk_path}")
    print(f"Saved plot: {topk_path.with_suffix('.png')}")
    if not elbow_df.empty:
        print(f"Saved summary: {analysis_dir / 'elbow_summary.csv'}")


if __name__ == "__main__":
    main()
