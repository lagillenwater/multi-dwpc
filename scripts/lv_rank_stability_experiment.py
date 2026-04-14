#!/usr/bin/env python3
"""LV metapath rank-stability analysis over explicit replicate summary artifacts."""

from __future__ import annotations

import argparse
import os
import sys
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

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_replicate_analysis import FEATURE_KEYS, build_b_seed_runs, load_summary_bank  # noqa: E402
from src.replicate_analysis import rank_features, summarize_rank_stability  # noqa: E402


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_top_k_values(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one top-k value")
    return values


def _parse_optional_rbo_p(arg: str) -> float | None:
    text = str(arg).strip().lower()
    if text in {"none", "off", "false", "na", "nan"}:
        return None
    value = float(arg)
    if not (0.0 < value < 1.0):
        raise ValueError("--rbo-p must be in (0, 1), or use 'none' to disable RBO")
    return value


def _top_k_labels_from_columns(df: pd.DataFrame) -> list[str]:
    labels = []
    for col in df.columns:
        prefix = "mean_topk_jaccard_"
        if col.startswith(prefix):
            labels.append(col[len(prefix):])
    return sorted(set(labels), key=int)


def _color_map(ids: list[str]) -> dict[str, str]:
    fallback = plt.get_cmap("tab10")
    colors = {}
    for idx, entity_id in enumerate(ids):
        colors[entity_id] = LV_COLORS.get(entity_id, fallback(idx % 10))
    return colors


def _plot_overall(overall_df: pd.DataFrame, y_col: str, y_label: str, title: str, out_path: Path) -> None:
    if overall_df.empty:
        return
    controls = sorted(overall_df["control"].astype(str).unique().tolist())
    colors = {"permuted": "#1f77b4", "random": "#d62728"}
    fig, axes = plt.subplots(1, len(controls), figsize=(6.5 * len(controls), 4.8), sharey=True)
    if len(controls) == 1:
        axes = [axes]
    for ax, control in zip(axes, controls):
        subset = overall_df[overall_df["control"].astype(str) == control].copy().sort_values("b")
        ax.plot(subset["b"].astype(int), subset[y_col].astype(float), marker="o", color=colors.get(control, "#333333"))
        ax.set_xlabel("B")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overlap_and_rank_points(entity_df: pd.DataFrame, output_path: Path) -> None:
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    lv_ids = sorted(entity_df["lv_id"].dropna().astype(str).unique().tolist())
    top_k_labels = _top_k_labels_from_columns(entity_df)
    if not controls or not lv_ids:
        return

    metric_specs: list[tuple[str, str, str]] = []
    if "5" in top_k_labels:
        metric_specs.append(("top5", "mean_topk_jaccard_5", "Mean top-5 Jaccard across seeds"))
    if "10" in top_k_labels:
        metric_specs.append(("top10", "mean_topk_jaccard_10", "Mean top-10 Jaccard across seeds"))
    if "mean_rbo" in entity_df.columns:
        metric_specs.append(("rbo", "mean_rbo", "Mean RBO across seeds"))
    metric_specs.append(("all", "mean_spearman_rho", "Mean Spearman rho across seeds"))

    plot_rows = []
    for metric_key, metric_col, metric_label in metric_specs:
        subset = entity_df[["control", "b", "lv_id", "target_set_id", metric_col]].copy()
        subset = subset.rename(columns={metric_col: "metric_value"})
        subset["metric_key"] = metric_key
        subset["metric_label"] = metric_label
        plot_rows.append(subset)
    plot_df = pd.concat(plot_rows, ignore_index=True)
    mean_df = (
        plot_df.groupby(["metric_key", "control", "b", "lv_id"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "mean_metric_value"})
    )

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
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--analysis-output-dir", default=None)
    parser.add_argument("--b-values", default="2,4,6,8,10,20")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--top-k-metapaths", default="5,10")
    parser.add_argument("--rbo-p", default="0.9", help="RBO persistence in (0,1), or 'none' to disable")
    parser.add_argument(
        "--rank-metric",
        default="effect_size_d",
        choices=["diff", "effect_size_d"],
        help="Metric to use for ranking metapaths (default: effect_size_d)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    exp_root = Path(args.analysis_output_dir) if args.analysis_output_dir else output_dir / "lv_rank_stability_experiment"
    exp_root.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_bank(output_dir)
    runs_df = build_b_seed_runs(summary_df, _parse_int_list(args.b_values), _parse_int_list(args.seeds))

    # Select ranking metric (effect_size_d or diff)
    rank_metric = str(args.rank_metric)
    if rank_metric not in runs_df.columns:
        print(f"Warning: {rank_metric} not in runs_df, falling back to 'diff'")
        rank_metric = "diff"

    rank_df = rank_features(
        runs_df,
        rank_group_keys=["control", "b", "seed", "lv_id", "target_set_id"],
        feature_col="metapath",
        score_col=rank_metric,
        rank_col="metapath_rank",
    )
    pairwise_df, entity_df, overall_df = summarize_rank_stability(
        rank_df,
        outer_keys=["control", "b", "lv_id", "target_set_id"],
        replicate_col="seed",
        feature_col="metapath",
        rank_col="metapath_rank",
        top_k=_parse_top_k_values(args.top_k_metapaths),
        rbo_p=_parse_optional_rbo_p(args.rbo_p),
    )
    if not entity_df.empty:
        entity_df = entity_df.rename(columns={"n_entities": "n_lv_target_sets"})
    if not overall_df.empty:
        overall_df = overall_df.rename(columns={"n_entities": "n_lv_target_sets"})

    runs_df.to_csv(exp_root / "all_runs_long.csv", index=False)
    rank_df.to_csv(exp_root / "metapath_rank_table.csv", index=False)
    pairwise_df.to_csv(exp_root / "pairwise_metrics.csv", index=False)
    entity_df.to_csv(exp_root / "lv_stability_summary.csv", index=False)
    overall_df.to_csv(exp_root / "overall_stability_summary.csv", index=False)

    _plot_overall(
        overall_df,
        y_col="mean_spearman_rho",
        y_label="Mean Spearman rho across seeds",
        title="LV metapath rank stability by B",
        out_path=exp_root / "spearman_overall_by_group.png",
    )
    if "mean_rbo" in overall_df.columns:
        _plot_overall(
            overall_df,
            y_col="mean_rbo",
            y_label="Mean RBO across seeds",
            title="LV metapath RBO stability by B",
            out_path=exp_root / "rbo_overall_by_group.png",
        )
    _plot_overlap_and_rank_points(entity_df, exp_root / "topk_jaccard_overall_by_group.png")

    print(f"Saved rank table: {exp_root / 'metapath_rank_table.csv'}")
    print(f"Saved pairwise metrics: {exp_root / 'pairwise_metrics.csv'}")
    print(f"Saved overall summary: {exp_root / 'overall_stability_summary.csv'}")


if __name__ == "__main__":
    main()
