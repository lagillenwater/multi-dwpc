#!/usr/bin/env python3
"""Year metapath rank-stability analysis over explicit replicate summary artifacts."""

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

YEAR_COLORS = {
    "2016": "#1f77b4",
    "2024": "#ff7f0e",
}


def _save_dual(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def _default_results_dir(score_source: str) -> str:
    if str(score_source) == "api":
        return "output/dwpc_com/all_GO_positive_growth/results"
    return "output/dwpc_direct/all_GO_positive_growth/results"

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.replicate_analysis import rank_features, summarize_rank_stability  # noqa: E402
from src.result_normalization import (  # noqa: E402
    load_normalized_year_results,
    summarize_normalized_year_results,
)
from src.year_replicate_analysis import build_b_seed_runs, load_summary_bank  # noqa: E402


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_top_k_values(arg: str) -> list[int | str]:
    values: list[int | str] = []
    for tok in str(arg).split(","):
        token = tok.strip()
        if not token:
            continue
        values.append(int(token))
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
    def _sort_key(label: str) -> tuple[int, int | str]:
        if label == "all":
            return (1, label)
        return (0, int(label))
    return sorted(set(labels), key=_sort_key)


def _plot_overall(overall_df: pd.DataFrame, y_col: str, y_label: str, title: str, out_path: Path) -> None:
    years = sorted(overall_df["year"].dropna().astype(int).unique().tolist())
    b_values = sorted(overall_df["b"].dropna().astype(int).unique().tolist())
    controls = sorted(overall_df["control"].dropna().astype(str).unique().tolist())
    if not years or not controls or not b_values:
        return

    year_colors = {year: color for year, color in zip(years, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])}

    fig, axes = plt.subplots(1, len(controls), figsize=(6.6 * len(controls), 4.8), sharey=True)
    if len(controls) == 1:
        axes = [axes]
    for ax, control in zip(axes, controls):
        subset = overall_df[overall_df["control"].astype(str) == control].copy()
        for year in years:
            year_df = subset[subset["year"].astype(int) == int(year)].copy().sort_values("b")
            if year_df.empty:
                continue
            ax.plot(
                year_df["b"].astype(int),
                year_df[y_col].astype(float),
                marker="o",
                linewidth=2.2,
                color=year_colors.get(year, "#333333"),
                label=str(year),
            )
        ax.set_xlabel("B")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(title="Year")
    fig.suptitle(title)
    fig.tight_layout()
    _save_dual(fig, out_path)
    plt.close(fig)


def _plot_overlap_and_rank_points(entity_df: pd.DataFrame, output_path: Path) -> None:
    entity_df = entity_df.copy()
    entity_df["year"] = entity_df["year"].astype(int)
    controls = sorted(entity_df["control"].dropna().astype(str).unique().tolist())
    years = sorted(entity_df["year"].dropna().astype(int).unique().tolist())
    top_k_labels = _top_k_labels_from_columns(entity_df)
    if not controls or not years:
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
        subset = entity_df[["control", "b", "year", "go_id", metric_col]].copy()
        subset = subset.rename(columns={metric_col: "metric_value"})
        subset["metric_key"] = metric_key
        subset["metric_label"] = metric_label
        plot_rows.append(subset)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    mean_df = (
        plot_df.groupby(["metric_key", "control", "b", "year"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "mean_metric_value"})
    )

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
            for year in years:
                color = YEAR_COLORS.get(str(year), "#333333")
                points = control_points[control_points["year"].astype(int) == int(year)].copy()
                if points.empty:
                    continue
                b_values = points["b"].astype(float).to_numpy()
                jitter = rng.uniform(-0.14, 0.14, size=len(points))
                ax.scatter(
                    b_values + jitter,
                    points["metric_value"].astype(float),
                    s=28,
                    alpha=0.30,
                    color=color,
                    edgecolors="none",
                )
                line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
                ax.plot(
                    line["b"].astype(float),
                    line["mean_metric_value"].astype(float),
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                    color=color,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-source", default="direct", choices=["direct", "api"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--workspace-dir", default=None)
    parser.add_argument("--summaries-dir", default=None)
    parser.add_argument("--output-dir", default="output/year_rank_stability_exp")
    parser.add_argument("--b-values", default="1,2,5,10,20")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--top-k-metapaths", default="5,10")
    parser.add_argument("--rbo-p", default="0.9", help="RBO persistence in (0,1), or 'none' to disable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else Path(_default_results_dir(args.score_source))
    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else results_dir.parent
    summary_dir_candidate = workspace_dir / "replicate_summaries"
    if args.summaries_dir:
        summary_source = Path(args.summaries_dir)
        summary_df = load_summary_bank(summary_source)
    elif summary_dir_candidate.exists():
        summary_source = summary_dir_candidate
        summary_df = load_summary_bank(summary_source)
    else:
        summary_source = workspace_dir
        if (summary_source / "replicate_manifest.csv").exists() or summary_source.is_file():
            summary_df = load_summary_bank(summary_source)
        else:
            normalized_df = load_normalized_year_results(
                results_dir,
                score_source=str(args.score_source),
                data_dir=Path(args.data_dir),
            )
            summary_df = summarize_normalized_year_results(normalized_df)
            if summary_df.empty:
                raise ValueError(
                    f"No summary rows produced from normalized {args.score_source} results under {results_dir}"
                )
    exp_dir = Path(args.output_dir) / "year_rank_stability_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    runs_df = build_b_seed_runs(summary_df, _parse_int_list(args.b_values), _parse_int_list(args.seeds))
    rank_df = rank_features(
        runs_df,
        rank_group_keys=["year", "control", "b", "seed", "go_id"],
        feature_col="metapath",
        score_col="diff",
        rank_col="metapath_rank",
    )
    pairwise_df, go_summary_df, overall_df = summarize_rank_stability(
        rank_df,
        outer_keys=["year", "control", "b", "go_id"],
        replicate_col="seed",
        feature_col="metapath",
        rank_col="metapath_rank",
        top_k=_parse_top_k_values(args.top_k_metapaths),
        rbo_p=_parse_optional_rbo_p(args.rbo_p),
    )
    if not overall_df.empty:
        overall_df = overall_df.rename(columns={"n_entities": "n_go_terms"})

    runs_df.to_csv(exp_dir / "all_runs_long.csv", index=False)
    rank_df.to_csv(exp_dir / "metapath_rank_table.csv", index=False)
    pairwise_df.to_csv(exp_dir / "pairwise_metrics.csv", index=False)
    go_summary_df.to_csv(exp_dir / "go_term_stability_summary.csv", index=False)
    overall_df.to_csv(exp_dir / "overall_stability_summary.csv", index=False)

    _plot_overall(
        overall_df,
        y_col="mean_spearman_rho",
        y_label="Mean Spearman rho across seeds",
        title="Year metapath rank stability by B",
        out_path=exp_dir / "spearman_overall_by_group.pdf",
    )
    if "mean_rbo" in overall_df.columns:
        _plot_overall(
            overall_df,
            y_col="mean_rbo",
            y_label="Mean RBO across seeds",
            title="Year metapath RBO stability by B",
            out_path=exp_dir / "rbo_overall_by_group.pdf",
        )
    _plot_overlap_and_rank_points(go_summary_df, exp_dir / "topk_jaccard_overall_by_group.pdf")

    print(f"Saved rank table: {exp_dir / 'metapath_rank_table.csv'}")
    print(f"Saved pairwise metrics: {exp_dir / 'pairwise_metrics.csv'}")
    print(f"Saved overall summary: {exp_dir / 'overall_stability_summary.csv'}")


if __name__ == "__main__":
    main()
