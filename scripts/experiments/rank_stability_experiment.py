#!/usr/bin/env python3
"""Metapath rank-stability analysis over explicit replicate summary artifacts.

Replaces the prior `year_rank_stability_experiment.py` and
`lv_rank_stability_experiment.py`. Dispatches on `--analysis-type {year,lv}`.

Year mode: ranks metapaths per (year, control, b, seed, go_id), summarizes
top-k Jaccard / Spearman rho stability across seeds, plots violin overlays
per year.

LV mode: ranks metapaths per (control, b, seed, lv_id), summarizes the same
metrics, plots per-LV scatter + mean line.

Outputs (under `<output-dir>/{year,lv}_rank_stability_experiment/`):
    all_runs_long.csv
    metapath_rank_table.csv
    pairwise_metrics.csv
    {go_term,lv}_stability_summary.csv      # entity-level
    overall_stability_summary.csv
    spearman_overall_by_group.pdf (+ .png)
    topk_jaccard_overall_by_group.pdf (+ .png)
"""

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

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.replicate_analysis import (  # noqa: E402
    build_b_seed_runs,
    load_domain_summary_bank,
    rank_features,
    summarize_rank_stability,
)
from src.result_normalization import (  # noqa: E402
    load_normalized_year_results,
    summarize_normalized_year_results,
)


YEAR_COLORS = {
    "2016": "#1f77b4",
    "2024": "#ff7f0e",
}

LV_COLORS = {
    "LV246": "#1f77b4",
    "LV57": "#ff7f0e",
    "LV603": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_dual(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def _default_results_dir(score_source: str) -> str:
    if str(score_source) == "api":
        return "output/dwpc_com/all_GO_positive_growth/results"
    return "output/dwpc_direct/all_GO_positive_growth/results"


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


def _lv_color_map(lv_ids: list[str]) -> dict:
    fallback = plt.get_cmap("tab10")
    colors = {}
    for idx, lv_id in enumerate(lv_ids):
        colors[lv_id] = LV_COLORS.get(lv_id, fallback(idx % 10))
    return colors


# ---------------------------------------------------------------------------
# Year-specific plots
# ---------------------------------------------------------------------------


def _plot_overall_year(
    overall_df: pd.DataFrame, y_col: str, y_label: str, title: str, output_path: Path
) -> None:
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
    _save_dual(fig, output_path)
    plt.close(fig)


def _plot_overlap_and_rank_points_year(entity_df: pd.DataFrame, output_path: Path) -> None:
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

    width = 0.32 if len(years) > 1 else 0.55
    offsets = {int(y): ((idx - (len(years) - 1) / 2.0) * width) for idx, y in enumerate(years)}
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
                b_vals_sorted = sorted(points["b"].astype(int).unique().tolist())
                dists = [
                    points[points["b"].astype(int) == int(b)]["metric_value"].astype(float).to_numpy()
                    for b in b_vals_sorted
                ]
                positions = [float(b) + float(offsets[int(year)]) for b in b_vals_sorted]
                parts = ax.violinplot(
                    dists,
                    positions=positions,
                    widths=width * 0.9,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.25)
                line = control_mean[control_mean["year"].astype(int) == int(year)].copy().sort_values("b")
                ax.plot(
                    line["b"].astype(float) + float(offsets[int(year)]),
                    line["mean_metric_value"].astype(float),
                    linewidth=2.2,
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


# ---------------------------------------------------------------------------
# LV-specific plots
# ---------------------------------------------------------------------------


def _plot_overall_lv(
    overall_df: pd.DataFrame, y_col: str, y_label: str, title: str, output_path: Path
) -> None:
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
    _save_dual(fig, output_path)
    plt.close(fig)


def _plot_overlap_and_rank_points_lv(entity_df: pd.DataFrame, output_path: Path) -> None:
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
    metric_specs.append(("all", "mean_spearman_rho", "Mean Spearman rho across seeds"))

    plot_rows = []
    for metric_key, metric_col, metric_label in metric_specs:
        subset = entity_df[["control", "b", "lv_id", metric_col]].copy()
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
# Year-specific input discovery
# ---------------------------------------------------------------------------


def _load_year_summary(args: argparse.Namespace) -> pd.DataFrame:
    """Year-side fallback chain: manifest -> summaries dir -> normalize raw results."""
    results_dir = (
        Path(args.results_dir) if args.results_dir
        else Path(_default_results_dir(args.score_source))
    )
    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else results_dir.parent
    summary_dir_candidate = workspace_dir / "replicate_summaries"

    if args.summaries_dir:
        return load_domain_summary_bank(Path(args.summaries_dir), domain="year")
    if summary_dir_candidate.exists():
        return load_domain_summary_bank(summary_dir_candidate, domain="year")

    summary_source = workspace_dir
    if (summary_source / "replicate_manifest.csv").exists() or summary_source.is_file():
        return load_domain_summary_bank(summary_source, domain="year")

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
    return summary_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis-type", required=True, choices=["year", "lv"],
        help="year (cross-year cohort rank stability) or lv (per-LV rank stability).",
    )
    parser.add_argument("--output-dir", required=True,
                        help="Root output dir; experiment subdir is `<analysis>_rank_stability_experiment`.")
    parser.add_argument("--analysis-output-dir", default=None,
                        help="Override the experiment subdir entirely.")
    parser.add_argument("--b-values", default="2,4,6,8,10,20",
                        help="Comma-separated B values. Default: 2,4,6,8,10,20.")
    parser.add_argument("--seeds", default="11,22,33,44,55",
                        help="Comma-separated seeds. Default: 11,22,33,44,55.")
    parser.add_argument("--top-k-metapaths", default="5,10",
                        help="Comma-separated top-k for Jaccard overlap metrics. Default: 5,10.")
    parser.add_argument("--rbo-p", default="0.9",
                        help="RBO persistence in (0, 1), or 'none' to disable.")
    parser.add_argument(
        "--rank-metric", default="effect_size_z", choices=["diff", "effect_size_z"],
        help="Metric to use for ranking metapaths. Default: effect_size_z.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip rank/stability compute; read summaries from the experiment dir and re-render plots only.",
    )

    # Year-only flags (input discovery / raw-results fallback)
    parser.add_argument("--score-source", default="direct", choices=["direct", "api"],
                        help="[year only] direct (default) or api scoring source for the raw-results fallback.")
    parser.add_argument("--results-dir", default=None,
                        help="[year only] Raw results directory; default depends on --score-source.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"),
                        help="[year only] Data dir for raw-results normalization.")
    parser.add_argument("--workspace-dir", default=None,
                        help="[year only] Workspace dir containing `replicate_manifest.csv`. Default: parent of --results-dir.")
    parser.add_argument("--summaries-dir", default=None,
                        help="[year only] Direct path to a `replicate_summaries/` directory. Skips fallback chain.")

    return parser.parse_args()


def _resolve_exp_dir(args: argparse.Namespace) -> Path:
    if args.analysis_output_dir:
        return Path(args.analysis_output_dir)
    return Path(args.output_dir) / f"{args.analysis_type}_rank_stability_experiment"


def _load_summary(args: argparse.Namespace) -> pd.DataFrame:
    if args.analysis_type == "year":
        return _load_year_summary(args)
    return load_domain_summary_bank(Path(args.output_dir), domain="lv")


def main() -> None:
    args = parse_args()
    domain = args.analysis_type
    exp_dir = _resolve_exp_dir(args)
    entity_summary_filename = (
        "go_term_stability_summary.csv" if domain == "year" else "lv_stability_summary.csv"
    )

    if args.plot_only:
        overall_path = exp_dir / "overall_stability_summary.csv"
        entity_path = exp_dir / entity_summary_filename
        for p in (overall_path, entity_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"--plot-only requires {p} to exist. Run without --plot-only first."
                )
        overall_df = pd.read_csv(overall_path)
        entity_df = pd.read_csv(entity_path)
        print(f"--plot-only: loaded {overall_path} ({len(overall_df)} rows) and {entity_path} ({len(entity_df)} rows)")
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)
        summary_df = _load_summary(args)
        runs_df = build_b_seed_runs(
            summary_df,
            _parse_int_list(args.b_values),
            _parse_int_list(args.seeds),
            domain=domain,
        )

        rank_metric = str(args.rank_metric)
        if rank_metric not in runs_df.columns:
            print(f"Warning: {rank_metric} not in runs_df, falling back to 'diff'")
            rank_metric = "diff"

        if domain == "year":
            rank_group_keys = ["year", "control", "b", "seed", "go_id"]
            outer_keys = ["year", "control", "b", "go_id"]
            entity_rename = {"n_entities": "n_go_terms"}
        else:
            rank_group_keys = ["control", "b", "seed", "lv_id"]
            outer_keys = ["control", "b", "lv_id"]
            entity_rename = {"n_entities": "n_lvs"}

        rank_df = rank_features(
            runs_df,
            rank_group_keys=rank_group_keys,
            feature_col="metapath",
            score_col=rank_metric,
            rank_col="metapath_rank",
        )
        pairwise_df, entity_df, overall_df = summarize_rank_stability(
            rank_df,
            outer_keys=outer_keys,
            replicate_col="seed",
            feature_col="metapath",
            rank_col="metapath_rank",
            top_k=_parse_top_k_values(args.top_k_metapaths),
            rbo_p=_parse_optional_rbo_p(args.rbo_p),
        )
        if not entity_df.empty:
            entity_df = entity_df.rename(columns=entity_rename)
        if not overall_df.empty:
            overall_df = overall_df.rename(columns=entity_rename)

        runs_df.to_csv(exp_dir / "all_runs_long.csv", index=False)
        rank_df.to_csv(exp_dir / "metapath_rank_table.csv", index=False)
        pairwise_df.to_csv(exp_dir / "pairwise_metrics.csv", index=False)
        entity_df.to_csv(exp_dir / entity_summary_filename, index=False)
        overall_df.to_csv(exp_dir / "overall_stability_summary.csv", index=False)

    plot_overall_fn = _plot_overall_year if domain == "year" else _plot_overall_lv
    plot_points_fn = (
        _plot_overlap_and_rank_points_year if domain == "year" else _plot_overlap_and_rank_points_lv
    )
    title_label = "Year" if domain == "year" else "LV"

    plot_overall_fn(
        overall_df,
        y_col="mean_spearman_rho",
        y_label="Mean Spearman rho across seeds",
        title=f"{title_label} metapath rank stability by B",
        output_path=exp_dir / "spearman_overall_by_group.pdf",
    )
    plot_points_fn(entity_df, exp_dir / "topk_jaccard_overall_by_group.pdf")

    if not args.plot_only:
        print(f"Saved rank table: {exp_dir / 'metapath_rank_table.csv'}")
        print(f"Saved pairwise metrics: {exp_dir / 'pairwise_metrics.csv'}")
        print(f"Saved overall summary: {exp_dir / 'overall_stability_summary.csv'}")


if __name__ == "__main__":
    main()
