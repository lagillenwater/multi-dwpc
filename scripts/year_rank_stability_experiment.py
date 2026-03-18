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

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.replicate_analysis import rank_features, summarize_rank_stability  # noqa: E402
from src.year_replicate_analysis import build_b_seed_runs, load_summary_bank  # noqa: E402


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="output/dwpc_direct/all_GO_positive_growth/results")
    parser.add_argument("--summaries-dir", default=None)
    parser.add_argument("--output-dir", default="output/year_rank_stability_exp")
    parser.add_argument("--b-values", default="1,2,5,10,20")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--top-k-metapaths", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries_dir = Path(args.summaries_dir) if args.summaries_dir else Path(args.results_dir).parent / "replicate_summaries"
    exp_dir = Path(args.output_dir) / "year_rank_stability_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_bank(summaries_dir)
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
        top_k=args.top_k_metapaths,
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
        out_path=exp_dir / "spearman_overall_by_group.png",
    )
    _plot_overall(
        overall_df,
        y_col="mean_topk_jaccard",
        y_label=f"Mean top-{args.top_k_metapaths} Jaccard across seeds",
        title="Year top-k overlap stability by B",
        out_path=exp_dir / "topk_jaccard_overall_by_group.png",
    )

    print(f"Saved rank table: {exp_dir / 'metapath_rank_table.csv'}")
    print(f"Saved pairwise metrics: {exp_dir / 'pairwise_metrics.csv'}")
    print(f"Saved overall summary: {exp_dir / 'overall_stability_summary.csv'}")


if __name__ == "__main__":
    main()
