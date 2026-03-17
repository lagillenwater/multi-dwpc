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
from src.replicate_analysis import build_diff_runs, rank_features, summarize_rank_stability  # noqa: E402


JOIN_KEYS = ["year", "go_id", "metapath"]


def _load_summary_bank(summaries_dir: Path) -> pd.DataFrame:
    files = sorted(Path(summaries_dir).glob("summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No year replicate summaries found under {summaries_dir}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _plot_overall(overall_df: pd.DataFrame, y_col: str, y_label: str, title: str, out_path: Path) -> None:
    years = sorted(overall_df["year"].dropna().astype(int).unique().tolist())
    controls = sorted(overall_df["control"].dropna().astype(str).unique().tolist())
    if not years or not controls:
        return

    width = 0.8 / max(len(controls), 1)
    x = np.arange(len(years))
    colors = {"permuted": "#1f77b4", "random": "#d62728"}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, control in enumerate(controls):
        vals = []
        for year in years:
            sub = overall_df[
                (overall_df["year"].astype(int) == int(year))
                & (overall_df["control"].astype(str) == control)
            ]
            vals.append(float(sub[y_col].iloc[0]) if not sub.empty else np.nan)
        ax.bar(
            x + (i - (len(controls) - 1) / 2) * width,
            vals,
            width=width,
            label=control,
            color=colors.get(control),
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Control")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="output/dwpc_direct/all_GO_positive_growth/results")
    parser.add_argument("--summaries-dir", default=None)
    parser.add_argument("--output-dir", default="output/year_rank_stability_exp")
    parser.add_argument("--top-k-metapaths", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries_dir = Path(args.summaries_dir) if args.summaries_dir else Path(args.results_dir).parent / "replicate_summaries"
    exp_dir = Path(args.output_dir) / "year_rank_stability_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _load_summary_bank(summaries_dir)
    runs_df = build_diff_runs(summary_df, join_keys=JOIN_KEYS)
    rank_df = rank_features(
        runs_df,
        rank_group_keys=["year", "control", "go_id", "replicate"],
        feature_col="metapath",
        score_col="diff",
        rank_col="metapath_rank",
    )
    pairwise_df, go_summary_df, overall_df = summarize_rank_stability(
        rank_df,
        outer_keys=["year", "control", "go_id"],
        replicate_col="replicate",
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
        y_label="Mean Spearman rho across replicates",
        title="Year metapath rank stability by control and year",
        out_path=exp_dir / "spearman_overall_by_group.png",
    )
    _plot_overall(
        overall_df,
        y_col="mean_topk_jaccard",
        y_label=f"Mean top-{args.top_k_metapaths} Jaccard",
        title="Year top-k overlap stability by control and year",
        out_path=exp_dir / "topk_jaccard_overall_by_group.png",
    )

    print(f"Saved rank table: {exp_dir / 'metapath_rank_table.csv'}")
    print(f"Saved pairwise metrics: {exp_dir / 'pairwise_metrics.csv'}")
    print(f"Saved overall summary: {exp_dir / 'overall_stability_summary.csv'}")


if __name__ == "__main__":
    main()
