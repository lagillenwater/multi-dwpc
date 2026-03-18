#!/usr/bin/env python3
"""Year null-variance analysis over explicit replicate summary artifacts."""

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
from src.replicate_analysis import summarize_feature_variance, summarize_overall_variance  # noqa: E402
from src.year_replicate_analysis import build_b_seed_runs, load_summary_bank  # noqa: E402


FEATURE_KEYS = ["year", "control", "b", "go_id", "metapath"]


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _plot_overall(overall_df: pd.DataFrame, y_col: str, y_label: str, title: str, output_path: Path) -> None:
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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="output/dwpc_direct/all_GO_positive_growth/results")
    parser.add_argument("--summaries-dir", default=None)
    parser.add_argument("--output-dir", default="output/year_null_variance_exp")
    parser.add_argument("--b-values", default="1,2,5,10,20")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries_dir = Path(args.summaries_dir) if args.summaries_dir else Path(args.results_dir).parent / "replicate_summaries"
    exp_dir = Path(args.output_dir) / "year_null_variance_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_bank(summaries_dir)
    runs_df = build_b_seed_runs(summary_df, _parse_int_list(args.b_values), _parse_int_list(args.seeds))
    feature_df = summarize_feature_variance(runs_df, FEATURE_KEYS)
    overall_df = summarize_overall_variance(
        feature_df,
        ["year", "control", "b"],
        runs_df=runs_df,
        replicate_col="seed",
    )

    runs_df.to_csv(exp_dir / "all_runs_long.csv", index=False)
    feature_df.to_csv(exp_dir / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(exp_dir / "overall_variance_summary.csv", index=False)

    _plot_overall(
        overall_df,
        y_col="mean_diff_var",
        y_label="Mean feature variance of diff across GO terms/metapaths",
        title="Year null variance by B",
        output_path=exp_dir / "variance_overall_by_group.png",
    )
    _plot_overall(
        overall_df,
        y_col="mean_diff_std",
        y_label="Mean feature SD of diff across GO terms/metapaths",
        title="Year null SD by B",
        output_path=exp_dir / "sd_overall_by_group.png",
    )

    print(f"Saved runs: {exp_dir / 'all_runs_long.csv'}")
    print(f"Saved feature summary: {exp_dir / 'feature_variance_summary.csv'}")
    print(f"Saved overall summary: {exp_dir / 'overall_variance_summary.csv'}")


if __name__ == "__main__":
    main()
