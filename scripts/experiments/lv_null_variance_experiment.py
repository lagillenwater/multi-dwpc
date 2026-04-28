#!/usr/bin/env python3
"""LV null-variance analysis over explicit replicate summary artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_replicate_analysis import FEATURE_KEYS, build_b_seed_runs, load_summary_bank  # noqa: E402
from src.replicate_analysis import summarize_feature_variance, summarize_overall_variance  # noqa: E402


def _save_dual(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


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
        ax.plot(
            subset["b"].astype(int),
            subset[y_col].astype(float),
            marker="o",
            linewidth=2.2,
            color=colors.get(control, "#333333"),
        )
        ax.set_xlabel("B")
        ax.set_title(control)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    _save_dual(fig, out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--analysis-output-dir", default=None)
    parser.add_argument("--b-values", default="1,2,5,10,20")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    exp_root = Path(args.analysis_output_dir) if args.analysis_output_dir else output_dir / "lv_null_variance_experiment"
    exp_root.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_bank(output_dir)
    runs_df = build_b_seed_runs(summary_df, _parse_int_list(args.b_values), _parse_int_list(args.seeds))
    feature_df = summarize_feature_variance(
        runs_df,
        feature_keys=["control", "b", *FEATURE_KEYS],
    )
    overall_df = summarize_overall_variance(
        feature_df,
        overall_keys=["control", "b"],
        runs_df=runs_df,
        replicate_col="seed",
    )

    runs_df.to_csv(exp_root / "all_runs_wide.csv", index=False)
    feature_df.to_csv(exp_root / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(exp_root / "overall_variance_summary.csv", index=False)

    _plot_overall(
        overall_df,
        y_col="mean_diff_var",
        y_label="Mean feature variance of diff across LVs/metapaths",
        title="LV null variance by B",
        out_path=exp_root / "variance_overall_by_group.pdf",
    )
    _plot_overall(
        overall_df,
        y_col="mean_diff_std",
        y_label="Mean feature SD of diff across LVs/metapaths",
        title="LV null SD by B",
        out_path=exp_root / "sd_overall_by_group.pdf",
    )

    print(f"Saved runs: {exp_root / 'all_runs_wide.csv'}")
    print(f"Saved feature summary: {exp_root / 'feature_variance_summary.csv'}")
    print(f"Saved overall summary: {exp_root / 'overall_variance_summary.csv'}")


if __name__ == "__main__":
    main()
