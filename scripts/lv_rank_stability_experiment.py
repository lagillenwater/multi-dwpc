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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--analysis-output-dir", default=None)
    parser.add_argument("--b-values", default="1,2,5,10,20,50")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--top-k-metapaths", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    exp_root = Path(args.analysis_output_dir) if args.analysis_output_dir else output_dir / "lv_rank_stability_experiment"
    exp_root.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_bank(output_dir)
    runs_df = build_b_seed_runs(summary_df, _parse_int_list(args.b_values), _parse_int_list(args.seeds))
    rank_df = rank_features(
        runs_df,
        rank_group_keys=["control", "b", "seed", "lv_id", "target_set_id"],
        feature_col="metapath",
        score_col="diff",
        rank_col="metapath_rank",
    )
    pairwise_df, entity_df, overall_df = summarize_rank_stability(
        rank_df,
        outer_keys=["control", "b", "lv_id", "target_set_id"],
        replicate_col="seed",
        feature_col="metapath",
        rank_col="metapath_rank",
        top_k=args.top_k_metapaths,
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
    _plot_overall(
        overall_df,
        y_col="mean_topk_jaccard",
        y_label=f"Mean top-{args.top_k_metapaths} Jaccard across seeds",
        title="LV top-k overlap stability by B",
        out_path=exp_root / "topk_jaccard_overall_by_group.png",
    )

    print(f"Saved rank table: {exp_root / 'metapath_rank_table.csv'}")
    print(f"Saved pairwise metrics: {exp_root / 'pairwise_metrics.csv'}")
    print(f"Saved overall summary: {exp_root / 'overall_stability_summary.csv'}")


if __name__ == "__main__":
    main()
