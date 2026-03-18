#!/usr/bin/env python3
"""Plot LV metapath-rank comparisons against a reference seed."""

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


SEED_COLORS = {
    22: "#1f77b4",
    33: "#ff7f0e",
    44: "#2ca02c",
    55: "#d62728",
}


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value)).strip("_")


def _load_rank_table(path: Path) -> pd.DataFrame:
    required_columns = {"b", "seed", "lv_id", "target_set_id", "metapath", "metapath_rank"}
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "control" not in df.columns:
        df = df.copy()
        df["control"] = "combined"
    return df


def _plot_group(group: pd.DataFrame, ref_seed: int, title: str, output_path: Path) -> None:
    ref_df = (
        group[group["seed"].astype(int) == int(ref_seed)][["metapath", "metapath_rank"]]
        .rename(columns={"metapath_rank": "ref_rank"})
        .copy()
    )
    if ref_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    max_rank = 0
    for seed in sorted(group["seed"].astype(int).unique().tolist()):
        if int(seed) == int(ref_seed):
            continue
        comp_df = (
            group[group["seed"].astype(int) == int(seed)][["metapath", "metapath_rank"]]
            .rename(columns={"metapath_rank": "comp_rank"})
            .copy()
        )
        merged = ref_df.merge(comp_df, on="metapath", how="inner")
        if merged.empty:
            continue
        max_rank = max(max_rank, int(merged[["ref_rank", "comp_rank"]].max().max()))
        ax.scatter(
            merged["ref_rank"].astype(float),
            merged["comp_rank"].astype(float),
            s=44,
            alpha=0.75,
            color=SEED_COLORS.get(int(seed), None),
            label=f"seed {seed}",
        )

    if max_rank == 0:
        plt.close(fig)
        return

    ax.plot([0.5, max_rank + 0.5], [0.5, max_rank + 0.5], linestyle="--", color="#999999", linewidth=1.5)
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.set_xlabel(f"Seed {ref_seed} rank")
    ax.set_ylabel("Comparison-seed rank")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(title="Seed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/lv_experiment/lv_rank_stability_experiment",
        help="Directory containing metapath_rank_table.csv",
    )
    parser.add_argument("--reference-seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    rank_df = _load_rank_table(analysis_dir / "metapath_rank_table.csv")
    output_dir = analysis_dir / f"rank_scatter_ref_seed_{args.reference_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, group in rank_df.groupby(["control", "b", "lv_id", "target_set_id"], sort=True):
        control, b, lv_id, target_set_id = key
        title = (
            f"Metapath Rank Scatter (control={control}, B={int(b)}, "
            f"{lv_id}, {target_set_id})"
        )
        out_name = (
            f"metapath_rank_scatter_ref_seed_{args.reference_seed}_"
            f"{_sanitize(control)}_b{int(b)}_{_sanitize(lv_id)}_{_sanitize(target_set_id)}.png"
        )
        out_path = output_dir / out_name
        _plot_group(group.copy(), args.reference_seed, title, out_path)

    print(f"Saved seed-comparison plots under: {output_dir}")


if __name__ == "__main__":
    main()
