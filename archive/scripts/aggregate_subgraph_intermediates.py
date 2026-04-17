#!/usr/bin/env python3
"""Aggregate per-GO subgraph intermediate results from array job outputs.

Concatenates path_instances_GO:*.csv, intermediate_sharing_GO:*.csv, and
intermediate_summary_GO:*.csv into combined files, then prints global stats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _concat_pattern(input_dir: Path, prefix: str) -> pd.DataFrame:
    pattern = f"{prefix}_GO:*.csv"
    files = sorted(input_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="output/year_subgraph_intermediates")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    paths_df = _concat_pattern(input_dir, "path_instances")
    sharing_df = _concat_pattern(input_dir, "intermediate_sharing")
    summary_df = _concat_pattern(input_dir, "intermediate_summary")

    if not paths_df.empty:
        paths_df.to_csv(output_dir / "path_instances_all.csv", index=False)
        print(f"Saved: {output_dir / 'path_instances_all.csv'} ({len(paths_df):,} rows)")

    if not sharing_df.empty:
        sharing_df.to_csv(output_dir / "intermediate_sharing_all.csv", index=False)
        print(f"Saved: {output_dir / 'intermediate_sharing_all.csv'} ({len(sharing_df):,} rows)")

    if not summary_df.empty:
        summary_df.to_csv(output_dir / "intermediate_summary_all.csv", index=False)
        print(f"Saved: {output_dir / 'intermediate_summary_all.csv'} ({len(summary_df):,} rows)")

        print(f"\n--- Intermediate node sharing (aggregated) ---")
        print(f"  (GO term, metapath) groups: {len(summary_df):,}")
        has_shared = summary_df[summary_df["n_shared_intermediates"] > 0]
        print(f"  Groups with shared intermediates: {len(has_shared):,} "
              f"({len(has_shared)/len(summary_df):.1%})")
        print(f"  Max genes sharing one intermediate: {summary_df['max_genes_per_intermediate'].max()}")
        print(f"  Median max_genes_per_intermediate: {summary_df['max_genes_per_intermediate'].median():.0f}")
        print(f"  Median frac_intermediates_shared: {summary_df['frac_intermediates_shared'].median():.3f}")
        print(f"  Mean frac_intermediates_shared:   {summary_df['frac_intermediates_shared'].mean():.3f}")

        # Distribution of n_genes per group
        if "n_genes" in summary_df.columns:
            print(f"\n  Genes per (GO, metapath) group:")
            print(f"    median: {summary_df['n_genes'].median():.0f}, "
                  f"mean: {summary_df['n_genes'].mean():.1f}, "
                  f"max: {summary_df['n_genes'].max()}")
    else:
        print("No intermediate summary files found.")


if __name__ == "__main__":
    main()
