#!/usr/bin/env python3
"""Identify LV top pairs and top paths from an existing LV workspace."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_subgraphs import extract_top_subgraphs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="output/lv_experiment",
        help="LV workspace directory containing lv_metapath_results.csv and shared inputs",
    )
    parser.add_argument("--top-metapaths", type=int, default=2)
    parser.add_argument("--top-pairs", type=int, default=10)
    parser.add_argument("--top-paths", type=int, default=5)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument(
        "--metapath-rank-metric",
        default="consensus_score",
        choices=["consensus_score", "consensus_rank", "min_d", "min_diff", "diff_perm", "diff_rand", "d_perm", "d_rand"],
    )
    parser.add_argument(
        "--pair-rank-metric",
        default="dwpc",
        choices=["dwpc", "contrast_min", "contrast_perm", "contrast_rand", "contrast_mean"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    top_pairs_df, top_paths_df = extract_top_subgraphs(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        top_metapaths=args.top_metapaths,
        top_pairs=args.top_pairs,
        top_paths=args.top_paths,
        degree_d=args.degree_d,
        metapath_rank_metric=args.metapath_rank_metric,
        pair_rank_metric=args.pair_rank_metric,
    )
    print(f"Saved top pairs: {output_dir / 'top_pairs.csv'} ({len(top_pairs_df):,} rows)")
    print(f"Saved top paths: {output_dir / 'top_paths.csv'} ({len(top_paths_df):,} rows)")


if __name__ == "__main__":
    main()
