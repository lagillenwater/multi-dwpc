#!/usr/bin/env python3
"""Extract and plot LV top subgraphs as single-network panels per top metapath."""

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
from src.lv_subgraphs import plot_top_subgraphs  # noqa: E402


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
        "--pair-rank-metric",
        default="dwpc",
        choices=["dwpc", "contrast_min", "contrast_perm", "contrast_rand", "contrast_mean"],
    )
    parser.add_argument("--min-shared-intermediates", type=int, default=0)
    parser.add_argument("--min-genes", type=int, default=2)
    parser.add_argument("--max-genes", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Reuse the existing LV extraction script to build top_pairs.csv and top_paths.csv.
    import subprocess

    cmd = [
        sys.executable,
        "scripts/identify_lv_top_paths.py",
        "--output-dir",
        str(output_dir),
        "--top-metapaths",
        str(args.top_metapaths),
        "--top-pairs",
        str(args.top_pairs),
        "--top-paths",
        str(args.top_paths),
        "--degree-d",
        str(args.degree_d),
        "--pair-rank-metric",
        str(args.pair_rank_metric),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    n_written = plot_top_subgraphs(
        output_dir=output_dir,
        min_shared_intermediates=args.min_shared_intermediates,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
    )
    print(f"Saved LV top-subgraph plots: {output_dir / 'plots'} ({n_written} files)")


if __name__ == "__main__":
    main()
