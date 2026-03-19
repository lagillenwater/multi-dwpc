#!/usr/bin/env python3
"""Extract and plot year top subgraphs with side-by-side year comparisons."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--top-n", type=int, default=5, help="Top GO terms per metapath.")
    parser.add_argument("--pair-top-n", type=int, default=10, help="Top gene-BP pairs per GO term.")
    parser.add_argument("--top-paths", type=int, default=5, help="Top enumerated paths per pair.")
    parser.add_argument("--top-quantile", type=float, default=None)
    parser.add_argument("--top-z", type=float, default=None)
    parser.add_argument("--metapath", default=None)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--label-top-genes", type=int, default=8)
    parser.add_argument("--label-top-path-nodes", type=int, default=0)
    parser.add_argument("--align-nodes", action="store_true")
    parser.add_argument("--rank-by-paired-diff", action="store_true")
    parser.add_argument("--rank-by-path-change", action="store_true")
    parser.add_argument("--paired-stat", default="mean_dwpc")
    parser.add_argument("--paired-top-n", type=int, default=10)
    parser.add_argument("--paired-abs", action="store_true")
    parser.add_argument("--path-change-top-n", type=int, default=10)
    parser.add_argument("--path-change-metric", choices=["jaccard", "count"], default="jaccard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extract_cmd = [
        sys.executable,
        "scripts/identify_year_top_paths.py",
        "--years",
        *[str(year) for year in args.years],
        "--top-n",
        str(args.top_n),
        "--pair-top-n",
        str(args.pair_top_n),
        "--top-paths",
        str(args.top_paths),
        "--degree-d",
        str(args.degree_d),
        "--chunksize",
        str(args.chunksize),
    ]
    if args.top_quantile is not None:
        extract_cmd.extend(["--top-quantile", str(args.top_quantile)])
    if args.top_z is not None:
        extract_cmd.extend(["--top-z", str(args.top_z)])
    if args.metapath:
        extract_cmd.extend(["--metapath", args.metapath])
    _run(extract_cmd)

    pair_plot_cmd = [
        sys.executable,
        "scripts/plot_top_paths_networks.py",
        "--years",
        *[str(year) for year in args.years],
        "--side-by-side",
        "--label-top-genes",
        str(args.label_top_genes),
    ]
    if args.metapath:
        pair_plot_cmd.extend(["--metapath", args.metapath])
    _run(pair_plot_cmd)

    path_plot_cmd = [
        sys.executable,
        "scripts/plot_path_instances_networks.py",
        "--years",
        *[str(year) for year in args.years],
        "--side-by-side",
        "--label-top",
        str(args.label_top_path_nodes),
    ]
    if args.metapath:
        path_plot_cmd.extend(["--metapath", args.metapath])
    if args.align_nodes:
        path_plot_cmd.append("--align-nodes")
    if args.rank_by_paired_diff:
        path_plot_cmd.extend(
            [
                "--rank-by-paired-diff",
                "--paired-stat",
                str(args.paired_stat),
                "--paired-top-n",
                str(args.paired_top_n),
            ]
        )
        if args.paired_abs:
            path_plot_cmd.append("--paired-abs")
    if args.rank_by_path_change:
        path_plot_cmd.extend(
            [
                "--rank-by-path-change",
                "--path-change-top-n",
                str(args.path_change_top_n),
                "--path-change-metric",
                str(args.path_change_metric),
            ]
        )
    _run(path_plot_cmd)


if __name__ == "__main__":
    main()
