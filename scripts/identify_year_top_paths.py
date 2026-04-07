#!/usr/bin/env python3
"""Run the existing year top-BP, top-pair, and top-path utilities together."""

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
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-metapaths", type=int, default=None)
    parser.add_argument("--pair-top-n", type=int, default=10)
    parser.add_argument("--top-paths", type=int, default=5)
    parser.add_argument("--pair-cumulative-frac", type=float, default=None)
    parser.add_argument("--pair-min-n", type=int, default=1)
    parser.add_argument("--pair-max-n", type=int, default=None)
    parser.add_argument("--path-cumulative-frac", type=float, default=None)
    parser.add_argument("--path-min-count", type=int, default=1)
    parser.add_argument("--path-max-count", type=int, default=None)
    parser.add_argument("--top-quantile", type=float, default=None)
    parser.add_argument("--top-z", type=float, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--support-path", default=None)
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--support-sort-metric", default="min_d", choices=["min_d", "min_diff", "real_mean"])
    parser.add_argument("--metapath", default=None)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--plot-path-instances",
        action="store_true",
        help="Generate side-by-side path-instance plots after extracting the top paths.",
    )
    parser.add_argument("--plot-label-top", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    top_bp_cmd = [
        sys.executable,
        "scripts/top_bps_by_metapath.py",
        "--years",
        *[str(year) for year in args.years],
        "--top-n",
        str(args.top_n),
        "--pair-top-n",
        str(args.pair_top_n),
        "--chunksize",
        str(args.chunksize),
    ]
    if args.results_dir:
        top_bp_cmd.extend(["--results-dir", str(args.results_dir)])
    if args.output_dir:
        top_bp_cmd.extend(["--output-dir", str(args.output_dir)])
    if args.top_metapaths is not None:
        top_bp_cmd.extend(["--top-metapaths", str(args.top_metapaths)])
    if args.top_quantile is not None:
        top_bp_cmd.extend(["--top-quantile", str(args.top_quantile)])
    if args.top_z is not None:
        top_bp_cmd.extend(["--top-z", str(args.top_z)])
    if args.support_path:
        top_bp_cmd.extend(["--support-path", str(args.support_path)])
    if args.support_only:
        top_bp_cmd.append("--support-only")
    if args.support_sort_metric:
        top_bp_cmd.extend(["--support-sort-metric", str(args.support_sort_metric)])
    if args.pair_cumulative_frac is not None:
        top_bp_cmd.extend(["--pair-cumulative-frac", str(args.pair_cumulative_frac)])
        top_bp_cmd.extend(["--pair-min-n", str(args.pair_min_n)])
        if args.pair_max_n is not None:
            top_bp_cmd.extend(["--pair-max-n", str(args.pair_max_n)])
    _run(top_bp_cmd)

    top_path_cmd = [
        sys.executable,
        "scripts/extract_top_paths_local.py",
        "--years",
        *[str(year) for year in args.years],
        "--top-pairs",
        str(args.pair_top_n),
        "--top-paths",
        str(args.top_paths),
        "--degree-d",
        str(args.degree_d),
    ]
    if args.output_dir:
        top_path_cmd.extend(["--top-dir", str(args.output_dir), "--output-dir", str(args.output_dir)])
    if args.path_cumulative_frac is not None:
        top_path_cmd.extend(["--path-cumulative-frac", str(args.path_cumulative_frac)])
        top_path_cmd.extend(["--path-min-count", str(args.path_min_count)])
        if args.path_max_count is not None:
            top_path_cmd.extend(["--path-max-count", str(args.path_max_count)])
    if args.metapath:
        top_path_cmd.extend(["--metapath", args.metapath])
    _run(top_path_cmd)

    if args.plot_path_instances:
        plot_cmd = [
            sys.executable,
            "scripts/plot_path_instances_networks.py",
            "--years",
            *[str(year) for year in args.years],
            "--label-top",
            str(args.plot_label_top),
        ]
        if len(args.years) == 2:
            plot_cmd.extend(["--side-by-side", "--align-nodes"])
        if args.metapath:
            plot_cmd.extend(["--metapath", args.metapath])
        _run(plot_cmd)


if __name__ == "__main__":
    main()
