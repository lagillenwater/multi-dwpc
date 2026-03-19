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
    parser.add_argument("--top-quantile", type=float, default=None)
    parser.add_argument("--top-z", type=float, default=None)
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
    if args.top_metapaths is not None:
        top_bp_cmd.extend(["--top-metapaths", str(args.top_metapaths)])
    if args.top_quantile is not None:
        top_bp_cmd.extend(["--top-quantile", str(args.top_quantile)])
    if args.top_z is not None:
        top_bp_cmd.extend(["--top-z", str(args.top_z)])
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
