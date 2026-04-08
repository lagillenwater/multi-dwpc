#!/usr/bin/env python3
"""Extract and plot LV top subgraphs as single-network panels per top metapath."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_subgraphs import plot_top_subgraphs  # noqa: E402


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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
        "--analysis-dir",
        default=None,
        help="Optional LV rank-stability analysis directory for rank-based top-path plotting.",
    )
    parser.add_argument(
        "--pair-rank-metric",
        default="dwpc",
        choices=["dwpc", "contrast_min", "contrast_perm", "contrast_rand", "contrast_mean"],
    )
    parser.add_argument("--control", default=None, help="Optional control filter in rank mode.")
    parser.add_argument("--b", type=int, default=None, help="Optional B filter in rank mode; defaults to max.")
    parser.add_argument("--seed", type=int, default=11, help="Seed filter in rank mode (default 11).")
    parser.add_argument(
        "--metapath-selection-method",
        default="top_n",
        choices=["top_n", "effective_number"],
    )
    parser.add_argument(
        "--effective-min-n",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--effective-max-n",
        type=int,
        default=None,
    )
    parser.add_argument("--min-shared-intermediates", type=int, default=0)
    parser.add_argument("--min-genes", type=int, default=2)
    parser.add_argument("--max-genes", type=int, default=8)
    return parser.parse_args()


def _plot_from_rank_runs(args: argparse.Namespace, output_dir: Path, analysis_dir: Path) -> None:
    top_paths_runs = analysis_dir / "top_paths_runs.csv"
    if not top_paths_runs.exists():
        cmd = [
            sys.executable,
            "scripts/identify_lv_rank_top_paths.py",
            "--analysis-dir",
            str(analysis_dir),
            "--workspace-dir",
            str(output_dir),
            "--top-metapaths-per-run",
            str(args.top_metapaths),
            "--top-pairs",
            str(args.top_pairs),
            "--top-paths",
            str(args.top_paths),
            "--degree-d",
            str(args.degree_d),
            "--metapath-rank-metric",
            str(args.metapath_rank_metric),
            "--pair-rank-metric",
            str(args.pair_rank_metric),
        ]
        _run(cmd)

    top_paths_df = pd.read_csv(top_paths_runs)
    if top_paths_df.empty:
        raise ValueError(f"No rows found in {top_paths_runs}")

    selected_b = int(args.b) if args.b is not None else int(top_paths_df["b"].astype(int).max())
    selected_seed = int(args.seed)
    controls = (
        [str(args.control)]
        if args.control is not None
        else sorted(top_paths_df["control"].astype(str).unique().tolist())
    )

    wrote_any = False
    for control in controls:
        subset = top_paths_df[
            (top_paths_df["control"].astype(str) == control)
            & (top_paths_df["b"].astype(int) == selected_b)
            & (top_paths_df["seed"].astype(int) == selected_seed)
        ].copy()
        if subset.empty:
            print(f"No rank top-path rows for control={control}, b={selected_b}, seed={selected_seed}.")
            continue

        stage_dir = analysis_dir / f"rank_top_subgraphs_{control}_b{selected_b}_seed{selected_seed}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_paths_path = stage_dir / "top_paths.csv"
        subset.to_csv(stage_paths_path, index=False)
        n_written = plot_top_subgraphs(
            output_dir=stage_dir,
            min_shared_intermediates=args.min_shared_intermediates,
            min_genes=args.min_genes,
            max_genes=args.max_genes,
        )
        print(f"Saved LV rank top-subgraph plots: {stage_dir / 'plots'} ({n_written} files)")
        wrote_any = True

    if not wrote_any:
        raise ValueError(
            "No LV rank top-subgraph plots were generated. "
            f"Checked controls={controls}, b={selected_b}, seed={selected_seed}."
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_dir / "lv_rank_stability_experiment"

    if (output_dir / "lv_metapath_results.csv").exists():
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
            "--metapath-selection-method",
            str(args.metapath_selection_method),
            "--pair-rank-metric",
            str(args.pair_rank_metric),
        ]
        if args.control is not None:
            cmd.extend(["--control", str(args.control)])
        if args.b is not None:
            cmd.extend(["--b", str(args.b)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.effective_min_n is not None:
            cmd.extend(["--effective-min-n", str(args.effective_min_n)])
        if args.effective_max_n is not None:
            cmd.extend(["--effective-max-n", str(args.effective_max_n)])
        _run(cmd)

        n_written = plot_top_subgraphs(
            output_dir=output_dir,
            min_shared_intermediates=args.min_shared_intermediates,
            min_genes=args.min_genes,
            max_genes=args.max_genes,
        )
        print(f"Saved LV top-subgraph plots: {output_dir / 'plots'} ({n_written} files)")
        return

    if analysis_dir.exists():
        _plot_from_rank_runs(args=args, output_dir=output_dir, analysis_dir=analysis_dir)
        return

    raise FileNotFoundError(
        "Could not find either a standard LV workspace "
        f"({output_dir / 'lv_metapath_results.csv'}) or a rank-stability analysis directory ({analysis_dir})."
    )


if __name__ == "__main__":
    main()
