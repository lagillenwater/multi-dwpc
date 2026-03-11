#!/usr/bin/env python3
"""
Run a compact LV null-variance experiment across fixed replicate counts.

Experiment design:
- Build/reuse a compact LV workspace with limited metapaths.
- Sweep B in {1, 2, 5, 10, 20, 50} (or user-specified list).
- For each B and random seed, run nulls+stats and collect diff metrics.
- Summarize variance of diff_perm and diff_rand across seeds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_RESULT_COLUMNS = {
    "lv_id",
    "target_set_id",
    "metapath",
    "diff_perm",
    "diff_rand",
    "d_perm",
    "d_rand",
    "p_perm",
    "p_rand",
    "p_perm_fdr",
    "p_rand_fdr",
}


def _parse_int_list(arg: str) -> list[int]:
    values = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pipeline_script(root: Path) -> Path:
    path = root / "scripts" / "lv_multidwpc_analysis.py"
    if not path.exists():
        raise FileNotFoundError(
            "Missing LV pipeline orchestrator: "
            f"{path}\n"
            "This repository checkout does not include LV pipeline files. "
            "Use a branch that contains them (for example: "
            "`git switch main && git pull --ff-only`) and rerun."
        )
    return path


def _assert_lv_pipeline_files(root: Path) -> None:
    required = [
        root / "scripts" / "lv_multidwpc_analysis.py",
        root / "src" / "lv_inputs.py",
        root / "src" / "lv_nulls.py",
        root / "src" / "lv_precompute.py",
        root / "src" / "lv_stats.py",
        root / "src" / "lv_pairs.py",
        root / "src" / "lv_targets.py",
        root / "src" / "lv_dwpc.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "LV variance experiment requires LV pipeline files that are missing "
            "in this checkout:\n"
            f"  - {joined}\n"
            "Switch to a branch that includes the LV pipeline "
            "(e.g., `git switch main && git pull --ff-only`)."
        )


def _stage_cmd(
    python_exe: str,
    pipeline_path: Path,
    stage: str,
    output_dir: Path,
) -> list[str]:
    return [
        python_exe,
        str(pipeline_path),
        "--stage",
        stage,
        "--output-dir",
        str(output_dir),
    ]


def _prepare_workspace(args: argparse.Namespace, root: Path) -> None:
    pipeline_path = _pipeline_script(root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # top-genes
    cmd = _stage_cmd(args.python_exe, pipeline_path, "top-genes", output_dir)
    cmd.extend(["--lvs", args.lvs, "--top-fraction", str(args.top_fraction)])
    if args.smoke:
        cmd.append("--smoke")
    else:
        if not args.lv_loadings:
            raise ValueError(
                "--lv-loadings is required unless --smoke is enabled."
            )
        cmd.extend(["--lv-loadings", args.lv_loadings])
    if args.gene_reference:
        cmd.extend(["--gene-reference", args.gene_reference])
    if args.resume_setup:
        cmd.append("--resume")
    if args.force_setup:
        cmd.append("--force")
    _run(cmd, cwd=root)

    # target-sets
    cmd = _stage_cmd(args.python_exe, pipeline_path, "target-sets", output_dir)
    cmd.extend(["--lvs", args.lvs])
    if args.include_brown_adipose:
        cmd.append("--include-brown-adipose")
    if args.resume_setup:
        cmd.append("--resume")
    if args.force_setup:
        cmd.append("--force")
    _run(cmd, cwd=root)

    # precompute-scores
    cmd = _stage_cmd(args.python_exe, pipeline_path, "precompute-scores", output_dir)
    cmd.extend(
        [
            "--n-workers-precompute",
            str(args.n_workers_precompute),
            "--max-metapath-length",
            str(args.max_metapath_length),
        ]
    )
    if args.metapath_limit_per_target is not None:
        cmd.extend(
            ["--metapath-limit-per-target", str(args.metapath_limit_per_target)]
        )
    if args.include_direct_metapaths:
        cmd.append("--include-direct-metapaths")
    if args.resume_setup:
        cmd.append("--resume")
    if args.force_setup:
        cmd.append("--force")
    _run(cmd, cwd=root)


def _collect_long_form(run_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "b",
        "seed",
        "lv_id",
        "target_set_id",
        "metapath",
        "node_type",
        "target_set_label",
    ]
    out_frames = []
    specs = [
        ("permuted", "diff_perm", "d_perm", "p_perm", "p_perm_fdr"),
        ("random", "diff_rand", "d_rand", "p_rand", "p_rand_fdr"),
    ]
    for control, diff_col, d_col, p_col, p_fdr_col in specs:
        part = run_df[id_cols + [diff_col, d_col, p_col, p_fdr_col]].copy()
        part = part.rename(
            columns={
                diff_col: "diff",
                d_col: "d",
                p_col: "p",
                p_fdr_col: "p_fdr",
            }
        )
        part["control"] = control
        out_frames.append(part)
    return pd.concat(out_frames, ignore_index=True)


def _feature_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["control", "lv_id", "target_set_id", "metapath", "node_type", "b"]
    for key, group in long_df.groupby(keys, dropna=False):
        diff_vals = group["diff"].to_numpy(dtype=float)
        d_vals = group["d"].to_numpy(dtype=float)
        rows.append(
            {
                "control": key[0],
                "lv_id": key[1],
                "target_set_id": key[2],
                "metapath": key[3],
                "node_type": key[4],
                "b": int(key[5]),
                "n_seeds": int(len(group)),
                "diff_mean": float(np.nanmean(diff_vals)),
                "diff_std": float(np.nanstd(diff_vals, ddof=1))
                if len(group) > 1
                else np.nan,
                "diff_var": float(np.nanvar(diff_vals, ddof=1))
                if len(group) > 1
                else np.nan,
                "d_mean": float(np.nanmean(d_vals)),
                "d_std": float(np.nanstd(d_vals, ddof=1))
                if len(group) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["control", "lv_id", "target_set_id", "metapath", "b"]
    )


def _overall_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        feature_df.groupby(["control", "b"], as_index=False)
        .agg(
            n_features=("metapath", "size"),
            mean_diff_std=("diff_std", "mean"),
            mean_diff_var=("diff_var", "mean"),
            median_diff_std=("diff_std", "median"),
            median_diff_var=("diff_var", "median"),
        )
        .sort_values(["control", "b"])
    )
    return out


def _plot_overall_variance(overall_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"permuted": "#1f77b4", "random": "#d62728"}
    for control, group in overall_df.groupby("control"):
        group = group.sort_values("b")
        ax.plot(
            group["b"],
            group["mean_diff_var"],
            marker="o",
            linewidth=2.0,
            label=control,
            color=colors.get(control),
        )
    ax.set_xlabel("Null replicate count (B)")
    ax.set_ylabel("Mean variance of diff across features")
    ax.set_title("DWPC Difference Variance vs B")
    ax.grid(alpha=0.3)
    ax.legend(title="Control")
    fig.tight_layout()
    fig.savefig(output_dir / "variance_overall_by_b.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_sd(overall_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"permuted": "#1f77b4", "random": "#d62728"}
    for control, group in overall_df.groupby("control"):
        group = group.sort_values("b")
        ax.plot(
            group["b"],
            group["mean_diff_std"],
            marker="o",
            linewidth=2.0,
            label=control,
            color=colors.get(control),
        )
    ax.set_xlabel("Null replicate count (B)")
    ax.set_ylabel("Mean SD of diff across features")
    ax.set_title("DWPC Difference SD vs B")
    ax.grid(alpha=0.3)
    ax.legend(title="Control")
    fig.tight_layout()
    fig.savefig(output_dir / "sd_overall_by_b.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_sd(
    feature_df: pd.DataFrame, control: str, output_dir: Path, max_features: int
) -> None:
    subset = feature_df[feature_df["control"] == control].copy()
    if subset.empty:
        return

    subset["feature_label"] = (
        subset["lv_id"].astype(str)
        + " | "
        + subset["target_set_id"].astype(str)
        + " | "
        + subset["metapath"].astype(str)
    )
    rank = (
        subset.groupby("feature_label")["diff_var"]
        .mean()
        .sort_values(ascending=False)
        .head(max_features)
        .index
    )
    subset = subset[subset["feature_label"].isin(rank)].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, group in subset.groupby("feature_label", sort=True):
        group = group.sort_values("b")
        ax.plot(
            group["b"],
            group["diff_std"],
            marker="o",
            linewidth=1.4,
            alpha=0.9,
            label=label,
        )
    ax.set_xlabel("Null replicate count (B)")
    ax.set_ylabel("SD(diff) across seeds")
    ax.set_title(f"Per-feature SD stabilization vs B ({control})")
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / f"sd_by_b_per_feature_{control}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    root = _repo_root()
    _assert_lv_pipeline_files(root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    b_values = _parse_int_list(args.b_values)
    seeds = _parse_int_list(args.seeds)

    print("[setup] Preparing compact LV workspace...")
    _prepare_workspace(args=args, root=root)

    pipeline_path = _pipeline_script(root)
    result_rows = []
    qc_rows = []

    for b in b_values:
        for seed in seeds:
            print(f"[experiment] Running B={b}, seed={seed}")
            null_cmd = _stage_cmd(
                args.python_exe, pipeline_path, "nulls", output_dir
            )
            null_cmd.extend(
                [
                    "--n-degree-bins",
                    str(args.n_degree_bins),
                    "--b-min",
                    str(b),
                    "--b-max",
                    str(b),
                    "--b-batch",
                    str(args.b_batch),
                    "--adaptive-p-low",
                    str(args.adaptive_p_low),
                    "--adaptive-p-high",
                    str(args.adaptive_p_high),
                    "--random-seed",
                    str(seed),
                    "--force",
                ]
            )
            _run(null_cmd, cwd=root)

            stats_cmd = _stage_cmd(
                args.python_exe, pipeline_path, "stats", output_dir
            )
            stats_cmd.append("--force")
            _run(stats_cmd, cwd=root)

            results_path = output_dir / "lv_metapath_results.csv"
            if not results_path.exists():
                raise FileNotFoundError(f"Expected stats output not found: {results_path}")
            run_df = pd.read_csv(results_path)
            missing = REQUIRED_RESULT_COLUMNS - set(run_df.columns)
            if missing:
                raise ValueError(f"Missing expected columns in stats output: {sorted(missing)}")
            run_df["b"] = int(b)
            run_df["seed"] = int(seed)
            result_rows.append(run_df)

            qc_path = output_dir / "null_sampling_qc.csv"
            if qc_path.exists():
                qc_df = pd.read_csv(qc_path)
                qc_df["b"] = int(b)
                qc_df["seed"] = int(seed)
                qc_rows.append(qc_df)

    runs_df = pd.concat(result_rows, ignore_index=True)
    long_df = _collect_long_form(runs_df)
    feature_df = _feature_summary(long_df)
    overall_df = _overall_summary(feature_df)

    exp_dir = output_dir / "null_variance_experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(exp_dir / "all_runs_wide.csv", index=False)
    long_df.to_csv(exp_dir / "all_runs_long.csv", index=False)
    feature_df.to_csv(exp_dir / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(exp_dir / "overall_variance_summary.csv", index=False)
    if qc_rows:
        pd.concat(qc_rows, ignore_index=True).to_csv(
            exp_dir / "null_sampling_qc_all_runs.csv", index=False
        )

    _plot_overall_variance(overall_df=overall_df, output_dir=exp_dir)
    _plot_overall_sd(overall_df=overall_df, output_dir=exp_dir)
    _plot_feature_sd(
        feature_df=feature_df,
        control="permuted",
        output_dir=exp_dir,
        max_features=args.max_feature_lines,
    )
    _plot_feature_sd(
        feature_df=feature_df,
        control="random",
        output_dir=exp_dir,
        max_features=args.max_feature_lines,
    )

    print("\n[done] LV null variance experiment complete.")
    print(f"  Workspace: {output_dir}")
    print(f"  Experiment outputs: {exp_dir}")
    print(f"  B values: {b_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Features analyzed: {feature_df[['lv_id', 'target_set_id', 'metapath']].drop_duplicates().shape[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-B null sweeps for LV pipeline and quantify diff variance."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="output/lv_null_variance_exp",
        help="LV workspace/output directory for this experiment.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to invoke LV pipeline stages.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use synthetic LV loadings for faster smoke-style experiment.",
    )
    parser.add_argument(
        "--lv-loadings",
        default=None,
        help="Path to LV loadings table (required unless --smoke).",
    )
    parser.add_argument(
        "--lvs",
        default="LV603,LV246,LV57",
        help="Comma-separated LV IDs for the experiment.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.005,
        help="Top fraction for LV gene selection.",
    )
    parser.add_argument(
        "--gene-reference",
        default="data/nodes/Gene.tsv",
        help="Gene reference table used by top-genes stage.",
    )
    parser.add_argument(
        "--include-brown-adipose",
        action="store_true",
        help="Include brown adipose in LV246 target set.",
    )
    parser.add_argument(
        "--resume-setup",
        action="store_true",
        help="Reuse existing setup artifacts for top-genes/target-sets/precompute.",
    )
    parser.add_argument(
        "--force-setup",
        action="store_true",
        help="Force recomputation of setup stages.",
    )
    parser.add_argument(
        "--metapath-limit-per-target",
        type=int,
        default=2,
        help="Limit metapaths per target type to keep the experiment compact.",
    )
    parser.add_argument(
        "--max-metapath-length",
        type=int,
        default=3,
        help="Maximum metapath length for precompute stage.",
    )
    parser.add_argument(
        "--n-workers-precompute",
        type=int,
        default=2,
        help="Worker count for precompute stage.",
    )
    parser.add_argument(
        "--include-direct-metapaths",
        action="store_true",
        help="Include direct metapaths in precompute stage.",
    )
    parser.add_argument(
        "--b-values",
        default="1,2,5,10,20,50",
        help="Comma-separated fixed B values to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
        help="Comma-separated random seeds per B.",
    )
    parser.add_argument(
        "--n-degree-bins",
        type=int,
        default=10,
        help="Degree bins for null sampling.",
    )
    parser.add_argument(
        "--b-batch",
        type=int,
        default=1,
        help="Batch size for null loop; 1 keeps B accounting explicit.",
    )
    parser.add_argument(
        "--adaptive-p-low",
        type=float,
        default=0.005,
        help="Adaptive lower p threshold passed to null stage.",
    )
    parser.add_argument(
        "--adaptive-p-high",
        type=float,
        default=0.20,
        help="Adaptive upper p threshold passed to null stage.",
    )
    parser.add_argument(
        "--max-feature-lines",
        type=int,
        default=12,
        help="Maximum feature lines in per-feature SD plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args=args)


if __name__ == "__main__":
    main()
