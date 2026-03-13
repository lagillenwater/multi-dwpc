#!/usr/bin/env python3
"""
Run LV metapath/path rank-stability experiment across fixed null replicate counts.

Experiment design:
- Build/reuse a compact LV workspace with limited metapaths.
- Sweep B in {1, 2, 5, 10, 20, 50} (or user-specified list).
- For each B and seed, run nulls+stats and optionally top-subgraphs.
- Quantify metapath rank stability and path-instance selection/rank stability.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zlib
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_METAPATH_COLUMNS = {
    "lv_id",
    "target_set_id",
    "metapath",
    "min_d",
    "min_diff",
}

REQUIRED_TOP_PAIRS_COLUMNS = {
    "lv_id",
    "target_set_id",
    "metapath",
    "pair_rank",
}

REQUIRED_TOP_PATHS_COLUMNS = {
    "lv_id",
    "target_set_id",
    "metapath_g_orientation",
    "target_id",
    "gene_identifier",
    "pair_rank",
    "path_rank",
    "path_nodes_ids",
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
            "Use a branch that contains LV pipeline files "
            "(for example: `git switch main && git pull --ff-only`) and rerun."
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
        root / "src" / "lv_subgraphs.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "LV rank-stability experiment requires LV pipeline files that are missing "
            "in this checkout:\n"
            f"  - {joined}\n"
            "Switch to a branch that includes the LV pipeline "
            "(for example: `git switch main && git pull --ff-only`)."
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
            raise ValueError("--lv-loadings is required unless --smoke is enabled.")
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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(items: list[set[str]]) -> float:
    if len(items) < 2:
        return np.nan
    vals = [_jaccard(a, b) for a, b in combinations(items, 2)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def _collect_run_tables(
    output_dir: Path,
    b: int,
    seed: int,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metapath_path = output_dir / "lv_metapath_results.csv"
    top_pairs_path = output_dir / "top_pairs.csv"
    top_paths_path = output_dir / "top_paths.csv"
    qc_path = output_dir / "null_sampling_qc.csv"

    if not metapath_path.exists():
        raise FileNotFoundError(f"Expected stats output not found: {metapath_path}")

    metapath_df = pd.read_csv(metapath_path)
    missing = REQUIRED_METAPATH_COLUMNS - set(metapath_df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns in {metapath_path}: {sorted(missing)}"
        )
    metapath_df["b"] = int(b)
    metapath_df["seed"] = int(seed)
    metapath_df["run_id"] = run_id

    if top_pairs_path.exists():
        top_pairs_df = pd.read_csv(top_pairs_path)
        missing = REQUIRED_TOP_PAIRS_COLUMNS - set(top_pairs_df.columns)
        if missing:
            raise ValueError(
                f"Missing expected columns in {top_pairs_path}: {sorted(missing)}"
            )
    else:
        top_pairs_df = pd.DataFrame(columns=sorted(REQUIRED_TOP_PAIRS_COLUMNS))
    top_pairs_df["b"] = int(b)
    top_pairs_df["seed"] = int(seed)
    top_pairs_df["run_id"] = run_id

    if top_paths_path.exists():
        top_paths_df = pd.read_csv(top_paths_path)
        missing = REQUIRED_TOP_PATHS_COLUMNS - set(top_paths_df.columns)
        if missing:
            raise ValueError(
                f"Missing expected columns in {top_paths_path}: {sorted(missing)}"
            )
    else:
        top_paths_df = pd.DataFrame(columns=sorted(REQUIRED_TOP_PATHS_COLUMNS))
    top_paths_df["b"] = int(b)
    top_paths_df["seed"] = int(seed)
    top_paths_df["run_id"] = run_id

    if qc_path.exists():
        qc_df = pd.read_csv(qc_path)
    else:
        qc_df = pd.DataFrame()
    qc_df["b"] = int(b)
    qc_df["seed"] = int(seed)
    qc_df["run_id"] = run_id
    return metapath_df, top_pairs_df, top_paths_df, qc_df


def _rank_metapaths_within_run(metapath_runs_df: pd.DataFrame) -> pd.DataFrame:
    df = metapath_runs_df.copy()
    sort_cols = [
        "run_id",
        "lv_id",
        "target_set_id",
        "min_d",
        "min_diff",
        "metapath",
    ]
    df = df.sort_values(
        sort_cols,
        ascending=[True, True, True, False, False, True],
    )
    df["metapath_rank"] = (
        df.groupby(["run_id", "lv_id", "target_set_id"]).cumcount() + 1
    )
    return df


def _metapath_pairwise_metrics(
    rank_df: pd.DataFrame, top_k_metapaths: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_keys = ["b", "lv_id", "target_set_id"]
    for key, group in rank_df.groupby(group_keys, dropna=False):
        by_seed = {}
        for seed, seed_df in group.groupby("seed"):
            ranks = dict(zip(seed_df["metapath"], seed_df["metapath_rank"]))
            by_seed[int(seed)] = ranks

        seeds = sorted(by_seed)
        for seed_a, seed_b in combinations(seeds, 2):
            r_a = by_seed[seed_a]
            r_b = by_seed[seed_b]
            common = sorted(set(r_a) & set(r_b))
            if len(common) < 2:
                continue
            x = pd.Series([r_a[m] for m in common], dtype=float)
            y = pd.Series([r_b[m] for m in common], dtype=float)
            rho = x.corr(y, method="spearman")
            top_a = {m for m, r in r_a.items() if r <= top_k_metapaths}
            top_b = {m for m, r in r_b.items() if r <= top_k_metapaths}
            rows.append(
                {
                    "b": int(key[0]),
                    "lv_id": key[1],
                    "target_set_id": key[2],
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "n_common_metapaths": int(len(common)),
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "topk_jaccard": _jaccard(top_a, top_b),
                }
            )

    pairwise_df = pd.DataFrame(rows)
    if pairwise_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "b",
                "n_pairs",
                "mean_spearman_rho",
                "median_spearman_rho",
                "mean_topk_jaccard",
                "median_topk_jaccard",
            ]
        )
        return pairwise_df, summary_df

    summary_df = (
        pairwise_df.groupby("b", as_index=False)
        .agg(
            n_pairs=("spearman_rho", "size"),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            mean_topk_jaccard=("topk_jaccard", "mean"),
            median_topk_jaccard=("topk_jaccard", "median"),
        )
        .sort_values("b")
    )
    return pairwise_df, summary_df


def _selected_metapath_jaccard(top_pairs_runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if top_pairs_runs_df.empty:
        return pd.DataFrame(columns=["b", "mean_selected_metapath_jaccard"])

    keys = ["b", "lv_id", "target_set_id"]
    for key, group in top_pairs_runs_df.groupby(keys, dropna=False):
        sets = []
        for _, run_df in group.groupby("run_id"):
            sets.append(set(run_df["metapath"].astype(str).unique()))
        rows.append(
            {
                "b": int(key[0]),
                "lv_id": key[1],
                "target_set_id": key[2],
                "mean_selected_metapath_jaccard": _mean_pairwise_jaccard(sets),
            }
        )
    return pd.DataFrame(rows)


def _path_pairwise_selection_metrics(
    top_paths_runs_df: pd.DataFrame,
    top_k_path_instances: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if top_paths_runs_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "b",
                    "lv_id",
                    "target_set_id",
                    "metapath_g_orientation",
                    "target_id",
                    "seed_a",
                    "seed_b",
                    "path_jaccard",
                ]
            ),
            pd.DataFrame(columns=["b", "n_pairs", "mean_path_jaccard"]),
        )

    df = top_paths_runs_df.copy()
    df["path_key"] = (
        df["gene_identifier"].astype(str) + "||" + df["path_nodes_ids"].astype(str)
    )
    df = df.sort_values(
        [
            "run_id",
            "lv_id",
            "target_set_id",
            "metapath_g_orientation",
            "target_id",
            "pair_rank",
            "path_rank",
            "path_nodes_ids",
        ]
    )
    df["instance_rank"] = (
        df.groupby(
            [
                "run_id",
                "lv_id",
                "target_set_id",
                "metapath_g_orientation",
                "target_id",
            ]
        ).cumcount()
        + 1
    )
    df = df[df["instance_rank"] <= int(top_k_path_instances)].copy()

    keys = ["b", "lv_id", "target_set_id", "metapath_g_orientation", "target_id"]
    for key, group in df.groupby(keys, dropna=False):
        by_seed = {}
        for seed, run_df in group.groupby("seed"):
            by_seed[int(seed)] = set(run_df["path_key"].astype(str))
        seeds = sorted(by_seed)
        for seed_a, seed_b in combinations(seeds, 2):
            rows.append(
                {
                    "b": int(key[0]),
                    "lv_id": key[1],
                    "target_set_id": key[2],
                    "metapath_g_orientation": key[3],
                    "target_id": str(key[4]),
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "path_jaccard": _jaccard(by_seed[seed_a], by_seed[seed_b]),
                }
            )

    pairwise_df = pd.DataFrame(rows)
    if pairwise_df.empty:
        summary_df = pd.DataFrame(columns=["b", "n_pairs", "mean_path_jaccard"])
        return pairwise_df, summary_df

    summary_df = (
        pairwise_df.groupby("b", as_index=False)
        .agg(
            n_pairs=("path_jaccard", "size"),
            mean_path_jaccard=("path_jaccard", "mean"),
            median_path_jaccard=("path_jaccard", "median"),
        )
        .sort_values("b")
    )
    return pairwise_df, summary_df


def _path_rank_variability(
    top_paths_runs_df: pd.DataFrame, top_k_path_instances: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_paths_runs_df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "b",
                    "lv_id",
                    "target_set_id",
                    "metapath_g_orientation",
                    "target_id",
                    "path_key",
                    "n_runs",
                    "rank_sd",
                ]
            ),
            pd.DataFrame(columns=["b", "n_paths_shared", "mean_rank_sd"]),
        )

    df = top_paths_runs_df.copy()
    df["path_key"] = (
        df["gene_identifier"].astype(str) + "||" + df["path_nodes_ids"].astype(str)
    )
    df = df.sort_values(
        [
            "run_id",
            "lv_id",
            "target_set_id",
            "metapath_g_orientation",
            "target_id",
            "pair_rank",
            "path_rank",
            "path_nodes_ids",
        ]
    )
    df["instance_rank"] = (
        df.groupby(
            [
                "run_id",
                "lv_id",
                "target_set_id",
                "metapath_g_orientation",
                "target_id",
            ]
        ).cumcount()
        + 1
    )
    df = df[df["instance_rank"] <= int(top_k_path_instances)].copy()

    rows = []
    keys = [
        "b",
        "lv_id",
        "target_set_id",
        "metapath_g_orientation",
        "target_id",
        "path_key",
    ]
    for key, group in df.groupby(keys, dropna=False):
        if group["seed"].nunique() < 2:
            continue
        ranks = group["instance_rank"].astype(float).to_numpy()
        rank_sd = float(np.nanstd(ranks, ddof=1)) if len(ranks) > 1 else np.nan
        rows.append(
            {
                "b": int(key[0]),
                "lv_id": key[1],
                "target_set_id": key[2],
                "metapath_g_orientation": key[3],
                "target_id": str(key[4]),
                "path_key": key[5],
                "n_runs": int(group["seed"].nunique()),
                "rank_sd": rank_sd,
            }
        )
    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        summary_df = pd.DataFrame(columns=["b", "n_paths_shared", "mean_rank_sd"])
        return detail_df, summary_df

    summary_df = (
        detail_df.groupby("b", as_index=False)
        .agg(
            n_paths_shared=("rank_sd", "size"),
            mean_rank_sd=("rank_sd", "mean"),
            median_rank_sd=("rank_sd", "median"),
        )
        .sort_values("b")
    )
    return detail_df, summary_df


def _plot_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    work = df[[x_col, y_col]].dropna().copy()
    if work.empty:
        return
    work = work.sort_values(x_col)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(work[x_col], work[y_col], marker="o", linewidth=2.0, color="#1f77b4")
    # Force explicit replicate-count ticks (for example: 1, 2, 5, 10, 20, 50).
    x_vals = sorted({int(v) for v in work[x_col].tolist()})
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_xlabel("Null replicate count (B)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metapath_pairwise_scatter_by_b(
    pairwise_df: pd.DataFrame, output_dir: Path
) -> None:
    if pairwise_df.empty:
        return
    required = {"b", "lv_id", "seed_a", "seed_b", "spearman_rho"}
    if not required.issubset(pairwise_df.columns):
        return

    lvs = sorted(pairwise_df["lv_id"].dropna().astype(str).unique().tolist())
    if not lvs:
        return
    cmap = plt.get_cmap("tab10")
    lv_colors = {lv: cmap(i % 10) for i, lv in enumerate(lvs)}

    def _seed_pair_key(token: str) -> tuple[int, int]:
        left, right = token.split("-")
        return int(left), int(right)

    for b, group in pairwise_df.groupby("b", dropna=False):
        work = group.copy()
        if work.empty:
            continue
        work["seed_pair"] = (
            work["seed_a"].astype(int).astype(str)
            + "-"
            + work["seed_b"].astype(int).astype(str)
        )
        pair_order = sorted(work["seed_pair"].unique().tolist(), key=_seed_pair_key)
        pair_to_x = {pair: idx for idx, pair in enumerate(pair_order)}
        offsets = {
            lv: (idx - (len(lvs) - 1) / 2.0) * 0.12 for idx, lv in enumerate(lvs)
        }

        fig_w = max(8.0, 0.75 * len(pair_order))
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        for lv in lvs:
            sub = work[work["lv_id"].astype(str) == lv]
            if sub.empty:
                continue
            xs = [pair_to_x[p] + offsets[lv] for p in sub["seed_pair"].tolist()]
            ys = sub["spearman_rho"].to_numpy(dtype=float)
            ax.scatter(
                xs,
                ys,
                s=44,
                alpha=0.85,
                color=lv_colors[lv],
                label=lv,
            )

        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels(pair_order, rotation=45, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Seed pair (seed_a-seed_b)")
        ax.set_ylabel("Pairwise Spearman rho")
        ax.set_title(f"Metapath Pairwise Spearman by Seed Pair (B={int(b)})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="LV", frameon=False)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"metapath_pairwise_spearman_scatter_b{int(b)}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _plot_seed_reference_rank_scatter_by_lv_b(rank_df: pd.DataFrame, output_dir: Path) -> None:
    """
    For each (B, LV, target_set), plot metapath ranks for reference seed_a vs seed_b.

    - x-axis: rank in reference seed_a (lowest seed id available in that group)
    - y-axis: rank in each other seed_b
    - color: seed_b
    """
    required = {"b", "lv_id", "target_set_id", "seed", "metapath", "metapath_rank"}
    if rank_df.empty or not required.issubset(rank_df.columns):
        return

    def _sanitize(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))

    for (b, lv_id, target_set_id), group in rank_df.groupby(
        ["b", "lv_id", "target_set_id"], dropna=False
    ):
        seeds = sorted(group["seed"].dropna().astype(int).unique().tolist())
        if len(seeds) < 2:
            continue
        seed_a = int(seeds[0])
        ref = group[group["seed"].astype(int) == seed_a][["metapath", "metapath_rank"]].copy()
        ref = ref.rename(columns={"metapath_rank": "rank_a"})
        if ref.empty:
            continue

        others = [int(s) for s in seeds if int(s) != seed_a]
        cmap = plt.get_cmap("tab10")
        color_map = {seed_b: cmap(i % 10) for i, seed_b in enumerate(others)}
        rows = []
        for seed_b in others:
            sub = group[group["seed"].astype(int) == seed_b][["metapath", "metapath_rank"]].copy()
            sub = sub.rename(columns={"metapath_rank": "rank_b"})
            merged = ref.merge(sub, on="metapath", how="inner")
            if merged.empty:
                continue
            merged["seed_b"] = seed_b
            rows.append(merged)
        if not rows:
            continue
        plot_df = pd.concat(rows, ignore_index=True)

        max_rank = float(
            np.nanmax(
                np.concatenate(
                    [
                        plot_df["rank_a"].to_numpy(dtype=float),
                        plot_df["rank_b"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        lim_low, lim_high = 0.5, max_rank + 0.5
        fig_h = max(5.0, 0.2 * max_rank + 2.0)
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        for seed_b in others:
            sub = plot_df[plot_df["seed_b"] == seed_b]
            if sub.empty:
                continue
            seed_token = f"{int(b)}|{lv_id}|{target_set_id}|{seed_a}|{seed_b}"
            jitter_seed = zlib.crc32(seed_token.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(jitter_seed)
            x_jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
            y_jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
            ax.scatter(
                sub["rank_a"].to_numpy(dtype=float) + x_jitter,
                sub["rank_b"].to_numpy(dtype=float) + y_jitter,
                s=35,
                alpha=0.8,
                color=color_map[seed_b],
                label=f"seed {seed_b}",
            )

        ax.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=1.0, alpha=0.4)
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_xlabel(f"Seed {seed_a} rank")
        ax.set_ylabel("Seed_b rank")
        ax.set_title(f"Metapath Rank Scatter (B={int(b)}, {lv_id}, {target_set_id})")
        ax.grid(alpha=0.25)
        ax.legend(title="Seed_b", frameon=False, fontsize=8)
        fig.tight_layout()
        out_name = (
            f"metapath_rank_scatter_ref_seed_b{int(b)}_"
            f"{_sanitize(str(lv_id))}_{_sanitize(str(target_set_id))}.png"
        )
        fig.savefig(output_dir / out_name, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_seed_reference_pair_rank_scatter_by_lv_b(
    top_pairs_runs_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    For each (B, LV, target_set), plot pair ranks for reference seed_a vs seed_b.

    Pair identity is defined by (metapath, target_id, gene_identifier).
    """
    required = {
        "b",
        "lv_id",
        "target_set_id",
        "seed",
        "metapath",
        "target_id",
        "gene_identifier",
        "pair_rank",
    }
    if top_pairs_runs_df.empty or not required.issubset(top_pairs_runs_df.columns):
        return

    def _sanitize(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))

    df = top_pairs_runs_df.copy()
    for col in ["metapath", "target_id", "gene_identifier"]:
        df[col] = df[col].astype(str)
    df["pair_rank"] = pd.to_numeric(df["pair_rank"], errors="coerce")
    df = df[df["pair_rank"].notna()].copy()
    if df.empty:
        return

    for (b, lv_id, target_set_id), group in df.groupby(
        ["b", "lv_id", "target_set_id"], dropna=False
    ):
        seeds = sorted(group["seed"].dropna().astype(int).unique().tolist())
        if len(seeds) < 2:
            continue
        seed_a = int(seeds[0])
        ref = group[group["seed"].astype(int) == seed_a][
            ["metapath", "target_id", "gene_identifier", "pair_rank"]
        ].copy()
        ref = ref.rename(columns={"pair_rank": "rank_a"})
        if ref.empty:
            continue

        others = [int(s) for s in seeds if int(s) != seed_a]
        cmap = plt.get_cmap("tab10")
        color_map = {seed_b: cmap(i % 10) for i, seed_b in enumerate(others)}
        rows = []
        for seed_b in others:
            sub = group[group["seed"].astype(int) == seed_b][
                ["metapath", "target_id", "gene_identifier", "pair_rank"]
            ].copy()
            sub = sub.rename(columns={"pair_rank": "rank_b"})
            merged = ref.merge(
                sub,
                on=["metapath", "target_id", "gene_identifier"],
                how="inner",
            )
            if merged.empty:
                continue
            merged["seed_b"] = seed_b
            rows.append(merged)
        if not rows:
            continue
        plot_df = pd.concat(rows, ignore_index=True)

        max_rank = float(
            np.nanmax(
                np.concatenate(
                    [
                        plot_df["rank_a"].to_numpy(dtype=float),
                        plot_df["rank_b"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        lim_low, lim_high = 0.5, max_rank + 0.5
        fig_h = max(5.0, 0.2 * max_rank + 2.0)
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        for seed_b in others:
            sub = plot_df[plot_df["seed_b"] == seed_b]
            if sub.empty:
                continue
            seed_token = f"{int(b)}|{lv_id}|{target_set_id}|pair|{seed_a}|{seed_b}"
            jitter_seed = zlib.crc32(seed_token.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(jitter_seed)
            x_jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
            y_jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
            ax.scatter(
                sub["rank_a"].to_numpy(dtype=float) + x_jitter,
                sub["rank_b"].to_numpy(dtype=float) + y_jitter,
                s=35,
                alpha=0.8,
                color=color_map[seed_b],
                label=f"seed {seed_b}",
            )

        ax.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=1.0, alpha=0.4)
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_xlabel(f"Seed {seed_a} pair rank")
        ax.set_ylabel("Seed_b pair rank")
        ax.set_title(f"Pair Rank Scatter (B={int(b)}, {lv_id}, {target_set_id})")
        ax.grid(alpha=0.25)
        ax.legend(title="Seed_b", frameon=False, fontsize=8)
        fig.tight_layout()
        out_name = (
            f"pair_rank_scatter_ref_seed_b{int(b)}_"
            f"{_sanitize(str(lv_id))}_{_sanitize(str(target_set_id))}.png"
        )
        fig.savefig(output_dir / out_name, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_seed_reference_path_rank_scatter_by_lv_b(
    top_paths_runs_df: pd.DataFrame, output_dir: Path
) -> None:
    """
    For each (B, LV, target_set), plot path ranks for reference seed_a vs seed_b.

    Path identity is defined by
    (metapath_g_orientation, target_id, gene_identifier, path_nodes_ids).
    """
    required = {
        "b",
        "lv_id",
        "target_set_id",
        "seed",
        "metapath_g_orientation",
        "target_id",
        "gene_identifier",
        "path_nodes_ids",
        "path_rank",
    }
    if top_paths_runs_df.empty or not required.issubset(top_paths_runs_df.columns):
        return

    def _sanitize(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))

    df = top_paths_runs_df.copy()
    for col in ["metapath_g_orientation", "target_id", "gene_identifier", "path_nodes_ids"]:
        df[col] = df[col].astype(str)
    df["path_rank"] = pd.to_numeric(df["path_rank"], errors="coerce")
    df = df[df["path_rank"].notna()].copy()
    if df.empty:
        return

    for (b, lv_id, target_set_id), group in df.groupby(
        ["b", "lv_id", "target_set_id"], dropna=False
    ):
        seeds = sorted(group["seed"].dropna().astype(int).unique().tolist())
        if len(seeds) < 2:
            continue
        seed_a = int(seeds[0])
        ref = group[group["seed"].astype(int) == seed_a][
            [
                "metapath_g_orientation",
                "target_id",
                "gene_identifier",
                "path_nodes_ids",
                "path_rank",
            ]
        ].copy()
        ref = ref.rename(columns={"path_rank": "rank_a"})
        if ref.empty:
            continue

        others = [int(s) for s in seeds if int(s) != seed_a]
        cmap = plt.get_cmap("tab10")
        color_map = {seed_b: cmap(i % 10) for i, seed_b in enumerate(others)}
        rows = []
        for seed_b in others:
            sub = group[group["seed"].astype(int) == seed_b][
                [
                    "metapath_g_orientation",
                    "target_id",
                    "gene_identifier",
                    "path_nodes_ids",
                    "path_rank",
                ]
            ].copy()
            sub = sub.rename(columns={"path_rank": "rank_b"})
            merged = ref.merge(
                sub,
                on=[
                    "metapath_g_orientation",
                    "target_id",
                    "gene_identifier",
                    "path_nodes_ids",
                ],
                how="inner",
            )
            if merged.empty:
                continue
            merged["seed_b"] = seed_b
            rows.append(merged)
        if not rows:
            continue
        plot_df = pd.concat(rows, ignore_index=True)

        max_rank = float(
            np.nanmax(
                np.concatenate(
                    [
                        plot_df["rank_a"].to_numpy(dtype=float),
                        plot_df["rank_b"].to_numpy(dtype=float),
                    ]
                )
            )
        )
        lim_low, lim_high = 0.5, max_rank + 0.5
        fig_h = max(5.0, 0.2 * max_rank + 2.0)
        fig, ax = plt.subplots(figsize=(7.0, fig_h))
        for seed_b in others:
            sub = plot_df[plot_df["seed_b"] == seed_b]
            if sub.empty:
                continue
            seed_token = f"{int(b)}|{lv_id}|{target_set_id}|path|{seed_a}|{seed_b}"
            jitter_seed = zlib.crc32(seed_token.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(jitter_seed)
            x_jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
            y_jitter = rng.normal(loc=0.0, scale=0.06, size=len(sub))
            ax.scatter(
                sub["rank_a"].to_numpy(dtype=float) + x_jitter,
                sub["rank_b"].to_numpy(dtype=float) + y_jitter,
                s=35,
                alpha=0.8,
                color=color_map[seed_b],
                label=f"seed {seed_b}",
            )

        ax.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=1.0, alpha=0.4)
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_xlabel(f"Seed {seed_a} path rank")
        ax.set_ylabel("Seed_b path rank")
        ax.set_title(f"Path Rank Scatter (B={int(b)}, {lv_id}, {target_set_id})")
        ax.grid(alpha=0.25)
        ax.legend(title="Seed_b", frameon=False, fontsize=8)
        fig.tight_layout()
        out_name = (
            f"path_rank_scatter_ref_seed_b{int(b)}_"
            f"{_sanitize(str(lv_id))}_{_sanitize(str(target_set_id))}.png"
        )
        fig.savefig(output_dir / out_name, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _analyze_and_write_outputs(
    metapath_runs_df: pd.DataFrame,
    top_pairs_runs_df: pd.DataFrame,
    top_paths_runs_df: pd.DataFrame,
    exp_dir: Path,
    top_k_metapaths: int,
    top_k_path_instances: int,
) -> None:
    rank_df = _rank_metapaths_within_run(metapath_runs_df)
    mp_pairwise_df, mp_summary_df = _metapath_pairwise_metrics(
        rank_df, top_k_metapaths=top_k_metapaths
    )
    selected_mp_df = _selected_metapath_jaccard(top_pairs_runs_df)
    selected_mp_summary = (
        selected_mp_df.groupby("b", as_index=False)["mean_selected_metapath_jaccard"]
        .mean()
        .sort_values("b")
        if not selected_mp_df.empty
        else pd.DataFrame(columns=["b", "mean_selected_metapath_jaccard"])
    )
    path_pairwise_df, path_summary_df = _path_pairwise_selection_metrics(
        top_paths_runs_df,
        top_k_path_instances=top_k_path_instances,
    )
    path_rank_detail_df, path_rank_summary_df = _path_rank_variability(
        top_paths_runs_df,
        top_k_path_instances=top_k_path_instances,
    )

    rank_df.to_csv(exp_dir / "metapath_rank_table.csv", index=False)
    mp_pairwise_df.to_csv(exp_dir / "metapath_pairwise_metrics.csv", index=False)
    mp_summary_df.to_csv(exp_dir / "metapath_stability_by_b.csv", index=False)
    selected_mp_df.to_csv(exp_dir / "selected_metapath_jaccard_detail.csv", index=False)
    selected_mp_summary.to_csv(
        exp_dir / "selected_metapath_jaccard_by_b.csv", index=False
    )
    path_pairwise_df.to_csv(exp_dir / "path_selection_pairwise_metrics.csv", index=False)
    path_summary_df.to_csv(exp_dir / "path_selection_stability_by_b.csv", index=False)
    path_rank_detail_df.to_csv(exp_dir / "path_rank_variability_detail.csv", index=False)
    path_rank_summary_df.to_csv(exp_dir / "path_rank_variability_by_b.csv", index=False)

    _plot_series(
        mp_summary_df,
        x_col="b",
        y_col="mean_spearman_rho",
        title="Metapath Rank Stability vs B",
        y_label="Mean pairwise Spearman rho",
        output_path=exp_dir / "metapath_spearman_vs_b.png",
    )
    _plot_series(
        mp_summary_df,
        x_col="b",
        y_col="mean_topk_jaccard",
        title=f"Metapath Top-{top_k_metapaths} Overlap vs B",
        y_label=f"Mean pairwise Jaccard (top-{top_k_metapaths})",
        output_path=exp_dir / "metapath_topk_jaccard_vs_b.png",
    )
    _plot_series(
        selected_mp_summary,
        x_col="b",
        y_col="mean_selected_metapath_jaccard",
        title="Selected Metapath Overlap vs B",
        y_label="Mean pairwise Jaccard (top-subgraph metapaths)",
        output_path=exp_dir / "selected_metapath_jaccard_vs_b.png",
    )
    _plot_series(
        path_summary_df,
        x_col="b",
        y_col="mean_path_jaccard",
        title=f"Path-Instance Selection Overlap vs B (top {top_k_path_instances})",
        y_label="Mean pairwise Jaccard",
        output_path=exp_dir / "path_selection_jaccard_vs_b.png",
    )
    _plot_series(
        path_rank_summary_df,
        x_col="b",
        y_col="mean_rank_sd",
        title=f"Shared Path Rank Variability vs B (top {top_k_path_instances})",
        y_label="Mean SD of instance rank",
        output_path=exp_dir / "path_rank_sd_vs_b.png",
    )
    _plot_metapath_pairwise_scatter_by_b(
        pairwise_df=mp_pairwise_df,
        output_dir=exp_dir,
    )
    _plot_seed_reference_rank_scatter_by_lv_b(
        rank_df=rank_df,
        output_dir=exp_dir,
    )
    _plot_seed_reference_pair_rank_scatter_by_lv_b(
        top_pairs_runs_df=top_pairs_runs_df,
        output_dir=exp_dir,
    )
    _plot_seed_reference_path_rank_scatter_by_lv_b(
        top_paths_runs_df=top_paths_runs_df,
        output_dir=exp_dir,
    )


def run_experiment(args: argparse.Namespace) -> None:
    root = _repo_root()
    _assert_lv_pipeline_files(root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    b_values = _parse_int_list(args.b_values)
    seeds = _parse_int_list(args.seeds)
    exp_dir = output_dir / "rank_stability_experiment"
    runs_dir = exp_dir / "runs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        print("[setup] Preparing compact LV workspace...")
        _prepare_workspace(args=args, root=root)

        pipeline_path = _pipeline_script(root)
        metapath_frames = []
        top_pairs_frames = []
        top_paths_frames = []
        qc_frames = []

        for b in b_values:
            for seed in seeds:
                run_id = f"b{b}_s{seed}"
                print(f"[experiment] Running B={b}, seed={seed}")

                null_cmd = _stage_cmd(args.python_exe, pipeline_path, "nulls", output_dir)
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

                stats_cmd = _stage_cmd(args.python_exe, pipeline_path, "stats", output_dir)
                stats_cmd.append("--force")
                _run(stats_cmd, cwd=root)

                if not args.skip_path_analysis:
                    subgraph_cmd = _stage_cmd(
                        args.python_exe, pipeline_path, "top-subgraphs", output_dir
                    )
                    subgraph_cmd.extend(
                        [
                            "--top-metapaths",
                            str(args.top_metapaths),
                            "--top-pairs",
                            str(args.top_pairs),
                            "--top-paths",
                            str(args.top_paths),
                            "--pair-rank-metric",
                            str(args.pair_rank_metric),
                            "--degree-d",
                            str(args.degree_d),
                            "--force",
                        ]
                    )
                    _run(subgraph_cmd, cwd=root)

                metapath_df, top_pairs_df, top_paths_df, qc_df = _collect_run_tables(
                    output_dir=output_dir,
                    b=int(b),
                    seed=int(seed),
                    run_id=run_id,
                )
                metapath_frames.append(metapath_df)
                if not top_pairs_df.empty:
                    top_pairs_frames.append(top_pairs_df)
                if not top_paths_df.empty:
                    top_paths_frames.append(top_paths_df)
                if not qc_df.empty:
                    qc_frames.append(qc_df)

                if args.copy_run_artifacts:
                    run_out = runs_dir / run_id
                    run_out.mkdir(parents=True, exist_ok=True)
                    for src in [
                        output_dir / "lv_metapath_results.csv",
                        output_dir / "top_pairs.csv",
                        output_dir / "top_paths.csv",
                        output_dir / "null_sampling_qc.csv",
                    ]:
                        if src.exists():
                            shutil.copy2(src, run_out / src.name)

        metapath_runs_df = pd.concat(metapath_frames, ignore_index=True)
        top_pairs_runs_df = (
            pd.concat(top_pairs_frames, ignore_index=True)
            if top_pairs_frames
            else pd.DataFrame(
                columns=sorted(REQUIRED_TOP_PAIRS_COLUMNS | {"b", "seed", "run_id"})
            )
        )
        top_paths_runs_df = (
            pd.concat(top_paths_frames, ignore_index=True)
            if top_paths_frames
            else pd.DataFrame(
                columns=sorted(REQUIRED_TOP_PATHS_COLUMNS | {"b", "seed", "run_id"})
            )
        )
        qc_runs_df = (
            pd.concat(qc_frames, ignore_index=True) if qc_frames else pd.DataFrame()
        )

        metapath_runs_df.to_csv(exp_dir / "metapath_runs.csv", index=False)
        top_pairs_runs_df.to_csv(exp_dir / "top_pairs_runs.csv", index=False)
        top_paths_runs_df.to_csv(exp_dir / "top_paths_runs.csv", index=False)
        if not qc_runs_df.empty:
            qc_runs_df.to_csv(exp_dir / "null_sampling_qc_runs.csv", index=False)
    else:
        metapath_path = exp_dir / "metapath_runs.csv"
        top_pairs_path = exp_dir / "top_pairs_runs.csv"
        top_paths_path = exp_dir / "top_paths_runs.csv"
        if not metapath_path.exists():
            raise FileNotFoundError(
                f"Analyze-only mode requires existing file: {metapath_path}"
            )
        metapath_runs_df = pd.read_csv(metapath_path)
        top_pairs_runs_df = (
            pd.read_csv(top_pairs_path) if top_pairs_path.exists() else pd.DataFrame()
        )
        top_paths_runs_df = (
            pd.read_csv(top_paths_path) if top_paths_path.exists() else pd.DataFrame()
        )

    _analyze_and_write_outputs(
        metapath_runs_df=metapath_runs_df,
        top_pairs_runs_df=top_pairs_runs_df,
        top_paths_runs_df=top_paths_runs_df,
        exp_dir=exp_dir,
        top_k_metapaths=int(args.top_k_metapaths),
        top_k_path_instances=int(args.top_k_path_instances),
    )

    print("\n[done] LV rank-stability experiment complete.")
    print(f"  Workspace: {output_dir}")
    print(f"  Experiment outputs: {exp_dir}")
    print(f"  B values: {b_values}")
    print(f"  Seeds: {seeds}")
    print(
        "  Features analyzed: "
        f"{metapath_runs_df[['lv_id', 'target_set_id', 'metapath']].drop_duplicates().shape[0]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-B LV null sweeps and quantify metapath/path rank stability."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="output/lv_rank_stability_exp",
        help="LV workspace/output directory for this experiment.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to invoke LV pipeline stages.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip new pipeline runs and only analyze existing experiment CSVs.",
    )
    parser.add_argument(
        "--copy-run-artifacts",
        action="store_true",
        help="Copy per-run CSV artifacts into rank_stability_experiment/runs/.",
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
        "--skip-path-analysis",
        action="store_true",
        help="Skip top-subgraphs stage and path-level stability metrics.",
    )
    parser.add_argument(
        "--top-metapaths",
        type=int,
        default=10,
        help="Top supported metapaths per LV-target for subgraph extraction.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="Top gene-target pairs per selected metapath.",
    )
    parser.add_argument(
        "--top-paths",
        type=int,
        default=5,
        help="Top path instances per selected pair.",
    )
    parser.add_argument(
        "--pair-rank-metric",
        choices=["dwpc", "contrast_min", "contrast_perm", "contrast_rand", "contrast_mean"],
        default="contrast_min",
        help=(
            "Metric for selecting top pairs before path extraction. "
            "Default `contrast_min` ranks by the conservative minimum of "
            "(dwpc - perm_null_mean, dwpc - rand_null_mean)."
        ),
    )
    parser.add_argument(
        "--degree-d",
        type=float,
        default=0.5,
        help="Degree damping for path instance extraction.",
    )
    parser.add_argument(
        "--top-k-metapaths",
        type=int,
        default=10,
        help="Top-k threshold for metapath overlap stability metrics.",
    )
    parser.add_argument(
        "--top-k-path-instances",
        type=int,
        default=20,
        help="Top-k path instances per group used for path selection overlap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args=args)


if __name__ == "__main__":
    main()
