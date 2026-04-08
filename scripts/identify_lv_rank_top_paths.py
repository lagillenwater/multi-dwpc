#!/usr/bin/env python3
"""Extract top LV pairs and path instances for each rank-stability run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.dwpc_direct import reverse_metapath_abbrev  # noqa: E402
from src.lv_dwpc import compute_real_pair_dwpc  # noqa: E402
from src.lv_pairs import build_lv_target_pairs  # noqa: E402
from src.lv_subgraphs import EdgeLoader, _enumerate_paths, _load_node_maps, _parse_metapath  # noqa: E402


PAIR_RANK_CHOICES = ["dwpc", "contrast_min", "contrast_perm", "contrast_rand", "contrast_mean"]
METAPATH_RANK_CHOICES = [
    "metapath_rank",
    "consensus_score",
    "consensus_rank",
    "min_d",
    "min_diff",
    "diff_perm",
    "diff_rand",
    "d_perm",
    "d_rand",
]
ASCENDING_METRICS = {"consensus_rank", "metapath_rank", "p_perm_fdr", "p_rand_fdr"}


def _load_rank_table(path: Path) -> pd.DataFrame:
    required = {
        "b",
        "seed",
        "lv_id",
        "target_set_id",
        "target_set_label",
        "node_type",
        "metapath",
        "metapath_rank",
    }
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "control" not in df.columns:
        df = df.copy()
        df["control"] = "combined"
    if "run_id" not in df.columns:
        df = df.copy()
        df["run_id"] = (
            "b"
            + df["b"].astype(int).astype(str)
            + "_s"
            + df["seed"].astype(int).astype(str)
        )
    return df


def _load_runs_table(path: Path) -> pd.DataFrame:
    required = {
        "b",
        "seed",
        "lv_id",
        "target_set_id",
        "target_set_label",
        "node_type",
        "metapath",
        "control",
        "diff",
    }
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "run_id" not in df.columns:
        df = df.copy()
        df["run_id"] = (
            "b"
            + df["b"].astype(int).astype(str)
            + "_s"
            + df["seed"].astype(int).astype(str)
        )
    return df


def _prepare_pair_table(workspace_dir: Path) -> Path:
    pairs_path = workspace_dir / "lv_gene_target_pairs.csv"
    if pairs_path.exists():
        return pairs_path
    build_lv_target_pairs(
        top_genes_path=workspace_dir / "lv_top_genes.csv",
        target_sets_path=workspace_dir / "target_sets.csv",
        lv_target_map_path=workspace_dir / "lv_target_map.csv",
        output_pairs_path=pairs_path,
    )
    return pairs_path


def _add_pair_rank_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for required in ["perm_null_mean", "rand_null_mean"]:
        if required not in work.columns:
            work[required] = pd.NA
    work["pair_diff_perm"] = work["dwpc"] - work["perm_null_mean"]
    work["pair_diff_rand"] = work["dwpc"] - work["rand_null_mean"]
    work["pair_diff_min"] = work[["pair_diff_perm", "pair_diff_rand"]].min(axis=1)
    work["pair_diff_mean"] = work[["pair_diff_perm", "pair_diff_rand"]].mean(axis=1)
    return work


def _rank_score_col(metric: str) -> str:
    mapping = {
        "dwpc": "dwpc",
        "contrast_min": "pair_diff_min",
        "contrast_perm": "pair_diff_perm",
        "contrast_rand": "pair_diff_rand",
        "contrast_mean": "pair_diff_mean",
    }
    if metric not in mapping:
        raise ValueError(f"Unsupported pair-rank metric: {metric}")
    return mapping[metric]


def _effective_number(scores: pd.Series) -> float:
    vals = scores.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


def _effective_k(scores: pd.Series, *, min_n: int = 1, max_n: int | None = None) -> int:
    k = int(np.ceil(_effective_number(scores)))
    k = max(int(min_n), k)
    if max_n is not None:
        k = min(k, int(max_n))
    return k


def _build_consensus_runs_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    control_map = {"permuted": "perm", "random": "rand"}
    work = runs_df.copy()
    work["control"] = work["control"].astype(str)
    work = work[work["control"].isin(control_map)].copy()
    if work.empty:
        raise ValueError("No permuted/random rows found in all_runs_long.csv")

    index_cols = [
        col
        for col in [
            "b",
            "seed",
            "run_id",
            "lv_id",
            "target_set_id",
            "target_set_label",
            "node_type",
            "metapath",
        ]
        if col in work.columns
    ]
    metric_cols = [col for col in ["diff", "d", "p", "p_fdr", "null_mean_score", "real_mean_score"] if col in work.columns]
    if "diff" not in metric_cols:
        raise ValueError("all_runs_long.csv must contain a 'diff' column for consensus ranking")

    split_frames = []
    for control, prefix in control_map.items():
        sub = work[work["control"] == control].copy()
        if sub.empty:
            continue
        rename_map = {
            "diff": f"diff_{prefix}",
            "d": f"d_{prefix}",
            "p": f"p_{prefix}",
            "p_fdr": f"p_{prefix}_fdr",
            "null_mean_score": f"{prefix}_null_mean",
        }
        keep_cols = index_cols + [col for col in metric_cols if col in rename_map or col == "real_mean_score"]
        sub = sub[keep_cols].rename(columns=rename_map)
        split_frames.append(sub)

    if len(split_frames) < 2:
        raise ValueError("Need both permuted and random rows to build LV consensus ranking")

    merged = split_frames[0]
    for sub in split_frames[1:]:
        merged = merged.merge(sub, on=index_cols + [col for col in ["real_mean_score"] if col in merged.columns and col in sub.columns], how="inner")

    if merged.empty:
        raise ValueError("No shared LV-target-metapath rows remained after merging permuted and random runs")

    group_cols = [col for col in ["b", "seed", "run_id", "lv_id", "target_set_id"] if col in merged.columns]
    merged["rank_perm"] = (
        merged.groupby(group_cols)["diff_perm"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    merged["rank_rand"] = (
        merged.groupby(group_cols)["diff_rand"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    merged["consensus_rank"] = 0.5 * (merged["rank_perm"] + merged["rank_rand"])
    merged["consensus_score"] = 0.5 * (
        (1.0 / merged["rank_perm"].replace(0, np.nan).astype(float))
        + (1.0 / merged["rank_rand"].replace(0, np.nan).astype(float))
    )
    if "d_perm" in merged.columns and "d_rand" in merged.columns:
        merged["min_d"] = merged[["d_perm", "d_rand"]].min(axis=1)
    if "diff_perm" in merged.columns and "diff_rand" in merged.columns:
        merged["min_diff"] = merged[["diff_perm", "diff_rand"]].min(axis=1)
    if "p_perm_fdr" in merged.columns and "p_rand_fdr" in merged.columns:
        merged["supported"] = (
            (merged["p_perm_fdr"].astype(float) < 0.05)
            & (merged["p_rand_fdr"].astype(float) < 0.05)
            & (merged["diff_perm"].astype(float) > 0)
            & (merged["diff_rand"].astype(float) > 0)
        )
    else:
        merged["supported"] = pd.NA
    return merged


def _select_consensus_metapaths(
    consensus_df: pd.DataFrame,
    *,
    rank_metric: str,
    selection_method: str,
    top_n: int,
    effective_min_n: int,
    effective_max_n: int | None,
) -> pd.DataFrame:
    if rank_metric not in consensus_df.columns:
        raise ValueError(f"Requested metapath rank metric '{rank_metric}' not found in consensus runs table")

    group_cols = [col for col in ["b", "seed", "run_id", "lv_id", "target_set_id"] if col in consensus_df.columns]
    rows = []
    for _, group in consensus_df.groupby(group_cols, sort=True):
        work = group.copy()
        sort_ascending = rank_metric in ASCENDING_METRICS
        work = work.sort_values(
            [rank_metric, "consensus_score", "min_d", "min_diff", "metapath"],
            ascending=[sort_ascending, False, False, False, True],
        ).reset_index(drop=True)
        if selection_method == "effective_number":
            if rank_metric in {"consensus_rank", "metapath_rank"}:
                effective_scores = 1.0 / work[rank_metric].replace(0, np.nan).astype(float)
            else:
                effective_scores = pd.to_numeric(work[rank_metric], errors="coerce").fillna(0.0).clip(lower=0.0)
            eff_n = _effective_number(effective_scores)
            k = _effective_k(
                effective_scores,
                min_n=effective_min_n,
                max_n=effective_max_n,
            )
            selected = work.head(k).copy()
            selected["effective_n_all"] = float(eff_n)
            selected["selection_method"] = "effective_number"
        else:
            selected = work.head(int(top_n)).copy()
            selected["effective_n_all"] = pd.NA
            selected["selection_method"] = "top_n"
        selected["control"] = "combined"
        selected["metapath_rank_metric"] = str(rank_metric)
        selected["metapath_rank"] = range(1, len(selected) + 1)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _extract_paths(top_pairs_df: pd.DataFrame, degree_d: float, top_paths: int) -> pd.DataFrame:
    edge_loader = EdgeLoader(REPO_ROOT / "data" / "edges")
    rows = []
    for row in top_pairs_df.itertuples(index=False):
        metapath_src_to_gene = reverse_metapath_abbrev(str(row.metapath))
        nodes, edges = _parse_metapath(metapath_src_to_gene)
        maps = _load_node_maps(REPO_ROOT, list(dict.fromkeys(nodes)))
        target_id = str(row.target_id)
        gene_id = str(row.gene_identifier)
        source_pos = maps.id_to_pos[nodes[0]].get(target_id)
        gene_pos = maps.id_to_pos["G"].get(gene_id)
        if source_pos is None or gene_pos is None:
            continue
        paths = _enumerate_paths(
            source_pos=int(source_pos),
            gene_pos=int(gene_pos),
            nodes=nodes,
            edges=edges,
            edge_loader=edge_loader,
            top_k=top_paths,
            degree_d=degree_d,
        )
        for path_rank, (path_score, pos_path) in enumerate(paths, start=1):
            node_ids = []
            node_names = []
            for node_type, pos in zip(nodes, pos_path):
                node_id = maps.pos_to_id[node_type].get(int(pos))
                node_name = maps.id_to_name[node_type].get(str(node_id), str(node_id))
                node_ids.append(str(node_id))
                node_names.append(str(node_name))
            rows.append(
                {
                    "control": row.control,
                    "b": int(row.b),
                    "seed": int(row.seed),
                    "run_id": row.run_id,
                    "lv_id": row.lv_id,
                    "target_set_id": row.target_set_id,
                    "target_set_label": row.target_set_label,
                    "node_type": row.node_type,
                    "metapath": metapath_src_to_gene,
                    "metapath_g_orientation": row.metapath,
                    "target_id": row.target_id,
                    "target_name": row.target_name,
                    "gene_identifier": row.gene_identifier,
                    "gene_symbol": row.gene_symbol,
                    "pair_rank": int(row.pair_rank),
                    "path_rank": int(path_rank),
                    "path_score": float(path_score),
                    "path_nodes_ids": "|".join(node_ids),
                    "path_nodes_names": "|".join(node_names),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        default="output/lv_experiment/lv_rank_stability_experiment",
        help="Directory containing metapath_rank_table.csv",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Parent LV workspace directory containing lv_top_genes.csv/target_sets.csv/lv_target_map.csv",
    )
    parser.add_argument("--top-metapaths-per-run", type=int, default=2)
    parser.add_argument(
        "--metapath-selection-method",
        default="top_n",
        choices=["top_n", "effective_number"],
        help="How to choose metapaths per LV-target run.",
    )
    parser.add_argument(
        "--metapath-rank-metric",
        default="metapath_rank",
        choices=METAPATH_RANK_CHOICES,
        help="Metapath ranking metric. Use consensus_score/consensus_rank for dual-null consensus selection from all_runs_long.csv.",
    )
    parser.add_argument("--effective-min-n", type=int, default=1)
    parser.add_argument("--effective-max-n", type=int, default=None)
    parser.add_argument("--top-pairs", type=int, default=10)
    parser.add_argument("--top-paths", type=int, default=5)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument("--pair-rank-metric", default="dwpc", choices=PAIR_RANK_CHOICES)
    parser.add_argument("--b", type=int, default=None, help="Optional B filter.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter.")
    parser.add_argument("--control", default=None, help="Optional control filter when using metapath_rank.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else analysis_dir.parent

    if args.metapath_rank_metric == "metapath_rank":
        rank_df = _load_rank_table(analysis_dir / "metapath_rank_table.csv")
        selected = rank_df.copy()
        if args.control is not None:
            selected = selected[selected["control"].astype(str) == str(args.control)].copy()
        if args.b is not None:
            selected = selected[selected["b"].astype(int) == int(args.b)].copy()
        if args.seed is not None:
            selected = selected[selected["seed"].astype(int) == int(args.seed)].copy()
        selected = selected[selected["metapath_rank"].astype(int) <= int(args.top_metapaths_per_run)].copy()
        selected["metapath_rank_metric"] = "metapath_rank"
        selected["selection_method"] = "top_n"
        selected["effective_n_all"] = pd.NA
        if selected.empty:
            raise ValueError("No metapaths selected after applying metapath_rank filters")
    else:
        runs_df = _load_runs_table(analysis_dir / "all_runs_long.csv")
        if args.b is not None:
            runs_df = runs_df[runs_df["b"].astype(int) == int(args.b)].copy()
        if args.seed is not None:
            runs_df = runs_df[runs_df["seed"].astype(int) == int(args.seed)].copy()
        if runs_df.empty:
            raise ValueError("No all_runs_long rows remain after applying B/seed filters")
        consensus_df = _build_consensus_runs_table(runs_df)
        selected = _select_consensus_metapaths(
            consensus_df,
            rank_metric=str(args.metapath_rank_metric),
            selection_method=str(args.metapath_selection_method),
            top_n=int(args.top_metapaths_per_run),
            effective_min_n=int(args.effective_min_n),
            effective_max_n=args.effective_max_n,
        )
        if selected.empty:
            raise ValueError("No metapaths selected after applying consensus/effective-number selection")
        selected.to_csv(analysis_dir / "metapath_selection_runs.csv", index=False)

    pairs_path = _prepare_pair_table(workspace_dir)
    manifest_override = selected[["node_type", "metapath"]].drop_duplicates().copy()
    pair_dwpc, _ = compute_real_pair_dwpc(
        pairs_path=pairs_path,
        data_dir=REPO_ROOT / "data",
        metapath_stats_path=REPO_ROOT / "data" / "metapath-dwpc-stats.tsv",
        output_dwpc_path=analysis_dir / "lv_pair_dwpc_rank_selected.csv",
        output_metapath_manifest_path=analysis_dir / "lv_metapaths_rank_selected.csv",
        damping=0.5,
        max_length=3,
        exclude_direct=True,
        manifest_override=manifest_override,
        use_disk_cache=True,
    )

    merge_cols = ["lv_id", "target_set_id", "target_set_label", "node_type", "metapath"]
    keep_cols = merge_cols + [
        "control",
        "b",
        "seed",
        "run_id",
        "metapath_rank",
        "metapath_rank_metric",
        "selection_method",
        "effective_n_all",
        "consensus_score",
        "consensus_rank",
        "perm_null_mean",
        "rand_null_mean",
        "diff_perm",
        "diff_rand",
        "min_diff",
    ]
    keep_cols = [col for col in keep_cols if col in selected.columns]
    run_df = selected[keep_cols].drop_duplicates().copy()
    merged = pair_dwpc.merge(run_df, on=merge_cols, how="inner")
    if merged.empty:
        raise ValueError("No pair-level rows matched the selected run metapaths")

    merged = _add_pair_rank_metrics(merged)
    score_col = _rank_score_col(args.pair_rank_metric)
    merged["pair_rank_metric"] = str(args.pair_rank_metric)
    merged["pair_rank_score"] = merged[score_col]

    rank_group_cols = ["control", "b", "seed", "run_id", "lv_id", "target_set_id", "metapath"]
    merged = merged.sort_values(
        rank_group_cols + ["pair_rank_score", "dwpc", "gene_rank", "gene_identifier"],
        ascending=[True, True, True, True, True, True, True, False, False, True, True],
    )
    top_pairs_df = merged.groupby(rank_group_cols, as_index=False, group_keys=False).head(int(args.top_pairs)).copy()
    top_pairs_df["pair_rank"] = top_pairs_df.groupby(rank_group_cols).cumcount() + 1

    top_pairs_cols = [
        "control",
        "b",
        "seed",
        "run_id",
        "lv_id",
        "target_set_id",
        "target_set_label",
        "node_type",
        "target_id",
        "target_name",
        "target_position",
        "gene_identifier",
        "gene_symbol",
        "loading",
        "gene_rank",
        "source_idx",
        "target_idx",
        "metapath",
        "metapath_rank",
        "metapath_rank_metric",
        "selection_method",
        "effective_n_all",
        "consensus_score",
        "consensus_rank",
        "dwpc",
        "pair_rank_metric",
        "pair_rank_score",
        "pair_diff_perm",
        "pair_diff_rand",
        "pair_diff_min",
        "pair_diff_mean",
        "pair_rank",
    ]
    top_pairs_cols = [col for col in top_pairs_cols if col in top_pairs_df.columns]
    top_pairs_out = top_pairs_df[top_pairs_cols].copy()
    top_pairs_out.to_csv(analysis_dir / "top_pairs_runs.csv", index=False)

    top_paths_df = _extract_paths(top_pairs_out, degree_d=args.degree_d, top_paths=int(args.top_paths))
    if not top_paths_df.empty:
        top_paths_df = top_paths_df.sort_values(
            ["control", "b", "seed", "lv_id", "target_set_id", "metapath_g_orientation", "pair_rank", "path_rank"]
        ).reset_index(drop=True)
    top_paths_df.to_csv(analysis_dir / "top_paths_runs.csv", index=False)

    print(f"Saved top pairs: {analysis_dir / 'top_pairs_runs.csv'} ({len(top_pairs_out):,} rows)")
    print(f"Saved top paths: {analysis_dir / 'top_paths_runs.csv'} ({len(top_paths_df):,} rows)")
    if args.metapath_rank_metric != "metapath_rank":
        print(f"Saved metapath selection table: {analysis_dir / 'metapath_selection_runs.csv'}")


if __name__ == "__main__":
    main()
