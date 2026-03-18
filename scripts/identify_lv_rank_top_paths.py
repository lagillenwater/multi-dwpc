#!/usr/bin/env python3
"""Extract top LV pairs and path instances for each rank-stability run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

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
    parser.add_argument("--top-pairs", type=int, default=10)
    parser.add_argument("--top-paths", type=int, default=5)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument("--pair-rank-metric", default="dwpc", choices=PAIR_RANK_CHOICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else analysis_dir.parent

    rank_df = _load_rank_table(analysis_dir / "metapath_rank_table.csv")
    selected = rank_df[rank_df["metapath_rank"].astype(int) <= int(args.top_metapaths_per_run)].copy()
    if selected.empty:
        raise ValueError("No metapaths selected after applying top-metapaths-per-run")

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
        "dwpc",
        "pair_rank_metric",
        "pair_rank_score",
        "pair_diff_perm",
        "pair_diff_rand",
        "pair_diff_min",
        "pair_diff_mean",
        "pair_rank",
    ]
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


if __name__ == "__main__":
    main()
