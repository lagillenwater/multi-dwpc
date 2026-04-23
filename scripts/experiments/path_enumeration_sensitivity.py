#!/usr/bin/env python3
"""Sweep path enumeration parameters (top_k, degree_d) and measure impact.

Tests how n_intermediates_found, sharing metrics, and wall time change
as a function of top_k and degree_d for representative (gene_set, metapath)
pairs.

Usage:
    python scripts/experiments/path_enumeration_sensitivity.py \
        --analysis-type lv \
        --lv-output-dirs output/lv_single_target_refactor \
        --b 10 \
        --output-dir output/path_enum_sensitivity

    # With specific pairs
    python scripts/experiments/path_enumeration_sensitivity.py \
        --analysis-type lv \
        --lv-output-dirs output/lv_single_target_refactor \
        --b 10 \
        --entity-metapath-pairs '{"LV246": ["GaDlAiD"]}' \
        --output-dir output/path_enum_sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.path_enumeration import EdgeLoader, load_node_maps, parse_metapath  # noqa: E402
from src.intermediate_sharing import (  # noqa: E402
    compute_intermediate_coverage,
    enumerate_gene_intermediates,
    load_dwpc_from_numpy,
    load_runs_at_b,
)
from src.lv_dwpc import NODE_TYPE_TO_ABBREV  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-type", choices=["lv", "year"], required=True)
    p.add_argument("--lv-output-dirs", nargs="+", required=True)
    p.add_argument("--b", type=int, required=True)
    p.add_argument("--top-k-values", default="50,100,500,1000")
    p.add_argument("--degree-d-values", default="0.3,0.5,0.7")
    p.add_argument("--z-threshold", type=float, default=0.5)
    p.add_argument("--entity-metapath-pairs", default=None, help='JSON dict: {"LV246": ["GaDlAiD"]}')
    p.add_argument("--max-pairs", type=int, default=10, help="Max (entity, metapath) pairs if auto-selecting")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = [int(x) for x in args.top_k_values.split(",")]
    degree_d_values = [float(x) for x in args.degree_d_values.split(",")]
    id_col = "lv_id" if args.analysis_type == "lv" else "go_id"
    lv_dirs = [Path(d) for d in args.lv_output_dirs]

    group_cols = (
        ["lv_id", "target_id", "target_name", "node_type", "metapath"]
        if args.analysis_type == "lv"
        else ["go_id", "metapath", "year"]
    )

    runs_frames = []
    for d in lv_dirs:
        subdir = "lv_rank_stability_experiment" if args.analysis_type == "lv" else "year_rank_stability_experiment"
        rp = d / subdir / "all_runs_long.csv"
        if rp.exists():
            runs_frames.append(load_runs_at_b(rp, args.b, group_cols))
    if not runs_frames:
        print("No all_runs_long.csv found")
        return
    results_df = pd.concat(runs_frames, ignore_index=True)
    selected = results_df[results_df["effect_size_z"] > args.z_threshold].copy()

    if args.entity_metapath_pairs:
        pairs_dict = json.loads(args.entity_metapath_pairs)
        pairs = [(eid, mp) for eid, mps in pairs_dict.items() for mp in mps]
    else:
        top = selected.nlargest(args.max_pairs, "effect_size_z")
        pairs = list(zip(top[id_col], top["metapath"]))

    if not pairs:
        print("No (entity, metapath) pairs to test")
        return

    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    all_node_types = set()
    for _, mp in pairs:
        nodes, _ = parse_metapath(mp)
        all_node_types.update(nodes)
    maps = load_node_maps(REPO_ROOT, list(all_node_types))

    top_genes_path = None
    for d in lv_dirs:
        candidate = d / ("lv_top_genes.csv" if args.analysis_type == "lv" else "go_top_genes.csv")
        if candidate.exists():
            top_genes_path = candidate
            break
    if top_genes_path is None:
        print("No top_genes.csv found")
        return
    top_genes_df = pd.read_csv(top_genes_path)

    n_combos = len(pairs) * len(top_k_values) * len(degree_d_values)
    print(f"Running {len(pairs)} pairs x {len(top_k_values)} top_k x {len(degree_d_values)} degree_d = {n_combos} combinations")

    rows = []
    for gs_id, metapath in pairs:
        gs_genes = top_genes_df[top_genes_df[id_col] == gs_id]["gene_identifier"].astype(int).tolist()
        if not gs_genes:
            continue

        gs_row = selected[(selected[id_col] == gs_id) & (selected["metapath"] == metapath)]
        if gs_row.empty:
            continue

        target_pos = None
        if args.analysis_type == "lv" and "target_id" in gs_row.columns:
            tid = gs_row["target_id"].iloc[0]
            nt = gs_row["node_type"].iloc[0] if "node_type" in gs_row.columns else "BP"
            nt_key = NODE_TYPE_TO_ABBREV.get(nt, nt)
            target_pos = maps.id_to_pos.get(nt_key, {}).get(tid)
        else:
            target_pos = maps.id_to_pos.get("BP", {}).get(gs_id)

        if target_pos is None:
            continue

        for top_k in top_k_values:
            for deg_d in degree_d_values:
                t0 = time.time()
                gene_ints, _ = enumerate_gene_intermediates(
                    gs_genes, target_pos, metapath, edge_loader, maps,
                    path_top_k=top_k, degree_d=deg_d,
                    gene_set_id=gs_id,
                )
                elapsed = time.time() - t0

                n_with_paths = len(gene_ints)
                all_ints = set()
                for v in gene_ints.values():
                    all_ints.update(v)

                coverage_stats, _ = compute_intermediate_coverage(gene_ints) if n_with_paths > 0 else ({}, [])

                rows.append({
                    "gene_set_id": gs_id,
                    "metapath": metapath,
                    "top_k": top_k,
                    "degree_d": deg_d,
                    "n_genes_total": len(gs_genes),
                    "n_genes_with_paths": n_with_paths,
                    "n_unique_intermediates": len(all_ints),
                    "top1_coverage": coverage_stats.get("top1_intermediate_coverage", 0),
                    "top5_coverage": coverage_stats.get("top5_intermediate_coverage", 0),
                    "wall_time_s": elapsed,
                })

        print(f"  {gs_id} / {metapath}: done")

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "path_enumeration_sensitivity.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(result_df)} rows)")


if __name__ == "__main__":
    main()
