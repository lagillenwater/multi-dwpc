"""
Pair construction utilities for LV multi-DWPC analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_lv_target_pairs(
    top_genes_path: Path,
    target_sets_path: Path,
    lv_target_map_path: Path,
    output_pairs_path: Path,
) -> pd.DataFrame:
    """
    Build LV-specific gene-target pairs from top genes and target sets.

    Returns
    -------
    pd.DataFrame
        One row per (lv_id, gene_identifier, target_id) pair with metadata.
    """
    top_genes = pd.read_csv(top_genes_path)
    target_sets = pd.read_csv(target_sets_path)
    lv_map = pd.read_csv(lv_target_map_path)

    required_top = {"lv_id", "gene_identifier", "gene_symbol", "loading", "rank"}
    required_targets = {
        "target_set_id",
        "target_set_label",
        "node_type",
        "target_id",
        "target_name",
        "target_position",
    }
    required_map = {"lv_id", "target_set_id"}

    missing_top = required_top - set(top_genes.columns)
    missing_targets = required_targets - set(target_sets.columns)
    missing_map = required_map - set(lv_map.columns)
    if missing_top:
        raise ValueError(
            f"Missing columns in top genes file {top_genes_path}: {sorted(missing_top)}"
        )
    if missing_targets:
        raise ValueError(
            f"Missing columns in target sets file {target_sets_path}: {sorted(missing_targets)}"
        )
    if missing_map:
        raise ValueError(
            f"Missing columns in LV map file {lv_target_map_path}: {sorted(missing_map)}"
        )

    top_genes["gene_identifier"] = top_genes["gene_identifier"].astype(str)
    lv_genes = top_genes.merge(lv_map, on="lv_id", how="inner")
    if lv_genes.empty:
        raise ValueError("No LV genes matched LV-to-target mapping.")

    pairs = lv_genes.merge(target_sets, on="target_set_id", how="inner")
    if pairs.empty:
        raise ValueError("No target rows matched LV-to-target mapping.")

    pairs = pairs.rename(
        columns={
            "rank": "gene_rank",
        }
    )
    pairs = pairs.sort_values(
        ["lv_id", "target_set_id", "gene_rank", "target_id"]
    ).reset_index(drop=True)

    output_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_pairs_path, index=False)
    return pairs
