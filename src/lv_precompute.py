"""
Precompute gene-by-feature score matrices for optimized LV analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dwpc_direct import HetMat, get_dwpc_raw_mean, reverse_metapath_abbrev
from src.lv_dwpc import (
    DIRECT_GENE_TO_TARGET_METAPATHS,
    NODE_TYPE_TO_ABBREV,
)


def _load_metapath_stats(stats_path: Path) -> pd.DataFrame:
    df = pd.read_csv(stats_path, sep="\t")
    required_cols = {"metapath", "length"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metapath stats missing required columns {sorted(missing)}: {stats_path}"
        )
    return df


def _select_gene_to_target_metapaths(
    stats_df: pd.DataFrame,
    target_abbrev: str,
    max_length: int,
    exclude_direct: bool,
    limit: int | None,
) -> list[str]:
    subset = stats_df[
        stats_df["metapath"].astype(str).str.startswith(target_abbrev)
        & stats_df["metapath"].astype(str).str.endswith("G")
        & (stats_df["length"] <= max_length)
    ].copy()
    if subset.empty:
        return []

    sort_cols = ["length"]
    asc = [True]
    if "n_pairs" in subset.columns:
        sort_cols.append("n_pairs")
        asc.append(False)
    subset = subset.sort_values(sort_cols, ascending=asc)

    gene_to_target = []
    seen = set()
    for metapath in subset["metapath"]:
        mp_g = reverse_metapath_abbrev(metapath)
        if mp_g in seen:
            continue
        seen.add(mp_g)
        gene_to_target.append(mp_g)

    if exclude_direct:
        gene_to_target = [
            metapath
            for metapath in gene_to_target
            if metapath not in DIRECT_GENE_TO_TARGET_METAPATHS
        ]
    if limit is not None and limit > 0:
        gene_to_target = gene_to_target[:limit]

    return gene_to_target


def _build_feature_manifest(
    target_sets: pd.DataFrame,
    stats_df: pd.DataFrame,
    max_metapath_length: int,
    exclude_direct: bool,
    metapath_limit_per_target: int | None,
) -> pd.DataFrame:
    rows = []
    dedup_target_sets = target_sets[
        ["target_set_id", "target_set_label", "node_type"]
    ].drop_duplicates()

    for _, target_meta in dedup_target_sets.iterrows():
        target_set_id = target_meta["target_set_id"]
        target_set_label = target_meta["target_set_label"]
        node_type = target_meta["node_type"]
        target_abbrev = NODE_TYPE_TO_ABBREV[node_type]
        metapaths = _select_gene_to_target_metapaths(
            stats_df=stats_df,
            target_abbrev=target_abbrev,
            max_length=max_metapath_length,
            exclude_direct=exclude_direct,
            limit=metapath_limit_per_target,
        )
        target_rows = target_sets[target_sets["target_set_id"] == target_set_id]
        target_ids = sorted(target_rows["target_id"].astype(str).tolist())
        n_targets = len(target_ids)
        target_id_list = "|".join(target_ids)

        for metapath in metapaths:
            rows.append(
                {
                    "target_set_id": target_set_id,
                    "target_set_label": target_set_label,
                    "node_type": node_type,
                    "target_abbrev": target_abbrev,
                    "metapath": metapath,
                    "n_targets": n_targets,
                    "target_ids": target_id_list,
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No features were generated for precompute stage.")

    manifest = manifest.sort_values(
        ["target_set_id", "metapath"]
    ).reset_index(drop=True)
    manifest.insert(0, "feature_idx", np.arange(len(manifest), dtype=np.int64))
    return manifest


def _map_target_positions(
    hetmat: HetMat,
    target_sets: pd.DataFrame,
) -> dict[str, np.ndarray]:
    target_pos_by_set: dict[str, np.ndarray] = {}
    for target_set_id, group in target_sets.groupby("target_set_id", sort=True):
        node_type = group["node_type"].iloc[0]
        node_df = hetmat.get_nodes(node_type)
        id_to_idx = dict(zip(node_df["identifier"], node_df.index))

        wanted_ids = []
        for value in group["target_id"].tolist():
            coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(coerced):
                wanted_ids.append(str(value))
            else:
                wanted_ids.append(int(coerced))

        mapped = [id_to_idx.get(item) for item in wanted_ids]
        mapped = [idx for idx in mapped if idx is not None]
        if not mapped:
            raise ValueError(f"No targets mapped for target_set_id={target_set_id}")
        target_pos_by_set[target_set_id] = np.asarray(mapped, dtype=np.int64)

    return target_pos_by_set


def precompute_gene_feature_scores(
    output_dir: Path,
    data_dir: Path,
    metapath_stats_path: Path,
    damping: float = 0.5,
    max_metapath_length: int = 3,
    metapath_limit_per_target: int | None = None,
    n_workers_precompute: int = 4,
    include_direct_metapaths: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Precompute per-gene scores for each (target_set, metapath) feature.
    """
    output_dir = Path(output_dir)
    top_genes_path = output_dir / "lv_top_genes.csv"
    target_sets_path = output_dir / "target_sets.csv"
    lv_map_path = output_dir / "lv_target_map.csv"

    for path in (top_genes_path, target_sets_path, lv_map_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    top_genes = pd.read_csv(top_genes_path)
    target_sets = pd.read_csv(target_sets_path)
    lv_map = pd.read_csv(lv_map_path)

    stats_df = _load_metapath_stats(metapath_stats_path)
    feature_manifest = _build_feature_manifest(
        target_sets=target_sets,
        stats_df=stats_df,
        max_metapath_length=max_metapath_length,
        exclude_direct=not include_direct_metapaths,
        metapath_limit_per_target=metapath_limit_per_target,
    )

    hetmat = HetMat(data_dir=data_dir, damping=damping, use_disk_cache=True)
    gene_nodes = hetmat.get_nodes("Gene")
    gene_ids = gene_nodes["identifier"].to_numpy()
    n_genes = len(gene_ids)
    n_features = len(feature_manifest)

    # Warm cache for unique metapaths using existing parallel helper.
    unique_metapaths = feature_manifest["metapath"].drop_duplicates().tolist()
    hetmat.precompute_matrices(
        unique_metapaths,
        n_workers=max(1, int(n_workers_precompute)),
        show_progress=True,
    )

    target_pos_by_set = _map_target_positions(hetmat=hetmat, target_sets=target_sets)
    score_matrix = np.zeros((n_genes, n_features), dtype=np.float32)

    for row in feature_manifest.itertuples(index=False):
        feature_idx = int(row.feature_idx)
        target_positions = target_pos_by_set[row.target_set_id]
        dwpc_matrix = hetmat.compute_dwpc_matrix(row.metapath, damping=damping)
        raw_mean = get_dwpc_raw_mean(hetmat.metapath_stats, row.metapath)
        if raw_mean == 0:
            raise ValueError(f"raw mean is zero for metapath {row.metapath}")

        accum = np.zeros(n_genes, dtype=np.float64)
        for target_pos in target_positions:
            column = np.asarray(dwpc_matrix[:, int(target_pos)].todense()).ravel()
            transformed = np.arcsinh(column / raw_mean)
            accum += transformed
        score_matrix[:, feature_idx] = (accum / len(target_positions)).astype(np.float32)

    # Real LV feature means.
    gene_id_to_row = {}
    for idx, gene_id in enumerate(gene_ids):
        gene_id_to_row[str(int(gene_id))] = idx
        gene_id_to_row[str(gene_id)] = idx

    real_index_rows = []
    for row in top_genes.itertuples(index=False):
        token = str(row.gene_identifier)
        idx = gene_id_to_row.get(token)
        if idx is None:
            parsed = pd.to_numeric(pd.Series([row.gene_identifier]), errors="coerce").iloc[0]
            if not pd.isna(parsed):
                idx = gene_id_to_row.get(str(int(parsed)))
        if idx is None:
            continue
        real_index_rows.append(
            {
                "lv_id": row.lv_id,
                "gene_identifier": row.gene_identifier,
                "gene_symbol": row.gene_symbol,
                "loading": float(row.loading),
                "rank": float(row.rank),
                "gene_row_idx": int(idx),
            }
        )

    lv_real_gene_indices = pd.DataFrame(real_index_rows).sort_values(
        ["lv_id", "rank"]
    ).reset_index(drop=True)

    # Restrict to mapped LVs in lv_map.
    allowed_lvs = set(lv_map["lv_id"].astype(str).tolist())
    lv_real_gene_indices = lv_real_gene_indices[
        lv_real_gene_indices["lv_id"].astype(str).isin(allowed_lvs)
    ].copy()
    if lv_real_gene_indices.empty:
        raise ValueError("No LV real genes were mapped to score matrix rows.")

    real_mean_rows = []
    for lv_id, group in lv_real_gene_indices.groupby("lv_id", sort=True):
        idx = group["gene_row_idx"].to_numpy(dtype=np.int64)
        lv_mean = score_matrix[idx].mean(axis=0)
        lv_target_set = lv_map[lv_map["lv_id"] == lv_id]["target_set_id"].iloc[0]
        for feature in feature_manifest.itertuples(index=False):
            if feature.target_set_id != lv_target_set:
                continue
            real_mean_rows.append(
                {
                    "lv_id": lv_id,
                    "target_set_id": feature.target_set_id,
                    "target_set_label": feature.target_set_label,
                    "node_type": feature.node_type,
                    "feature_idx": int(feature.feature_idx),
                    "metapath": feature.metapath,
                    "real_mean": float(lv_mean[int(feature.feature_idx)]),
                    "n_real_genes": int(len(idx)),
                }
            )

    real_feature_scores = pd.DataFrame(real_mean_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "gene_feature_scores.npy", score_matrix)
    np.save(output_dir / "gene_ids.npy", gene_ids)
    feature_manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
    lv_real_gene_indices.to_csv(output_dir / "lv_real_gene_indices.csv", index=False)
    real_feature_scores.to_csv(output_dir / "real_feature_scores.csv", index=False)

    return feature_manifest, lv_real_gene_indices, real_feature_scores
