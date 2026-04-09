"""
DWPC computation for LV-target pair tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dwpc_direct import (
    HetMat,
    create_node_index_mapping,
    load_metapath_stats,
    reverse_metapath_abbrev,
)


NODE_TYPE_TO_ABBREV = {
    "Biological Process": "BP",
    "Anatomy": "A",
    "Disease": "D",
    "Side Effect": "SE",
}

DIRECT_GENE_TO_TARGET_METAPATHS = {"GpBP", "GdA", "GeA", "GuA", "GaD", "GdD", "GuD"}


def _load_metapath_stats(stats_path: Path) -> pd.DataFrame:
    if stats_path.exists():
        df = pd.read_csv(stats_path, sep="\t")
    else:
        # Keep LV stages robust by bootstrapping stats the same way HetMat does.
        try:
            df = load_metapath_stats(stats_path.parent)
        except Exception as exc:
            raise FileNotFoundError(
                f"Metapath stats file not found and auto-download failed: {stats_path}"
            ) from exc
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
    # stats are stored mostly as target->...->G; reverse to gene->...->target
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

    reversed_metapaths = []
    seen = set()
    for metapath in subset["metapath"]:
        reversed_mp = reverse_metapath_abbrev(metapath)
        if reversed_mp in seen:
            continue
        seen.add(reversed_mp)
        reversed_metapaths.append(reversed_mp)

    if exclude_direct:
        reversed_metapaths = [
            metapath
            for metapath in reversed_metapaths
            if metapath not in DIRECT_GENE_TO_TARGET_METAPATHS
        ]
    if limit is not None and limit > 0:
        reversed_metapaths = reversed_metapaths[:limit]
    return reversed_metapaths


def _map_pair_indices(hetmat: HetMat, pairs_df: pd.DataFrame) -> pd.DataFrame:
    mapped_frames = []
    for node_type, group in pairs_df.groupby("node_type", sort=True):
        local_group = group.copy()
        coerced_gene_ids = pd.to_numeric(
            local_group["gene_identifier"], errors="coerce"
        )
        if coerced_gene_ids.notna().all():
            local_group["gene_identifier"] = coerced_gene_ids.astype(np.int64)
        else:
            local_group["gene_identifier"] = local_group["gene_identifier"].astype(str)

        mapped = create_node_index_mapping(
            hetmat=hetmat,
            df=local_group,
            source_type="Gene",
            target_type=node_type,
            source_id_col="gene_identifier",
            target_id_col="target_id",
        )
        mapped_frames.append(mapped)

    out = pd.concat(mapped_frames, ignore_index=True)
    out["source_idx"] = pd.to_numeric(out["source_idx"], errors="coerce")
    out["target_idx"] = pd.to_numeric(out["target_idx"], errors="coerce")
    out = out.dropna(subset=["source_idx", "target_idx"]).copy()
    out["source_idx"] = out["source_idx"].astype(np.int64)
    out["target_idx"] = out["target_idx"].astype(np.int64)
    return out


def _build_metapath_manifest(
    stats_df: pd.DataFrame,
    target_node_types: list[str],
    max_length: int,
    exclude_direct: bool,
    limit_per_target: int | None,
) -> pd.DataFrame:
    records = []
    for node_type in sorted(target_node_types):
        target_abbrev = NODE_TYPE_TO_ABBREV[node_type]
        metapaths = _select_gene_to_target_metapaths(
            stats_df=stats_df,
            target_abbrev=target_abbrev,
            max_length=max_length,
            exclude_direct=exclude_direct,
            limit=limit_per_target,
        )
        for metapath in metapaths:
            records.append(
                {
                    "node_type": node_type,
                    "target_abbrev": target_abbrev,
                    "metapath": metapath,
                }
            )
    return pd.DataFrame(records)


def compute_real_pair_dwpc(
    pairs_path: Path,
    data_dir: Path,
    metapath_stats_path: Path,
    output_dwpc_path: Path,
    output_metapath_manifest_path: Path,
    damping: float = 0.5,
    max_length: int = 3,
    exclude_direct: bool = True,
    metapath_limit_per_target: int | None = None,
    manifest_override: pd.DataFrame | None = None,
    use_disk_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute DWPC for LV-target pairs across selected metapaths.
    """
    pairs = pd.read_csv(pairs_path)
    required_cols = {
        "lv_id",
        "target_set_id",
        "node_type",
        "target_id",
        "target_name",
        "target_position",
        "gene_identifier",
        "gene_symbol",
        "loading",
        "gene_rank",
    }
    missing_cols = required_cols - set(pairs.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns in pair file {pairs_path}: {sorted(missing_cols)}"
        )
    pairs["gene_identifier"] = pairs["gene_identifier"].astype(str)

    stats_df = _load_metapath_stats(metapath_stats_path)
    node_types = sorted(pairs["node_type"].dropna().unique().tolist())
    unsupported = [node_type for node_type in node_types if node_type not in NODE_TYPE_TO_ABBREV]
    if unsupported:
        raise ValueError(f"Unsupported node types for metapath selection: {unsupported}")

    if manifest_override is not None:
        manifest_df = manifest_override.copy()
        required_manifest_cols = {"node_type", "metapath"}
        missing_manifest = required_manifest_cols - set(manifest_df.columns)
        if missing_manifest:
            raise ValueError(
                "manifest_override missing required columns: "
                f"{sorted(missing_manifest)}"
            )
        manifest_df = manifest_df.drop_duplicates(subset=["node_type", "metapath"]).copy()
        if "target_abbrev" not in manifest_df.columns:
            manifest_df["target_abbrev"] = manifest_df["node_type"].map(NODE_TYPE_TO_ABBREV)
    else:
        manifest_df = _build_metapath_manifest(
            stats_df=stats_df,
            target_node_types=node_types,
            max_length=max_length,
            exclude_direct=exclude_direct,
            limit_per_target=metapath_limit_per_target,
        )
    if manifest_df.empty:
        raise ValueError(
            "No metapaths selected for current node types and filter settings."
        )

    hetmat = HetMat(data_dir=data_dir, damping=damping, use_disk_cache=use_disk_cache)
    indexed_pairs = _map_pair_indices(hetmat=hetmat, pairs_df=pairs)
    if indexed_pairs.empty:
        raise ValueError("All pair rows were dropped during source/target index mapping.")

    result_frames = []
    for node_type, group in indexed_pairs.groupby("node_type", sort=True):
        mp_list = (
            manifest_df.loc[manifest_df["node_type"] == node_type, "metapath"]
            .drop_duplicates()
            .tolist()
        )
        if not mp_list:
            continue

        src_idx = group["source_idx"].to_numpy(dtype=np.int64)
        tgt_idx = group["target_idx"].to_numpy(dtype=np.int64)
        base = group[
            [
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
            ]
        ].copy()

        for metapath in mp_list:
            dwpc_vals = hetmat.get_dwpc_for_pairs(
                metapath=metapath,
                source_indices=src_idx,
                target_indices=tgt_idx,
                damping=damping,
                transform=True,
            )
            frame = base.copy()
            frame["metapath"] = metapath
            frame["dwpc"] = dwpc_vals
            result_frames.append(frame)

    if not result_frames:
        raise ValueError("No DWPC result rows were generated.")

    dwpc_df = pd.concat(result_frames, ignore_index=True)
    dwpc_df = dwpc_df.sort_values(
        ["lv_id", "target_set_id", "metapath", "gene_rank", "target_id"]
    ).reset_index(drop=True)

    output_dwpc_path.parent.mkdir(parents=True, exist_ok=True)
    output_metapath_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dwpc_df.to_csv(output_dwpc_path, index=False)
    manifest_df.to_csv(output_metapath_manifest_path, index=False)

    return dwpc_df, manifest_df
