"""
Precompute gene-by-feature score matrices for optimized LV analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.dwpc_direct import (
    HetMat,
    get_dwpc_raw_mean,
    load_metapath_stats,
    reverse_metapath_abbrev,
)
from src.lv_dwpc import (
    DIRECT_GENE_TO_TARGET_METAPATHS,
    NODE_TYPE_TO_ABBREV,
)

SCORES_MEMMAP_FILENAME = "gene_feature_scores.f32.memmap"
SCORES_MEMMAP_META_FILENAME = "gene_feature_scores.memmap.meta.json"
PRECOMPUTE_PROGRESS_FILENAME = "precompute_completed_feature_idx.npy"


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
    # Try target->G metapaths first (e.g., BPpG for Biological Process)
    subset = stats_df[
        stats_df["metapath"].astype(str).str.startswith(target_abbrev)
        & stats_df["metapath"].astype(str).str.endswith("G")
        & (stats_df["length"] <= max_length)
    ].copy()

    # If no target->G metapaths found, try G->target metapaths (e.g., GbCcSE for Side Effect)
    # Some node types like SE only have metapaths stored in G->target direction
    if subset.empty:
        subset = stats_df[
            stats_df["metapath"].astype(str).str.startswith("G")
            & stats_df["metapath"].astype(str).str.endswith(target_abbrev)
            & (stats_df["length"] <= max_length)
        ].copy()
        # These are already in G->target format, no need to reverse later
        already_gene_to_target = True
    else:
        already_gene_to_target = False

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
        if already_gene_to_target:
            mp_g = metapath  # Already in G->target format
        else:
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


def prepare_feature_manifest(
    output_dir: Path,
    metapath_stats_path: Path,
    max_metapath_length: int = 3,
    metapath_limit_per_target: int | None = None,
    include_direct_metapaths: bool = False,
) -> pd.DataFrame:
    """
    Build and persist the LV feature manifest without computing score matrices.
    """
    output_dir = Path(output_dir)
    target_sets_path = output_dir / "target_sets.csv"
    lv_map_path = output_dir / "lv_target_map.csv"

    for path in (target_sets_path, lv_map_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    target_sets = pd.read_csv(target_sets_path)
    stats_df = _load_metapath_stats(metapath_stats_path)
    feature_manifest = _build_feature_manifest(
        target_sets=target_sets,
        stats_df=stats_df,
        max_metapath_length=max_metapath_length,
        exclude_direct=not include_direct_metapaths,
        metapath_limit_per_target=metapath_limit_per_target,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = output_dir / "feature_manifest.csv"
    if feature_manifest_path.exists():
        existing_manifest = pd.read_csv(feature_manifest_path)
        if not existing_manifest.equals(feature_manifest):
            raise ValueError(
                "Existing feature_manifest.csv does not match current configuration. "
                "Use a new output directory or remove old precompute artifacts."
            )
    feature_manifest.to_csv(feature_manifest_path, index=False)
    return feature_manifest


def list_feature_metapaths(output_dir: Path) -> list[str]:
    feature_manifest_path = Path(output_dir) / "feature_manifest.csv"
    if not feature_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_manifest_path}. Prepare LV metadata before listing metapaths."
        )
    feature_manifest = pd.read_csv(feature_manifest_path, usecols=["metapath"])
    return sorted(feature_manifest["metapath"].astype(str).unique().tolist())


def warmup_feature_metapath_cache(
    output_dir: Path,
    data_dir: Path,
    metapath: str,
    damping: float = 0.5,
) -> None:
    """
    Compute and persist the shared DWPC cache entry for one LV feature metapath.
    """
    feature_manifest_path = Path(output_dir) / "feature_manifest.csv"
    if not feature_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {feature_manifest_path}. Prepare LV metadata before warming cache."
        )
    feature_manifest = pd.read_csv(feature_manifest_path, usecols=["metapath"])
    valid_metapaths = set(feature_manifest["metapath"].astype(str).tolist())
    if str(metapath) not in valid_metapaths:
        available = "\n  - ".join(sorted(valid_metapaths))
        raise ValueError(
            f"Unknown LV warmup metapath: {metapath}\nAvailable metapaths:\n  - {available}"
        )

    hetmat = HetMat(
        data_dir=data_dir,
        damping=damping,
        use_disk_cache=True,
        write_disk_cache=True,
    )
    _ = hetmat.compute_dwpc_matrix_csc(str(metapath), damping=damping)
    hetmat.clear_metapath_from_memory(metapath=str(metapath), damping=damping)


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


def _mean_transformed_scores_for_targets(
    dwpc_matrix_csc: sparse.csc_matrix,
    target_positions: np.ndarray,
    raw_mean: float,
    n_genes: int,
) -> np.ndarray:
    """
    Compute per-gene mean arcsinh(DWPC/raw_mean) across target columns.

    This avoids repeated dense column materialization by working in sparse form.
    """
    if len(target_positions) == 0:
        return np.zeros(n_genes, dtype=np.float64)

    sub = dwpc_matrix_csc[:, target_positions]
    if sub.nnz == 0:
        return np.zeros(n_genes, dtype=np.float64)

    transformed = sub.copy()
    transformed.data = np.arcsinh(transformed.data / raw_mean)
    row_sum = np.asarray(transformed.sum(axis=1)).ravel()
    return row_sum / float(len(target_positions))


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_completed_atomic(path: Path, completed: set[int]) -> None:
    arr = np.asarray(sorted(completed), dtype=np.int64)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, arr)
    tmp_path.replace(path)


def _load_completed(path: Path, n_features: int) -> set[int]:
    if not path.exists():
        return set()
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.int64).ravel()
    return {int(x) for x in arr.tolist() if 0 <= int(x) < int(n_features)}


def _prepare_score_memmap(
    output_dir: Path,
    n_genes: int,
    n_features: int,
) -> tuple[np.memmap, set[int], Path, Path, Path]:
    score_memmap_path = output_dir / SCORES_MEMMAP_FILENAME
    score_meta_path = output_dir / SCORES_MEMMAP_META_FILENAME
    progress_path = output_dir / PRECOMPUTE_PROGRESS_FILENAME

    resume_ready = (
        score_memmap_path.exists()
        and score_meta_path.exists()
        and progress_path.exists()
    )
    if resume_ready:
        meta = json.loads(score_meta_path.read_text(encoding="utf-8"))
        valid_shape = (
            int(meta.get("n_genes", -1)) == int(n_genes)
            and int(meta.get("n_features", -1)) == int(n_features)
            and str(meta.get("dtype", "")) == "float32"
        )
        if valid_shape:
            score_matrix = np.memmap(
                score_memmap_path,
                mode="r+",
                dtype=np.float32,
                shape=(n_genes, n_features),
            )
            completed = _load_completed(progress_path, n_features=n_features)
            return score_matrix, completed, score_memmap_path, score_meta_path, progress_path

    score_matrix = np.memmap(
        score_memmap_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_genes, n_features),
    )
    score_matrix[:] = 0.0
    score_matrix.flush()
    _write_json_atomic(
        score_meta_path,
        {
            "n_genes": int(n_genes),
            "n_features": int(n_features),
            "dtype": "float32",
        },
    )
    completed: set[int] = set()
    _write_completed_atomic(progress_path, completed)
    return score_matrix, completed, score_memmap_path, score_meta_path, progress_path


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

    feature_manifest = prepare_feature_manifest(
        output_dir=output_dir,
        metapath_stats_path=metapath_stats_path,
        max_metapath_length=max_metapath_length,
        metapath_limit_per_target=metapath_limit_per_target,
        include_direct_metapaths=include_direct_metapaths,
    )

    # Read existing DWPC disk cache entries but do not write new files.
    # This preserves speedup from existing caches without unbounded disk growth.
    hetmat = HetMat(
        data_dir=data_dir,
        damping=damping,
        use_disk_cache=True,
        write_disk_cache=False,
    )
    gene_nodes = hetmat.get_nodes("Gene")
    gene_ids = gene_nodes["identifier"].to_numpy()
    n_genes = len(gene_ids)
    n_features = len(feature_manifest)
    _ = n_workers_precompute  # retained for CLI compatibility in streamed precompute mode

    target_pos_by_set = _map_target_positions(hetmat=hetmat, target_sets=target_sets)
    score_matrix, completed, score_memmap_path, score_meta_path, progress_path = (
        _prepare_score_memmap(output_dir=output_dir, n_genes=n_genes, n_features=n_features)
    )
    if completed:
        print(
            f"Resuming precompute from checkpoint: "
            f"{len(completed)}/{n_features} features already complete."
        )

    for metapath, mp_features in feature_manifest.groupby("metapath", sort=False):
        pending_rows = [
            row for row in mp_features.itertuples(index=False)
            if int(row.feature_idx) not in completed
        ]
        if not pending_rows:
            continue

        # Use CSC once per metapath for repeated fast column slicing.
        dwpc_matrix_csc = hetmat.compute_dwpc_matrix_csc(metapath, damping=damping)
        raw_mean = get_dwpc_raw_mean(hetmat.metapath_stats, metapath)
        if raw_mean == 0:
            raise ValueError(f"raw mean is zero for metapath {metapath}")

        for row in pending_rows:
            feature_idx = int(row.feature_idx)
            target_positions = target_pos_by_set[row.target_set_id]
            mean_scores = _mean_transformed_scores_for_targets(
                dwpc_matrix_csc=dwpc_matrix_csc,
                target_positions=target_positions,
                raw_mean=raw_mean,
                n_genes=n_genes,
            )
            score_matrix[:, feature_idx] = mean_scores.astype(np.float32)
            completed.add(feature_idx)

        score_matrix.flush()
        _write_completed_atomic(progress_path, completed)
        # Stream by metapath: free matrix memory before the next metapath.
        hetmat.clear_metapath_from_memory(metapath=metapath, damping=damping)

    if len(completed) != int(n_features):
        raise RuntimeError(
            f"Precompute incomplete: {len(completed)}/{n_features} feature columns filled."
        )

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

    np.save(output_dir / "gene_feature_scores.npy", np.asarray(score_matrix))
    np.save(output_dir / "gene_ids.npy", gene_ids)
    lv_real_gene_indices.to_csv(output_dir / "lv_real_gene_indices.csv", index=False)
    real_feature_scores.to_csv(output_dir / "real_feature_scores.csv", index=False)

    try:
        del score_matrix
    except Exception:
        pass
    for path in (score_memmap_path, score_meta_path, progress_path):
        if path.exists():
            path.unlink()

    return feature_manifest, lv_real_gene_indices, real_feature_scores
