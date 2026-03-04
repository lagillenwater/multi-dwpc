"""
Vectorized null sampling with adaptive replicate counts and streaming stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class StreamingState:
    n_eff: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    exceed: np.ndarray
    d_samples: list[list[float]]
    overlap_sum: float = 0.0
    overlap_count: int = 0


NODE_ABBREVS = ["BP", "CC", "MF", "PW", "SE", "PC", "G", "A", "D", "C", "S"]


def _parse_edge_endpoints(edge_name: str) -> tuple[str, str]:
    nodes = []
    i = 0
    while i < len(edge_name):
        matched = False
        for ab in NODE_ABBREVS:
            if edge_name.startswith(ab, i):
                nodes.append(ab)
                i += len(ab)
                matched = True
                break
        if not matched:
            i += 1
    if len(nodes) < 2:
        raise ValueError(f"Cannot parse edge endpoints for '{edge_name}'")
    return nodes[0], nodes[-1]


def _compute_gene_degrees(data_dir: Path, gene_count: int) -> np.ndarray:
    edges_dir = data_dir / "edges"
    degree = np.zeros(gene_count, dtype=np.int64)

    for edge_path in sorted(edges_dir.glob("*.sparse.npz")):
        edge_name = edge_path.name.replace(".sparse.npz", "")
        src_type, dst_type = _parse_edge_endpoints(edge_name)
        mat = sparse.load_npz(edge_path).tocsr()

        if src_type == "G":
            row_nnz = np.asarray(mat.getnnz(axis=1)).astype(np.int64).ravel()
            degree[: len(row_nnz)] += row_nnz
        if dst_type == "G":
            col_nnz = np.asarray(mat.getnnz(axis=0)).astype(np.int64).ravel()
            degree[: len(col_nnz)] += col_nnz

    return degree


def _assign_degree_bins(
    degree: np.ndarray, n_bins: int
) -> tuple[np.ndarray, int]:
    rank = pd.Series(degree).rank(method="first")
    bins = pd.qcut(rank, q=n_bins, labels=False, duplicates="drop")
    if bins.isna().any():
        bins = bins.fillna(0)
    bin_ids = bins.astype(int).to_numpy()
    return bin_ids, int(bin_ids.max()) + 1


def _build_real_slots(
    lv_real_gene_indices: pd.DataFrame,
    degree_bins: np.ndarray,
) -> pd.DataFrame:
    df = lv_real_gene_indices.copy()
    df["gene_row_idx"] = pd.to_numeric(df["gene_row_idx"], errors="coerce").astype(int)
    df["degree_bin"] = df["gene_row_idx"].map(
        lambda idx: int(degree_bins[int(idx)])
    )
    df = df.sort_values(
        ["lv_id", "degree_bin", "rank", "gene_identifier"]
    ).reset_index(drop=True)
    df["slot_in_bin"] = df.groupby(["lv_id", "degree_bin"]).cumcount()
    df["slot_global"] = df.groupby(["lv_id"]).cumcount()
    return df


def _build_bin_counts(real_slots: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {}
    for lv_id, group in real_slots.groupby("lv_id", sort=True):
        counts = (
            group.groupby("degree_bin")["gene_row_idx"]
            .size()
            .astype(int)
            .to_dict()
        )
        out[str(lv_id)] = {int(k): int(v) for k, v in counts.items()}
    return out


def _sample_random_slots(
    rng: np.random.Generator,
    lv_id: str,
    bin_counts: Dict[str, Dict[int, int]],
    random_pool_rows: Dict[str, Dict[int, np.ndarray]],
    bin_to_gene_rows: Dict[int, np.ndarray],
) -> np.ndarray:
    slots = []
    for bin_id, count in bin_counts[lv_id].items():
        filtered = random_pool_rows[lv_id][bin_id]
        if len(filtered) < count:
            filtered = bin_to_gene_rows[bin_id]
        chosen = rng.choice(filtered, size=count, replace=False)
        slots.extend(chosen.tolist())
    return np.asarray(slots, dtype=np.int64)


def _sample_permuted_slots(
    rng: np.random.Generator,
    lv_id: str,
    lv_ids: list[str],
    bin_counts: Dict[str, Dict[int, int]],
    pooled_bin_rows: Dict[int, np.ndarray],
) -> np.ndarray:
    # Build per-bin allocations by shuffling pooled LV genes.
    slots = []
    for bin_id in sorted(pooled_bin_rows):
        pool = pooled_bin_rows[bin_id]
        shuffled = rng.permutation(pool)
        cursor = 0
        allocation = {}
        for other_lv in lv_ids:
            count = bin_counts[other_lv].get(bin_id, 0)
            allocation[other_lv] = shuffled[cursor: cursor + count]
            cursor += count
        chosen = allocation[lv_id]
        if len(chosen) > 0:
            slots.extend(chosen.tolist())
    return np.asarray(slots, dtype=np.int64)


def _init_streaming_state(n_features: int) -> StreamingState:
    return StreamingState(
        n_eff=np.zeros(n_features, dtype=np.int32),
        mean=np.zeros(n_features, dtype=np.float64),
        m2=np.zeros(n_features, dtype=np.float64),
        exceed=np.zeros(n_features, dtype=np.int32),
        d_samples=[[] for _ in range(n_features)],
    )


def _update_streaming(
    state: StreamingState,
    active_idx: np.ndarray,
    null_means: np.ndarray,
    real_means: np.ndarray,
    null_slot_scores: np.ndarray,
    real_slot_scores: np.ndarray,
) -> None:
    if len(active_idx) == 0:
        return

    prev_n = state.n_eff[active_idx].astype(np.float64)
    next_n = prev_n + 1.0
    delta = null_means - state.mean[active_idx]
    new_mean = state.mean[active_idx] + (delta / next_n)
    new_m2 = state.m2[active_idx] + delta * (null_means - new_mean)

    state.mean[active_idx] = new_mean
    state.m2[active_idx] = new_m2
    state.n_eff[active_idx] += 1
    state.exceed[active_idx] += (null_means >= real_means).astype(np.int32)

    diff = real_slot_scores - null_slot_scores
    diff_mean = diff.mean(axis=0)
    diff_std = diff.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_vals = diff_mean / diff_std
    d_vals = np.where(np.isfinite(d_vals), d_vals, np.nan)

    for local_col, feature_idx in enumerate(active_idx.tolist()):
        state.d_samples[feature_idx].append(float(d_vals[local_col]))


def _state_to_feature_rows(
    state: StreamingState,
    lv_id: str,
    null_type: str,
    feature_manifest: pd.DataFrame,
) -> list[dict]:
    rows = []
    for feature in feature_manifest.itertuples(index=False):
        idx = int(feature.feature_idx)
        n_eff = int(state.n_eff[idx])
        if n_eff <= 0:
            continue
        p = (1 + int(state.exceed[idx])) / (1 + n_eff)
        std = np.nan
        if n_eff > 1:
            std = float(np.sqrt(state.m2[idx] / (n_eff - 1)))
        d_values = np.asarray(state.d_samples[idx], dtype=np.float64)
        finite = d_values[np.isfinite(d_values)]
        if len(finite) == 0:
            d_mean = np.nan
            d_median = np.nan
            d_iqr = np.nan
        else:
            d_mean = float(np.mean(finite))
            d_median = float(np.median(finite))
            q75 = float(np.percentile(finite, 75))
            q25 = float(np.percentile(finite, 25))
            d_iqr = q75 - q25

        rows.append(
            {
                "lv_id": lv_id,
                "null_type": null_type,
                "feature_idx": idx,
                "target_set_id": feature.target_set_id,
                "target_set_label": feature.target_set_label,
                "node_type": feature.node_type,
                "metapath": feature.metapath,
                "n_eff": n_eff,
                "null_mean": float(state.mean[idx]),
                "null_std": std,
                "exceed_count": int(state.exceed[idx]),
                "p_empirical": p,
                "d_mean": d_mean,
                "d_median": d_median,
                "d_iqr": d_iqr,
            }
        )
    return rows


def run_vectorized_nulls(
    output_dir: Path,
    data_dir: Path,
    n_degree_bins: int = 10,
    b_min: int = 200,
    b_max: int = 1000,
    b_batch: int = 100,
    adaptive_p_low: float = 0.005,
    adaptive_p_high: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run degree-matched null sampling with adaptive replicate counts.
    """
    output_dir = Path(output_dir)
    feature_manifest = pd.read_csv(output_dir / "feature_manifest.csv")
    real_feature_scores = pd.read_csv(output_dir / "real_feature_scores.csv")
    lv_real_gene_indices = pd.read_csv(output_dir / "lv_real_gene_indices.csv")
    lv_map = pd.read_csv(output_dir / "lv_target_map.csv")

    score_matrix = np.load(output_dir / "gene_feature_scores.npy")
    gene_ids = np.load(output_dir / "gene_ids.npy")
    n_genes, n_features = score_matrix.shape
    if n_features != len(feature_manifest):
        raise ValueError("Feature manifest length does not match score matrix columns.")

    degree = _compute_gene_degrees(data_dir=data_dir, gene_count=n_genes)
    degree_bins, n_bins_eff = _assign_degree_bins(degree=degree, n_bins=n_degree_bins)

    degree_table = pd.DataFrame(
        {
            "gene_row_idx": np.arange(n_genes, dtype=np.int64),
            "gene_identifier": gene_ids,
            "degree": degree,
            "degree_bin": degree_bins,
        }
    )
    degree_table.to_csv(output_dir / "gene_degree_table.csv", index=False)

    real_slots = _build_real_slots(
        lv_real_gene_indices=lv_real_gene_indices, degree_bins=degree_bins
    )
    real_slots.to_csv(output_dir / "lv_real_slots.csv", index=False)

    bin_counts = _build_bin_counts(real_slots)
    lv_ids = sorted(real_slots["lv_id"].astype(str).unique().tolist())
    bin_to_gene_rows = {
        int(bin_id): group["gene_row_idx"].to_numpy(dtype=np.int64)
        for bin_id, group in degree_table.groupby("degree_bin")
    }
    real_bin_rows = {
        lv_id: {
            int(bin_id): group["gene_row_idx"].to_numpy(dtype=np.int64)
            for bin_id, group in real_slots[real_slots["lv_id"] == lv_id].groupby("degree_bin")
        }
        for lv_id in lv_ids
    }
    random_pool_rows = {
        lv_id: {
            bin_id: np.setdiff1d(
                bin_to_gene_rows[bin_id],
                real_bin_rows[lv_id].get(bin_id, np.asarray([], dtype=np.int64)),
                assume_unique=False,
            )
            for bin_id in bin_to_gene_rows
        }
        for lv_id in lv_ids
    }
    pooled_bin_rows = {
        int(bin_id): group["gene_row_idx"].to_numpy(dtype=np.int64)
        for bin_id, group in real_slots.groupby("degree_bin")
    }

    # Real references by LV.
    real_feature_by_lv = {
        lv_id: real_feature_scores[real_feature_scores["lv_id"] == lv_id]
        for lv_id in lv_ids
    }
    lv_to_target_set = dict(zip(lv_map["lv_id"].astype(str), lv_map["target_set_id"].astype(str)))
    allowed_features_by_lv = {}
    for lv_id in lv_ids:
        target_set_id = lv_to_target_set[lv_id]
        allowed = feature_manifest[feature_manifest["target_set_id"] == target_set_id][
            "feature_idx"
        ].to_numpy(dtype=np.int64)
        allowed_features_by_lv[lv_id] = allowed
    real_means_by_lv = {
        lv_id: np.full(n_features, np.nan, dtype=np.float64) for lv_id in lv_ids
    }
    for lv_id in lv_ids:
        for row in real_feature_by_lv[lv_id].itertuples(index=False):
            real_means_by_lv[lv_id][int(row.feature_idx)] = float(row.real_mean)

    real_slot_idx_by_lv = {}
    real_slot_scores_by_lv = {}
    for lv_id in lv_ids:
        rows = real_slots[real_slots["lv_id"] == lv_id].sort_values(
            ["degree_bin", "slot_in_bin"]
        )
        idx = rows["gene_row_idx"].to_numpy(dtype=np.int64)
        real_slot_idx_by_lv[lv_id] = idx
        real_slot_scores_by_lv[lv_id] = score_matrix[idx]

    rng = np.random.default_rng(random_seed)
    summary_rows = []
    qc_rows = []

    for null_type in ("random", "permuted"):
        for lv_id in lv_ids:
            state = _init_streaming_state(n_features=n_features)
            active = np.zeros(n_features, dtype=bool)
            active[allowed_features_by_lv[lv_id]] = True

            while active.any():
                active_idx = np.where(active)[0]
                if len(active_idx) == 0:
                    break

                for _ in range(int(b_batch)):
                    if null_type == "random":
                        sampled_idx = _sample_random_slots(
                            rng=rng,
                            lv_id=lv_id,
                            bin_counts=bin_counts,
                            random_pool_rows=random_pool_rows,
                            bin_to_gene_rows=bin_to_gene_rows,
                        )
                    else:
                        sampled_idx = _sample_permuted_slots(
                            rng=rng,
                            lv_id=lv_id,
                            lv_ids=lv_ids,
                            bin_counts=bin_counts,
                            pooled_bin_rows=pooled_bin_rows,
                        )

                    sampled_scores = score_matrix[sampled_idx][:, active_idx]
                    sampled_means = sampled_scores.mean(axis=0)
                    real_slot_scores = real_slot_scores_by_lv[lv_id][:, active_idx]
                    real_means = real_means_by_lv[lv_id][active_idx]

                    _update_streaming(
                        state=state,
                        active_idx=active_idx,
                        null_means=sampled_means,
                        real_means=real_means,
                        null_slot_scores=sampled_scores,
                        real_slot_scores=real_slot_scores,
                    )

                    real_idx_set = set(real_slot_idx_by_lv[lv_id].tolist())
                    overlap = len(real_idx_set.intersection(set(sampled_idx.tolist())))
                    state.overlap_sum += overlap / max(1, len(real_idx_set))
                    state.overlap_count += 1

                # Adaptive stopping.
                candidate = np.where(active & (state.n_eff >= int(b_min)))[0]
                if len(candidate) > 0:
                    p_vals = (1 + state.exceed[candidate]) / (1 + state.n_eff[candidate])
                    done_mask = (
                        (p_vals < adaptive_p_low)
                        | (p_vals > adaptive_p_high)
                        | (state.n_eff[candidate] >= int(b_max))
                    )
                    active[candidate[done_mask]] = False
                active[state.n_eff >= int(b_max)] = False

                if np.all(state.n_eff >= int(b_max)):
                    active[:] = False

            summary_rows.extend(
                _state_to_feature_rows(
                    state=state,
                    lv_id=lv_id,
                    null_type=null_type,
                    feature_manifest=feature_manifest,
                )
            )
            scored_feature_idx = allowed_features_by_lv[lv_id]
            scored_n_eff = state.n_eff[scored_feature_idx]
            qc_rows.append(
                {
                    "lv_id": lv_id,
                    "null_type": null_type,
                    # QC B-eff metrics should only summarize features that were
                    # actually scored for this LV's mapped target set.
                    "mean_b_eff": float(np.nanmean(scored_n_eff)),
                    "min_b_eff": int(np.nanmin(scored_n_eff)),
                    "max_b_eff": int(np.nanmax(scored_n_eff)),
                    "n_features_scored": int(len(scored_feature_idx)),
                    "mean_overlap_with_real": (
                        float(state.overlap_sum / state.overlap_count)
                        if state.overlap_count > 0
                        else np.nan
                    ),
                    "n_degree_bins_requested": int(n_degree_bins),
                    "n_degree_bins_effective": int(n_bins_eff),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["lv_id", "null_type", "feature_idx"]
    )
    qc_df = pd.DataFrame(qc_rows).sort_values(["lv_id", "null_type"])

    summary_df.to_csv(output_dir / "null_streaming_summary.csv", index=False)
    qc_df.to_csv(output_dir / "null_sampling_qc.csv", index=False)
    return summary_df, qc_df
