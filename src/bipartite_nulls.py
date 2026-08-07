"""Generic bipartite null generators for replicate experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def degree_preserving_permutations(
    edge_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    n_permutations: int = 1,
    random_state: int = 42,
    n_swap_attempts_per_edge: int = 10,
) -> list[pd.DataFrame]:
    """Generate degree-preserving permutations by bipartite edge swaps."""
    work = edge_df.copy()
    work = work.drop_duplicates(subset=[source_col, target_col]).copy()
    if len(work) != len(edge_df):
        raise ValueError(
            "degree_preserving_permutations expects a binary edge list with no duplicate source-target pairs."
        )

    metadata_cols = [col for col in work.columns if col != target_col]
    source_meta = work[metadata_cols].drop_duplicates(subset=[source_col])
    unique_sources = work[source_col].drop_duplicates().tolist()
    if len(source_meta) != len(unique_sources):
        raise ValueError(
            f"Source metadata is not unique for {source_col}; expected {len(unique_sources)} rows but found {len(source_meta)}"
        )

    edge_list = list(work[[source_col, target_col]].itertuples(index=False, name=None))
    edge_set = set(edge_list)
    n_edges = len(edge_list)
    if n_edges == 0:
        return [work.copy() for _ in range(n_permutations)]

    permuted = []
    for perm_idx in range(n_permutations):
        rng = np.random.RandomState(random_state + perm_idx)
        perm_edges = list(edge_list)
        perm_edge_set = set(edge_set)
        n_attempts = max(1, int(n_swap_attempts_per_edge) * n_edges)

        for _ in range(n_attempts):
            idx_a, idx_b = rng.choice(n_edges, size=2, replace=False)
            src_a, tgt_a = perm_edges[idx_a]
            src_b, tgt_b = perm_edges[idx_b]
            if src_a == src_b or tgt_a == tgt_b:
                continue

            new_a = (src_a, tgt_b)
            new_b = (src_b, tgt_a)
            if new_a in perm_edge_set or new_b in perm_edge_set:
                continue

            perm_edge_set.remove((src_a, tgt_a))
            perm_edge_set.remove((src_b, tgt_b))
            perm_edge_set.add(new_a)
            perm_edge_set.add(new_b)
            perm_edges[idx_a] = new_a
            perm_edges[idx_b] = new_b

        perm_df = pd.DataFrame(perm_edges, columns=[source_col, target_col])
        perm_df = perm_df.merge(source_meta, on=source_col, how="left")
        perm_df = perm_df[edge_df.columns]
        permuted.append(perm_df)

    return permuted


def calculate_target_membership_counts(
    edge_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    target_universe: list[str] | list[int] | np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    counts = (
        edge_df[[source_col, target_col]]
        .drop_duplicates()
        .groupby(target_col)[source_col]
        .nunique()
        .reset_index(name="promiscuity")
    )
    if target_universe is None:
        return counts

    universe_df = pd.DataFrame({target_col: pd.Series(list(target_universe))})
    counts = universe_df.merge(counts, on=target_col, how="left")
    counts["promiscuity"] = counts["promiscuity"].fillna(0).astype(int)
    return counts


def _assign_rank_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Partition `values` into `n_bins` fixed strata by rank (ties broken by position)."""
    rank = values.rank(method="first")
    bins = pd.qcut(rank, q=n_bins, labels=False, duplicates="drop")
    return bins.astype(int)


def generate_promiscuity_matched_samples(
    edge_df: pd.DataFrame,
    all_annotations_df: pd.DataFrame,
    source_col: str,
    target_col: str,
    target_universe: list[str] | list[int] | np.ndarray | pd.Series | None = None,
    n_bins: int = 10,
    random_state: int = 42,
    include_match_metadata: bool = True,
) -> pd.DataFrame:
    """Fixed-bin stratified SRSWOR: replace each source's real targets with an
    equal-sized draw per promiscuity bin, from a shared bin partition of the
    target universe. Matches the sampling design HetNetEX/HetNetEx-MD's exact
    resampling-moment formulas assume (a fixed stratum-to-population mapping),
    unlike a per-target sliding tolerance window.
    """
    required = {source_col, target_col}
    missing = required - set(edge_df.columns)
    if missing:
        raise ValueError(f"edge_df missing required columns: {sorted(missing)}")
    missing = required - set(all_annotations_df.columns)
    if missing:
        raise ValueError(f"all_annotations_df missing required columns: {sorted(missing)}")

    source_meta_cols = [col for col in edge_df.columns if col != target_col]
    source_meta = edge_df[source_meta_cols].drop_duplicates(subset=[source_col])
    promiscuity_df = calculate_target_membership_counts(
        all_annotations_df,
        source_col=source_col,
        target_col=target_col,
        target_universe=target_universe,
    )
    promiscuity_df["bin_id"] = _assign_rank_bins(promiscuity_df["promiscuity"], n_bins)

    with_prom = edge_df.merge(promiscuity_df, on=target_col, how="left")
    with_prom["promiscuity"] = with_prom["promiscuity"].fillna(0).astype(int)

    source_target_sets = (
        edge_df.groupby(source_col)[target_col].apply(set).to_dict()
    )
    all_results = []

    for source_id, source_group in with_prom.groupby(source_col, sort=True):
        targets_in_source = source_target_sets[source_id]
        candidate_pool = promiscuity_df[
            ~promiscuity_df[target_col].isin(targets_in_source)
        ]
        seed = int(
            hashlib.md5(f"{random_state}_{source_id}".encode()).hexdigest(), 16
        ) % (2**32)
        rng = np.random.RandomState(seed)

        sampled_chunks = []
        for bin_id, bin_group in source_group.groupby("bin_id"):
            k = len(bin_group)
            bin_candidates = candidate_pool[candidate_pool["bin_id"] == bin_id]
            if len(bin_candidates) < k:
                raise ValueError(
                    f"Candidate pool too small in bin {bin_id} for source={source_id}: "
                    f"{len(bin_candidates)} candidates for {k} required targets."
                )
            chosen = rng.choice(len(bin_candidates), size=k, replace=False)
            sampled_chunks.append(bin_candidates.iloc[chosen])

        sampled = pd.concat(sampled_chunks, ignore_index=True)
        out = pd.DataFrame({
            source_col: source_id,
            target_col: sampled[target_col].to_numpy(),
        })
        if include_match_metadata:
            out["bin_id"] = sampled["bin_id"].to_numpy()
            out["sampled_promiscuity"] = sampled["promiscuity"].to_numpy(dtype=int)
        all_results.append(out)

    result = pd.concat(all_results, ignore_index=True)
    result = result.merge(source_meta, on=source_col, how="left")

    ordered = list(edge_df.columns)
    extra = [col for col in result.columns if col not in ordered]
    return result[ordered + extra]
