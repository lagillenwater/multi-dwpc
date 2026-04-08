"""General pathway-level coverage and multi-DWPC robustness analysis.

Compares multi-metapath (multi-DWPC) coverage against single-best-metapath
baselines at the pathway level.  Works for both year (GO-term) and LV
(latent-variable / target-set) experiments via column configuration.

Key concepts
------------
- **pathway group**: the unit of aggregation, e.g. (year, go_id) or
  (lv_id, target_set_id).
- **gene score**: max DWPC across selected metapaths and (optionally)
  target nodes for a given gene within a pathway group.
- **single-metapath baseline**: the same gene score computed using only
  the top-1 ranked metapath per pathway group.
- **rescue**: genes or targets with zero score under top-1 but nonzero
  score under the full multi-metapath set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CoverageConfig:
    """Column-mapping configuration for a specific experiment type."""

    # Columns that together define a pathway group
    group_cols: list[str]

    # Column containing gene identifiers
    gene_col: str

    # Column containing target-node identifiers (None if single-target, e.g. year/GO)
    target_col: str | None

    # Column containing DWPC values
    dwpc_col: str = "dwpc"

    # Column containing metapath abbreviations
    metapath_col: str = "metapath"

    # Column containing gene weights (e.g. loading); None means uniform
    weight_col: str | None = None

    # Columns to carry through to output for labeling (optional)
    label_cols: list[str] = field(default_factory=list)


def rank_metapaths_by_dwpc_mass(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
) -> pd.DataFrame:
    """Rank metapaths within each pathway group by total DWPC mass.

    Returns the input frame with an added ``metapath_rank`` column
    (1 = highest total DWPC within the group).
    """
    mass = (
        pair_dwpc.groupby(cfg.group_cols + [cfg.metapath_col], as_index=False)
        .agg(_dwpc_mass=(cfg.dwpc_col, "sum"))
    )
    mass["metapath_rank"] = (
        mass.groupby(cfg.group_cols)["_dwpc_mass"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return pair_dwpc.merge(
        mass[cfg.group_cols + [cfg.metapath_col, "metapath_rank"]],
        on=cfg.group_cols + [cfg.metapath_col],
        how="left",
    )


def rank_metapaths_from_results(
    pair_dwpc: pd.DataFrame,
    results_df: pd.DataFrame,
    cfg: CoverageConfig,
    *,
    rank_col: str = "consensus_score",
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank metapaths using an external results table (e.g. metapath_results or support).

    Parameters
    ----------
    results_df
        Must contain the group columns, metapath column, and *rank_col*.
    rank_col
        Column in *results_df* to rank by.
    ascending
        If True, lower values of *rank_col* rank higher (e.g. consensus_rank).
    """
    sub = results_df[cfg.group_cols + [cfg.metapath_col, rank_col]].drop_duplicates()
    sub = sub.copy()
    sub["metapath_rank"] = (
        sub.groupby(cfg.group_cols)[rank_col]
        .rank(method="first", ascending=ascending)
        .astype(int)
    )
    return pair_dwpc.merge(
        sub[cfg.group_cols + [cfg.metapath_col, "metapath_rank"]],
        on=cfg.group_cols + [cfg.metapath_col],
        how="left",
    )


# ---------------------------------------------------------------------------
# Core gene-score computation
# ---------------------------------------------------------------------------

def _gene_scores(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
    *,
    max_rank: int | None = None,
) -> pd.DataFrame:
    """Compute per-gene pathway score = max DWPC across metapaths (and targets).

    Parameters
    ----------
    max_rank
        If given, only metapaths with ``metapath_rank <= max_rank`` are used.
        Requires a ``metapath_rank`` column.
    """
    df = pair_dwpc
    if max_rank is not None:
        df = df[df["metapath_rank"] <= max_rank]
    return (
        df.groupby(cfg.group_cols + [cfg.gene_col], as_index=False)
        .agg(gene_score=(cfg.dwpc_col, "max"))
    )


# ---------------------------------------------------------------------------
# Pathway-level metrics
# ---------------------------------------------------------------------------

def _build_gene_frame(
    gene_scores_df: pd.DataFrame,
    gene_universe: pd.DataFrame | None,
    cfg: CoverageConfig,
) -> pd.DataFrame:
    """Merge gene scores with full gene universe, filling missing with 0."""
    if gene_universe is not None:
        # gene_universe must contain group_cols (or a subset) + gene_col + optional weight_col
        merge_keys = [c for c in cfg.group_cols if c in gene_universe.columns] + [cfg.gene_col]
        merged = gene_universe.merge(gene_scores_df, on=merge_keys, how="left")
        merged["gene_score"] = merged["gene_score"].fillna(0.0)
    else:
        merged = gene_scores_df.copy()
    merged["has_score"] = merged["gene_score"] > 0
    if cfg.weight_col and cfg.weight_col in merged.columns:
        merged["abs_weight"] = merged[cfg.weight_col].abs()
    return merged


def compute_pathway_metrics(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
    *,
    gene_universe: pd.DataFrame | None = None,
    max_rank: int | None = None,
    condition_label: str = "all",
) -> pd.DataFrame:
    """Compute pathway-level coverage metrics for a single condition.

    Parameters
    ----------
    gene_universe
        DataFrame of all genes per pathway group (with optional weight column).
        If None, the gene set is derived from pair_dwpc itself.
    max_rank
        Restrict to metapaths with rank <= max_rank.
    condition_label
        String label for this condition (e.g. "top1", "all", "top3").
    """
    gs = _gene_scores(pair_dwpc, cfg, max_rank=max_rank)
    gf = _build_gene_frame(gs, gene_universe, cfg)

    has_weights = cfg.weight_col and "abs_weight" in gf.columns

    # -- gene coverage --
    gene_agg = gf.groupby(cfg.group_cols, as_index=False).agg(
        n_genes=(cfg.gene_col, "size"),
        n_genes_covered=("has_score", "sum"),
    )
    gene_agg["gene_coverage"] = gene_agg["n_genes_covered"] / gene_agg["n_genes"]

    # -- loading-weighted coverage (when weights exist) --
    if has_weights:
        total_w = gf.groupby(cfg.group_cols, as_index=False).agg(
            total_weight=("abs_weight", "sum"),
        )
        covered_w = (
            gf[gf["has_score"]]
            .groupby(cfg.group_cols, as_index=False)
            .agg(covered_weight=("abs_weight", "sum"))
        )
        gene_agg = gene_agg.merge(total_w, on=cfg.group_cols, how="left")
        gene_agg = gene_agg.merge(covered_w, on=cfg.group_cols, how="left")
        gene_agg["covered_weight"] = gene_agg["covered_weight"].fillna(0.0)
        gene_agg["loading_weighted_coverage"] = np.where(
            gene_agg["total_weight"] > 0,
            gene_agg["covered_weight"] / gene_agg["total_weight"],
            0.0,
        )

    # -- target coverage (multi-target only) --
    if cfg.target_col is not None:
        df_subset = pair_dwpc
        if max_rank is not None:
            df_subset = df_subset[df_subset["metapath_rank"] <= max_rank]
        target_max = (
            df_subset.groupby(cfg.group_cols + [cfg.target_col], as_index=False)
            .agg(max_dwpc=(cfg.dwpc_col, "max"))
        )
        target_max["hit"] = target_max["max_dwpc"] > 0
        target_agg = target_max.groupby(cfg.group_cols, as_index=False).agg(
            n_targets=(cfg.target_col, "size"),
            n_targets_hit=("hit", "sum"),
        )
        target_agg["target_coverage"] = target_agg["n_targets_hit"] / target_agg["n_targets"]
        gene_agg = gene_agg.merge(target_agg, on=cfg.group_cols, how="left")

    # -- pathway strength --
    gf["log1p_score"] = np.log1p(gf["gene_score"])
    strength = gf.groupby(cfg.group_cols, as_index=False).agg(
        pathway_strength_median=("log1p_score", "median"),
        pathway_strength_mean=("log1p_score", "mean"),
    )
    gene_agg = gene_agg.merge(strength, on=cfg.group_cols, how="left")

    # -- number of metapaths used --
    mp_df = pair_dwpc
    if max_rank is not None:
        mp_df = mp_df[mp_df["metapath_rank"] <= max_rank]
    n_mp = (
        mp_df.groupby(cfg.group_cols, as_index=False)
        .agg(n_metapaths=(cfg.metapath_col, "nunique"))
    )
    gene_agg = gene_agg.merge(n_mp, on=cfg.group_cols, how="left")

    gene_agg["condition"] = condition_label
    return gene_agg


# ---------------------------------------------------------------------------
# Multi vs single comparison
# ---------------------------------------------------------------------------

def compare_conditions(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
    *,
    gene_universe: pd.DataFrame | None = None,
    max_ranks: list[int] | None = None,
) -> pd.DataFrame:
    """Compute pathway metrics under top-1 and all-metapath conditions, plus rescue rates.

    Parameters
    ----------
    max_ranks
        List of metapath rank cutoffs to evaluate.  Defaults to [1, total].
        Each produces a row per pathway group.
    """
    if "metapath_rank" not in pair_dwpc.columns:
        raise ValueError(
            "pair_dwpc must have a 'metapath_rank' column. "
            "Call rank_metapaths_by_dwpc_mass() or rank_metapaths_from_results() first."
        )

    n_max = int(pair_dwpc["metapath_rank"].max())
    if max_ranks is None:
        max_ranks = sorted({1, n_max})

    frames = []
    for k in max_ranks:
        label = f"top{k}" if k < n_max else "all"
        df = compute_pathway_metrics(
            pair_dwpc, cfg,
            gene_universe=gene_universe,
            max_rank=k,
            condition_label=label,
        )
        frames.append(df)

    stacked = pd.concat(frames, ignore_index=True)
    return stacked


def compute_rescue_table(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
    *,
    gene_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-pathway-group rescue: genes/targets gained by multi over top-1.

    Returns one row per pathway group with rescue counts and rates.
    """
    gs_top1 = _gene_scores(pair_dwpc, cfg, max_rank=1)
    gs_all = _gene_scores(pair_dwpc, cfg, max_rank=None)

    top1 = _build_gene_frame(gs_top1, gene_universe, cfg)
    multi = _build_gene_frame(gs_all, gene_universe, cfg)

    merge_keys = cfg.group_cols + [cfg.gene_col]
    compare = top1[merge_keys + ["has_score"]].rename(columns={"has_score": "covered_top1"})
    compare = compare.merge(
        multi[merge_keys + ["has_score"]].rename(columns={"has_score": "covered_multi"}),
        on=merge_keys,
        how="outer",
    )
    compare["covered_top1"] = compare["covered_top1"].fillna(False)
    compare["covered_multi"] = compare["covered_multi"].fillna(False)
    compare["rescued"] = ~compare["covered_top1"] & compare["covered_multi"]

    rescue_agg = compare.groupby(cfg.group_cols, as_index=False).agg(
        n_genes=(cfg.gene_col, "size"),
        n_covered_top1=("covered_top1", "sum"),
        n_covered_multi=("covered_multi", "sum"),
        n_rescued=("rescued", "sum"),
    )
    rescue_agg["gene_rescue_rate"] = np.where(
        rescue_agg["n_genes"] > 0,
        rescue_agg["n_rescued"] / rescue_agg["n_genes"],
        0.0,
    )
    rescue_agg["gene_coverage_top1"] = rescue_agg["n_covered_top1"] / rescue_agg["n_genes"]
    rescue_agg["gene_coverage_multi"] = rescue_agg["n_covered_multi"] / rescue_agg["n_genes"]
    rescue_agg["gene_coverage_gain"] = rescue_agg["gene_coverage_multi"] - rescue_agg["gene_coverage_top1"]

    # Target rescue (multi-target only)
    if cfg.target_col is not None:
        for label, max_rank in [("top1", 1), ("multi", None)]:
            sub = pair_dwpc if max_rank is None else pair_dwpc[pair_dwpc["metapath_rank"] <= max_rank]
            t_max = (
                sub.groupby(cfg.group_cols + [cfg.target_col], as_index=False)
                .agg(max_dwpc=(cfg.dwpc_col, "max"))
            )
            t_max[f"hit_{label}"] = t_max["max_dwpc"] > 0
            t_agg = t_max.groupby(cfg.group_cols, as_index=False).agg(
                **{f"n_targets_{label}": (cfg.target_col, "size"),
                   f"n_targets_hit_{label}": (f"hit_{label}", "sum")},
            )
            rescue_agg = rescue_agg.merge(t_agg, on=cfg.group_cols, how="left")

        rescue_agg["target_rescue"] = (
            rescue_agg["n_targets_hit_multi"].fillna(0)
            - rescue_agg["n_targets_hit_top1"].fillna(0)
        ).astype(int)
        rescue_agg["target_rescue_rate"] = np.where(
            rescue_agg["n_targets_multi"].fillna(0) > 0,
            rescue_agg["target_rescue"] / rescue_agg["n_targets_multi"],
            0.0,
        )

    return rescue_agg


def compute_metapath_complementarity(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
) -> pd.DataFrame:
    """Per (pathway-group, metapath): fraction of covered genes unique to that metapath.

    A high uniqueness fraction means the metapath contributes non-redundant
    signal -- the core argument for multi-DWPC over single-metapath analysis.
    """
    nonzero = pair_dwpc[pair_dwpc[cfg.dwpc_col] > 0].copy()
    if nonzero.empty:
        return pd.DataFrame()

    # For each gene in each group, count how many metapaths cover it
    gene_mp_count = (
        nonzero.groupby(cfg.group_cols + [cfg.gene_col], as_index=False)
        .agg(n_metapaths_covering=(cfg.metapath_col, "nunique"))
    )
    gene_mp_count["unique_to_one"] = gene_mp_count["n_metapaths_covering"] == 1

    # Expand back: for each (group, metapath, gene), is that gene unique to this metapath?
    gene_mp = (
        nonzero[cfg.group_cols + [cfg.metapath_col, cfg.gene_col]]
        .drop_duplicates()
    )
    gene_mp = gene_mp.merge(
        gene_mp_count[cfg.group_cols + [cfg.gene_col, "unique_to_one"]],
        on=cfg.group_cols + [cfg.gene_col],
        how="left",
    )

    result = gene_mp.groupby(cfg.group_cols + [cfg.metapath_col], as_index=False).agg(
        n_genes_covered=(cfg.gene_col, "size"),
        n_genes_unique=("unique_to_one", "sum"),
    )
    result["uniqueness_fraction"] = np.where(
        result["n_genes_covered"] > 0,
        result["n_genes_unique"] / result["n_genes_covered"],
        0.0,
    )
    return result


def compute_cumulative_coverage(
    pair_dwpc: pd.DataFrame,
    cfg: CoverageConfig,
    *,
    gene_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cumulative gene coverage as metapaths are added in rank order.

    Returns one row per (pathway_group, k) where k is the number of
    metapaths included (1, 2, ..., n_selected).
    """
    if "metapath_rank" not in pair_dwpc.columns:
        raise ValueError("pair_dwpc must have a 'metapath_rank' column.")

    max_k = int(pair_dwpc["metapath_rank"].max())
    frames = []
    for k in range(1, max_k + 1):
        metrics = compute_pathway_metrics(
            pair_dwpc, cfg,
            gene_universe=gene_universe,
            max_rank=k,
            condition_label=str(k),
        )
        metrics["k"] = k
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)
