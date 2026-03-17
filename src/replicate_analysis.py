"""Shared analysis helpers for explicit null-replicate experiments."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


DEFAULT_SUMMARY_COLUMNS = {
    "domain",
    "name",
    "control",
    "replicate",
    "mean_score",
}


def build_diff_runs(
    summary_df: pd.DataFrame,
    join_keys: list[str],
    score_col: str = "mean_score",
    control_col: str = "control",
    replicate_col: str = "replicate",
    real_control: str = "real",
) -> pd.DataFrame:
    """Join real summaries to null summaries and compute diff = real - null."""
    if summary_df.empty:
        return pd.DataFrame()

    required = set(join_keys) | {score_col, control_col, replicate_col}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Summary dataframe missing required columns: {sorted(missing)}")

    real_df = summary_df[summary_df[control_col].astype(str) == str(real_control)].copy()
    if real_df.empty:
        raise ValueError(f"No rows found with {control_col}={real_control!r}")
    null_df = summary_df[summary_df[control_col].astype(str) != str(real_control)].copy()
    if null_df.empty:
        raise ValueError("No null-control rows found in summary dataframe")

    real_keep = join_keys + [score_col]
    real_df = real_df[real_keep].drop_duplicates(subset=join_keys).rename(
        columns={score_col: "real_mean_score"}
    )
    null_keep = [
        col
        for col in null_df.columns
        if col in join_keys or col in {control_col, replicate_col, score_col, "name"}
    ]
    null_df = null_df[null_keep].rename(columns={score_col: "null_mean_score"})

    merged = null_df.merge(real_df, on=join_keys, how="inner")
    merged["diff"] = merged["real_mean_score"] - merged["null_mean_score"]
    return merged


def summarize_feature_variance(
    runs_df: pd.DataFrame,
    feature_keys: list[str],
    diff_col: str = "diff",
    extra_metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-feature variance summaries across replicate runs."""
    if runs_df.empty:
        return pd.DataFrame()

    extra_metric_cols = extra_metric_cols or []
    rows = []
    for key, group in runs_df.groupby(feature_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        diff_vals = group[diff_col].to_numpy(dtype=float)
        row = {feature_keys[i]: key[i] for i in range(len(feature_keys))}
        row.update(
            {
                "n_replicates": int(len(group)),
                "diff_mean": float(np.nanmean(diff_vals)),
                "diff_std": float(np.nanstd(diff_vals, ddof=1)) if len(group) > 1 else np.nan,
                "diff_var": float(np.nanvar(diff_vals, ddof=1)) if len(group) > 1 else np.nan,
            }
        )
        for metric_col in extra_metric_cols:
            vals = group[metric_col].to_numpy(dtype=float)
            row[f"{metric_col}_mean"] = float(np.nanmean(vals))
            row[f"{metric_col}_std"] = (
                float(np.nanstd(vals, ddof=1)) if len(group) > 1 else np.nan
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(feature_keys).reset_index(drop=True)


def summarize_overall_variance(
    feature_df: pd.DataFrame,
    overall_keys: list[str],
    runs_df: pd.DataFrame | None = None,
    replicate_col: str | None = None,
) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()

    out = (
        feature_df.groupby(overall_keys, as_index=False)
        .agg(
            n_features=("diff_var", "size"),
            mean_diff_std=("diff_std", "mean"),
            mean_diff_var=("diff_var", "mean"),
            median_diff_std=("diff_std", "median"),
            median_diff_var=("diff_var", "median"),
        )
        .sort_values(overall_keys)
        .reset_index(drop=True)
    )
    if runs_df is not None and replicate_col is not None and not runs_df.empty:
        rep_df = (
            runs_df.groupby(overall_keys, as_index=False)[replicate_col]
            .nunique()
            .rename(columns={replicate_col: "n_replicates"})
        )
        out = out.merge(rep_df, on=overall_keys, how="left")
    return out


def rank_features(
    runs_df: pd.DataFrame,
    rank_group_keys: list[str],
    feature_col: str = "metapath",
    score_col: str = "diff",
    rank_col: str = "feature_rank",
) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    sort_cols = rank_group_keys + [score_col, feature_col]
    ascending = [True] * len(rank_group_keys) + [False, True]
    rank_df = runs_df.sort_values(sort_cols, ascending=ascending).copy()
    rank_df[rank_col] = rank_df.groupby(rank_group_keys).cumcount() + 1
    return rank_df


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def summarize_rank_stability(
    rank_df: pd.DataFrame,
    outer_keys: list[str],
    replicate_col: str,
    feature_col: str = "metapath",
    rank_col: str = "feature_rank",
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rank_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    pairwise_rows = []
    for key, group in rank_df.groupby(outer_keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        by_rep = {}
        for rep, rep_df in group.groupby(replicate_col):
            by_rep[rep] = dict(
                zip(
                    rep_df[feature_col].astype(str),
                    rep_df[rank_col].astype(int),
                )
            )
        for rep_a, rep_b in combinations(sorted(by_rep), 2):
            ranks_a = by_rep[rep_a]
            ranks_b = by_rep[rep_b]
            common = sorted(set(ranks_a) & set(ranks_b))
            if len(common) < 2:
                continue
            x = pd.Series([ranks_a[m] for m in common], dtype=float)
            y = pd.Series([ranks_b[m] for m in common], dtype=float)
            top_a = {m for m, r in ranks_a.items() if r <= int(top_k)}
            top_b = {m for m, r in ranks_b.items() if r <= int(top_k)}
            row = {outer_keys[i]: key[i] for i in range(len(outer_keys))}
            row.update(
                {
                    "replicate_a": rep_a,
                    "replicate_b": rep_b,
                    "n_common_features": int(len(common)),
                    "spearman_rho": float(x.corr(y, method="spearman")),
                    "topk_jaccard": _jaccard(top_a, top_b),
                }
            )
            pairwise_rows.append(row)

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        return pairwise_df, pd.DataFrame(), pd.DataFrame()

    entity_summary = (
        pairwise_df.groupby(outer_keys, as_index=False)
        .agg(
            n_pairs=("spearman_rho", "size"),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            mean_topk_jaccard=("topk_jaccard", "mean"),
            median_topk_jaccard=("topk_jaccard", "median"),
        )
        .sort_values(outer_keys)
        .reset_index(drop=True)
    )

    overall_keys = [key for key in outer_keys if key not in {"go_id", "lv_id", "target_set_id"}]
    if not overall_keys:
        overall_keys = outer_keys
    overall_df = (
        entity_summary.groupby(overall_keys, as_index=False)
        .agg(
            n_entities=(outer_keys[-1], "size"),
            n_pairs=("n_pairs", "sum"),
            mean_spearman_rho=("mean_spearman_rho", "mean"),
            median_spearman_rho=("median_spearman_rho", "median"),
            mean_topk_jaccard=("mean_topk_jaccard", "mean"),
            median_topk_jaccard=("median_topk_jaccard", "median"),
        )
        .sort_values(overall_keys)
        .reset_index(drop=True)
    )
    return pairwise_df, entity_summary, overall_df
