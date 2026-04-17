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


def build_b_seed_runs(
    summary_df: pd.DataFrame,
    b_values: list[int],
    seeds: list[int],
    *,
    join_keys: list[str],
    replicate_pool_keys: list[str],
    control_col: str = "control",
    replicate_col: str = "replicate",
    score_col: str = "mean_score",
    real_control: str = "real",
) -> pd.DataFrame:
    """Generic B/seed resampling for explicit real-vs-null replicate analyses.

    Computes both raw difference (diff) and permutation z-statistic (z = diff / null_std).
    The z-statistic uses the standard deviation of null scores across sampled replicates.
    """
    if summary_df.empty:
        return pd.DataFrame()

    required = set(join_keys) | set(replicate_pool_keys) | {control_col, replicate_col, score_col}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Summary dataframe missing required columns: {sorted(missing)}")

    real_df = summary_df[summary_df[control_col].astype(str) == str(real_control)].copy()
    if real_df.empty:
        raise ValueError(f"No real rows found with {control_col}={real_control!r}")
    real_df = real_df[join_keys + [score_col]].drop_duplicates(subset=join_keys).rename(
        columns={score_col: "real_mean_score"}
    )

    null_df = summary_df[summary_df[control_col].astype(str) != str(real_control)].copy()
    if null_df.empty:
        raise ValueError("No null-control rows found in summary dataframe")

    max_b = max(int(b) for b in b_values)
    available = (
        null_df.groupby(replicate_pool_keys, dropna=False)[replicate_col]
        .nunique()
        .to_dict()
    )
    for pool_key, count in available.items():
        if int(count) < max_b:
            label = pool_key if isinstance(pool_key, tuple) else (pool_key,)
            raise ValueError(
                f"Requested max B={max_b} but only {count} replicate summaries are available "
                f"for pool={label}."
            )

    rows = []
    for pool_key, group in null_df.groupby(replicate_pool_keys, sort=True, dropna=False):
        key_values = pool_key if isinstance(pool_key, tuple) else (pool_key,)
        pool_meta = {
            replicate_pool_keys[idx]: key_values[idx]
            for idx in range(len(replicate_pool_keys))
        }
        rep_ids = sorted(group[replicate_col].astype(int).unique().tolist())
        rep_ids_arr = np.asarray(rep_ids, dtype=int)
        for b in sorted(int(x) for x in b_values):
            for seed in sorted(int(x) for x in seeds):
                rng = np.random.RandomState(seed)
                selected = rng.choice(rep_ids_arr, size=b, replace=False)
                subset = group[group[replicate_col].isin(selected)].copy()
                # Compute mean and std of null scores across sampled replicates
                agg = (
                    subset.groupby(join_keys, as_index=False)[score_col]
                    .agg(null_mean_score=("mean"), null_std_score=("std"))
                )
                merged = agg.merge(real_df, on=join_keys, how="inner")
                for key, value in pool_meta.items():
                    if key not in merged.columns:
                        merged[key] = value
                merged["b"] = int(b)
                merged["seed"] = int(seed)
                merged["diff"] = merged["real_mean_score"] - merged["null_mean_score"]
                # Compute permutation z-statistic (z = diff / null_std)
                # Use ddof=1 for sample std; handle cases where std=0 or NaN
                merged["null_std_score"] = merged["null_std_score"].replace(0, np.nan)
                merged["permutation_z"] = merged["diff"] / merged["null_std_score"]
                rows.append(merged)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    ordered = [
        *replicate_pool_keys, "b", "seed", *join_keys,
        "real_mean_score", "null_mean_score", "null_std_score", "diff", "permutation_z"
    ]
    ordered = [col for idx, col in enumerate(ordered) if col in out.columns and col not in ordered[:idx]]
    return out[ordered]


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


def _spearman_from_ranks(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return float(x_rank.corr(y_rank, method="pearson"))


def _rbo_score(rank_a: list[str], rank_b: list[str], p: float) -> float:
    """Compute extrapolated finite-list rank-biased overlap for two ranked lists."""
    if not rank_a or not rank_b:
        return np.nan
    depth = min(len(rank_a), len(rank_b))
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    overlap_sum = 0.0
    overlap_at_depth = 0.0
    for d in range(1, depth + 1):
        seen_a.add(rank_a[d - 1])
        seen_b.add(rank_b[d - 1])
        overlap_at_depth = len(seen_a & seen_b) / float(d)
        overlap_sum += overlap_at_depth * (p ** (d - 1))
    return float(((1.0 - p) * overlap_sum) + (overlap_at_depth * (p ** depth)))


def _normalize_top_k_specs(top_k: int | str | list[int | str] | tuple[int | str, ...]) -> list[tuple[str, int | None]]:
    if isinstance(top_k, (int, str)):
        raw_values = [top_k]
    else:
        raw_values = list(top_k)

    specs: list[tuple[str, int | None]] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text in {"all", "full"}:
            label = "all"
            limit = None
        else:
            limit = int(value)
            if limit < 1:
                raise ValueError("top_k values must be positive integers or 'all'")
            label = str(limit)
        if label in seen:
            continue
        seen.add(label)
        specs.append((label, limit))

    if not specs:
        raise ValueError("Expected at least one top_k value")
    return specs


def summarize_rank_stability(
    rank_df: pd.DataFrame,
    outer_keys: list[str],
    replicate_col: str,
    feature_col: str = "metapath",
    rank_col: str = "feature_rank",
    top_k: int | str | list[int | str] | tuple[int | str, ...] = 10,
    rbo_p: float | None = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rank_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    top_k_specs = _normalize_top_k_specs(top_k)
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
            ordered_a = [m for m, _ in sorted(ranks_a.items(), key=lambda item: (item[1], item[0]))]
            ordered_b = [m for m, _ in sorted(ranks_b.items(), key=lambda item: (item[1], item[0]))]
            row = {outer_keys[i]: key[i] for i in range(len(outer_keys))}
            row.update(
                {
                    "replicate_a": rep_a,
                    "replicate_b": rep_b,
                    "n_common_features": int(len(common)),
                    "spearman_rho": _spearman_from_ranks(x, y),
                }
            )
            if rbo_p is not None:
                row["rbo"] = _rbo_score(ordered_a, ordered_b, p=float(rbo_p))
            for label, limit in top_k_specs:
                if limit is None:
                    top_a = set(ranks_a)
                    top_b = set(ranks_b)
                else:
                    top_a = {m for m, r in ranks_a.items() if r <= int(limit)}
                    top_b = {m for m, r in ranks_b.items() if r <= int(limit)}
                row[f"topk_jaccard_{label}"] = _jaccard(top_a, top_b)
            if len(top_k_specs) == 1:
                row["topk_jaccard"] = row[f"topk_jaccard_{top_k_specs[0][0]}"]
            pairwise_rows.append(row)

    pairwise_df = pd.DataFrame(pairwise_rows)
    if pairwise_df.empty:
        return pairwise_df, pd.DataFrame(), pd.DataFrame()

    agg_dict: dict[str, tuple[str, str]] = {
        "n_pairs": ("spearman_rho", "size"),
        "mean_spearman_rho": ("spearman_rho", "mean"),
        "median_spearman_rho": ("spearman_rho", "median"),
    }
    if rbo_p is not None and "rbo" in pairwise_df.columns:
        agg_dict["mean_rbo"] = ("rbo", "mean")
        agg_dict["median_rbo"] = ("rbo", "median")
    for label, _ in top_k_specs:
        agg_dict[f"mean_topk_jaccard_{label}"] = (f"topk_jaccard_{label}", "mean")
        agg_dict[f"median_topk_jaccard_{label}"] = (f"topk_jaccard_{label}", "median")

    entity_summary = (
        pairwise_df.groupby(outer_keys, as_index=False)
        .agg(**agg_dict)
        .sort_values(outer_keys)
        .reset_index(drop=True)
    )
    if len(top_k_specs) == 1:
        label = top_k_specs[0][0]
        entity_summary["mean_topk_jaccard"] = entity_summary[f"mean_topk_jaccard_{label}"]
        entity_summary["median_topk_jaccard"] = entity_summary[f"median_topk_jaccard_{label}"]

    overall_keys = [key for key in outer_keys if key not in {"go_id", "lv_id", "target_set_id"}]
    if not overall_keys:
        overall_keys = outer_keys
    overall_agg: dict[str, tuple[str, str]] = {
        "n_entities": (outer_keys[-1], "size"),
        "n_pairs": ("n_pairs", "sum"),
        "mean_spearman_rho": ("mean_spearman_rho", "mean"),
        "median_spearman_rho": ("median_spearman_rho", "median"),
    }
    if "mean_rbo" in entity_summary.columns:
        overall_agg["mean_rbo"] = ("mean_rbo", "mean")
        overall_agg["median_rbo"] = ("median_rbo", "median")
    for label, _ in top_k_specs:
        overall_agg[f"mean_topk_jaccard_{label}"] = (f"mean_topk_jaccard_{label}", "mean")
        overall_agg[f"median_topk_jaccard_{label}"] = (f"median_topk_jaccard_{label}", "median")

    overall_df = (
        entity_summary.groupby(overall_keys, as_index=False)
        .agg(**overall_agg)
        .sort_values(overall_keys)
        .reset_index(drop=True)
    )
    if len(top_k_specs) == 1:
        label = top_k_specs[0][0]
        overall_df["mean_topk_jaccard"] = overall_df[f"mean_topk_jaccard_{label}"]
        overall_df["median_topk_jaccard"] = overall_df[f"median_topk_jaccard_{label}"]
    return pairwise_df, entity_summary, overall_df
