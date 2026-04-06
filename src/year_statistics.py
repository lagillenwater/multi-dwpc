"""Shared builders for year-level aggregated metapath statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_METAPATHS = {"BPpG", "GpBP"}
DEFAULT_STATISTICS = [
    "mean_dwpc",
    "mean_dwpc_nonzero",
    "median_dwpc",
    "median_dwpc_nonzero",
    "mean_pvalue",
    "mean_pvalue_nonzero",
    "median_pvalue",
    "median_pvalue_nonzero",
    "mean_std",
    "mean_std_nonzero",
    "median_std",
    "median_std_nonzero",
]


def _standardize_year_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce direct/API year result frames to a common column set."""
    out = df.copy()
    rename_map = {}
    if "metapath" not in out.columns and "metapath_abbreviation" in out.columns:
        rename_map["metapath_abbreviation"] = "metapath"
    if "go_id" not in out.columns and "neo4j_source_id" in out.columns:
        rename_map["neo4j_source_id"] = "go_id"
    if rename_map:
        out = out.rename(columns=rename_map)

    required = {"go_id", "metapath", "dwpc"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Year result frame missing required columns: {sorted(missing)}")
    return out


def _safe_series_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) > 0 else np.nan


def _safe_series_median(values: pd.Series) -> float:
    return float(values.median()) if len(values) > 0 else np.nan


def _aggregate_one_year_group(group: pd.DataFrame) -> dict[str, float | int]:
    dwpc_all = group["dwpc"].dropna()
    dwpc_nonzero = dwpc_all[dwpc_all > 0]

    if "p_value" in group.columns:
        pval_all = group["p_value"].dropna()
        pval_nonzero = pval_all[pval_all > 0] if len(pval_all) > 0 else pd.Series(dtype=float)
    else:
        pval_all = pd.Series(dtype=float)
        pval_nonzero = pd.Series(dtype=float)

    if "dgp_nonzero_sd" in group.columns:
        std_all = group["dgp_nonzero_sd"].dropna()
        std_nonzero = std_all[std_all > 0] if len(std_all) > 0 else pd.Series(dtype=float)
    else:
        std_all = pd.Series(dtype=float)
        std_nonzero = pd.Series(dtype=float)

    return {
        "mean_dwpc": _safe_series_mean(dwpc_all),
        "mean_dwpc_nonzero": _safe_series_mean(dwpc_nonzero),
        "median_dwpc": _safe_series_median(dwpc_all),
        "median_dwpc_nonzero": _safe_series_median(dwpc_nonzero),
        "mean_pvalue": _safe_series_mean(pval_all),
        "mean_pvalue_nonzero": _safe_series_mean(pval_nonzero),
        "median_pvalue": _safe_series_median(pval_all),
        "median_pvalue_nonzero": _safe_series_median(pval_nonzero),
        "mean_std": _safe_series_mean(std_all),
        "mean_std_nonzero": _safe_series_mean(std_nonzero),
        "median_std": _safe_series_median(std_all),
        "median_std_nonzero": _safe_series_median(std_nonzero),
        "n_total": int(len(dwpc_all)),
        "n_nonzero": int(len(dwpc_nonzero)),
    }


def load_labeled_year_result_files(
    data_dir: Path | str,
    files_config: dict[str, str],
    exclude_metapaths: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load a labeled set of direct/API year result CSVs."""
    data_dir = Path(data_dir)
    exclude_metapaths = set(DEFAULT_EXCLUDE_METAPATHS if exclude_metapaths is None else exclude_metapaths)
    datasets: dict[str, pd.DataFrame] = {}
    for label, filename in files_config.items():
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        frame = _standardize_year_result_frame(pd.read_csv(filepath))
        if exclude_metapaths:
            frame = frame[~frame["metapath"].astype(str).isin(exclude_metapaths)].copy()
        datasets[str(label)] = frame
    return datasets


def build_aggregated_year_statistics_panel(
    normalized_df: pd.DataFrame,
    *,
    dataset_col: str = "name",
    exclude_metapaths: set[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate GO-term x metapath statistics from normalized year results.

    The input is the shared Step 6 year schema, optionally with extra columns such
    as `p_value` or `dgp_nonzero_sd`. The output preserves dataset metadata and
    adds a `dataset` label column used by downstream year-statistics analyses.
    """
    if normalized_df.empty:
        return pd.DataFrame()

    exclude_metapaths = set(DEFAULT_EXCLUDE_METAPATHS if exclude_metapaths is None else exclude_metapaths)
    required = {"go_id", "metapath", "dwpc", dataset_col}
    missing = required - set(normalized_df.columns)
    if missing:
        raise ValueError(
            f"Normalized year dataframe missing required columns for aggregation: {sorted(missing)}"
        )

    work = normalized_df.copy()
    if exclude_metapaths:
        work = work[~work["metapath"].astype(str).isin(exclude_metapaths)].copy()
    work["dataset"] = work[dataset_col].astype(str)

    meta_cols = [
        col
        for col in ["domain", "name", "control", "replicate", "year", "score_source"]
        if col in work.columns
    ]
    group_cols = [*meta_cols, "dataset", "go_id", "metapath"]

    rows = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {group_cols[idx]: keys[idx] for idx in range(len(group_cols))}
        row.update(_aggregate_one_year_group(group))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_aggregated_year_statistics(
    datasets: dict[str, pd.DataFrame] | pd.DataFrame,
    *,
    dataset_col: str = "name",
    exclude_metapaths: set[str] | None = None,
) -> pd.DataFrame:
    """Aggregate GO-term x metapath statistics across labeled or normalized year datasets."""
    if isinstance(datasets, pd.DataFrame):
        return build_aggregated_year_statistics_panel(
            datasets,
            dataset_col=dataset_col,
            exclude_metapaths=exclude_metapaths,
        )

    exclude_metapaths = set(DEFAULT_EXCLUDE_METAPATHS if exclude_metapaths is None else exclude_metapaths)
    rows = []
    for label, df in datasets.items():
        work = _standardize_year_result_frame(df)
        if exclude_metapaths:
            work = work[~work["metapath"].astype(str).isin(exclude_metapaths)].copy()
        for (go_id, metapath), group in work.groupby(["go_id", "metapath"], dropna=False):
            row = {
                "go_id": go_id,
                "metapath": str(metapath),
                "dataset": str(label),
            }
            row.update(_aggregate_one_year_group(group))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def detect_supported_statistics(agg_df: pd.DataFrame) -> list[str]:
    """Return the current statistic panel supported by an aggregated year-statistics frame."""
    statistics = DEFAULT_STATISTICS[:4]
    for prefix in ("pvalue", "std"):
        stat_cols = [col for col in DEFAULT_STATISTICS if prefix in col]
        if any(col in agg_df.columns and agg_df[col].notna().any() for col in stat_cols):
            statistics.extend(stat_cols)
    return statistics
