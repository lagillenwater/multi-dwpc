"""Shared builders for year-level aggregated metapath statistics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_METAPATHS = {"BPpG", "GpBP"}


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


def build_aggregated_year_statistics(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate GO-term x metapath statistics across labeled year result datasets."""
    rows = []
    for label, df in datasets.items():
        work = _standardize_year_result_frame(df)
        for (go_id, metapath), group in work.groupby(["go_id", "metapath"], dropna=False):
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

            rows.append(
                {
                    "go_id": go_id,
                    "metapath": str(metapath),
                    "dataset": str(label),
                    "mean_dwpc": dwpc_all.mean() if len(dwpc_all) > 0 else np.nan,
                    "mean_dwpc_nonzero": dwpc_nonzero.mean() if len(dwpc_nonzero) > 0 else np.nan,
                    "median_dwpc": dwpc_all.median() if len(dwpc_all) > 0 else np.nan,
                    "median_dwpc_nonzero": dwpc_nonzero.median() if len(dwpc_nonzero) > 0 else np.nan,
                    "mean_pvalue": pval_all.mean() if len(pval_all) > 0 else np.nan,
                    "mean_pvalue_nonzero": pval_nonzero.mean() if len(pval_nonzero) > 0 else np.nan,
                    "median_pvalue": pval_all.median() if len(pval_all) > 0 else np.nan,
                    "median_pvalue_nonzero": pval_nonzero.median() if len(pval_nonzero) > 0 else np.nan,
                    "mean_std": std_all.mean() if len(std_all) > 0 else np.nan,
                    "mean_std_nonzero": std_nonzero.mean() if len(std_nonzero) > 0 else np.nan,
                    "median_std": std_all.median() if len(std_all) > 0 else np.nan,
                    "median_std_nonzero": std_nonzero.median() if len(std_nonzero) > 0 else np.nan,
                    "n_total": int(len(dwpc_all)),
                    "n_nonzero": int(len(dwpc_nonzero)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def detect_supported_statistics(agg_df: pd.DataFrame) -> list[str]:
    """Return the current statistic panel supported by an aggregated year-statistics frame."""
    statistics = [
        "mean_dwpc",
        "mean_dwpc_nonzero",
        "median_dwpc",
        "median_dwpc_nonzero",
    ]
    for prefix in ("pvalue", "std"):
        stat_cols = [
            f"mean_{prefix}",
            f"mean_{prefix}_nonzero",
            f"median_{prefix}",
            f"median_{prefix}_nonzero",
        ]
        if any(col in agg_df.columns and agg_df[col].notna().any() for col in stat_cols):
            statistics.extend(stat_cols)
    return statistics
