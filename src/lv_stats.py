"""
Final statistics assembly for LV multi-DWPC optimized pipeline.

Each LV maps to a single target. Results are grouped by lv_id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def _apply_bh_fdr(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    # FDR correction within each LV
    for _, idx in out.groupby("lv_id").groups.items():
        subset = out.loc[idx]
        pvals = subset[p_col].to_numpy(dtype=float)
        valid = np.isfinite(pvals)
        if valid.sum() == 0:
            continue
        corrected = np.full(len(subset), np.nan, dtype=float)
        corrected[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
        out.loc[idx, out_col] = corrected
    return out


def _add_consensus_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Rank within each LV
    out["rank_perm"] = (
        out.groupby("lv_id")["diff_perm"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    out["rank_rand"] = (
        out.groupby("lv_id")["diff_rand"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    out["consensus_rank"] = 0.5 * (out["rank_perm"] + out["rank_rand"])
    out["consensus_score"] = 0.5 * (
        (1.0 / out["rank_perm"].replace(0, np.nan))
        + (1.0 / out["rank_rand"].replace(0, np.nan))
    )
    return out


def build_final_stats(output_dir: Path) -> pd.DataFrame:
    """
    Merge real and null summaries into final metapath-level result table.
    """
    output_dir = Path(output_dir)
    real = pd.read_csv(output_dir / "real_feature_scores.csv")
    nulls = pd.read_csv(output_dir / "null_streaming_summary.csv")

    perm = nulls[nulls["null_type"] == "permuted"].copy()
    rand = nulls[nulls["null_type"] == "random"].copy()

    perm = perm.rename(
        columns={
            "null_mean": "perm_null_mean",
            "null_std": "perm_null_std",
            "p_empirical": "p_perm",
            "d_median": "d_perm",
            "n_eff": "b_eff_perm",
        }
    )
    rand = rand.rename(
        columns={
            "null_mean": "rand_null_mean",
            "null_std": "rand_null_std",
            "p_empirical": "p_rand",
            "d_median": "d_rand",
            "n_eff": "b_eff_rand",
        }
    )

    key = [
        "lv_id",
        "target_id",
        "target_name",
        "node_type",
        "feature_idx",
        "metapath",
    ]
    merged = real.merge(
        perm[key + ["perm_null_mean", "perm_null_std", "p_perm", "d_perm", "b_eff_perm"]],
        on=key,
        how="left",
    ).merge(
        rand[key + ["rand_null_mean", "rand_null_std", "p_rand", "d_rand", "b_eff_rand"]],
        on=key,
        how="left",
    )

    merged["diff_perm"] = merged["real_mean"] - merged["perm_null_mean"]
    merged["diff_rand"] = merged["real_mean"] - merged["rand_null_mean"]
    merged["min_diff"] = np.minimum(merged["diff_perm"], merged["diff_rand"])
    merged["min_d"] = np.minimum(merged["d_perm"], merged["d_rand"])

    merged = _apply_bh_fdr(merged, p_col="p_perm", out_col="p_perm_fdr")
    merged = _apply_bh_fdr(merged, p_col="p_rand", out_col="p_rand_fdr")
    merged = _add_consensus_rank_columns(merged)

    merged["supported"] = (
        (merged["p_perm_fdr"] < 0.05)
        & (merged["p_rand_fdr"] < 0.05)
        & (merged["diff_perm"] > 0)
        & (merged["diff_rand"] > 0)
    )

    merged = merged.sort_values(
        ["lv_id", "supported", "consensus_score", "min_d", "min_diff"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    out_path = output_dir / "lv_metapath_results.csv"
    merged.to_csv(out_path, index=False)
    return merged
