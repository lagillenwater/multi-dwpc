"""
Final statistics assembly for LV multi-DWPC optimized pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def _apply_bh_fdr(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    group_cols = ["lv_id", "target_set_id"]
    for _, idx in out.groupby(group_cols).groups.items():
        subset = out.loc[idx]
        pvals = subset[p_col].to_numpy(dtype=float)
        valid = np.isfinite(pvals)
        if valid.sum() == 0:
            continue
        corrected = np.full(len(subset), np.nan, dtype=float)
        corrected[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
        out.loc[idx, out_col] = corrected
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
        "target_set_id",
        "target_set_label",
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

    merged["supported"] = (
        (merged["p_perm_fdr"] < 0.05)
        & (merged["p_rand_fdr"] < 0.05)
        & (merged["diff_perm"] > 0)
        & (merged["diff_rand"] > 0)
    )

    merged = merged.sort_values(
        ["lv_id", "target_set_id", "supported", "min_d", "min_diff"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)

    out_path = output_dir / "lv_metapath_results.csv"
    merged.to_csv(out_path, index=False)
    return merged
