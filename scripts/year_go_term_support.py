#!/usr/bin/env python3
"""Build GO-term-level and global metapath support tables from year replicate summaries."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.year_replicate_analysis import load_summary_bank  # noqa: E402


def _apply_bh_fdr(df: pd.DataFrame, p_col: str, out_col: str, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
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


def _empirical_pvalue(real_value: float, null_values: np.ndarray) -> float:
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0 or not math.isfinite(float(real_value)):
        return np.nan
    return float((1.0 + np.sum(valid >= float(real_value))) / float(valid.size + 1))


def _effect_size(real_value: float, null_values: np.ndarray) -> float:
    valid = np.asarray(null_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0 or not math.isfinite(float(real_value)):
        return np.nan
    null_mean = float(np.mean(valid))
    null_std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    diff = float(real_value) - null_mean
    if null_std > 0:
        return diff / null_std
    if diff > 0:
        return np.inf
    if diff < 0:
        return -np.inf
    return 0.0


def _combine_pvalues_fisher(pvals: pd.Series) -> float:
    vals = pvals.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    vals = np.clip(vals, 1e-300, 1.0)
    stat = float(-2.0 * np.sum(np.log(vals)))
    return float(chi2.sf(stat, 2 * vals.size))


def build_go_term_support(summary_df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "control", "replicate", "go_id", "metapath", "mean_score"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Summary dataframe missing required columns: {sorted(missing)}")

    work = summary_df.copy()
    work["year"] = work["year"].astype(int)
    work["control"] = work["control"].astype(str)
    work["replicate"] = work["replicate"].astype(int)
    work["go_id"] = work["go_id"].astype(str)
    work["metapath"] = work["metapath"].astype(str)

    real_df = work[work["control"] == "real"].copy()
    if real_df.empty:
        raise ValueError("No real rows found in summary dataframe")
    real_scores = (
        real_df.groupby(["year", "go_id", "metapath"], as_index=False)["mean_score"]
        .mean()
        .rename(columns={"mean_score": "real_mean"})
    )

    null_df = work[work["control"] != "real"].copy()
    if null_df.empty:
        raise ValueError("No null rows found in summary dataframe")
    null_rep = (
        null_df.groupby(["year", "go_id", "metapath", "control", "replicate"], as_index=False)["mean_score"]
        .mean()
        .rename(columns={"mean_score": "null_mean_score"})
    )

    rows: list[dict] = []
    for row in real_scores.itertuples(index=False):
        out_row = {
            "year": int(row.year),
            "go_id": str(row.go_id),
            "metapath": str(row.metapath),
            "real_mean": float(row.real_mean),
        }
        for control, prefix in [("permuted", "perm"), ("random", "rand")]:
            subset = null_rep[
                (null_rep["year"].astype(int) == int(row.year))
                & (null_rep["go_id"].astype(str) == str(row.go_id))
                & (null_rep["metapath"].astype(str) == str(row.metapath))
                & (null_rep["control"].astype(str) == control)
            ].copy()
            vals = subset["null_mean_score"].to_numpy(dtype=float)
            out_row[f"{prefix}_null_mean"] = float(np.mean(vals)) if vals.size else np.nan
            out_row[f"{prefix}_null_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            out_row[f"p_{prefix}"] = _empirical_pvalue(float(row.real_mean), vals)
            out_row[f"d_{prefix}"] = _effect_size(float(row.real_mean), vals)
            out_row[f"b_eff_{prefix}"] = int(vals.size)
        rows.append(out_row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["diff_perm"] = out["real_mean"] - out["perm_null_mean"]
    out["diff_rand"] = out["real_mean"] - out["rand_null_mean"]
    out["min_diff"] = np.minimum(out["diff_perm"], out["diff_rand"])
    out["min_d"] = np.minimum(out["d_perm"], out["d_rand"])
    out = _apply_bh_fdr(out, p_col="p_perm", out_col="p_perm_fdr", group_cols=["year", "go_id"])
    out = _apply_bh_fdr(out, p_col="p_rand", out_col="p_rand_fdr", group_cols=["year", "go_id"])
    out["supported"] = (
        (out["p_perm_fdr"] < 0.05)
        & (out["p_rand_fdr"] < 0.05)
        & (out["diff_perm"] > 0)
        & (out["diff_rand"] > 0)
    )
    out["fdr_sum"] = out["p_perm_fdr"] + out["p_rand_fdr"]
    return out.sort_values(
        ["year", "go_id", "supported", "min_d", "min_diff", "fdr_sum", "metapath"],
        ascending=[True, True, False, False, False, True, True],
    ).reset_index(drop=True)


def build_global_metapath_support(go_support_df: pd.DataFrame) -> pd.DataFrame:
    if go_support_df.empty:
        return pd.DataFrame()

    agg = (
        go_support_df.groupby(["year", "metapath"], as_index=False)
        .agg(
            n_go_terms=("go_id", "nunique"),
            n_go_terms_supported=("supported", "sum"),
            mean_real_mean=("real_mean", "mean"),
            mean_diff_perm=("diff_perm", "mean"),
            mean_diff_rand=("diff_rand", "mean"),
            mean_min_diff=("min_diff", "mean"),
            mean_min_d=("min_d", "mean"),
            median_min_d=("min_d", "median"),
            combined_p_perm=("p_perm", _combine_pvalues_fisher),
            combined_p_rand=("p_rand", _combine_pvalues_fisher),
        )
    )
    agg["support_fraction"] = agg["n_go_terms_supported"] / agg["n_go_terms"].replace(0, np.nan)
    agg = _apply_bh_fdr(agg, p_col="combined_p_perm", out_col="combined_p_perm_fdr", group_cols=["year"])
    agg = _apply_bh_fdr(agg, p_col="combined_p_rand", out_col="combined_p_rand_fdr", group_cols=["year"])
    agg["supported_global"] = (
        (agg["combined_p_perm_fdr"] < 0.05)
        & (agg["combined_p_rand_fdr"] < 0.05)
        & (agg["mean_diff_perm"] > 0)
        & (agg["mean_diff_rand"] > 0)
    )
    agg["combined_fdr_sum"] = agg["combined_p_perm_fdr"] + agg["combined_p_rand_fdr"]
    return agg.sort_values(
        ["year", "supported_global", "support_fraction", "mean_min_d", "combined_fdr_sum", "metapath"],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-dir",
        default=str(REPO_ROOT / "output" / "dwpc_direct" / "all_GO_positive_growth"),
        help="Year replicate workspace directory containing replicate_summaries/.",
    )
    parser.add_argument("--summaries-dir", default=None)
    parser.add_argument(
        "--go-support-output",
        default=str(REPO_ROOT / "output" / "year_direct_go_term_support.csv"),
    )
    parser.add_argument(
        "--global-support-output",
        default=str(REPO_ROOT / "output" / "year_direct_global_metapath_support.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_source = Path(args.summaries_dir) if args.summaries_dir else Path(args.workspace_dir) / "replicate_summaries"
    summary_df = load_summary_bank(summary_source)
    go_df = build_go_term_support(summary_df)
    global_df = build_global_metapath_support(go_df)

    go_path = Path(args.go_support_output)
    global_path = Path(args.global_support_output)
    go_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.parent.mkdir(parents=True, exist_ok=True)
    go_df.to_csv(go_path, index=False)
    global_df.to_csv(global_path, index=False)

    print(f"Saved GO-term support: {go_path}")
    print(f"Saved global metapath support: {global_path}")


if __name__ == "__main__":
    main()
