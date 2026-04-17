#!/usr/bin/env python3
"""Build year metapath support statistics from explicit null replicate summaries."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.year_replicate_analysis import load_summary_bank  # noqa: E402


def _apply_bh_fdr(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    for _, idx in out.groupby("year").groups.items():
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


def build_year_metapath_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "control", "replicate", "go_id", "metapath", "mean_score"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"Summary dataframe missing required columns: {sorted(missing)}")

    work = summary_df.copy()
    work["year"] = work["year"].astype(int)
    work["control"] = work["control"].astype(str)
    work["replicate"] = work["replicate"].astype(int)
    work["metapath"] = work["metapath"].astype(str)

    real_df = work[work["control"] == "real"].copy()
    if real_df.empty:
        raise ValueError("No real rows found in summary dataframe")
    real_mp = (
        real_df.groupby(["year", "metapath"], as_index=False)
        .agg(real_mean=("mean_score", "mean"), n_real_go_terms=("go_id", "nunique"))
    )

    null_df = work[work["control"] != "real"].copy()
    if null_df.empty:
        raise ValueError("No null rows found in summary dataframe")
    null_rep = (
        null_df.groupby(["year", "control", "replicate", "metapath"], as_index=False)["mean_score"]
        .mean()
        .rename(columns={"mean_score": "null_mean_score"})
    )

    rows: list[dict] = []
    for row in real_mp.itertuples(index=False):
        out_row = {
            "year": int(row.year),
            "metapath": str(row.metapath),
            "real_mean": float(row.real_mean),
            "n_real_go_terms": int(row.n_real_go_terms),
        }
        for control, prefix in [("permuted", "perm"), ("random", "rand")]:
            subset = null_rep[
                (null_rep["year"].astype(int) == int(row.year))
                & (null_rep["control"].astype(str) == control)
                & (null_rep["metapath"].astype(str) == str(row.metapath))
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
    out = _apply_bh_fdr(out, p_col="p_perm", out_col="p_perm_fdr")
    out = _apply_bh_fdr(out, p_col="p_rand", out_col="p_rand_fdr")
    out["supported"] = (
        (out["p_perm_fdr"] < 0.05)
        & (out["p_rand_fdr"] < 0.05)
        & (out["diff_perm"] > 0)
        & (out["diff_rand"] > 0)
    )
    out["fdr_sum"] = out["p_perm_fdr"] + out["p_rand_fdr"]
    return out.sort_values(
        ["year", "supported", "min_d", "min_diff", "fdr_sum", "metapath"],
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
        "--output-path",
        default=str(REPO_ROOT / "output" / "year_direct_metapath_support.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_source = (
        Path(args.summaries_dir)
        if args.summaries_dir
        else Path(args.workspace_dir) / "replicate_summaries"
    )
    summary_df = load_summary_bank(summary_source)
    out_df = build_year_metapath_results(summary_df)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved year metapath support: {output_path}")
    if not out_df.empty:
        support_counts = (
            out_df.groupby("year", as_index=False)["supported"]
            .sum()
            .rename(columns={"supported": "n_supported"})
        )
        for row in support_counts.itertuples(index=False):
            print(f"Year {int(row.year)} supported metapaths: {int(row.n_supported)}")


if __name__ == "__main__":
    main()
