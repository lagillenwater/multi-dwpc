#!/usr/bin/env python3
"""Build GO-term-level and global metapath support tables from year replicate summaries."""

from __future__ import annotations

import argparse
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
    groupby_arg: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    for _, idx in out.groupby(groupby_arg).groups.items():
        subset = out.loc[idx]
        pvals = subset[p_col].to_numpy(dtype=float)
        valid = np.isfinite(pvals)
        if valid.sum() == 0:
            continue
        corrected = np.full(len(subset), np.nan, dtype=float)
        corrected[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
        out.loc[idx, out_col] = corrected
    return out


def _safe_effect_size(real_mean: pd.Series, null_mean: pd.Series, null_std: pd.Series) -> pd.Series:
    diff = real_mean.astype(float) - null_mean.astype(float)
    out = pd.Series(np.nan, index=diff.index, dtype=float)

    valid_std = null_std.notna() & (null_std.astype(float) > 0)
    out.loc[valid_std] = diff.loc[valid_std] / null_std.loc[valid_std].astype(float)

    zero_std = null_std.fillna(0.0).astype(float) <= 0
    out.loc[zero_std & (diff > 0)] = np.inf
    out.loc[zero_std & (diff < 0)] = -np.inf
    out.loc[zero_std & (diff == 0)] = 0.0
    return out


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

    # Join all null replicate summaries to the real table once, then compute null
    # summary statistics and empirical p-values via grouped aggregations.
    null_vs_real = null_rep.merge(real_scores, on=["year", "go_id", "metapath"], how="inner")
    if null_vs_real.empty:
        return pd.DataFrame()
    null_vs_real["null_ge_real"] = (
        null_vs_real["null_mean_score"].astype(float) >= null_vs_real["real_mean"].astype(float)
    ).astype(int)

    grouped = (
        null_vs_real.groupby(["year", "go_id", "metapath", "control"], as_index=False)
        .agg(
            real_mean=("real_mean", "first"),
            null_mean=("null_mean_score", "mean"),
            null_std=("null_mean_score", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
            n_ge_real=("null_ge_real", "sum"),
            b_eff=("null_mean_score", "size"),
        )
    )
    grouped["p_empirical"] = (1.0 + grouped["n_ge_real"].astype(float)) / (
        grouped["b_eff"].astype(float) + 1.0
    )

    out = real_scores.copy()
    for control, prefix in [("permuted", "perm"), ("random", "rand")]:
        sub = grouped[grouped["control"].astype(str) == control].copy()
        if sub.empty:
            continue
        sub = sub.rename(
            columns={
                "null_mean": f"{prefix}_null_mean",
                "null_std": f"{prefix}_null_std",
                "p_empirical": f"p_{prefix}",
                "b_eff": f"b_eff_{prefix}",
            }
        )
        out = out.merge(
            sub[
                [
                    "year",
                    "go_id",
                    "metapath",
                    "real_mean",
                    f"{prefix}_null_mean",
                    f"{prefix}_null_std",
                    f"p_{prefix}",
                    f"b_eff_{prefix}",
                ]
            ],
            on=["year", "go_id", "metapath", "real_mean"],
            how="left",
        )

    out["diff_perm"] = out["real_mean"] - out["perm_null_mean"]
    out["diff_rand"] = out["real_mean"] - out["rand_null_mean"]
    out["d_perm"] = _safe_effect_size(out["real_mean"], out["perm_null_mean"], out["perm_null_std"])
    out["d_rand"] = _safe_effect_size(out["real_mean"], out["rand_null_mean"], out["rand_null_std"])
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


def build_go_term_support_from_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "control", "b", "seed", "go_id", "metapath", "real_mean_score", "null_mean_score"}
    missing = required - set(runs_df.columns)
    if missing:
        raise ValueError(f"Runs dataframe missing required columns: {sorted(missing)}")

    work = runs_df.copy()
    work["year"] = work["year"].astype(int)
    work["control"] = work["control"].astype(str)
    work["b"] = work["b"].astype(int)
    work["seed"] = work["seed"].astype(int)
    work["go_id"] = work["go_id"].astype(str)
    work["metapath"] = work["metapath"].astype(str)

    grouped = (
        work.groupby(["year", "go_id", "metapath", "control", "b"], as_index=False)
        .agg(
            real_mean=("real_mean_score", "first"),
            null_mean=("null_mean_score", "mean"),
            null_std=("null_mean_score", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
            b_eff=("null_mean_score", "size"),
        )
    )
    # Recompute the empirical count after aggregation using a vectorized pass to avoid
    # relying on groupby lambdas with external state.
    compare_df = work.copy()
    compare_df["null_ge_real"] = (
        compare_df["null_mean_score"].astype(float) >= compare_df["real_mean_score"].astype(float)
    ).astype(int)
    ge_counts = (
        compare_df.groupby(["year", "go_id", "metapath", "control", "b"], as_index=False)["null_ge_real"]
        .sum()
        .rename(columns={"null_ge_real": "n_ge_real"})
    )
    grouped = grouped.drop(columns=["n_ge_real"]).merge(
        ge_counts,
        on=["year", "go_id", "metapath", "control", "b"],
        how="left",
    )
    grouped["p_empirical"] = (1.0 + grouped["n_ge_real"].astype(float)) / (
        grouped["b_eff"].astype(float) + 1.0
    )

    base = (
        work[["year", "go_id", "metapath", "b", "real_mean_score"]]
        .drop_duplicates()
        .rename(columns={"real_mean_score": "real_mean"})
    )
    out = base.copy()
    for control, prefix in [("permuted", "perm"), ("random", "rand")]:
        sub = grouped[grouped["control"].astype(str) == control].copy()
        if sub.empty:
            continue
        sub = sub.rename(
            columns={
                "null_mean": f"{prefix}_null_mean",
                "null_std": f"{prefix}_null_std",
                "p_empirical": f"p_{prefix}",
                "b_eff": f"b_eff_{prefix}",
            }
        )
        out = out.merge(
            sub[
                [
                    "year",
                    "go_id",
                    "metapath",
                    "b",
                    "real_mean",
                    f"{prefix}_null_mean",
                    f"{prefix}_null_std",
                    f"p_{prefix}",
                    f"b_eff_{prefix}",
                ]
            ],
            on=["year", "go_id", "metapath", "b", "real_mean"],
            how="left",
        )

    out["diff_perm"] = out["real_mean"] - out["perm_null_mean"]
    out["diff_rand"] = out["real_mean"] - out["rand_null_mean"]
    out["d_perm"] = _safe_effect_size(out["real_mean"], out["perm_null_mean"], out["perm_null_std"])
    out["d_rand"] = _safe_effect_size(out["real_mean"], out["rand_null_mean"], out["rand_null_std"])
    out["min_diff"] = np.minimum(out["diff_perm"], out["diff_rand"])
    out["min_d"] = np.minimum(out["d_perm"], out["d_rand"])
    out = _apply_bh_fdr(out, p_col="p_perm", out_col="p_perm_fdr", group_cols=["year", "b", "go_id"])
    out = _apply_bh_fdr(out, p_col="p_rand", out_col="p_rand_fdr", group_cols=["year", "b", "go_id"])
    out["supported"] = (
        (out["p_perm_fdr"] < 0.05)
        & (out["p_rand_fdr"] < 0.05)
        & (out["diff_perm"] > 0)
        & (out["diff_rand"] > 0)
    )
    out["fdr_sum"] = out["p_perm_fdr"] + out["p_rand_fdr"]
    return out.sort_values(
        ["b", "year", "go_id", "supported", "min_d", "min_diff", "fdr_sum", "metapath"],
        ascending=[True, True, True, False, False, False, True, True],
    ).reset_index(drop=True)


def build_global_metapath_support(go_support_df: pd.DataFrame) -> pd.DataFrame:
    if go_support_df.empty:
        return pd.DataFrame()

    group_cols = ["year", "metapath"]
    if "b" in go_support_df.columns:
        group_cols = ["b", *group_cols]

    agg = (
        go_support_df.groupby(group_cols, as_index=False)
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
    fdr_group_cols = ["year"]
    if "b" in agg.columns:
        fdr_group_cols = ["b", "year"]
    agg = _apply_bh_fdr(agg, p_col="combined_p_perm", out_col="combined_p_perm_fdr", group_cols=fdr_group_cols)
    agg = _apply_bh_fdr(agg, p_col="combined_p_rand", out_col="combined_p_rand_fdr", group_cols=fdr_group_cols)
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
    parser.add_argument("--runs-path", default=None, help="Optional B/seed runs file from year_null_variance_experiment/all_runs_long.csv")
    parser.add_argument("--b", type=int, default=None, help="Optional B to filter when using --runs-path")
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
    if args.runs_path:
        runs_df = pd.read_csv(Path(args.runs_path))
        if args.b is not None:
            runs_df = runs_df[runs_df["b"].astype(int) == int(args.b)].copy()
        go_df = build_go_term_support_from_runs(runs_df)
    else:
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
