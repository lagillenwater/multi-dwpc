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


def _effective_number_from_scores(scores: pd.Series) -> float:
    vals = scores.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


def _add_consensus_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["year", "go_id"]
    if "b" in out.columns:
        group_cols = ["b", *group_cols]

    out["rank_perm"] = (
        out.groupby(group_cols)["diff_perm"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    out["rank_rand"] = (
        out.groupby(group_cols)["diff_rand"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    out["consensus_rank"] = 0.5 * (out["rank_perm"] + out["rank_rand"])
    out["consensus_score"] = 0.5 * (
        (1.0 / out["rank_perm"].replace(0, np.nan))
        + (1.0 / out["rank_rand"].replace(0, np.nan))
    )
    return out


def _add_effective_n_selection(
    df: pd.DataFrame,
    *,
    score_col: str,
    indicator_col: str,
    effective_n_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out[indicator_col] = False
    out[effective_n_col] = np.nan
    group_cols = ["year", "go_id"]
    if "b" in out.columns:
        group_cols = ["b", *group_cols]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        group = out.loc[idx].copy()
        if group.empty:
            continue
        scores = group[score_col].astype(float).fillna(0.0).clip(lower=0.0)
        eff_n = _effective_number_from_scores(scores)
        k = max(1, int(np.ceil(eff_n)))
        ranked = group.assign(_score=scores.to_numpy()).sort_values(
            ["_score", "consensus_rank", "metapath"],
            ascending=[False, True, True],
        )
        selected_idx = ranked.head(k).index
        out.loc[idx, effective_n_col] = float(eff_n)
        out.loc[selected_idx, indicator_col] = True
    return out


def _add_dual_null_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diff_perm"] = out["real_mean"] - out["perm_null_mean"]
    out["diff_rand"] = out["real_mean"] - out["rand_null_mean"]
    out["d_perm"] = _safe_effect_size(out["real_mean"], out["perm_null_mean"], out["perm_null_std"])
    out["d_rand"] = _safe_effect_size(out["real_mean"], out["rand_null_mean"], out["rand_null_std"])
    out["min_diff"] = np.minimum(out["diff_perm"], out["diff_rand"])
    out["min_d"] = np.minimum(out["d_perm"], out["d_rand"])
    out["mean_std_score"] = 0.5 * (out["d_perm"] + out["d_rand"])
    return out


def _aggregate_global_summary(go_support_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    group_cols = ["year", "metapath"]
    if "b" in go_support_df.columns:
        group_cols = ["b", *group_cols]

    summary = (
        go_support_df.groupby(group_cols, as_index=False)
        .agg(
            n_go_terms=("go_id", "nunique"),
            mean_real_mean=("real_mean", "mean"),
            mean_diff_perm=("diff_perm", "mean"),
            mean_diff_rand=("diff_rand", "mean"),
            mean_min_diff=("min_diff", "mean"),
            mean_min_d=("min_d", "mean"),
            mean_std_score=("mean_std_score", "mean"),
            mean_consensus_score=("consensus_score", "mean"),
            median_consensus_rank=("consensus_rank", "median"),
            median_min_d=("min_d", "median"),
            median_std_score=("mean_std_score", "median"),
        )
    )
    rename_map = {
        "n_go_terms": f"n_go_terms_{suffix}",
        "mean_real_mean": f"mean_real_mean_{suffix}",
        "mean_diff_perm": f"mean_diff_perm_{suffix}",
        "mean_diff_rand": f"mean_diff_rand_{suffix}",
        "mean_min_diff": f"mean_min_diff_{suffix}",
        "mean_min_d": f"mean_min_d_{suffix}",
        "mean_std_score": f"mean_std_score_{suffix}",
        "mean_consensus_score": f"mean_consensus_score_{suffix}",
        "median_consensus_rank": f"median_consensus_rank_{suffix}",
        "median_min_d": f"median_min_d_{suffix}",
        "median_std_score": f"median_std_score_{suffix}",
    }
    return summary.rename(columns=rename_map)


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

    out = _add_dual_null_summary_columns(out)
    out = _apply_bh_fdr(out, p_col="p_perm", out_col="p_perm_fdr", group_cols=["year", "go_id"])
    out = _apply_bh_fdr(out, p_col="p_rand", out_col="p_rand_fdr", group_cols=["year", "go_id"])
    out["supported"] = (
        (out["p_perm_fdr"] < 0.05)
        & (out["p_rand_fdr"] < 0.05)
        & (out["diff_perm"] > 0)
        & (out["diff_rand"] > 0)
    )
    out["fdr_sum"] = out["p_perm_fdr"] + out["p_rand_fdr"]
    out = _add_consensus_rank_columns(out)
    out = _add_effective_n_selection(
        out,
        score_col="consensus_score",
        indicator_col="selected_by_effective_n_all",
        effective_n_col="effective_n_all",
    )
    supported_subset = out[out["supported"]].copy()
    if not supported_subset.empty:
        supported_subset = _add_effective_n_selection(
            supported_subset,
            score_col="consensus_score",
            indicator_col="selected_by_effective_n_supported_only",
            effective_n_col="effective_n_supported_only",
        )
        out = out.merge(
            supported_subset[
                [
                    "year",
                    "go_id",
                    "metapath",
                    "selected_by_effective_n_supported_only",
                    "effective_n_supported_only",
                ]
            ],
            on=["year", "go_id", "metapath"],
            how="left",
        )
    else:
        out["selected_by_effective_n_supported_only"] = np.nan
        out["effective_n_supported_only"] = np.nan
    return out.sort_values(
        ["year", "go_id", "selected_by_effective_n_all", "consensus_score", "supported", "fdr_sum", "metapath"],
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

    compare_df = work.copy()
    compare_df["null_ge_real"] = (
        compare_df["null_mean_score"].astype(float) >= compare_df["real_mean_score"].astype(float)
    ).astype(int)

    ge_counts = (
        compare_df.groupby(["year", "go_id", "metapath", "control", "b"], as_index=False)["null_ge_real"]
        .sum()
        .rename(columns={"null_ge_real": "n_ge_real"})
    )

    grouped = grouped.merge(
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

    out = _add_dual_null_summary_columns(out)

    out = _apply_bh_fdr(out, p_col="p_perm", out_col="p_perm_fdr", group_cols=["year", "b"])
    out = _apply_bh_fdr(out, p_col="p_rand", out_col="p_rand_fdr", group_cols=["year", "b"])

    out["supported"] = (
        (out["p_perm_fdr"] < 0.05)
        & (out["p_rand_fdr"] < 0.05)
        & (out["diff_perm"] > 0)
        & (out["diff_rand"] > 0)
    )
    out["fdr_sum"] = out["p_perm_fdr"] + out["p_rand_fdr"]
    out = _add_consensus_rank_columns(out)
    out = _add_effective_n_selection(
        out,
        score_col="consensus_score",
        indicator_col="selected_by_effective_n_all",
        effective_n_col="effective_n_all",
    )
    supported_subset = out[out["supported"]].copy()
    if not supported_subset.empty:
        supported_subset = _add_effective_n_selection(
            supported_subset,
            score_col="consensus_score",
            indicator_col="selected_by_effective_n_supported_only",
            effective_n_col="effective_n_supported_only",
        )
        out = out.merge(
            supported_subset[
                [
                    "b",
                    "year",
                    "go_id",
                    "metapath",
                    "selected_by_effective_n_supported_only",
                    "effective_n_supported_only",
                ]
            ],
            on=["b", "year", "go_id", "metapath"],
            how="left",
        )
    else:
        out["selected_by_effective_n_supported_only"] = np.nan
        out["effective_n_supported_only"] = np.nan

    return out.sort_values(
        ["b", "year", "go_id", "selected_by_effective_n_all", "consensus_score", "supported", "fdr_sum", "metapath"],
        ascending=[True, True, True, False, False, False, True, True],
    ).reset_index(drop=True)

def build_global_metapath_support(go_support_df: pd.DataFrame) -> pd.DataFrame:
    if go_support_df.empty:
        return pd.DataFrame()

    group_cols = ["year", "metapath"]
    if "b" in go_support_df.columns:
        group_cols = ["b", *group_cols]

    agg_full = _aggregate_global_summary(go_support_df, suffix="all")
    supported_only = go_support_df[go_support_df["supported"].fillna(False)].copy()
    if supported_only.empty:
        agg_supported = agg_full[group_cols].copy()
        for col in [
            "n_go_terms_supported_only",
            "mean_real_mean_supported_only",
            "mean_diff_perm_supported_only",
            "mean_diff_rand_supported_only",
            "mean_min_diff_supported_only",
            "mean_min_d_supported_only",
            "mean_std_score_supported_only",
            "median_min_d_supported_only",
            "median_std_score_supported_only",
        ]:
            agg_supported[col] = np.nan
    else:
        agg_supported = _aggregate_global_summary(supported_only, suffix="supported_only")

    agg_inferential = (
        go_support_df.groupby(group_cols, as_index=False)
        .agg(
            combined_p_perm=("p_perm", _combine_pvalues_fisher),
            combined_p_rand=("p_rand", _combine_pvalues_fisher),
            n_go_terms_supported=("supported", "sum"),
            n_go_terms_selected_all=("selected_by_effective_n_all", "sum"),
        )
    )
    supported_selected = (
        go_support_df[go_support_df["selected_by_effective_n_supported_only"].fillna(False)]
        .groupby(group_cols, as_index=False)
        .agg(n_go_terms_selected_supported_only=("go_id", "nunique"))
    )

    agg = agg_full.merge(agg_supported, on=group_cols, how="left").merge(
        agg_inferential, on=group_cols, how="left"
    )
    agg = agg.merge(supported_selected, on=group_cols, how="left")
    agg["support_fraction"] = agg["n_go_terms_supported"] / agg["n_go_terms_all"].replace(0, np.nan)
    agg["selected_fraction_all"] = agg["n_go_terms_selected_all"] / agg["n_go_terms_all"].replace(0, np.nan)
    agg["selected_fraction_supported_only"] = (
        agg["n_go_terms_selected_supported_only"] / agg["n_go_terms_supported_only"].replace(0, np.nan)
    )
    fdr_group_cols = ["year"]
    if "b" in agg.columns:
        fdr_group_cols = ["b", "year"]
    agg = _apply_bh_fdr(agg, p_col="combined_p_perm", out_col="combined_p_perm_fdr", group_cols=fdr_group_cols)
    agg = _apply_bh_fdr(agg, p_col="combined_p_rand", out_col="combined_p_rand_fdr", group_cols=fdr_group_cols)
    agg["supported_global"] = (
        (agg["combined_p_perm_fdr"] < 0.05)
        & (agg["combined_p_rand_fdr"] < 0.05)
        & (agg["mean_diff_perm_all"] > 0)
        & (agg["mean_diff_rand_all"] > 0)
    )
    agg["combined_fdr_sum"] = agg["combined_p_perm_fdr"] + agg["combined_p_rand_fdr"]
    return agg.sort_values(
        ["year", "selected_fraction_all", "mean_consensus_score_all", "supported_global", "combined_fdr_sum", "metapath"],
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
