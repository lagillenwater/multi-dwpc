#!/usr/bin/env python3
"""Sampled year-API statistic sensitivity and cross-statistic rank consistency analysis."""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.replicate_analysis import build_b_seed_runs, summarize_feature_variance, summarize_overall_variance  # noqa: E402
from src.result_normalization import (  # noqa: E402
    load_neo4j_mappings,
    normalize_api_year_result,
    parse_year_dataset_name,
)


DEFAULT_RESULTS_DIR = REPO_ROOT / "output" / "dwpc_com" / "all_GO_positive_growth" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "year_api_sampled_statistics"
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _parse_int_list(arg: str) -> list[int]:
    values = [int(tok.strip()) for tok in str(arg).split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _rbo_score(rank_a: list[str], rank_b: list[str], p: float = 0.9) -> float:
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


def _safe_stat(x: pd.Series, fn: str) -> float:
    if len(x) == 0:
        return np.nan
    if fn == "mean":
        return float(x.mean())
    if fn == "median":
        return float(x.median())
    raise ValueError(f"Unsupported fn: {fn}")


def _dataset_stat_summary(df_norm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    meta = df_norm[["domain", "name", "control", "replicate", "year"]].iloc[0].to_dict()
    for (go_id, metapath), group in df_norm.groupby(["go_id", "metapath"], dropna=False):
        dwpc_all = group["dwpc"].dropna()
        dwpc_nonzero = dwpc_all[dwpc_all > 0]

        pval_all = group["p_value"].dropna() if "p_value" in group.columns else pd.Series(dtype=float)
        pval_nonzero = pval_all[pval_all > 0] if len(pval_all) else pd.Series(dtype=float)

        std_all = (
            group["dgp_nonzero_sd"].dropna()
            if "dgp_nonzero_sd" in group.columns
            else pd.Series(dtype=float)
        )
        std_nonzero = std_all[std_all > 0] if len(std_all) else pd.Series(dtype=float)

        stat_values = {
            "mean_dwpc": _safe_stat(dwpc_all, "mean"),
            "mean_dwpc_nonzero": _safe_stat(dwpc_nonzero, "mean"),
            "median_dwpc": _safe_stat(dwpc_all, "median"),
            "median_dwpc_nonzero": _safe_stat(dwpc_nonzero, "median"),
            "mean_pvalue": _safe_stat(pval_all, "mean"),
            "mean_pvalue_nonzero": _safe_stat(pval_nonzero, "mean"),
            "median_pvalue": _safe_stat(pval_all, "median"),
            "median_pvalue_nonzero": _safe_stat(pval_nonzero, "median"),
            "mean_std": _safe_stat(std_all, "mean"),
            "mean_std_nonzero": _safe_stat(std_nonzero, "mean"),
            "median_std": _safe_stat(std_all, "median"),
            "median_std_nonzero": _safe_stat(std_nonzero, "median"),
        }

        for statistic, raw_value in stat_values.items():
            if pd.isna(raw_value):
                continue
            score_value = float(raw_value)
            if "pvalue" in statistic:
                score_value = float(-math.log10(max(score_value, 1e-300)))
            rows.append(
                {
                    **meta,
                    "go_id": str(go_id),
                    "metapath": str(metapath),
                    "statistic": str(statistic),
                    "mean_score": score_value,
                }
            )

    return pd.DataFrame(rows)


def _sample_go_terms(summary_df: pd.DataFrame, n_go_terms: int, seed: int) -> list[str]:
    real_df = summary_df[summary_df["control"].astype(str) == "real"].copy()
    real_2016 = set(real_df[real_df["year"].astype(int) == 2016]["go_id"].astype(str).unique().tolist())
    real_2024 = set(real_df[real_df["year"].astype(int) == 2024]["go_id"].astype(str).unique().tolist())
    pool = sorted(real_2016 & real_2024)
    if len(pool) < n_go_terms:
        pool = sorted(real_2016 | real_2024)
    if not pool:
        raise ValueError("No GO terms available in API real datasets for sampling.")
    sample_n = min(int(n_go_terms), len(pool))
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(np.arange(len(pool), dtype=int), size=sample_n, replace=False)
    return sorted([pool[int(i)] for i in idx])


def _pairwise_rank_consistency(sampled_summary_df: pd.DataFrame) -> pd.DataFrame:
    real_df = sampled_summary_df[sampled_summary_df["control"].astype(str) == "real"].copy()
    rows: list[dict] = []
    for year, year_df in real_df.groupby("year", sort=True):
        stat_to_scores: dict[str, pd.Series] = {}
        stat_to_ranked: dict[str, list[str]] = {}
        for stat, stat_df in year_df.groupby("statistic", sort=True):
            mp_scores = (
                stat_df.groupby("metapath", as_index=False)["mean_score"]
                .mean()
                .sort_values(["mean_score", "metapath"], ascending=[False, True])
                .reset_index(drop=True)
            )
            stat_to_scores[str(stat)] = pd.Series(
                mp_scores["mean_score"].to_numpy(),
                index=mp_scores["metapath"].astype(str).tolist(),
                dtype=float,
            )
            stat_to_ranked[str(stat)] = mp_scores["metapath"].astype(str).tolist()

        for stat_a, stat_b in combinations(sorted(stat_to_scores), 2):
            vec_a = stat_to_scores[stat_a]
            vec_b = stat_to_scores[stat_b]
            common = sorted(set(vec_a.index) & set(vec_b.index))
            if len(common) < 2:
                continue
            rho = float(vec_a.loc[common].rank(method="average").corr(vec_b.loc[common].rank(method="average")))
            rbo = _rbo_score(stat_to_ranked[stat_a], stat_to_ranked[stat_b], p=0.9)
            rows.append(
                {
                    "year": int(year),
                    "statistic_a": stat_a,
                    "statistic_b": stat_b,
                    "n_common_metapaths": int(len(common)),
                    "spearman_rho": rho,
                    "rbo": rbo,
                }
            )
    return pd.DataFrame(rows)


def _build_null_pool(
    sampled_summary_df: pd.DataFrame,
    *,
    target_pool_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a year/control/statistic null-replicate pool up to target size.

    If fewer true API null replicates are available, synthetic replicates are
    generated by bootstrapping donor replicate + donor GO rows within each
    (year, control, statistic) pool.
    """
    if target_pool_size < 1:
        raise ValueError("--target-null-pool-size must be >= 1")

    real_df = sampled_summary_df[sampled_summary_df["control"].astype(str) == "real"].copy()
    null_df = sampled_summary_df[sampled_summary_df["control"].astype(str) != "real"].copy()
    if null_df.empty:
        return sampled_summary_df.copy(), pd.DataFrame()

    rng = np.random.RandomState(int(seed))
    pooled_parts: list[pd.DataFrame] = []
    pool_meta_rows: list[dict] = []

    pool_keys = ["year", "control", "statistic"]
    for key, group in null_df.groupby(pool_keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        year, control, statistic = int(key[0]), str(key[1]), str(key[2])
        group = group.copy()
        existing_rep_ids = sorted(group["replicate"].astype(int).unique().tolist())

        if len(existing_rep_ids) >= int(target_pool_size):
            keep_rep_ids = existing_rep_ids[: int(target_pool_size)]
            kept = group[group["replicate"].astype(int).isin(keep_rep_ids)].copy()
            pooled_parts.append(kept)
            pool_meta_rows.append(
                {
                    "year": year,
                    "control": control,
                    "statistic": statistic,
                    "target_pool_size": int(target_pool_size),
                    "n_true_replicates": int(len(existing_rep_ids)),
                    "n_kept_true_replicates": int(len(keep_rep_ids)),
                    "n_synthetic_replicates": 0,
                }
            )
            continue

        pooled_parts.append(group)
        go_terms = sorted(group["go_id"].astype(str).unique().tolist())
        metapaths = sorted(group["metapath"].astype(str).unique().tolist())
        existing_max_rep = int(max(existing_rep_ids)) if existing_rep_ids else 0
        needed = int(target_pool_size) - len(existing_rep_ids)

        # Cache donor rows by (replicate, go_id) for fast bootstrap lookups.
        donor_cache: dict[tuple[int, str], pd.DataFrame] = {}
        for rep_id, rep_df in group.groupby("replicate", sort=False):
            rep_int = int(rep_id)
            for go_id, go_df in rep_df.groupby("go_id", sort=False):
                donor_cache[(rep_int, str(go_id))] = go_df.copy()

        synthetic_rows: list[pd.DataFrame] = []
        for offset in range(1, needed + 1):
            new_rep = existing_max_rep + offset
            for target_go in go_terms:
                donor_rep = int(rng.choice(existing_rep_ids))
                donor_go = str(rng.choice(go_terms))
                donor = donor_cache.get((donor_rep, donor_go))
                if donor is None or donor.empty:
                    donor = group[group["replicate"].astype(int) == donor_rep].copy()
                donor = donor.copy()
                donor["go_id"] = str(target_go)
                donor["replicate"] = int(new_rep)
                donor["name"] = donor["name"].astype(str) + "_bootstrap"
                donor["replicate_origin"] = "synthetic_bootstrap"
                synthetic_rows.append(donor)

        if synthetic_rows:
            synth_df = pd.concat(synthetic_rows, ignore_index=True)
            synth_df = (
                synth_df.groupby(
                    ["domain", "name", "control", "replicate", "year", "go_id", "metapath", "statistic"],
                    as_index=False,
                )["mean_score"]
                .mean()
            )
            missing_mp = set(metapaths) - set(synth_df["metapath"].astype(str).unique().tolist())
            if missing_mp:
                # Fill any missing metapaths with zeros so pool rows remain rectangular.
                fill_rows = []
                for rep in sorted(synth_df["replicate"].astype(int).unique().tolist()):
                    for go in go_terms:
                        for metapath in sorted(missing_mp):
                            fill_rows.append(
                                {
                                    "domain": "year",
                                    "name": f"bootstrap_fill_{year}_{control}",
                                    "control": control,
                                    "replicate": int(rep),
                                    "year": year,
                                    "go_id": str(go),
                                    "metapath": str(metapath),
                                    "statistic": statistic,
                                    "mean_score": 0.0,
                                }
                            )
                if fill_rows:
                    synth_df = pd.concat([synth_df, pd.DataFrame(fill_rows)], ignore_index=True)
            pooled_parts.append(synth_df)

        pool_meta_rows.append(
            {
                "year": year,
                "control": control,
                "statistic": statistic,
                "target_pool_size": int(target_pool_size),
                "n_true_replicates": int(len(existing_rep_ids)),
                "n_kept_true_replicates": int(len(existing_rep_ids)),
                "n_synthetic_replicates": int(needed),
            }
        )

    pooled_null_df = pd.concat(pooled_parts, ignore_index=True) if pooled_parts else pd.DataFrame()
    pooled_summary_df = pd.concat([real_df, pooled_null_df], ignore_index=True)
    pool_meta_df = pd.DataFrame(pool_meta_rows).sort_values(["year", "control", "statistic"]).reset_index(drop=True)
    return pooled_summary_df, pool_meta_df


def _plot_sensitivity(overall_var_df: pd.DataFrame, out_pdf: Path) -> None:
    if overall_var_df.empty:
        return
    years = sorted(overall_var_df["year"].astype(int).unique().tolist())
    controls = sorted(overall_var_df["control"].astype(str).unique().tolist())
    if not years or not controls:
        return
    fig, axes = plt.subplots(len(years), len(controls), figsize=(6.5 * len(controls), 4.5 * len(years)), sharex=True)
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        if len(years) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    for i, year in enumerate(years):
        for j, control in enumerate(controls):
            ax = axes[i, j]
            subset = overall_var_df[
                (overall_var_df["year"].astype(int) == int(year))
                & (overall_var_df["control"].astype(str) == str(control))
            ].copy()
            for stat, stat_df in subset.groupby("statistic", sort=True):
                stat_df = stat_df.sort_values("b")
                ax.plot(
                    stat_df["b"].astype(int),
                    stat_df["mean_diff_std"].astype(float),
                    marker="o",
                    linewidth=1.6,
                    alpha=0.9,
                    label=str(stat),
                )
            ax.grid(alpha=0.25)
            ax.set_xlabel("B")
            ax.set_ylabel("Mean diff SD")
            ax.set_title(f"{year} | {control}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Statistic")
    fig.suptitle("Statistic sensitivity to null replicate count (API sampled GO terms)")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_consistency_heatmap(pairwise_df: pd.DataFrame, metric: str, out_pdf_prefix: Path) -> None:
    if pairwise_df.empty:
        return
    for year, year_df in pairwise_df.groupby("year", sort=True):
        stats = sorted(set(year_df["statistic_a"].astype(str)) | set(year_df["statistic_b"].astype(str)))
        if not stats:
            continue
        mat = np.full((len(stats), len(stats)), np.nan, dtype=float)
        idx = {stat: i for i, stat in enumerate(stats)}
        for stat in stats:
            mat[idx[stat], idx[stat]] = 1.0
        for row in year_df.itertuples(index=False):
            i = idx[str(row.statistic_a)]
            j = idx[str(row.statistic_b)]
            value = float(getattr(row, metric))
            mat[i, j] = value
            mat[j, i] = value
        fig, ax = plt.subplots(figsize=(0.65 * len(stats) + 3, 0.65 * len(stats) + 2))
        im = ax.imshow(mat, vmin=-1 if metric == "spearman_rho" else 0, vmax=1, cmap="viridis")
        ax.set_xticks(np.arange(len(stats)))
        ax.set_yticks(np.arange(len(stats)))
        ax.set_xticklabels(stats, rotation=90)
        ax.set_yticklabels(stats)
        ax.set_title(f"{metric} consistency across statistics ({year})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(Path(f"{out_pdf_prefix}_{metric}_{int(year)}.pdf"), bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--n-go-terms", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--b-values", default="1,2,5,10,20")
    parser.add_argument("--resample-seeds", default="11,22,33,44,55")
    parser.add_argument("--target-null-pool-size", type=int, default=100)
    parser.add_argument("--null-bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("res_all_GO_positive_growth_*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No API result files found under {results_dir}")

    neo4j_to_entrez, neo4j_to_go = load_neo4j_mappings(data_dir)

    summary_frames: list[pd.DataFrame] = []
    for path in result_files:
        dataset_name = path.stem.replace("res_", "", 1)
        try:
            parse_year_dataset_name(dataset_name)
        except ValueError:
            continue
        raw_df = pd.read_csv(path)
        if "neo4j_target_id" not in raw_df.columns and "neo4j_pseudo_target_id" in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df["neo4j_target_id"] = raw_df["neo4j_pseudo_target_id"]
        norm_df = normalize_api_year_result(
            raw_df,
            dataset_name=dataset_name,
            neo4j_to_entrez=neo4j_to_entrez,
            neo4j_to_go=neo4j_to_go,
        )
        if norm_df.empty:
            continue
        stat_df = _dataset_stat_summary(norm_df)
        if not stat_df.empty:
            summary_frames.append(stat_df)

    if not summary_frames:
        raise ValueError("No normalized API summary rows were generated from result files.")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    sampled_go_terms = _sample_go_terms(summary_df, int(args.n_go_terms), int(args.sample_seed))
    sampled_summary_df = summary_df[summary_df["go_id"].astype(str).isin(sampled_go_terms)].copy()
    pooled_summary_df, pool_meta_df = _build_null_pool(
        sampled_summary_df,
        target_pool_size=int(args.target_null_pool_size),
        seed=int(args.null_bootstrap_seed),
    )

    sampled_summary_df.to_csv(output_dir / "statistic_summary_long.csv", index=False)
    pooled_summary_df.to_csv(output_dir / "statistic_summary_long_with_null_pool.csv", index=False)
    pool_meta_df.to_csv(output_dir / "null_pool_summary.csv", index=False)
    pd.DataFrame({"go_id": sampled_go_terms}).to_csv(output_dir / "sampled_go_terms.csv", index=False)

    requested_b_values = sorted(set(_parse_int_list(args.b_values)))
    resample_seeds = sorted(set(_parse_int_list(args.resample_seeds)))

    runs_frames: list[pd.DataFrame] = []
    for statistic, stat_df in pooled_summary_df.groupby("statistic", sort=True):
        null_df = stat_df[stat_df["control"].astype(str) != "real"].copy()
        if null_df.empty:
            continue
        per_pool_counts = (
            null_df.groupby(["year", "control"], dropna=False)["replicate"]
            .nunique()
            .tolist()
        )
        if not per_pool_counts:
            continue
        max_allowed_b = int(min(per_pool_counts))
        b_values = [b for b in requested_b_values if b <= max_allowed_b]
        if not b_values:
            continue
        stat_runs_df = build_b_seed_runs(
            stat_df,
            b_values=b_values,
            seeds=resample_seeds,
            join_keys=["year", "go_id", "metapath"],
            replicate_pool_keys=["year", "control"],
        )
        if stat_runs_df.empty:
            continue
        stat_runs_df.insert(0, "statistic", str(statistic))
        runs_frames.append(stat_runs_df)

    if not runs_frames:
        raise ValueError("No B/seed runs were generated. Check available API null replicates and B values.")

    runs_df = pd.concat(runs_frames, ignore_index=True)
    runs_df.to_csv(output_dir / "b_seed_runs_long.csv", index=False)

    feature_df = summarize_feature_variance(
        runs_df,
        feature_keys=["statistic", "year", "control", "b", "go_id", "metapath"],
    )
    overall_df = summarize_overall_variance(
        feature_df,
        overall_keys=["statistic", "year", "control", "b"],
        runs_df=runs_df,
        replicate_col="seed",
    )
    feature_df.to_csv(output_dir / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(output_dir / "overall_variance_summary.csv", index=False)

    sens_rows: list[dict] = []
    for (statistic, year, control), grp in overall_df.groupby(["statistic", "year", "control"], sort=True):
        grp = grp.sort_values("b")
        if grp.empty:
            continue
        first = grp.iloc[0]
        last = grp.iloc[-1]
        sens_rows.append(
            {
                "statistic": str(statistic),
                "year": int(year),
                "control": str(control),
                "b_min": int(first["b"]),
                "b_max": int(last["b"]),
                "mean_diff_std_b_min": float(first["mean_diff_std"]),
                "mean_diff_std_b_max": float(last["mean_diff_std"]),
                "mean_diff_var_b_min": float(first["mean_diff_var"]),
                "mean_diff_var_b_max": float(last["mean_diff_var"]),
            }
        )
    pd.DataFrame(sens_rows).to_csv(output_dir / "statistic_sensitivity_summary.csv", index=False)

    pairwise_df = _pairwise_rank_consistency(sampled_summary_df)
    pairwise_df.to_csv(output_dir / "statistic_rank_consistency_pairwise.csv", index=False)
    if not pairwise_df.empty:
        summary_rank_df = (
            pairwise_df.groupby(["year", "statistic_a"], as_index=False)
            .agg(
                n_pairs=("spearman_rho", "size"),
                mean_spearman_rho=("spearman_rho", "mean"),
                mean_rbo=("rbo", "mean"),
            )
            .sort_values(["year", "statistic_a"])
            .reset_index(drop=True)
        )
    else:
        summary_rank_df = pd.DataFrame(
            columns=["year", "statistic_a", "n_pairs", "mean_spearman_rho", "mean_rbo"]
        )
    summary_rank_df.to_csv(output_dir / "statistic_rank_consistency_summary.csv", index=False)

    _plot_sensitivity(overall_df, output_dir / "variance_sensitivity_by_statistic.pdf")
    _plot_consistency_heatmap(
        pairwise_df,
        metric="spearman_rho",
        out_pdf_prefix=output_dir / "statistic_rank_consistency",
    )
    _plot_consistency_heatmap(
        pairwise_df,
        metric="rbo",
        out_pdf_prefix=output_dir / "statistic_rank_consistency",
    )

    print(f"Sampled GO terms: {len(sampled_go_terms)}")
    print(f"Statistics present: {sampled_summary_df['statistic'].nunique()}")
    if not pool_meta_df.empty:
        synth_total = int(pool_meta_df["n_synthetic_replicates"].sum())
        print(f"Synthetic null replicates added across pools: {synth_total}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
