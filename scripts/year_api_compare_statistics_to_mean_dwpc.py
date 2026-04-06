#!/usr/bin/env python3
"""Compare year statistics to mean DWPC using real-minus-control profiles."""

from __future__ import annotations

import argparse
import math
import sys
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
from src.result_normalization import load_normalized_year_results, parse_year_dataset_name  # noqa: E402
from src.year_statistics import build_aggregated_year_statistics_panel, detect_supported_statistics  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "data"
ANCHOR_STAT = "mean_dwpc"


def _default_results_dir(score_source: str) -> Path:
    if str(score_source) == "api":
        return REPO_ROOT / "output" / "dwpc_com" / "all_GO_positive_growth" / "results"
    return REPO_ROOT / "output" / "dwpc_direct" / "all_GO_positive_growth" / "results"


def _default_output_dir(score_source: str) -> Path:
    if str(score_source) == "api":
        return REPO_ROOT / "output" / "year_api_stat_compare"
    return REPO_ROOT / "output" / "year_direct_stat_compare"


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


def _transform_stat(stat: str, value: float) -> float:
    if pd.isna(value):
        return np.nan
    if "pvalue" in str(stat):
        return float(-math.log10(max(float(value), 1e-300)))
    return float(value)


def _build_real_minus_control_profiles(
    agg_df: pd.DataFrame,
    statistics: list[str],
) -> pd.DataFrame:
    """Build per-(year,control,go,metapath,statistic) diff = real - mean(control)."""
    work = agg_df.copy()
    if work.empty:
        return pd.DataFrame()

    meta_rows = []
    for dataset in sorted(work["dataset"].astype(str).unique().tolist()):
        parsed = parse_year_dataset_name(dataset)
        meta_rows.append(
            {
                "dataset": dataset,
                "year": int(parsed["year"]),
                "control": str(parsed["control"]),
                "replicate": int(parsed["replicate"]),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    work = work.merge(meta_df, on="dataset", how="left")

    diff_frames: list[pd.DataFrame] = []
    for stat in statistics:
        if stat not in work.columns:
            continue
        stat_df = work[["year", "control", "replicate", "go_id", "metapath", stat]].copy()
        stat_df = stat_df.rename(columns={stat: "value"}).dropna(subset=["value"])
        if stat_df.empty:
            continue
        stat_df["value"] = stat_df["value"].map(lambda v: _transform_stat(stat, v))

        real_df = stat_df[stat_df["control"].astype(str) == "real"].copy()
        if real_df.empty:
            continue
        real_df = (
            real_df.groupby(["year", "go_id", "metapath"], as_index=False)["value"]
            .mean()
            .rename(columns={"value": "real_value"})
        )

        null_df = stat_df[stat_df["control"].astype(str) != "real"].copy()
        if null_df.empty:
            continue
        null_df = (
            null_df.groupby(["year", "control", "go_id", "metapath"], as_index=False)["value"]
            .mean()
            .rename(columns={"value": "null_value"})
        )

        merged = null_df.merge(real_df, on=["year", "go_id", "metapath"], how="inner")
        if merged.empty:
            continue
        merged["diff"] = merged["real_value"] - merged["null_value"]
        merged["statistic"] = str(stat)
        merged["comparison"] = (
            merged["year"].astype(int).astype(str)
            + "_real_minus_"
            + merged["control"].astype(str)
        )
        diff_frames.append(
            merged[
                [
                    "year",
                    "control",
                    "comparison",
                    "go_id",
                    "metapath",
                    "statistic",
                    "real_value",
                    "null_value",
                    "diff",
                ]
            ]
        )

    if not diff_frames:
        return pd.DataFrame()
    return pd.concat(diff_frames, ignore_index=True)


def _compute_rank_consistency(
    diff_df: pd.DataFrame,
    statistics: list[str],
    rbo_p: float,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (year, control, comparison), cmp_df in diff_df.groupby(["year", "control", "comparison"], sort=True):
        anchor_df = cmp_df[cmp_df["statistic"].astype(str) == ANCHOR_STAT].copy()
        if anchor_df.empty:
            continue
        anchor_mp = (
            anchor_df.groupby("metapath", as_index=False)["diff"]
            .mean()
            .dropna()
            .sort_values(["diff", "metapath"], ascending=[False, True])
        )
        anchor_scores = pd.Series(
            anchor_mp["diff"].to_numpy(dtype=float),
            index=anchor_mp["metapath"].astype(str).tolist(),
            dtype=float,
        )
        anchor_ranked = anchor_mp["metapath"].astype(str).tolist()
        if len(anchor_ranked) < 2:
            continue
        anchor_rank_pos = {metapath: idx + 1 for idx, metapath in enumerate(anchor_ranked)}
        anchor_top = anchor_ranked[: int(top_n)]

        for stat in statistics:
            if stat == ANCHOR_STAT:
                continue
            stat_df = cmp_df[cmp_df["statistic"].astype(str) == str(stat)].copy()
            if stat_df.empty:
                continue
            stat_mp = (
                stat_df.groupby("metapath", as_index=False)["diff"]
                .mean()
                .dropna()
                .sort_values(["diff", "metapath"], ascending=[False, True])
            )
            stat_scores = pd.Series(
                stat_mp["diff"].to_numpy(dtype=float),
                index=stat_mp["metapath"].astype(str).tolist(),
                dtype=float,
            )
            stat_ranked = stat_mp["metapath"].astype(str).tolist()
            stat_rank_pos = {metapath: idx + 1 for idx, metapath in enumerate(stat_ranked)}
            stat_top = stat_ranked[: int(top_n)]

            common = sorted(set(anchor_scores.index) & set(stat_scores.index))
            if len(common) < 2:
                continue
            rho = float(
                anchor_scores.loc[common].rank(method="average").corr(
                    stat_scores.loc[common].rank(method="average")
                )
            )
            rbo = _rbo_score(anchor_ranked, stat_ranked, p=rbo_p)

            top_union = set(anchor_top) | set(stat_top)
            top_intersection = set(anchor_top) & set(stat_top)
            top_jaccard = (
                float(len(top_intersection)) / float(len(top_union))
                if top_union
                else np.nan
            )
            common_top = sorted(set(anchor_top) & set(stat_top))
            if len(common_top) >= 2:
                top_rho = float(
                    pd.Series([anchor_rank_pos[m] for m in common_top], dtype=float).rank(method="average").corr(
                        pd.Series([stat_rank_pos[m] for m in common_top], dtype=float).rank(method="average")
                    )
                )
            else:
                top_rho = np.nan
            top_rbo = _rbo_score(anchor_top, stat_top, p=rbo_p)

            rows.append(
                {
                    "year": int(year),
                    "control": str(control),
                    "comparison": str(comparison),
                    "anchor_statistic": ANCHOR_STAT,
                    "statistic": str(stat),
                    "top_n": int(top_n),
                    "n_common_metapaths": int(len(common)),
                    "spearman_rho": rho,
                    "rbo": rbo,
                    "n_common_top_paths": int(len(common_top)),
                    "top_n_jaccard": top_jaccard,
                    "top_n_spearman_rho": top_rho,
                    "top_n_rbo": top_rbo,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["year", "control", "statistic"]).reset_index(drop=True)


def _compute_variance_consistency(
    diff_df: pd.DataFrame,
    statistics: list[str],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (year, control, comparison), cmp_df in diff_df.groupby(["year", "control", "comparison"], sort=True):
        anchor_df = cmp_df[cmp_df["statistic"].astype(str) == ANCHOR_STAT].copy()
        if anchor_df.empty:
            continue
        anchor_rank = (
            anchor_df.groupby("metapath", as_index=False)["diff"]
            .mean()
            .sort_values(["diff", "metapath"], ascending=[False, True])
        )
        anchor_top_paths = set(anchor_rank.head(int(top_n))["metapath"].astype(str).tolist())
        if not anchor_top_paths:
            continue

        anchor_var = (
            anchor_df.groupby("metapath", as_index=False)["diff"]
            .var(ddof=1)
            .rename(columns={"diff": "anchor_var"})
            .dropna()
        )
        anchor_var = anchor_var[anchor_var["metapath"].astype(str).isin(anchor_top_paths)].copy()
        if anchor_var.empty:
            continue

        for stat in statistics:
            if stat == ANCHOR_STAT:
                continue
            stat_df = cmp_df[cmp_df["statistic"].astype(str) == str(stat)].copy()
            if stat_df.empty:
                continue
            stat_var = (
                stat_df.groupby("metapath", as_index=False)["diff"]
                .var(ddof=1)
                .rename(columns={"diff": "stat_var"})
                .dropna()
            )
            merged = anchor_var.merge(stat_var, on="metapath", how="inner")
            if len(merged) < 2:
                continue
            pearson = float(merged["anchor_var"].corr(merged["stat_var"], method="pearson"))
            spearman = float(
                merged["anchor_var"].rank(method="average").corr(
                    merged["stat_var"].rank(method="average")
                )
            )
            rows.append(
                {
                    "year": int(year),
                    "control": str(control),
                    "comparison": str(comparison),
                    "anchor_statistic": ANCHOR_STAT,
                    "statistic": str(stat),
                    "top_n": int(top_n),
                    "n_common_metapaths": int(len(merged)),
                    "pearson_var_corr": pearson,
                    "spearman_var_corr": spearman,
                    "anchor_var_mean": float(merged["anchor_var"].mean()),
                    "stat_var_mean": float(merged["stat_var"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["year", "control", "statistic"]).reset_index(drop=True)


def _plot_metric_lines(df: pd.DataFrame, metric_col: str, out_pdf: Path, title: str) -> None:
    if df.empty:
        return
    work = df[["comparison", "statistic", metric_col]].dropna().copy()
    if work.empty:
        return
    statistics = sorted(work["statistic"].astype(str).unique().tolist())
    x = np.arange(len(statistics))
    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(statistics) + 4), 6))
    for comparison, cmp_df in work.groupby("comparison", sort=True):
        y_vals = []
        for stat in statistics:
            row = cmp_df[cmp_df["statistic"].astype(str) == str(stat)]
            y_vals.append(float(row.iloc[0][metric_col]) if not row.empty else np.nan)
        ax.plot(x, y_vals, marker="o", linewidth=1.8, alpha=0.9, label=str(comparison))
    ax.set_xticks(x)
    ax.set_xticklabels(statistics, rotation=90)
    ax.set_ylim(-1.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.legend(title="Comparison", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    out_pdf: Path,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    if df.empty:
        return
    work = df[["comparison", "statistic", metric_col]].dropna().copy()
    if work.empty:
        return
    pivot = work.pivot(index="comparison", columns="statistic", values=metric_col)
    if pivot.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(8.5, 0.7 * len(pivot.columns) + 3), max(4.5, 0.55 * len(pivot.index) + 2))
    )
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_variance_scatter_by_comparison(
    diff_df: pd.DataFrame,
    statistics: list[str],
    out_dir: Path,
) -> None:
    for comparison, cmp_df in diff_df.groupby("comparison", sort=True):
        anchor_df = cmp_df[cmp_df["statistic"].astype(str) == ANCHOR_STAT].copy()
        if anchor_df.empty:
            continue
        anchor_var = (
            anchor_df.groupby("metapath", as_index=False)["diff"]
            .var(ddof=1)
            .rename(columns={"diff": "anchor_var"})
            .dropna()
        )
        if anchor_var.empty:
            continue
        compare_stats = [s for s in statistics if s != ANCHOR_STAT]
        if not compare_stats:
            continue

        n_cols = min(3, len(compare_stats))
        n_rows = int(math.ceil(len(compare_stats) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 4.8 * n_rows),
            sharex=False,
            sharey=False,
        )
        axes = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)

        idx = 0
        for stat in compare_stats:
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]
            stat_series = cmp_df[cmp_df["statistic"].astype(str) == str(stat)][["metapath", "diff"]].copy()
            if stat_series.empty:
                ax.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(stat)
                ax.set_xlabel(f"Var({ANCHOR_STAT})")
                ax.set_ylabel(f"Var({stat})")
                ax.grid(alpha=0.2)
                idx += 1
                continue
            stat_var = (
                stat_series.groupby("metapath", as_index=False)["diff"]
                .var(ddof=1)
                .rename(columns={"diff": "stat_var"})
                .dropna()
            )
            merged = anchor_var.merge(stat_var, on="metapath", how="inner")
            if len(merged) >= 2:
                ax.scatter(
                    merged["anchor_var"],
                    merged["stat_var"],
                    s=18,
                    alpha=0.55,
                    color="#1f77b4",
                    edgecolors="none",
                )
                rho = merged["anchor_var"].rank(method="average").corr(
                    merged["stat_var"].rank(method="average")
                )
                ax.set_title(f"{stat}\nSpearman={rho:.3f}, n={len(merged)}")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "insufficient overlap",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(stat)
            ax.set_xlabel(f"Var({ANCHOR_STAT})")
            ax.set_ylabel(f"Var({stat})")
            ax.grid(alpha=0.2)
            idx += 1

        while idx < n_rows * n_cols:
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")
            idx += 1

        fig.suptitle(f"Metapath diff-variance vs {ANCHOR_STAT}: {comparison}")
        fig.tight_layout()
        fig.savefig(out_dir / f"variance_scatter_vs_{ANCHOR_STAT}_{comparison}.pdf", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-source", default="api", choices=["direct", "api"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rbo-p", type=float, default=0.9)
    parser.add_argument("--top-n-paths", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir(args.score_source)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.score_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_df = load_normalized_year_results(
        results_dir,
        score_source=str(args.score_source),
        data_dir=data_dir,
    )
    agg_df = build_aggregated_year_statistics_panel(normalized_df)
    if agg_df.empty:
        raise ValueError("Aggregated year statistics are empty.")

    statistics = detect_supported_statistics(agg_df)
    if ANCHOR_STAT not in statistics:
        raise ValueError(f"Anchor statistic {ANCHOR_STAT!r} not available.")
    if int(args.top_n_paths) < 1:
        raise ValueError("--top-n-paths must be >= 1")

    diff_df = _build_real_minus_control_profiles(agg_df, statistics)
    if diff_df.empty:
        raise ValueError("No real-minus-control profiles could be built from current API datasets.")

    rank_df = _compute_rank_consistency(
        diff_df,
        statistics,
        rbo_p=float(args.rbo_p),
        top_n=int(args.top_n_paths),
    )
    var_df = _compute_variance_consistency(
        diff_df,
        statistics,
        top_n=int(args.top_n_paths),
    )

    rank_plot_df = rank_df[~rank_df["statistic"].astype(str).str.contains("std", case=False, regex=False)].copy()
    var_plot_df = var_df[~var_df["statistic"].astype(str).str.contains("std", case=False, regex=False)].copy()

    agg_df.to_csv(output_dir / "aggregated_statistics_all_datasets.csv", index=False)
    diff_df.to_csv(output_dir / "real_minus_control_profiles.csv", index=False)
    rank_df.to_csv(output_dir / "rank_consistency_vs_mean_dwpc.csv", index=False)
    var_df.to_csv(output_dir / "variance_consistency_vs_mean_dwpc.csv", index=False)

    _plot_metric_lines(
        rank_plot_df,
        metric_col="spearman_rho",
        out_pdf=output_dir / "rank_spearman_consistency_vs_mean_dwpc_lines.pdf",
        title=f"Top-{int(args.top_n_paths)} path ranking consistency vs mean DWPC (Spearman rho)",
    )
    _plot_metric_lines(
        rank_plot_df,
        metric_col="top_n_jaccard",
        out_pdf=output_dir / "rank_topn_jaccard_consistency_vs_mean_dwpc_lines.pdf",
        title=f"Top-{int(args.top_n_paths)} path overlap vs mean DWPC (Jaccard)",
    )
    _plot_metric_lines(
        rank_plot_df,
        metric_col="top_n_rbo",
        out_pdf=output_dir / "rank_topn_rbo_consistency_vs_mean_dwpc_lines.pdf",
        title=f"Top-{int(args.top_n_paths)} path overlap vs mean DWPC (RBO)",
    )
    _plot_metric_lines(
        var_plot_df,
        metric_col="spearman_var_corr",
        out_pdf=output_dir / "variance_spearman_consistency_vs_mean_dwpc_lines.pdf",
        title=f"Top-{int(args.top_n_paths)} path diff-variance consistency vs mean DWPC (Spearman)",
    )
    _plot_metric_lines(
        var_plot_df,
        metric_col="pearson_var_corr",
        out_pdf=output_dir / "variance_pearson_consistency_vs_mean_dwpc_lines.pdf",
        title=f"Top-{int(args.top_n_paths)} path diff-variance consistency vs mean DWPC (Pearson)",
    )
    _plot_metric_heatmap(
        rank_plot_df,
        metric_col="spearman_rho",
        out_pdf=output_dir / "rank_spearman_consistency_vs_mean_dwpc_heatmap.pdf",
        title=f"Top-{int(args.top_n_paths)} path ranking consistency vs mean DWPC (Spearman rho)",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_metric_heatmap(
        rank_plot_df,
        metric_col="top_n_jaccard",
        out_pdf=output_dir / "rank_topn_jaccard_consistency_vs_mean_dwpc_heatmap.pdf",
        title=f"Top-{int(args.top_n_paths)} path overlap vs mean DWPC (Jaccard)",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_metric_heatmap(
        rank_plot_df,
        metric_col="top_n_rbo",
        out_pdf=output_dir / "rank_topn_rbo_consistency_vs_mean_dwpc_heatmap.pdf",
        title=f"Top-{int(args.top_n_paths)} path overlap vs mean DWPC (RBO)",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_metric_heatmap(
        var_plot_df,
        metric_col="spearman_var_corr",
        out_pdf=output_dir / "variance_spearman_consistency_vs_mean_dwpc_heatmap.pdf",
        title=f"Top-{int(args.top_n_paths)} path diff-variance consistency vs mean DWPC (Spearman)",
        vmin=-1.0,
        vmax=1.0,
    )
    _plot_metric_heatmap(
        var_plot_df,
        metric_col="pearson_var_corr",
        out_pdf=output_dir / "variance_pearson_consistency_vs_mean_dwpc_heatmap.pdf",
        title=f"Top-{int(args.top_n_paths)} path diff-variance consistency vs mean DWPC (Pearson)",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_stats = [s for s in statistics if "std" not in str(s).lower()]
    _plot_variance_scatter_by_comparison(diff_df, plot_stats, output_dir)

    print(f"Comparisons analyzed: {len(sorted(diff_df['comparison'].astype(str).unique().tolist()))}")
    print(f"Statistics compared to {ANCHOR_STAT}: {max(0, len(statistics) - 1)}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
