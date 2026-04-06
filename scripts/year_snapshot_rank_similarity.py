#!/usr/bin/env python3
"""Compare real metapath rankings between year snapshots."""

from __future__ import annotations

import argparse
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
from src.result_normalization import load_normalized_year_results  # noqa: E402
from src.year_statistics import (  # noqa: E402
    build_aggregated_year_statistics_panel,
    build_year_statistic_summary_long,
    detect_supported_statistics,
)


DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _default_results_dir(score_source: str) -> Path:
    if str(score_source) == "api":
        return REPO_ROOT / "output" / "dwpc_com" / "all_GO_positive_growth" / "results"
    return REPO_ROOT / "output" / "dwpc_direct" / "all_GO_positive_growth" / "results"


def _default_output_dir(score_source: str) -> Path:
    if str(score_source) == "api":
        return REPO_ROOT / "output" / "year_api_snapshot_rank_similarity"
    return REPO_ROOT / "output" / "year_direct_snapshot_rank_similarity"


def _parse_int_list(arg: str) -> list[int]:
    values = sorted({int(tok.strip()) for tok in str(arg).split(",") if tok.strip()})
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_stat_list(arg: str | None, available: list[str]) -> list[str]:
    if arg is None or not str(arg).strip():
        return ["mean_dwpc"]
    requested = [tok.strip() for tok in str(arg).split(",") if tok.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unsupported statistics requested: {missing}")
    return requested


def _display_statistic_name(statistic: str) -> str:
    stat = str(statistic)
    if "pvalue" in stat:
        return stat.replace("pvalue", "log10pvalue")
    return stat


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


def _build_real_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    work = summary_df[summary_df["control"].astype(str) == "real"].copy()
    if work.empty:
        raise ValueError("No real rows found in year statistic summary.")

    ranking_df = (
        work.groupby(["year", "statistic", "metapath"], as_index=False)
        .agg(
            mean_score=("mean_score", "mean"),
            n_go_terms=("go_id", "nunique"),
        )
        .sort_values(["year", "statistic", "mean_score", "metapath"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    ranking_df["rank"] = (
        ranking_df.groupby(["year", "statistic"], sort=False)
        .cumcount()
        .add(1)
    )
    return ranking_df


def _compare_rankings(
    ranking_df: pd.DataFrame,
    *,
    year_a: int,
    year_b: int,
    statistics: list[str],
    top_k_values: list[int],
    rbo_p: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    topk_rows: list[dict] = []

    for statistic in statistics:
        a_df = ranking_df[
            (ranking_df["year"].astype(int) == int(year_a))
            & (ranking_df["statistic"].astype(str) == str(statistic))
        ].copy()
        b_df = ranking_df[
            (ranking_df["year"].astype(int) == int(year_b))
            & (ranking_df["statistic"].astype(str) == str(statistic))
        ].copy()
        if a_df.empty or b_df.empty:
            continue

        a_ranked = a_df.sort_values(["rank", "metapath"])["metapath"].astype(str).tolist()
        b_ranked = b_df.sort_values(["rank", "metapath"])["metapath"].astype(str).tolist()
        a_rank_pos = dict(zip(a_df["metapath"].astype(str), a_df["rank"].astype(int)))
        b_rank_pos = dict(zip(b_df["metapath"].astype(str), b_df["rank"].astype(int)))
        common = sorted(set(a_rank_pos) & set(b_rank_pos))
        if len(common) < 2:
            continue

        rho = float(
            pd.Series([a_rank_pos[m] for m in common], dtype=float).corr(
                pd.Series([b_rank_pos[m] for m in common], dtype=float),
                method="pearson",
            )
        )
        summary_rows.append(
            {
                "year_a": int(year_a),
                "year_b": int(year_b),
                "statistic": str(statistic),
                "n_metapaths_year_a": int(len(a_ranked)),
                "n_metapaths_year_b": int(len(b_ranked)),
                "n_common_metapaths": int(len(common)),
                "spearman_rho": rho,
                "rbo": _rbo_score(a_ranked, b_ranked, p=float(rbo_p)),
            }
        )

        for top_k in top_k_values:
            top_a = a_ranked[: int(top_k)]
            top_b = b_ranked[: int(top_k)]
            top_a_set = set(top_a)
            top_b_set = set(top_b)
            overlap = top_a_set & top_b_set
            union = top_a_set | top_b_set
            shift_vals = [abs(int(a_rank_pos[m]) - int(b_rank_pos[m])) for m in top_a if m in b_rank_pos]
            topk_rows.append(
                {
                    "year_a": int(year_a),
                    "year_b": int(year_b),
                    "statistic": str(statistic),
                    "top_k": int(top_k),
                    "overlap_count": int(len(overlap)),
                    "jaccard": float(len(overlap) / len(union)) if union else np.nan,
                    "retention_from_year_a": float(len(overlap) / len(top_a_set)) if top_a_set else np.nan,
                    "retention_from_year_b": float(len(overlap) / len(top_b_set)) if top_b_set else np.nan,
                    "median_abs_rank_shift_year_a_topk": float(np.median(shift_vals)) if shift_vals else np.nan,
                    "mean_abs_rank_shift_year_a_topk": float(np.mean(shift_vals)) if shift_vals else np.nan,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(topk_rows)


def _plot_metric_summary(
    summary_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    *,
    top_k: int,
    output_path: Path,
) -> None:
    if summary_df.empty or topk_df.empty:
        return

    plot_df = summary_df.merge(
        topk_df[topk_df["top_k"].astype(int) == int(top_k)][
            ["statistic", "jaccard", "median_abs_rank_shift_year_a_topk"]
        ],
        on="statistic",
        how="left",
    )
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("statistic").reset_index(drop=True)
    labels = [_display_statistic_name(stat) for stat in plot_df["statistic"].astype(str).tolist()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(max(10, 1.2 * len(labels) + 4), 8.5))
    metric_specs = [
        ("spearman_rho", "Spearman rho", (-1.0, 1.0)),
        ("rbo", "RBO", (0.0, 1.0)),
        ("jaccard", f"Top-{int(top_k)} Jaccard", (0.0, 1.0)),
        ("median_abs_rank_shift_year_a_topk", f"Median abs rank shift\n({int(top_k)} from year A)", None),
    ]

    for ax, (col, title, ylim) in zip(axes.flat, metric_specs):
        vals = plot_df[col].astype(float).to_numpy()
        ax.bar(x, vals, color="#4c78a8", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.suptitle("Year snapshot metapath-ranking similarity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_scatter(
    ranking_df: pd.DataFrame,
    *,
    statistic: str,
    year_a: int,
    year_b: int,
    output_path: Path,
) -> None:
    a_df = ranking_df[
        (ranking_df["year"].astype(int) == int(year_a))
        & (ranking_df["statistic"].astype(str) == str(statistic))
    ][["metapath", "rank"]].rename(columns={"rank": "rank_a"})
    b_df = ranking_df[
        (ranking_df["year"].astype(int) == int(year_b))
        & (ranking_df["statistic"].astype(str) == str(statistic))
    ][["metapath", "rank"]].rename(columns={"rank": "rank_b"})
    merged = a_df.merge(b_df, on="metapath", how="inner")
    if len(merged) < 2:
        return

    max_rank = int(merged[["rank_a", "rank_b"]].max().max())
    rho = float(merged["rank_a"].corr(merged["rank_b"], method="pearson"))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(
        merged["rank_a"].astype(float),
        merged["rank_b"].astype(float),
        s=22,
        alpha=0.65,
        color="#1f77b4",
        edgecolors="none",
    )
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#999999", linewidth=1.4)
    ax.set_xlabel(f"{int(year_a)} rank")
    ax.set_ylabel(f"{int(year_b)} rank")
    ax.set_title(f"{_display_statistic_name(statistic)}\nSpearman rho = {rho:.3f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-source", default="direct", choices=["direct", "api"])
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--year-a", type=int, default=2016)
    parser.add_argument("--year-b", type=int, default=2024)
    parser.add_argument("--statistics", default="mean_dwpc")
    parser.add_argument("--top-k-values", default="5,10,20")
    parser.add_argument("--rbo-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir(args.score_source)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.score_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_df = load_normalized_year_results(
        results_dir,
        score_source=str(args.score_source),
        data_dir=Path(args.data_dir),
    )
    agg_df = build_aggregated_year_statistics_panel(normalized_df)
    if agg_df.empty:
        raise ValueError("Aggregated year statistics are empty.")

    available_statistics = detect_supported_statistics(agg_df)
    statistics = _parse_stat_list(args.statistics, available_statistics)
    summary_long_df = build_year_statistic_summary_long(agg_df, statistics=statistics)
    ranking_df = _build_real_rankings(summary_long_df)
    ranking_df.to_csv(output_dir / "snapshot_metapath_rankings.csv", index=False)

    top_k_values = _parse_int_list(args.top_k_values)
    summary_df, topk_df = _compare_rankings(
        ranking_df,
        year_a=int(args.year_a),
        year_b=int(args.year_b),
        statistics=statistics,
        top_k_values=top_k_values,
        rbo_p=float(args.rbo_p),
    )
    if summary_df.empty:
        raise ValueError("No snapshot ranking comparisons were generated.")

    summary_df.to_csv(output_dir / "snapshot_rank_similarity_summary.csv", index=False)
    topk_df.to_csv(output_dir / "snapshot_rank_similarity_topk.csv", index=False)

    plot_top_k = 10 if 10 in top_k_values else top_k_values[0]
    _plot_metric_summary(
        summary_df,
        topk_df,
        top_k=int(plot_top_k),
        output_path=output_dir / "snapshot_rank_similarity_overview.pdf",
    )
    _plot_rank_scatter(
        ranking_df,
        statistic=str(statistics[0]),
        year_a=int(args.year_a),
        year_b=int(args.year_b),
        output_path=output_dir / f"rank_scatter_{statistics[0]}_{int(args.year_a)}_vs_{int(args.year_b)}.pdf",
    )

    print(f"Saved rankings: {output_dir / 'snapshot_metapath_rankings.csv'}")
    print(f"Saved summary: {output_dir / 'snapshot_rank_similarity_summary.csv'}")
    print(f"Saved top-k summary: {output_dir / 'snapshot_rank_similarity_topk.csv'}")
    print(f"Saved plots under: {output_dir}")


if __name__ == "__main__":
    main()
