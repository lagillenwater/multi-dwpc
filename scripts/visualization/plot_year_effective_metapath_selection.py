#!/usr/bin/env python3
"""Plot metapath selection frequency across GO terms for each year."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


YEAR_COLORS = {
    "2016": "#1f77b4",
    "2024": "#ff7f0e",
}


def _save_dual(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes"})
    )


def _build_summary(df: pd.DataFrame, selection_col: str) -> tuple[pd.DataFrame, list[int], dict[int, int]]:
    work = df[["year", "go_id", "metapath", selection_col]].copy()
    work["year"] = work["year"].astype(int)
    work["go_id"] = work["go_id"].astype(str)
    work["metapath"] = work["metapath"].astype(str)
    work[selection_col] = _normalize_bool(work[selection_col])

    years = sorted(work["year"].dropna().unique().tolist())
    go_totals = (
        work.groupby("year", as_index=False)["go_id"]
        .nunique()
        .set_index("year")["go_id"]
        .astype(int)
        .to_dict()
    )

    counts = (
        work.groupby(["year", "metapath"], as_index=False)[selection_col]
        .sum()
        .rename(columns={selection_col: "n_go_terms_selected"})
    )
    counts["selected_fraction"] = counts.apply(
        lambda row: float(row["n_go_terms_selected"]) / float(go_totals[int(row["year"])])
        if go_totals[int(row["year"])] > 0
        else np.nan,
        axis=1,
    )

    count_wide = (
        counts.pivot(index="metapath", columns="year", values="n_go_terms_selected")
        .fillna(0)
        .astype(int)
    )
    frac_wide = counts.pivot(index="metapath", columns="year", values="selected_fraction").fillna(0.0)

    selected_mask = count_wide.max(axis=1) > 0
    count_wide = count_wide.loc[selected_mask].copy()
    frac_wide = frac_wide.loc[count_wide.index].copy()

    summary = pd.DataFrame(index=count_wide.index)
    for year in years:
        summary[f"n_go_terms_selected_{year}"] = count_wide.get(year, 0).astype(int)
        summary[f"selected_fraction_{year}"] = frac_wide.get(year, 0.0).astype(float)
    summary["max_n_go_terms_selected"] = count_wide.max(axis=1).astype(int)
    summary["total_n_go_terms_selected"] = count_wide.sum(axis=1).astype(int)
    summary = (
        summary.sort_values(
            ["max_n_go_terms_selected", "total_n_go_terms_selected"],
            ascending=[False, False],
        )
        .reset_index()
        .rename(columns={"index": "metapath"})
    )
    return summary, years, go_totals


def _plot_summary(
    summary: pd.DataFrame,
    *,
    years: list[int],
    go_totals: dict[int, int],
    output_path: Path,
    title: str,
) -> None:
    n_metapaths = len(summary)
    fig_height = max(8.0, 0.32 * n_metapaths + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))

    y = np.arange(n_metapaths, dtype=float)
    bar_height = 0.38 if len(years) > 1 else 0.6
    offsets = {
        int(year): (idx - (len(years) - 1) / 2.0) * bar_height
        for idx, year in enumerate(years)
    }

    for year in years:
        col = f"n_go_terms_selected_{year}"
        color = YEAR_COLORS.get(str(year), "#333333")
        ax.barh(
            y + offsets[int(year)],
            summary[col].astype(float).to_numpy(),
            height=bar_height * 0.9,
            color=color,
            alpha=0.85,
            label=f"{year} (n={go_totals[int(year)]})",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(summary["metapath"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GO terms selecting metapath by effective-number criterion")
    ax.set_ylabel("Metapath")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Year", loc="best")
    fig.tight_layout()
    _save_dual(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-path",
        default="output/year_direct_go_term_support_b5.csv",
        help="GO-term support CSV containing selected_by_effective_n_all.",
    )
    parser.add_argument(
        "--selection-col",
        default="selected_by_effective_n_all",
        help="Boolean selection column to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/metapath_analysis/effective_metapath_selection_b5",
        help="Directory for the summary CSV and plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support_path = Path(args.support_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(support_path)
    required = {"year", "go_id", "metapath", args.selection_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{support_path} is missing required columns: {missing}")

    summary, years, go_totals = _build_summary(df, args.selection_col)
    if summary.empty:
        raise ValueError("No metapaths were selected for any GO term.")

    b_vals = sorted(df["b"].dropna().astype(int).unique().tolist()) if "b" in df.columns else []
    b_label = f" (B={b_vals[0]})" if len(b_vals) == 1 else ""

    summary_path = output_dir / "effective_metapath_selection_frequency.csv"
    plot_path = output_dir / "effective_metapath_selection_frequency.pdf"
    summary.to_csv(summary_path, index=False)
    _plot_summary(
        summary,
        years=years,
        go_totals=go_totals,
        output_path=plot_path,
        title=f"Metapath selection frequency across GO terms{b_label}",
    )

    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
