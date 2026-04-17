#!/usr/bin/env python3
"""Generate global summary metrics for LV or Year intermediate sharing analysis.

Aggregates per-gene-set metrics from intermediate sharing results:
- Metapath counts and effect sizes
- Gene coverage statistics
- Intermediate sharing and convergence metrics

Usage:
    python scripts/generate_global_summary.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing --output-dir output/lv_global_summary

    python scripts/generate_global_summary.py --analysis-type year \
        --input-dir output/year_intermediate_sharing --output-dir output/year_global_summary

    # With chosen B from elbow selection
    python scripts/generate_global_summary.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --chosen-b-json output/lv_full_analysis/chosen_b.json \
        --output-dir output/lv_global_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save_figure(fig, out_stem: Path) -> None:
    """Save figure as both PDF and PNG using a common stem."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_stem.with_suffix(ext), bbox_inches="tight", dpi=200)


def _load_dropped_lvs(input_dir: Path, b_value: int | None) -> pd.DataFrame:
    """Load dropped_lvs.csv from the intermediate-sharing output if present."""
    data_dir = input_dir / f"b{b_value}" if b_value is not None else input_dir
    path = data_dir / "dropped_lvs.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_lv_selection_diagnostics(
    by_metapath_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    effect_size_threshold: float,
    analysis_type: str,
    out_dir: Path,
) -> None:
    """Emit two PDFs + PNGs showing why each LV was kept or dropped.

    - `lv_selection_max_effect_size`: horizontal bar of max effect size per LV,
      colored by selected/dropped, with the filter threshold marked.
    - `lv_selection_effect_distribution`: strip + count of per-metapath effect
      sizes for each selected LV; dropped LVs shown at their max_d.
    """
    if by_metapath_df.empty and dropped_df.empty:
        return
    id_col = "lv_id" if analysis_type == "lv" else "go_id"

    selected_summary = pd.DataFrame()
    if not by_metapath_df.empty and id_col in by_metapath_df.columns:
        selected_summary = (
            by_metapath_df.groupby(id_col, as_index=False)
            .agg(
                max_effect_size_z=("effect_size_z", "max"),
                n_metapaths_selected=("effect_size_z", "size"),
            )
            .assign(status="selected")
        )

    dropped_summary = pd.DataFrame()
    if not dropped_df.empty and id_col in dropped_df.columns:
        dropped_summary = dropped_df[[id_col, "max_effect_size_z"]].copy()
        dropped_summary["n_metapaths_selected"] = 0
        dropped_summary["status"] = "dropped"

    all_lvs = pd.concat([selected_summary, dropped_summary], ignore_index=True)
    if all_lvs.empty:
        return
    all_lvs = all_lvs.sort_values("max_effect_size_z", ascending=False).reset_index(drop=True)

    # Cap the number of entities drawn so year-scale datasets (hundreds of
    # GO terms) don't produce 170-inch-tall figures. Keep the top entities by
    # max effect size and surface the total count in the title.
    MAX_BARS = 50
    full_count = len(all_lvs)
    bars_df = all_lvs.head(MAX_BARS).copy()

    # Plain horizontal bar chart (one color for all entities). Threshold line
    # is the only status indicator; bars can be negative when an entity's best
    # metapath has a negative effect size.
    max_d = bars_df["max_effect_size_z"].astype(float)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.28 * len(bars_df))))
    y = np.arange(len(bars_df))
    ax.barh(y, max_d, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(bars_df[id_col].astype(str).tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(
        float(effect_size_threshold),
        color="red", linestyle="--", linewidth=1.3,
        label=f"z threshold = {effect_size_threshold}",
    )
    lo, hi = float(min(0.0, max_d.min())), float(max(effect_size_threshold, max_d.max()))
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - 0.05 * span, hi + 0.05 * span)
    ax.set_xlabel("Max z across metapaths")
    ax.set_ylabel(id_col)
    title = f"{id_col} selection: max effect size per {id_col}"
    if full_count > MAX_BARS:
        title += f"  (top {len(bars_df)} of {full_count})"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_dir / "lv_selection_max_effect_size")
    plt.close(fig)

    # Plot 2: strip of effect sizes for selected LVs (one row per LV)
    if by_metapath_df.empty or id_col not in by_metapath_df.columns:
        return
    selected_ids_all = all_lvs[all_lvs["status"] == "selected"][id_col].astype(str).tolist()
    if not selected_ids_all:
        return

    # Same cap as plot 1 so both diagnostics agree on the displayed subset.
    selected_ids = selected_ids_all[:MAX_BARS]
    full_selected = len(selected_ids_all)

    strip_df = by_metapath_df[by_metapath_df[id_col].astype(str).isin(selected_ids)].copy()
    if strip_df.empty:
        return

    id_order = selected_ids  # already sorted by max_effect_size_z above
    row_index = {lv: i for i, lv in enumerate(id_order)}
    strip_df["row"] = strip_df[id_col].astype(str).map(row_index)

    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.35 * len(id_order) + 1.5)))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.22, 0.22, size=len(strip_df))
    ax.scatter(
        strip_df["effect_size_z"].astype(float),
        strip_df["row"].astype(float) + jitter,
        s=22, alpha=0.55, color="#1f77b4", edgecolors="none",
    )

    ax.axvline(
        float(effect_size_threshold),
        color="red", linestyle="--", linewidth=1.3,
        label=f"z threshold = {effect_size_threshold}",
    )
    ax.set_yticks(np.arange(len(id_order)))
    ax.set_yticklabels(id_order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("z (per metapath)")
    ax.set_ylabel(id_col)
    title = f"Per-metapath effect size distribution for selected {id_col}"
    if full_selected > len(id_order):
        title += f"  (top {len(id_order)} of {full_selected})"
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, out_dir / "lv_selection_effect_distribution")
    plt.close(fig)


def _load_all_entities_z_from_runs(
    runs_path: Path, b: int | None, analysis_type: str
) -> pd.DataFrame:
    """Load upstream all_runs_long.csv and aggregate to one z per (entity, metapath).

    Uses permuted null only. Averages effect_size_z (true z) across seeds at the
    chosen B. Returns a frame with columns [entity_id, metapath, effect_size_z,
    target_name] matching what plot_top_metapaths_per_entity expects.
    """
    runs = pd.read_csv(runs_path)
    if b is not None and "b" in runs.columns:
        runs = runs[runs["b"] == int(b)]
    if "control" in runs.columns:
        runs = runs[runs["control"] == "permuted"]
    if runs.empty:
        return pd.DataFrame()

    if "effect_size_z" in runs.columns:
        z_col = "effect_size_z"
    elif "effect_size_d" in runs.columns:
        z_col = "effect_size_d"
    elif "d" in runs.columns:
        z_col = "d"
    else:
        raise ValueError(
            f"{runs_path} has neither 'effect_size_z', 'effect_size_d', nor 'd' column"
        )

    id_col = "lv_id" if analysis_type == "lv" else "go_id"
    group_cols = [id_col, "metapath"]
    if "target_name" in runs.columns:
        group_cols.insert(1, "target_name")

    agg = runs.groupby(group_cols, as_index=False).agg(
        effect_size_z=(z_col, "mean"),
    )
    return agg


def plot_top_metapaths_per_entity(
    by_metapath_df: pd.DataFrame,
    effect_size_threshold: float,
    analysis_type: str,
    out_dir: Path,
    top_n: int = 5,
    max_panels: int = 50,
    ncols: int = 3,
    filename_suffix: str = "",
) -> None:
    """One panel per selected LV/GO term, horizontal bars of its top-N metapaths by z."""
    if by_metapath_df.empty:
        return
    id_col = "lv_id" if analysis_type == "lv" else "go_id"
    if id_col not in by_metapath_df.columns or "effect_size_z" not in by_metapath_df.columns:
        return

    ordering = (
        by_metapath_df.groupby(id_col)["effect_size_z"].max()
        .sort_values(ascending=False)
    )
    entity_ids = ordering.head(max_panels).index.astype(str).tolist()
    if not entity_ids:
        return
    full_count = len(ordering)

    name_lookup: dict[str, str] = {}
    if "target_name" in by_metapath_df.columns:
        for eid, group in by_metapath_df.groupby(id_col):
            names = group["target_name"].dropna().astype(str).unique().tolist()
            if names:
                name_lookup[str(eid)] = names[0]

    n_panels = len(entity_ids)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, 2.4 * nrows),
        squeeze=False,
    )

    vmax = float(by_metapath_df["effect_size_z"].max())
    vmin = min(0.0, float(by_metapath_df["effect_size_z"].min()))
    span = max(vmax - vmin, 1e-6)
    xlim = (vmin - 0.03 * span, vmax + 0.05 * span)

    for idx, eid in enumerate(entity_ids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sub = (
            by_metapath_df[by_metapath_df[id_col].astype(str) == eid]
            .nlargest(top_n, "effect_size_z")
        )
        y = np.arange(len(sub))
        ax.barh(y, sub["effect_size_z"].astype(float),
                color="#2ca02c", edgecolor="black", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["metapath"].astype(str).tolist(), fontsize=7)
        ax.invert_yaxis()
        ax.axvline(float(effect_size_threshold),
                   color="red", linestyle="--", linewidth=1.0)
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
        panel_title = eid
        target = name_lookup.get(eid)
        if target:
            panel_title = f"{eid}: {target}"
        ax.set_title(panel_title, fontsize=9)

    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    suptitle = f"Top {top_n} metapaths by z per {id_col}"
    if full_count > n_panels:
        suptitle += f"  (top {n_panels} of {full_count})"
    fig.suptitle(suptitle, fontsize=11)
    fig.supxlabel("z", fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save_figure(fig, out_dir / f"top_metapaths_per_{id_col}{filename_suffix}")
    plt.close(fig)


def load_intermediate_sharing_data(
    input_dir: Path,
    b_value: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load intermediate sharing results.

    Args:
        input_dir: Directory containing intermediate sharing results
        b_value: If specified, load from b{value}/ subdirectory

    Returns:
        Tuple of (by_metapath_df, top_intermediates_df)
    """
    if b_value is not None:
        data_dir = input_dir / f"b{b_value}"
    else:
        data_dir = input_dir

    by_metapath_path = data_dir / "intermediate_sharing_by_metapath.csv"
    top_int_path = data_dir / "top_intermediates_by_metapath.csv"

    if not by_metapath_path.exists():
        raise FileNotFoundError(f"Not found: {by_metapath_path}")

    by_metapath_df = pd.read_csv(by_metapath_path)

    if top_int_path.exists():
        top_int_df = pd.read_csv(top_int_path)
    else:
        top_int_df = pd.DataFrame()

    return by_metapath_df, top_int_df


_MEDIAN_METRICS = [
    ("top1_intermediate_coverage", "median_top1_coverage"),
    ("top5_intermediate_coverage", "median_top5_coverage"),
    ("pct_intermediates_shared_quarter", "median_pct_shared_quarter"),
    ("pct_intermediates_shared_majority", "median_pct_shared_majority"),
    ("pct_intermediates_shared_all", "median_pct_shared_all"),
    ("n_unique_intermediates", "median_n_intermediates"),
    ("n_intermediates_cover_80pct", "median_n_for_80pct_coverage"),
    ("median_jaccard_to_group", "median_jaccard"),
]

_MAX_METRICS = [
    ("top1_intermediate_coverage", "max_top1_coverage"),
]

_SUM_METRICS = [
    ("n_unique_intermediates", "total_unique_intermediates"),
]


def compute_global_summary(
    by_metapath_df: pd.DataFrame,
    top_int_df: pd.DataFrame,
    analysis_type: str,
    b_value: int | None = None,
) -> pd.DataFrame:
    if analysis_type == "lv":
        group_cols = ["lv_id", "target_id", "target_name", "node_type"]
    else:
        group_cols = ["go_id", "target_id", "target_name", "node_type"]

    group_cols = [c for c in group_cols if c in by_metapath_df.columns]

    summary_rows = []

    for group_key, group in by_metapath_df.groupby(group_cols, sort=True):
        if isinstance(group_key, tuple):
            row = dict(zip(group_cols, group_key))
        else:
            row = {group_cols[0]: group_key}

        if b_value is not None:
            row["b"] = b_value
        elif "b" in group.columns:
            row["b"] = group["b"].iloc[0]

        row["n_metapaths_selected"] = len(group)

        top_mp_row = group.loc[group["effect_size_z"].idxmax()]
        row["top_metapath"] = top_mp_row["metapath"]
        row["top_metapath_d"] = top_mp_row["effect_size_z"]

        row["median_effect_size_z"] = group["effect_size_z"].median()
        row["max_effect_size_z"] = group["effect_size_z"].max()
        row["min_effect_size_z"] = group["effect_size_z"].min()

        if "n_genes_total" in group.columns:
            row["n_genes_total"] = int(group["n_genes_total"].iloc[0] or 0)
        elif {"n_genes_2016", "n_genes_2024_added"}.issubset(group.columns):
            n2016 = int(group["n_genes_2016"].iloc[0] or 0)
            n2024 = int(group["n_genes_2024_added"].iloc[0] or 0)
            row["n_genes_2016"] = n2016
            row["n_genes_2024_added"] = n2024
            row["n_genes_total"] = n2016 + n2024
        else:
            row["n_genes_total"] = 0

        if "n_genes_with_paths" in group.columns:
            row["max_genes_with_paths_per_mp"] = group["n_genes_with_paths"].max()
            row["median_genes_with_paths_per_mp"] = group["n_genes_with_paths"].median()
        else:
            row["max_genes_with_paths_per_mp"] = 0
            row["median_genes_with_paths_per_mp"] = 0

        if row["n_genes_total"] > 0:
            row["approx_pct_genes_covered"] = (
                row["max_genes_with_paths_per_mp"] / row["n_genes_total"] * 100
            )
        else:
            row["approx_pct_genes_covered"] = 0.0

        for src_col, dest_col in _MEDIAN_METRICS:
            if src_col in group.columns:
                row[dest_col] = group[src_col].median()

        for src_col, dest_col in _MAX_METRICS:
            if src_col in group.columns:
                row[dest_col] = group[src_col].max()

        for src_col, dest_col in _SUM_METRICS:
            if src_col in group.columns:
                row[dest_col] = group[src_col].sum()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if "n_metapaths_selected" in summary_df.columns:
        summary_df = summary_df.sort_values("n_metapaths_selected", ascending=False)

    return summary_df.reset_index(drop=True)


def compute_cross_b_summary(
    input_dir: Path,
    b_values: list[int],
    analysis_type: str,
) -> pd.DataFrame:
    """Compute summary across multiple B values.

    Args:
        input_dir: Directory containing b{value}/ subdirectories
        b_values: List of B values to aggregate
        analysis_type: "lv" or "year"

    Returns:
        DataFrame with metrics across B values
    """
    all_summaries = []

    for b in b_values:
        try:
            by_metapath_df, top_int_df = load_intermediate_sharing_data(input_dir, b)
            summary = compute_global_summary(by_metapath_df, top_int_df, analysis_type, b)
            all_summaries.append(summary)
        except FileNotFoundError:
            print(f"Warning: No data for B={b}, skipping")
            continue

    if not all_summaries:
        return pd.DataFrame()

    return pd.concat(all_summaries, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-type",
        choices=["lv", "year"],
        required=True,
        help="Type of analysis (lv or year)",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing intermediate sharing results",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for global summary",
    )
    parser.add_argument(
        "--chosen-b-json",
        help="Path to chosen_b.json from B selection (uses this B for summary)",
    )
    parser.add_argument(
        "--b-values",
        help="Comma-separated B values to summarize (e.g., '2,5,10,20,30')",
    )
    parser.add_argument(
        "--b",
        type=int,
        help="Single B value to use",
    )
    parser.add_argument(
        "--effect-size-threshold",
        type=float,
        default=0.5,
        help=(
            "Permutation z-statistic threshold used by the upstream filter, "
            "drawn on the LV selection diagnostic plots (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--all-runs-path",
        default=None,
        help=(
            "Optional path to upstream all_runs_long.csv "
            "(e.g. <lv_output_dir>/lv_rank_stability_experiment/all_runs_long.csv). "
            "When provided, produce an additional barplot of top-10 metapaths per "
            "entity computed across ALL entities (including ones that didn't clear "
            "the effect-size threshold)."
        ),
    )
    parser.add_argument(
        "--all-runs-top-n",
        type=int,
        default=10,
        help="Top-N metapaths per entity for the --all-runs-path plot (default: 10).",
    )
    return parser.parse_args()


def _load_and_summarize(
    input_dir: Path, b_value: int | None, analysis_type: str, output_dir: Path,
) -> pd.DataFrame:
    by_metapath_df, top_int_df = load_intermediate_sharing_data(input_dir, b_value)
    summary = compute_global_summary(by_metapath_df, top_int_df, analysis_type, b_value)
    summary.to_csv(output_dir / "global_summary.csv", index=False)
    print(f"Saved: {output_dir / 'global_summary.csv'}")
    return summary


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chosen_b = None
    if args.chosen_b_json:
        with open(args.chosen_b_json) as f:
            chosen_b = json.load(f)["chosen_b"]
        print(f"Using chosen B = {chosen_b}")

    summary = pd.DataFrame()

    if args.b_values:
        b_values = [int(b.strip()) for b in args.b_values.split(",")]
        cross_b_summary = compute_cross_b_summary(input_dir, b_values, args.analysis_type)
        if not cross_b_summary.empty:
            cross_b_summary.to_csv(output_dir / "global_summary_all_b.csv", index=False)
        if chosen_b is not None and chosen_b in b_values:
            summary = _load_and_summarize(input_dir, chosen_b, args.analysis_type, output_dir)
    else:
        b_value = args.b if args.b is not None else chosen_b
        summary = _load_and_summarize(input_dir, b_value, args.analysis_type, output_dir)

    if not summary.empty:
        print(f"Gene sets analyzed: {len(summary)}")
        print(f"Total metapaths selected: {summary['n_metapaths_selected'].sum()}")

    plot_b = args.b if args.b is not None else chosen_b
    if plot_b is not None:
        try:
            by_metapath_df_plot, _ = load_intermediate_sharing_data(input_dir, plot_b)
        except FileNotFoundError:
            by_metapath_df_plot = pd.DataFrame()
        dropped_df_plot = _load_dropped_lvs(input_dir, plot_b)
        plots_dir = output_dir / "plots"
        plot_lv_selection_diagnostics(
            by_metapath_df=by_metapath_df_plot,
            dropped_df=dropped_df_plot,
            effect_size_threshold=args.effect_size_threshold,
            analysis_type=args.analysis_type,
            out_dir=plots_dir,
        )
        plot_top_metapaths_per_entity(
            by_metapath_df=by_metapath_df_plot,
            effect_size_threshold=args.effect_size_threshold,
            analysis_type=args.analysis_type,
            out_dir=plots_dir,
        )
        if args.all_runs_path:
            runs_path = Path(args.all_runs_path)
            if runs_path.exists():
                all_runs_df = _load_all_entities_z_from_runs(runs_path, plot_b, args.analysis_type)
                if not all_runs_df.empty:
                    plot_top_metapaths_per_entity(
                        by_metapath_df=all_runs_df,
                        effect_size_threshold=args.effect_size_threshold,
                        analysis_type=args.analysis_type,
                        out_dir=plots_dir,
                        top_n=int(args.all_runs_top_n),
                        filename_suffix="_all_entities",
                    )
        print(f"Saved diagnostic plots to {plots_dir}/")


if __name__ == "__main__":
    main()
