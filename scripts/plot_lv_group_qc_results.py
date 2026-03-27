#!/usr/bin/env python3
"""Create a single dashboard figure for LV group QC results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

matplotlib.use("Agg")


GROUP_COLORS = {
    "LV246 | adipose_tissue": "#1f77b4",
    "LV57 | hypothyroidism": "#ff7f0e",
    "LV603 | neutrophil_bp": "#2ca02c",
}

NULL_STYLES = {
    "permuted": ("-", "o"),
    "random": ("--", "s"),
}


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _group_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_label"] = out["lv_id"].astype(str) + " | " + out["target_set_id"].astype(str)
    return out


def _ordered_group_labels(group_qc_df: pd.DataFrame) -> list[str]:
    df = _group_label(group_qc_df)
    return df["group_label"].astype(str).tolist()


def _group_color(label: str) -> str:
    return GROUP_COLORS.get(label, "#444444")


def _metric_specs(df: pd.DataFrame) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = [("mean_spearman_rho", "Spearman rho")]
    for k in (5, 10, 15):
        col = f"mean_topk_jaccard_{k}"
        if col in df.columns:
            specs.append((col, f"Top-{k} Jaccard"))
    return specs


def _descriptor_heatmap(
    ax: plt.Axes,
    descriptor_dev_df: pd.DataFrame,
    group_order: list[str],
) -> None:
    df = _group_label(descriptor_dev_df)
    pivot = df.pivot(index="group_label", columns="variable", values="deviation")
    pivot = pivot.reindex(group_order)
    ordered_cols = [
        "n_genes",
        "target_set_size",
        "n_candidate_metapaths",
        "gene_promiscuity_median",
        "gene_promiscuity_iqr",
        "gene_promiscuity_p90",
        "target_promiscuity_median",
        "target_promiscuity_iqr",
        "target_promiscuity_p90",
        "score_sparsity",
    ]
    ordered_cols = [col for col in ordered_cols if col in pivot.columns]
    pivot = pivot[ordered_cols]
    data = pivot.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    max_abs = max(1.0, float(max_abs))
    cmap = LinearSegmentedColormap.from_list("qc_div", ["#b2182b", "#f7f7f7", "#2166ac"])
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "NA" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="#222222")
    ax.set_xticks(np.arange(len(ordered_cols)))
    ax.set_xticklabels(
        [
            "n_genes",
            "targets",
            "metapaths",
            "gene_med",
            "gene_iqr",
            "gene_p90",
            "target_med",
            "target_iqr",
            "target_p90",
            "sparsity",
        ][: len(ordered_cols)],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(len(group_order)))
    ax.set_yticklabels(group_order)
    ax.set_title("Descriptor deviation\n(value - family median) / (p95 - p05)")
    ax.set_xticks(np.arange(-0.5, len(ordered_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(group_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def _null_match_panel(
    axes: list[plt.Axes],
    group_qc_df: pd.DataFrame,
    group_order: list[str],
) -> None:
    df = _group_label(group_qc_df).set_index("group_label").loc[group_order].reset_index()
    y = np.arange(len(df))
    panels = [
        ("random_match_mae", "Random match MAE", 1.0),
        ("random_within_tolerance_rate", "Random within-tolerance", 0.90),
        ("perm_edge_overlap_with_real_mean", "Permuted overlap with real", 0.70),
        ("perm_pairwise_overlap_mean", "Permuted replicate overlap", np.nan),
    ]
    for ax, (col, title, threshold) in zip(axes, panels):
        vals = df[col].astype(float).to_numpy()
        colors = [_group_color(label) for label in df["group_label"].tolist()]
        xmin = float(np.nanmin(vals))
        ax.hlines(y, xmin, vals, color=colors, alpha=0.30, linewidth=2.5)
        ax.scatter(vals, y, s=85, c=colors, edgecolors="white", linewidths=1.0, zorder=3)
        for idx, val in enumerate(vals):
            ax.text(val, y[idx] + 0.10, f"{val:.3f}", fontsize=7.5, ha="center", color=colors[idx])
        if not pd.isna(threshold):
            ax.axvline(threshold, color="#8c8c8c", linestyle="--", linewidth=1.0)
        data_min = float(np.nanmin(vals))
        data_max = float(np.nanmax(vals))
        span = max(0.01, data_max - data_min)
        left = max(0.0, min(data_min, threshold if not pd.isna(threshold) else data_min) - 0.25 * span)
        right = max(data_max, threshold if not pd.isna(threshold) else data_max) + 0.25 * span
        if "rate" in col:
            right = min(1.02, max(right, 1.0))
        ax.set_xlim(left, right)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.20, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(y)
        ax.set_yticklabels(df["group_label"].tolist())
        ax.invert_yaxis()


def _plot_within_null_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    group_order: list[str],
) -> None:
    for group_label in group_order:
        gdf = df[df["group_label"].astype(str) == group_label].copy()
        color = _group_color(group_label)
        for control in sorted(gdf["control"].astype(str).unique().tolist()):
            cdf = gdf[gdf["control"].astype(str) == control].copy().sort_values("b")
            if cdf.empty or metric_col not in cdf.columns:
                continue
            linestyle, marker = NULL_STYLES.get(control, ("-", "o"))
            ax.plot(
                cdf["b"].astype(float),
                cdf[metric_col].astype(float),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5.5,
                color=color,
                alpha=0.95,
                label=f"{group_label} [{control}]",
            )
    ax.set_title(metric_label)
    ax.set_xlabel("B")
    ax.set_ylabel("Within-null seed stability")
    ax.set_ylim(0.70, 1.02)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_between_null_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    group_order: list[str],
) -> None:
    for group_label in group_order:
        gdf = df[df["group_label"].astype(str) == group_label].copy().sort_values("b")
        if gdf.empty or metric_col not in gdf.columns:
            continue
        color = _group_color(group_label)
        ax.plot(
            gdf["b"].astype(float),
            gdf[metric_col].astype(float),
            linestyle="-",
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
            alpha=0.95,
            label=group_label,
        )
    ax.set_title(metric_label)
    ax.set_xlabel("B")
    ax.set_ylabel("Random vs permuted agreement")
    ax.set_ylim(0.70, 1.02)
    ax.grid(alpha=0.22, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _decision_text(
    ax: plt.Axes,
    group_qc_df: pd.DataFrame,
    group_order: list[str],
) -> None:
    df = _group_label(group_qc_df).set_index("group_label").loc[group_order].reset_index()
    ax.axis("off")
    lines = []
    for row in df.itertuples(index=False):
        status = str(getattr(row, "descriptor_status", "pass"))
        lines.append(
            f"{row.group_label}: {row.tier} | B={row.recommended_b} | descriptor={status}"
        )
    ax.text(
        0.0,
        1.0,
        "Summary\n\n" + "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-dir",
        default="output/lv_experiment_all_metapaths/lv_group_qc_experiment",
        help="Directory containing LV group QC CSV outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc_dir = Path(args.qc_dir)

    group_qc_df = _load_csv(
        qc_dir / "group_qc_summary.csv",
        ["lv_id", "target_set_id", "tier", "recommended_b", "descriptor_status"],
    )
    descriptor_dev_df = _load_csv(
        qc_dir / "descriptor_deviation.csv",
        ["lv_id", "target_set_id", "variable", "deviation"],
    )
    within_df = _load_csv(
        qc_dir / "within_null_stability_summary.csv",
        ["control", "b", "lv_id", "target_set_id", "mean_spearman_rho"],
    )
    between_df = _load_csv(
        qc_dir / "between_null_agreement.csv",
        ["b", "lv_id", "target_set_id", "mean_spearman_rho"],
    )

    group_qc_df = _group_label(group_qc_df)
    descriptor_dev_df = _group_label(descriptor_dev_df)
    within_df = _group_label(within_df)
    between_df = _group_label(between_df)
    group_order = _ordered_group_labels(group_qc_df)

    within_specs = _metric_specs(within_df)
    between_specs = _metric_specs(between_df)

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.2, 1.0, 1.0, 0.24],
        width_ratios=[1.2, 1.2, 1.2, 1.0],
        hspace=0.34,
        wspace=0.28,
    )

    ax_heat = fig.add_subplot(gs[0, :2])
    heat_im = _descriptor_heatmap(ax_heat, descriptor_dev_df, group_order)
    cbar = fig.colorbar(heat_im, ax=ax_heat, fraction=0.025, pad=0.01)
    cbar.set_label("Scaled deviation")

    null_grid = gs[0, 2:].subgridspec(2, 2, hspace=0.36, wspace=0.25)
    null_axes = [fig.add_subplot(null_grid[i, j]) for i in range(2) for j in range(2)]
    _null_match_panel(null_axes, group_qc_df, group_order)

    within_grid = gs[1, :].subgridspec(2, 2, hspace=0.35, wspace=0.25)
    within_axes = [fig.add_subplot(within_grid[i, j]) for i in range(2) for j in range(2)]
    for ax, (metric_col, metric_label) in zip(within_axes, within_specs):
        _plot_within_null_metric(ax, within_df, metric_col, metric_label, group_order)
    for ax in within_axes[len(within_specs) :]:
        ax.axis("off")
    if within_axes:
        handles, labels = within_axes[0].get_legend_handles_labels()
        if handles:
            within_axes[0].legend(handles, labels, fontsize=8, loc="lower right", ncol=2)

    between_grid = gs[2, :].subgridspec(2, 2, hspace=0.35, wspace=0.25)
    between_axes = [fig.add_subplot(between_grid[i, j]) for i in range(2) for j in range(2)]
    for ax, (metric_col, metric_label) in zip(between_axes, between_specs):
        _plot_between_null_metric(ax, between_df, metric_col, metric_label, group_order)
    for ax in between_axes[len(between_specs) :]:
        ax.axis("off")
    if between_axes:
        handles, labels = between_axes[0].get_legend_handles_labels()
        if handles:
            between_axes[0].legend(handles, labels, fontsize=8, loc="lower right")

    ax_text = fig.add_subplot(gs[3, :])
    _decision_text(ax_text, group_qc_df, group_order)

    fig.suptitle(
        "LV Group QC Dashboard\n"
        "Top: descriptor deviation and null-generator QC | "
        "Middle: within-null seed stability | Bottom: random-vs-permuted agreement",
        fontsize=18,
        y=0.98,
    )
    out_path = qc_dir / "lv_group_qc_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
