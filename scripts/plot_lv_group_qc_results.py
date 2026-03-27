#!/usr/bin/env python3
"""Plot LV per-group QC summaries and dual-null stability results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

matplotlib.use("Agg")


CONTROL_COLORS = {
    "permuted": "#1f77b4",
    "random": "#d62728",
}

TIER_COLORS = {
    "Production Ready": "#2ca02c",
    "Production With Higher B": "#ff7f0e",
    "Tune And Recheck": "#d62728",
    "Out Of Family": "#7f7f7f",
}


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _group_label(row: pd.Series) -> str:
    return f"{row['lv_id']} | {row['target_set_id']}"


def _ordered_groups(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["group_label"] = out.apply(_group_label, axis=1)
    out["tier_rank"] = out["tier"].map(
        {
            "Production Ready": 0,
            "Production With Higher B": 1,
            "Tune And Recheck": 2,
            "Out Of Family": 3,
        }
    ).fillna(9)
    return out.sort_values(["tier_rank", "lv_id", "target_set_id"]).reset_index(drop=True)


def _plot_decision_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df = _ordered_groups(summary_df)
    bool_cols = ["descriptor_in_envelope", "random_qc_pass", "permuted_qc_pass"]
    bool_labels = ["Envelope", "Random", "Permuted"]
    bool_matrix = summary_df[bool_cols].astype(int).to_numpy()

    fig = plt.figure(figsize=(12.5, max(4.5, 1.6 + 0.95 * len(summary_df))))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2.2], wspace=0.06)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1], sharey=ax_heat)

    cmap = ListedColormap(["#d95f5f", "#2f9e44"])
    ax_heat.imshow(bool_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for row_idx in range(bool_matrix.shape[0]):
        for col_idx in range(bool_matrix.shape[1]):
            ax_heat.text(
                col_idx,
                row_idx,
                "PASS" if bool_matrix[row_idx, col_idx] else "FAIL",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
    ax_heat.set_xticks(np.arange(len(bool_labels)))
    ax_heat.set_xticklabels(bool_labels)
    ax_heat.set_yticks(np.arange(len(summary_df)))
    ax_heat.set_yticklabels(summary_df["group_label"].tolist())
    ax_heat.set_title("QC Gates")
    ax_heat.set_xticks(np.arange(-0.5, len(bool_labels), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(summary_df), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    ax_text.set_xlim(0, 2.8)
    ax_text.set_ylim(len(summary_df) - 0.5, -0.5)
    ax_text.set_xticks([0.75, 2.05])
    ax_text.set_xticklabels(["Decision", "Production B"])
    ax_text.set_yticks(np.arange(len(summary_df)))
    ax_text.tick_params(axis="y", left=False, labelleft=False)
    ax_text.set_title("Decision")
    for idx, row in summary_df.iterrows():
        tier = str(row["tier"])
        tier_color = TIER_COLORS.get(tier, "#333333")
        ax_text.add_patch(
            Rectangle((0.10, idx - 0.34), 1.35, 0.68, facecolor=tier_color, alpha=0.12, edgecolor="none")
        )
        ax_text.text(
            0.18,
            idx,
            tier,
            va="center",
            ha="left",
            color=tier_color,
            fontsize=12,
            fontweight="bold",
        )
        ax_text.text(
            1.82,
            idx,
            f"B={row['recommended_b']}",
            va="center",
            ha="left",
            color="#222222",
            fontsize=12,
            fontweight="bold",
        )
    ax_text.grid(axis="x", alpha=0.18)
    fig.suptitle("LV Group QC Decision Summary", y=0.98, fontsize=17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _metric_specs(entity_df: pd.DataFrame) -> list[tuple[str, str, float]]:
    specs: list[tuple[str, str, float]] = []
    if "mean_topk_jaccard_5" in entity_df.columns:
        specs.append(("mean_topk_jaccard_5", "Mean top-5 Jaccard", 0.80))
    if "mean_topk_jaccard_10" in entity_df.columns:
        specs.append(("mean_topk_jaccard_10", "Mean top-10 Jaccard", 0.85))
    specs.append(("mean_spearman_rho", "Mean Spearman rho", 0.90))
    return specs


def _plot_dual_null_stability(
    summary_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    summary_df = _ordered_groups(summary_df)
    specs = _metric_specs(entity_df)
    groups = summary_df[["lv_id", "target_set_id", "group_label"]].drop_duplicates().to_dict("records")
    fig, axes = plt.subplots(
        len(specs),
        len(groups),
        figsize=(5.4 * len(groups), 4.1 * len(specs)),
        sharex=True,
        sharey=False,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        if len(specs) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    for col_idx, group in enumerate(groups):
        group_df = entity_df[
            (entity_df["lv_id"].astype(str) == str(group["lv_id"]))
            & (entity_df["target_set_id"].astype(str) == str(group["target_set_id"]))
        ].copy()
        for row_idx, (metric_col, metric_label, threshold) in enumerate(specs):
            ax = axes[row_idx, col_idx]
            metric_vals = []
            for control in sorted(group_df["control"].astype(str).unique().tolist()):
                control_df = group_df[group_df["control"].astype(str) == control].copy().sort_values("b")
                if control_df.empty:
                    continue
                metric_vals.extend(control_df[metric_col].astype(float).tolist())
                ax.plot(
                    control_df["b"].astype(float),
                    control_df[metric_col].astype(float),
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                    color=CONTROL_COLORS.get(control, "#333333"),
                    label=control,
                )
                for _, point in control_df.iterrows():
                    ax.text(
                        float(point["b"]),
                        float(point[metric_col]) + 0.012,
                        f"{float(point[metric_col]):.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=CONTROL_COLORS.get(control, "#333333"),
                    )
            ax.axhline(threshold, color="#999999", linestyle="--", linewidth=1.0)
            if metric_vals:
                y_min = min(metric_vals + [threshold])
                if y_min >= 0.75:
                    lower = max(0.70, y_min - 0.06)
                elif y_min >= 0.50:
                    lower = max(0.45, y_min - 0.08)
                else:
                    lower = max(0.0, y_min - 0.05)
                ax.set_ylim(lower, 1.02)
            ax.grid(alpha=0.25, linestyle=":")
            if row_idx == 0:
                tier = summary_df[
                    (summary_df["lv_id"].astype(str) == str(group["lv_id"]))
                    & (summary_df["target_set_id"].astype(str) == str(group["target_set_id"]))
                ]["tier"].iloc[0]
                ax.set_title(f"{group['group_label']}\n{tier}", color=TIER_COLORS.get(str(tier), "#222222"))
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(specs) - 1:
                ax.set_xlabel("B")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, -1].legend(title="Null", loc="best", frameon=True)
    title = "LV Dual-Null Stability By B"
    if len(specs) == 1:
        title += " (only Spearman rho available in aggregate)"
    fig.suptitle(title, y=0.99, fontsize=17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_null_match_qc(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df = _ordered_groups(summary_df)
    labels = summary_df["group_label"].tolist()
    y = np.arange(len(summary_df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
    axes = np.asarray(axes, dtype=object)

    panels = [
        (axes[0, 0], "random_match_mae", "Random match MAE", 1.0, CONTROL_COLORS["random"]),
        (
            axes[0, 1],
            "random_within_tolerance_rate",
            "Random within-tolerance rate",
            0.90,
            CONTROL_COLORS["random"],
        ),
        (
            axes[1, 0],
            "perm_edge_overlap_with_real_mean",
            "Permuted overlap with real",
            0.70,
            CONTROL_COLORS["permuted"],
        ),
        (
            axes[1, 1],
            "perm_pairwise_overlap_mean",
            "Permuted replicate overlap",
            np.nan,
            CONTROL_COLORS["permuted"],
        ),
    ]
    for ax, col, title, threshold, color in panels:
        vals = summary_df[col].astype(float).to_numpy()
        ax.hlines(y, np.nanmin(vals), vals, color=color, alpha=0.30, linewidth=2.5)
        ax.scatter(vals, y, s=95, color=color, edgecolors="white", linewidths=1.0, zorder=3)
        for idx, value in enumerate(vals):
            ax.text(value, y[idx] + 0.12, f"{value:.3f}", fontsize=8, color=color, ha="center")
        if not pd.isna(threshold):
            ax.axvline(threshold, color="#999999", linestyle="--", linewidth=1.0)
        data_min = float(np.nanmin(vals))
        data_max = float(np.nanmax(vals))
        if "rate" in col:
            lower = min(data_min, threshold if not pd.isna(threshold) else data_min) - 0.05
            upper = 1.01
            ax.set_xlim(max(0.0, lower), upper)
        else:
            span = max(0.03, data_max - data_min)
            lower = data_min - 0.25 * span
            upper = data_max + 0.25 * span
            if not pd.isna(threshold):
                lower = min(lower, threshold - 0.10 * max(1.0, threshold))
                upper = max(upper, threshold + 0.10 * max(1.0, threshold))
            ax.set_xlim(max(0.0, lower), upper)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
    for ax in axes[:, 1]:
        ax.set_yticks(y)
        ax.tick_params(axis="y", labelleft=False)
    for ax in axes.ravel():
        ax.invert_yaxis()

    fig.suptitle("LV Null-Match QC Summary", y=0.98, fontsize=17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-dir",
        default="output/lv_experiment/lv_group_qc_experiment",
        help="Directory containing group_qc_summary.csv",
    )
    parser.add_argument(
        "--analysis-dir",
        default="output/lv_experiment/lv_rank_stability_experiment",
        help="Directory containing lv_stability_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc_dir = Path(args.qc_dir)
    analysis_dir = Path(args.analysis_dir)

    summary_df = _load_csv(
        qc_dir / "group_qc_summary.csv",
        ["lv_id", "target_set_id", "tier", "recommended_b"],
    )
    entity_df = _load_csv(
        analysis_dir / "lv_stability_summary.csv",
        ["control", "b", "lv_id", "target_set_id", "mean_spearman_rho"],
    )

    decision_path = qc_dir / "group_decision_summary.png"
    _plot_decision_summary(summary_df, decision_path)
    print(f"Saved plot: {decision_path}")

    stability_path = qc_dir / "dual_null_stability_by_b.png"
    _plot_dual_null_stability(summary_df, entity_df, stability_path)
    print(f"Saved plot: {stability_path}")

    null_qc_path = qc_dir / "null_match_qc_summary.png"
    _plot_null_match_qc(summary_df, null_qc_path)
    print(f"Saved plot: {null_qc_path}")


if __name__ == "__main__":
    main()
