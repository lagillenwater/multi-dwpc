#!/usr/bin/env python3
"""Plot LV per-group QC summaries and dual-null stability results."""

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
    cols = ["descriptor_in_envelope", "random_qc_pass", "permuted_qc_pass"]
    labels = ["In envelope", "Random QC", "Permuted QC"]

    fig_w = max(8.0, 0.9 * len(summary_df))
    fig_h = max(4.8, 0.55 * len(summary_df) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    y_pos = np.arange(len(summary_df))

    for x_idx, col in enumerate(cols):
        values = summary_df[col].astype(int).to_numpy()
        colors = np.where(values > 0, "#2ca02c", "#d62728")
        ax.scatter(
            np.full(len(summary_df), x_idx),
            y_pos,
            s=130,
            c=colors,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )

    tier_x = len(cols) + 0.9
    for idx, row in summary_df.iterrows():
        ax.text(
            tier_x,
            y_pos[idx],
            str(row["tier"]),
            va="center",
            ha="left",
            color=TIER_COLORS.get(str(row["tier"]), "#333333"),
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            tier_x + 1.6,
            y_pos[idx],
            f"B={row['recommended_b']}",
            va="center",
            ha="left",
            color="#333333",
            fontsize=10,
        )

    ax.set_xlim(-0.5, tier_x + 2.4)
    ax.set_ylim(-0.7, len(summary_df) - 0.3)
    ax.set_xticks(list(range(len(labels))) + [tier_x, tier_x + 1.6])
    ax.set_xticklabels(labels + ["Tier", "Recommended B"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df["group_label"].tolist())
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("LV group QC decision summary")
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
        figsize=(5.2 * len(groups), 3.8 * len(specs)),
        sharex=True,
        sharey="row",
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
            for control in sorted(group_df["control"].astype(str).unique().tolist()):
                control_df = group_df[group_df["control"].astype(str) == control].copy().sort_values("b")
                if control_df.empty:
                    continue
                ax.plot(
                    control_df["b"].astype(float),
                    control_df[metric_col].astype(float),
                    marker="o",
                    linewidth=2.2,
                    markersize=6.5,
                    color=CONTROL_COLORS.get(control, "#333333"),
                    label=control,
                )
            ax.axhline(threshold, color="#999999", linestyle="--", linewidth=1.0)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if row_idx == 0:
                tier = summary_df[
                    (summary_df["lv_id"].astype(str) == str(group["lv_id"]))
                    & (summary_df["target_set_id"].astype(str) == str(group["target_set_id"]))
                ]["tier"].iloc[0]
                ax.set_title(f"{group['group_label']}\n{tier}")
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(specs) - 1:
                ax.set_xlabel("B")
    axes[0, -1].legend(title="Null", loc="best")
    fig.suptitle("LV dual-null stability by B")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_null_match_qc(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df = _ordered_groups(summary_df)
    labels = summary_df["group_label"].tolist()
    x = np.arange(len(summary_df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = np.asarray(axes, dtype=object)

    axes[0, 0].bar(x, summary_df["random_match_mae"].astype(float), color=CONTROL_COLORS["random"], alpha=0.8)
    axes[0, 0].axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    axes[0, 0].set_ylabel("Random match MAE")
    axes[0, 0].set_title("Random null: match error")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(
        x,
        summary_df["random_within_tolerance_rate"].astype(float),
        color=CONTROL_COLORS["random"],
        alpha=0.8,
    )
    axes[0, 1].axhline(0.90, color="#999999", linestyle="--", linewidth=1.0)
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_title("Random null: within tolerance")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(
        x,
        summary_df["perm_edge_overlap_with_real_mean"].astype(float),
        color=CONTROL_COLORS["permuted"],
        alpha=0.8,
    )
    axes[1, 0].axhline(0.70, color="#999999", linestyle="--", linewidth=1.0)
    axes[1, 0].set_ylabel("Mean overlap")
    axes[1, 0].set_title("Permuted null: overlap with real")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(
        x,
        summary_df["perm_pairwise_overlap_mean"].astype(float),
        color=CONTROL_COLORS["permuted"],
        alpha=0.8,
    )
    axes[1, 1].set_ylabel("Mean pairwise overlap")
    axes[1, 1].set_title("Permuted null: replicate overlap")
    axes[1, 1].grid(axis="y", alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    for ax in axes[0, :]:
        ax.set_xticks(x)
        ax.set_xticklabels([])

    fig.suptitle("LV null-match QC summary")
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
