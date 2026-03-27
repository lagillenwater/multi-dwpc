#!/usr/bin/env python3
"""Plot adjacent metapath score gaps from LV QC outputs."""

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


GROUP_COLORS = {
    "LV246 | adipose_tissue": "#1f77b4",
    "LV57 | hypothyroidism": "#ff7f0e",
    "LV603 | neutrophil_bp": "#2ca02c",
}

CONTROL_STYLES = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc-dir",
        default="output/lv_experiment_all_metapaths/lv_group_qc_experiment",
        help="Directory containing score_separation_table.csv",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=20,
        help="Maximum upper rank to display",
    )
    parser.add_argument(
        "--metric",
        choices=["raw", "standardized", "both"],
        default="both",
        help="Which score-gap metric to plot",
    )
    return parser.parse_args()


def _plot_metric(
    df: pd.DataFrame,
    qc_dir: Path,
    group_order: list[str],
    metric_col: str,
    metric_label: str,
    metric_slug: str,
) -> Path:
    n_groups = len(group_order)
    fig, axes = plt.subplots(
        n_groups,
        1,
        figsize=(11, max(3.2 * n_groups, 4.2)),
        sharex=True,
    )
    if n_groups == 1:
        axes = [axes]

    all_vals = df[metric_col].astype(float).to_numpy()
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    span = max(1e-6, ymax - ymin)

    for ax, group_label in zip(axes, group_order):
        gdf = df[df["group_label"].astype(str) == group_label].copy()
        color = GROUP_COLORS.get(group_label, "#444444")
        for control in sorted(gdf["control"].astype(str).unique().tolist()):
            cdf = gdf[gdf["control"].astype(str) == control].copy().sort_values("rank_upper")
            linestyle, marker = CONTROL_STYLES.get(control, ("-", "o"))
            ax.plot(
                cdf["rank_upper"].astype(int),
                cdf[metric_col].astype(float),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=5.0,
                color=color,
                alpha=0.95,
                label=control,
            )
        ax.set_title(group_label)
        ax.set_ylabel(metric_label)
        ax.set_ylim(ymin - 0.08 * span, ymax + 0.10 * span)
        ax.grid(alpha=0.22, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Upper rank k in gap between ranks k and k+1")
    fig.suptitle(f"LVQC score separation ({metric_slug})", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = qc_dir / f"lv_score_separation_{metric_slug}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    qc_dir = Path(args.qc_dir)
    df = _load_csv(
        qc_dir / "score_separation_table.csv",
        [
            "lv_id",
            "target_set_id",
            "control",
            "rank_upper",
            "rank_lower",
            "score_gap",
            "standardized_score_gap",
            "diff_upper",
            "diff_lower",
        ],
    )
    df = _group_label(df)
    df = df[df["rank_upper"].astype(int) <= int(args.max_rank)].copy()

    group_order = df["group_label"].drop_duplicates().astype(str).tolist()
    n_groups = len(group_order)
    if n_groups == 0:
        raise ValueError("No rows available after filtering score_separation_table.csv")
    metrics = []
    if args.metric in {"raw", "both"}:
        metrics.append(("score_gap", "Adjacent score gap", "raw"))
    if args.metric in {"standardized", "both"}:
        metrics.append(("standardized_score_gap", "Standardized adjacent score gap", "standardized"))

    out_paths = []
    for metric_col, metric_label, metric_slug in metrics:
        out_paths.append(
            _plot_metric(
                df=df,
                qc_dir=qc_dir,
                group_order=group_order,
                metric_col=metric_col,
                metric_label=metric_label,
                metric_slug=metric_slug,
            )
        )
    for out_path in out_paths:
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
