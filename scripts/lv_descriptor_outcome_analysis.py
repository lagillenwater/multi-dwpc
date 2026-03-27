#!/usr/bin/env python3
"""Build exploratory LV descriptor/outcome analyses and plots."""

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
    "LV603 | neutrophil_homeostasis": "#2ca02c",
}

NULL_STYLES = {
    "permuted": ("-", "o"),
    "random": ("--", "s"),
}


def _load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _group_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_label"] = out["lv_id"].astype(str) + " | " + out["target_set_id"].astype(str)
    return out


def _group_color(label: str) -> str:
    return GROUP_COLORS.get(label, "#444444")


def _iqr(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def _safe_p90(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    return float(values.quantile(0.90))


def _reverse_metapath_abbrev(metapath: str) -> str:
    node_abbrevs = {"G", "BP", "CC", "MF", "PW", "A", "D", "C", "SE", "S", "PC"}
    edge_abbrevs = {"p", "i", "c", "r", ">", "<", "a", "d", "u", "e", "b", "t", "l"}
    tokens: list[str] = []
    pos = 0
    while pos < len(metapath):
        if pos + 2 <= len(metapath) and metapath[pos : pos + 2] in node_abbrevs:
            tokens.append(metapath[pos : pos + 2])
            pos += 2
        elif metapath[pos] in node_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        elif metapath[pos] in edge_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        else:
            pos += 1
    direction_map = {">": "<", "<": ">"}
    return "".join(direction_map.get(token, token) for token in reversed(tokens))


def _build_metapath_length_map(stats_df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in stats_df.itertuples(index=False):
        mp = str(row.metapath)
        length = int(row.length)
        out[mp] = length
        out[_reverse_metapath_abbrev(mp)] = length
    return out


def _build_descriptor_outcome_table(qc_dir: Path) -> pd.DataFrame:
    snapshot_df = _load_csv(
        qc_dir / "lv_qc_metric_snapshot.csv",
        [
            "lv_id",
            "target_set_id",
            "target_set_label",
            "n_genes",
            "gene_promiscuity_median",
            "gene_promiscuity_iqr",
            "gene_promiscuity_p90",
            "gene_promiscuity_max",
            "target_set_size",
            "target_promiscuity_median",
            "target_promiscuity_iqr",
            "target_promiscuity_p90",
            "target_promiscuity_max",
            "n_candidate_metapaths",
            "n_metapaths_len_1",
            "n_metapaths_len_2",
            "n_metapaths_len_3",
            "score_sparsity",
            "mean_spearman_rho_permuted",
            "mean_spearman_rho_random",
            "mean_rbo_permuted",
            "mean_rbo_random",
            "between_null_mean_spearman_rho",
            "between_null_mean_rbo",
            "mean_standardized_top10_gap_permuted",
            "mean_standardized_top10_gap_random",
            "near_tie_count_top10_permuted",
            "near_tie_count_top10_random",
            "near_tie_count_top20_permuted",
            "near_tie_count_top20_random",
            "snapshot_b",
        ],
    )
    table = snapshot_df.rename(
        columns={
            "mean_spearman_rho_permuted": "within_null_mean_spearman_rho_permuted",
            "mean_spearman_rho_random": "within_null_mean_spearman_rho_random",
            "mean_rbo_permuted": "within_null_mean_rbo_permuted",
            "mean_rbo_random": "within_null_mean_rbo_random",
            "between_null_mean_spearman_rho": "random_vs_permuted_mean_spearman_rho",
            "between_null_mean_rbo": "random_vs_permuted_mean_rbo",
        }
    )
    table = _group_label(table)
    return table.sort_values(["lv_id", "target_set_id"]).reset_index(drop=True)


def _plot_qc_vs_predictors(table: pd.DataFrame, output_path: Path) -> None:
    pairs = [
        ("n_genes", "within_null_mean_spearman_rho_permuted"),
        ("n_genes", "within_null_mean_rbo_permuted"),
        ("gene_promiscuity_p90", "random_vs_permuted_mean_spearman_rho"),
        ("gene_promiscuity_p90", "random_vs_permuted_mean_rbo"),
        ("target_set_size", "mean_standardized_top10_gap_permuted"),
        ("target_promiscuity_p90", "mean_standardized_top10_gap_random"),
        ("n_candidate_metapaths", "within_null_mean_rbo_permuted"),
        ("n_candidate_metapaths", "random_vs_permuted_mean_rbo"),
        ("score_sparsity", "near_tie_count_top10_permuted"),
        ("score_sparsity", "near_tie_count_top20_random"),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(13, 18))
    for ax, (x_col, y_col) in zip(axes.flat, pairs):
        xs = pd.to_numeric(table[x_col], errors="coerce")
        ys = pd.to_numeric(table[y_col], errors="coerce")
        for row in table.itertuples(index=False):
            color = _group_color(row.group_label)
            ax.scatter(getattr(row, x_col), getattr(row, y_col), color=color, s=70, alpha=0.9)
            ax.annotate(str(row.lv_id), (getattr(row, x_col), getattr(row, y_col)), xytext=(5, 5), textcoords="offset points", fontsize=8)
        finite_x = xs[np.isfinite(xs)]
        finite_y = ys[np.isfinite(ys)]
        if not finite_x.empty:
            xspan = max(float(finite_x.max() - finite_x.min()), 1e-6)
            ax.set_xlim(float(finite_x.min()) - 0.10 * xspan, float(finite_x.max()) + 0.10 * xspan)
        if not finite_y.empty:
            yspan = max(float(finite_y.max() - finite_y.min()), 1e-6)
            ax.set_ylim(float(finite_y.min()) - 0.10 * yspan, float(finite_y.max()) + 0.10 * yspan)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.22, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("LVQC: QC outputs vs upstream predictors", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _correlation_table(table: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "n_genes",
        "gene_promiscuity_median",
        "gene_promiscuity_iqr",
        "gene_promiscuity_p90",
        "gene_promiscuity_max",
        "target_set_size",
        "target_promiscuity_median",
        "target_promiscuity_iqr",
        "target_promiscuity_p90",
        "target_promiscuity_max",
        "n_candidate_metapaths",
        "n_metapaths_len_1",
        "n_metapaths_len_2",
        "n_metapaths_len_3",
        "score_sparsity",
    ]
    outcomes = [
        "within_null_mean_spearman_rho_permuted",
        "within_null_mean_spearman_rho_random",
        "within_null_mean_rbo_permuted",
        "within_null_mean_rbo_random",
        "random_vs_permuted_mean_spearman_rho",
        "random_vs_permuted_mean_rbo",
        "mean_standardized_top10_gap_permuted",
        "mean_standardized_top10_gap_random",
        "near_tie_count_top10_permuted",
        "near_tie_count_top10_random",
        "near_tie_count_top20_permuted",
        "near_tie_count_top20_random",
    ]
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            subset = table[[predictor, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < 2:
                pearson_r = np.nan
                spearman_r = np.nan
            else:
                pearson_r = float(subset[predictor].corr(subset[outcome], method="pearson"))
                spearman_r = float(subset[predictor].corr(subset[outcome], method="spearman"))
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_points": int(len(subset)),
                    "pearson_r": pearson_r,
                    "spearman_r": spearman_r,
                }
            )
    return pd.DataFrame(rows).sort_values(["predictor", "outcome"]).reset_index(drop=True)


def _outlier_table(table: pd.DataFrame) -> pd.DataFrame:
    outcome_cols = [
        "within_null_mean_spearman_rho_permuted",
        "within_null_mean_spearman_rho_random",
        "within_null_mean_rbo_permuted",
        "within_null_mean_rbo_random",
        "random_vs_permuted_mean_spearman_rho",
        "random_vs_permuted_mean_rbo",
        "mean_standardized_top10_gap_permuted",
        "mean_standardized_top10_gap_random",
        "near_tie_count_top10_permuted",
        "near_tie_count_top10_random",
        "near_tie_count_top20_permuted",
        "near_tie_count_top20_random",
    ]
    z_df = pd.DataFrame(index=table.index)
    for col in outcome_cols:
        values = pd.to_numeric(table[col], errors="coerce")
        std = float(values.std(ddof=1)) if len(values.dropna()) >= 2 else np.nan
        if pd.notna(std) and std > 1e-12:
            z_df[col] = (values - float(values.mean())) / std
        else:
            z_df[col] = np.nan
    out = table[["group_label", "lv_id", "target_set_id"]].copy()
    out["max_abs_outcome_z"] = z_df.abs().max(axis=1)
    out["driver_metric"] = z_df.abs().idxmax(axis=1)
    return out.sort_values("max_abs_outcome_z", ascending=False).reset_index(drop=True)


def _plot_ranked_score_curves(
    analysis_dir: Path,
    table: pd.DataFrame,
    output_path: Path,
    snapshot_b: int,
) -> None:
    runs_df = _load_csv(
        analysis_dir / "all_runs_long.csv",
        ["control", "b", "seed", "lv_id", "target_set_id", "metapath", "diff"],
    )
    runs_df = runs_df[runs_df["b"].astype(int) == int(snapshot_b)].copy()
    group_order = table["group_label"].astype(str).tolist()
    fig, axes = plt.subplots(len(group_order), 1, figsize=(12, max(4.0 * len(group_order), 4.5)), sharex=False)
    if len(group_order) == 1:
        axes = [axes]
    for ax, group_label in zip(axes, group_order):
        lv_id, target_set_id = [part.strip() for part in group_label.split("|", maxsplit=1)]
        subset = runs_df[
            (runs_df["lv_id"].astype(str) == lv_id)
            & (runs_df["target_set_id"].astype(str) == target_set_id)
        ].copy()
        color = _group_color(group_label)
        for control in sorted(subset["control"].astype(str).unique().tolist()):
            control_df = (
                subset[subset["control"].astype(str) == control]
                .groupby("metapath", as_index=False)["diff"]
                .mean()
                .sort_values(["diff", "metapath"], ascending=[False, True])
                .reset_index(drop=True)
            )
            control_df["rank"] = np.arange(1, len(control_df) + 1)
            linestyle, marker = NULL_STYLES.get(control, ("-", "o"))
            ax.plot(
                control_df["rank"].astype(int),
                control_df["diff"].astype(float),
                linestyle=linestyle,
                marker=marker,
                markersize=3.2,
                linewidth=2.0,
                color=color,
                alpha=0.95,
                label=control,
            )
        ax.set_title(group_label)
        ax.set_xlabel("Metapath rank")
        ax.set_ylabel("Mean diff score")
        ax.grid(alpha=0.22, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"LVQC ranked score curves (B={int(snapshot_b)})", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_metapath_length_disagreement(
    qc_dir: Path,
    workspace_dir: Path,
    stats_path: Path,
    output_csv: Path,
    output_plot: Path,
    snapshot_b: int,
) -> None:
    scatter_df = _load_csv(
        qc_dir / "between_null_rank_scatter.csv",
        ["b", "lv_id", "target_set_id", "metapath", "mean_rank_permuted", "mean_rank_random"],
    )
    feature_manifest = _load_csv(workspace_dir / "feature_manifest.csv", ["target_set_id", "metapath"])
    stats_df = _load_csv(stats_path, ["metapath", "length"])
    length_map = _build_metapath_length_map(stats_df)

    scatter_df = scatter_df[scatter_df["b"].astype(int) == int(snapshot_b)].copy()
    scatter_df["abs_rank_shift"] = (
        scatter_df["mean_rank_permuted"].astype(float) - scatter_df["mean_rank_random"].astype(float)
    ).abs()
    feature_manifest = feature_manifest.copy()
    feature_manifest["metapath_length"] = feature_manifest["metapath"].map(length_map)
    merged = scatter_df.merge(feature_manifest, on=["target_set_id", "metapath"], how="left")
    summary = (
        merged.groupby(["lv_id", "target_set_id", "metapath_length"], as_index=False)
        .agg(
            n_metapaths=("metapath", "nunique"),
            mean_abs_rank_shift=("abs_rank_shift", "mean"),
            median_abs_rank_shift=("abs_rank_shift", "median"),
            p90_abs_rank_shift=("abs_rank_shift", _safe_p90),
        )
        .sort_values(["lv_id", "target_set_id", "metapath_length"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_csv, index=False)

    summary = _group_label(summary)
    group_order = summary["group_label"].drop_duplicates().astype(str).tolist()
    lengths = sorted(summary["metapath_length"].dropna().astype(int).unique().tolist())
    fig, axes = plt.subplots(len(group_order), 1, figsize=(10.5, max(3.6 * len(group_order), 4.0)), sharex=True)
    if len(group_order) == 1:
        axes = [axes]
    for ax, group_label in zip(axes, group_order):
        gdf = summary[summary["group_label"].astype(str) == group_label].copy()
        color = _group_color(group_label)
        ax.bar(
            gdf["metapath_length"].astype(int),
            gdf["mean_abs_rank_shift"].astype(float),
            color=color,
            alpha=0.85,
            width=0.6,
        )
        ax.set_title(group_label)
        ax.set_ylabel("Mean abs rank shift")
        ax.grid(alpha=0.22, linestyle=":", axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Metapath length")
    axes[-1].set_xticks(lengths)
    fig.suptitle(f"LVQC metapath-length disagreement (B={int(snapshot_b)})", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-dir", default="output/lv_experiment_all_metapaths/lv_group_qc_experiment")
    parser.add_argument("--workspace-dir", default="output/lv_experiment_all_metapaths")
    parser.add_argument("--analysis-dir", default="output/lv_experiment_all_metapaths/lv_rank_stability_experiment")
    parser.add_argument("--snapshot-b", type=int, default=5)
    parser.add_argument("--metapath-stats-path", default="data/metapath-dwpc-stats.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qc_dir = Path(args.qc_dir)
    workspace_dir = Path(args.workspace_dir)
    analysis_dir = Path(args.analysis_dir)

    table = _build_descriptor_outcome_table(qc_dir)
    table.to_csv(qc_dir / "lv_descriptor_outcome_table.csv", index=False)

    near_tie_cols = [
        "group_label",
        "near_tie_count_top10_permuted",
        "near_tie_count_top10_random",
        "near_tie_count_top20_permuted",
        "near_tie_count_top20_random",
    ]
    table[near_tie_cols].to_csv(qc_dir / "lv_near_tie_summary.csv", index=False)

    corr_df = _correlation_table(table)
    corr_df.to_csv(qc_dir / "lv_qc_predictor_correlations.csv", index=False)

    outlier_df = _outlier_table(table)
    outlier_df.to_csv(qc_dir / "lv_qc_outliers.csv", index=False)

    _plot_qc_vs_predictors(table, qc_dir / "lv_qc_vs_predictors.png")
    _plot_ranked_score_curves(
        analysis_dir=analysis_dir,
        table=table,
        output_path=qc_dir / "lv_ranked_score_curves.png",
        snapshot_b=int(args.snapshot_b),
    )
    _plot_metapath_length_disagreement(
        qc_dir=qc_dir,
        workspace_dir=workspace_dir,
        stats_path=Path(args.metapath_stats_path),
        output_csv=qc_dir / "lv_metapath_length_disagreement_summary.csv",
        output_plot=qc_dir / "lv_metapath_length_disagreement.png",
        snapshot_b=int(args.snapshot_b),
    )

    print(f"Saved descriptor/outcome table: {qc_dir / 'lv_descriptor_outcome_table.csv'}")
    print(f"Saved predictor plot bundle: {qc_dir / 'lv_qc_vs_predictors.png'}")
    print(f"Saved predictor correlations: {qc_dir / 'lv_qc_predictor_correlations.csv'}")
    print(f"Saved outlier summary: {qc_dir / 'lv_qc_outliers.csv'}")
    print(f"Saved ranked score curves: {qc_dir / 'lv_ranked_score_curves.png'}")
    print(f"Saved near-tie summary: {qc_dir / 'lv_near_tie_summary.csv'}")
    print(f"Saved metapath-length disagreement summary: {qc_dir / 'lv_metapath_length_disagreement_summary.csv'}")
    print(f"Saved metapath-length disagreement plot: {qc_dir / 'lv_metapath_length_disagreement.png'}")


if __name__ == "__main__":
    main()
