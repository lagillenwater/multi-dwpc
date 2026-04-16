#!/usr/bin/env python3
"""Year-specific intermediate-sharing plots.

Preserves the 2016-vs-2024 cohort split from
`intermediate_sharing_by_metapath.csv`. Pooling cohorts into a single
"sharing" number destroys the whole point of the year analysis, so this
script never does that -- every sharing/jaccard/coverage panel shows the
cohort breakdown explicitly.

Inputs (written by scripts/year_intermediate_sharing.py):
    <input-dir>/intermediate_sharing_by_metapath.csv
    <input-dir>/intermediate_sharing_summary.csv        (optional)
    <input-dir>/top_intermediates_by_metapath.csv       (optional)

Typical use (as the year full pipeline runs it):
    python3 scripts/plot_year_intermediate_sharing.py \\
        --input-dir output/year_full_analysis/intermediate_sharing/b10 \\
        --top-go-json output/year_full_analysis/top_go_ids.json \\
        --output-dir output/year_full_analysis/intermediate_sharing/b10/figures
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


COHORT_COLORS = {
    "2016_to_2016": "#1f77b4",     # blue: baseline intra-cohort
    "2024_to_2016": "#d62728",     # red: cross-cohort (key signal)
    "2024_to_2024": "#ff7f0e",     # orange: newly added intra-cohort
}

COHORT_LABELS = {
    "2016_to_2016": "2016 \u2194 2016",
    "2024_to_2016": "2024-added \u2192 2016",
    "2024_to_2024": "2024-added \u2194 2024-added",
}

BP_NODES_TSV = "data/nodes/Biological Process.tsv"


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 150) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_bp_name_lookup(repo_root: Path) -> dict[str, str]:
    path = repo_root / BP_NODES_TSV
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if "identifier" not in df.columns or "name" not in df.columns:
        return {}
    return dict(zip(df["identifier"].astype(str), df["name"].astype(str)))


def _short(s: str, n: int = 36) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _go_label(go_id: str, bp_names: dict[str, str]) -> str:
    name = bp_names.get(str(go_id))
    if name:
        return f"{go_id}\n{_short(name, 36)}"
    return str(go_id)


def plot_cohort_sharing_per_go(
    by_metapath_df: pd.DataFrame,
    out_dir: Path,
    bp_names: dict[str, str],
) -> None:
    """Per GO term: grouped bars showing median sharing percentage for each
    cohort pair across that GO's surviving metapaths. The cross-cohort
    (2024-added -> 2016) bar is the central "did the new genes integrate"
    signal; the other two bars are within-cohort baselines.
    """
    cohort_cols = {
        "2016_to_2016": "pct_2016_sharing_with_2016",
        "2024_to_2016": "pct_2024_sharing_with_2016",
        "2024_to_2024": "pct_2024_sharing_with_2024",
    }
    present = {k: v for k, v in cohort_cols.items() if v in by_metapath_df.columns}
    if not present:
        print("  [cohort_sharing] none of the expected sharing columns present; skipping")
        return

    agg = (
        by_metapath_df.groupby("go_id", as_index=False)
        .agg({v: "median" for v in present.values()})
    )
    go_ids = agg["go_id"].astype(str).tolist()
    if not go_ids:
        return

    n = len(go_ids)
    width = 0.27
    x = np.arange(n)

    fig_h = max(5.0, 0.45 * n + 2.5)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * n + 3.0), fig_h))

    bars_drawn = 0
    for offset, (key, col) in enumerate(present.items()):
        values = agg[col].astype(float).fillna(0.0).to_numpy()
        ax.bar(
            x + (offset - (len(present) - 1) / 2) * width,
            values,
            width=width,
            color=COHORT_COLORS[key],
            label=COHORT_LABELS[key],
            edgecolor="black",
            linewidth=0.3,
        )
        bars_drawn += 1

    ax.set_xticks(x)
    ax.set_xticklabels([_go_label(g, bp_names) for g in go_ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median % genes sharing intermediates (across surviving metapaths)")
    ax.set_title(
        "Cohort-wise intermediate sharing per GO term\n"
        "(within-cohort baselines flank the 2024\u21922016 cross-cohort signal)"
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_dir, "cohort_sharing_per_go")


def plot_cohort_jaccard_heatmap(
    by_metapath_df: pd.DataFrame,
    out_dir: Path,
    bp_names: dict[str, str],
) -> None:
    """Heatmap with one row per GO term, three columns for the cohort Jaccard
    medians. Lets you scan which GOs have the strongest cross-cohort mixing.
    """
    jac_cols = {
        "2016_to_2016": "median_jaccard_2016_to_2016",
        "2024_to_2016": "median_jaccard_2024_to_2016",
        "2024_to_2024": "median_jaccard_2024_to_2024",
    }
    present = [(k, v) for k, v in jac_cols.items() if v in by_metapath_df.columns]
    if not present:
        print("  [cohort_jaccard] no jaccard columns present; skipping")
        return

    agg = (
        by_metapath_df.groupby("go_id", as_index=False)
        .agg({v: "median" for _, v in present})
    )
    go_ids = agg["go_id"].astype(str).tolist()
    if not go_ids:
        return

    mat = agg[[v for _, v in present]].astype(float).fillna(0.0).to_numpy()
    labels = [COHORT_LABELS[k] for k, _ in present]

    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(present) + 5), max(4.0, 0.32 * len(go_ids) + 2.0)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.001, mat.max()))
    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(go_ids)))
    ax.set_yticklabels([_go_label(g, bp_names) for g in go_ids], fontsize=8)

    # Annotate each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.max() * 0.6 else "black")
    ax.set_title("Median Jaccard per cohort pair, per GO term")
    fig.colorbar(im, ax=ax, label="Median Jaccard")
    fig.tight_layout()
    save_figure(fig, out_dir, "cohort_jaccard_heatmap")


def plot_top_metapaths_per_go(
    by_metapath_df: pd.DataFrame,
    out_dir: Path,
    bp_names: dict[str, str],
    top_n: int = 10,
) -> None:
    """For each top GO, a horizontal bar chart of its top-N metapaths by
    effect size, colored by which cohort contributes the stronger sharing
    signal (2024\u21922016 dominant vs 2016\u21662016 dominant)."""
    if "metapath" not in by_metapath_df.columns or "effect_size_d" not in by_metapath_df.columns:
        print("  [top_metapaths] missing required columns; skipping")
        return

    go_ids = sorted(by_metapath_df["go_id"].astype(str).unique().tolist())
    n_panels = len(go_ids)
    if n_panels == 0:
        return

    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), squeeze=False)

    for idx, go_id in enumerate(go_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = by_metapath_df[by_metapath_df["go_id"].astype(str) == go_id].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values("effect_size_d", ascending=False).head(top_n)

        # Cohort-dominant color: compare 2024-to-2016 pct vs 2016-to-2016 pct per metapath
        def _color(row: pd.Series) -> str:
            cross = float(row.get("pct_2024_sharing_with_2016", 0) or 0)
            within = float(row.get("pct_2016_sharing_with_2016", 0) or 0)
            if cross >= within + 5:
                return COHORT_COLORS["2024_to_2016"]
            if within >= cross + 5:
                return COHORT_COLORS["2016_to_2016"]
            return "#888888"

        colors = [_color(r_) for _, r_ in sub.iterrows()]
        y = np.arange(len(sub))
        ax.barh(y, sub["effect_size_d"].astype(float), color=colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["metapath"].astype(str).tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("z")
        ax.set_title(_go_label(go_id, bp_names), fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    # Hide unused axes
    for k in range(n_panels, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    # Legend for dominant-cohort color
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COHORT_COLORS["2024_to_2016"], label="2024\u21922016 dominant"),
        plt.Rectangle((0, 0), 1, 1, color=COHORT_COLORS["2016_to_2016"], label="2016\u21662016 dominant"),
        plt.Rectangle((0, 0), 1, 1, color="#888888", label="mixed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
               fontsize=9, frameon=False)
    fig.suptitle(f"Top {top_n} metapaths per GO (colored by dominant cohort sharing)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_figure(fig, out_dir, "top_metapaths_per_go")


def plot_coverage_by_rank_cohort(
    by_metapath_df: pd.DataFrame,
    out_dir: Path,
    bp_names: dict[str, str],
) -> None:
    """Per GO: metapath_rank vs top1_intermediate_coverage, with each point
    colored by which cohort is sharing more (2024\u21922016 vs 2016\u21662016)."""
    if "top1_intermediate_coverage" not in by_metapath_df.columns:
        return
    if "metapath_rank" not in by_metapath_df.columns:
        return

    go_ids = sorted(by_metapath_df["go_id"].astype(str).unique().tolist())
    n_panels = len(go_ids)
    if n_panels == 0:
        return

    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)

    for idx, go_id in enumerate(go_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = by_metapath_df[by_metapath_df["go_id"].astype(str) == go_id].copy()
        sub = sub.sort_values("metapath_rank")

        cross = sub.get("pct_2024_sharing_with_2016", pd.Series(0, index=sub.index)).astype(float).fillna(0)
        within = sub.get("pct_2016_sharing_with_2016", pd.Series(0, index=sub.index)).astype(float).fillna(0)
        colors = [
            COHORT_COLORS["2024_to_2016"] if cr >= wi + 5
            else COHORT_COLORS["2016_to_2016"] if wi >= cr + 5
            else "#888888"
            for cr, wi in zip(cross, within)
        ]
        ax.scatter(
            sub["metapath_rank"].astype(float),
            sub["top1_intermediate_coverage"].astype(float),
            c=colors, s=36, edgecolors="none", alpha=0.85,
        )
        ax.set_xlabel("Metapath rank")
        ax.set_ylabel("Top-1 intermediate coverage (% genes)")
        ax.set_ylim(-5, 105)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(_go_label(go_id, bp_names), fontsize=10)
        ax.grid(alpha=0.25)

    for k in range(n_panels, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    fig.suptitle("Coverage vs metapath rank (per GO); point color = dominant cohort",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_dir, "coverage_by_rank_cohort")


def plot_cohort_gene_counts(
    by_metapath_df: pd.DataFrame,
    out_dir: Path,
    bp_names: dict[str, str],
) -> None:
    """Per GO: 2016 vs 2024-added gene counts as a grouped bar. Useful context
    for how many genes each cohort contributes to each top GO term."""
    cohort_cols = {"2016": "n_genes_2016", "2024-added": "n_genes_2024_added"}
    if not all(c in by_metapath_df.columns for c in cohort_cols.values()):
        return

    first_per_go = by_metapath_df.groupby("go_id", as_index=False).first()
    go_ids = first_per_go["go_id"].astype(str).tolist()
    if not go_ids:
        return

    x = np.arange(len(go_ids))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(go_ids) + 3.0), 5.0))
    ax.bar(
        x - width / 2, first_per_go[cohort_cols["2016"]].astype(float).fillna(0),
        width=width, color=COHORT_COLORS["2016_to_2016"], label="2016 baseline",
    )
    ax.bar(
        x + width / 2, first_per_go[cohort_cols["2024-added"]].astype(float).fillna(0),
        width=width, color=COHORT_COLORS["2024_to_2016"], label="2024-added",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([_go_label(g, bp_names) for g in go_ids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gene count")
    ax.set_title("Gene counts per cohort per top GO term")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, out_dir, "cohort_gene_counts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True,
                        help="Directory with intermediate_sharing_by_metapath.csv (usually a b{B} subdir).")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir for figures. Defaults to <input-dir>/figures.")
    parser.add_argument("--top-go-json", default=None,
                        help="Optional JSON list of go_ids to restrict plots to (e.g. top_go_ids.json).")
    parser.add_argument("--top-n-metapaths", type=int, default=10,
                        help="Top N metapaths to show in the per-GO metapath panels (default: 10).")
    parser.add_argument("--repo-root", default=".",
                        help="Repository root for resolving data/nodes/Biological Process.tsv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_path = input_dir / "intermediate_sharing_by_metapath.csv"
    if not by_path.exists():
        print(f"Missing {by_path}")
        return
    by_metapath_df = pd.read_csv(by_path)

    # Optional filter to top GOs
    if args.top_go_json:
        with open(args.top_go_json) as fh:
            top_ids = [str(x) for x in json.load(fh)]
        before = by_metapath_df["go_id"].nunique()
        by_metapath_df = by_metapath_df[by_metapath_df["go_id"].astype(str).isin(top_ids)].copy()
        print(f"Restricted to {by_metapath_df['go_id'].nunique()} of {before} GO terms from {args.top_go_json}")
    if by_metapath_df.empty:
        print("Nothing to plot after filtering.")
        return

    bp_names = load_bp_name_lookup(Path(args.repo_root))

    print("Generating year-specific intermediate-sharing plots...")
    plot_cohort_gene_counts(by_metapath_df, out_dir, bp_names)
    print("  - cohort_gene_counts")
    plot_cohort_sharing_per_go(by_metapath_df, out_dir, bp_names)
    print("  - cohort_sharing_per_go")
    plot_cohort_jaccard_heatmap(by_metapath_df, out_dir, bp_names)
    print("  - cohort_jaccard_heatmap")
    plot_top_metapaths_per_go(by_metapath_df, out_dir, bp_names, top_n=args.top_n_metapaths)
    print("  - top_metapaths_per_go")
    plot_coverage_by_rank_cohort(by_metapath_df, out_dir, bp_names)
    print("  - coverage_by_rank_cohort")

    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
