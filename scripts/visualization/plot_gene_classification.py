#!/usr/bin/env python3
"""Plot-only sibling of scripts/data_prep/percent_change_and_filtering.py figures.

Reads pre-computed classification + filtered CSVs from output/intermediate/
and re-renders the two PDFs without re-running classification, percent-change,
or IQR filtering. The original data_prep script remains the source of truth
for the CSVs.

Outputs:
    <output-dir>/gene_classification_stable_vs_added.{pdf,jpeg}
    <output-dir>/filtered_dataset_stable_vs_added.{pdf,jpeg}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _stable_added_panel(df: pd.DataFrame, suptitle_left: str, suptitle_right: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].hist(df["n_stable"], bins=30, alpha=0.7, label="Stable genes", edgecolor="black")
    axes[0].hist(df["n_added"], bins=30, alpha=0.7, label="Added genes", edgecolor="black", color="orange")
    axes[0].set_xlabel("Number of Genes per GO Term", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title(suptitle_left, fontsize=12)
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].scatter(df["n_stable"], df["n_added"], alpha=0.4, s=30)
    axes[1].set_xlabel("Number of Stable Genes", fontsize=11)
    axes[1].set_ylabel("Number of Added Genes", fontsize=11)
    axes[1].set_title(suptitle_right, fontsize=12)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def _save_dual(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.jpeg", dpi=300, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        default="output/intermediate",
        help="Directory holding go_gene_classification_summary.csv and all_GO_positive_growth.csv (default: output/intermediate).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the gene-classification PDFs/JPEGs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classification_path = input_dir / "go_gene_classification_summary.csv"
    filtered_path = input_dir / "all_GO_positive_growth.csv"
    for p in (classification_path, filtered_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input {p} is missing. Run scripts/data_prep/percent_change_and_filtering.py first."
            )

    classification_df = pd.read_csv(classification_path)
    filtered_df = pd.read_csv(filtered_path)

    fig1 = _stable_added_panel(
        classification_df,
        suptitle_left="Distribution: Stable vs Added Genes",
        suptitle_right="Stable vs Added Genes per GO Term",
    )
    _save_dual(fig1, output_dir, "gene_classification_stable_vs_added")
    plt.close(fig1)

    fig2 = _stable_added_panel(
        filtered_df,
        suptitle_left="Filtered Dataset: Stable vs Added Genes",
        suptitle_right="Filtered Dataset: Stable vs Added per GO Term",
    )
    _save_dual(fig2, output_dir, "filtered_dataset_stable_vs_added")
    plt.close(fig2)

    print(f"Saved {output_dir / 'gene_classification_stable_vs_added.pdf'}")
    print(f"Saved {output_dir / 'filtered_dataset_stable_vs_added.pdf'}")


if __name__ == "__main__":
    main()
