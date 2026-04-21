#!/usr/bin/env python3
"""Build go_top_genes.csv for year analysis.

Writes a per-(GO, gene) table that year_intermediate_sharing.py reads to
enumerate path instances. The table is the union of real-filtered 2016 and
real-filtered 2024-added GO-gene pairs (both positive-growth filtered).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()


DEFAULT_REAL_2016 = "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv"
DEFAULT_REAL_2024 = "output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-2016", default=DEFAULT_REAL_2016)
    p.add_argument("--real-2024", default=DEFAULT_REAL_2024)
    p.add_argument("--output-dir", required=True, help="Year experiment output dir; go_top_genes.csv is written here.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    real_2016 = pd.read_csv(REPO_ROOT / args.real_2016)[["go_id", "entrez_gene_id"]]
    real_2024 = pd.read_csv(REPO_ROOT / args.real_2024)[["go_id", "entrez_gene_id"]]

    combined = (
        pd.concat([real_2016, real_2024], ignore_index=True)
        .drop_duplicates()
        .rename(columns={"entrez_gene_id": "gene_identifier"})
        .astype({"gene_identifier": int})
        .sort_values(["go_id", "gene_identifier"])
        .reset_index(drop=True)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "go_top_genes.csv"
    combined.to_csv(output_path, index=False)
    print(f"Saved {output_path}: {len(combined)} (GO, gene) rows across {combined['go_id'].nunique()} GO terms")


if __name__ == "__main__":
    main()
