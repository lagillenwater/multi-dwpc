#!/usr/bin/env python3
"""Generate gene-centric connectivity table for LV or Year analysis.

Creates a table with one row per (gene, metapath) showing:
- Gene identifiers and names
- Metapath and its effect size
- Top intermediate nodes for this gene
- DWPC value

Usage:
    python scripts/generate_gene_table.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --lv-output-dirs output/lv_experiment \
        --output-dir output/lv_consumable

    # With specific B value
    python scripts/generate_gene_table.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --b 10 \
        --output-dir output/lv_consumable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

from src.intermediate_sharing import load_dwpc_from_numpy  # noqa: E402


def load_gene_names(repo_root: Path) -> dict[int, str]:
    """Load gene ID to gene symbol mapping."""
    gene_file = repo_root / "data" / "nodes" / "Gene.tsv"
    if not gene_file.exists():
        return {}

    df = pd.read_csv(gene_file, sep="\t")
    # identifier is Entrez Gene ID, name is gene symbol
    return dict(zip(df["identifier"].astype(int), df["name"]))


def load_dwpc_values(
    lv_output_dirs: list[Path],
    analysis_type: str,
) -> dict[tuple, float]:
    """Load DWPC values from numpy arrays. Delegates to shared loader."""
    return load_dwpc_from_numpy(lv_output_dirs, analysis_type=analysis_type)


def load_top_genes(
    lv_output_dirs: list[Path],
    analysis_type: str,
) -> pd.DataFrame:
    """Load top genes from LV/Year output directories."""
    frames = []

    for lv_output_dir in lv_output_dirs:
        if analysis_type == "lv":
            top_genes_path = lv_output_dir / "lv_top_genes.csv"
        else:
            top_genes_path = lv_output_dir / "go_top_genes.csv"

        if top_genes_path.exists():
            frames.append(pd.read_csv(top_genes_path))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates()


def generate_gene_table(
    input_dir: Path,
    lv_output_dirs: list[Path],
    analysis_type: str,
    b_value: int | None = None,
    gene_names: dict[int, str] | None = None,
    dwpc_lookup: dict[tuple, float] | None = None,
) -> pd.DataFrame:
    """Generate gene-centric connectivity table.

    Args:
        input_dir: Directory with intermediate sharing results
        lv_output_dirs: Directories with LV/Year experiment outputs
        analysis_type: "lv" or "year"
        b_value: B value subdirectory to use
        gene_names: Gene ID to name mapping
        dwpc_lookup: DWPC value lookup

    Returns:
        DataFrame with gene-level connectivity information
    """
    # Load intermediate sharing data
    if b_value is not None:
        data_dir = input_dir / f"b{b_value}"
    else:
        data_dir = input_dir

    by_metapath_path = data_dir / "intermediate_sharing_by_metapath.csv"
    top_int_path = data_dir / "top_intermediates_by_metapath.csv"

    if not by_metapath_path.exists():
        raise FileNotFoundError(f"Not found: {by_metapath_path}")

    by_metapath_df = pd.read_csv(by_metapath_path)

    # Load top intermediates
    if top_int_path.exists():
        top_int_df = pd.read_csv(top_int_path)
    else:
        top_int_df = pd.DataFrame()

    # Load top genes to get gene list per gene set
    top_genes_df = load_top_genes(lv_output_dirs, analysis_type)

    if analysis_type == "lv":
        id_col = "lv_id"
        gene_id_col = "gene_identifier"
    else:
        id_col = "go_id"
        gene_id_col = "gene_identifier"

    gene_table_rows = []

    # Process each gene set
    for gene_set_id in by_metapath_df[id_col].unique():
        # Get metapaths for this gene set
        mp_df = by_metapath_df[by_metapath_df[id_col] == gene_set_id]

        # Get genes for this gene set
        if not top_genes_df.empty and id_col in top_genes_df.columns:
            gene_ids = top_genes_df[
                top_genes_df[id_col] == gene_set_id
            ][gene_id_col].astype(int).tolist()
        else:
            gene_ids = []

        # Get target info
        target_name = mp_df["target_name"].iloc[0] if "target_name" in mp_df.columns else None
        target_id = mp_df["target_id"].iloc[0] if "target_id" in mp_df.columns else None

        # Process each metapath
        for _, mp_row in mp_df.iterrows():
            metapath = mp_row["metapath"]
            mp_rank = mp_row.get("metapath_rank", None)
            effect_size = mp_row.get("permutation_z", None)

            # Get top intermediates for this metapath
            if not top_int_df.empty:
                mp_int_df = top_int_df[
                    (top_int_df[id_col] == gene_set_id) &
                    (top_int_df["metapath"] == metapath)
                ].sort_values("intermediate_rank")
            else:
                mp_int_df = pd.DataFrame()

            # Process each gene
            for gene_id in gene_ids:
                # Get DWPC for this gene
                dwpc = None
                if dwpc_lookup:
                    dwpc = dwpc_lookup.get((gene_set_id, metapath, gene_id))

                # Skip genes without DWPC (not in top percentile)
                if dwpc is None or dwpc <= 0:
                    continue

                # Get gene name
                gene_name = gene_names.get(gene_id) if gene_names else None

                # Find top intermediate for this gene
                # Note: top_int_df has aggregate info, not per-gene
                # For now, use the global top intermediate for this metapath
                top_int_id = None
                top_int_name = None
                if not mp_int_df.empty:
                    top_row = mp_int_df.iloc[0]
                    top_int_id = top_row.get("intermediate_id")
                    top_int_name = top_row.get("intermediate_name")

                gene_table_rows.append({
                    "gene_set_id": gene_set_id,
                    "target_id": target_id,
                    "target_name": target_name,
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "metapath": metapath,
                    "metapath_rank": mp_rank,
                    "permutation_z": effect_size,
                    "dwpc": dwpc,
                    "top_intermediate_id": top_int_id,
                    "top_intermediate_name": top_int_name,
                })

    gene_table = pd.DataFrame(gene_table_rows)

    # Sort by gene set, metapath rank, then DWPC descending
    if not gene_table.empty:
        sort_cols = ["gene_set_id"]
        if "metapath_rank" in gene_table.columns:
            sort_cols.append("metapath_rank")
        sort_cols.append("dwpc")

        gene_table = gene_table.sort_values(
            sort_cols,
            ascending=[True] * (len(sort_cols) - 1) + [False]
        ).reset_index(drop=True)

    return gene_table


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
        "--lv-output-dirs",
        nargs="+",
        default=[],
        help="Paths to LV/Year experiment output directories",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for gene connectivity table",
    )
    parser.add_argument(
        "--b",
        type=int,
        help="B value subdirectory to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lv_output_dirs = [Path(d) for d in args.lv_output_dirs]

    # Load gene names
    print("Loading gene name mappings...")
    gene_names = load_gene_names(REPO_ROOT)
    print(f"Loaded {len(gene_names)} gene names")

    # Load DWPC values if lv_output_dirs provided
    dwpc_lookup = {}
    if lv_output_dirs:
        print("Loading DWPC values...")
        dwpc_lookup = load_dwpc_values(lv_output_dirs, args.analysis_type)
        print(f"Loaded {len(dwpc_lookup)} DWPC entries")

    # Generate gene table
    print(f"\nGenerating gene connectivity table...")
    gene_table = generate_gene_table(
        input_dir=input_dir,
        lv_output_dirs=lv_output_dirs,
        analysis_type=args.analysis_type,
        b_value=args.b,
        gene_names=gene_names,
        dwpc_lookup=dwpc_lookup,
    )

    # Save
    output_path = output_dir / "gene_connectivity_table.csv"
    gene_table.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Summary stats
    print(f"\n{'='*60}")
    print("Gene Connectivity Table Summary")
    print("=" * 60)
    print(f"Total rows: {len(gene_table)}")
    if not gene_table.empty:
        print(f"Gene sets: {gene_table['gene_set_id'].nunique()}")
        print(f"Unique genes: {gene_table['gene_id'].nunique()}")
        print(f"Unique metapaths: {gene_table['metapath'].nunique()}")
        if "dwpc" in gene_table.columns:
            print(f"DWPC range: {gene_table['dwpc'].min():.4f} - {gene_table['dwpc'].max():.4f}")


if __name__ == "__main__":
    main()
