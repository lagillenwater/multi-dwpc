#!/usr/bin/env python3
"""Generate metapath subgraph visualizations.

Creates bipartite graph visualizations showing:
- Genes (left) -> Intermediates (center) -> Target (right)
- Edge width proportional to connection strength
- Node color by type

Usage:
    python scripts/plot_metapath_subgraphs.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --gene-table output/lv_consumable/gene_connectivity_table.csv \
        --output-dir output/lv_consumable/subgraphs

    # For specific gene set and metapath
    python scripts/plot_metapath_subgraphs.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --gene-set-id LV246 --metapath GdDlA \
        --output-dir output/lv_consumable/subgraphs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

# Node type colors
NODE_COLORS = {
    "G": "#4CAF50",  # Gene - green
    "A": "#2196F3",  # Anatomy - blue
    "D": "#F44336",  # Disease - red
    "C": "#9C27B0",  # Compound - purple
    "BP": "#FF9800",  # Biological Process - orange
    "CC": "#00BCD4",  # Cellular Component - cyan
    "MF": "#FFEB3B",  # Molecular Function - yellow
    "PW": "#795548",  # Pathway - brown
    "PC": "#E91E63",  # Pharmacologic Class - pink
    "SE": "#607D8B",  # Side Effect - gray
    "S": "#9E9E9E",  # Symptom - light gray
}

DEFAULT_COLOR = "#BDBDBD"


def load_gene_names(repo_root: Path) -> dict[int, str]:
    """Load gene ID to gene symbol mapping."""
    gene_file = repo_root / "data" / "nodes" / "Gene.tsv"
    if not gene_file.exists():
        return {}
    df = pd.read_csv(gene_file, sep="\t")
    return dict(zip(df["identifier"].astype(int), df["name"]))


def parse_intermediate_id(int_id: str) -> tuple[str, str]:
    """Parse intermediate ID like 'A:UBERON:0001234' into (type, id)."""
    if ":" not in int_id:
        return ("?", int_id)
    parts = int_id.split(":", 1)
    return (parts[0], parts[1])


def plot_metapath_subgraph(
    gene_set_id: str,
    target_name: str,
    metapath: str,
    genes: list[dict],
    intermediates: list[dict],
    output_path: Path,
    max_genes: int = 20,
    max_intermediates: int = 15,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """Plot a bipartite subgraph for a metapath.

    Args:
        gene_set_id: Gene set identifier (LV or GO)
        target_name: Target phenotype/anatomy name
        metapath: Metapath string
        genes: List of dicts with gene_id, gene_name, dwpc
        intermediates: List of dicts with intermediate_id, intermediate_name, n_genes_using
        output_path: Path to save the figure
        max_genes: Maximum genes to show
        max_intermediates: Maximum intermediates to show
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Limit nodes for readability
    genes = genes[:max_genes]
    intermediates = intermediates[:max_intermediates]

    n_genes = len(genes)
    n_ints = len(intermediates)

    if n_genes == 0 or n_ints == 0:
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Layout positions
    # Genes on left (x=0.1), intermediates in center (x=0.5), target on right (x=0.9)
    gene_y = np.linspace(0.9, 0.1, n_genes) if n_genes > 1 else [0.5]
    int_y = np.linspace(0.9, 0.1, n_ints) if n_ints > 1 else [0.5]

    gene_x = 0.15
    int_x = 0.5
    target_x = 0.85

    # Normalize DWPC values for edge widths
    dwpc_values = [g.get("dwpc", 1) for g in genes]
    max_dwpc = max(dwpc_values) if dwpc_values else 1
    min_dwpc = min(dwpc_values) if dwpc_values else 0

    # Normalize intermediate usage for edge widths
    usage_values = [i.get("n_genes_using", 1) for i in intermediates]
    max_usage = max(usage_values) if usage_values else 1

    # Draw edges from genes to intermediates (simplified: connect all)
    # In reality, we'd need per-gene intermediate data
    for i, gene in enumerate(genes):
        gy = gene_y[i]
        dwpc = gene.get("dwpc", 1)

        # Normalize edge width
        if max_dwpc > min_dwpc:
            edge_width = 0.5 + 2.5 * (dwpc - min_dwpc) / (max_dwpc - min_dwpc)
        else:
            edge_width = 1.5

        # Connect to top intermediates (simplified visualization)
        # Use alpha based on rank
        for j, interm in enumerate(intermediates[:5]):  # Connect to top 5
            iy = int_y[j]
            alpha = 0.3 - 0.05 * j  # Fade with rank
            ax.plot(
                [gene_x, int_x], [gy, iy],
                color="#90A4AE", alpha=max(0.1, alpha),
                linewidth=edge_width * 0.5, zorder=1
            )

    # Draw edges from intermediates to target
    for j, interm in enumerate(intermediates):
        iy = int_y[j]
        usage = interm.get("n_genes_using", 1)

        # Edge width based on usage
        edge_width = 0.5 + 3.0 * usage / max_usage

        ax.plot(
            [int_x, target_x], [iy, 0.5],
            color="#78909C", alpha=0.5,
            linewidth=edge_width, zorder=1
        )

    # Draw gene nodes
    for i, gene in enumerate(genes):
        gy = gene_y[i]
        gene_name = gene.get("gene_name") or str(gene.get("gene_id", "?"))
        dwpc = gene.get("dwpc", 0)

        # Node size based on DWPC
        if max_dwpc > min_dwpc:
            node_size = 200 + 400 * (dwpc - min_dwpc) / (max_dwpc - min_dwpc)
        else:
            node_size = 400

        ax.scatter(gene_x, gy, s=node_size, c=NODE_COLORS["G"],
                   edgecolors="white", linewidths=1.5, zorder=3)

        # Label
        label = gene_name[:12] if len(gene_name) > 12 else gene_name
        ax.text(gene_x - 0.02, gy, label, ha="right", va="center",
                fontsize=8, fontweight="medium")

    # Draw intermediate nodes
    for j, interm in enumerate(intermediates):
        iy = int_y[j]
        int_id = interm.get("intermediate_id", "")
        int_name = interm.get("intermediate_name") or int_id
        usage = interm.get("n_genes_using", 1)
        pct = interm.get("pct_genes_using", 0)

        # Parse node type for color
        node_type, _ = parse_intermediate_id(int_id)
        color = NODE_COLORS.get(node_type, DEFAULT_COLOR)

        # Node size based on usage
        node_size = 200 + 600 * usage / max_usage

        ax.scatter(int_x, iy, s=node_size, c=color,
                   edgecolors="white", linewidths=1.5, zorder=3)

        # Label with percentage
        label = int_name[:20] if len(int_name) > 20 else int_name
        ax.text(int_x + 0.02, iy, f"{label} ({pct:.0f}%)", ha="left", va="center",
                fontsize=8)

    # Draw target node
    ax.scatter(target_x, 0.5, s=800, c=NODE_COLORS.get("D", DEFAULT_COLOR),
               edgecolors="white", linewidths=2, zorder=3, marker="s")
    ax.text(target_x, 0.5 - 0.08, target_name[:25], ha="center", va="top",
            fontsize=10, fontweight="bold")

    # Column labels
    ax.text(gene_x, 0.98, "Genes", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(int_x, 0.98, "Intermediates", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(target_x, 0.98, "Target", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

    # Title
    ax.set_title(f"{gene_set_id}: {metapath}\n{target_name}",
                 fontsize=14, fontweight="bold", pad=20)

    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_subgraphs(
    input_dir: Path,
    output_dir: Path,
    analysis_type: str,
    gene_table_path: Path | None = None,
    b_value: int | None = None,
    gene_set_filter: str | None = None,
    metapath_filter: str | None = None,
    top_k_metapaths: int = 3,
) -> None:
    """Generate subgraph visualizations for all or filtered gene sets/metapaths.

    Args:
        input_dir: Directory with intermediate sharing results
        output_dir: Output directory for subgraph images
        analysis_type: "lv" or "year"
        gene_table_path: Path to gene connectivity table
        b_value: B value subdirectory to use
        gene_set_filter: Filter to specific gene set ID
        metapath_filter: Filter to specific metapath
        top_k_metapaths: Number of top metapaths per gene set to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
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
        print("Warning: No top_intermediates_by_metapath.csv found")
        top_int_df = pd.DataFrame()

    # Load gene table if provided
    if gene_table_path and gene_table_path.exists():
        gene_table = pd.read_csv(gene_table_path)
    else:
        gene_table = pd.DataFrame()

    # Load gene names
    gene_names = load_gene_names(REPO_ROOT)

    # Determine ID column
    id_col = "lv_id" if analysis_type == "lv" else "go_id"

    # Apply filters
    if gene_set_filter:
        by_metapath_df = by_metapath_df[by_metapath_df[id_col] == gene_set_filter]

    if metapath_filter:
        by_metapath_df = by_metapath_df[by_metapath_df["metapath"] == metapath_filter]

    # Process each gene set
    for gene_set_id in by_metapath_df[id_col].unique():
        gs_df = by_metapath_df[by_metapath_df[id_col] == gene_set_id]

        # Get top metapaths by effect size
        top_mps = gs_df.nsmallest(top_k_metapaths, "metapath_rank") if "metapath_rank" in gs_df.columns else gs_df.head(top_k_metapaths)

        for _, mp_row in top_mps.iterrows():
            metapath = mp_row["metapath"]
            target_name = mp_row.get("target_name", "Unknown")

            # Get intermediates for this metapath
            if not top_int_df.empty:
                mp_ints = top_int_df[
                    (top_int_df[id_col] == gene_set_id) &
                    (top_int_df["metapath"] == metapath)
                ].to_dict("records")
            else:
                mp_ints = []

            # Get genes from gene table
            if not gene_table.empty and "gene_set_id" in gene_table.columns:
                mp_genes = gene_table[
                    (gene_table["gene_set_id"] == gene_set_id) &
                    (gene_table["metapath"] == metapath)
                ].to_dict("records")
            else:
                # Fallback: no gene-level data
                mp_genes = []

            # If no gene table, create placeholder genes
            if not mp_genes:
                n_genes = mp_row.get("n_genes_with_paths", 0)
                mp_genes = [{"gene_id": i, "gene_name": f"Gene_{i}", "dwpc": 1.0}
                            for i in range(min(int(n_genes), 20))]

            # Generate plot
            safe_mp = metapath.replace("/", "_").replace("\\", "_")
            output_path = output_dir / f"{gene_set_id}_{safe_mp}.png"

            print(f"Generating: {output_path.name}")
            plot_metapath_subgraph(
                gene_set_id=gene_set_id,
                target_name=target_name,
                metapath=metapath,
                genes=mp_genes,
                intermediates=mp_ints,
                output_path=output_path,
            )

            # Also save PDF version
            pdf_path = output_dir / f"{gene_set_id}_{safe_mp}.pdf"
            plot_metapath_subgraph(
                gene_set_id=gene_set_id,
                target_name=target_name,
                metapath=metapath,
                genes=mp_genes,
                intermediates=mp_ints,
                output_path=pdf_path,
            )


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
        "--gene-table",
        help="Path to gene_connectivity_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for subgraph images",
    )
    parser.add_argument(
        "--b",
        type=int,
        help="B value subdirectory to use",
    )
    parser.add_argument(
        "--gene-set-id",
        help="Filter to specific gene set ID",
    )
    parser.add_argument(
        "--metapath",
        help="Filter to specific metapath",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top metapaths per gene set to visualize (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    gene_table_path = Path(args.gene_table) if args.gene_table else None

    print(f"Generating metapath subgraph visualizations...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    generate_subgraphs(
        input_dir=input_dir,
        output_dir=output_dir,
        analysis_type=args.analysis_type,
        gene_table_path=gene_table_path,
        b_value=args.b,
        gene_set_filter=args.gene_set_id,
        metapath_filter=args.metapath,
        top_k_metapaths=args.top_k,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
