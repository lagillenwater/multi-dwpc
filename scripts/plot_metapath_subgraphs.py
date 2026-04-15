#!/usr/bin/env python3
"""Generate multi-hop metapath subgraph visualizations.

Parses the metapath abbreviation to lay out one column per node position
(source genes -> hop-1 intermediates -> ... -> target), colors nodes by
node type, and sizes intermediate nodes by how many genes use them.

Edges:
  - Intermediates in the LAST intermediate column connect to the target.
    (Drawn with line width proportional to n_genes_using.)
  - No gene -> intermediate fanout: we don't persist per-(gene, intermediate)
    path data, so drawing edges there would be misleading. Columns alone
    communicate "these genes connect to these intermediates".
  - Between intermediate columns (for metapaths with 3+ edges): no edges
    drawn, for the same reason.

Usage:
    python scripts/plot_metapath_subgraphs.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --gene-table output/lv_consumable/gene_connectivity_table.csv \
        --output-dir output/lv_consumable/subgraphs

    python scripts/plot_metapath_subgraphs.py --analysis-type lv \
        --input-dir output/lv_intermediate_sharing \
        --gene-set-id LV246 --metapath GaDlAiD \
        --output-dir output/lv_consumable/subgraphs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

# Node type colors (single- or multi-letter hetio abbrevs)
NODE_COLORS = {
    "G": "#4CAF50",    # Gene - green
    "A": "#2196F3",    # Anatomy - blue
    "D": "#F44336",    # Disease - red
    "C": "#9C27B0",    # Compound - purple
    "BP": "#FF9800",   # Biological Process - orange
    "CC": "#00BCD4",   # Cellular Component - cyan
    "MF": "#FFEB3B",   # Molecular Function - yellow
    "PW": "#795548",   # Pathway - brown
    "PC": "#E91E63",   # Pharmacologic Class - pink
    "SE": "#607D8B",   # Side Effect - gray
    "S": "#9E9E9E",    # Symptom - light gray
}

DEFAULT_COLOR = "#BDBDBD"

NODE_TYPE_FULL_NAMES = {
    "G": "Gene",
    "A": "Anatomy",
    "D": "Disease",
    "C": "Compound",
    "BP": "Biological Process",
    "CC": "Cellular Component",
    "MF": "Molecular Function",
    "PW": "Pathway",
    "PC": "Pharmacologic Class",
    "SE": "Side Effect",
    "S": "Symptom",
}


def parse_metapath_nodes(metapath: str) -> list[str]:
    """Split a metapath abbreviation into its sequence of node-type tokens.

    Uppercase runs are node types; single lowercase characters between them
    are edge types. Examples:
        "GaDlAiD"  -> ["G", "D", "A", "D"]
        "GpBPpGpBP" -> ["G", "BP", "G", "BP"]
    """
    nodes: list[str] = []
    i = 0
    while i < len(metapath):
        ch = metapath[i]
        if ch.isupper():
            j = i
            while j < len(metapath) and metapath[j].isupper():
                j += 1
            nodes.append(metapath[i:j])
            i = j
        else:
            i += 1
    return nodes


def load_gene_names(repo_root: Path) -> dict[int, str]:
    """Load gene ID to gene symbol mapping."""
    gene_file = repo_root / "data" / "nodes" / "Gene.tsv"
    if not gene_file.exists():
        return {}
    df = pd.read_csv(gene_file, sep="\t")
    return dict(zip(df["identifier"].astype(int), df["name"]))


def parse_intermediate_id(int_id: str) -> tuple[str, str]:
    """Parse intermediate ID like 'A:UBERON:0001234' into (type, id_rest)."""
    if ":" not in int_id:
        return ("?", int_id)
    parts = int_id.split(":", 1)
    return (parts[0], parts[1])


def _save_figure(fig, out_path_stem: Path) -> None:
    """Save figure as both .pdf and .png using a common stem."""
    out_path_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_path_stem.with_suffix(ext), bbox_inches="tight", dpi=200, facecolor="white")


def _bucket_intermediates_by_hop(
    intermediates: list[dict],
    hop_node_types: list[str],
) -> list[list[dict]]:
    """Group intermediate records into columns by their node-type prefix.

    Args:
        intermediates: list of dicts with "intermediate_id" like "A:UBERON:123".
        hop_node_types: node-type abbrevs for intermediate positions only,
            e.g. ["D", "A"] for a G-D-A-D metapath.

    Returns:
        list of buckets, one per intermediate hop, preserving input order.
    """
    buckets: list[list[dict]] = [[] for _ in hop_node_types]
    for interm in intermediates:
        int_id = interm.get("intermediate_id", "") or ""
        node_type, _ = parse_intermediate_id(str(int_id))
        for hi, expected in enumerate(hop_node_types):
            if node_type == expected:
                buckets[hi].append(interm)
                break
    return buckets


def _column_x_positions(n_cols: int) -> list[float]:
    """Return n_cols evenly-spaced x positions with margins."""
    if n_cols <= 1:
        return [0.5]
    return list(np.linspace(0.1, 0.9, n_cols))


def plot_metapath_subgraph(
    gene_set_id: str,
    target_name: str,
    metapath: str,
    genes: list[dict],
    intermediates: list[dict],
    output_stem: Path,
    max_genes: int = 20,
    max_intermediates_per_hop: int = 12,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot a multi-hop metapath subgraph.

    The number of columns comes from the metapath itself: one column per
    node type in the path (source, intermediates..., target). Intermediates
    are bucketed into hop columns by matching their node-type prefix to the
    metapath. Edges are only drawn between the final intermediate column and
    the target; gene->intermediate and intermediate->intermediate edges are
    omitted because per-gene path data is not persisted.

    Args:
        gene_set_id: LV or GO identifier used as the plot title.
        target_name: Target phenotype/anatomy name.
        metapath: Metapath abbreviation, e.g. "GaDlAiD".
        genes: List of dicts with gene_id, gene_name, dwpc.
        intermediates: List of dicts with intermediate_id, intermediate_name,
            n_genes_using, pct_genes_using.
        output_stem: Output path without extension; .pdf and .png will be saved.
    """
    node_types = parse_metapath_nodes(metapath)
    if len(node_types) < 2:
        # Fall back to a two-column layout if the metapath couldn't be parsed.
        node_types = ["G", "?"]

    source_type = node_types[0]
    target_type = node_types[-1]
    hop_node_types = node_types[1:-1]  # intermediate positions only
    n_cols = len(node_types)

    genes = genes[:max_genes]
    hop_buckets_raw = _bucket_intermediates_by_hop(intermediates, hop_node_types)
    hop_buckets: list[list[dict]] = [b[:max_intermediates_per_hop] for b in hop_buckets_raw]

    if figsize is None:
        figsize = (max(10.0, 2.8 * n_cols + 2.5), max(8.0, 0.42 * max_genes + 3.0))
    fig, ax = plt.subplots(figsize=figsize)

    if not genes and all(not b for b in hop_buckets):
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        _save_figure(fig, output_stem)
        plt.close(fig)
        return

    x_positions = _column_x_positions(n_cols)
    gene_x = x_positions[0]
    hop_xs = x_positions[1:-1]
    target_x = x_positions[-1]

    def _y_coords(n: int) -> list[float]:
        if n == 0:
            return []
        if n == 1:
            return [0.5]
        return list(np.linspace(0.9, 0.1, n))

    gene_y = _y_coords(len(genes))

    # Draw gene nodes (no edges out of this column)
    gene_color = NODE_COLORS.get(source_type, DEFAULT_COLOR)
    dwpc_values = [float(g.get("dwpc", 1.0)) for g in genes]
    max_dwpc = max(dwpc_values) if dwpc_values else 1.0
    min_dwpc = min(dwpc_values) if dwpc_values else 0.0

    for i, gene in enumerate(genes):
        gy = gene_y[i]
        gene_name = gene.get("gene_name") or str(gene.get("gene_id", "?"))
        dwpc = float(gene.get("dwpc", 1.0))
        if max_dwpc > min_dwpc:
            node_size = 160 + 340 * (dwpc - min_dwpc) / (max_dwpc - min_dwpc)
        else:
            node_size = 320
        ax.scatter(gene_x, gy, s=node_size, c=gene_color,
                   edgecolors="white", linewidths=1.2, zorder=3)
        label = gene_name[:12] if len(gene_name) > 12 else gene_name
        ax.text(gene_x - 0.015, gy, label, ha="right", va="center",
                fontsize=8, fontweight="medium")

    # Draw intermediate node columns
    last_hop_x: float | None = None
    last_hop_ys: list[float] = []
    last_hop_items: list[dict] = []
    for hop_idx, hop_x in enumerate(hop_xs):
        items = hop_buckets[hop_idx]
        if not items:
            ax.text(hop_x, 0.5, "(no data)", ha="center", va="center",
                    fontsize=9, color="#888888", style="italic")
            continue
        node_type = hop_node_types[hop_idx]
        color = NODE_COLORS.get(node_type, DEFAULT_COLOR)
        usages = [float(it.get("n_genes_using", 1)) for it in items]
        max_usage = max(usages) if usages else 1.0
        ys = _y_coords(len(items))
        for j, interm in enumerate(items):
            iy = ys[j]
            int_id = str(interm.get("intermediate_id", ""))
            int_name = interm.get("intermediate_name") or int_id
            usage = float(interm.get("n_genes_using", 1))
            pct = interm.get("pct_genes_using")
            node_size = 180 + 520 * (usage / max_usage if max_usage > 0 else 0)
            ax.scatter(hop_x, iy, s=node_size, c=color,
                       edgecolors="white", linewidths=1.2, zorder=3)
            label = str(int_name)[:22]
            pct_suffix = f" ({pct:.0f}%)" if pct is not None and not pd.isna(pct) else ""
            ax.text(hop_x + 0.012, iy, f"{label}{pct_suffix}",
                    ha="left", va="center", fontsize=7)
        last_hop_x = hop_x
        last_hop_ys = ys
        last_hop_items = items

    # Draw edges: last intermediate column -> target
    if last_hop_x is not None and last_hop_items:
        usages = [float(it.get("n_genes_using", 1)) for it in last_hop_items]
        max_usage = max(usages) if usages else 1.0
        for iy, interm in zip(last_hop_ys, last_hop_items):
            usage = float(interm.get("n_genes_using", 1))
            edge_width = 0.5 + 2.5 * (usage / max_usage if max_usage > 0 else 0)
            ax.plot(
                [last_hop_x, target_x], [iy, 0.5],
                color="#78909C", alpha=0.45,
                linewidth=edge_width, zorder=1,
            )

    # Draw target node
    target_color = NODE_COLORS.get(target_type, DEFAULT_COLOR)
    ax.scatter(target_x, 0.5, s=900, c=target_color,
               edgecolors="white", linewidths=2, zorder=3, marker="s")
    ax.text(target_x, 0.5 - 0.07, str(target_name)[:30], ha="center", va="top",
            fontsize=10, fontweight="bold")

    # Column headers (type label + position in path)
    header_y = 0.985
    def _col_label(node_type: str, position_label: str) -> str:
        full = NODE_TYPE_FULL_NAMES.get(node_type, node_type)
        return f"{position_label}\n({full})"

    ax.text(gene_x, header_y, _col_label(source_type, "Source genes"),
            ha="center", va="top", fontsize=11, fontweight="bold")
    for hop_idx, hop_x in enumerate(hop_xs):
        ax.text(
            hop_x, header_y,
            _col_label(hop_node_types[hop_idx], f"Hop {hop_idx + 1}"),
            ha="center", va="top", fontsize=11, fontweight="bold",
        )
    ax.text(target_x, header_y, _col_label(target_type, "Target"),
            ha="center", va="top", fontsize=11, fontweight="bold")

    # Title with metapath breakdown
    hop_arrow = " \u2192 ".join(node_types)
    ax.set_title(
        f"{gene_set_id}: {metapath}   [{hop_arrow}]\n{target_name}",
        fontsize=13, fontweight="bold", pad=12,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _save_figure(fig, output_stem)
    plt.close(fig)


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
    """Generate subgraph visualizations for all or filtered gene sets/metapaths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = input_dir / f"b{b_value}" if b_value is not None else input_dir

    by_metapath_path = data_dir / "intermediate_sharing_by_metapath.csv"
    top_int_path = data_dir / "top_intermediates_by_metapath.csv"

    if not by_metapath_path.exists():
        raise FileNotFoundError(f"Not found: {by_metapath_path}")

    by_metapath_df = pd.read_csv(by_metapath_path)

    top_int_df = pd.read_csv(top_int_path) if top_int_path.exists() else pd.DataFrame()
    if not top_int_path.exists():
        print("Warning: No top_intermediates_by_metapath.csv found")

    gene_table = pd.read_csv(gene_table_path) if (gene_table_path and gene_table_path.exists()) else pd.DataFrame()
    gene_names = load_gene_names(REPO_ROOT)

    id_col = "lv_id" if analysis_type == "lv" else "go_id"

    if gene_set_filter:
        by_metapath_df = by_metapath_df[by_metapath_df[id_col].astype(str) == str(gene_set_filter)]

    if metapath_filter:
        by_metapath_df = by_metapath_df[by_metapath_df["metapath"] == metapath_filter]

    for gene_set_id in by_metapath_df[id_col].astype(str).unique():
        gs_df = by_metapath_df[by_metapath_df[id_col].astype(str) == gene_set_id]
        top_mps = (
            gs_df.nsmallest(top_k_metapaths, "metapath_rank")
            if "metapath_rank" in gs_df.columns
            else gs_df.head(top_k_metapaths)
        )

        for _, mp_row in top_mps.iterrows():
            metapath = str(mp_row["metapath"])
            target_name = str(mp_row.get("target_name", "Unknown"))

            if not top_int_df.empty:
                mp_ints = top_int_df[
                    (top_int_df[id_col].astype(str) == gene_set_id)
                    & (top_int_df["metapath"] == metapath)
                ].to_dict("records")
            else:
                mp_ints = []

            if not gene_table.empty and "gene_set_id" in gene_table.columns:
                mp_genes = gene_table[
                    (gene_table["gene_set_id"].astype(str) == gene_set_id)
                    & (gene_table["metapath"] == metapath)
                ].to_dict("records")
            else:
                mp_genes = []

            if not mp_genes:
                n_genes = int(mp_row.get("n_genes_with_paths", 0) or 0)
                mp_genes = [
                    {"gene_id": i, "gene_name": gene_names.get(i, f"Gene_{i}"), "dwpc": 1.0}
                    for i in range(min(n_genes, 20))
                ]

            safe_mp = metapath.replace("/", "_").replace("\\", "_")
            output_stem = output_dir / f"{gene_set_id}_{safe_mp}"
            print(f"Generating: {output_stem.name}")
            plot_metapath_subgraph(
                gene_set_id=gene_set_id,
                target_name=target_name,
                metapath=metapath,
                genes=mp_genes,
                intermediates=mp_ints,
                output_stem=output_stem,
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
    parser.add_argument("--gene-table", help="Path to gene_connectivity_table.csv")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for subgraph images",
    )
    parser.add_argument("--b", type=int, help="B value subdirectory to use")
    parser.add_argument("--gene-set-id", help="Filter to specific gene set ID")
    parser.add_argument("--metapath", help="Filter to specific metapath")
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

    print("Generating metapath subgraph visualizations...")
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
