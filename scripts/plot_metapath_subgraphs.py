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


def metapath_display(metapath: str) -> str:
    """Hyphen-separate the metapath abbreviation between node types and edges.

    "GaDlAiD" -> "G-a-D-l-A-i-D"
    "GpBPpG"  -> "G-p-BP-p-G"
    """
    tokens: list[str] = []
    i = 0
    while i < len(metapath):
        ch = metapath[i]
        if ch.isupper():
            j = i
            while j < len(metapath) and metapath[j].isupper():
                j += 1
            tokens.append(metapath[i:j])
            i = j
        else:
            tokens.append(ch)
            i += 1
    return "-".join(tokens)


def load_gene_names(repo_root: Path) -> dict[int, str]:
    """Load gene ID to gene symbol mapping."""
    gene_file = repo_root / "data" / "nodes" / "Gene.tsv"
    if not gene_file.exists():
        return {}
    df = pd.read_csv(gene_file, sep="\t")
    return dict(zip(df["identifier"].astype(int), df["name"]))


# Maps folder-name (TSV stem in data/nodes/) -> qualified-id prefix used
# throughout the pipeline (matches NODE_COLORS keys).
_TSV_STEM_TO_TYPE = {
    "Gene": "G",
    "Anatomy": "A",
    "Disease": "D",
    "Compound": "C",
    "Biological Process": "BP",
    "Cellular Component": "CC",
    "Molecular Function": "MF",
    "Pathway": "PW",
    "Pharmacologic Class": "PC",
    "Side Effect": "SE",
    "Symptom": "S",
}


def load_all_node_names(repo_root: Path) -> dict[str, str]:
    """Return a dict mapping `"{type_abbrev}:{identifier}"` -> human name.

    Reads every `data/nodes/*.tsv` file (hetio convention: `identifier`, `name`
    columns) and aggregates into a single qualified-id lookup, matching the
    key format written by `lv_intermediate_sharing.py` / `gene_paths.csv`.
    """
    result: dict[str, str] = {}
    nodes_dir = repo_root / "data" / "nodes"
    if not nodes_dir.exists():
        return result
    for tsv in nodes_dir.glob("*.tsv"):
        abbrev = _TSV_STEM_TO_TYPE.get(tsv.stem)
        if abbrev is None:
            continue
        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception:
            continue
        if "identifier" not in df.columns or "name" not in df.columns:
            continue
        for ident, name in zip(df["identifier"].astype(str), df["name"].astype(str)):
            result[f"{abbrev}:{ident}"] = name
    return result


def _format_intermediate_label(qualified_id: str, node_name_lookup: dict[str, str]) -> str:
    """Human name if we have one, else the raw identifier (without type prefix)."""
    name = node_name_lookup.get(qualified_id)
    if name:
        return name
    # Fallback: strip the type prefix so it's at least not "A:UBERON:..."
    _, rest = parse_intermediate_id(qualified_id)
    return rest or qualified_id


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


def _per_hop_intermediate_buckets(
    intermediates: list[dict],
    hop_node_types: list[str],
    path_edges: pd.DataFrame | None,
    top_n_per_hop: int,
) -> list[list[dict]]:
    """Top-N intermediates per hop based on distinct source-gene coverage.

    For each intermediate position in the metapath, this counts how many
    distinct source genes have at least one path passing through each
    candidate intermediate at that position. Ranks and picks the top N per
    hop -- so every hop column populates with its own top list, even when
    the same node type appears at multiple hops (e.g., G-G-G-A).

    Each returned row carries:
        intermediate_id, intermediate_name, n_genes_using_at_hop,
        pct_genes_using_at_hop, hop_index (1-based).

    Names are pulled from the passed-in `intermediates` list when present;
    otherwise the caller's lookup will fill them in later.
    """
    n_hops = len(hop_node_types)
    buckets: list[list[dict]] = [[] for _ in hop_node_types]

    name_by_id: dict[str, str] = {}
    for interm in intermediates or []:
        iid = str(interm.get("intermediate_id", "") or "")
        name = interm.get("intermediate_name")
        if iid and name and not (isinstance(name, float) and pd.isna(name)):
            name_by_id[iid] = str(name)

    if path_edges is None or path_edges.empty:
        # Fallback to type-matching against the supplied intermediates list.
        for interm in intermediates or []:
            int_id = str(interm.get("intermediate_id", "") or "")
            node_type, _ = parse_intermediate_id(int_id)
            for hi, expected in enumerate(hop_node_types):
                if node_type == expected:
                    buckets[hi].append(interm)
                    break
        return buckets

    total_genes = int(path_edges["gene_id"].nunique()) if "gene_id" in path_edges.columns else 0
    for hop_idx in range(1, n_hops + 1):
        col = f"hop_{hop_idx}_id"
        if col not in path_edges.columns:
            continue
        sub = path_edges[[c for c in ("gene_id", col) if c in path_edges.columns]].dropna()
        if sub.empty:
            continue
        gene_counts = (
            sub.groupby(col)["gene_id"].nunique()
            .sort_values(ascending=False)
            .head(top_n_per_hop)
        )
        for int_id, n_genes in gene_counts.items():
            iid = str(int_id)
            pct = (float(n_genes) / total_genes * 100.0) if total_genes else 0.0
            buckets[hop_idx - 1].append(
                {
                    "intermediate_id": iid,
                    "intermediate_name": name_by_id.get(iid),
                    "n_genes_using": int(n_genes),
                    "pct_genes_using": pct,
                    "hop_index": hop_idx,
                }
            )

    return buckets


def _column_x_positions(n_cols: int) -> list[float]:
    """Return n_cols evenly-spaced x positions with margins."""
    if n_cols <= 1:
        return [0.5]
    return list(np.linspace(0.1, 0.9, n_cols))


def _y_coords(n: int) -> list[float]:
    """Return n evenly-spaced y positions from top (0.9) to bottom (0.1)."""
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    return list(np.linspace(0.9, 0.1, n))


def plot_metapath_subgraph(
    gene_set_id: str,
    target_name: str,
    metapath: str,
    genes: list[dict],
    intermediates: list[dict],
    output_stem: Path,
    path_edges: pd.DataFrame | None = None,
    node_name_lookup: dict[str, str] | None = None,
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

    node_name_lookup = node_name_lookup or {}
    genes = genes[:max_genes]
    hop_buckets: list[list[dict]] = _per_hop_intermediate_buckets(
        intermediates,
        hop_node_types,
        path_edges=path_edges,
        top_n_per_hop=max_intermediates_per_hop,
    )

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

    gene_y = _y_coords(len(genes))

    # Position map for edge drawing (keys: qualified node IDs like "G:123" / "A:UBERON:...").
    node_positions: dict[str, tuple[float, float]] = {}
    for i, gene in enumerate(genes):
        gid = gene.get("gene_id")
        if gid is not None:
            node_positions[f"{source_type}:{gid}"] = (gene_x, gene_y[i])

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
            # Prefer the upstream-provided name; fall back to the loaded
            # node-TSV name lookup; finally strip the type prefix off the id.
            int_name = interm.get("intermediate_name")
            if not int_name or (isinstance(int_name, float) and pd.isna(int_name)):
                int_name = _format_intermediate_label(int_id, node_name_lookup)
            usage = float(interm.get("n_genes_using", 1))
            pct = interm.get("pct_genes_using")
            node_size = 180 + 520 * (usage / max_usage if max_usage > 0 else 0)
            ax.scatter(hop_x, iy, s=node_size, c=color,
                       edgecolors="white", linewidths=1.2, zorder=3)
            label = str(int_name)[:22]
            pct_suffix = (
                f" ({pct:.0f}% genes)"
                if pct is not None and not pd.isna(pct) else ""
            )
            ax.text(hop_x + 0.012, iy, f"{label}{pct_suffix}",
                    ha="left", va="center", fontsize=7)
            if int_id:
                node_positions[int_id] = (hop_x, iy)
        last_hop_x = hop_x
        last_hop_ys = ys
        last_hop_items = items

    target_pos = (target_x, 0.5)

    # Real edges: aggregate (from_id, to_id) pairs across the path records
    # for every adjacent hop transition.
    edges_drawn = False
    if path_edges is not None and not path_edges.empty:
        n_hops = len(node_types)
        edge_counts: dict[tuple[int, str, str], int] = {}
        for row in path_edges.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(path_edges.columns, row))
            for i in range(n_hops - 1):
                from_id = row_dict.get(f"hop_{i}_id")
                to_id = row_dict.get(f"hop_{i + 1}_id")
                if from_id is None or (isinstance(from_id, float) and pd.isna(from_id)):
                    continue
                if to_id is None or (isinstance(to_id, float) and pd.isna(to_id)):
                    continue
                key = (i, str(from_id), str(to_id))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        if edge_counts:
            max_count = max(edge_counts.values())
            for (hop_idx, fid, tid), cnt in edge_counts.items():
                from_pos = node_positions.get(fid)
                if hop_idx + 1 == len(node_types) - 1:
                    to_pos = target_pos
                else:
                    to_pos = node_positions.get(tid)
                if from_pos is None or to_pos is None:
                    continue
                frac = cnt / max_count if max_count else 0.0
                ax.plot(
                    [from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                    color="#546E7A",
                    alpha=0.15 + 0.55 * frac,
                    linewidth=0.4 + 2.2 * frac,
                    zorder=1,
                )
                edges_drawn = True

    # Fallback when no path-level data: at least draw last-intermediate -> target
    if not edges_drawn and last_hop_x is not None and last_hop_items:
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

    # Column headers: "Source", "Hop 1", ..., and target-type full name
    header_y = 0.985
    ax.text(gene_x, header_y, "Source", ha="center", va="top",
            fontsize=11, fontweight="bold")
    for hop_idx, hop_x in enumerate(hop_xs):
        ax.text(hop_x, header_y, f"Hop {hop_idx + 1}", ha="center", va="top",
                fontsize=11, fontweight="bold")
    target_header = NODE_TYPE_FULL_NAMES.get(target_type, target_type)
    ax.text(target_x, header_y, target_header, ha="center", va="top",
            fontsize=11, fontweight="bold")

    # Two-line title: "LV57: hypothyroidism" / "G-a-D-l-A-i-D"
    ax.set_title(
        f"{gene_set_id}: {target_name}\n{metapath_display(metapath)}",
        fontsize=13, fontweight="bold", pad=12,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _save_figure(fig, output_stem)
    plt.close(fig)


def plot_combined_lv_subgraph(
    gene_set_id: str,
    target_name: str,
    metapaths: list[str],
    genes: list[dict],
    path_edges: pd.DataFrame,
    intermediate_names: dict[str, str],
    output_stem: Path,
    max_genes: int = 20,
    top_n_intermediates: int = 25,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Pooled per-LV subgraph across multiple metapaths.

    Columns are Source | Hop 1 | Hop 2 | ... | Hop max_hops | Target. Shorter
    metapaths jump from their last intermediate directly to Target, producing
    edges that span the unused intermediate columns. Each intermediate is
    placed in the hop column it traverses most often; if it serves more than
    one hop across metapaths, the majority wins and a footnote is appended
    to its label.
    """
    if path_edges is None or path_edges.empty:
        print(f"  [combined] {gene_set_id}: no path-level data, skipping")
        return

    # Determine hop counts per metapath (intermediate hops only) and max.
    hop_counts = {mp: max(len(parse_metapath_nodes(mp)) - 2, 0) for mp in metapaths}
    max_hops = max(hop_counts.values()) if hop_counts else 1
    if max_hops == 0:
        max_hops = 1
    n_cols = max_hops + 2  # source + hops + target

    # Rank intermediates by traversal count across all top-K metapaths.
    intermediate_rows: list[dict] = []
    edge_counts: dict[tuple[int, str, str], int] = {}  # (hop_idx_from, from_id, to_id) -> count
    hop_assignments: dict[str, dict[int, int]] = {}  # int_id -> {hop_idx: count}
    traversal_counts: dict[str, int] = {}
    intermediate_types: dict[str, str] = {}
    source_edges: dict[tuple[str, str], int] = {}  # (source_qualified, first_hop_id) -> count
    target_edges: dict[tuple[int, str], int] = {}  # (last_hop_idx, last_hop_id) -> count (last hop -> target)

    for row in path_edges.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(path_edges.columns, row))
        mp = str(row_dict.get("metapath", ""))
        mp_hop_count = hop_counts.get(mp, 0)  # intermediate hop count
        if mp_hop_count == 0:
            continue  # metapath has no intermediates
        # hop_0 = source gene, hop_1..hop_mp_hop_count = intermediates, hop_{mp_hop_count+1} = target
        source_id = row_dict.get("hop_0_id")
        target_id = row_dict.get(f"hop_{mp_hop_count + 1}_id")

        # Record per-intermediate traversals (one increment per path per intermediate).
        for int_idx in range(1, mp_hop_count + 1):
            int_id = row_dict.get(f"hop_{int_idx}_id")
            int_type = row_dict.get(f"hop_{int_idx}_type")
            if int_id is None or (isinstance(int_id, float) and pd.isna(int_id)):
                continue
            int_id = str(int_id)
            traversal_counts[int_id] = traversal_counts.get(int_id, 0) + 1
            if int_type and not (isinstance(int_type, float) and pd.isna(int_type)):
                intermediate_types.setdefault(int_id, str(int_type))
            hop_assignments.setdefault(int_id, {})
            hop_assignments[int_id][int_idx] = hop_assignments[int_id].get(int_idx, 0) + 1

        # Accumulate edges.
        # source -> first hop (column 1)
        first_int_id = row_dict.get("hop_1_id")
        if source_id is not None and first_int_id is not None:
            key = (str(source_id), str(first_int_id))
            source_edges[key] = source_edges.get(key, 0) + 1

        # Intermediate -> intermediate
        for int_idx in range(1, mp_hop_count):
            fid = row_dict.get(f"hop_{int_idx}_id")
            tid = row_dict.get(f"hop_{int_idx + 1}_id")
            if fid is None or tid is None:
                continue
            if (isinstance(fid, float) and pd.isna(fid)) or (isinstance(tid, float) and pd.isna(tid)):
                continue
            edge_counts[(int_idx, str(fid), str(tid))] = (
                edge_counts.get((int_idx, str(fid), str(tid)), 0) + 1
            )

        # last intermediate -> target (may skip hop columns)
        last_int_id = row_dict.get(f"hop_{mp_hop_count}_id")
        if last_int_id is not None and target_id is not None:
            target_edges[(mp_hop_count, str(last_int_id))] = (
                target_edges.get((mp_hop_count, str(last_int_id)), 0) + 1
            )

    if not traversal_counts:
        print(f"  [combined] {gene_set_id}: no intermediate traversals found")
        return

    # Top N intermediates overall
    ranked = sorted(traversal_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n_intermediates]
    kept_ids = {iid for iid, _ in ranked}

    # Assign each kept intermediate to its majority hop index
    int_hop: dict[str, int] = {}
    for iid, _ in ranked:
        hops = hop_assignments.get(iid, {})
        if not hops:
            continue
        int_hop[iid] = max(hops.items(), key=lambda kv: kv[1])[0]

    # Layout
    if figsize is None:
        figsize = (max(11.0, 2.6 * n_cols + 3.0), max(8.0, 0.42 * max_genes + 3.0))
    fig, ax = plt.subplots(figsize=figsize)

    x_positions = _column_x_positions(n_cols)
    source_x = x_positions[0]
    hop_xs = x_positions[1:-1]
    target_x = x_positions[-1]

    genes = genes[:max_genes]
    gene_y = _y_coords(len(genes)) if genes else []

    # Group kept intermediates by hop index for positioning
    hop_items: dict[int, list[str]] = {h: [] for h in range(1, max_hops + 1)}
    for iid, h in int_hop.items():
        if iid in kept_ids:
            hop_items.setdefault(h, []).append(iid)
    # Sort each hop's intermediates by traversal count desc
    for h in hop_items:
        hop_items[h].sort(key=lambda iid: traversal_counts.get(iid, 0), reverse=True)

    # Build position map for drawing edges
    node_positions: dict[str, tuple[float, float]] = {}
    for i, gene in enumerate(genes):
        gid = gene.get("gene_id")
        if gid is not None:
            node_positions[f"G:{gid}"] = (source_x, gene_y[i])
    hop_ys: dict[int, list[float]] = {}
    for hop_idx in range(1, max_hops + 1):
        items = hop_items.get(hop_idx, [])
        ys = _y_coords(len(items))
        hop_ys[hop_idx] = ys
        for rank, iid in enumerate(items):
            node_positions[iid] = (hop_xs[hop_idx - 1], ys[rank])

    target_pos = (target_x, 0.5)

    # Draw edges (lowest z-order)
    all_counts = (
        list(source_edges.values())
        + [c for (_, _, _), c in edge_counts.items()]
        + list(target_edges.values())
    )
    max_count = max(all_counts) if all_counts else 1

    def _edge_style(cnt: int) -> tuple[float, float]:
        frac = cnt / max_count if max_count else 0.0
        return (0.15 + 0.55 * frac, 0.4 + 2.2 * frac)  # alpha, width

    for (sid, fid), cnt in source_edges.items():
        # `sid` is already qualified (e.g., "G:123") as written by lv_intermediate_sharing.
        from_pos = node_positions.get(str(sid))
        to_pos = node_positions.get(fid)
        if from_pos is None or to_pos is None:
            continue
        a, lw = _edge_style(cnt)
        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                color="#546E7A", alpha=a, linewidth=lw, zorder=1)

    for (hop_idx, fid, tid), cnt in edge_counts.items():
        from_pos = node_positions.get(fid)
        to_pos = node_positions.get(tid)
        if from_pos is None or to_pos is None:
            continue
        a, lw = _edge_style(cnt)
        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                color="#546E7A", alpha=a, linewidth=lw, zorder=1)

    for (hop_idx, fid), cnt in target_edges.items():
        from_pos = node_positions.get(fid)
        if from_pos is None:
            continue
        a, lw = _edge_style(cnt)
        ax.plot([from_pos[0], target_pos[0]], [from_pos[1], target_pos[1]],
                color="#546E7A", alpha=a, linewidth=lw, zorder=1)

    # Draw source genes
    gene_color = NODE_COLORS.get("G", DEFAULT_COLOR)
    for i, gene in enumerate(genes):
        gy = gene_y[i]
        gene_name = gene.get("gene_name") or str(gene.get("gene_id", "?"))
        ax.scatter(source_x, gy, s=320, c=gene_color,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(source_x - 0.015, gy, str(gene_name)[:12],
                ha="right", va="center", fontsize=8, fontweight="medium")

    # Draw intermediates, color by their own node type
    for hop_idx in range(1, max_hops + 1):
        items = hop_items.get(hop_idx, [])
        ys = hop_ys.get(hop_idx, [])
        for rank, iid in enumerate(items):
            cnt = traversal_counts.get(iid, 1)
            node_type = intermediate_types.get(iid, "?")
            color = NODE_COLORS.get(node_type, DEFAULT_COLOR)
            node_size = 180 + 520 * (cnt / max_count if max_count > 0 else 0)
            ax.scatter(hop_xs[hop_idx - 1], ys[rank], s=node_size, c=color,
                       edgecolors="white", linewidths=1.2, zorder=3)
            name = intermediate_names.get(iid)
            if not name:
                name = _format_intermediate_label(iid, intermediate_names)
            label = str(name)[:22]
            ax.text(
                hop_xs[hop_idx - 1] + 0.012, ys[rank],
                f"{label} ({cnt} traversals)",
                ha="left", va="center", fontsize=7,
            )

    # Target
    ax.scatter(target_x, 0.5, s=900, c=NODE_COLORS.get("D", DEFAULT_COLOR),
               edgecolors="white", linewidths=2, zorder=3, marker="s")
    ax.text(target_x, 0.5 - 0.07, str(target_name)[:30], ha="center", va="top",
            fontsize=10, fontweight="bold")

    # Headers
    header_y = 0.985
    ax.text(source_x, header_y, "Source", ha="center", va="top",
            fontsize=11, fontweight="bold")
    for hop_idx in range(1, max_hops + 1):
        ax.text(hop_xs[hop_idx - 1], header_y, f"Hop {hop_idx}",
                ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(target_x, header_y, NODE_TYPE_FULL_NAMES.get("D", "Target"),
            ha="center", va="top", fontsize=11, fontweight="bold")

    # Title
    mp_display = ", ".join(metapath_display(m) for m in metapaths)
    ax.set_title(
        f"{gene_set_id}: {target_name}  (combined across {len(metapaths)} metapath"
        f"{'s' if len(metapaths) != 1 else ''})\n{mp_display}",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Color legend for the node types present
    types_present = sorted(set(intermediate_types.get(iid, "?") for iid in kept_ids))
    legend_handles = []
    for t in types_present:
        full = NODE_TYPE_FULL_NAMES.get(t, t)
        legend_handles.append(
            plt.Line2D([], [], marker="o", color="w",
                       markerfacecolor=NODE_COLORS.get(t, DEFAULT_COLOR),
                       markersize=10, label=full)
        )
    if legend_handles:
        ax.legend(handles=legend_handles, title="Intermediate type",
                  loc="lower right", fontsize=8, framealpha=0.9)

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
    gene_paths_path = data_dir / "gene_paths.csv"

    if not by_metapath_path.exists():
        raise FileNotFoundError(f"Not found: {by_metapath_path}")

    by_metapath_df = pd.read_csv(by_metapath_path)

    top_int_df = pd.read_csv(top_int_path) if top_int_path.exists() else pd.DataFrame()
    if not top_int_path.exists():
        print("Warning: No top_intermediates_by_metapath.csv found")

    gene_paths_df = pd.read_csv(gene_paths_path) if gene_paths_path.exists() else pd.DataFrame()
    if gene_paths_df.empty:
        print(f"Warning: No gene_paths.csv at {gene_paths_path} -- edges will be minimal")

    gene_table = pd.read_csv(gene_table_path) if (gene_table_path and gene_table_path.exists()) else pd.DataFrame()
    gene_names = load_gene_names(REPO_ROOT)
    # Qualified-id -> human-name lookup spanning every node type.
    node_name_lookup = load_all_node_names(REPO_ROOT)

    id_col = "lv_id" if analysis_type == "lv" else "go_id"

    if gene_set_filter:
        by_metapath_df = by_metapath_df[by_metapath_df[id_col].astype(str) == str(gene_set_filter)]

    if metapath_filter:
        by_metapath_df = by_metapath_df[by_metapath_df["metapath"] == metapath_filter]

    # Combined lookup: upstream-provided names win, node-TSV names backfill.
    int_name_lookup: dict[str, str] = dict(node_name_lookup)
    if not top_int_df.empty and "intermediate_id" in top_int_df.columns:
        for iid, name in zip(
            top_int_df["intermediate_id"].astype(str),
            top_int_df.get("intermediate_name", pd.Series([None] * len(top_int_df))),
        ):
            if iid and (name is not None and not pd.isna(name)):
                int_name_lookup[iid] = str(name)

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

            if not gene_paths_df.empty:
                mp_paths = gene_paths_df[
                    (gene_paths_df[id_col].astype(str) == gene_set_id)
                    & (gene_paths_df["metapath"] == metapath)
                ]
            else:
                mp_paths = pd.DataFrame()

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
                path_edges=mp_paths if not mp_paths.empty else None,
                node_name_lookup=int_name_lookup,
            )

        # One combined per-LV plot pooling top-K metapaths. Uses the same
        # `gene_paths.csv` path-level data already in `gene_paths_df`.
        if not gene_paths_df.empty:
            top_metapaths = top_mps["metapath"].astype(str).tolist()
            combined_paths = gene_paths_df[
                (gene_paths_df[id_col].astype(str) == gene_set_id)
                & (gene_paths_df["metapath"].astype(str).isin(top_metapaths))
            ]
            if not combined_paths.empty:
                combined_target = str(top_mps.iloc[0].get("target_name", "Unknown"))
                # Reuse gene list from first metapath; fall back to placeholder.
                if not gene_table.empty and "gene_set_id" in gene_table.columns:
                    combined_genes = gene_table[
                        gene_table["gene_set_id"].astype(str) == gene_set_id
                    ].drop_duplicates(subset=["gene_id"]).to_dict("records")
                else:
                    combined_gene_ids = combined_paths["gene_id"].dropna().unique().tolist()
                    combined_genes = [
                        {
                            "gene_id": int(g),
                            "gene_name": gene_names.get(int(g), str(g)),
                            "dwpc": 1.0,
                        }
                        for g in combined_gene_ids
                    ]
                combined_stem = output_dir / f"{gene_set_id}_combined"
                print(f"Generating: {combined_stem.name}")
                plot_combined_lv_subgraph(
                    gene_set_id=gene_set_id,
                    target_name=combined_target,
                    metapaths=top_metapaths,
                    genes=combined_genes,
                    path_edges=combined_paths,
                    intermediate_names=int_name_lookup,
                    output_stem=combined_stem,
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
