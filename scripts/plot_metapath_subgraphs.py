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
    # Don't slice yet -- the filter-by-surviving-paths step below needs to see
    # every candidate gene before we cap at max_genes.
    n_intermediate_hops = len(hop_node_types)

    # Step 1: per-hop top-N candidate intermediates (based on source-gene coverage).
    hop_buckets: list[list[dict]] = _per_hop_intermediate_buckets(
        intermediates,
        hop_node_types,
        path_edges=path_edges,
        top_n_per_hop=max_intermediates_per_hop,
    )
    allowed_per_hop: dict[int, set[str]] = {}
    for hi, bucket in enumerate(hop_buckets):
        allowed_per_hop[hi + 1] = {
            str(it.get("intermediate_id", "") or "") for it in bucket if it.get("intermediate_id")
        }

    # Step 2: keep only paths where EVERY intermediate is in its hop's allowed set.
    # This prevents split/ghost edges -- drawn paths are always end-to-end complete.
    surviving_rows: list[dict] = []
    if path_edges is not None and not path_edges.empty:
        for row in path_edges.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(path_edges.columns, row))
            ok = True
            for hi in range(1, n_intermediate_hops + 1):
                iid = row_dict.get(f"hop_{hi}_id")
                if iid is None or (isinstance(iid, float) and pd.isna(iid)):
                    ok = False
                    break
                if str(iid) not in allowed_per_hop.get(hi, set()):
                    ok = False
                    break
            if ok:
                surviving_rows.append(row_dict)

    # Step 3: derive visible sources + actually-used intermediates from surviving paths.
    visible_source_ids: set[str] = set()
    used_per_hop: dict[int, set[str]] = {i: set() for i in range(1, n_intermediate_hops + 1)}
    edge_counts: dict[tuple[int, str, str], int] = {}
    for row in surviving_rows:
        src = row.get("hop_0_id")
        if src is not None and not (isinstance(src, float) and pd.isna(src)):
            visible_source_ids.add(str(src))
        for hi in range(1, n_intermediate_hops + 1):
            iid = row.get(f"hop_{hi}_id")
            if iid is not None and not (isinstance(iid, float) and pd.isna(iid)):
                used_per_hop[hi].add(str(iid))
        for i in range(n_cols - 1):
            fid = row.get(f"hop_{i}_id")
            tid = row.get(f"hop_{i + 1}_id")
            if fid is None or tid is None:
                continue
            if (isinstance(fid, float) and pd.isna(fid)) or (
                isinstance(tid, float) and pd.isna(tid)
            ):
                continue
            key = (i, str(fid), str(tid))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    # Step 4: filter displayed genes + intermediates to those on at least one surviving path.
    if surviving_rows:
        genes_with_paths = [
            g for g in genes
            if f"{source_type}:{g.get('gene_id')}" in visible_source_ids
        ]
        genes_to_draw = genes_with_paths[:max_genes]
        final_buckets: list[list[dict]] = []
        for hi in range(n_intermediate_hops):
            bucket = hop_buckets[hi]
            used = used_per_hop.get(hi + 1, set())
            final_buckets.append(
                [it for it in bucket if str(it.get("intermediate_id", "") or "") in used]
            )
    else:
        # No surviving paths (no path data, or no complete coverage): fall back to
        # the raw bucketed view with no edges.
        genes_to_draw = list(genes)[:max_genes]
        final_buckets = hop_buckets

    if figsize is None:
        figsize = (
            max(10.0, 2.8 * n_cols + 2.5),
            max(8.0, 0.42 * max(len(genes_to_draw), 1) + 3.0),
        )
    fig, ax = plt.subplots(figsize=figsize)

    if not genes_to_draw and all(not b for b in final_buckets):
        ax.text(0.5, 0.5, "No complete paths to display", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        _save_figure(fig, output_stem)
        plt.close(fig)
        return

    x_positions = _column_x_positions(n_cols)
    gene_x = x_positions[0]
    hop_xs = x_positions[1:-1]
    target_x = x_positions[-1]

    gene_y = _y_coords(len(genes_to_draw))
    target_pos = (target_x, 0.5)

    # Position maps. Hop positions use (hop_idx, qualified_id) so the same id
    # appearing at multiple hops (e.g. G-G-G-A) gets distinct coordinates.
    source_positions: dict[str, tuple[float, float]] = {}
    for i, gene in enumerate(genes_to_draw):
        gid = gene.get("gene_id")
        if gid is not None:
            source_positions[f"{source_type}:{gid}"] = (gene_x, gene_y[i])

    hop_positions: dict[int, dict[str, tuple[float, float]]] = {}
    for hop_idx, hop_x in enumerate(hop_xs):
        items = final_buckets[hop_idx]
        ys = _y_coords(len(items))
        pos_map: dict[str, tuple[float, float]] = {}
        for j, interm in enumerate(items):
            int_id = str(interm.get("intermediate_id", "") or "")
            if int_id:
                pos_map[int_id] = (hop_x, ys[j])
        hop_positions[hop_idx + 1] = pos_map

    # Draw edges first (lowest z-order) so nodes cover their endpoints.
    if edge_counts:
        max_count = max(edge_counts.values())
        for (i, fid, tid), cnt in edge_counts.items():
            # Determine from_pos
            if i == 0:
                from_pos = source_positions.get(fid)
            else:
                from_pos = hop_positions.get(i, {}).get(fid)
            # Determine to_pos
            if i + 1 == n_cols - 1:
                to_pos = target_pos
            else:
                to_pos = hop_positions.get(i + 1, {}).get(tid)
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

    # Draw source gene nodes
    gene_color = NODE_COLORS.get(source_type, DEFAULT_COLOR)
    dwpc_values = [float(g.get("dwpc", 1.0)) for g in genes_to_draw]
    max_dwpc = max(dwpc_values) if dwpc_values else 1.0
    min_dwpc = min(dwpc_values) if dwpc_values else 0.0

    for i, gene in enumerate(genes_to_draw):
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

    # Draw intermediate nodes
    for hop_idx, hop_x in enumerate(hop_xs):
        items = final_buckets[hop_idx]
        if not items:
            # Only happens when no path in top-N per-hop selection survived;
            # be explicit rather than silently showing a blank column.
            ax.text(hop_x, 0.5, "(no complete paths at this hop)",
                    ha="center", va="center", fontsize=9, color="#888888", style="italic")
            continue
        node_type = hop_node_types[hop_idx]
        color = NODE_COLORS.get(node_type, DEFAULT_COLOR)
        usages = [float(it.get("n_genes_using", 1)) for it in items]
        max_usage = max(usages) if usages else 1.0
        ys = _y_coords(len(items))
        for j, interm in enumerate(items):
            iy = ys[j]
            int_id = str(interm.get("intermediate_id", "") or "")
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


def _collect_pooled_path_metadata(
    path_edges: pd.DataFrame, hop_counts: dict[str, int], max_hops: int
) -> tuple[list[tuple[dict, int]], dict[int, dict[str, int]], dict[int, dict[str, set[str]]], dict[str, str]]:
    """Walk path_edges once, gather path rows + per-hop-per-intermediate stats.

    Returns (path_rows, per_hop_traversals, per_hop_gene_sets, intermediate_types).
    """
    per_hop_traversals: dict[int, dict[str, int]] = {h: {} for h in range(1, max_hops + 1)}
    per_hop_gene_sets: dict[int, dict[str, set[str]]] = {h: {} for h in range(1, max_hops + 1)}
    intermediate_types: dict[str, str] = {}
    path_rows: list[tuple[dict, int]] = []

    for row in path_edges.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(path_edges.columns, row))
        mp = str(row_dict.get("metapath", ""))
        mp_hop_count = hop_counts.get(mp, 0)
        if mp_hop_count == 0:
            continue
        path_rows.append((row_dict, mp_hop_count))
        source_id = row_dict.get("hop_0_id")
        for hi in range(1, mp_hop_count + 1):
            iid = row_dict.get(f"hop_{hi}_id")
            itype = row_dict.get(f"hop_{hi}_type")
            if iid is None or (isinstance(iid, float) and pd.isna(iid)):
                continue
            iid = str(iid)
            per_hop_traversals[hi][iid] = per_hop_traversals[hi].get(iid, 0) + 1
            per_hop_gene_sets[hi].setdefault(iid, set())
            if source_id is not None and not (isinstance(source_id, float) and pd.isna(source_id)):
                per_hop_gene_sets[hi][iid].add(str(source_id))
            if itype and not (isinstance(itype, float) and pd.isna(itype)):
                intermediate_types.setdefault(iid, str(itype))

    return path_rows, per_hop_traversals, per_hop_gene_sets, intermediate_types


def _render_pooled_subgraph(
    *,
    surviving: list[tuple[dict, int]],
    max_hops: int,
    metapaths: list[str],
    genes: list[dict],
    intermediate_types: dict[str, str],
    intermediate_names: dict[str, str],
    gene_set_id: str,
    target_name: str,
    subtitle: str,
    output_stem: Path,
    max_genes: int = 20,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Draw a pooled per-LV subgraph from already-selected `surviving` path rows."""
    if not surviving:
        print(f"  [pooled] {gene_set_id}: no paths to render for '{subtitle}'")
        return

    # Collect visible sources, used-per-hop ids, edge counts.
    visible_source_ids: set[str] = set()
    used_per_hop: dict[int, dict[str, int]] = {h: {} for h in range(1, max_hops + 1)}
    used_gene_sets_per_hop: dict[int, dict[str, set[str]]] = {h: {} for h in range(1, max_hops + 1)}
    edges: dict[tuple[int, str, str], int] = {}  # (hop_idx_of_from, from_id, to_id)
    # Target is always drawn at column index max_hops + 1; encode it with tid=None in the key.

    for row_dict, mp_hop_count in surviving:
        src = row_dict.get("hop_0_id")
        if src is not None and not (isinstance(src, float) and pd.isna(src)):
            visible_source_ids.add(str(src))

        for hi in range(1, mp_hop_count + 1):
            iid = str(row_dict.get(f"hop_{hi}_id"))
            used_per_hop[hi][iid] = used_per_hop[hi].get(iid, 0) + 1
            used_gene_sets_per_hop[hi].setdefault(iid, set())
            if src is not None and not (isinstance(src, float) and pd.isna(src)):
                used_gene_sets_per_hop[hi][iid].add(str(src))

        # Edges within the path
        # source (hop 0) -> hop 1
        if src is not None and mp_hop_count >= 1:
            first = row_dict.get("hop_1_id")
            if first is not None and not (isinstance(first, float) and pd.isna(first)):
                edges[(0, str(src), str(first))] = edges.get((0, str(src), str(first)), 0) + 1
        # intermediate -> intermediate
        for hi in range(1, mp_hop_count):
            fid = row_dict.get(f"hop_{hi}_id")
            tid = row_dict.get(f"hop_{hi + 1}_id")
            if fid is None or tid is None:
                continue
            if (isinstance(fid, float) and pd.isna(fid)) or (isinstance(tid, float) and pd.isna(tid)):
                continue
            edges[(hi, str(fid), str(tid))] = edges.get((hi, str(fid), str(tid)), 0) + 1
        # last intermediate -> target (may skip columns if mp_hop_count < max_hops)
        last = row_dict.get(f"hop_{mp_hop_count}_id")
        if last is not None and not (isinstance(last, float) and pd.isna(last)):
            # Use sentinel "__target__" for the target side of the key.
            edges[(mp_hop_count, str(last), "__target__")] = (
                edges.get((mp_hop_count, str(last), "__target__"), 0) + 1
            )

    # Step 5: layout. An intermediate can appear in more than one hop, so we key
    # positions by (hop_idx, qualified_id).
    n_cols = max_hops + 2
    if figsize is None:
        figsize = (max(11.0, 2.6 * n_cols + 3.0), max(8.0, 0.42 * max_genes + 3.0))
    fig, ax = plt.subplots(figsize=figsize)

    x_positions = _column_x_positions(n_cols)
    source_x = x_positions[0]
    hop_xs = x_positions[1:-1]
    target_x = x_positions[-1]

    # Filter BEFORE slicing so a source gene needed by a surviving path isn't
    # cut off just because it's past position `max_genes` in the input list.
    genes_with_paths = [
        g for g in genes
        if f"G:{g.get('gene_id')}" in visible_source_ids
    ]
    genes_to_draw = genes_with_paths[:max_genes]
    gene_y = _y_coords(len(genes_to_draw)) if genes_to_draw else []

    source_positions: dict[str, tuple[float, float]] = {}
    for i, gene in enumerate(genes_to_draw):
        gid = gene.get("gene_id")
        if gid is not None:
            source_positions[f"G:{gid}"] = (source_x, gene_y[i])

    # Per-hop items sorted by gene coverage desc.
    hop_items: dict[int, list[str]] = {}
    for hi in range(1, max_hops + 1):
        iids = list(used_per_hop.get(hi, {}).keys())
        iids.sort(
            key=lambda i: (
                len(used_gene_sets_per_hop[hi].get(i, set())),
                used_per_hop[hi].get(i, 0),
            ),
            reverse=True,
        )
        hop_items[hi] = iids

    # Position map keyed by (hop_idx, qualified_id).
    hop_positions: dict[int, dict[str, tuple[float, float]]] = {}
    for hi in range(1, max_hops + 1):
        items = hop_items[hi]
        ys = _y_coords(len(items))
        pos_map: dict[str, tuple[float, float]] = {}
        for rank, iid in enumerate(items):
            pos_map[iid] = (hop_xs[hi - 1], ys[rank])
        hop_positions[hi] = pos_map

    target_pos = (target_x, 0.5)

    # Total source-gene pool size for pct display.
    total_source_genes = len(visible_source_ids) if visible_source_ids else 1

    # Step 6: draw edges (lowest zorder).
    max_count = max(edges.values()) if edges else 1

    # Skip edges are short-metapath tails that jump directly to the target,
    # bypassing one or more intermediate hop columns. They're the visually
    # interesting ones for single-intermediate metapaths, so color them
    # distinctly (amber) rather than the default gray.
    REGULAR_EDGE_COLOR = "#546E7A"
    SKIP_EDGE_COLOR = "#8A0102"
    any_skip = False

    def _edge_style(cnt: int) -> tuple[float, float]:
        frac = cnt / max_count if max_count else 0.0
        return (0.15 + 0.55 * frac, 0.4 + 2.2 * frac)

    for (hop_idx_from, fid, tid), cnt in edges.items():
        if hop_idx_from == 0:
            from_pos = source_positions.get(fid)
        else:
            from_pos = hop_positions.get(hop_idx_from, {}).get(fid)
        if tid == "__target__":
            to_pos = target_pos
        else:
            to_pos = hop_positions.get(hop_idx_from + 1, {}).get(tid)
        if from_pos is None or to_pos is None:
            continue
        is_skip = (tid == "__target__") and (hop_idx_from < max_hops)
        edge_color = SKIP_EDGE_COLOR if is_skip else REGULAR_EDGE_COLOR
        if is_skip:
            any_skip = True
        a, lw = _edge_style(cnt)
        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                color=edge_color, alpha=a, linewidth=lw, zorder=1)

    # Step 7: draw source genes.
    gene_color = NODE_COLORS.get("G", DEFAULT_COLOR)
    for i, gene in enumerate(genes_to_draw):
        gy = gene_y[i]
        gene_name = gene.get("gene_name") or str(gene.get("gene_id", "?"))
        ax.scatter(source_x, gy, s=320, c=gene_color,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(source_x - 0.015, gy, str(gene_name)[:12],
                ha="right", va="center", fontsize=8, fontweight="medium")

    # Step 8: draw intermediates. Labels show "% of source genes at this hop".
    for hi in range(1, max_hops + 1):
        items = hop_items.get(hi, [])
        for rank, iid in enumerate(items):
            gene_cov = len(used_gene_sets_per_hop[hi].get(iid, set()))
            cnt = used_per_hop[hi].get(iid, 0)
            pct = (gene_cov / total_source_genes * 100.0) if total_source_genes else 0.0
            node_type = intermediate_types.get(iid, "?")
            color = NODE_COLORS.get(node_type, DEFAULT_COLOR)
            node_size = 180 + 520 * (cnt / max_count if max_count > 0 else 0)
            pos = hop_positions[hi][iid]
            ax.scatter(pos[0], pos[1], s=node_size, c=color,
                       edgecolors="white", linewidths=1.2, zorder=3)
            name = intermediate_names.get(iid)
            if not name:
                name = _format_intermediate_label(iid, intermediate_names)
            label = str(name)[:22]
            ax.text(
                pos[0] + 0.012, pos[1],
                f"{label} ({pct:.0f}% genes)",
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
    plural = "s" if len(metapaths) != 1 else ""
    ax.set_title(
        f"{gene_set_id}: {target_name}  ({subtitle}; {len(metapaths)} metapath{plural})",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Color legend for the node types of intermediates actually drawn.
    drawn_ids: set[str] = set()
    for items in hop_items.values():
        drawn_ids.update(items)
    types_present = sorted(set(intermediate_types.get(iid, "?") for iid in drawn_ids))
    legend_handles = []
    for t in types_present:
        full = NODE_TYPE_FULL_NAMES.get(t, t)
        legend_handles.append(
            plt.Line2D([], [], marker="o", color="w",
                       markerfacecolor=NODE_COLORS.get(t, DEFAULT_COLOR),
                       markersize=10, label=full)
        )
    # Edge color legend entries: regular full-path edges, and skip-edges that
    # jump from a middle hop directly to the target (from shorter metapaths).
    legend_handles.append(
        plt.Line2D([], [], color=REGULAR_EDGE_COLOR, linewidth=2.4,
                   label="Full-path edge")
    )
    if any_skip:
        legend_handles.append(
            plt.Line2D([], [], color=SKIP_EDGE_COLOR, linewidth=2.4,
                       label="Shortcut to target")
        )
    if legend_handles:
        ax.legend(handles=legend_handles, title="Legend",
                  loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _save_figure(fig, output_stem)
    plt.close(fig)


def plot_lv_top_shared_subgraph(
    gene_set_id: str,
    target_name: str,
    metapaths: list[str],
    genes: list[dict],
    path_edges: pd.DataFrame,
    intermediate_names: dict[str, str],
    output_stem: Path,
    max_genes: int = 20,
    top_n_per_hop: int = 15,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Pooled subgraph weighted toward intermediates shared across many source
    genes. Per-hop top-N intermediates by gene coverage define the allow-set;
    a path survives only if every hop is in the allow-set. Highlights nodes
    that bridge many genes."""
    if path_edges is None or path_edges.empty:
        print(f"  [top_shared] {gene_set_id}: no path-level data, skipping")
        return

    hop_counts = {mp: max(len(parse_metapath_nodes(mp)) - 2, 0) for mp in metapaths}
    max_hops = max(hop_counts.values()) if hop_counts else 1
    if max_hops == 0:
        max_hops = 1

    path_rows, _, per_hop_gene_sets, intermediate_types = _collect_pooled_path_metadata(
        path_edges, hop_counts, max_hops
    )
    if not path_rows:
        print(f"  [top_shared] {gene_set_id}: no usable paths found")
        return

    per_hop_traversals: dict[int, dict[str, int]] = {h: {} for h in range(1, max_hops + 1)}
    for row_dict, mp_hop_count in path_rows:
        for hi in range(1, mp_hop_count + 1):
            iid = row_dict.get(f"hop_{hi}_id")
            if iid is None or (isinstance(iid, float) and pd.isna(iid)):
                continue
            iid = str(iid)
            per_hop_traversals[hi][iid] = per_hop_traversals[hi].get(iid, 0) + 1

    allowed_per_hop: dict[int, set[str]] = {}
    for hi in range(1, max_hops + 1):
        candidates = per_hop_gene_sets[hi]
        if not candidates:
            allowed_per_hop[hi] = set()
            continue
        ranked = sorted(
            candidates.items(),
            key=lambda kv: (len(kv[1]), per_hop_traversals[hi].get(kv[0], 0)),
            reverse=True,
        )
        allowed_per_hop[hi] = {iid for iid, _ in ranked[:top_n_per_hop]}

    surviving: list[tuple[dict, int]] = []
    for row_dict, mp_hop_count in path_rows:
        ok = True
        for hi in range(1, mp_hop_count + 1):
            iid = row_dict.get(f"hop_{hi}_id")
            if iid is None or (isinstance(iid, float) and pd.isna(iid)):
                ok = False
                break
            if str(iid) not in allowed_per_hop.get(hi, set()):
                ok = False
                break
        if ok:
            surviving.append((row_dict, mp_hop_count))

    _render_pooled_subgraph(
        surviving=surviving,
        max_hops=max_hops,
        metapaths=metapaths,
        genes=genes,
        intermediate_types=intermediate_types,
        intermediate_names=intermediate_names,
        gene_set_id=gene_set_id,
        target_name=target_name,
        subtitle="top shared intermediates",
        output_stem=output_stem,
        max_genes=max_genes,
        figsize=figsize,
    )


def plot_lv_top_paths_subgraph(
    gene_set_id: str,
    target_name: str,
    metapaths: list[str],
    genes: list[dict],
    path_edges: pd.DataFrame,
    intermediate_names: dict[str, str],
    output_stem: Path,
    max_genes: int = 20,
    top_n_paths: int = 40,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Pooled subgraph showing the top N highest-scoring individual paths from
    the LV to its target, regardless of intermediate sharing. Lets unique or
    rare intermediates show up if they belong to a strong path."""
    if path_edges is None or path_edges.empty:
        print(f"  [top_paths] {gene_set_id}: no path-level data, skipping")
        return

    hop_counts = {mp: max(len(parse_metapath_nodes(mp)) - 2, 0) for mp in metapaths}
    max_hops = max(hop_counts.values()) if hop_counts else 1
    if max_hops == 0:
        max_hops = 1

    path_rows, _, _, intermediate_types = _collect_pooled_path_metadata(
        path_edges, hop_counts, max_hops
    )
    if not path_rows:
        print(f"  [top_paths] {gene_set_id}: no usable paths found")
        return

    def _row_score(row_dict: dict) -> float:
        val = row_dict.get("path_score")
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return -float("inf")
        try:
            return float(val)
        except (TypeError, ValueError):
            return -float("inf")

    # Stratified selection: top N/K paths per metapath, so single-intermediate
    # metapaths (GaDrD, etc.) aren't squeezed out by longer ones whose scores
    # typically dominate a global ranking.
    k = max(len(hop_counts), 1)
    per_metapath_budget = max(1, top_n_paths // k)
    grouped: dict[str, list[tuple[dict, int]]] = {}
    for row_dict, mp_hop_count in path_rows:
        mp = str(row_dict.get("metapath", ""))
        grouped.setdefault(mp, []).append((row_dict, mp_hop_count))

    surviving: list[tuple[dict, int]] = []
    for mp, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda pair: _row_score(pair[0]), reverse=True)
        surviving.extend(rows_sorted[:per_metapath_budget])
    # Sort the final selection by score for deterministic ordering.
    surviving.sort(key=lambda pair: _row_score(pair[0]), reverse=True)

    _render_pooled_subgraph(
        surviving=surviving,
        max_hops=max_hops,
        metapaths=metapaths,
        genes=genes,
        intermediate_types=intermediate_types,
        intermediate_names=intermediate_names,
        gene_set_id=gene_set_id,
        target_name=target_name,
        subtitle=f"top {len(surviving)} paths by score",
        output_stem=output_stem,
        max_genes=max_genes,
        figsize=figsize,
    )


# Back-compat alias so existing callers don't break.
plot_combined_lv_subgraph = plot_lv_top_shared_subgraph


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

    # Tolerate an empty gene-table CSV: treat it like "no gene table" so the
    # viz still renders with placeholder gene rows instead of crashing.
    if gene_table_path and gene_table_path.exists():
        try:
            gene_table = pd.read_csv(gene_table_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {gene_table_path} is empty; proceeding without gene-table data")
            gene_table = pd.DataFrame()
    else:
        gene_table = pd.DataFrame()
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
            raw_target_name = mp_row.get("target_name")
            if raw_target_name is None or (isinstance(raw_target_name, float) and pd.isna(raw_target_name)):
                raw_target_name = ""
            target_name = str(raw_target_name).strip()
            # For year mode the target IS the gene-set's GO term; resolve its
            # human name via the node TSV lookup rather than falling back to
            # the literal string "Unknown".
            if (not target_name or target_name == "Unknown") and analysis_type == "year":
                resolved = int_name_lookup.get(f"BP:{gene_set_id}")
                if resolved:
                    target_name = str(resolved)
                else:
                    target_name = gene_set_id
            if not target_name:
                target_name = "Unknown"

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

            # Derive source genes from the per-path records when gene_table
            # has no entries. Using actual hop_0_id values (not placeholders
            # like Gene_0 / Gene_1) ensures they match visible_source_ids so
            # the Source column and its edges actually render.
            if not mp_genes and not mp_paths.empty and "gene_id" in mp_paths.columns:
                mp_gene_ids = (
                    mp_paths["gene_id"].dropna().unique().tolist()
                )
                mp_genes = []
                for g in mp_gene_ids:
                    try:
                        gid = int(g)
                    except (TypeError, ValueError):
                        continue
                    mp_genes.append({
                        "gene_id": gid,
                        "gene_name": gene_names.get(gid, str(gid)),
                        "dwpc": 1.0,
                    })

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

        # Two pooled per-LV plots. Both use ALL surviving metapaths for the LV
        # (not just the top-K used for individual per-metapath plots) so users
        # see the full connectivity picture.
        if not gene_paths_df.empty:
            all_metapaths = gs_df["metapath"].astype(str).unique().tolist()
            combined_paths = gene_paths_df[
                (gene_paths_df[id_col].astype(str) == gene_set_id)
                & (gene_paths_df["metapath"].astype(str).isin(all_metapaths))
            ]
            if not combined_paths.empty:
                raw_combined_target = gs_df.iloc[0].get("target_name")
                if raw_combined_target is None or (
                    isinstance(raw_combined_target, float) and pd.isna(raw_combined_target)
                ):
                    raw_combined_target = ""
                combined_target = str(raw_combined_target).strip()
                if (not combined_target or combined_target == "Unknown") and analysis_type == "year":
                    resolved = int_name_lookup.get(f"BP:{gene_set_id}")
                    combined_target = str(resolved) if resolved else gene_set_id
                if not combined_target:
                    combined_target = "Unknown"
                # Every source gene_id appearing in combined_paths must be in
                # combined_genes or its edges will be skipped at draw time.
                # Start from gene_table (preferred names/scores), then backfill
                # any missing ids from combined_paths using Gene.tsv symbols.
                combined_genes_by_id: dict[int, dict] = {}
                if not gene_table.empty and "gene_set_id" in gene_table.columns:
                    gt_rows = gene_table[
                        gene_table["gene_set_id"].astype(str) == gene_set_id
                    ].drop_duplicates(subset=["gene_id"]).to_dict("records")
                    for rec in gt_rows:
                        try:
                            gid = int(rec.get("gene_id"))
                        except (TypeError, ValueError):
                            continue
                        combined_genes_by_id[gid] = rec
                path_gene_ids = combined_paths["gene_id"].dropna().unique().tolist()
                for g in path_gene_ids:
                    try:
                        gid = int(g)
                    except (TypeError, ValueError):
                        continue
                    if gid not in combined_genes_by_id:
                        combined_genes_by_id[gid] = {
                            "gene_id": gid,
                            "gene_name": gene_names.get(gid, str(gid)),
                            "dwpc": 1.0,
                        }
                combined_genes = list(combined_genes_by_id.values())

                shared_stem = output_dir / f"{gene_set_id}_top_shared"
                print(f"Generating: {shared_stem.name}")
                plot_lv_top_shared_subgraph(
                    gene_set_id=gene_set_id,
                    target_name=combined_target,
                    metapaths=all_metapaths,
                    genes=combined_genes,
                    path_edges=combined_paths,
                    intermediate_names=int_name_lookup,
                    output_stem=shared_stem,
                )

                paths_stem = output_dir / f"{gene_set_id}_top_paths"
                print(f"Generating: {paths_stem.name}")
                plot_lv_top_paths_subgraph(
                    gene_set_id=gene_set_id,
                    target_name=combined_target,
                    metapaths=all_metapaths,
                    genes=combined_genes,
                    path_edges=combined_paths,
                    intermediate_names=int_name_lookup,
                    output_stem=paths_stem,
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
