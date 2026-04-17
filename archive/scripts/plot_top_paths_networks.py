# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""
Visualize top gene-BP pairs as bipartite networks per metapath.

Inputs:
  output/metapath_analysis/top_paths/top_gene_bp_pairs_2016.csv
  output/metapath_analysis/top_paths/top_gene_bp_pairs_2024.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.lines import Line2D


def _scale(values, min_size=0.6, max_size=3.5):
    vmin = np.min(values)
    vmax = np.max(values)
    if vmax == vmin:
        return np.full_like(values, (min_size + max_size) / 2.0, dtype=float)
    return min_size + (values - vmin) / (vmax - vmin) * (max_size - min_size)


def _scale_with_bounds(values, vmin, vmax, min_size=0.6, max_size=3.5):
    if vmax == vmin:
        return np.full_like(values, (min_size + max_size) / 2.0, dtype=float)
    return min_size + (values - vmin) / (vmax - vmin) * (max_size - min_size)


def _sanitize(name: str, max_len: int = 80) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name[:max_len] if len(name) > max_len else name


def _parse_metanodes(metapath: str) -> list[str]:
    node_abbrevs = ["BP", "CC", "MF", "PW", "SE", "PC", "G", "A", "D", "C", "S"]
    node_abbrevs = sorted(node_abbrevs, key=len, reverse=True)
    nodes = []
    i = 0
    while i < len(metapath):
        matched = False
        for ab in node_abbrevs:
            if metapath.startswith(ab, i):
                nodes.append(ab)
                i += len(ab)
                matched = True
                break
        if not matched:
            i += 1
    return nodes


def plot_metapath_bp_network(df, metapath, go_id, year, out_dir, label_top_genes=8):
    df = df[(df["metapath"] == metapath) & (df["go_id"] == go_id)].copy()
    if df.empty:
        return

    genes = df["gene_name"].fillna(df["entrez_gene_id"].astype(str)).unique().tolist()
    bp_label = df["go_name"].iloc[0] if isinstance(df["go_name"].iloc[0], str) else go_id

    # Positions
    gene_y = np.linspace(0, 1, len(genes)) if genes else []
    gene_pos = {g: (0.0, y) for g, y in zip(genes, gene_y)}
    bp_pos = {bp_label: (1.0, 0.5)}

    # Intermediate nodes (schema only)
    metanodes = _parse_metanodes(metapath)
    intermediates = metanodes[1:-1] if len(metanodes) >= 3 else []
    inter_pos = {}
    if intermediates:
        xs = np.linspace(0.25, 0.75, len(intermediates))
        for x, node in zip(xs, intermediates):
            inter_pos[node] = (x, 0.5)

    # Node sizes by DWPC sum (within this BP)
    gene_strength = df.groupby("gene_name")["dwpc"].sum().reindex(genes).fillna(0).values
    gene_sizes = _scale(gene_strength, min_size=30, max_size=180)
    bp_size = _scale(np.array([df["dwpc"].sum()]), min_size=120, max_size=260)[0]

    # Edge widths by DWPC
    edge_widths = _scale(df["dwpc"].values, min_size=0.4, max_size=3.0)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.2 * len(genes) + 2)))

    # Draw edges (genes to BP)
    for (row, w) in zip(df.itertuples(index=False), edge_widths):
        gene_label = row.gene_name if isinstance(row.gene_name, str) else str(row.entrez_gene_id)
        x1, y1 = gene_pos.get(gene_label, (0.0, 0.0))
        x2, y2 = bp_pos[bp_label]
        ax.plot([x1, x2], [y1, y2], color="#9aa0a6", linewidth=w, alpha=0.7)

    # Draw intermediate schema chain
    if intermediates:
        chain_nodes = ["G"] + intermediates + ["BP"]
        chain_pos = [(0.0, 0.5)] + [inter_pos[n] for n in intermediates] + [(1.0, 0.5)]
        for (x1, y1), (x2, y2) in zip(chain_pos, chain_pos[1:]):
            ax.plot([x1, x2], [y1, y2], color="#c7c7c7", linewidth=1.5, linestyle="--", alpha=0.8)
        for node in intermediates:
            x, y = inter_pos[node]
            ax.scatter([x], [y], s=180, c="#ffffff", edgecolors="#4d4d4d", linewidths=1.0, zorder=3)
            ax.text(x, y, node, ha="center", va="center", fontsize=9, zorder=4)

    # Draw nodes
    ax.scatter([gene_pos[g][0] for g in genes], [gene_pos[g][1] for g in genes],
               s=gene_sizes, c="#4c78a8", alpha=0.85, edgecolors="white", linewidths=0.5, label="Genes")
    ax.scatter([bp_pos[bp_label][0]], [bp_pos[bp_label][1]],
               s=bp_size, c="#f58518", alpha=0.9, edgecolors="white", linewidths=0.5, label="Biological Process")

    # Labels
    ax.text(bp_pos[bp_label][0] + 0.02, bp_pos[bp_label][1], bp_label, fontsize=8, va="center")

    if label_top_genes > 0 and len(genes) > 0:
        gene_strength_series = pd.Series(gene_strength, index=genes).sort_values(ascending=False)
        top_genes = gene_strength_series.head(label_top_genes).index.tolist()
        for g in top_genes:
            x, y = gene_pos[g]
            ax.text(x - 0.02, y, g, fontsize=7, va="center", ha="right")

    ax.set_title(f"{metapath} | {year} | {go_id}", fontsize=12)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

    out_path = out_dir / f"{year}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _gene_label_map(df):
    label_map = {}
    for row in df.itertuples(index=False):
        if getattr(row, "entrez_gene_id", None) is None:
            continue
        if row.entrez_gene_id not in label_map and isinstance(row.gene_name, str) and row.gene_name.strip():
            label_map[row.entrez_gene_id] = row.gene_name.strip()
    return label_map


def _draw_bp_network(
    ax,
    genes,
    gene_pos,
    bp_pos,
    intermediates,
    inter_pos,
    dwpc_values,
    gene_sizes,
    bp_size,
    edge_widths,
    title,
    label_top_genes,
    bp_label,
):
    for (gene_label, w) in zip(genes, edge_widths):
        x1, y1 = gene_pos.get(gene_label, (0.0, 0.0))
        x2, y2 = bp_pos[bp_label]
        ax.plot([x1, x2], [y1, y2], color="#9aa0a6", linewidth=w, alpha=0.7)

    if intermediates:
        chain_pos = [(0.0, 0.5)] + [inter_pos[n] for n in intermediates] + [(1.0, 0.5)]
        for (x1, y1), (x2, y2) in zip(chain_pos, chain_pos[1:]):
            ax.plot([x1, x2], [y1, y2], color="#c7c7c7", linewidth=1.5, linestyle="--", alpha=0.8)
        for node in intermediates:
            x, y = inter_pos[node]
            ax.scatter([x], [y], s=180, c="#ffffff", edgecolors="#4d4d4d", linewidths=1.0, zorder=3)
            ax.text(x, y, node, ha="center", va="center", fontsize=9, zorder=4)

    ax.scatter(
        [gene_pos[g][0] for g in genes],
        [gene_pos[g][1] for g in genes],
        s=gene_sizes,
        c="#4c78a8",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        label="Genes",
    )
    ax.scatter(
        [bp_pos[bp_label][0]],
        [bp_pos[bp_label][1]],
        s=bp_size,
        c="#f58518",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        label="Biological Process",
    )

    ax.text(bp_pos[bp_label][0] + 0.02, bp_pos[bp_label][1], bp_label, fontsize=8, va="center")

    if label_top_genes > 0 and len(genes) > 0:
        gene_strength_series = pd.Series(dwpc_values, index=genes).sort_values(ascending=False)
        top_genes = gene_strength_series.head(label_top_genes).index.tolist()
        for g in top_genes:
            x, y = gene_pos[g]
            ax.text(x - 0.02, y, g, fontsize=7, va="center", ha="right")

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")


def plot_metapath_bp_diff_network(
    df_a,
    df_b,
    metapath,
    go_id,
    go_name,
    year_a,
    year_b,
    out_dir,
    label_top_genes=8,
):
    sub_a = df_a[(df_a["metapath"] == metapath) & (df_a["go_id"] == go_id)].copy()
    sub_b = df_b[(df_b["metapath"] == metapath) & (df_b["go_id"] == go_id)].copy()
    if sub_a.empty and sub_b.empty:
        return

    go_label = go_name if isinstance(go_name, str) and go_name.strip() else go_id

    label_map = _gene_label_map(pd.concat([sub_a, sub_b], ignore_index=True))

    dwpc_a = sub_a.groupby("entrez_gene_id")["dwpc"].sum()
    dwpc_b = sub_b.groupby("entrez_gene_id")["dwpc"].sum()
    gene_ids = sorted(set(dwpc_a.index).union(dwpc_b.index))
    if not gene_ids:
        return

    dwpc_a = dwpc_a.reindex(gene_ids).fillna(0.0)
    dwpc_b = dwpc_b.reindex(gene_ids).fillna(0.0)
    delta = dwpc_b.values - dwpc_a.values

    genes = [label_map.get(gid, str(gid)) for gid in gene_ids]
    order = np.argsort(-np.abs(delta))
    genes = [genes[i] for i in order]
    delta = delta[order]

    gene_y = np.linspace(0, 1, len(genes)) if genes else []
    gene_pos = {g: (0.0, y) for g, y in zip(genes, gene_y)}
    bp_pos = {go_label: (1.0, 0.5)}

    metanodes = _parse_metanodes(metapath)
    intermediates = metanodes[1:-1] if len(metanodes) >= 3 else []
    inter_pos = {}
    if intermediates:
        xs = np.linspace(0.25, 0.75, len(intermediates))
        for x, node in zip(xs, intermediates):
            inter_pos[node] = (x, 0.5)

    gene_sizes = _scale(np.abs(delta), min_size=30, max_size=180)
    bp_size = _scale(np.array([np.abs(delta).sum()]), min_size=120, max_size=260)[0]
    edge_widths = _scale(np.abs(delta), min_size=0.4, max_size=3.0)
    edge_colors = np.where(delta > 0, "#4c78a8", np.where(delta < 0, "#e45756", "#9aa0a6"))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.2 * len(genes) + 2)))

    for (gene_label, w, color) in zip(genes, edge_widths, edge_colors):
        x1, y1 = gene_pos.get(gene_label, (0.0, 0.0))
        x2, y2 = bp_pos[go_label]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=w, alpha=0.8)

    if intermediates:
        chain_pos = [(0.0, 0.5)] + [inter_pos[n] for n in intermediates] + [(1.0, 0.5)]
        for (x1, y1), (x2, y2) in zip(chain_pos, chain_pos[1:]):
            ax.plot([x1, x2], [y1, y2], color="#c7c7c7", linewidth=1.5, linestyle="--", alpha=0.8)
        for node in intermediates:
            x, y = inter_pos[node]
            ax.scatter([x], [y], s=180, c="#ffffff", edgecolors="#4d4d4d", linewidths=1.0, zorder=3)
            ax.text(x, y, node, ha="center", va="center", fontsize=9, zorder=4)

    ax.scatter(
        [gene_pos[g][0] for g in genes],
        [gene_pos[g][1] for g in genes],
        s=gene_sizes,
        c="#4c78a8",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        label="Genes",
    )
    ax.scatter(
        [bp_pos[go_label][0]],
        [bp_pos[go_label][1]],
        s=bp_size,
        c="#f58518",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        label="Biological Process",
    )

    ax.text(bp_pos[go_label][0] + 0.02, bp_pos[go_label][1], go_label, fontsize=8, va="center")

    if label_top_genes > 0 and len(genes) > 0:
        top_genes = [genes[i] for i in np.argsort(-np.abs(delta))[:label_top_genes]]
        for g in top_genes:
            x, y = gene_pos[g]
            ax.text(x - 0.02, y, g, fontsize=7, va="center", ha="right")

    ax.set_title(f"{metapath} | {year_b} - {year_a} | {go_id}", fontsize=12)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Genes", markerfacecolor="#4c78a8", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Biological Process", markerfacecolor="#f58518", markersize=8),
        Line2D([0], [0], color="#4c78a8", lw=2, label=f"Increase ({year_b} > {year_a})"),
        Line2D([0], [0], color="#e45756", lw=2, label=f"Decrease ({year_b} < {year_a})"),
        Line2D([0], [0], color="#9aa0a6", lw=2, label="No change"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    out_path = out_dir / f"{year_a}_{year_b}_diff.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metapath_bp_side_by_side(
    df_a,
    df_b,
    metapath,
    go_id,
    go_name,
    year_a,
    year_b,
    out_dir,
    label_top_genes=8,
):
    sub_a = df_a[(df_a["metapath"] == metapath) & (df_a["go_id"] == go_id)].copy()
    sub_b = df_b[(df_b["metapath"] == metapath) & (df_b["go_id"] == go_id)].copy()
    if sub_a.empty and sub_b.empty:
        return

    go_label = go_name if isinstance(go_name, str) and go_name.strip() else go_id

    label_map = _gene_label_map(pd.concat([sub_a, sub_b], ignore_index=True))
    dwpc_a = sub_a.groupby("entrez_gene_id")["dwpc"].sum()
    dwpc_b = sub_b.groupby("entrez_gene_id")["dwpc"].sum()
    gene_ids = sorted(set(dwpc_a.index).union(dwpc_b.index))
    if not gene_ids:
        return

    values_a = dwpc_a.reindex(gene_ids).fillna(0.0).values
    values_b = dwpc_b.reindex(gene_ids).fillna(0.0).values
    combined = values_a + values_b
    order = np.argsort(-combined)

    values_a = values_a[order]
    values_b = values_b[order]
    genes = [label_map.get(gid, str(gid)) for gid in gene_ids]
    genes = [genes[i] for i in order]

    gene_y = np.linspace(0, 1, len(genes)) if genes else []
    gene_pos = {g: (0.0, y) for g, y in zip(genes, gene_y)}
    bp_pos = {go_label: (1.0, 0.5)}

    metanodes = _parse_metanodes(metapath)
    intermediates = metanodes[1:-1] if len(metanodes) >= 3 else []
    inter_pos = {}
    if intermediates:
        xs = np.linspace(0.25, 0.75, len(intermediates))
        for x, node in zip(xs, intermediates):
            inter_pos[node] = (x, 0.5)

    vmin = float(min(values_a.min(), values_b.min()))
    vmax = float(max(values_a.max(), values_b.max()))
    gene_sizes_a = _scale_with_bounds(values_a, vmin, vmax, min_size=30, max_size=180)
    gene_sizes_b = _scale_with_bounds(values_b, vmin, vmax, min_size=30, max_size=180)
    edge_widths_a = _scale_with_bounds(values_a, vmin, vmax, min_size=0.4, max_size=3.0)
    edge_widths_b = _scale_with_bounds(values_b, vmin, vmax, min_size=0.4, max_size=3.0)

    sum_a = float(values_a.sum())
    sum_b = float(values_b.sum())
    bp_vmin = min(sum_a, sum_b)
    bp_vmax = max(sum_a, sum_b)
    bp_size_a = _scale_with_bounds(np.array([sum_a]), bp_vmin, bp_vmax, min_size=120, max_size=260)[0]
    bp_size_b = _scale_with_bounds(np.array([sum_b]), bp_vmin, bp_vmax, min_size=120, max_size=260)[0]

    height = max(5, 0.2 * len(genes) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, height))

    _draw_bp_network(
        axes[0],
        genes,
        gene_pos,
        bp_pos,
        intermediates,
        inter_pos,
        values_a,
        gene_sizes_a,
        bp_size_a,
        edge_widths_a,
        f"{metapath} | {year_a} | {go_id}",
        label_top_genes,
        go_label,
    )
    _draw_bp_network(
        axes[1],
        genes,
        gene_pos,
        bp_pos,
        intermediates,
        inter_pos,
        values_b,
        gene_sizes_b,
        bp_size_b,
        edge_widths_b,
        f"{metapath} | {year_b} | {go_id}",
        label_top_genes,
        go_label,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Genes", markerfacecolor="#4c78a8", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Biological Process", markerfacecolor="#f58518", markersize=8),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False)

    out_path = out_dir / f"{year_a}_{year_b}_side_by_side.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def reverse_metapath_abbrev(metapath: str) -> str:
    node_abbrevs = {"G", "BP", "CC", "MF", "PW", "A", "D", "C", "SE", "S", "PC"}
    edge_abbrevs = {"p", "i", "c", "r", ">", "<", "a", "d", "u", "e", "b", "t", "l"}
    tokens = []
    pos = 0
    while pos < len(metapath):
        if pos + 2 <= len(metapath) and metapath[pos:pos+2] in node_abbrevs:
            tokens.append(metapath[pos:pos+2])
            pos += 2
        elif metapath[pos] in node_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        elif metapath[pos] in edge_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        elif metapath[pos] == ">":
            tokens.append(">")
            pos += 1
        elif metapath[pos] == "<":
            tokens.append("<")
            pos += 1
        else:
            pos += 1
    direction_map = {">": "<", "<": ">"}
    reversed_tokens = []
    for token in reversed(tokens):
        if token in direction_map:
            reversed_tokens.append(direction_map[token])
        else:
            reversed_tokens.append(token)
    return "".join(reversed_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bipartite networks for top gene-BP pairs per metapath.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--metapath", default=None, help="Optional metapath filter (exact match or reverse).")
    parser.add_argument("--label-top-genes", type=int, default=8)
    parser.add_argument("--diff", action="store_true", help="Plot differences between two years.")
    parser.add_argument("--side-by-side", action="store_true", help="Plot side-by-side panels for two years.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    top_dir = repo_root / "output" / "metapath_analysis" / "top_paths"
    out_dir = top_dir / "networks"
    out_dir.mkdir(parents=True, exist_ok=True)

    year_dfs = {}
    for year in args.years:
        path = top_dir / f"top_gene_bp_pairs_{year}.csv"
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue
        df = pd.read_csv(path)
        year_dfs[str(year)] = df

    if not year_dfs:
        return

    if args.diff and args.side_by_side:
        raise SystemExit("Choose only one of --diff or --side-by-side.")

    if args.side_by_side:
        if len(args.years) != 2:
            raise SystemExit("Side-by-side mode requires exactly two years (e.g., --years 2016 2024).")
        year_a, year_b = [str(y) for y in args.years]
        if year_a not in year_dfs or year_b not in year_dfs:
            missing = [y for y in (year_a, year_b) if y not in year_dfs]
            raise SystemExit(f"Missing data for years: {', '.join(missing)}")

        df_a = year_dfs[year_a]
        df_b = year_dfs[year_b]

        if args.metapath:
            mp = args.metapath
            metapath_values = set(pd.concat([df_a["metapath"], df_b["metapath"]]).unique())
            if mp not in metapath_values:
                mp_rev = reverse_metapath_abbrev(mp)
                if mp_rev in metapath_values:
                    mp = mp_rev
                else:
                    raise SystemExit(f"Metapath {args.metapath} not found in either year.")
            metapaths = [mp]
        else:
            metapaths = sorted(set(df_a["metapath"]).union(set(df_b["metapath"])))

        out_dir = top_dir / "networks_side_by_side"
        out_dir.mkdir(parents=True, exist_ok=True)

        keys_a = df_a[["metapath", "go_id", "go_name"]].drop_duplicates()
        keys_b = df_b[["metapath", "go_id", "go_name"]].drop_duplicates()
        key_map = {}
        for row in keys_a.itertuples(index=False):
            key_map[(row.metapath, row.go_id)] = row.go_name
        for row in keys_b.itertuples(index=False):
            key_map.setdefault((row.metapath, row.go_id), row.go_name)

        for mp in metapaths:
            mp_dir = out_dir / _sanitize(mp)
            for (metapath, go_id), go_name in sorted(key_map.items()):
                if metapath != mp:
                    continue
                bp_dir = mp_dir / _sanitize(f"{go_id}__{go_name}")
                bp_dir.mkdir(parents=True, exist_ok=True)
                plot_metapath_bp_side_by_side(
                    df_a,
                    df_b,
                    mp,
                    go_id,
                    go_name,
                    year_a,
                    year_b,
                    bp_dir,
                    label_top_genes=args.label_top_genes,
                )
        print(f"Saved side-by-side network plots for {year_a} vs {year_b} to {out_dir}")
        return

    if args.diff:
        if len(args.years) != 2:
            raise SystemExit("Diff mode requires exactly two years (e.g., --years 2016 2024).")
        year_a, year_b = [str(y) for y in args.years]
        if year_a not in year_dfs or year_b not in year_dfs:
            missing = [y for y in (year_a, year_b) if y not in year_dfs]
            raise SystemExit(f"Missing data for years: {', '.join(missing)}")

        df_a = year_dfs[year_a]
        df_b = year_dfs[year_b]

        if args.metapath:
            mp = args.metapath
            metapath_values = set(pd.concat([df_a["metapath"], df_b["metapath"]]).unique())
            if mp not in metapath_values:
                mp_rev = reverse_metapath_abbrev(mp)
                if mp_rev in metapath_values:
                    mp = mp_rev
                else:
                    raise SystemExit(f"Metapath {args.metapath} not found in either year.")
            metapaths = [mp]
        else:
            metapaths = sorted(set(df_a["metapath"]).union(set(df_b["metapath"])))

        out_dir = top_dir / "networks_diff"
        out_dir.mkdir(parents=True, exist_ok=True)

        keys_a = df_a[["metapath", "go_id", "go_name"]].drop_duplicates()
        keys_b = df_b[["metapath", "go_id", "go_name"]].drop_duplicates()
        key_map = {}
        for row in keys_a.itertuples(index=False):
            key_map[(row.metapath, row.go_id)] = row.go_name
        for row in keys_b.itertuples(index=False):
            key_map.setdefault((row.metapath, row.go_id), row.go_name)

        for mp in metapaths:
            mp_dir = out_dir / _sanitize(mp)
            for (metapath, go_id), go_name in sorted(key_map.items()):
                if metapath != mp:
                    continue
                bp_dir = mp_dir / _sanitize(f"{go_id}__{go_name}")
                bp_dir.mkdir(parents=True, exist_ok=True)
                plot_metapath_bp_diff_network(
                    df_a,
                    df_b,
                    mp,
                    go_id,
                    go_name,
                    year_a,
                    year_b,
                    bp_dir,
                    label_top_genes=args.label_top_genes,
                )
        print(f"Saved diff network plots for {year_a} vs {year_b} to {out_dir}")
        return

    for year, df in year_dfs.items():
        metapath_values = set(df["metapath"].unique())
        if args.metapath:
            mp = args.metapath
            if mp not in metapath_values:
                mp_rev = reverse_metapath_abbrev(mp)
                if mp_rev in metapath_values:
                    mp = mp_rev
                else:
                    print(f"Metapath {args.metapath} not found for {year}. Skipping.")
                    continue
            metapaths = [mp]
        else:
            metapaths = sorted(metapath_values)

        for mp in metapaths:
            mp_dir = out_dir / _sanitize(mp)
            for go_id, group in df[df["metapath"] == mp].groupby("go_id"):
                go_name = group["go_name"].iloc[0] if isinstance(group["go_name"].iloc[0], str) else go_id
                bp_dir = mp_dir / _sanitize(f"{go_id}__{go_name}")
                bp_dir.mkdir(parents=True, exist_ok=True)
                plot_metapath_bp_network(df, mp, go_id, year, bp_dir, label_top_genes=args.label_top_genes)
        print(f"Saved network plots for {year} to {out_dir}")


if __name__ == "__main__":
    main()
