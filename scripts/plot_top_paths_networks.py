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


def _scale(values, min_size=0.6, max_size=3.5):
    vmin = np.min(values)
    vmax = np.max(values)
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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    top_dir = repo_root / "output" / "metapath_analysis" / "top_paths"
    out_dir = top_dir / "networks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        path = top_dir / f"top_gene_bp_pairs_{year}.csv"
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue
        df = pd.read_csv(path)
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
