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
Plot networks from enumerated path instances (local enumeration).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sanitize(name: str, max_len: int = 80) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name[:max_len] if len(name) > max_len else name


def reverse_metapath_abbrev(metapath: str) -> str:
    node_abbrevs = {"G", "BP", "CC", "MF", "PW", "A", "D", "C", "SE", "S", "PC"}
    edge_abbrevs = {"p", "i", "c", "r", ">", "<", "a", "d", "u", "e", "b", "t", "l"}
    tokens = []
    pos = 0
    while pos < len(metapath):
        if pos + 2 <= len(metapath) and metapath[pos:pos + 2] in node_abbrevs:
            tokens.append(metapath[pos:pos + 2])
            pos += 2
        elif metapath[pos] in node_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        elif metapath[pos] in edge_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        else:
            pos += 1
    direction_map = {">": "<", "<": ">"}
    reversed_tokens = []
    for token in reversed(tokens):
        reversed_tokens.append(direction_map.get(token, token))
    return "".join(reversed_tokens)


def _build_layers(paths, n_layers):
    layers = [[] for _ in range(n_layers)]
    for path in paths:
        for idx, node in enumerate(path):
            if node not in layers[idx]:
                layers[idx].append(node)
    return layers


def _build_positions(layers):
    positions = {}
    n_layers = len(layers)
    for idx, layer in enumerate(layers):
        ys = np.linspace(0, 1, len(layer)) if layer else []
        x = idx / (n_layers - 1) if n_layers > 1 else 0.5
        for node, y in zip(layer, ys):
            positions[(idx, node)] = (x, y)
    return positions


def _counts_from_paths(paths):
    counts = {}
    for path in paths:
        for idx, node in enumerate(path):
            counts[(idx, node)] = counts.get((idx, node), 0) + 1
    return counts


def _draw_network(ax, paths, layers, positions, title, label_top=8):
    counts = _counts_from_paths(paths)
    sizes = {k: 80 + 30 * v for k, v in counts.items()}

    for path in paths:
        for idx in range(len(path) - 1):
            x1, y1 = positions[(idx, path[idx])]
            x2, y2 = positions[(idx + 1, path[idx + 1])]
            ax.plot([x1, x2], [y1, y2], color="#9aa0a6", linewidth=1.2, alpha=0.6)

    for idx, layer in enumerate(layers):
        for node in layer:
            if (idx, node) not in sizes:
                sizes[(idx, node)] = 80
            x, y = positions[(idx, node)]
            color = "#4c78a8" if idx == 0 or idx == len(layers) - 1 else "#f58518"
            ax.scatter([x], [y], s=sizes[(idx, node)], c=color, edgecolors="white", linewidths=0.5, alpha=0.9)

    for idx, layer in enumerate(layers):
        layer_counts = sorted(((node, counts.get((idx, node), 0)) for node in layer), key=lambda x: x[1], reverse=True)
        for node, _ in layer_counts[:label_top]:
            x, y = positions[(idx, node)]
            ax.text(x + 0.01, y, node, fontsize=7, va="center")

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")


def plot_group_years(df_by_year, metapath, go_id, go_name, years, out_dir, label_top=8):
    year_paths = {}
    for year in years:
        df = df_by_year.get(year)
        if df is None:
            continue
        sub = df[(df["metapath"] == metapath) & (df["go_id"] == go_id)]
        if sub.empty:
            continue
        year_paths[year] = [p.split("|") for p in sub["path_nodes_names"].tolist()]

    if len(year_paths) == 0:
        return
    # Require presence in all years for side-by-side comparison
    if any(year not in year_paths for year in years):
        return

    # Build shared layers/positions across years for alignment
    all_paths = [p for paths in year_paths.values() for p in paths]
    n_layers = len(all_paths[0])
    layers = _build_layers(all_paths, n_layers)
    positions = _build_positions(layers)

    fig, axes = plt.subplots(1, len(year_paths), figsize=(6 * len(year_paths), max(5, 0.3 * max(len(l) for l in layers) + 2)))
    if len(year_paths) == 1:
        axes = [axes]

    for ax, year in zip(axes, year_paths.keys()):
        paths = year_paths[year]
        title = f"{metapath} | {year} | {go_id} {go_name}"
        _draw_network(ax, paths, layers, positions, title, label_top=label_top)

    out_dir.mkdir(parents=True, exist_ok=True)
    year_label = "_".join(str(y) for y in years)
    out_path = out_dir / f"{year_label}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot path-instance networks.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--metapath", default=None)
    parser.add_argument("--label-top", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    top_dir = repo_root / "output" / "metapath_analysis" / "top_paths"
    out_root = top_dir / "path_networks"
    out_root.mkdir(parents=True, exist_ok=True)

    year_dfs = {}
    for year in args.years:
        path = top_dir / f"path_instances_{year}.csv"
        if not path.exists():
            print(f"Missing {path}, skipping.")
            continue
        df = pd.read_csv(path)
        if args.metapath:
            mp = args.metapath
            if mp not in set(df["metapath"]):
                # Try reverse orientation
                mp_rev = reverse_metapath_abbrev(mp)
                if mp_rev in set(df["metapath"]):
                    mp = mp_rev
                else:
                    print(f"Metapath {args.metapath} not found for {year}. Skipping.")
                    continue
            df = df[df["metapath"] == mp]

        year_dfs[int(year)] = df

    if not year_dfs:
        return

    all_years = sorted(year_dfs.keys())
    keys = None
    for df in year_dfs.values():
        current = set(
            tuple(row)
            for row in df[["metapath", "go_id", "go_name"]].drop_duplicates().itertuples(index=False, name=None)
        )
        keys = current if keys is None else keys.intersection(current)
    for (metapath, go_id, go_name) in sorted(keys or []):
        mp_dir = out_root / _sanitize(metapath)
        bp_dir = mp_dir / _sanitize(f"{go_id}__{go_name}")
        plot_group_years(year_dfs, metapath, go_id, go_name, all_years, bp_dir, label_top=args.label_top)

    print(f"Saved path-instance networks to {out_root}")


if __name__ == "__main__":
    main()
