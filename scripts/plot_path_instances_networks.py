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
from collections import defaultdict

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


def _normalize_metapath_bp(metapath: str) -> str:
    if metapath.startswith("BP"):
        return metapath
    return reverse_metapath_abbrev(metapath)


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
            year_paths[year] = []
            continue
        year_paths[year] = [p.split("|") for p in sub["path_nodes_names"].tolist()]

    if not year_paths:
        return

    # Build shared layers/positions across years for alignment (union of paths)
    all_paths = [p for paths in year_paths.values() for p in paths]
    if not all_paths:
        return
    n_layers = len(all_paths[0])
    layers = _build_layers(all_paths, n_layers)
    positions = _build_positions(layers)

    plot_years = [y for y in years if y in year_paths]
    fig, axes = plt.subplots(1, len(plot_years), figsize=(6 * len(plot_years), max(5, 0.3 * max(len(l) for l in layers) + 2)))
    if len(plot_years) == 1:
        axes = [axes]

    for ax, year in zip(axes, plot_years):
        paths = year_paths.get(year, [])
        title = f"{metapath} | {year} | {go_id} {go_name}"
        _draw_network(ax, paths, layers, positions, title, label_top=label_top)
        if not paths:
            ax.text(0.5, 0.02, f"No paths in {year}", fontsize=8, ha="center", va="bottom")

    out_dir.mkdir(parents=True, exist_ok=True)
    year_label = "_".join(str(y) for y in years)
    out_path = out_dir / f"{year_label}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_paired_diff(paired_dir: Path, year: str, control: str, stat: str) -> pd.DataFrame:
    path = paired_dir / f"paired_go_terms_{year}_vs_{control}.csv"
    if not path.exists():
        return pd.DataFrame()
    diff_col = f"{stat}_diff"
    usecols = ["go_id", "metapath", diff_col]
    df = pd.read_csv(path, usecols=usecols)
    return df.rename(columns={diff_col: f"{control}_diff"})


def build_ranked_go_ids(
    paired_dir: Path,
    years: list[str],
    stat: str,
    top_n: int,
    use_abs: bool,
) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = defaultdict(set)
    for year in years:
        perm = _load_paired_diff(paired_dir, year, "perm", stat)
        rand = _load_paired_diff(paired_dir, year, "random", stat)
        if perm.empty and rand.empty:
            continue

        if perm.empty:
            merged = rand
        elif rand.empty:
            merged = perm
        else:
            merged = perm.merge(rand, on=["go_id", "metapath"], how="outer")

        if "perm_diff" in merged.columns and "random_diff" in merged.columns:
            merged["diff"] = merged[["perm_diff", "random_diff"]].mean(axis=1, skipna=True)
        elif "perm_diff" in merged.columns:
            merged["diff"] = merged["perm_diff"]
        else:
            merged["diff"] = merged["random_diff"]

        merged = merged.dropna(subset=["diff"])
        if merged.empty:
            continue

        merged["rank_value"] = merged["diff"].abs() if use_abs else merged["diff"]
        merged["metapath_bp"] = merged["metapath"].map(_normalize_metapath_bp)
        top = merged.sort_values("rank_value", ascending=False).groupby("metapath_bp").head(top_n)
        for row in top.itertuples(index=False):
            allowed[row.metapath_bp].add(row.go_id)

    return allowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot path-instance networks.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--metapath", default=None)
    parser.add_argument("--label-top", type=int, default=8)
    parser.add_argument("--side-by-side", action="store_true", help="Side-by-side output for exactly two years.")
    parser.add_argument("--rank-by-paired-diff", action="store_true", help="Select GO terms by paired real-control diffs.")
    parser.add_argument("--paired-stat", default="mean_dwpc", help="Statistic to rank by (default: mean_dwpc).")
    parser.add_argument("--paired-top-n", type=int, default=10, help="Top GO terms per metapath (default: 10).")
    parser.add_argument("--paired-abs", action="store_true", help="Rank by absolute diff instead of signed.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    top_dir = repo_root / "output" / "metapath_analysis" / "top_paths"
    out_root = top_dir / ("path_networks_side_by_side" if args.side_by_side else "path_networks")
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

    if args.side_by_side and len(args.years) != 2:
        raise SystemExit("Side-by-side mode requires exactly two years (e.g., --years 2016 2024).")

    all_years = sorted(year_dfs.keys())
    year_labels = [str(y) for y in all_years]
    allowed_go_ids = None
    if args.rank_by_paired_diff:
        paired_dir = repo_root / "output" / "metapath_analysis" / "divergence_scores"
        allowed_go_ids = build_ranked_go_ids(
            paired_dir,
            year_labels,
            args.paired_stat,
            args.paired_top_n,
            args.paired_abs,
        )

    key_map = {}
    for df in year_dfs.values():
        for row in df[["metapath", "go_id", "go_name"]].drop_duplicates().itertuples(index=False):
            key = (row.metapath, row.go_id)
            if key not in key_map or not key_map[key]:
                key_map[key] = row.go_name

    for (metapath, go_id), go_name in sorted(key_map.items()):
        if allowed_go_ids is not None:
            allowed = allowed_go_ids.get(metapath, set())
            if go_id not in allowed:
                continue
        mp_dir = out_root / _sanitize(metapath)
        bp_dir = mp_dir / _sanitize(f"{go_id}__{go_name}")
        plot_group_years(year_dfs, metapath, go_id, go_name, all_years, bp_dir, label_top=args.label_top)

    print(f"Saved path-instance networks to {out_root}")


if __name__ == "__main__":
    main()
