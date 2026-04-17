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


def _degree_from_paths(paths):
    degrees = {}
    for path in paths:
        for idx in range(len(path) - 1):
            n1 = (idx, path[idx])
            n2 = (idx + 1, path[idx + 1])
            degrees[n1] = degrees.get(n1, 0) + 1
            degrees[n2] = degrees.get(n2, 0) + 1
    return degrees


def _draw_network(ax, paths, layers, positions, title, label_top=0):
    counts = _counts_from_paths(paths)
    degrees = _degree_from_paths(paths)
    if degrees:
        vals = np.array(list(degrees.values()), dtype=float)
        vmin = float(vals.min())
        vmax = float(vals.max())
    else:
        vmin = vmax = 0.0
    sizes = {}
    for k, v in degrees.items():
        if vmax == vmin:
            size = 140.0
        else:
            # Inverse scaling: higher degree -> smaller node size.
            size = 260.0 - (v - vmin) / (vmax - vmin) * (260.0 - 80.0)
        sizes[k] = size

    for path in paths:
        for idx in range(len(path) - 1):
            x1, y1 = positions[(idx, path[idx])]
            x2, y2 = positions[(idx + 1, path[idx + 1])]
            ax.plot([x1, x2], [y1, y2], color="#9aa0a6", linewidth=1.2, alpha=0.6)

    for idx, layer in enumerate(layers):
        for node in layer:
            if (idx, node) not in sizes:
                sizes[(idx, node)] = 100
            x, y = positions[(idx, node)]
            color = "#4c78a8" if idx == 0 or idx == len(layers) - 1 else "#f58518"
            ax.scatter([x], [y], s=sizes[(idx, node)], c=color, edgecolors="white", linewidths=0.5, alpha=0.9)

    for idx, layer in enumerate(layers):
        layer_counts = sorted(((node, counts.get((idx, node), 0)) for node in layer), key=lambda x: x[1], reverse=True)
        if label_top and label_top > 0:
            label_nodes = [node for node, _ in layer_counts[:label_top]]
        else:
            label_nodes = [node for node, _ in layer_counts]
        for i, node in enumerate(label_nodes):
            x, y = positions[(idx, node)]
            if idx == 0:
                dx, dy, ha = -0.02, 0.0, "right"
            elif idx == len(layers) - 1:
                dx, dy, ha = 0.02, 0.0, "left"
            else:
                dy = 0.018 if i % 2 == 0 else -0.018
                dx, ha = 0.012, "left"
            ax.text(
                x + dx,
                y + dy,
                node,
                fontsize=7,
                va="center",
                ha=ha,
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
            )

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")


def plot_group_years(
    df_by_year,
    metapath,
    go_id,
    go_name,
    years,
    out_dir,
    label_top=8,
    align_nodes=True,
):
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

    plot_years = [y for y in years if y in year_paths]
    all_paths = [p for paths in year_paths.values() for p in paths]
    if not all_paths:
        return
    n_layers = len(all_paths[0])
    size_layers = _build_layers(all_paths, n_layers)
    max_layer_size = max((len(l) for l in size_layers), default=0)
    fig, axes = plt.subplots(
        1,
        len(plot_years),
        figsize=(6 * len(plot_years), max(5, 0.3 * max_layer_size + 2)),
    )
    if len(plot_years) == 1:
        axes = [axes]
    aligned_layers = None
    aligned_positions = None
    if align_nodes:
        aligned_layers = _build_layers(all_paths, n_layers)
        aligned_positions = _build_positions(aligned_layers)

    for ax, year in zip(axes, plot_years):
        paths = year_paths.get(year, [])
        title = f"{metapath} | {year} | {go_id} {go_name}"
        if not paths:
            ax.set_title(title, fontsize=11)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.axis("off")
            ax.text(0.5, 0.02, f"No paths in {year}", fontsize=8, ha="center", va="bottom")
            continue
        if align_nodes:
            layers = aligned_layers
            positions = aligned_positions
        else:
            n_layers = len(paths[0])
            layers = _build_layers(paths, n_layers)
            positions = _build_positions(layers)
        _draw_network(ax, paths, layers, positions, title, label_top=label_top)

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


def build_ranked_go_ids_by_path_change(
    df_by_year: dict[int, pd.DataFrame],
    years: list[int],
    top_n: int,
    metric: str,
) -> dict[str, set[str]]:
    if len(years) != 2:
        raise ValueError("Path-change ranking requires exactly two years.")

    year_a, year_b = years
    df_a = df_by_year.get(year_a)
    df_b = df_by_year.get(year_b)
    if df_a is None or df_b is None:
        return {}

    def _path_sets(df: pd.DataFrame) -> pd.Series:
        return df.groupby(["metapath", "go_id"])["path_nodes_ids"].apply(lambda s: set(s))

    sets_a = _path_sets(df_a)
    sets_b = _path_sets(df_b)

    records = []
    keys = set(sets_a.index).intersection(set(sets_b.index))
    for key in keys:
        paths_a = sets_a.get(key, set())
        paths_b = sets_b.get(key, set())
        if not paths_a or not paths_b:
            continue
        if metric == "jaccard":
            union = paths_a | paths_b
            if not union:
                continue
            score = 1.0 - (len(paths_a & paths_b) / len(union))
        elif metric == "count":
            score = abs(len(paths_a) - len(paths_b))
        else:
            raise ValueError(f"Unknown path-change metric: {metric}")
        records.append({"metapath": key[0], "go_id": key[1], "score": score})

    if not records:
        return {}

    df = pd.DataFrame(records)
    top = df.sort_values("score", ascending=False).groupby("metapath").head(top_n)
    allowed: dict[str, set[str]] = defaultdict(set)
    for row in top.itertuples(index=False):
        allowed[row.metapath].add(row.go_id)
    return allowed


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot path-instance networks.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--metapath", default=None)
    parser.add_argument("--label-top", type=int, default=0, help="Max labels per layer; 0 = label all.")
    parser.add_argument("--side-by-side", action="store_true", help="Side-by-side output for exactly two years.")
    parser.add_argument("--rank-by-paired-diff", action="store_true", help="Select GO terms by paired real-control diffs.")
    parser.add_argument("--rank-by-path-change", action="store_true", help="Select GO terms by path changes between years.")
    parser.add_argument("--paired-stat", default="mean_dwpc", help="Statistic to rank by (default: mean_dwpc).")
    parser.add_argument("--paired-top-n", type=int, default=10, help="Top GO terms per metapath (default: 10).")
    parser.add_argument("--paired-abs", action="store_true", help="Rank by absolute diff instead of signed.")
    parser.add_argument("--path-change-top-n", type=int, default=10, help="Top GO terms per metapath (default: 10).")
    parser.add_argument("--path-change-metric", choices=["jaccard", "count"], default="jaccard")
    parser.add_argument("--align-nodes", action="store_true", help="Align node positions across years.")
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
    if args.rank_by_paired_diff and args.rank_by_path_change:
        raise SystemExit("Choose only one ranking method: --rank-by-paired-diff or --rank-by-path-change.")
    if args.rank_by_paired_diff:
        paired_dir = repo_root / "output" / "metapath_analysis" / "divergence_scores"
        allowed_go_ids = build_ranked_go_ids(
            paired_dir,
            year_labels,
            args.paired_stat,
            args.paired_top_n,
            args.paired_abs,
        )
    if args.rank_by_path_change:
        if len(year_dfs) != 2:
            raise SystemExit("Path-change ranking requires exactly two years (e.g., --years 2016 2024).")
        allowed_go_ids = build_ranked_go_ids_by_path_change(
            year_dfs,
            all_years,
            args.path_change_top_n,
            args.path_change_metric,
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
        plot_group_years(
            year_dfs,
            metapath,
            go_id,
            go_name,
            all_years,
            bp_dir,
            label_top=args.label_top,
            align_nodes=args.align_nodes,
        )

    print(f"Saved path-instance networks to {out_root}")


if __name__ == "__main__":
    main()
