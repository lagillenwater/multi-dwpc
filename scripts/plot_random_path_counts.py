#!/usr/bin/env python3
"""
Sample random node pairs and plot exact simple-path counts by path length.

The script builds an undirected graph from the HetMat sparse edge files, samples
node pairs from the largest connected component, counts exact simple paths of
lengths 2-6 between each pair, and writes both CSV outputs and a summary plot.

By default, pair sampling excludes the top 50% highest-degree nodes to keep
exact length-6 counting tractable. Set --degree-percentile-cap 100 to sample
from the full node set.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


_ADJACENCY: Optional[Sequence[Tuple[int, ...]]] = None
_REVERSE_ADJACENCY: Optional[Sequence[Tuple[int, ...]]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="HetMat data directory containing metagraph.json, nodes/, and edges/.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/random_path_count_sampling",
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=20,
        help="Number of random node pairs to sample.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum exact path length to count.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=6,
        help="Maximum exact path length to count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed used for pair sampling.",
    )
    parser.add_argument(
        "--degree-percentile-cap",
        type=float,
        default=50.0,
        help=(
            "Sample only from nodes at or below this degree percentile. "
            "Use 100 to include all nodes."
        ),
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat sparse edge files as directed instead of collapsing to an undirected graph.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for path counting. "
            "Defaults to min(num_pairs, available CPUs)."
        ),
    )
    return parser.parse_args()


def load_metagraph(data_dir: Path) -> Dict:
    with open(data_dir / "metagraph.json") as handle:
        return json.load(handle)


def get_abbrev_maps(metagraph: Dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    kind_to_abbrev = metagraph["kind_to_abbrev"]
    abbrev_to_kind = {abbrev: kind for kind, abbrev in kind_to_abbrev.items()}
    return kind_to_abbrev, abbrev_to_kind


def sorted_abbrevs(abbrev_to_kind: Dict[str, str]) -> List[str]:
    return sorted(abbrev_to_kind, key=len, reverse=True)


def parse_edge_endpoints(stem: str, abbrevs: Sequence[str]) -> Tuple[str, str]:
    for src in abbrevs:
        if not stem.startswith(src):
            continue
        remainder = stem[len(src):]
        for dst in abbrevs:
            if remainder.endswith(dst):
                return src, dst
    raise ValueError(f"Unable to parse edge endpoints from {stem}")


def load_nodes(
    data_dir: Path,
    metagraph: Dict,
    kind_to_abbrev: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    records: List[Dict[str, object]] = []
    offsets: Dict[str, int] = {}
    offset = 0

    for kind in metagraph["metanode_kinds"]:
        abbrev = kind_to_abbrev[kind]
        node_path = data_dir / "nodes" / f"{kind}.tsv"
        node_df = pd.read_csv(node_path, sep="\t")
        node_df = node_df.sort_values("position").reset_index(drop=True)
        offsets[abbrev] = offset

        for row in node_df.itertuples(index=False):
            records.append(
                {
                    "global_id": offset + int(row.position),
                    "kind": kind,
                    "abbrev": abbrev,
                    "identifier": str(row.identifier),
                    "name": str(row.name),
                    "position": int(row.position),
                }
            )
        offset += len(node_df)

    return pd.DataFrame.from_records(records), offsets


def build_graph(
    data_dir: Path,
    node_df: pd.DataFrame,
    offsets: Dict[str, int],
    abbrevs: Sequence[str],
    directed: bool,
) -> Tuple[List[Set[int]], List[Set[int]]]:
    adjacency_sets = [set() for _ in range(len(node_df))]
    reverse_sets = [set() for _ in range(len(node_df))]
    edges_dir = data_dir / "edges"

    for edge_path in sorted(edges_dir.glob("*.sparse.npz")):
        src_abbrev, dst_abbrev = parse_edge_endpoints(edge_path.stem.replace(".sparse", ""), abbrevs)
        matrix = sparse.load_npz(edge_path).tocoo()
        src_offset = offsets[src_abbrev]
        dst_offset = offsets[dst_abbrev]

        for row, col in zip(matrix.row, matrix.col):
            src = src_offset + int(row)
            dst = dst_offset + int(col)
            if src == dst:
                continue
            adjacency_sets[src].add(dst)
            reverse_sets[dst].add(src)
            if not directed:
                adjacency_sets[dst].add(src)
                reverse_sets[src].add(dst)

    adjacency = [tuple(sorted(neighbors)) for neighbors in adjacency_sets]
    reverse_adjacency = [tuple(sorted(neighbors)) for neighbors in reverse_sets]
    return adjacency, reverse_adjacency


def largest_component(adjacency: Sequence[Set[int]]) -> Set[int]:
    seen: Set[int] = set()
    largest: Set[int] = set()

    for start in range(len(adjacency)):
        if start in seen or not adjacency[start]:
            continue

        queue = deque([start])
        component = {start}
        seen.add(start)

        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                component.add(neighbor)
                queue.append(neighbor)

        if len(component) > len(largest):
            largest = component

    return largest


def sample_pairs(
    node_df: pd.DataFrame,
    adjacency: Sequence[Set[int]],
    num_pairs: int,
    seed: int,
    degree_percentile_cap: float,
    directed: bool,
    min_length: int,
    max_length: int,
) -> List[Tuple[int, int]]:
    if not 0 < degree_percentile_cap <= 100:
        raise ValueError("--degree-percentile-cap must be in (0, 100].")

    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=int)
    eligible = degrees > 0

    if not directed:
        component = largest_component(adjacency)
        if not component:
            raise ValueError("No connected component with edges was found.")
        component_mask = node_df["global_id"].isin(component).to_numpy()
        eligible &= component_mask

    if degree_percentile_cap < 100:
        degree_cap = int(np.percentile(degrees[eligible], degree_percentile_cap))
        eligible &= degrees <= degree_cap
    else:
        degree_cap = int(degrees.max())

    candidates = node_df.loc[eligible, "global_id"].to_numpy(dtype=int)
    if len(candidates) < 2:
        raise ValueError("Not enough eligible nodes to sample random pairs.")

    max_pairs = len(candidates) * (len(candidates) - 1) // 2
    if num_pairs > max_pairs:
        raise ValueError(f"Requested {num_pairs} pairs but only {max_pairs} unique pairs are available.")

    rng = np.random.default_rng(seed)
    sampled: Set[Tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(2000, num_pairs * 500)

    while len(sampled) < num_pairs and attempts < max_attempts:
        src = int(rng.choice(candidates))
        distances = bounded_distances(adjacency, src, max_length)
        reachable = [
            int(node_id)
            for node_id, distance in distances.items()
            if node_id != src and min_length <= distance <= max_length and eligible[node_id]
        ]
        if not reachable:
            attempts += 1
            continue

        dst = int(rng.choice(reachable))
        pair = (src, dst)
        if not directed:
            pair = tuple(sorted(pair))
        sampled.add(pair)
        attempts += 1

    if len(sampled) < num_pairs:
        raise RuntimeError(
            f"Unable to sample {num_pairs} unique pairs after {attempts} attempts. "
            f"Eligible nodes: {len(candidates)}, degree cap: {degree_cap}."
        )

    print(
        f"Sampled {len(sampled)} pairs from {len(candidates)} eligible nodes "
        f"(degree cap <= {degree_cap}, directed={directed})."
    )
    return sorted(sampled)


def bounded_distances(adjacency: Sequence[Set[int]], target: int, max_depth: int) -> Dict[int, int]:
    distances = {target: 0}
    queue = deque([target])

    while queue:
        node = queue.popleft()
        depth = distances[node]
        if depth == max_depth:
            continue
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = depth + 1
            queue.append(neighbor)

    return distances


def count_simple_paths(
    adjacency: Sequence[Set[int]],
    reverse_adjacency: Sequence[Set[int]],
    source: int,
    target: int,
    min_length: int,
    max_length: int,
) -> Dict[int, int]:
    if source == target:
        raise ValueError("Source and target must differ for path counting.")

    distances = bounded_distances(reverse_adjacency, target, max_length)
    counts = {length: 0 for length in range(min_length, max_length + 1)}
    visited = {source}

    def dfs(node: int, depth: int) -> None:
        remaining = max_length - depth
        distance_to_target = distances.get(node)
        if distance_to_target is None or distance_to_target > remaining:
            return

        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue

            next_depth = depth + 1
            if neighbor == target:
                if min_length <= next_depth <= max_length:
                    counts[next_depth] += 1
                continue

            if next_depth >= max_length:
                continue

            neighbor_distance = distances.get(neighbor)
            if neighbor_distance is None or neighbor_distance > (max_length - next_depth):
                continue

            visited.add(neighbor)
            dfs(neighbor, next_depth)
            visited.remove(neighbor)

    dfs(source, 0)
    return counts


def _init_worker(
    adjacency: Sequence[Tuple[int, ...]],
    reverse_adjacency: Sequence[Tuple[int, ...]],
) -> None:
    global _ADJACENCY, _REVERSE_ADJACENCY
    _ADJACENCY = adjacency
    _REVERSE_ADJACENCY = reverse_adjacency


def _count_pair_worker(task: Tuple[int, str, int, int, int, int]) -> List[Dict[str, object]]:
    pair_id, pair_label, source, target, min_length, max_length = task
    if _ADJACENCY is None or _REVERSE_ADJACENCY is None:
        raise RuntimeError("Worker graph state is not initialized.")

    counts = count_simple_paths(
        adjacency=_ADJACENCY,
        reverse_adjacency=_REVERSE_ADJACENCY,
        source=source,
        target=target,
        min_length=min_length,
        max_length=max_length,
    )
    return [
        {
            "pair_id": pair_id,
            "pair_label": pair_label,
            "path_length": int(length),
            "path_count": int(value),
        }
        for length, value in counts.items()
    ]


def _count_pair_worker_with_id(
    task: Tuple[int, str, int, int, int, int]
) -> Tuple[int, List[Dict[str, object]]]:
    return task[0], _count_pair_worker(task)


def make_pair_records(node_df: pd.DataFrame, pairs: Iterable[Tuple[int, int]]) -> pd.DataFrame:
    nodes = node_df.set_index("global_id")
    records: List[Dict[str, object]] = []

    for pair_id, (source, target) in enumerate(pairs, start=1):
        source_row = nodes.loc[source]
        target_row = nodes.loc[target]
        records.append(
            {
                "pair_id": pair_id,
                "source_global_id": source,
                "source_kind": source_row["kind"],
                "source_abbrev": source_row["abbrev"],
                "source_identifier": source_row["identifier"],
                "source_name": source_row["name"],
                "target_global_id": target,
                "target_kind": target_row["kind"],
                "target_abbrev": target_row["abbrev"],
                "target_identifier": target_row["identifier"],
                "target_name": target_row["name"],
                "pair_label": (
                    f"P{pair_id}: {source_row['abbrev']}:{source_row['identifier']} "
                    f"to {target_row['abbrev']}:{target_row['identifier']}"
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def make_count_records(
    pair_df: pd.DataFrame,
    adjacency: Sequence[Set[int]],
    reverse_adjacency: Sequence[Set[int]],
    min_length: int,
    max_length: int,
    workers: int,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    tasks = [
        (
            int(row.pair_id),
            row.pair_label,
            int(row.source_global_id),
            int(row.target_global_id),
            min_length,
            max_length,
        )
        for row in pair_df.itertuples(index=False)
    ]
    pair_lookup = {
        int(row.pair_id): (
            row.source_abbrev,
            row.source_identifier,
            row.target_abbrev,
            row.target_identifier,
        )
        for row in pair_df.itertuples(index=False)
    }

    if workers <= 1:
        _init_worker(adjacency=adjacency, reverse_adjacency=reverse_adjacency)
        for task in tasks:
            pair_records = _count_pair_worker(task)
            records.extend(pair_records)
            pair_id = task[0]
            source_abbrev, source_identifier, target_abbrev, target_identifier = pair_lookup[pair_id]
            print(
                f"Counted paths for pair {pair_id:02d} "
                f"({source_abbrev}:{source_identifier} -> "
                f"{target_abbrev}:{target_identifier})."
            )
    else:
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = multiprocessing.get_context()

        with mp_context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(adjacency, reverse_adjacency),
        ) as pool:
            for pair_id, pair_records in pool.imap_unordered(_count_pair_worker_with_id, tasks):
                records.extend(pair_records)
                source_abbrev, source_identifier, target_abbrev, target_identifier = pair_lookup[pair_id]
                print(
                    f"Counted paths for pair {pair_id:02d} "
                    f"({source_abbrev}:{source_identifier} -> "
                    f"{target_abbrev}:{target_identifier})."
                )

    return pd.DataFrame.from_records(records).sort_values(["pair_id", "path_length"]).reset_index(drop=True)


def plot_counts(count_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, count_df["pair_id"].nunique()))

    for color, (pair_id, pair_rows) in zip(colors, count_df.groupby("pair_id")):
        pair_rows = pair_rows.sort_values("path_length")
        ax.plot(
            pair_rows["path_length"],
            pair_rows["path_count"],
            marker="o",
            linewidth=1.5,
            alpha=0.7,
            color=color,
            label=f"P{pair_id}",
        )

    mean_counts = (
        count_df.groupby("path_length", as_index=False)["path_count"]
        .mean()
        .rename(columns={"path_count": "mean_path_count"})
    )
    ax.plot(
        mean_counts["path_length"],
        mean_counts["mean_path_count"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        marker="o",
        label="Mean",
    )

    ax.set_title(
        f"Exact simple-path counts for {count_df['pair_id'].nunique()} random node pairs"
    )
    ax.set_xlabel("Path length")
    ax.set_ylabel("Number of paths")
    ax.set_xticks(sorted(count_df["path_length"].unique()))
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.min_length < 1:
        raise ValueError("--min-length must be at least 1.")
    if args.max_length < args.min_length:
        raise ValueError("--max-length must be >= --min-length.")
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = (repo_root / args.data_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metagraph = load_metagraph(data_dir)
    kind_to_abbrev, abbrev_to_kind = get_abbrev_maps(metagraph)
    abbrevs = sorted_abbrevs(abbrev_to_kind)

    node_df, offsets = load_nodes(data_dir, metagraph, kind_to_abbrev)
    print("Loading HetMat nodes and edges...")
    adjacency, reverse_adjacency = build_graph(
        data_dir=data_dir,
        node_df=node_df,
        offsets=offsets,
        abbrevs=abbrevs,
        directed=args.directed,
    )
    print(f"Built graph with {len(node_df)} nodes.")
    workers = args.workers or min(args.num_pairs, os.cpu_count() or 1)
    pairs = sample_pairs(
        node_df=node_df,
        adjacency=adjacency,
        num_pairs=args.num_pairs,
        seed=args.seed,
        degree_percentile_cap=args.degree_percentile_cap,
        directed=args.directed,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    pair_df = make_pair_records(node_df=node_df, pairs=pairs)
    print(f"Counting paths with {workers} worker(s)...")
    count_df = make_count_records(
        pair_df=pair_df,
        adjacency=adjacency,
        reverse_adjacency=reverse_adjacency,
        min_length=args.min_length,
        max_length=args.max_length,
        workers=workers,
    )

    counts_path = output_dir / "random_path_counts_by_length.csv"
    pairs_path = output_dir / "random_path_count_pairs.csv"
    figure_path = output_dir / "random_path_counts_by_length.png"

    pair_df.to_csv(pairs_path, index=False)
    count_df.to_csv(counts_path, index=False)
    plot_counts(count_df=count_df, output_path=figure_path)

    print(f"Wrote pair metadata to {pairs_path}")
    print(f"Wrote path counts to {counts_path}")
    print(f"Wrote plot to {figure_path}")


if __name__ == "__main__":
    main()
