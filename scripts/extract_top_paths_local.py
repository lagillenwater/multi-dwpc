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
Enumerate top path instances locally (no API) for visualization.

Uses direct DWPC outputs and adjacency matrices in data/edges/.
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


NODE_FILES = {
    "G": "Gene.tsv",
    "BP": "Biological Process.tsv",
    "CC": "Cellular Component.tsv",
    "MF": "Molecular Function.tsv",
    "PW": "Pathway.tsv",
    "A": "Anatomy.tsv",
    "D": "Disease.tsv",
    "C": "Compound.tsv",
    "S": "Symptom.tsv",
    "SE": "Side Effect.tsv",
    "PC": "Pharmacologic Class.tsv",
}


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


def parse_metapath(metapath: str) -> Tuple[List[str], List[str]]:
    node_abbrevs = ["BP", "CC", "MF", "PW", "SE", "PC", "G", "A", "D", "C", "S"]
    node_abbrevs = sorted(node_abbrevs, key=len, reverse=True)
    nodes: List[str] = []
    edges: List[str] = []
    i = 0
    edge_token = ""
    while i < len(metapath):
        matched = False
        for ab in node_abbrevs:
            if metapath.startswith(ab, i):
                if nodes:
                    edges.append(edge_token)
                    edge_token = ""
                nodes.append(ab)
                i += len(ab)
                matched = True
                break
        if not matched:
            edge_token += metapath[i]
            i += 1
    return nodes, edges


@dataclass
class NodeMaps:
    id_to_pos: Dict[str, Dict[str, int]]
    pos_to_id: Dict[str, Dict[int, str]]
    id_to_name: Dict[str, Dict[str, str]]


def load_node_maps(repo_root: Path, node_types: List[str]) -> NodeMaps:
    id_to_pos = {}
    pos_to_id = {}
    id_to_name = {}
    for node_type in node_types:
        filename = NODE_FILES[node_type]
        df = pd.read_csv(repo_root / "data" / "nodes" / filename, sep="\t")
        id_to_pos[node_type] = dict(zip(df["identifier"], df["position"]))
        pos_to_id[node_type] = dict(zip(df["position"], df["identifier"]))
        id_to_name[node_type] = dict(zip(df["identifier"], df["name"]))
    return NodeMaps(id_to_pos, pos_to_id, id_to_name)


class EdgeLoader:
    def __init__(self, edges_dir: Path):
        self.edges_dir = edges_dir
        self.edge_files = {p.name for p in edges_dir.glob("*.sparse.npz")}
        self.cache = {}
        self.deg_cache = {}

    def _canonical_edge_token(self, edge_token: str) -> Tuple[str, bool]:
        reverse = "<" in edge_token and ">" not in edge_token
        edge_base = edge_token.replace("<", "").replace(">", "")
        if "<" in edge_token or ">" in edge_token:
            file_edge = f"{edge_base}>"
        else:
            file_edge = edge_base
        return file_edge, reverse

    def load(self, src: str, edge_token: str, dst: str) -> sparse.csr_matrix:
        key = (src, edge_token, dst)
        if key in self.cache:
            return self.cache[key]

        file_edge, reverse = self._canonical_edge_token(edge_token)
        name_direct = f"{src}{file_edge}{dst}.sparse.npz"
        name_reverse = f"{dst}{file_edge}{src}.sparse.npz"

        if name_direct in self.edge_files:
            mat = sparse.load_npz(self.edges_dir / name_direct)
            if reverse:
                mat = mat.T
        elif name_reverse in self.edge_files:
            mat = sparse.load_npz(self.edges_dir / name_reverse).T
            if reverse:
                mat = mat.T
        else:
            raise FileNotFoundError(f"No edge matrix for {src}{edge_token}{dst}")

        mat = mat.tocsr()
        self.cache[key] = mat
        return mat

    def degree(self, src: str, edge_token: str, dst: str) -> np.ndarray:
        key = (src, edge_token, dst)
        if key in self.deg_cache:
            return self.deg_cache[key]
        mat = self.load(src, edge_token, dst)
        deg = np.asarray(mat.getnnz(axis=1)).astype(float)
        self.deg_cache[key] = deg
        return deg


def enumerate_paths(
    bp_pos: int,
    gene_pos: int,
    nodes: List[str],
    edges: List[str],
    edge_loader: EdgeLoader,
    top_k: int,
    degree_d: float = 0.5,
) -> List[Tuple[float, List[int]]]:
    num_edges = len(edges)
    if num_edges == 1:
        A0 = edge_loader.load(nodes[0], edges[0], nodes[1])
        if A0[bp_pos, gene_pos] == 0:
            return []
        return [(1.0, [bp_pos, gene_pos])]

    A0 = edge_loader.load(nodes[0], edges[0], nodes[1])
    neighbors1 = A0.getrow(bp_pos).indices

    if num_edges == 2:
        A1 = edge_loader.load(nodes[1], edges[1], nodes[2])
        deg1 = edge_loader.degree(nodes[1], edges[1], nodes[2])
        targets = set(A1[:, gene_pos].nonzero()[0])
        heap = []
        for n1 in neighbors1:
            if n1 in targets:
                score = 1.0 / math.sqrt(max(deg1[n1], 1.0)) ** degree_d
                heapq.heappush(heap, (score, [bp_pos, n1, gene_pos]))
        return sorted(heap, key=lambda x: x[0], reverse=True)[:top_k]

    if num_edges == 3:
        A1 = edge_loader.load(nodes[1], edges[1], nodes[2])
        A2 = edge_loader.load(nodes[2], edges[2], nodes[3])
        deg1 = edge_loader.degree(nodes[1], edges[1], nodes[2])
        deg2 = edge_loader.degree(nodes[2], edges[2], nodes[3])
        targets = set(A2[:, gene_pos].nonzero()[0])

        heap: List[Tuple[float, List[int]]] = []
        for n1 in neighbors1:
            n2s = A1.getrow(n1).indices
            if len(n2s) == 0:
                continue
            for n2 in n2s:
                if n2 in targets:
                    score = 1.0 / (
                        (math.sqrt(max(deg1[n1], 1.0)) ** degree_d)
                        * (math.sqrt(max(deg2[n2], 1.0)) ** degree_d)
                    )
                    if len(heap) < top_k:
                        heapq.heappush(heap, (score, [bp_pos, n1, n2, gene_pos]))
                    else:
                        if score > heap[0][0]:
                            heapq.heapreplace(heap, (score, [bp_pos, n1, n2, gene_pos]))
        return sorted(heap, key=lambda x: x[0], reverse=True)

    raise ValueError(f"Metapath length {num_edges} not supported for enumeration.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top path instances locally.")
    parser.add_argument("--years", nargs="+", default=["2016", "2024"])
    parser.add_argument("--top-pairs", type=int, default=5)
    parser.add_argument("--top-paths", type=int, default=5)
    parser.add_argument("--metapath", default=None, help="Optional metapath filter (BP-first or G-first).")
    parser.add_argument("--degree-d", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    base_name = "all_GO_positive_growth"
    top_dir = repo_root / "output" / "metapath_analysis" / "top_paths"
    results_dir = repo_root / "output" / "dwpc_direct" / base_name / "results"
    edges_dir = repo_root / "data" / "edges"

    edge_loader = EdgeLoader(edges_dir)

    for year in args.years:
        top_pairs_path = top_dir / f"top_gene_bp_pairs_{year}.csv"
        if not top_pairs_path.exists():
            print(f"Missing {top_pairs_path}, skipping.")
            continue
        df = pd.read_csv(top_pairs_path)

        if args.metapath:
            mp = args.metapath
            if mp not in set(df["metapath"]):
                mp_rev = reverse_metapath_abbrev(mp)
                if mp_rev in set(df["metapath"]):
                    mp = mp_rev
                else:
                    print(f"Metapath {args.metapath} not found for {year}. Skipping.")
                    continue
            df = df[df["metapath"] == mp]

        df = df.sort_values("dwpc", ascending=False)
        df = df.groupby("metapath").head(args.top_pairs).copy()

        output_rows = []
        for _, row in df.iterrows():
            metapath_g = row["metapath"]
            metapath_bp = reverse_metapath_abbrev(metapath_g)
            nodes, edges = parse_metapath(metapath_bp)

            node_types = list(dict.fromkeys(nodes))
            maps = load_node_maps(repo_root, node_types)

            bp_id = row["go_id"]
            gene_id = row["entrez_gene_id"]
            bp_pos = maps.id_to_pos["BP"].get(bp_id)
            gene_pos = maps.id_to_pos["G"].get(gene_id)
            if bp_pos is None or gene_pos is None:
                continue

            try:
                paths = enumerate_paths(
                    bp_pos,
                    gene_pos,
                    nodes,
                    edges,
                    edge_loader,
                    top_k=args.top_paths,
                    degree_d=args.degree_d,
                )
            except Exception as exc:
                print(f"Failed {metapath_bp} {bp_id} {gene_id}: {exc}")
                continue

            for rank, (score, pos_path) in enumerate(paths, start=1):
                id_path = []
                name_path = []
                for node_type, pos in zip(nodes, pos_path):
                    node_id = maps.pos_to_id[node_type].get(int(pos))
                    node_name = maps.id_to_name[node_type].get(node_id, str(node_id))
                    id_path.append(str(node_id))
                    name_path.append(str(node_name))

                output_rows.append(
                    {
                        "year": int(year),
                        "metapath": metapath_bp,
                        "metapath_g_orientation": metapath_g,
                        "go_id": bp_id,
                        "go_name": row.get("go_name"),
                        "entrez_gene_id": gene_id,
                        "gene_name": row.get("gene_name"),
                        "path_rank": rank,
                        "path_score": score,
                        "path_nodes_ids": "|".join(id_path),
                        "path_nodes_names": "|".join(name_path),
                    }
                )

        out_path = top_dir / f"path_instances_{year}.csv"
        pd.DataFrame(output_rows).to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
