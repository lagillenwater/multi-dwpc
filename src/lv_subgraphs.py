"""
Top-subgraph extraction and plotting for LV analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from src.dwpc_direct import reverse_metapath_abbrev
from src.lv_dwpc import compute_real_pair_dwpc
from src.lv_pairs import build_lv_target_pairs


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


def _sanitize(value: str, max_len: int = 100) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text[:max_len] if len(text) > max_len else text


def _parse_metapath(metapath: str) -> Tuple[List[str], List[str]]:
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


def _load_node_maps(repo_root: Path, node_types: List[str]) -> NodeMaps:
    id_to_pos = {}
    pos_to_id = {}
    id_to_name = {}
    for node_type in node_types:
        filename = NODE_FILES[node_type]
        df = pd.read_csv(repo_root / "data" / "nodes" / filename, sep="\t")
        id_to_pos[node_type] = dict(zip(df["identifier"].astype(str), df["position"]))
        pos_to_id[node_type] = dict(zip(df["position"], df["identifier"].astype(str)))
        id_to_name[node_type] = dict(zip(df["identifier"].astype(str), df["name"].astype(str)))
    return NodeMaps(id_to_pos, pos_to_id, id_to_name)


class EdgeLoader:
    def __init__(self, edges_dir: Path):
        self.edges_dir = edges_dir
        self.edge_files = {p.name for p in edges_dir.glob("*.sparse.npz")}
        self.cache: Dict[Tuple[str, str, str], sparse.csr_matrix] = {}
        self.deg_cache: Dict[Tuple[str, str, str], np.ndarray] = {}

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


def _enumerate_paths(
    source_pos: int,
    gene_pos: int,
    nodes: List[str],
    edges: List[str],
    edge_loader: EdgeLoader,
    top_k: int,
    degree_d: float = 0.5,
) -> List[Tuple[float, List[int]]]:
    num_edges = len(edges)
    if num_edges == 1:
        mat = edge_loader.load(nodes[0], edges[0], nodes[1])
        if mat[source_pos, gene_pos] == 0:
            return []
        return [(1.0, [source_pos, gene_pos])]

    mat0 = edge_loader.load(nodes[0], edges[0], nodes[1])
    nbr1 = mat0.getrow(source_pos).indices

    if num_edges == 2:
        mat1 = edge_loader.load(nodes[1], edges[1], nodes[2])
        deg1 = edge_loader.degree(nodes[1], edges[1], nodes[2])
        targets = set(mat1[:, gene_pos].nonzero()[0])
        rows = []
        for n1 in nbr1:
            if n1 in targets:
                score = 1.0 / (np.sqrt(max(float(deg1[n1]), 1.0)) ** degree_d)
                rows.append((score, [source_pos, n1, gene_pos]))
        return sorted(rows, key=lambda x: x[0], reverse=True)[:top_k]

    if num_edges == 3:
        mat1 = edge_loader.load(nodes[1], edges[1], nodes[2])
        mat2 = edge_loader.load(nodes[2], edges[2], nodes[3])
        deg1 = edge_loader.degree(nodes[1], edges[1], nodes[2])
        deg2 = edge_loader.degree(nodes[2], edges[2], nodes[3])
        targets = set(mat2[:, gene_pos].nonzero()[0])
        heap: List[Tuple[float, List[int]]] = []
        for n1 in nbr1:
            n2s = mat1.getrow(n1).indices
            for n2 in n2s:
                if n2 in targets:
                    score = 1.0 / (
                        (np.sqrt(max(float(deg1[n1]), 1.0)) ** degree_d)
                        * (np.sqrt(max(float(deg2[n2]), 1.0)) ** degree_d)
                    )
                    heap.append((score, [source_pos, n1, n2, gene_pos]))
        return sorted(heap, key=lambda x: x[0], reverse=True)[:top_k]

    return []


def extract_top_subgraphs(
    repo_root: Path,
    output_dir: Path,
    top_metapaths: int,
    top_pairs: int,
    top_paths: int,
    damping: float = 0.5,
    degree_d: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create top pair and path-instance outputs for supported metapaths.
    """
    results_path = output_dir / "lv_metapath_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing stats file: {results_path}")

    results = pd.read_csv(results_path)
    supported = results[results["supported"] == True].copy()  # noqa: E712
    if supported.empty:
        supported = results.copy()

    supported["fdr_sum"] = supported["p_perm_fdr"] + supported["p_rand_fdr"]
    supported = supported.sort_values(
        ["lv_id", "target_set_id", "min_d", "min_diff", "fdr_sum"],
        ascending=[True, True, False, False, True],
    )
    top_mp = supported.groupby(["lv_id", "target_set_id"]).head(top_metapaths).copy()

    manifest_override = top_mp[["node_type", "metapath"]].drop_duplicates().copy()

    pairs = build_lv_target_pairs(
        top_genes_path=output_dir / "lv_top_genes.csv",
        target_sets_path=output_dir / "target_sets.csv",
        lv_target_map_path=output_dir / "lv_target_map.csv",
        output_pairs_path=output_dir / "lv_gene_target_pairs.csv",
    )

    pair_dwpc, _ = compute_real_pair_dwpc(
        pairs_path=output_dir / "lv_gene_target_pairs.csv",
        data_dir=repo_root / "data",
        metapath_stats_path=repo_root / "data" / "metapath-dwpc-stats.tsv",
        output_dwpc_path=output_dir / "lv_pair_dwpc_top_selected.csv",
        output_metapath_manifest_path=output_dir / "lv_metapaths_top_selected.csv",
        damping=damping,
        max_length=3,
        exclude_direct=True,
        manifest_override=manifest_override,
        use_disk_cache=True,
    )

    keep_cols = ["lv_id", "target_set_id", "metapath"]
    filtered_pairs = pair_dwpc.merge(top_mp[keep_cols], on=keep_cols, how="inner")
    filtered_pairs = filtered_pairs.sort_values(
        ["lv_id", "target_set_id", "metapath", "dwpc"],
        ascending=[True, True, True, False],
    )
    top_pairs_df = (
        filtered_pairs.groupby(["lv_id", "target_set_id", "metapath"]).head(top_pairs).copy()
    )
    top_pairs_df["pair_rank"] = top_pairs_df.groupby(
        ["lv_id", "target_set_id", "metapath"]
    ).cumcount() + 1
    top_pairs_df.to_csv(output_dir / "top_pairs.csv", index=False)

    edge_loader = EdgeLoader(repo_root / "data" / "edges")
    path_rows = []
    for row in top_pairs_df.itertuples(index=False):
        metapath_g = row.metapath
        metapath_src_to_gene = reverse_metapath_abbrev(metapath_g)
        nodes, edges = _parse_metapath(metapath_src_to_gene)
        node_types = list(dict.fromkeys(nodes))
        maps = _load_node_maps(repo_root, node_types)

        source_id = str(row.target_id)
        gene_id = str(row.gene_identifier)
        source_pos = maps.id_to_pos[nodes[0]].get(source_id)
        gene_pos = maps.id_to_pos["G"].get(gene_id)

        # Try integer string fallback.
        if source_pos is None:
            try:
                source_pos = maps.id_to_pos[nodes[0]].get(str(int(float(source_id))))
            except ValueError:
                source_pos = None
        if gene_pos is None:
            try:
                gene_pos = maps.id_to_pos["G"].get(str(int(float(gene_id))))
            except ValueError:
                gene_pos = None

        if source_pos is None or gene_pos is None:
            continue

        paths = _enumerate_paths(
            source_pos=int(source_pos),
            gene_pos=int(gene_pos),
            nodes=nodes,
            edges=edges,
            edge_loader=edge_loader,
            top_k=top_paths,
            degree_d=degree_d,
        )
        for rank, (score, pos_path) in enumerate(paths, start=1):
            id_path = []
            name_path = []
            for node_type, pos in zip(nodes, pos_path):
                node_id = maps.pos_to_id[node_type].get(int(pos))
                node_name = maps.id_to_name[node_type].get(str(node_id), str(node_id))
                id_path.append(str(node_id))
                name_path.append(str(node_name))

            path_rows.append(
                {
                    "lv_id": row.lv_id,
                    "target_set_id": row.target_set_id,
                    "target_set_label": row.target_set_label,
                    "node_type": row.node_type,
                    "metapath": metapath_src_to_gene,
                    "metapath_g_orientation": metapath_g,
                    "target_id": row.target_id,
                    "target_name": row.target_name,
                    "gene_identifier": row.gene_identifier,
                    "gene_symbol": row.gene_symbol,
                    "pair_rank": int(row.pair_rank),
                    "path_rank": rank,
                    "path_score": float(score),
                    "path_nodes_ids": "|".join(id_path),
                    "path_nodes_names": "|".join(name_path),
                }
            )

    top_paths_columns = [
        "lv_id",
        "target_set_id",
        "target_set_label",
        "node_type",
        "metapath",
        "metapath_g_orientation",
        "target_id",
        "target_name",
        "gene_identifier",
        "gene_symbol",
        "pair_rank",
        "path_rank",
        "path_score",
        "path_nodes_ids",
        "path_nodes_names",
    ]
    top_paths_df = pd.DataFrame(path_rows, columns=top_paths_columns)
    top_paths_df.to_csv(output_dir / "top_paths.csv", index=False)
    return top_pairs_df, top_paths_df


def _build_layers(paths: list[list[str]], n_layers: int) -> list[list[str]]:
    layers = [[] for _ in range(n_layers)]
    for path in paths:
        for idx, node in enumerate(path):
            if node not in layers[idx]:
                layers[idx].append(node)
    return layers


def _build_positions(layers: list[list[str]]) -> dict[tuple[int, str], tuple[float, float]]:
    positions = {}
    n_layers = len(layers)
    for idx, layer in enumerate(layers):
        ys = np.linspace(0, 1, len(layer)) if layer else []
        x = idx / (n_layers - 1) if n_layers > 1 else 0.5
        for node, y in zip(layer, ys):
            positions[(idx, node)] = (x, y)
    return positions


def _select_shared_intermediate_gene_paths(
    group: pd.DataFrame,
    min_shared_intermediates: int,
    min_genes: int,
    max_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top genes per LV/metapath/target group using pair-level DWPC rank.
    """
    work = group.copy()
    work["path_nodes"] = work["path_nodes_names"].astype(str).str.split("|")
    work["intermediate_nodes"] = work["path_nodes"].map(
        lambda nodes: tuple(nodes[1:-1])
    )

    min_shared_intermediates = max(int(min_shared_intermediates), 0)
    min_genes = max(int(min_genes), 2)
    max_genes = max(int(max_genes), min_genes)

    if min_shared_intermediates > 0:
        work = work[
            work["intermediate_nodes"].map(len) >= min_shared_intermediates
        ].copy()
    if work.empty:
        return work, pd.DataFrame()

    work["gene_identifier_str"] = work["gene_identifier"].astype(str)
    work["gene_symbol"] = work["gene_symbol"].fillna("").astype(str)
    gene_summary = (
        work.groupby(["gene_identifier_str", "gene_symbol"], as_index=False)
        .agg(
            best_pair_rank=("pair_rank", "min"),
            best_path_score=("path_score", "max"),
            n_paths=("intermediate_nodes", "size"),
            n_shared_patterns=("intermediate_nodes", "nunique"),
        )
        .sort_values(
            ["best_pair_rank", "best_path_score", "n_paths"],
            ascending=[True, False, False],
        )
    )
    top_gene_ids = set(gene_summary["gene_identifier_str"].head(max_genes))
    work = work[work["gene_identifier_str"].isin(top_gene_ids)].copy()

    n_unique_genes = work["gene_identifier_str"].nunique()
    if n_unique_genes < min_genes:
        return work.iloc[0:0].copy(), pd.DataFrame()

    gene_summary = gene_summary[
        gene_summary["gene_identifier_str"].isin(top_gene_ids)
    ].copy()
    return work, gene_summary


def plot_top_subgraphs(
    output_dir: Path,
    min_shared_intermediates: int = 0,
    min_genes: int = 2,
    max_genes: int = 8,
) -> int:
    """
    Plot LV-level multi-gene network diagrams from top_paths.csv.
    """
    output_dir = Path(output_dir)
    top_paths_path = output_dir / "top_paths.csv"
    if not top_paths_path.exists():
        raise FileNotFoundError(f"Missing path file: {top_paths_path}")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    top_paths = pd.read_csv(top_paths_path)
    if top_paths.empty:
        return 0

    key_cols = ["lv_id", "target_set_id", "metapath", "target_id"]
    n_written = 0
    for key, group in top_paths.groupby(key_cols, sort=True):
        selected_paths, _ = _select_shared_intermediate_gene_paths(
            group=group,
            min_shared_intermediates=min_shared_intermediates,
            min_genes=min_genes,
            max_genes=max_genes,
        )
        if selected_paths.empty:
            continue

        paths = selected_paths["path_nodes"].tolist()
        if not paths:
            continue

        n_layers = len(paths[0])
        layers = _build_layers(paths, n_layers)
        positions = _build_positions(layers)
        gene_label_df = selected_paths[
            ["gene_identifier_str", "gene_symbol"]
        ].drop_duplicates()
        gene_label_df["gene_label"] = (
            gene_label_df["gene_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .fillna(gene_label_df["gene_identifier_str"])
        )
        gene_labels = gene_label_df["gene_label"].tolist()
        cmap = plt.get_cmap("tab20", max(len(gene_labels), 1))
        gene_color_map = {label: cmap(idx) for idx, label in enumerate(gene_labels)}

        fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * max(len(x) for x in layers))))
        for path in paths:
            gene_label = path[-1]
            edge_color = gene_color_map.get(gene_label, "#a0a0a0")
            for idx in range(len(path) - 1):
                x1, y1 = positions[(idx, path[idx])]
                x2, y2 = positions[(idx + 1, path[idx + 1])]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color,
                    linewidth=1.0,
                    alpha=0.5,
                )

        for idx, layer in enumerate(layers):
            for node in layer:
                x, y = positions[(idx, node)]
                if idx == 0:
                    color = "#4c78a8"
                elif idx == len(layers) - 1:
                    color = gene_color_map.get(node, "#54a24b")
                else:
                    color = "#f58518"
                ax.scatter([x], [y], s=130, c=color, edgecolors="white", linewidths=0.7)
                ax.text(
                    x + 0.012,
                    y,
                    str(node),
                    fontsize=7,
                    va="center",
                    ha="left",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.4),
                )

        lv_id, target_set_id, metapath, target_id = key
        target_name = str(selected_paths.iloc[0]["target_name"])
        gene_text = ", ".join(gene_labels[:10])
        if len(gene_labels) > 10:
            gene_text += ", ..."
        title = (
            f"{lv_id} | {target_set_id} | {metapath}\n"
            f"target={target_name} ({target_id}) | genes={len(gene_labels)} | "
            f"intermediates>={min_shared_intermediates}\n"
            f"{gene_text}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")

        out_name = (
            f"{_sanitize(lv_id)}__{_sanitize(target_set_id)}__{_sanitize(metapath)}__"
            f"{_sanitize(str(target_id))}__multigene.png"
        )
        fig.tight_layout()
        fig.savefig(plots_dir / out_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_written += 1

    return n_written
