#!/usr/bin/env python3
"""Generate gene x intermediate and target x intermediate matrices for visualization.

For both year (GO term) and LV analyses, this script:
1. Enumerates path instances for selected (source, target, metapath) combinations
2. Outputs gene x intermediate matrices (binary: does gene use this intermediate?)
3. Outputs target x intermediate matrices (count: how many genes reach target via intermediate?)

These matrices enable heatmap visualizations of shared vs diffuse mechanisms.

Usage:
    # Year analysis (GO terms)
    python scripts/intermediate_matrices.py --mode year --go-id GO:0000002

    # LV analysis
    python scripts/intermediate_matrices.py --mode lv --lv-id LV246 --target-set-id adipose_tissue
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_top_paths_local import (  # noqa: E402
    EdgeLoader,
    enumerate_paths,
    load_node_maps,
    parse_metapath,
    reverse_metapath_abbrev,
    select_paths,
)


def _effective_number(scores: np.ndarray) -> float:
    """Compute effective number from score distribution."""
    vals = scores[np.isfinite(scores)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    weights = vals / vals.sum()
    entropy = float(-(weights * np.log(weights)).sum())
    return float(np.exp(entropy))


def _enumerate_gene_intermediates(
    genes: list[int],
    target_id: str,
    target_type: str,
    metapath_abbrev: str,
    edge_loader: EdgeLoader,
    maps,
    dwpc_lookup: dict[tuple, float],
    *,
    path_top_k: int = 5000,
    path_min_count: int = 1,
    path_max_count: int | None = None,
    degree_d: float = 0.5,
    dwpc_threshold: float = 0.0,
) -> list[dict]:
    """Enumerate paths for genes and return list of {gene, intermediate, position} records."""
    metapath_full = reverse_metapath_abbrev(metapath_abbrev)
    nodes, edges = parse_metapath(metapath_full)

    target_pos = maps.id_to_pos.get(target_type, {}).get(target_id)
    if target_pos is None:
        return []

    records = []
    for gene_id in genes:
        gene_pos = maps.id_to_pos.get("G", {}).get(gene_id)
        if gene_pos is None:
            continue

        # Get DWPC for this pair - filter by threshold
        dwpc = dwpc_lookup.get((target_id, metapath_abbrev, gene_id), 0.0)
        if dwpc <= dwpc_threshold:
            continue

        try:
            candidate_paths = enumerate_paths(
                target_pos, gene_pos, nodes, edges, edge_loader,
                top_k=path_top_k, degree_d=degree_d,
            )
            paths = select_paths(
                candidate_paths,
                selection_method="effective_number",
                top_paths=5,
                path_cumulative_frac=None,
                path_min_count=path_min_count,
                path_max_count=path_max_count,
            )
        except Exception:
            continue

        for score, pos_path in paths:
            # Intermediate nodes are positions 1 to -2 (excluding first target and last Gene)
            for i, (node_type, pos) in enumerate(zip(nodes, pos_path)):
                if 0 < i < len(nodes) - 1:  # intermediate positions
                    node_id = maps.pos_to_id[node_type].get(int(pos))
                    node_name = maps.id_to_name[node_type].get(node_id, str(node_id))
                    if node_id is not None:
                        records.append({
                            "gene_id": gene_id,
                            "gene_name": maps.id_to_name.get("G", {}).get(gene_id, str(gene_id)),
                            "intermediate_type": node_type,
                            "intermediate_id": node_id,
                            "intermediate_name": node_name,
                            "intermediate_position": i,
                            "path_score": score,
                            "dwpc": dwpc,
                        })

    return records


def _load_year_data(
    go_id: str,
    support_path: Path,
    direct_results_dir: Path,
    selection_col: str,
    max_rank: int,
    dwpc_threshold_str: str,
    chunksize: int = 200_000,
) -> tuple[list[str], pd.DataFrame, dict, float]:
    """Load data for year analysis (GO terms)."""
    # Load support table to get selected metapaths
    support = pd.read_csv(support_path)
    support[selection_col] = (
        support[selection_col]
        .astype(str).str.strip().str.lower()
        .isin({"1", "true", "t", "yes"})
    )

    # Get metapaths selected in 2024 for this GO term
    selected = support[
        (support["go_id"] == go_id) &
        (support["year"] == 2024) &
        support[selection_col]
    ].copy()

    if selected.empty:
        return [], pd.DataFrame(), {}, 0.0

    # Rank and filter
    selected["rank"] = selected["consensus_score"].rank(method="first", ascending=False)
    selected = selected[selected["rank"] <= max_rank]
    metapaths = selected["metapath"].tolist()

    # Load DWPC data for 2024
    pattern = "dwpc_*_2024_real.csv"
    matches = sorted(direct_results_dir.glob(pattern))
    chunks = []
    for path in matches:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk = chunk[chunk["go_id"] == go_id]
            if not chunk.empty:
                chunks.append(chunk)

    if not chunks:
        return metapaths, pd.DataFrame(), {}, 0.0

    dwpc_df = pd.concat(chunks, ignore_index=True)

    # Create DWPC lookup
    dwpc_lookup = {}
    for _, row in dwpc_df.iterrows():
        dwpc_lookup[(row["go_id"], row["metapath"], int(row["entrez_gene_id"]))] = row["dwpc"]

    # Parse threshold
    if dwpc_threshold_str.startswith("p"):
        percentile = float(dwpc_threshold_str[1:])
        nonzero = dwpc_df[dwpc_df["dwpc"] > 0]["dwpc"].values
        dwpc_threshold = float(np.percentile(nonzero, percentile)) if len(nonzero) > 0 else 0.0
    else:
        dwpc_threshold = float(dwpc_threshold_str)

    return metapaths, dwpc_df, dwpc_lookup, dwpc_threshold


def _load_lv_data(
    lv_id: str,
    target_set_id: str,
    lv_output_dir: Path,
    dwpc_threshold_str: str,
) -> tuple[list[str], pd.DataFrame, dict, float, str]:
    """Load data for LV analysis."""
    # Load feature manifest to get metapaths and target info
    manifest_path = lv_output_dir / "feature_manifest.csv"
    if not manifest_path.exists():
        return [], pd.DataFrame(), {}, 0.0, ""

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[
        (manifest["lv_id"] == lv_id) &
        (manifest["target_set_id"] == target_set_id)
    ]

    if manifest.empty:
        return [], pd.DataFrame(), {}, 0.0, ""

    target_type = manifest["node_type"].iloc[0]
    metapaths = manifest["metapath"].unique().tolist()

    # Load gene scores
    scores_path = lv_output_dir / "gene_feature_scores.npy"
    genes_path = lv_output_dir / "gene_ids.npy"

    if not scores_path.exists() or not genes_path.exists():
        return metapaths, pd.DataFrame(), {}, 0.0, target_type

    scores = np.load(scores_path)
    gene_ids = np.load(genes_path)

    # Build DWPC lookup from numpy arrays
    dwpc_lookup = {}
    all_dwpc_values = []

    for _, row in manifest.iterrows():
        feature_idx = row["feature_idx"]
        metapath = row["metapath"]
        target_id = row["target_id"]
        feature_scores = scores[:, feature_idx]

        for gene_idx, gene_id in enumerate(gene_ids):
            dwpc = float(feature_scores[gene_idx])
            if dwpc > 0:
                dwpc_lookup[(target_id, metapath, int(gene_id))] = dwpc
                all_dwpc_values.append(dwpc)

    # Parse threshold
    if dwpc_threshold_str.startswith("p"):
        percentile = float(dwpc_threshold_str[1:])
        dwpc_threshold = float(np.percentile(all_dwpc_values, percentile)) if all_dwpc_values else 0.0
    else:
        dwpc_threshold = float(dwpc_threshold_str)

    # Create a simple DataFrame with gene info
    dwpc_df = pd.DataFrame({"gene_id": gene_ids})

    return metapaths, dwpc_df, dwpc_lookup, dwpc_threshold, target_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["year", "lv"], required=True,
        help="Analysis mode: 'year' for GO term analysis, 'lv' for LV analysis",
    )
    # Year-specific args
    parser.add_argument("--go-id", help="GO term ID (for year mode)")
    parser.add_argument(
        "--support-path", default="output/year_direct_go_term_support_b5.csv",
        help="Path to support table (year mode)",
    )
    parser.add_argument(
        "--direct-results-dir", default="output/dwpc_direct/all_GO_positive_growth/results",
        help="Directory with DWPC results (year mode)",
    )
    # LV-specific args
    parser.add_argument("--lv-id", help="LV ID (for lv mode)")
    parser.add_argument("--target-set-id", help="Target set ID (for lv mode)")
    parser.add_argument(
        "--lv-output-dir", default="output/lv_experiment",
        help="LV output directory (lv mode)",
    )
    # Common args
    parser.add_argument("--selection-col", default="selected_by_effective_n_all")
    parser.add_argument("--max-metapath-rank", type=int, default=5)
    parser.add_argument("--dwpc-threshold", type=str, default="p75")
    parser.add_argument("--path-top-k", type=int, default=5000)
    parser.add_argument("--degree-d", type=float, default=0.5)
    parser.add_argument("--output-dir", default="output/intermediate_matrices")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_dir = REPO_ROOT / "data" / "edges"
    edge_loader = EdgeLoader(edges_dir)

    # Load data based on mode
    if args.mode == "year":
        if not args.go_id:
            print("Error: --go-id required for year mode", file=sys.stderr)
            sys.exit(1)

        metapaths, dwpc_df, dwpc_lookup, dwpc_threshold = _load_year_data(
            args.go_id,
            Path(args.support_path),
            Path(args.direct_results_dir),
            args.selection_col,
            args.max_metapath_rank,
            args.dwpc_threshold,
        )
        target_id = args.go_id
        target_type = "BP"
        output_prefix = f"year_{args.go_id}"
        genes = dwpc_df["entrez_gene_id"].unique().astype(int).tolist() if not dwpc_df.empty else []

    else:  # lv mode
        if not args.lv_id or not args.target_set_id:
            print("Error: --lv-id and --target-set-id required for lv mode", file=sys.stderr)
            sys.exit(1)

        metapaths, dwpc_df, dwpc_lookup, dwpc_threshold, target_type = _load_lv_data(
            args.lv_id,
            args.target_set_id,
            Path(args.lv_output_dir),
            args.dwpc_threshold,
        )
        # For LV, target_id comes from manifest - get first one
        target_ids = list(set(k[0] for k in dwpc_lookup.keys()))
        target_id = target_ids[0] if target_ids else ""
        output_prefix = f"lv_{args.lv_id}_{args.target_set_id}"
        genes = dwpc_df["gene_id"].unique().astype(int).tolist() if not dwpc_df.empty else []

    if not metapaths:
        print(f"No selected metapaths found")
        sys.exit(0)

    print(f"Mode: {args.mode}")
    print(f"Target: {target_id} ({target_type})")
    print(f"Metapaths: {len(metapaths)}")
    print(f"Genes: {len(genes)}")
    print(f"DWPC threshold: {dwpc_threshold:.3f}")

    # Load node maps
    node_types = ["BP", "G", "C", "PW", "MF", "CC", "A", "D", "SE"]
    maps = load_node_maps(REPO_ROOT, node_types)

    # Enumerate intermediates for all metapaths
    all_records = []
    for metapath in metapaths:
        print(f"  Processing {metapath}...")
        records = _enumerate_gene_intermediates(
            genes, target_id, target_type, metapath, edge_loader, maps, dwpc_lookup,
            path_top_k=args.path_top_k,
            degree_d=args.degree_d,
            dwpc_threshold=dwpc_threshold,
        )
        for r in records:
            r["metapath"] = metapath
            r["target_id"] = target_id
            r["target_type"] = target_type
        all_records.extend(records)

    if not all_records:
        print("No path instances found")
        sys.exit(0)

    records_df = pd.DataFrame(all_records)
    print(f"Total records: {len(records_df)}")

    # Save raw records
    records_df.to_csv(out_dir / f"{output_prefix}_path_records.csv", index=False)
    print(f"Saved: {out_dir / f'{output_prefix}_path_records.csv'}")

    # Create gene x intermediate matrix (binary)
    # Aggregate across metapaths - gene uses intermediate if it appears in any metapath
    gene_int_df = records_df[["gene_id", "gene_name", "intermediate_id", "intermediate_name", "intermediate_type"]].drop_duplicates()

    # Create unique intermediate label
    gene_int_df["intermediate_label"] = (
        gene_int_df["intermediate_type"] + ":" +
        gene_int_df["intermediate_id"].astype(str) + " (" +
        gene_int_df["intermediate_name"] + ")"
    )

    # Pivot to matrix
    gene_int_matrix = gene_int_df.pivot_table(
        index=["gene_id", "gene_name"],
        columns="intermediate_label",
        aggfunc="size",
        fill_value=0,
    )
    gene_int_matrix = (gene_int_matrix > 0).astype(int)  # binary
    gene_int_matrix.to_csv(out_dir / f"{output_prefix}_gene_x_intermediate.csv")
    print(f"Saved: {out_dir / f'{output_prefix}_gene_x_intermediate.csv'}")
    print(f"  Shape: {gene_int_matrix.shape}")

    # Create intermediate summary (how many genes use each intermediate)
    int_summary = gene_int_df.groupby(
        ["intermediate_type", "intermediate_id", "intermediate_name", "intermediate_label"]
    ).agg(n_genes=("gene_id", "nunique")).reset_index()
    int_summary = int_summary.sort_values("n_genes", ascending=False)
    int_summary.to_csv(out_dir / f"{output_prefix}_intermediate_summary.csv", index=False)
    print(f"Saved: {out_dir / f'{output_prefix}_intermediate_summary.csv'}")

    # Per-metapath gene x intermediate matrices
    for metapath in metapaths:
        mp_records = records_df[records_df["metapath"] == metapath]
        if mp_records.empty:
            continue

        mp_gene_int = mp_records[["gene_id", "gene_name", "intermediate_id", "intermediate_name", "intermediate_type"]].drop_duplicates()
        mp_gene_int["intermediate_label"] = (
            mp_gene_int["intermediate_type"] + ":" +
            mp_gene_int["intermediate_id"].astype(str) + " (" +
            mp_gene_int["intermediate_name"] + ")"
        )

        mp_matrix = mp_gene_int.pivot_table(
            index=["gene_id", "gene_name"],
            columns="intermediate_label",
            aggfunc="size",
            fill_value=0,
        )
        mp_matrix = (mp_matrix > 0).astype(int)

        mp_safe = metapath.replace("<", "lt").replace(">", "gt")
        mp_matrix.to_csv(out_dir / f"{output_prefix}_{mp_safe}_gene_x_intermediate.csv")

    print(f"\nSaved per-metapath matrices for {len(metapaths)} metapaths")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Genes with paths: {gene_int_matrix.shape[0]}")
    print(f"Unique intermediates: {gene_int_matrix.shape[1]}")
    print(f"Matrix density: {gene_int_matrix.values.mean():.3f}")
    print(f"Mean intermediates per gene: {gene_int_matrix.sum(axis=1).mean():.1f}")
    print(f"Mean genes per intermediate: {gene_int_matrix.sum(axis=0).mean():.1f}")

    # Generate heatmap figures
    _generate_heatmaps(
        gene_int_matrix,
        int_summary,
        records_df,
        output_prefix,
        out_dir,
        target_id,
    )


def _generate_heatmaps(
    gene_int_matrix: pd.DataFrame,
    int_summary: pd.DataFrame,
    records_df: pd.DataFrame,
    output_prefix: str,
    out_dir: Path,
    target_id: str,
    max_intermediates: int = 50,
    max_genes: int = 50,
) -> None:
    """Generate heatmap visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Get matrix values
    matrix_values = gene_int_matrix.values
    n_genes, n_intermediates = matrix_values.shape

    # Select top intermediates by number of genes using them
    top_intermediates = int_summary.nlargest(min(max_intermediates, n_intermediates), "n_genes")
    top_int_labels = top_intermediates["intermediate_label"].tolist()

    # Filter matrix to top intermediates
    cols_to_keep = [c for c in gene_int_matrix.columns if c in top_int_labels]
    if not cols_to_keep:
        print("No intermediates to plot")
        return

    filtered_matrix = gene_int_matrix[cols_to_keep]

    # If too many genes, select those with most intermediate connections
    if n_genes > max_genes:
        gene_counts = filtered_matrix.sum(axis=1)
        top_gene_idx = gene_counts.nlargest(max_genes).index
        filtered_matrix = filtered_matrix.loc[top_gene_idx]

    # Simplify labels for display
    gene_labels = [f"{name}" for _, name in filtered_matrix.index]
    int_labels = [label.split("(")[1].rstrip(")") if "(" in label else label for label in filtered_matrix.columns]

    # Figure 1: Gene x Intermediate heatmap
    fig_height = max(6, len(gene_labels) * 0.25)
    fig_width = max(10, len(int_labels) * 0.3)
    fig, ax = plt.subplots(figsize=(min(fig_width, 20), min(fig_height, 16)))

    sns.heatmap(
        filtered_matrix.values,
        xticklabels=int_labels,
        yticklabels=gene_labels,
        cmap="Blues",
        cbar_kws={"label": "Gene uses intermediate"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Intermediate nodes")
    ax.set_ylabel("Genes")
    ax.set_title(f"Gene x Intermediate connectivity\n{target_id} (top {len(int_labels)} intermediates)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{output_prefix}_gene_x_intermediate_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / f'{output_prefix}_gene_x_intermediate_heatmap.[png,pdf]'}")

    # Figure 2: Intermediate usage bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(30, len(int_summary))
    plot_data = int_summary.nlargest(top_n, "n_genes")

    colors = plt.cm.tab10([plt.cm.tab10.colors.index(c) % 10 for c in range(len(plot_data))])
    type_colors = {t: plt.cm.Set2(i) for i, t in enumerate(plot_data["intermediate_type"].unique())}
    bar_colors = [type_colors[t] for t in plot_data["intermediate_type"]]

    bars = ax.barh(
        range(len(plot_data)),
        plot_data["n_genes"],
        color=bar_colors,
    )
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["intermediate_name"], fontsize=8)
    ax.set_xlabel("Number of genes")
    ax.set_title(f"Top {top_n} shared intermediate nodes\n{target_id}")
    ax.invert_yaxis()

    # Add legend for intermediate types
    handles = [plt.Rectangle((0, 0), 1, 1, color=type_colors[t]) for t in type_colors]
    ax.legend(handles, list(type_colors.keys()), loc="lower right", title="Node type")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{output_prefix}_intermediate_usage.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / f'{output_prefix}_intermediate_usage.[png,pdf]'}")

    # Figure 3: Per-metapath Jaccard heatmap (genes x genes based on shared intermediates)
    if n_genes <= 50:
        # Compute Jaccard similarity between genes
        jaccard_matrix = np.zeros((n_genes, n_genes))
        for i in range(n_genes):
            for j in range(n_genes):
                set_i = set(np.where(matrix_values[i] > 0)[0])
                set_j = set(np.where(matrix_values[j] > 0)[0])
                if set_i or set_j:
                    jaccard_matrix[i, j] = len(set_i & set_j) / len(set_i | set_j)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            jaccard_matrix,
            xticklabels=gene_labels,
            yticklabels=gene_labels,
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Jaccard similarity"},
            ax=ax,
            square=True,
        )
        ax.set_title(f"Gene-gene Jaccard similarity (shared intermediates)\n{target_id}")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fig.savefig(out_dir / f"{output_prefix}_gene_jaccard_heatmap.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_dir / f'{output_prefix}_gene_jaccard_heatmap.[png,pdf]'}")

    # Figure 4: Metapath x Intermediate heatmap (how many genes per metapath use each intermediate)
    mp_int_counts = records_df.groupby(
        ["metapath", "intermediate_name"]
    ).agg(n_genes=("gene_id", "nunique")).reset_index()

    if not mp_int_counts.empty:
        mp_int_pivot = mp_int_counts.pivot_table(
            index="metapath",
            columns="intermediate_name",
            values="n_genes",
            fill_value=0,
        )

        # Keep only top intermediates
        top_int_names = int_summary.nlargest(min(30, len(int_summary)), "n_genes")["intermediate_name"].tolist()
        cols_to_plot = [c for c in mp_int_pivot.columns if c in top_int_names]

        if cols_to_plot:
            mp_int_filtered = mp_int_pivot[cols_to_plot]

            fig_height = max(4, len(mp_int_filtered) * 0.4)
            fig, ax = plt.subplots(figsize=(14, fig_height))

            sns.heatmap(
                mp_int_filtered,
                cmap="YlGnBu",
                cbar_kws={"label": "Number of genes"},
                ax=ax,
                linewidths=0.5,
                linecolor="white",
            )
            ax.set_xlabel("Intermediate nodes")
            ax.set_ylabel("Metapath")
            ax.set_title(f"Metapath x Intermediate connectivity\n{target_id}")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()

            for ext in ["png", "pdf"]:
                fig.savefig(out_dir / f"{output_prefix}_metapath_x_intermediate_heatmap.{ext}", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_dir / f'{output_prefix}_metapath_x_intermediate_heatmap.[png,pdf]'}")


if __name__ == "__main__":
    main()
