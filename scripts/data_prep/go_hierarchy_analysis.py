"""
GO hierarchy analysis for the publication pipeline.

This script computes hierarchy metrics, correlation plots, and the
parents_GO_postive_growth dataset for publication reproductions.
"""

from pathlib import Path
import sys
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def ensure_obo_files(repo_root: Path) -> tuple[Path, Path]:
    """Download GO ontology OBO files if needed."""
    go_ontology_dir = repo_root / "gene_ontology/gene-ontology/download"
    go_ontology_dir.mkdir(parents=True, exist_ok=True)

    obo_2024_path = go_ontology_dir / "go-basic.obo"
    obo_2016_path = go_ontology_dir / "go-basic-2016-02-01.obo"

    go_obo_2024_url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    go_obo_2016_url = "http://release.geneontology.org/2016-02-01/ontology/go-basic.obo"

    if not obo_2024_path.exists():
        print(f"Downloading GO OBO 2024 from {go_obo_2024_url}...")
        urllib.request.urlretrieve(go_obo_2024_url, obo_2024_path)
        print(f"Saved to {obo_2024_path}")

    if not obo_2016_path.exists():
        print(f"Downloading GO OBO 2016 from {go_obo_2016_url}...")
        urllib.request.urlretrieve(go_obo_2016_url, obo_2016_path)
        print(f"Saved to {obo_2016_path}")

    return obo_2016_path, obo_2024_path


def main() -> None:
    """Run GO hierarchy analysis and parent-term selection."""
    if Path.cwd().name == "notebooks":
        repo_root = Path("..").resolve()
    else:
        repo_root = Path.cwd()

    print(f"Repo root: {repo_root}")
    sys.path.insert(0, str(repo_root))

    from src.go_processing import (
        add_hierarchy_metrics,
        calculate_percent_change,
        identify_leaf_terms,
        identify_parent_terms,
    )

    output_intermediate = repo_root / "output/intermediate"
    output_intermediate.mkdir(parents=True, exist_ok=True)
    output_images = repo_root / "output/images"
    output_images.mkdir(parents=True, exist_ok=True)

    com_go_terms = pd.read_csv(
        output_intermediate / "common_go_terms.csv"
    )
    classification_summary = pd.read_csv(
        output_intermediate / "go_gene_classification_summary.csv"
    )

    com_go_terms = com_go_terms.merge(
        classification_summary[["go_id", "n_stable", "n_added"]],
        on="go_id",
        how="left",
    )

    com_go_terms = calculate_percent_change(
        com_go_terms,
        count_col_2016="no_of_genes_in_hetio_GO_2016",
        count_col_2024="no_of_genes_in_GO_2024",
    )

    obo_2016_path, obo_2024_path = ensure_obo_files(repo_root)

    cache_file = repo_root / "output/cached_hierarchy_metrics.csv"
    if cache_file.exists():
        print(f"Loading cached hierarchy metrics from {cache_file}")
        com_go_terms_w_hierarchy = pd.read_csv(cache_file)
    else:
        print("Calculating hierarchy metrics (this may take a few minutes)...")
        com_go_terms_w_hierarchy = add_hierarchy_metrics(
            com_go_terms,
            obo_path_2016=str(obo_2016_path),
            obo_path_2024=str(obo_2024_path),
        )
        com_go_terms_w_hierarchy.to_csv(cache_file, index=False)
        print(f"Saved hierarchy metrics to cache: {cache_file}")

    df_with_hierarchy = com_go_terms_w_hierarchy.dropna(
        subset=["normalized_depth_2016", "normalized_depth_2024", "pct_change_genes"]
    ).copy()

    corr_2016, p_2016 = stats.spearmanr(
        df_with_hierarchy["normalized_depth_2016"],
        df_with_hierarchy["pct_change_genes"],
    )
    corr_2024, p_2024 = stats.spearmanr(
        df_with_hierarchy["normalized_depth_2024"],
        df_with_hierarchy["pct_change_genes"],
    )
    corr_change, p_change = stats.spearmanr(
        df_with_hierarchy["norm_depth_change"],
        df_with_hierarchy["pct_change_genes"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)

    axes[0].scatter(
        df_with_hierarchy["normalized_depth_2016"],
        df_with_hierarchy["pct_change_genes"],
        alpha=0.3,
        s=20,
    )
    axes[0].set_xlabel("Normalized Depth (2016)", fontsize=11)
    axes[0].set_ylabel("Percent Change in Gene Count", fontsize=11)
    axes[0].set_title(
        f"2016 Depth vs Change\nSpearman r={corr_2016:.3f}, p={p_2016:.2e}",
        fontsize=11,
    )
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].scatter(
        df_with_hierarchy["normalized_depth_2024"],
        df_with_hierarchy["pct_change_genes"],
        alpha=0.3,
        s=20,
        color="orange",
    )
    axes[1].set_xlabel("Normalized Depth (2024)", fontsize=11)
    axes[1].set_ylabel("Percent Change in Gene Count", fontsize=11)
    axes[1].set_title(
        f"2024 Depth vs Change\nSpearman r={corr_2024:.3f}, p={p_2024:.2e}",
        fontsize=11,
    )
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].scatter(
        df_with_hierarchy["norm_depth_change"],
        df_with_hierarchy["pct_change_genes"],
        alpha=0.3,
        s=20,
        color="green",
    )
    axes[2].set_xlabel("Change in Normalized Depth", fontsize=11)
    axes[2].set_ylabel("Percent Change in Gene Count", fontsize=11)
    axes[2].set_title(
        f"Depth Change vs Gene Change\nSpearman r={corr_change:.3f}, p={p_change:.2e}",
        fontsize=11,
    )
    axes[2].axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].axvline(x=0, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_images / "hierarchy_vs_percent_change.pdf", dpi=300,
                bbox_inches="tight")
    plt.savefig(output_images / "hierarchy_vs_percent_change.jpeg", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    all_go_positive_growth = pd.read_csv(
        output_intermediate / "all_GO_positive_growth.csv"
    )

    leaf_cache = repo_root / "output/cached_leaf_terms_2024.txt"
    parent_cache = repo_root / "output/cached_parent_terms_2024.txt"

    if leaf_cache.exists() and parent_cache.exists():
        with open(leaf_cache, "r") as f:
            leaf_terms_2024 = set(line.strip() for line in f)
        with open(parent_cache, "r") as f:
            parent_terms_2024 = set(line.strip() for line in f)
        print("Loaded cached leaf and parent terms.")
    else:
        leaf_terms_2024 = identify_leaf_terms(
            str(obo_2024_path), namespace="biological_process"
        )
        parent_terms_2024 = identify_parent_terms(
            leaf_terms_2024, str(obo_2024_path), namespace="biological_process"
        )
        with open(leaf_cache, "w") as f:
            f.write("\n".join(sorted(leaf_terms_2024)))
        with open(parent_cache, "w") as f:
            f.write("\n".join(sorted(parent_terms_2024)))
        print("Saved leaf and parent term caches.")

    parents_go_positive_growth = all_go_positive_growth[
        all_go_positive_growth["go_id"].isin(parent_terms_2024)
    ].reset_index(drop=True)

    parents_go_positive_growth.to_csv(
        output_intermediate / "parents_GO_postive_growth.csv",
        index=False,
    )
    print(
        "Saved parents_GO_postive_growth.csv: "
        f"{len(parents_go_positive_growth)} GO terms"
    )


if __name__ == "__main__":
    main()
