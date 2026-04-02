"""Helpers for validating direct DWPC values against historical API outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dwpc_direct import HetMat, reverse_metapath_abbrev
from result_normalization import load_neo4j_mappings


def load_api_results(api_results_path: Path | str) -> pd.DataFrame:
    """Load an API result CSV and validate the required columns."""
    api_results_path = Path(api_results_path)
    if not api_results_path.exists():
        raise FileNotFoundError(f"API results not found: {api_results_path}")
    df = pd.read_csv(api_results_path)
    required = {"neo4j_source_id", "neo4j_target_id", "metapath_abbreviation", "dwpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"API results {api_results_path} missing required columns: {sorted(missing)}"
        )
    return df


def list_api_metapaths(api_df: pd.DataFrame) -> list[str]:
    """Return sorted API metapath abbreviations present in a result frame."""
    return sorted(api_df["metapath_abbreviation"].dropna().astype(str).unique().tolist())


def _build_index_maps(hetmat: HetMat) -> tuple[dict, dict]:
    genes = hetmat.get_nodes("Gene")
    bps = hetmat.get_nodes("Biological Process")
    entrez_to_idx = dict(zip(genes["identifier"], genes["position"]))
    go_to_idx = dict(zip(bps["identifier"], bps["position"]))
    return entrez_to_idx, go_to_idx


def sample_metapath_concordance(
    hetmat: HetMat,
    api_df: pd.DataFrame,
    *,
    data_dir: Path | str,
    metapath_api: str,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Sample API rows for one metapath and compare direct vs API DWPC values."""
    api_subset = api_df[api_df["metapath_abbreviation"].astype(str) == str(metapath_api)].copy()
    if api_subset.empty:
        return pd.DataFrame(), {
            "metapath_api": str(metapath_api),
            "status": "skipped",
            "reason": "no api rows",
        }

    neo4j_to_entrez, neo4j_to_go = load_neo4j_mappings(Path(data_dir))
    api_subset["go_id"] = api_subset["neo4j_source_id"].map(neo4j_to_go)
    api_subset["entrez_gene_id"] = api_subset["neo4j_target_id"].map(neo4j_to_entrez)
    api_subset = api_subset.dropna(subset=["go_id", "entrez_gene_id"]).copy()
    if api_subset.empty:
        return pd.DataFrame(), {
            "metapath_api": str(metapath_api),
            "status": "skipped",
            "reason": "no mapped rows",
        }

    sample_size = min(int(n_samples), len(api_subset))
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(api_subset.index.to_numpy(), size=sample_size, replace=False)
    sample_df = api_subset.loc[sample_idx].copy().reset_index(drop=True)

    entrez_to_idx, go_to_idx = _build_index_maps(hetmat)
    sample_df["source_idx"] = sample_df["entrez_gene_id"].map(entrez_to_idx)
    sample_df["target_idx"] = sample_df["go_id"].map(go_to_idx)
    sample_df = sample_df.dropna(subset=["source_idx", "target_idx"]).copy()
    if sample_df.empty:
        return pd.DataFrame(), {
            "metapath_api": str(metapath_api),
            "status": "skipped",
            "reason": "no valid indexed rows",
        }

    sample_df["source_idx"] = sample_df["source_idx"].astype(int)
    sample_df["target_idx"] = sample_df["target_idx"].astype(int)
    metapath_direct = reverse_metapath_abbrev(str(metapath_api))
    sample_df["direct_dwpc"] = hetmat.get_dwpc_for_pairs(
        metapath_direct,
        sample_df["source_idx"].to_numpy(),
        sample_df["target_idx"].to_numpy(),
        transform=True,
    )
    sample_df["api_dwpc"] = sample_df["dwpc"].astype(float)
    sample_df["diff"] = sample_df["direct_dwpc"] - sample_df["api_dwpc"]
    sample_df["abs_diff"] = sample_df["diff"].abs()
    sample_df["metapath_api"] = str(metapath_api)

    correlation = sample_df["direct_dwpc"].corr(sample_df["api_dwpc"])
    exact_matches = int((sample_df["abs_diff"] < 1e-5).sum())
    summary = {
        "metapath_api": str(metapath_api),
        "status": "validated",
        "n_pairs": int(len(sample_df)),
        "exact_matches": exact_matches,
        "pct_exact": 100.0 * exact_matches / float(len(sample_df)),
        "correlation": float(correlation) if pd.notna(correlation) else np.nan,
        "mean_abs_diff": float(sample_df["abs_diff"].mean()),
        "median_abs_diff": float(sample_df["abs_diff"].median()),
        "max_abs_diff": float(sample_df["abs_diff"].max()),
    }
    return sample_df, summary
