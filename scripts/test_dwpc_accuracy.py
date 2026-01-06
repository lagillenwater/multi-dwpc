"""
Test script to validate DWPC direct computation against API results.

This script compares DWPC values computed directly via matrix multiplication
against the gold-standard values from the connectivity-search-backend API.

Usage:
    python scripts/test_dwpc_accuracy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from dwpc_direct import HetMat


def load_neo4j_mappings(data_dir: Path) -> tuple[dict, dict]:
    """Load neo4j ID to external identifier mappings."""
    gene_map = pd.read_csv(data_dir / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(data_dir / "neo4j_bp_mapping.csv")

    neo4j_to_entrez = dict(zip(gene_map["neo4j_id"], gene_map["identifier"]))
    neo4j_to_go = dict(zip(bp_map["neo4j_id"], bp_map["identifier"]))

    return neo4j_to_entrez, neo4j_to_go


def validate_metapath(
    hetmat: HetMat,
    api_df: pd.DataFrame,
    metapath_api: str,
    metapath_direct: str,
    neo4j_to_entrez: dict,
    neo4j_to_go: dict,
    n_samples: int = 1000,
    seed: int = 42
) -> dict:
    """
    Validate DWPC computation for a specific metapath.

    Parameters
    ----------
    hetmat : HetMat
        Initialized HetMat instance
    api_df : pd.DataFrame
        API results DataFrame
    metapath_api : str
        Metapath abbreviation as used in API (e.g., "BPpG")
    metapath_direct : str
        Metapath abbreviation for direct computation (e.g., "GpBP")
    neo4j_to_entrez : dict
        Neo4j ID to Entrez ID mapping
    neo4j_to_go : dict
        Neo4j ID to GO ID mapping
    n_samples : int
        Number of pairs to sample for validation
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Validation results
    """
    api_mp = api_df[api_df["metapath_abbreviation"] == metapath_api].copy()

    if len(api_mp) == 0:
        return {"metapath": metapath_direct, "status": "skipped", "reason": "no API data"}

    api_mp["go_id"] = api_mp["neo4j_source_id"].map(neo4j_to_go)
    api_mp["entrez_id"] = api_mp["neo4j_target_id"].map(neo4j_to_entrez)
    api_mapped = api_mp.dropna(subset=["go_id", "entrez_id"]).copy()

    if len(api_mapped) == 0:
        return {"metapath": metapath_direct, "status": "skipped", "reason": "no mapped pairs"}

    genes = hetmat.get_nodes("Gene")
    bps = hetmat.get_nodes("Biological Process")
    entrez_to_idx = dict(zip(genes["identifier"], genes["position"]))
    go_to_idx = dict(zip(bps["identifier"], bps["position"]))

    np.random.seed(seed)
    sample_size = min(n_samples, len(api_mapped))
    sample_indices = np.random.choice(len(api_mapped), size=sample_size, replace=False)
    api_sample = api_mapped.iloc[sample_indices].copy()

    api_sample["gene_idx"] = api_sample["entrez_id"].map(entrez_to_idx)
    api_sample["bp_idx"] = api_sample["go_id"].map(go_to_idx)
    valid_pairs = api_sample.dropna(subset=["gene_idx", "bp_idx"]).copy()
    valid_pairs["gene_idx"] = valid_pairs["gene_idx"].astype(int)
    valid_pairs["bp_idx"] = valid_pairs["bp_idx"].astype(int)

    if len(valid_pairs) == 0:
        return {"metapath": metapath_direct, "status": "skipped", "reason": "no valid pairs"}

    gene_indices = valid_pairs["gene_idx"].values
    bp_indices = valid_pairs["bp_idx"].values

    computed_dwpc = hetmat.get_dwpc_for_pairs(
        metapath_direct, gene_indices, bp_indices, transform=True
    )

    valid_pairs["computed_dwpc"] = computed_dwpc
    valid_pairs["api_dwpc"] = valid_pairs["dwpc"]
    valid_pairs["diff"] = valid_pairs["computed_dwpc"] - valid_pairs["api_dwpc"]
    valid_pairs["abs_diff"] = valid_pairs["diff"].abs()

    correlation = valid_pairs["computed_dwpc"].corr(valid_pairs["api_dwpc"])
    exact_matches = (valid_pairs["abs_diff"] < 1e-5).sum()

    return {
        "metapath": metapath_direct,
        "status": "validated",
        "n_pairs": len(valid_pairs),
        "exact_matches": exact_matches,
        "pct_exact": 100 * exact_matches / len(valid_pairs),
        "correlation": correlation,
        "mean_abs_diff": valid_pairs["abs_diff"].mean(),
        "max_abs_diff": valid_pairs["abs_diff"].max(),
    }


def main():
    print("=" * 70)
    print("DWPC ACCURACY VALIDATION")
    print("=" * 70)

    data_dir = repo_root / "data"
    api_results_path = repo_root / "output/dwpc_com/dataset2/results/res_dataset2_2016_real.csv"

    if not api_results_path.exists():
        print(f"ERROR: API results not found at {api_results_path}")
        print("Run the API-based DWPC computation first.")
        sys.exit(1)

    neo4j_gene_map = data_dir / "neo4j_gene_mapping.csv"
    neo4j_bp_map = data_dir / "neo4j_bp_mapping.csv"

    if not neo4j_gene_map.exists() or not neo4j_bp_map.exists():
        print("ERROR: Neo4j ID mappings not found.")
        print("These are created during validation. Run the validation manually first.")
        sys.exit(1)

    print("\nLoading data...")
    hetmat = HetMat(data_dir, damping=0.5)
    api_df = pd.read_csv(api_results_path)
    neo4j_to_entrez, neo4j_to_go = load_neo4j_mappings(data_dir)

    print(f"  HetMat loaded (damping={hetmat.damping})")
    print(f"  API results: {len(api_df)} rows")
    print(f"  Gene mappings: {len(neo4j_to_entrez)}")
    print(f"  BP mappings: {len(neo4j_to_go)}")

    metapath_pairs = [
        ("BPpG", "GpBP"),
        ("BPpGiG", "GiGpBP"),
        ("BPpGcG", "GcGpBP"),
        ("BPpGiGiG", "GiGiGpBP"),
        ("BPpGcGcG", "GcGcGpBP"),
        ("BPpGcGiG", "GiGcGpBP"),
    ]

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS (1000 pairs per metapath)")
    print("=" * 70)

    results = []
    all_passed = True

    for api_mp, direct_mp in metapath_pairs:
        result = validate_metapath(
            hetmat, api_df, api_mp, direct_mp,
            neo4j_to_entrez, neo4j_to_go,
            n_samples=1000
        )
        results.append(result)

        if result["status"] == "validated":
            passed = result["pct_exact"] == 100.0 and result["correlation"] > 0.9999
            status_str = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"\n{direct_mp} ({api_mp}):")
            print(f"  Pairs tested:    {result['n_pairs']}")
            print(f"  Exact matches:   {result['exact_matches']} ({result['pct_exact']:.1f}%)")
            print(f"  Correlation:     {result['correlation']:.6f}")
            print(f"  Mean abs diff:   {result['mean_abs_diff']:.2e}")
            print(f"  Max abs diff:    {result['max_abs_diff']:.2e}")
            print(f"  Status:          {status_str}")
        else:
            print(f"\n{direct_mp}: SKIPPED ({result.get('reason', 'unknown')})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    validated = [r for r in results if r["status"] == "validated"]
    total_pairs = sum(r["n_pairs"] for r in validated)
    total_exact = sum(r["exact_matches"] for r in validated)

    print(f"  Metapaths validated: {len(validated)}/{len(metapath_pairs)}")
    print(f"  Total pairs tested:  {total_pairs}")
    print(f"  Total exact matches: {total_exact} ({100*total_exact/total_pairs:.1f}%)")

    if all_passed:
        print("\n  RESULT: ALL TESTS PASSED")
        return 0
    else:
        print("\n  RESULT: SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
