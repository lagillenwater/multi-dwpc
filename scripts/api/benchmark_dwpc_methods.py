"""
Benchmark script comparing DWPC computation methods.

Compares:
1. API-based lookup via connectivity-search-backend Docker stack
2. Direct computation via hetmatpy matrix multiplication

Usage:
    python scripts/benchmark_dwpc_methods.py
"""

import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from dwpc_direct import HetMat

API_BASE_URL = "http://localhost:8015/v1"
SAMPLE_SIZES = [10, 50, 100, 500]
METAPATHS_TO_TEST = ["GpBP", "GiGpBP", "GiGiGpBP"]


def load_test_pairs(data_dir: Path, n_pairs: int, seed: int = 42) -> pd.DataFrame:
    """Load a sample of gene-BP pairs for benchmarking."""
    pairs_path = repo_root / "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Test pairs not found at {pairs_path}")

    df = pd.read_csv(pairs_path)
    np.random.seed(seed)
    sample_idx = np.random.choice(len(df), size=min(n_pairs, len(df)), replace=False)
    return df.iloc[sample_idx].copy()


def get_neo4j_ids(data_dir: Path) -> tuple[dict, dict]:
    """Load mappings from external IDs to neo4j IDs."""
    gene_map = pd.read_csv(data_dir / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(data_dir / "neo4j_bp_mapping.csv")

    entrez_to_neo4j = dict(zip(gene_map["identifier"], gene_map["neo4j_id"]))
    go_to_neo4j = dict(zip(bp_map["identifier"], bp_map["neo4j_id"]))

    return entrez_to_neo4j, go_to_neo4j


def benchmark_api(
    pairs_df: pd.DataFrame,
    entrez_to_neo4j: dict,
    go_to_neo4j: dict,
    metapath: str
) -> dict:
    """Benchmark API-based DWPC lookup."""
    # Map to neo4j IDs
    pairs_df = pairs_df.copy()
    pairs_df["gene_neo4j"] = pairs_df["entrez_gene_id"].map(entrez_to_neo4j)
    pairs_df["bp_neo4j"] = pairs_df["go_id"].map(go_to_neo4j)
    valid_pairs = pairs_df.dropna(subset=["gene_neo4j", "bp_neo4j"])

    n_pairs = len(valid_pairs)
    dwpc_values = []
    errors = 0

    start_time = time.perf_counter()

    with httpx.Client(timeout=30.0) as client:
        for _, row in valid_pairs.iterrows():
            source = int(row["gene_neo4j"])
            target = int(row["bp_neo4j"])

            try:
                url = f"{API_BASE_URL}/metapaths/source/{source}/target/{target}/"
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

                # Find the metapath in results
                dwpc = None
                for pc in data.get("path_counts", []):
                    if pc.get("metapath_abbreviation") == metapath:
                        dwpc = pc.get("dwpc")
                        break
                dwpc_values.append(dwpc)
            except Exception:
                errors += 1
                dwpc_values.append(None)

    elapsed = time.perf_counter() - start_time

    return {
        "method": "API",
        "metapath": metapath,
        "n_pairs": n_pairs,
        "time_seconds": elapsed,
        "time_per_pair_ms": (elapsed / n_pairs) * 1000 if n_pairs > 0 else 0,
        "errors": errors,
    }


def benchmark_direct(
    pairs_df: pd.DataFrame,
    hetmat: HetMat,
    metapath: str
) -> dict:
    """Benchmark direct DWPC computation."""
    gene_nodes = hetmat.get_nodes("Gene")
    bp_nodes = hetmat.get_nodes("Biological Process")

    entrez_to_idx = dict(zip(gene_nodes["identifier"], gene_nodes["position"]))
    go_to_idx = dict(zip(bp_nodes["identifier"], bp_nodes["position"]))

    pairs_df = pairs_df.copy()
    pairs_df["gene_idx"] = pairs_df["entrez_gene_id"].map(entrez_to_idx)
    pairs_df["bp_idx"] = pairs_df["go_id"].map(go_to_idx)
    valid_pairs = pairs_df.dropna(subset=["gene_idx", "bp_idx"])
    valid_pairs["gene_idx"] = valid_pairs["gene_idx"].astype(int)
    valid_pairs["bp_idx"] = valid_pairs["bp_idx"].astype(int)

    n_pairs = len(valid_pairs)
    gene_indices = valid_pairs["gene_idx"].values
    bp_indices = valid_pairs["bp_idx"].values

    start_time = time.perf_counter()

    dwpc_values = hetmat.get_dwpc_for_pairs(
        metapath, gene_indices, bp_indices, transform=True
    )

    elapsed = time.perf_counter() - start_time

    return {
        "method": "Direct",
        "metapath": metapath,
        "n_pairs": n_pairs,
        "time_seconds": elapsed,
        "time_per_pair_ms": (elapsed / n_pairs) * 1000 if n_pairs > 0 else 0,
        "errors": 0,
    }


def check_api_available() -> bool:
    """Check if the API is available."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE_URL}/")
            return response.status_code == 200
    except Exception:
        return False


def main():
    print("=" * 70)
    print("DWPC COMPUTATION BENCHMARK")
    print("=" * 70)

    data_dir = repo_root / "data"

    # Check API availability
    api_available = check_api_available()
    if api_available:
        print(f"API available at {API_BASE_URL}")
    else:
        print(f"API not available at {API_BASE_URL}")
        print("Start the Docker stack to enable API benchmarking.")
        print("Proceeding with direct computation benchmark only.\n")

    # Load HetMat
    print("\nLoading HetMat...")
    hetmat = HetMat(data_dir, damping=0.5)
    print(f"  Loaded (damping={hetmat.damping})")

    # Load neo4j mappings if API is available
    if api_available:
        print("\nLoading neo4j ID mappings...")
        entrez_to_neo4j, go_to_neo4j = get_neo4j_ids(data_dir)
        print(f"  Gene mappings: {len(entrez_to_neo4j)}")
        print(f"  BP mappings: {len(go_to_neo4j)}")

    results = []

    for n_pairs in SAMPLE_SIZES:
        print(f"\n{'=' * 70}")
        print(f"SAMPLE SIZE: {n_pairs} pairs")
        print("=" * 70)

        pairs_df = load_test_pairs(data_dir, n_pairs)
        print(f"Loaded {len(pairs_df)} test pairs")

        for metapath in METAPATHS_TO_TEST:
            print(f"\n  Metapath: {metapath}")

            # Benchmark direct computation
            direct_result = benchmark_direct(pairs_df, hetmat, metapath)
            results.append(direct_result)
            print(f"    Direct: {direct_result['time_seconds']:.3f}s "
                  f"({direct_result['time_per_pair_ms']:.2f} ms/pair)")

            # Benchmark API if available
            if api_available:
                api_result = benchmark_api(
                    pairs_df, entrez_to_neo4j, go_to_neo4j, metapath
                )
                results.append(api_result)
                print(f"    API:    {api_result['time_seconds']:.3f}s "
                      f"({api_result['time_per_pair_ms']:.2f} ms/pair)")

                if api_result['time_per_pair_ms'] > 0:
                    speedup = api_result['time_per_pair_ms'] / direct_result['time_per_pair_ms']
                    print(f"    Speedup: {speedup:.1f}x faster with direct computation")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print("\nAll results:")
    print(results_df.to_string(index=False))

    # Save results
    output_path = repo_root / "output" / "benchmark_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    if api_available:
        # Compute average speedup
        direct_times = results_df[results_df["method"] == "Direct"].groupby("metapath")["time_per_pair_ms"].mean()
        api_times = results_df[results_df["method"] == "API"].groupby("metapath")["time_per_pair_ms"].mean()

        print("\nAverage speedup by metapath:")
        for mp in METAPATHS_TO_TEST:
            if mp in direct_times.index and mp in api_times.index:
                speedup = api_times[mp] / direct_times[mp]
                print(f"  {mp}: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
