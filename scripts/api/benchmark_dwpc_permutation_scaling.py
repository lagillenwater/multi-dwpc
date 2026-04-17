#!/usr/bin/env python3
"""Benchmark API vs direct DWPC computation as a function of permutation count.

For a fixed pair sample and metapath list, this script:
  1. Generates B permuted GO->Gene assignments via degree-preserving swaps.
  2. Times API-based DWPC retrieval (run_metapaths_for_df) per replicate.
  3. Times direct (matrix) DWPC retrieval (get_dwpc_for_pairs) per replicate.
  4. Writes a CSV of wall times and plots total time vs B on a log scale.

Conventions (matching scripts/year_api_sampled_statistics_analysis.py):
  - Source = GO / Biological Process  (neo4j_source_id, go_id)
  - Target = Gene                     (neo4j_target_id, entrez_gene_id)
  - Default metapaths start with "BP" (BPpG, BPpGpBP, ...)

Usage:
    python scripts/benchmark_dwpc_permutation_scaling.py
    python scripts/benchmark_dwpc_permutation_scaling.py --skip-api --b-values 2,10,30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.bipartite_nulls import degree_preserving_permutations  # noqa: E402
from src.dwpc_api import run_metapaths_for_df  # noqa: E402
from dwpc_direct import HetMat  # noqa: E402

API_BASE_URL = "http://localhost:8015/v1"
DEFAULT_BASE_NAME = "all_GO_positive_growth"
DEFAULT_METAPATHS = "BPpG,BPpGpBP,BPpGpBPpG"


def check_api_available() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{API_BASE_URL}/").status_code == 200
    except Exception:
        return False


def load_source_target_pairs(n_pairs: int, seed: int) -> pd.DataFrame:
    pairs_path = (
        REPO_ROOT / "output" / "intermediate" / f"hetio_bppg_{DEFAULT_BASE_NAME}_filtered.csv"
    )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    df = pd.read_csv(pairs_path)[["go_id", "entrez_gene_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(len(df), size=min(int(n_pairs), len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def attach_neo4j_ids(pairs: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    gene_map = pd.read_csv(data_dir / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(data_dir / "neo4j_bp_mapping.csv")
    entrez_to_neo4j = dict(zip(gene_map["identifier"], gene_map["neo4j_id"]))
    go_to_neo4j = dict(zip(bp_map["identifier"], bp_map["neo4j_id"]))
    out = pairs.copy()
    out["neo4j_source_id"] = out["go_id"].map(go_to_neo4j)
    out["neo4j_target_id"] = out["entrez_gene_id"].map(entrez_to_neo4j)
    out = out.dropna(subset=["neo4j_source_id", "neo4j_target_id"])
    return out.astype({"neo4j_source_id": int, "neo4j_target_id": int}).reset_index(drop=True)


def attach_hetmat_indices(pairs: pd.DataFrame, hetmat: HetMat) -> pd.DataFrame:
    gene_nodes = hetmat.get_nodes("Gene")
    bp_nodes = hetmat.get_nodes("Biological Process")
    entrez_to_idx = dict(zip(gene_nodes["identifier"].astype(str), gene_nodes["position"]))
    go_to_idx = dict(zip(bp_nodes["identifier"].astype(str), bp_nodes["position"]))
    out = pairs.copy()
    out["gene_idx"] = out["entrez_gene_id"].astype(str).map(entrez_to_idx)
    out["bp_idx"] = out["go_id"].astype(str).map(go_to_idx)
    out = out.dropna(subset=["gene_idx", "bp_idx"])
    return out.astype({"gene_idx": int, "bp_idx": int}).reset_index(drop=True)


def generate_permutations(pairs: pd.DataFrame, b: int, seed: int) -> list[pd.DataFrame]:
    if int(b) == 0:
        return []
    return degree_preserving_permutations(
        edge_df=pairs,
        source_col="go_id",
        target_col="entrez_gene_id",
        n_permutations=int(b),
        random_state=int(seed),
        n_swap_attempts_per_edge=10,
    )


async def _time_api_call(
    pairs_with_neo: pd.DataFrame,
    parquet_dir: Path,
    tag: str,
    max_concurrency: int,
    client: httpx.AsyncClient,
) -> float:
    t0 = time.perf_counter()
    await run_metapaths_for_df(
        pairs_with_neo[["neo4j_source_id", "neo4j_target_id"]],
        col_source="neo4j_source_id",
        col_target="neo4j_target_id",
        base_out_dir=parquet_dir,
        group=tag,
        clear_group=True,
        max_concurrency=int(max_concurrency),
        retries=3,
        backoff_first=2.0,
        show_progress=True,
        check_health=False,
        client=client,
    )
    return time.perf_counter() - t0


async def _run_all_api_timings(
    perms_by_b: dict[int, list[pd.DataFrame]],
    b_values: list[int],
    n_pairs: int,
    n_metapaths: int,
    data_dir: Path,
    parquet_dir: Path,
    max_concurrency: int,
) -> list[dict]:
    limits = httpx.Limits(
        max_keepalive_connections=200,
        max_connections=250,
        keepalive_expiry=60.0,
    )
    timeout = httpx.Timeout(timeout=90.0, connect=10.0, read=70.0, write=10.0)

    print("Shared API client configuration:")
    print(f"  Max connections: {limits.max_connections}")
    print(f"  Keepalive connections: {limits.max_keepalive_connections}")
    print(f"  Global concurrency: {max_concurrency}")
    print(f"  Timeout: {90.0}s")

    api_rows: list[dict] = []
    client = httpx.AsyncClient(limits=limits, timeout=timeout)
    try:
        for b in b_values:
            perms = perms_by_b[b]
            t_api_total = 0.0
            for rep_idx, perm_df in enumerate(perms):
                perm_neo = attach_neo4j_ids(perm_df, data_dir)
                tag = f"bench_b{b:03d}_r{rep_idx:03d}"
                print(
                    f"  [bench] B={b} replicate {rep_idx + 1}/{b} -- "
                    f"calling API ({len(perm_neo)} pairs, concurrency={max_concurrency})",
                    flush=True,
                )
                t_rep = await _time_api_call(
                    perm_neo, parquet_dir, tag, max_concurrency, client
                )
                t_api_total += t_rep
                print(
                    f"  [bench] B={b} replicate {rep_idx + 1}/{b} done in {t_rep:.2f}s",
                    flush=True,
                )
            api_rows.append(
                {
                    "method": "api",
                    "b": int(b),
                    "n_pairs": int(n_pairs),
                    "n_metapaths": int(n_metapaths),
                    "time_seconds": float(t_api_total),
                    "time_per_replicate_ms": float(t_api_total * 1000.0 / b),
                }
            )
            print(
                f"  B={b:>3d}  api:    {t_api_total:7.3f}s total"
                f"   ({t_api_total * 1000.0 / b:6.1f} ms/replicate)"
            )
    finally:
        try:
            await asyncio.wait_for(client.aclose(), timeout=30.0)
        except asyncio.TimeoutError:
            print(
                "Warning: shared httpx client.aclose() timed out after 30s; "
                "continuing.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"Warning: shared httpx client.aclose() raised {exc!r}; continuing.",
                flush=True,
            )
    return api_rows


def _time_direct_call(
    pairs_with_idx: pd.DataFrame, hetmat: HetMat, metapaths: list[str]
) -> float:
    bp_idx = pairs_with_idx["bp_idx"].values
    gene_idx = pairs_with_idx["gene_idx"].values
    t0 = time.perf_counter()
    for metapath in metapaths:
        hetmat.get_dwpc_for_pairs(metapath, bp_idx, gene_idx, transform=True)
    return time.perf_counter() - t0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-values", default="2,4,6,8,10,20,30")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metapaths",
        default=DEFAULT_METAPATHS,
        help="Comma-separated metapath abbreviations to time on the direct path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "benchmark_permutation_scaling"),
    )
    parser.add_argument("--skip-api", action="store_true", help="Benchmark direct only")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded direct call per metapath before timing to exclude JIT/cache effects",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=20,
        help=(
            "Max concurrent HTTP requests per API replicate. Higher values can "
            "actually be slower on a local Docker stack due to Neo4j contention "
            "(default: 20)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    b_values = sorted({int(x) for x in args.b_values.split(",") if x.strip()})
    metapaths = [s.strip() for s in str(args.metapaths).split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(exist_ok=True)

    data_dir = REPO_ROOT / "data"
    pairs = load_source_target_pairs(n_pairs=args.n_pairs, seed=args.seed)
    print(f"Loaded {len(pairs)} source-target pairs")
    print(f"B grid: {b_values}")
    print(f"Metapaths: {metapaths}")

    api_available = False if args.skip_api else check_api_available()
    print(f"API available: {api_available}")

    hetmat = HetMat(data_dir, damping=0.5)

    t_warmup = 0.0
    if args.warmup:
        warmup_pairs = attach_hetmat_indices(pairs, hetmat)
        t_warmup_start = time.perf_counter()
        for mp in metapaths:
            hetmat.get_dwpc_for_pairs(mp, warmup_pairs["bp_idx"].values, warmup_pairs["gene_idx"].values)
        t_warmup = time.perf_counter() - t_warmup_start
        print(f"Warmup complete ({t_warmup * 1000.0:.1f} ms)")

    rows: list[dict] = []
    perms_by_b: dict[int, list[pd.DataFrame]] = {}
    for b in b_values:
        perms = generate_permutations(pairs, b=b, seed=args.seed)
        perms_by_b[b] = perms

        t_direct_total = 0.0
        for perm_df in perms:
            perm_idx = attach_hetmat_indices(perm_df, hetmat)
            t_direct_total += _time_direct_call(perm_idx, hetmat, metapaths)
        t_direct_total += t_warmup
        rows.append(
            {
                "method": "direct",
                "b": int(b),
                "n_pairs": int(len(pairs)),
                "n_metapaths": len(metapaths),
                "time_seconds": float(t_direct_total),
                "time_per_replicate_ms": float(t_direct_total * 1000.0 / b),
            }
        )
        print(
            f"  B={b:>3d}  direct: {t_direct_total:7.3f}s total"
            f"   ({t_direct_total * 1000.0 / b:6.1f} ms/replicate)"
        )

    if api_available:
        api_rows = asyncio.run(
            _run_all_api_timings(
                perms_by_b=perms_by_b,
                b_values=b_values,
                n_pairs=len(pairs),
                n_metapaths=len(metapaths),
                data_dir=data_dir,
                parquet_dir=parquet_dir,
                max_concurrency=args.max_concurrency,
            )
        )
        rows.extend(api_rows)

    results_df = pd.DataFrame(rows)
    results_path = out_dir / "benchmark_permutation_scaling.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    method_colors = {"api": "#d62728", "direct": "#1f77b4"}

    ax = axes[0]
    for method, color in method_colors.items():
        sub = results_df[results_df["method"] == method].sort_values("b")
        if sub.empty:
            continue
        ax.plot(
            sub["b"].astype(int),
            sub["time_seconds"].astype(float),
            marker="o", linewidth=2.0, color=color, label=method.upper(),
        )
    ax.set_xlabel("B (permutation count)")
    ax.set_ylabel("Total wall time (s)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(title="Method", loc="best")
    ax.set_title("Total time vs B (log y)")

    ax = axes[1]
    for method, color in method_colors.items():
        sub = results_df[results_df["method"] == method].sort_values("b")
        if sub.empty:
            continue
        ax.plot(
            sub["b"].astype(int),
            sub["time_per_replicate_ms"].astype(float),
            marker="o", linewidth=2.0, color=color, label=method.upper(),
        )
    ax.set_xlabel("B (permutation count)")
    ax.set_ylabel("Time per replicate (ms)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(title="Method", loc="best")
    ax.set_title("Per-replicate cost vs B (log y)")

    fig.suptitle(
        f"DWPC permutation scaling | {len(pairs)} pairs x {len(metapaths)} metapaths"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_permutation_scaling.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'benchmark_permutation_scaling.pdf'}")

    if {"api", "direct"}.issubset(set(results_df["method"].unique())):
        speedup_df = (
            results_df.pivot(index="b", columns="method", values="time_seconds")
            .assign(speedup=lambda d: d["api"] / d["direct"])
        )
        print("\nSpeedup (api_total / direct_total):")
        print(speedup_df.to_string())
        speedup_df.to_csv(out_dir / "benchmark_speedup_summary.csv")


if __name__ == "__main__":
    main()
