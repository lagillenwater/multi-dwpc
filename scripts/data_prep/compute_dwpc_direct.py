#!/usr/bin/env python3
"""
Compute DWPC directly from HetMat matrices without the Docker API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Setup repo root
if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT / "src"))

from dwpc_direct import HetMat, load_metapath_stats, reverse_metapath_abbrev  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "dwpc_direct" / "all_GO_positive_growth"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_INTERMEDIATE_DIR = REPO_ROOT / "output" / "intermediate"
DEFAULT_DAMPING = 0.5
DEFAULT_N_WORKERS = 4
BASE_NAME = "all_GO_positive_growth"


def _load_metapaths(data_dir: Path) -> list[str]:
    metapath_stats = load_metapath_stats(data_dir)
    bp_to_g = metapath_stats[
        metapath_stats["metapath"].str.startswith("BP")
        & metapath_stats["metapath"].str.endswith("G")
    ]
    bp_to_g = bp_to_g[bp_to_g["metapath"] != "BPpG"]
    return [reverse_metapath_abbrev(mp) for mp in sorted(bp_to_g["metapath"].unique())]


def _discover_datasets(repo_root: Path, intermediate_dir: Path) -> list[dict]:
    datasets = [
        {
            "name": f"{BASE_NAME}_2016_real",
            "path": intermediate_dir / "hetio_bppg_all_GO_positive_growth_filtered.csv",
            "gene_col": "entrez_gene_id",
            "go_col": "go_id",
            "year": 2016,
            "type": "real",
            "control": "real",
            "replicate": 0,
        },
        {
            "name": f"{BASE_NAME}_2024_real",
            "path": intermediate_dir / "hetio_bppg_all_GO_positive_growth_2024_filtered.csv",
            "gene_col": "entrez_gene_id",
            "go_col": "go_id",
            "year": 2024,
            "type": "real",
            "control": "real",
            "replicate": 0,
        },
    ]

    perm_dir_2016 = repo_root / "output" / "permutations" / f"{BASE_NAME}_2016"
    perm_dir_2024 = repo_root / "output" / "permutations" / f"{BASE_NAME}_2024"
    random_dir_2016 = repo_root / "output" / "random_samples" / f"{BASE_NAME}_2016"
    random_dir_2024 = repo_root / "output" / "random_samples" / f"{BASE_NAME}_2024"

    for perm_file in sorted(perm_dir_2016.glob("perm_*.csv")) if perm_dir_2016.exists() else []:
        perm_num = perm_file.stem.split("_")[1]
        datasets.append(
            {
                "name": f"{BASE_NAME}_2016_perm_{perm_num}",
                "path": perm_file,
                "gene_col": "entrez_gene_id",
                "go_col": "go_id",
                "year": 2016,
                "type": "permuted",
                "control": "permuted",
                "replicate": int(perm_num),
            }
        )

    for perm_file in sorted(perm_dir_2024.glob("perm_*.csv")) if perm_dir_2024.exists() else []:
        perm_num = perm_file.stem.split("_")[1]
        datasets.append(
            {
                "name": f"{BASE_NAME}_2024_perm_{perm_num}",
                "path": perm_file,
                "gene_col": "entrez_gene_id",
                "go_col": "go_id",
                "year": 2024,
                "type": "permuted",
                "control": "permuted",
                "replicate": int(perm_num),
            }
        )

    for random_file in sorted(random_dir_2016.glob("random_*.csv")) if random_dir_2016.exists() else []:
        random_num = random_file.stem.split("_")[1]
        datasets.append(
            {
                "name": f"{BASE_NAME}_2016_random_{random_num}",
                "path": random_file,
                "gene_col": "entrez_gene_id",
                "go_col": "go_id",
                "year": 2016,
                "type": "random",
                "control": "random",
                "replicate": int(random_num),
            }
        )

    for random_file in sorted(random_dir_2024.glob("random_*.csv")) if random_dir_2024.exists() else []:
        random_num = random_file.stem.split("_")[1]
        datasets.append(
            {
                "name": f"{BASE_NAME}_2024_random_{random_num}",
                "path": random_file,
                "gene_col": "entrez_gene_id",
                "go_col": "go_id",
                "year": 2024,
                "type": "random",
                "control": "random",
                "replicate": int(random_num),
            }
        )

    return datasets


def summarize_dataset_results(config: dict, results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "domain",
                "name",
                "control",
                "replicate",
                "year",
                "go_id",
                "metapath",
                "mean_score",
                "n_pairs",
            ]
        )

    summary_df = (
        results_df.groupby(["go_id", "metapath"], as_index=False)["dwpc"]
        .agg(mean_score="mean", n_pairs="size")
    )
    summary_df.insert(0, "year", int(config["year"]))
    summary_df.insert(0, "replicate", int(config["replicate"]))
    summary_df.insert(0, "control", str(config["control"]))
    summary_df.insert(0, "name", str(config["name"]))
    summary_df.insert(0, "domain", "year")
    return summary_df


def _prepare_dataset_indices(
    config: dict,
    gene_id_to_idx: dict,
    bp_id_to_idx: dict,
) -> dict | None:
    """Load a dataset's GO-gene pairs and map identifiers to matrix indices.

    Returns None if the source file is missing.
    """
    path = config["path"]
    if not path.exists():
        return None

    df = pd.read_csv(path)
    n_pairs = len(df)
    df["source_idx"] = df[config["gene_col"]].map(gene_id_to_idx)
    df["target_idx"] = df[config["go_col"]].map(bp_id_to_idx)

    n_unmapped_genes = int(df["source_idx"].isna().sum())
    n_unmapped_bps = int(df["target_idx"].isna().sum())
    if n_unmapped_genes > 0 or n_unmapped_bps > 0:
        print(f"  Warning [{config['name']}]: {n_unmapped_genes} unmapped genes, {n_unmapped_bps} unmapped BPs")

    df_mapped = df.dropna(subset=["source_idx", "target_idx"]).copy()
    df_mapped["source_idx"] = df_mapped["source_idx"].astype(int)
    df_mapped["target_idx"] = df_mapped["target_idx"].astype(int)

    return {
        "n_input_pairs": n_pairs,
        "n_mapped_pairs": len(df_mapped),
        "source_indices": df_mapped["source_idx"].values,
        "target_indices": df_mapped["target_idx"].values,
        "gene_ids": df_mapped[config["gene_col"]].values,
        "go_ids": df_mapped[config["go_col"]].values,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DWPC directly from HetMat matrices.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="HetMat data directory.")
    parser.add_argument(
        "--intermediate-dir",
        default=str(DEFAULT_INTERMEDIATE_DIR),
        help="Directory containing filtered GO-gene input datasets.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--damping", type=float, default=DEFAULT_DAMPING, help="DWPC damping exponent.")
    parser.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS, help="Parallel workers for matrix precompute.")
    parser.add_argument(
        "--read-only-cache",
        action="store_true",
        help="Read cached DWPC matrices from disk but do not write new cache files.",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List dataset names in deterministic order and exit.")
    parser.add_argument("--list-metapaths", action="store_true", help="List year metapaths in deterministic order and exit.")
    parser.add_argument("--dataset-name", default=None, help="Process exactly one named dataset.")
    parser.add_argument("--warmup-metapath", default=None, help="Warm exactly one metapath cache entry, then exit.")
    parser.add_argument(
        "--warmup-cache",
        action="store_true",
        help="Precompute and cache all year metapath DWPC matrices, then exit without processing datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    intermediate_dir = Path(args.intermediate_dir)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    summaries_dir = output_dir / "replicate_summaries"
    manifest_path = output_dir / "replicate_manifest.csv"

    metapaths = _load_metapaths(data_dir)
    if args.list_metapaths:
        for metapath in metapaths:
            print(metapath)
        return

    datasets = _discover_datasets(REPO_ROOT, intermediate_dir)
    if args.list_datasets:
        for dataset in datasets:
            print(dataset["name"])
        return

    if args.dataset_name:
        matches = [d for d in datasets if d["name"] == args.dataset_name]
        if not matches:
            available = "\n  - ".join(d["name"] for d in datasets)
            raise ValueError(
                f"Unknown dataset name: {args.dataset_name}\nAvailable datasets:\n  - {available}"
            )
        datasets = matches

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Data directory: {data_dir}")
    print(f"Intermediate directory: {intermediate_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Damping exponent: {args.damping}")
    print(f"Parallel workers: {args.n_workers}")
    print(f"Configured datasets: {len(datasets)}")
    for ds in datasets:
        status = "EXISTS" if Path(ds["path"]).exists() else "MISSING"
        print(f"  {ds['name']}: {status}")

    print(f"Computing {len(metapaths)} metapaths")

    hetmat = HetMat(
        data_dir,
        damping=float(args.damping),
        use_disk_cache=True,
        write_disk_cache=not args.read_only_cache,
    )
    gene_nodes = hetmat.get_nodes("Gene")
    bp_nodes = hetmat.get_nodes("Biological Process")
    gene_id_to_idx = dict(zip(gene_nodes["identifier"], gene_nodes["position"]))
    bp_id_to_idx = dict(zip(bp_nodes["identifier"], bp_nodes["position"]))

    if args.warmup_metapath:
        if args.warmup_metapath not in metapaths:
            available = "\n  - ".join(metapaths)
            raise ValueError(
                f"Unknown metapath for warmup: {args.warmup_metapath}\nAvailable metapaths:\n  - {available}"
            )
        metapaths = [args.warmup_metapath]

    if args.warmup_cache or args.warmup_metapath:
        print("Precomputing DWPC matrices...")
        precompute_start = time.perf_counter()
        hetmat.precompute_matrices(metapaths, n_workers=int(args.n_workers), show_progress=True)
        print(f"Precomputation complete: {time.perf_counter() - precompute_start:.1f}s")
        if args.warmup_metapath:
            print(f"Cache warmup complete for metapath {args.warmup_metapath}; exiting without dataset processing.")
        else:
            print("Cache warmup complete; exiting without dataset processing.")
        return

    partials_dir = results_dir / "_partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    manifest_rows = []
    active_configs = []

    for config in datasets:
        output_path = results_dir / f"dwpc_{config['name']}.csv"
        summary_path = summaries_dir / f"summary_{config['name']}.csv"
        manifest_rows.append(
            {
                "domain": "year",
                "name": config["name"],
                "control": config["control"],
                "replicate": int(config["replicate"]),
                "year": int(config["year"]),
                "source_path": str(config["path"]),
                "result_path": str(output_path),
                "summary_path": str(summary_path),
            }
        )
        if output_path.exists() and summary_path.exists():
            existing_df = pd.read_csv(output_path)
            print(f"  Already processed: {config['name']} ({len(existing_df)} rows)")
            all_summaries.append(
                {"name": config["name"], "status": "skipped (exists)", "n_results": len(existing_df)}
            )
            continue

        prep = _prepare_dataset_indices(config, gene_id_to_idx, bp_id_to_idx)
        if prep is None:
            print(f"  Skipping {config['name']}: file not found")
            all_summaries.append({"name": config["name"], "status": "skipped", "reason": "file not found"})
            continue

        active_configs.append(
            {
                **config,
                **prep,
                "output_path": output_path,
                "summary_path": summary_path,
                "partial_dir": partials_dir / config["name"],
            }
        )

    print(f"\nActive datasets to compute: {len(active_configs)}")
    print(f"Already complete: {len(datasets) - len(active_configs)}")

    batch_start = time.perf_counter()

    if active_configs:
        print(
            f"\nIterating {len(metapaths)} metapaths over {len(active_configs)} datasets "
            f"(one matrix in memory at a time)..."
        )
        for mp_idx, metapath in enumerate(metapaths, start=1):
            mp_start = time.perf_counter()
            print(f"\n[metapath {mp_idx}/{len(metapaths)}] {metapath}")
            for config in active_configs:
                partial_path = config["partial_dir"] / f"{metapath}.csv"
                if partial_path.exists():
                    continue
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dwpc_values = hetmat.get_dwpc_for_pairs(
                        metapath, config["source_indices"], config["target_indices"]
                    )
                    pd.DataFrame(
                        {
                            "entrez_gene_id": config["gene_ids"],
                            "go_id": config["go_ids"],
                            "source_idx": config["source_indices"],
                            "target_idx": config["target_indices"],
                            "metapath": metapath,
                            "dwpc": dwpc_values,
                        }
                    ).to_csv(partial_path, index=False)
                except Exception as exc:
                    print(f"  Error {config['name']} x {metapath}: {exc}")
            hetmat.clear_metapath_from_memory(metapath)
            print(f"  done in {time.perf_counter() - mp_start:.1f}s; matrix cleared from RAM")

        print("\nConcatenating partials into per-dataset result files...")
        for config in active_configs:
            partial_dir = config["partial_dir"]
            partial_files = sorted(partial_dir.glob("*.csv")) if partial_dir.exists() else []
            if not partial_files:
                all_summaries.append(
                    {"name": config["name"], "status": "no partials written"}
                )
                continue
            results_df = pd.concat(
                [pd.read_csv(p) for p in partial_files], ignore_index=True
            )
            results_df.to_csv(config["output_path"], index=False)
            summarize_dataset_results(config, results_df).to_csv(config["summary_path"], index=False)
            for p in partial_files:
                p.unlink()
            partial_dir.rmdir()
            print(f"  {config['name']}: {len(results_df)} rows -> {config['output_path']}")
            all_summaries.append(
                {
                    "name": config["name"],
                    "status": "completed",
                    "n_input_pairs": config["n_input_pairs"],
                    "n_mapped_pairs": config["n_mapped_pairs"],
                    "n_results": len(results_df),
                }
            )

    try:
        partials_dir.rmdir()
    except OSError:
        pass

    summary_df = pd.DataFrame(all_summaries)
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["year", "control", "replicate", "name"])
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nBatch complete: {(time.perf_counter() - batch_start) / 60:.1f} minutes")
    if not summary_df.empty:
        print("\nProcessing Summary:")
        print(summary_df.to_string(index=False))
    print(f"\nReplicate manifest: {manifest_path}")


if __name__ == "__main__":
    main()
