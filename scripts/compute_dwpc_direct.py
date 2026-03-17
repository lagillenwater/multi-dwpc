#!/usr/bin/env python3
"""
Compute DWPC directly from HetMat matrices without the Docker API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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


def process_dataset(
    config: dict,
    hetmat: HetMat,
    metapaths: list[str],
    gene_id_to_idx: dict,
    bp_id_to_idx: dict,
    output_path: Path,
    summary_path: Path,
) -> dict:
    name = config["name"]
    path = config["path"]

    if not path.exists():
        return {"name": name, "status": "skipped", "reason": "file not found"}

    df = pd.read_csv(path)
    n_pairs = len(df)
    df["source_idx"] = df[config["gene_col"]].map(gene_id_to_idx)
    df["target_idx"] = df[config["go_col"]].map(bp_id_to_idx)

    n_unmapped_genes = int(df["source_idx"].isna().sum())
    n_unmapped_bps = int(df["target_idx"].isna().sum())
    if n_unmapped_genes > 0 or n_unmapped_bps > 0:
        print(f"  Warning: {n_unmapped_genes} unmapped genes, {n_unmapped_bps} unmapped BPs")

    df_mapped = df.dropna(subset=["source_idx", "target_idx"]).copy()
    df_mapped["source_idx"] = df_mapped["source_idx"].astype(int)
    df_mapped["target_idx"] = df_mapped["target_idx"].astype(int)

    source_indices = df_mapped["source_idx"].values
    target_indices = df_mapped["target_idx"].values
    gene_ids = df_mapped[config["gene_col"]].values
    go_ids = df_mapped[config["go_col"]].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_summary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    if tmp_summary_path.exists():
        tmp_summary_path.unlink()
    summary_rows = []
    write_header = True
    total_results = 0

    for metapath in metapaths:
        try:
            dwpc_values = hetmat.get_dwpc_for_pairs(metapath, source_indices, target_indices)
            metapath_df = pd.DataFrame(
                {
                    "entrez_gene_id": gene_ids,
                    "go_id": go_ids,
                    "source_idx": source_indices,
                    "target_idx": target_indices,
                    "metapath": metapath,
                    "dwpc": dwpc_values,
                }
            )
            metapath_df.to_csv(
                tmp_output_path,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
            )
            write_header = False
            total_results += len(metapath_df)

            grouped = (
                metapath_df.groupby("go_id", as_index=False)["dwpc"]
                .agg(mean_score="mean", n_pairs="size")
            )
            grouped.insert(0, "metapath", metapath)
            summary_rows.append(grouped)
        except Exception as exc:
            print(f"  Error computing {metapath}: {exc}")
        finally:
            hetmat.clear_metapath_from_memory(metapath)

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_df.insert(0, "year", int(config["year"]))
        summary_df.insert(0, "replicate", int(config["replicate"]))
        summary_df.insert(0, "control", str(config["control"]))
        summary_df.insert(0, "name", str(config["name"]))
        summary_df.insert(0, "domain", "year")
        summary_df.to_csv(tmp_summary_path, index=False)
    else:
        summary_df = pd.DataFrame()

    if total_results > 0 and tmp_output_path.exists():
        tmp_output_path.replace(output_path)
    if len(summary_df) > 0 and tmp_summary_path.exists():
        tmp_summary_path.replace(summary_path)

    return {
        "name": name,
        "status": "completed",
        "n_input_pairs": n_pairs,
        "n_mapped_pairs": len(df_mapped),
        "n_results": int(total_results),
        "n_summary_rows": int(len(summary_df)),
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
    parser.add_argument("--list-datasets", action="store_true", help="List dataset names in deterministic order and exit.")
    parser.add_argument("--dataset-name", default=None, help="Process exactly one named dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    intermediate_dir = Path(args.intermediate_dir)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    summaries_dir = output_dir / "replicate_summaries"
    manifest_path = output_dir / "replicate_manifest.csv"

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

    metapaths = _load_metapaths(data_dir)
    print(f"Computing {len(metapaths)} metapaths")

    hetmat = HetMat(data_dir, damping=float(args.damping), use_disk_cache=True)
    gene_nodes = hetmat.get_nodes("Gene")
    bp_nodes = hetmat.get_nodes("Biological Process")
    gene_id_to_idx = dict(zip(gene_nodes["identifier"], gene_nodes["position"]))
    bp_id_to_idx = dict(zip(bp_nodes["identifier"], bp_nodes["position"]))

    print("Streaming metapaths without upfront global precompute...")

    all_summaries = []
    manifest_rows = []
    batch_start = time.perf_counter()
    for idx, config in enumerate(datasets, start=1):
        print(f"\n[{idx}/{len(datasets)}] Processing {config['name']}...")
        output_path = results_dir / f"dwpc_{config['name']}.csv"
        summary_path = summaries_dir / f"summary_{config['name']}.csv"
        manifest_row = {
            "domain": "year",
            "name": config["name"],
            "control": config["control"],
            "replicate": int(config["replicate"]),
            "year": int(config["year"]),
            "source_path": str(config["path"]),
            "result_path": str(output_path),
            "summary_path": str(summary_path),
        }
        if output_path.exists() and summary_path.exists():
            existing_df = pd.read_csv(output_path)
            print(f"  Already processed: {len(existing_df)} rows")
            all_summaries.append(
                {"name": config["name"], "status": "skipped (exists)", "n_results": len(existing_df)}
            )
            manifest_rows.append(manifest_row)
            continue
        if output_path.exists() and not summary_path.exists():
            print(f"  Found stale partial output without summary; recomputing {output_path.name}")

        start = time.perf_counter()
        summary = process_dataset(
            config=config,
            hetmat=hetmat,
            metapaths=metapaths,
            gene_id_to_idx=gene_id_to_idx,
            bp_id_to_idx=bp_id_to_idx,
            output_path=output_path,
            summary_path=summary_path,
        )
        if summary["status"] == "completed":
            elapsed = time.perf_counter() - start
            print(f"  Saved: {output_path}")
            print(f"  Saved: {summary_path}")
            print(f"  Rows: {summary['n_results']}")
            print(f"  Time: {elapsed:.1f}s")
            summary["time_seconds"] = elapsed
        all_summaries.append(summary)
        manifest_rows.append(manifest_row)

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
