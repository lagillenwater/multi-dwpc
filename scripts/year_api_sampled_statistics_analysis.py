#!/usr/bin/env python3
"""Local sampled year-API permutation sensitivity workflow."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.bipartite_nulls import degree_preserving_permutations  # noqa: E402
from src.dwpc_api import run_metapaths_for_df, validate_parquet_files  # noqa: E402
from src.replicate_analysis import build_b_seed_runs, summarize_feature_variance, summarize_overall_variance  # noqa: E402
from src.result_normalization import load_neo4j_mappings, normalize_api_year_result  # noqa: E402
from src.year_statistics import (  # noqa: E402
    build_aggregated_year_statistics_panel,
    build_year_statistic_summary_long,
)


DEFAULT_BASE_NAME = "all_GO_positive_growth"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "year_api_permutation_sensitivity"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_INTERMEDIATE_DIR = REPO_ROOT / "output" / "intermediate"

MAX_CONCURRENCY = 120
RETRIES = 10
BACKOFF_FIRST = 10.0
FINAL_MIN_COMPLETION_RATE = 0.2


def _parse_int_list(arg: str) -> list[int]:
    values = sorted({int(tok.strip()) for tok in str(arg).split(",") if tok.strip()})
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _load_year_edge_list(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"go_id", "entrez_gene_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df = df[["go_id", "entrez_gene_id"]].drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No GO-gene rows found in {path}")
    return df


def _sample_go_terms(real_2016: pd.DataFrame, real_2024: pd.DataFrame, n_go_terms: int, seed: int) -> list[str]:
    pool = sorted(set(real_2016["go_id"].astype(str)) & set(real_2024["go_id"].astype(str)))
    if len(pool) < int(n_go_terms):
        pool = sorted(set(real_2016["go_id"].astype(str)) | set(real_2024["go_id"].astype(str)))
    if not pool:
        raise ValueError("No GO terms available for sampled year API sensitivity workflow.")
    sample_n = min(int(n_go_terms), len(pool))
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(np.arange(len(pool), dtype=int), size=sample_n, replace=False)
    return sorted([pool[int(i)] for i in idx])


def _validate_permutation(real_df: pd.DataFrame, perm_df: pd.DataFrame, year: int, rep_id: int) -> None:
    real_sizes = real_df.groupby("go_id").size().sort_index()
    perm_sizes = perm_df.groupby("go_id").size().sort_index()
    if not real_sizes.equals(perm_sizes):
        raise ValueError(f"GO term sizes changed for year={year}, replicate={rep_id}")

    real_deg = real_df.groupby("entrez_gene_id")["go_id"].nunique().sort_index()
    perm_deg = perm_df.groupby("entrez_gene_id")["go_id"].nunique().sort_index()
    if not real_deg.equals(perm_deg):
        raise ValueError(f"Gene annotation degrees changed for year={year}, replicate={rep_id}")

    if len(perm_df[["go_id", "entrez_gene_id"]].drop_duplicates()) != len(perm_df):
        raise ValueError(f"Duplicate GO-gene edges introduced for year={year}, replicate={rep_id}")


def _load_neo4j_lookup_maps(data_dir: Path) -> tuple[dict, dict]:
    gene_map = pd.read_csv(data_dir / "neo4j_gene_mapping.csv")
    bp_map = pd.read_csv(data_dir / "neo4j_bp_mapping.csv")
    entrez_to_neo4j = dict(zip(gene_map["identifier"], gene_map["neo4j_id"]))
    go_to_neo4j = dict(zip(bp_map["identifier"], bp_map["neo4j_id"]))
    return entrez_to_neo4j, go_to_neo4j


def _attach_neo4j_ids(df: pd.DataFrame, entrez_to_neo4j: dict, go_to_neo4j: dict) -> pd.DataFrame:
    out = df.copy()
    out["neo4j_source_id"] = out["go_id"].map(go_to_neo4j)
    out["neo4j_target_id"] = out["entrez_gene_id"].map(entrez_to_neo4j)
    before = len(out)
    out = out.dropna(subset=["neo4j_source_id", "neo4j_target_id"]).copy()
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped:,} rows missing Neo4j IDs")
    out["neo4j_source_id"] = out["neo4j_source_id"].astype(int)
    out["neo4j_target_id"] = out["neo4j_target_id"].astype(int)
    return out


def _load_manifest_normalized_results(manifest_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    neo4j_to_entrez, neo4j_to_go = load_neo4j_mappings(data_dir)
    frames: list[pd.DataFrame] = []
    for row in manifest_df.itertuples(index=False):
        result_path = Path(row.result_path)
        if not result_path.exists():
            continue
        raw_df = pd.read_csv(result_path)
        if "neo4j_target_id" not in raw_df.columns and "neo4j_pseudo_target_id" in raw_df.columns:
            raw_df = raw_df.copy()
            raw_df["neo4j_target_id"] = raw_df["neo4j_pseudo_target_id"]
        norm_df = normalize_api_year_result(
            raw_df,
            dataset_name=str(row.name),
            neo4j_to_entrez=neo4j_to_entrez,
            neo4j_to_go=neo4j_to_go,
        )
        if not norm_df.empty:
            frames.append(norm_df)
    if not frames:
        raise ValueError("No normalized API result rows were loaded from the sampled dataset manifest.")
    return pd.concat(frames, ignore_index=True)


def _merge_group_parquets(group_dir: Path, output_csv: Path, expected_pairs: int, min_completion_rate: float) -> pd.DataFrame:
    validate_parquet_files(
        group_dir,
        expected_pairs=expected_pairs,
        min_completion_rate=float(min_completion_rate),
        show_progress=True,
    )
    parquet_files = sorted(group_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {group_dir}")
    frames = [pd.read_parquet(path) for path in parquet_files]
    res_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    return res_df


async def _lookup_dataset(
    dataset_row: pd.Series,
    *,
    data_dir: Path,
    parquet_dir: Path,
    min_completion_rate: float,
    max_concurrency: int,
    retries: int,
    backoff_first: float,
    cleanup_parquet: bool,
) -> None:
    csv_path = Path(dataset_row["result_path"])
    if csv_path.exists():
        print(f"[skip] {csv_path}")
        return

    input_path = Path(dataset_row["input_path"])
    group = str(dataset_row["name"])
    raw_df = pd.read_csv(input_path)
    entrez_to_neo4j, go_to_neo4j = _load_neo4j_lookup_maps(data_dir)
    api_df = _attach_neo4j_ids(raw_df, entrez_to_neo4j=entrez_to_neo4j, go_to_neo4j=go_to_neo4j)
    if api_df.empty:
        raise ValueError(f"No valid Neo4j-mapped rows for {group}")

    summary = await run_metapaths_for_df(
        api_df,
        col_source="neo4j_source_id",
        col_target="neo4j_target_id",
        base_out_dir=parquet_dir,
        group=group,
        clear_group=False,
        max_concurrency=int(max_concurrency),
        retries=int(retries),
        backoff_first=float(backoff_first),
    )
    group_dir = parquet_dir / group
    expected_pairs = int(api_df[["neo4j_source_id", "neo4j_target_id"]].drop_duplicates().shape[0])
    res_df = _merge_group_parquets(
        group_dir,
        output_csv=csv_path,
        expected_pairs=expected_pairs,
        min_completion_rate=float(min_completion_rate),
    )
    print(f"[saved] {csv_path} ({len(res_df):,} rows; {len(summary):,} pair summaries)")

    if cleanup_parquet:
        shutil.rmtree(group_dir, ignore_errors=True)


def _plot_sensitivity(overall_var_df: pd.DataFrame, out_pdf: Path) -> None:
    if overall_var_df.empty:
        return
    years = sorted(overall_var_df["year"].astype(int).unique().tolist())
    controls = sorted(overall_var_df["control"].astype(str).unique().tolist())
    if not years or not controls:
        return

    fig, axes = plt.subplots(len(years), len(controls), figsize=(6.5 * len(controls), 4.5 * len(years)), sharex=True)
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        if len(years) == 1:
            axes = axes[np.newaxis, :]
        else:
            axes = axes[:, np.newaxis]

    for i, year in enumerate(years):
        for j, control in enumerate(controls):
            ax = axes[i, j]
            subset = overall_var_df[
                (overall_var_df["year"].astype(int) == int(year))
                & (overall_var_df["control"].astype(str) == str(control))
            ].copy()
            for statistic, stat_df in subset.groupby("statistic", sort=True):
                stat_df = stat_df.sort_values("b")
                ax.plot(
                    stat_df["b"].astype(int),
                    stat_df["mean_diff_std"].astype(float),
                    marker="o",
                    linewidth=1.6,
                    alpha=0.9,
                    label=str(statistic),
                )
            ax.grid(alpha=0.25)
            ax.set_xlabel("B")
            ax.set_ylabel("Mean diff SD")
            ax.set_title(f"{year} | {control}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Statistic")
    fig.suptitle("API permutation sensitivity by statistic")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--intermediate-dir", default=str(DEFAULT_INTERMEDIATE_DIR))
    parser.add_argument("--base-name", default=DEFAULT_BASE_NAME)
    parser.add_argument("--n-go-terms", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--b-values", default="1,2,5,10,20,50,100")
    parser.add_argument("--resample-seeds", default="11,22,33,44,55")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--retries", type=int, default=RETRIES)
    parser.add_argument("--backoff-first", type=float, default=BACKOFF_FIRST)
    parser.add_argument("--min-completion-rate", type=float, default=FINAL_MIN_COMPLETION_RATE)
    parser.add_argument("--keep-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) if args.results_dir else output_dir / "results"
    data_dir = Path(args.data_dir)
    intermediate_dir = Path(args.intermediate_dir)
    base_name = str(args.base_name)
    n_go_terms = int(args.n_go_terms)
    n_permutations = int(args.n_permutations)

    inputs_dir = output_dir / "replicate_inputs"
    parquet_dir = output_dir / "metapaths_parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    real_2016 = _load_year_edge_list(intermediate_dir / f"hetio_bppg_{base_name}_filtered.csv")
    real_2024 = _load_year_edge_list(intermediate_dir / f"hetio_bppg_{base_name}_2024_filtered.csv")

    sampled_go_terms = _sample_go_terms(real_2016, real_2024, n_go_terms=n_go_terms, seed=int(args.sample_seed))
    pd.DataFrame({"go_id": sampled_go_terms}).to_csv(output_dir / "sampled_go_terms.csv", index=False)

    manifest_rows: list[dict] = []
    sampled_base = f"{base_name}_sample{n_go_terms}"
    year_real_map = {
        2016: real_2016[real_2016["go_id"].astype(str).isin(sampled_go_terms)].copy(),
        2024: real_2024[real_2024["go_id"].astype(str).isin(sampled_go_terms)].copy(),
    }

    for year, sampled_real_df in year_real_map.items():
        real_name = f"{sampled_base}_{year}_real"
        real_input_path = inputs_dir / f"{real_name}.csv"
        if not real_input_path.exists():
            sampled_real_df.to_csv(real_input_path, index=False)
        manifest_rows.append(
            {
                "name": real_name,
                "year": int(year),
                "control": "real",
                "replicate": 0,
                "input_path": str(real_input_path),
                "result_path": str(results_dir / f"res_{real_name}.csv"),
                "n_input_rows": int(len(sampled_real_df)),
            }
        )

        perm_dir = inputs_dir / f"{sampled_base}_{year}_permuted"
        perm_dir.mkdir(parents=True, exist_ok=True)
        for rep_id in range(1, n_permutations + 1):
            perm_name = f"{sampled_base}_{year}_perm_{rep_id:03d}"
            perm_input_path = perm_dir / f"perm_{rep_id:03d}.csv"
            if not perm_input_path.exists():
                perm_df = degree_preserving_permutations(
                    edge_df=sampled_real_df,
                    source_col="go_id",
                    target_col="entrez_gene_id",
                    n_permutations=1,
                    random_state=42 + int(rep_id) - 1,
                    n_swap_attempts_per_edge=10,
                )[0]
                _validate_permutation(sampled_real_df, perm_df, year=year, rep_id=rep_id)
                perm_df.to_csv(perm_input_path, index=False)
            manifest_rows.append(
                {
                    "name": perm_name,
                    "year": int(year),
                    "control": "permuted",
                    "replicate": int(rep_id),
                    "input_path": str(perm_input_path),
                    "result_path": str(results_dir / f"res_{perm_name}.csv"),
                    "n_input_rows": int(len(sampled_real_df)),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["year", "control", "replicate"]).reset_index(drop=True)
    manifest_df.to_csv(output_dir / "dataset_manifest.csv", index=False)

    for row in manifest_df.itertuples(index=False):
        asyncio.run(
            _lookup_dataset(
                pd.Series(row._asdict()),
                data_dir=data_dir,
                parquet_dir=parquet_dir,
                min_completion_rate=float(args.min_completion_rate),
                max_concurrency=int(args.max_concurrency),
                retries=int(args.retries),
                backoff_first=float(args.backoff_first),
                cleanup_parquet=not args.keep_parquet,
            )
        )

    normalized_df = _load_manifest_normalized_results(manifest_df, data_dir=data_dir)
    agg_df = build_aggregated_year_statistics_panel(normalized_df)
    summary_df = build_year_statistic_summary_long(agg_df)
    summary_df = summary_df[summary_df["control"].astype(str).isin(["real", "permuted"])].copy()
    summary_df.to_csv(output_dir / "statistic_summary_long.csv", index=False)
    agg_df.to_csv(output_dir / "aggregated_statistics_panel.csv", index=False)

    pool_meta_df = (
        summary_df[summary_df["control"].astype(str) != "real"]
        .groupby(["year", "control", "statistic"], as_index=False)
        .agg(
            n_true_replicates=("replicate", "nunique"),
            n_go_terms=("go_id", "nunique"),
            n_metapaths=("metapath", "nunique"),
        )
    )
    pool_meta_df["target_pool_size"] = int(n_permutations)
    pool_meta_df.to_csv(output_dir / "null_pool_summary.csv", index=False)

    requested_b_values = sorted(set(_parse_int_list(args.b_values)))
    resample_seeds = sorted(set(_parse_int_list(args.resample_seeds)))

    runs_frames: list[pd.DataFrame] = []
    for statistic, stat_df in summary_df.groupby("statistic", sort=True):
        null_df = stat_df[stat_df["control"].astype(str) != "real"].copy()
        if null_df.empty:
            continue
        per_pool_counts = (
            null_df.groupby(["year", "control"], dropna=False)["replicate"]
            .nunique()
            .tolist()
        )
        if not per_pool_counts:
            continue
        max_allowed_b = int(min(per_pool_counts))
        b_values = [b for b in requested_b_values if b <= max_allowed_b]
        if not b_values:
            continue
        stat_runs_df = build_b_seed_runs(
            stat_df,
            b_values=b_values,
            seeds=resample_seeds,
            join_keys=["year", "go_id", "metapath"],
            replicate_pool_keys=["year", "control"],
        )
        if stat_runs_df.empty:
            continue
        stat_runs_df.insert(0, "statistic", str(statistic))
        runs_frames.append(stat_runs_df)

    if not runs_frames:
        raise ValueError("No B/seed runs were generated. Check available API permutation replicates and B values.")

    runs_df = pd.concat(runs_frames, ignore_index=True)
    runs_df.to_csv(output_dir / "b_seed_runs_long.csv", index=False)

    feature_df = summarize_feature_variance(
        runs_df,
        feature_keys=["statistic", "year", "control", "b", "go_id", "metapath"],
    )
    overall_df = summarize_overall_variance(
        feature_df,
        overall_keys=["statistic", "year", "control", "b"],
        runs_df=runs_df,
        replicate_col="seed",
    )
    feature_df.to_csv(output_dir / "feature_variance_summary.csv", index=False)
    overall_df.to_csv(output_dir / "overall_variance_summary.csv", index=False)

    sens_rows: list[dict] = []
    for (statistic, year, control), grp in overall_df.groupby(["statistic", "year", "control"], sort=True):
        grp = grp.sort_values("b")
        if grp.empty:
            continue
        first = grp.iloc[0]
        last = grp.iloc[-1]
        sens_rows.append(
            {
                "statistic": str(statistic),
                "year": int(year),
                "control": str(control),
                "b_min": int(first["b"]),
                "b_max": int(last["b"]),
                "mean_diff_std_b_min": float(first["mean_diff_std"]),
                "mean_diff_std_b_max": float(last["mean_diff_std"]),
                "mean_diff_var_b_min": float(first["mean_diff_var"]),
                "mean_diff_var_b_max": float(last["mean_diff_var"]),
            }
        )
    pd.DataFrame(sens_rows).to_csv(output_dir / "statistic_sensitivity_summary.csv", index=False)

    _plot_sensitivity(overall_df, output_dir / "variance_sensitivity_by_statistic.pdf")

    print(f"Sampled GO terms: {len(sampled_go_terms)}")
    print(f"Requested permutation pool size: {n_permutations}")
    print(f"Available result files: {results_dir}")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
