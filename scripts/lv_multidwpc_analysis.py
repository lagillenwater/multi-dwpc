"""
LV multi-DWPC analysis orchestrator.
- load LV loadings
- select top 0.5% genes per LV
- emit top genes and summary tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup repo root
if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_inputs import extract_top_lv_genes  # noqa: E402
from src.lv_nulls import run_vectorized_nulls  # noqa: E402
from src.lv_pairs import build_lv_target_pairs  # noqa: E402
from src.lv_precompute import precompute_gene_feature_scores  # noqa: E402
from src.lv_stats import build_final_stats  # noqa: E402
from src.lv_subgraphs import extract_top_subgraphs, plot_top_subgraphs  # noqa: E402
from src.lv_targets import build_target_sets  # noqa: E402
from src.lv_dwpc import compute_real_pair_dwpc  # noqa: E402


DEFAULT_LVS = ("LV603", "LV246", "LV57")


def _parse_lvs(lv_arg: str) -> list[str]:
    tokens = [token.strip() for token in lv_arg.split(",")]
    return [token for token in tokens if token]


def _all_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _build_smoke_loadings(
    output_path: Path,
    gene_reference_path: Path,
    lv_ids: list[str],
    rows_per_lv: int = 800,
    seed: int = 13,
) -> Path:
    gene_df = pd.read_csv(gene_reference_path, sep="\t")
    if len(gene_df) < rows_per_lv:
        rows_per_lv = len(gene_df)

    base_genes = gene_df.head(rows_per_lv)["name"].tolist()
    rng = np.random.default_rng(seed)

    records = []
    for index, lv_id in enumerate(lv_ids):
        lv_seed_shift = (index + 1) * 0.15
        noise = rng.normal(loc=0.0, scale=1.0, size=rows_per_lv)
        loadings = noise + lv_seed_shift
        for gene_symbol, loading in zip(base_genes, loadings):
            records.append(
                {
                    "lv_id": lv_id,
                    "gene_symbol": gene_symbol,
                    "loading": float(loading),
                }
            )

    smoke_df = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_df.to_csv(output_path, index=False)
    return output_path


def run_top_genes_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_genes_path = output_dir / "lv_top_genes.csv"
    summary_path = output_dir / "lv_top_genes_summary.csv"
    cached_files = [top_genes_path, summary_path]
    if args.resume and _all_exist(cached_files) and not args.force:
        selected_df = pd.read_csv(top_genes_path)
        summary_df = pd.read_csv(summary_path)
        print("\nTop-gene extraction skipped (resume cache hit).")
        print(f"  Output: {top_genes_path}")
        print(f"  Summary: {summary_path}")
        print(f"  Selected rows: {len(selected_df):,}")
        print("\nPer-LV summary:")
        print(summary_df.to_string(index=False))
        return

    gene_reference_path = Path(args.gene_reference)
    if not gene_reference_path.is_absolute():
        gene_reference_path = (REPO_ROOT / gene_reference_path).resolve()

    if args.smoke:
        smoke_input_path = output_dir / "smoke_lv_loadings.csv"
        lv_loadings_path = _build_smoke_loadings(
            output_path=smoke_input_path,
            gene_reference_path=gene_reference_path,
            lv_ids=_parse_lvs(args.lvs),
        )
        print(f"[smoke] Created synthetic LV loadings: {lv_loadings_path}")
    else:
        if not args.lv_loadings:
            raise ValueError(
                "--lv-loadings is required unless --smoke is enabled."
            )
        lv_loadings_path = Path(args.lv_loadings)
        if not lv_loadings_path.is_absolute():
            lv_loadings_path = (REPO_ROOT / lv_loadings_path).resolve()

    selected_df, summary_df = extract_top_lv_genes(
        lv_loadings_path=lv_loadings_path,
        gene_reference_path=gene_reference_path,
        output_top_genes_path=top_genes_path,
        output_summary_path=summary_path,
        requested_lvs=_parse_lvs(args.lvs),
        top_fraction=args.top_fraction,
        lv_column=args.lv_column,
        gene_column=args.gene_column,
        loading_column=args.loading_column,
    )

    print("\nTop-gene extraction complete.")
    print(f"  Selected rows: {len(selected_df):,}")
    print(f"  Unique LVs: {selected_df['lv_id'].nunique()}")
    print(f"  Output: {top_genes_path}")
    print(f"  Summary: {summary_path}")
    print("\nPer-LV summary:")
    print(summary_df.to_string(index=False))


def run_target_sets_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_dir = REPO_ROOT / "data" / "nodes"
    target_sets_path = output_dir / "target_sets.csv"
    target_summary_path = output_dir / "target_sets_summary.csv"
    lv_target_map_path = output_dir / "lv_target_map.csv"
    cached_files = [target_sets_path, target_summary_path, lv_target_map_path]
    if args.resume and _all_exist(cached_files) and not args.force:
        targets_df = pd.read_csv(target_sets_path)
        summary_df = pd.read_csv(target_summary_path)
        lv_map_df = pd.read_csv(lv_target_map_path)
        print("\nTarget-set construction skipped (resume cache hit).")
        print(f"  Target rows: {len(targets_df):,}")
        print(f"  Output: {target_sets_path}")
        print(f"  Summary: {target_summary_path}")
        print(f"  LV map: {lv_target_map_path}")
        print("\nTarget-set summary:")
        print(summary_df.to_string(index=False))
        print("\nLV-to-target map:")
        print(lv_map_df.to_string(index=False))
        return

    targets_df, summary_df, lv_map_df = build_target_sets(
        nodes_dir=nodes_dir,
        output_target_sets_path=target_sets_path,
        output_summary_path=target_summary_path,
        output_lv_map_path=lv_target_map_path,
        lv_ids=_parse_lvs(args.lvs),
        include_brown_adipose=args.include_brown_adipose,
    )

    print("\nTarget-set construction complete.")
    print(f"  Target rows: {len(targets_df):,}")
    print(f"  Output: {target_sets_path}")
    print(f"  Summary: {target_summary_path}")
    print(f"  LV map: {lv_target_map_path}")
    print("\nTarget-set summary:")
    print(summary_df.to_string(index=False))
    print("\nLV-to-target map:")
    print(lv_map_df.to_string(index=False))


def run_real_dwpc_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_genes_path = output_dir / "lv_top_genes.csv"
    target_sets_path = output_dir / "target_sets.csv"
    lv_map_path = output_dir / "lv_target_map.csv"
    pairs_path = output_dir / "lv_gene_target_pairs.csv"
    dwpc_path = output_dir / "lv_pair_dwpc_real.csv"
    manifest_path = output_dir / "lv_metapaths_manifest.csv"

    if not top_genes_path.exists():
        raise FileNotFoundError(
            f"Missing {top_genes_path}. Run stage 'top-genes' first."
        )
    if not target_sets_path.exists() or not lv_map_path.exists():
        raise FileNotFoundError(
            "Missing target-set inputs. Run stage 'target-sets' first."
        )

    pairs_df = build_lv_target_pairs(
        top_genes_path=top_genes_path,
        target_sets_path=target_sets_path,
        lv_target_map_path=lv_map_path,
        output_pairs_path=pairs_path,
    )
    print("\nPair table complete.")
    print(f"  Pair rows: {len(pairs_df):,}")
    print(f"  Output: {pairs_path}")

    metapath_limit = args.metapath_limit_per_target
    if args.smoke and metapath_limit is None:
        metapath_limit = 5
        print(
            "[smoke] Using metapath limit per target type = 5 "
            "for faster validation."
        )

    dwpc_df, manifest_df = compute_real_pair_dwpc(
        pairs_path=pairs_path,
        data_dir=REPO_ROOT / "data",
        metapath_stats_path=REPO_ROOT / "data" / "metapath-dwpc-stats.tsv",
        output_dwpc_path=dwpc_path,
        output_metapath_manifest_path=manifest_path,
        damping=args.damping,
        max_length=args.max_metapath_length,
        exclude_direct=not args.include_direct_metapaths,
        metapath_limit_per_target=metapath_limit,
        use_disk_cache=True,
    )

    print("\nReal DWPC computation complete.")
    print(f"  DWPC rows: {len(dwpc_df):,}")
    print(f"  Unique metapaths: {dwpc_df['metapath'].nunique()}")
    print(f"  Output: {dwpc_path}")
    print(f"  Metapath manifest: {manifest_path}")
    print("\nMetapaths by node type:")
    print(
        manifest_df.groupby("node_type", as_index=False)["metapath"]
        .nunique()
        .rename(columns={"metapath": "n_metapaths"})
        .to_string(index=False)
    )


def run_precompute_scores_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precompute_cache = [
        output_dir / "gene_feature_scores.npy",
        output_dir / "gene_ids.npy",
        output_dir / "feature_manifest.csv",
        output_dir / "lv_real_gene_indices.csv",
        output_dir / "real_feature_scores.csv",
    ]
    if args.resume and _all_exist(precompute_cache) and not args.force:
        feature_manifest = pd.read_csv(output_dir / "feature_manifest.csv")
        lv_real_indices = pd.read_csv(output_dir / "lv_real_gene_indices.csv")
        real_feature_scores = pd.read_csv(output_dir / "real_feature_scores.csv")
        print("\nPrecompute scores skipped (resume cache hit).")
        print(f"  Features: {len(feature_manifest):,}")
        print(f"  LV real index rows: {len(lv_real_indices):,}")
        print(f"  Real feature score rows: {len(real_feature_scores):,}")
        print(f"  Output dir: {output_dir}")
        return

    metapath_limit = args.metapath_limit_per_target
    if args.smoke and metapath_limit is None:
        metapath_limit = 5
        print(
            "[smoke] Using metapath limit per target type = 5 "
            "for faster precompute validation."
        )

    feature_manifest, lv_real_indices, real_feature_scores = precompute_gene_feature_scores(
        output_dir=output_dir,
        data_dir=REPO_ROOT / "data",
        metapath_stats_path=REPO_ROOT / "data" / "metapath-dwpc-stats.tsv",
        damping=args.damping,
        max_metapath_length=args.max_metapath_length,
        metapath_limit_per_target=metapath_limit,
        n_workers_precompute=args.n_workers_precompute,
        include_direct_metapaths=args.include_direct_metapaths,
    )

    print("\nPrecompute scores complete.")
    print(f"  Features: {len(feature_manifest):,}")
    print(f"  LV real index rows: {len(lv_real_indices):,}")
    print(f"  Real feature score rows: {len(real_feature_scores):,}")
    print(f"  Output dir: {output_dir}")


def run_nulls_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    null_cache = [
        output_dir / "null_streaming_summary.csv",
        output_dir / "null_sampling_qc.csv",
    ]
    if args.resume and _all_exist(null_cache) and not args.force:
        summary_df = pd.read_csv(output_dir / "null_streaming_summary.csv")
        qc_df = pd.read_csv(output_dir / "null_sampling_qc.csv")
        print("\nNull sampling skipped (resume cache hit).")
        print(f"  Null summary rows: {len(summary_df):,}")
        print(f"  QC rows: {len(qc_df):,}")
        print(f"  Output: {output_dir / 'null_streaming_summary.csv'}")
        print(f"  QC: {output_dir / 'null_sampling_qc.csv'}")
        return

    summary_df, qc_df = run_vectorized_nulls(
        output_dir=output_dir,
        data_dir=REPO_ROOT / "data",
        n_degree_bins=args.n_degree_bins,
        b_min=args.b_min,
        b_max=args.b_max,
        b_batch=args.b_batch,
        adaptive_p_low=args.adaptive_p_low,
        adaptive_p_high=args.adaptive_p_high,
        random_seed=args.random_seed,
    )

    print("\nNull sampling complete.")
    print(f"  Null summary rows: {len(summary_df):,}")
    print(f"  QC rows: {len(qc_df):,}")
    print(f"  Output: {output_dir / 'null_streaming_summary.csv'}")
    print(f"  QC: {output_dir / 'null_sampling_qc.csv'}")


def run_stats_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "lv_metapath_results.csv"
    if args.resume and stats_path.exists() and not args.force:
        results = pd.read_csv(stats_path)
        print("\nStats assembly skipped (resume cache hit).")
        print(f"  Result rows: {len(results):,}")
        print(f"  Supported rows: {int(results['supported'].sum()):,}")
        print(f"  Output: {stats_path}")
        return

    results = build_final_stats(output_dir=output_dir)
    print("\nStats assembly complete.")
    print(f"  Result rows: {len(results):,}")
    print(f"  Supported rows: {int(results['supported'].sum()):,}")
    print(f"  Output: {output_dir / 'lv_metapath_results.csv'}")


def run_top_subgraphs_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_pair_path = output_dir / "top_pairs.csv"
    top_path_path = output_dir / "top_paths.csv"
    if args.resume and top_pair_path.exists() and top_path_path.exists() and not args.force:
        top_pairs_df = pd.read_csv(top_pair_path)
        top_paths_df = pd.read_csv(top_path_path)
        print("\nTop-subgraph extraction skipped (resume cache hit).")
        print(f"  Top pair rows: {len(top_pairs_df):,}")
        print(f"  Top path rows: {len(top_paths_df):,}")
        print(f"  Output: {top_pair_path}")
        print(f"  Output: {top_path_path}")
        return

    top_pairs_df, top_paths_df = extract_top_subgraphs(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        top_metapaths=args.top_metapaths,
        top_pairs=args.top_pairs,
        top_paths=args.top_paths,
        damping=args.damping,
        degree_d=args.degree_d,
    )
    print("\nTop-subgraph extraction complete.")
    print(f"  Top pair rows: {len(top_pairs_df):,}")
    print(f"  Top path rows: {len(top_paths_df):,}")
    print(f"  Output: {output_dir / 'top_pairs.csv'}")
    print(f"  Output: {output_dir / 'top_paths.csv'}")


def run_plot_subgraphs_stage(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    if args.resume and plots_dir.exists() and any(plots_dir.glob("*.png")) and not args.force:
        n_existing = len(list(plots_dir.glob("*.png")))
        print("\nTop-subgraph plotting skipped (resume cache hit).")
        print(f"  Plots present: {n_existing:,}")
        print(f"  Output dir: {plots_dir}")
        return

    n_written = plot_top_subgraphs(output_dir=output_dir)
    print("\nTop-subgraph plotting complete.")
    print(f"  Plots written: {n_written:,}")
    print(f"  Output dir: {output_dir / 'plots'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LV multi-DWPC analysis pipeline.")
    parser.add_argument(
        "--stage",
        default="top-genes",
        choices=[
            "top-genes",
            "target-sets",
            "real-dwpc",
            "precompute-scores",
            "nulls",
            "stats",
            "top-subgraphs",
            "plot-subgraphs",
            "pipeline-fast",
            "pipeline-subgraphs",
            "pipeline",
        ],
        help=(
            "Pipeline stage to run. "
            "'pipeline-fast' runs optimized stages through stats."
        ),
    )
    parser.add_argument(
        "--lv-loadings",
        default=None,
        help="Path to LV loadings table (csv/tsv/parquet).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "lv_multidwpc"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--gene-reference",
        default="data/nodes/Gene.tsv",
        help="Path to Hetionet Gene.tsv reference table.",
    )
    parser.add_argument(
        "--lvs",
        default=",".join(DEFAULT_LVS),
        help="Comma-separated list of LV IDs to keep (e.g., LV603,LV246,LV57).",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.005,
        help="Fraction of mapped genes to keep per LV.",
    )
    parser.add_argument("--lv-column", default=None, help="Optional LV column name.")
    parser.add_argument("--gene-column", default=None, help="Optional gene column name.")
    parser.add_argument(
        "--loading-column", default=None, help="Optional loading column name."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate synthetic LV inputs for quick pipeline testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse stage outputs when expected artifacts already exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute stage outputs even if cached artifacts exist.",
    )
    parser.add_argument(
        "--include-brown-adipose",
        action="store_true",
        help="Include UBERON:0001348 in adipose target set.",
    )
    parser.add_argument(
        "--include-direct-metapaths",
        action="store_true",
        help="Include direct 1-edge gene->target metapaths.",
    )
    parser.add_argument(
        "--max-metapath-length",
        type=int,
        default=3,
        help="Maximum metapath length to include from metapath stats.",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="DWPC damping exponent (0.5 matches API).",
    )
    parser.add_argument(
        "--metapath-limit-per-target",
        type=int,
        default=None,
        help="Optional cap on number of metapaths per target node type.",
    )
    parser.add_argument(
        "--n-workers-precompute",
        type=int,
        default=4,
        help="Workers for precompute matrix warmup.",
    )
    parser.add_argument(
        "--n-degree-bins",
        type=int,
        default=10,
        help="Degree quantile bins for null sampling.",
    )
    parser.add_argument("--b-min", type=int, default=200, help="Adaptive B minimum.")
    parser.add_argument("--b-max", type=int, default=1000, help="Adaptive B maximum.")
    parser.add_argument("--b-batch", type=int, default=100, help="Adaptive B batch size.")
    parser.add_argument(
        "--adaptive-p-low",
        type=float,
        default=0.005,
        help="Lower threshold for adaptive stopping.",
    )
    parser.add_argument(
        "--adaptive-p-high",
        type=float,
        default=0.20,
        help="Upper threshold for adaptive stopping.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for null sampling.",
    )
    parser.add_argument(
        "--top-metapaths",
        type=int,
        default=10,
        help="Top supported metapaths per LV-target for subgraph stage.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="Top gene-target pairs per selected metapath.",
    )
    parser.add_argument(
        "--top-paths",
        type=int,
        default=5,
        help="Top path instances per selected pair.",
    )
    parser.add_argument(
        "--degree-d",
        type=float,
        default=0.5,
        help="Degree damping for path instance extraction.",
    )
    args = parser.parse_args()

    if args.stage == "top-genes":
        run_top_genes_stage(args)
        return

    if args.stage == "target-sets":
        run_target_sets_stage(args)
        return

    if args.stage == "real-dwpc":
        run_real_dwpc_stage(args)
        return

    if args.stage == "precompute-scores":
        run_precompute_scores_stage(args)
        return

    if args.stage == "nulls":
        run_nulls_stage(args)
        return

    if args.stage == "stats":
        run_stats_stage(args)
        return

    if args.stage == "top-subgraphs":
        run_top_subgraphs_stage(args)
        return

    if args.stage == "plot-subgraphs":
        run_plot_subgraphs_stage(args)
        return

    if args.stage == "pipeline-fast":
        print("[note] Running optimized fast pipeline stages.")
        run_top_genes_stage(args)
        run_target_sets_stage(args)
        run_precompute_scores_stage(args)
        run_nulls_stage(args)
        run_stats_stage(args)
        return

    if args.stage == "pipeline-subgraphs":
        print("[note] Running subgraph extraction + plotting stages.")
        run_top_subgraphs_stage(args)
        run_plot_subgraphs_stage(args)
        return

    if args.stage == "pipeline":
        print("[note] Running legacy baseline pipeline stages.")
        run_top_genes_stage(args)
        run_target_sets_stage(args)
        run_real_dwpc_stage(args)
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
