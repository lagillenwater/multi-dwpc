"""
LV multi-DWPC analysis orchestrator.
- load LV loadings
- select top 1% genes per LV
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
from src.lv_targets import build_target_sets  # noqa: E402


DEFAULT_LVS = ("LV603", "LV246", "LV57")


def _parse_lvs(lv_arg: str) -> list[str]:
    tokens = [token.strip() for token in lv_arg.split(",")]
    return [token for token in tokens if token]


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

    top_genes_path = output_dir / "lv_top_genes.csv"
    summary_path = output_dir / "lv_top_genes_summary.csv"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="LV multi-DWPC analysis pipeline.")
    parser.add_argument(
        "--stage",
        default="top-genes",
        choices=["top-genes", "target-sets", "pipeline"],
        help=(
            "Pipeline stage to run. "
            "'pipeline' currently runs top-genes then target-sets."
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
        default=0.01,
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
        "--include-brown-adipose",
        action="store_true",
        help="Include UBERON:0001348 in adipose target set.",
    )
    args = parser.parse_args()

    if args.stage == "top-genes":
        run_top_genes_stage(args)
        return

    if args.stage == "target-sets":
        run_target_sets_stage(args)
        return

    if args.stage == "pipeline":
        print("[note] Running currently implemented pipeline stages.")
        run_top_genes_stage(args)
        run_target_sets_stage(args)
        return

    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
