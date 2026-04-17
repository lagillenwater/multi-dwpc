#!/usr/bin/env python3
"""Prepare shared LV inputs for explicit null-replicate experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.lv_explicit_replicates import write_manifest, write_real_artifact  # noqa: E402
from src.lv_precompute import (  # noqa: E402
    list_feature_metapaths,
    precompute_gene_feature_scores,
    prepare_feature_manifest,
    warmup_feature_metapath_cache,
)


DEFAULT_LVS = "LV603,LV246,LV57"


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument("--lv-loadings", default=None)
    parser.add_argument("--gene-reference", default="data/nodes/Gene.tsv")
    parser.add_argument("--lvs", default=DEFAULT_LVS)
    parser.add_argument("--top-fraction", type=float, default=0.005)
    parser.add_argument("--n-workers-precompute", type=int, default=4)
    parser.add_argument("--max-metapath-length", type=int, default=3)
    parser.add_argument("--metapath-limit-per-target", type=int, default=2)
    parser.add_argument("--all-metapaths", action="store_true")
    parser.add_argument("--prepare-metadata-only", action="store_true")
    parser.add_argument("--finalize-precompute", action="store_true")
    parser.add_argument("--list-metapaths", action="store_true")
    parser.add_argument("--warmup-metapath", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _prepare_metadata(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    pipeline = REPO_ROOT / "scripts" / "lv_multidwpc_analysis.py"
    common = [sys.executable, str(pipeline), "--output-dir", str(output_dir)]
    if args.resume:
        common.append("--resume")
    if args.force:
        common.append("--force")
    if args.smoke:
        common.append("--smoke")
    else:
        if not args.lv_loadings:
            raise ValueError("--lv-loadings is required unless --smoke is enabled")
        common.extend(["--lv-loadings", args.lv_loadings])
    common.extend(["--gene-reference", args.gene_reference, "--lvs", args.lvs, "--top-fraction", str(args.top_fraction)])

    _run(common + ["--stage", "top-genes"])
    _run(common + ["--stage", "targets"])

    prepare_feature_manifest(
        output_dir=output_dir,
        metapath_stats_path=REPO_ROOT / "data" / "metapath-dwpc-stats.tsv",
        max_metapath_length=args.max_metapath_length,
        metapath_limit_per_target=None if args.all_metapaths else args.metapath_limit_per_target,
        include_direct_metapaths=False,
    )
    write_real_artifact(output_dir, force=args.force)
    manifest = write_manifest(output_dir)
    print(f"Prepared LV metadata workspace: {output_dir}")
    print(f"Replicate manifest rows: {len(manifest)}")
    return output_dir


def _finalize_precompute(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    feature_manifest, lv_real_indices, real_feature_scores = precompute_gene_feature_scores(
        output_dir=output_dir,
        data_dir=REPO_ROOT / "data",
        metapath_stats_path=REPO_ROOT / "data" / "metapath-dwpc-stats.tsv",
        damping=0.5,
        max_metapath_length=args.max_metapath_length,
        metapath_limit_per_target=None if args.all_metapaths else args.metapath_limit_per_target,
        n_workers_precompute=args.n_workers_precompute,
        include_direct_metapaths=False,
    )
    print(f"Finalized LV precompute: {output_dir}")
    print(f"Features: {len(feature_manifest)}")
    print(f"LV real index rows: {len(lv_real_indices)}")
    print(f"Real feature score rows: {len(real_feature_scores)}")
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.list_metapaths:
        for metapath in list_feature_metapaths(output_dir):
            print(metapath)
        return

    if args.warmup_metapath:
        warmup_feature_metapath_cache(
            output_dir=output_dir,
            data_dir=REPO_ROOT / "data",
            metapath=args.warmup_metapath,
            damping=0.5,
        )
        print(f"Warmed LV cache for metapath: {args.warmup_metapath}")
        return

    if args.finalize_precompute:
        _finalize_precompute(args)
        return

    _prepare_metadata(args)
    if args.prepare_metadata_only:
        return
    _finalize_precompute(args)


if __name__ == "__main__":
    main()
