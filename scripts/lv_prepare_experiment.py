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
    parser.add_argument("--include-brown-adipose", action="store_true")
    parser.add_argument("--n-workers-precompute", type=int, default=4)
    parser.add_argument("--max-metapath-length", type=int, default=3)
    parser.add_argument("--metapath-limit-per-target", type=int, default=2)
    parser.add_argument("--all-metapaths", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    target_cmd = common + ["--stage", "target-sets"]
    if args.include_brown_adipose:
        target_cmd.append("--include-brown-adipose")
    _run(target_cmd)

    precompute_cmd = common + [
        "--stage",
        "precompute-scores",
        "--n-workers-precompute",
        str(args.n_workers_precompute),
        "--max-metapath-length",
        str(args.max_metapath_length),
    ]
    if not args.all_metapaths:
        precompute_cmd.extend(["--metapath-limit-per-target", str(args.metapath_limit_per_target)])
    _run(precompute_cmd)

    write_real_artifact(output_dir, force=args.force)
    manifest = write_manifest(output_dir)
    print(f"Prepared LV experiment workspace: {output_dir}")
    print(f"Replicate manifest rows: {len(manifest)}")


if __name__ == "__main__":
    main()
