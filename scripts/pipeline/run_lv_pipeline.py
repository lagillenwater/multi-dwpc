#!/usr/bin/env python3
"""Run the LV analysis pipeline locally (sequentially).

Mirrors the stages in hpc/submit_lv_full_pipeline.sh:
  1. B Selection
  2. Intermediate Sharing (at chosen B)
  3. Global Summary
  4. Gene Connectivity Table
  5. Subgraph Visualization
  6. Intermediate Sharing Plots

Usage:
    python scripts/pipeline/run_lv_pipeline.py --output-dir output/lv_full_analysis

    # Skip B selection (reuse existing chosen_b.json):
    python scripts/pipeline/run_lv_pipeline.py --skip-b-select

    # Run only through stage 2:
    python scripts/pipeline/run_lv_pipeline.py --stop-after-stage 2

    # Resume from stage 3:
    python scripts/pipeline/run_lv_pipeline.py --start-from-stage 3

    # Dry run (print commands without executing):
    python scripts/pipeline/run_lv_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read_chosen_b(path: Path) -> int:
    with open(path) as f:
        return int(json.load(f)["chosen_b"])


def _run(name: str, cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Stage: {name}")
    print(f"{'='*60}")
    cmd_str = " \\\n    ".join(cmd)
    print(f"  {cmd_str}")
    if dry_run:
        print("  [dry run -- skipped]")
        return
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"FAILED: {name} (exit code {result.returncode})")
        sys.exit(result.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="output/lv_full_analysis")
    p.add_argument("--lv-output-dirs", default="output/lv_single_target_refactor")
    p.add_argument("--effect-size-threshold", type=float, default=1.65)
    p.add_argument("--dwpc-percentile", type=float, default=75)
    p.add_argument("--dwpc-z-threshold", type=float, default=None,
        help="Gene DWPC z cutoff (overrides --dwpc-percentile when set).")
    p.add_argument("--path-z-threshold", type=float, default=None,
        help="Path-score z cutoff pooled per (LV, metapath) (overrides --path-top-k when set).")
    p.add_argument("--path-top-k", type=int, default=None,
        help="Legacy per-gene path cap (used only when --path-z-threshold is unset).")
    p.add_argument("--path-enumeration-cap", type=int, default=None,
        help="Max paths to enumerate per gene when --path-z-threshold is set.")
    p.add_argument("--skip-b-select", action="store_true")
    p.add_argument("--stop-after-stage", type=int, default=None)
    p.add_argument("--start-from-stage", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lv_dirs = args.lv_output_dirs
    py = sys.executable
    start = args.start_from_stage
    stop = args.stop_after_stage

    def should_run(stage: int) -> bool:
        if stage < start:
            return False
        if stop is not None and stage > stop:
            return False
        return True

    # Stage 1: B Selection
    chosen_b_path = out / "b_selection" / "chosen_b.json"
    if should_run(1):
        if args.skip_b_select and chosen_b_path.exists():
            print(f"Skipping B selection -- using {chosen_b_path}")
        else:
            variance_dir = rank_dir = ""
            for d in lv_dirs.split():
                v = Path(d) / "lv_null_variance_experiment"
                r = Path(d) / "lv_rank_stability_experiment"
                if v.is_dir():
                    variance_dir = str(v)
                if r.is_dir():
                    rank_dir = str(r)

            if not variance_dir or not rank_dir:
                print("Warning: variance/rank dirs not found -- defaulting B=10")
                (out / "b_selection").mkdir(parents=True, exist_ok=True)
                chosen_b_path.write_text(json.dumps({"chosen_b": 10, "note": "default"}))
            else:
                _run("B Selection", [
                    py, "scripts/pipeline/select_optimal_b.py",
                    "--analysis-type", "lv",
                    "--variance-dir", variance_dir,
                    "--rank-dir", rank_dir,
                    "--output-dir", str(out / "b_selection"),
                    "--aggregation", "median",
                ], dry_run=args.dry_run)

    if stop is not None and stop <= 1:
        return

    chosen_b = _read_chosen_b(chosen_b_path) if chosen_b_path.exists() else 10
    print(f"\nChosen B = {chosen_b}")

    # Stage 2: Intermediate Sharing
    if should_run(2):
        is_cmd = [
            py, "scripts/pipeline/lv_intermediate_sharing.py",
            "--lv-output-dirs", *lv_dirs.split(),
            "--b", str(chosen_b),
            "--effect-size-threshold", str(args.effect_size_threshold),
            "--dwpc-percentile", str(args.dwpc_percentile),
            "--output-dir", str(out / "intermediate_sharing"),
        ]
        if args.dwpc_z_threshold is not None:
            is_cmd += ["--dwpc-z-threshold", str(args.dwpc_z_threshold)]
        if args.path_z_threshold is not None:
            is_cmd += ["--path-z-threshold", str(args.path_z_threshold)]
        if args.path_top_k is not None:
            is_cmd += ["--path-top-k", str(args.path_top_k)]
        if args.path_enumeration_cap is not None:
            is_cmd += ["--path-enumeration-cap", str(args.path_enumeration_cap)]
        _run("Intermediate Sharing", is_cmd, dry_run=args.dry_run)

    # Stage 3: Global Summary
    if should_run(3):
        cmd = [
            py, "scripts/pipeline/generate_global_summary.py",
            "--analysis-type", "lv",
            "--input-dir", str(out / "intermediate_sharing"),
            "--b-values", str(chosen_b),
            "--chosen-b-json", str(chosen_b_path),
            "--effect-size-threshold", str(args.effect_size_threshold),
            "--output-dir", str(out / "global_summary"),
        ]
        all_runs = None
        for d in lv_dirs.split():
            candidate = Path(d) / "lv_rank_stability_experiment" / "all_runs_long.csv"
            if candidate.exists():
                all_runs = str(candidate)
                break
        if all_runs:
            cmd.extend(["--all-runs-path", all_runs])
        _run("Global Summary", cmd, dry_run=args.dry_run)

    # Stage 4: Gene Connectivity Table
    if should_run(4):
        _run("Gene Connectivity Table", [
            py, "scripts/pipeline/generate_gene_table.py",
            "--analysis-type", "lv",
            "--input-dir", str(out / "intermediate_sharing"),
            "--lv-output-dirs", *lv_dirs.split(),
            "--b", str(chosen_b),
            "--output-dir", str(out / "consumable"),
        ], dry_run=args.dry_run)

    # Stage 5: Subgraph Visualization
    if should_run(5):
        _run("Subgraph Visualization", [
            py, "scripts/visualization/plot_metapath_subgraphs.py",
            "--analysis-type", "lv",
            "--input-dir", str(out / "intermediate_sharing"),
            "--gene-table", str(out / "consumable" / "gene_connectivity_table.csv"),
            "--b", str(chosen_b),
            "--output-dir", str(out / "consumable" / "subgraphs"),
            "--top-k", "3",
        ], dry_run=args.dry_run)

    # Stage 6: Intermediate Sharing Plots
    if should_run(6):
        int_dir = out / "intermediate_sharing" / f"b{chosen_b}"
        if not int_dir.is_dir():
            int_dir = out / "intermediate_sharing"
        _run("Intermediate Sharing Plots", [
            py, "scripts/visualization/plot_lv_intermediate_sharing.py",
            "--input-dir", str(int_dir),
        ], dry_run=args.dry_run)

    print("\nLV pipeline complete.")


if __name__ == "__main__":
    main()
