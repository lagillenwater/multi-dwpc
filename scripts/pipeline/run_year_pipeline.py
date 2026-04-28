#!/usr/bin/env python3
"""Run the year analysis pipeline locally (sequentially).

Mirrors the stages in hpc/submit_year_full_pipeline.sh:
  1. B Selection
  2. Intermediate Sharing (at chosen B)
  3. Select Top GO Terms
  4. Global Summary
  5. Gene Connectivity Table
  6. Subgraph Visualization (per top GO term)
  7. Intermediate Sharing Plots
  8. Metapath Selection Comparison

Usage:
    python scripts/pipeline/run_year_pipeline.py --output-dir output/year_full_analysis

    # Skip B selection (reuse existing chosen_b.json):
    python scripts/pipeline/run_year_pipeline.py --skip-b-select

    # Run only through stage 3:
    python scripts/pipeline/run_year_pipeline.py --stop-after-stage 3

    # Resume from stage 4:
    python scripts/pipeline/run_year_pipeline.py --start-from-stage 4

    # Dry run (print commands without executing):
    python scripts/pipeline/run_year_pipeline.py --dry-run
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


def _read_json_list(path: Path) -> list[str]:
    with open(path) as f:
        return json.load(f)


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
    p.add_argument("--output-dir", default="output/year_full_analysis")
    p.add_argument("--year-output-dir", default="output/year_experiment")
    p.add_argument("--added-pairs-path", default="output/intermediate/upd_go_bp_2024_added.csv")
    p.add_argument("--effect-size-threshold", type=float, default=1.65)
    p.add_argument("--dwpc-percentile", type=float, default=75)
    p.add_argument("--dwpc-z-threshold", type=float, default=None,
        help="Gene DWPC z cutoff (overrides --dwpc-percentile when set).")
    p.add_argument("--path-z-threshold", type=float, default=None,
        help="Path-score z cutoff pooled per (GO, metapath) (overrides --path-top-k when set).")
    p.add_argument("--path-top-k", type=int, default=None,
        help="Legacy per-gene path cap (used only when --path-z-threshold is unset).")
    p.add_argument("--path-enumeration-cap", type=int, default=None,
        help="Max paths to enumerate per gene when --path-z-threshold is set.")
    p.add_argument("--top-go-terms", type=int, default=10)
    p.add_argument("--min-added-genes", type=int, default=5)
    p.add_argument("--skip-b-select", action="store_true")
    p.add_argument("--stop-after-stage", type=int, default=None)
    p.add_argument("--start-from-stage", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Forward --plot-only to subprocess calls (select_optimal_b, year_go_term_support, year_snapshot_rank_similarity). Use with --skip-b-select --start-from-stage 3 to regenerate every PDF without recomputing intermediates.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    year_dir = args.year_output_dir
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
            variance_dir = Path(year_dir) / "year_null_variance_experiment"
            rank_dir = Path(year_dir) / "year_rank_stability_experiment"

            if not variance_dir.is_dir() or not rank_dir.is_dir():
                print("Warning: variance/rank dirs not found -- defaulting B=10")
                (out / "b_selection").mkdir(parents=True, exist_ok=True)
                chosen_b_path.write_text(json.dumps({"chosen_b": 10, "note": "default"}))
            else:
                b_select_cmd = [
                    py, "scripts/pipeline/select_optimal_b.py",
                    "--analysis-type", "year",
                    "--variance-dir", str(variance_dir),
                    "--rank-dir", str(rank_dir),
                    "--output-dir", str(out / "b_selection"),
                    "--aggregation", "median",
                ]
                if args.plot_only:
                    b_select_cmd.append("--plot-only")
                _run("B Selection", b_select_cmd, dry_run=args.dry_run)

    if stop is not None and stop <= 1:
        return

    chosen_b = _read_chosen_b(chosen_b_path) if chosen_b_path.exists() else 10
    print(f"\nChosen B = {chosen_b}")

    # Stage 2: Intermediate Sharing
    if should_run(2):
        is_cmd = [
            py, "scripts/pipeline/year_intermediate_sharing.py",
            "--year-output-dir", year_dir,
            "--added-pairs-path", args.added_pairs_path,
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

    # Stage 3: Select Top GO Terms
    if should_run(3):
        _run("Select Top GO Terms", [
            py, "scripts/pipeline/select_top_go_terms.py",
            "--input-dir", str(out / "intermediate_sharing" / f"b{chosen_b}"),
            "--top-n", str(args.top_go_terms),
            "--min-added-genes", str(args.min_added_genes),
            "--output-dir", str(out),
        ], dry_run=args.dry_run)

    # Stage 4: Global Summary
    if should_run(4):
        _run("Global Summary", [
            py, "scripts/pipeline/generate_global_summary.py",
            "--analysis-type", "year",
            "--input-dir", str(out / "intermediate_sharing"),
            "--b-values", str(chosen_b),
            "--chosen-b-json", str(chosen_b_path),
            "--effect-size-threshold", str(args.effect_size_threshold),
            "--output-dir", str(out / "global_summary"),
        ], dry_run=args.dry_run)

    # Stage 5: Gene Connectivity Table
    if should_run(5):
        _run("Gene Connectivity Table", [
            py, "scripts/pipeline/generate_gene_table.py",
            "--analysis-type", "year",
            "--input-dir", str(out / "intermediate_sharing"),
            "--lv-output-dirs", year_dir,
            "--b", str(chosen_b),
            "--output-dir", str(out / "consumable"),
        ], dry_run=args.dry_run)

    # Stage 6: Subgraph Visualization (per top GO term)
    if should_run(6):
        top_go_path = out / "top_go_ids.json"
        if top_go_path.exists() and not args.dry_run:
            top_go_ids = _read_json_list(top_go_path)
            for go_id in top_go_ids:
                _run(f"Subgraph Viz: {go_id}", [
                    py, "scripts/visualization/plot_metapath_subgraphs.py",
                    "--analysis-type", "year",
                    "--input-dir", str(out / "intermediate_sharing"),
                    "--gene-table", str(out / "consumable" / "gene_connectivity_table.csv"),
                    "--b", str(chosen_b),
                    "--gene-set-id", go_id,
                    "--output-dir", str(out / "consumable" / "subgraphs" / go_id),
                    "--top-k", "3",
                ], dry_run=args.dry_run)
        elif args.dry_run:
            _run("Subgraph Viz: <top GO terms>", [
                py, "scripts/visualization/plot_metapath_subgraphs.py",
                "--analysis-type", "year",
                "--input-dir", str(out / "intermediate_sharing"),
                "--gene-table", str(out / "consumable" / "gene_connectivity_table.csv"),
                "--b", str(chosen_b),
                "--gene-set-id", "<GO_ID>",
                "--output-dir", str(out / "consumable" / "subgraphs" / "<GO_ID>"),
                "--top-k", "3",
            ], dry_run=True)
        else:
            print("Warning: top_go_ids.json not found -- skipping subgraph viz")

    # Stage 7: Intermediate Sharing Plots
    if should_run(7):
        cmd = [
            py, "scripts/visualization/plot_year_intermediate_sharing.py",
            "--input-dir", str(out / "intermediate_sharing" / f"b{chosen_b}"),
            "--output-dir", str(out / "intermediate_sharing" / f"b{chosen_b}" / "figures"),
            "--repo-root", ".",
        ]
        top_go_path = out / "top_go_ids.json"
        if top_go_path.exists():
            cmd.extend(["--top-go-json", str(top_go_path)])
        _run("Intermediate Sharing Plots", cmd, dry_run=args.dry_run)

    # Stage 8: Metapath Selection Comparison
    if should_run(8):
        runs_path = Path(year_dir) / "year_rank_stability_experiment" / "all_runs_long.csv"
        mp_dir = out / "metapath_analysis"

        go_support_cmd = [
            py, "scripts/visualization/year_go_term_support.py",
            "--runs-path", str(runs_path),
            "--b", str(chosen_b),
            "--go-support-output", str(mp_dir / "year_direct_go_term_support.csv"),
            "--global-support-output", str(mp_dir / "year_direct_global_metapath_support.csv"),
        ]
        if args.plot_only:
            go_support_cmd.append("--plot-only")
        _run("GO Term Support", go_support_cmd, dry_run=args.dry_run)

        _run("Metapath Selection Frequency", [
            py, "scripts/visualization/plot_year_effective_metapath_selection.py",
            "--support-path", str(mp_dir / "year_direct_go_term_support.csv"),
            "--output-dir", str(mp_dir / "selection_frequency"),
        ], dry_run=args.dry_run)

        rank_sim_cmd = [
            py, "scripts/visualization/year_snapshot_rank_similarity.py",
            "--support-path", str(mp_dir / "year_direct_global_metapath_support.csv"),
            "--rank-metric", "selected_fraction_all",
            "--b", str(chosen_b),
            "--output-dir", str(mp_dir / "rank_similarity"),
            "--label-metapaths",
        ]
        if args.plot_only:
            rank_sim_cmd.append("--plot-only")
        _run("Rank Similarity Scatter", rank_sim_cmd, dry_run=args.dry_run)

    print("\nYear pipeline complete.")


if __name__ == "__main__":
    main()
