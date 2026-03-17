"""
Run the year-comparison DWPC pipeline with optional staged execution.

This pipeline mirrors the LV staged orchestration style and drives:
1) year dataset preparation
2) permutation/random null generation
3) direct DWPC computation
4) downstream metapath analyses
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _init_log_dir(repo_root: Path, log_prefix: str) -> Path:
    log_root = repo_root / "output" / "logs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_root / f"{log_prefix}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _all_exist(repo_root: Path, artifacts: list[str]) -> bool:
    return all((repo_root / rel_path).exists() for rel_path in artifacts)


def _run_step(
    repo_root: Path,
    python_exe: str,
    script_path: str,
    script_args: list[str],
    artifacts: list[str],
    log_dir: Path,
    step_index: int,
    resume: bool,
    extra_env: dict[str, str] | None = None,
) -> None:
    if resume and artifacts and _all_exist(repo_root, artifacts):
        print(f"\nSkipping {script_path} (resume cache hit).")
        for rel_path in artifacts:
            print(f"  exists: {rel_path}")
        return

    script_full_path = repo_root / script_path
    cmd = [python_exe, str(script_full_path)] + script_args
    log_file = log_dir / f"{step_index:02d}_{script_full_path.stem}.log"

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Log: {log_file}")

    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_handle.write(line)
            print(line, end="")
        retcode = proc.wait()

    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd)


def _build_steps(args: argparse.Namespace) -> list[dict]:
    jaccard_args = ["--include-parents"] if args.include_parents else []

    prepare_steps = [
        {
            "script": "scripts/load_data.py",
            "args": [],
            "artifacts": [
                "output/intermediate/hetio_bppg_2016.csv",
                "output/intermediate/upd_go_bp_2024.csv",
            ],
            "env": None,
        },
        {
            "script": "scripts/percent_change_and_filtering.py",
            "args": [],
            "artifacts": ["output/intermediate/all_GO_positive_growth.csv"],
            "env": None,
        },
        {
            "script": "scripts/jaccard_similarity_and_filtering.py",
            "args": jaccard_args,
            "artifacts": [
                "output/intermediate/hetio_bppg_all_GO_positive_growth_filtered.csv",
                "output/intermediate/hetio_bppg_all_GO_positive_growth_2024_filtered.csv",
            ],
            "env": None,
        },
    ]

    null_steps = [
        {
            "script": "scripts/permutation_null_datasets.py",
            "args": [],
            "artifacts": [
                "output/permutations/all_GO_positive_growth_2016/perm_001.csv",
                "output/permutations/all_GO_positive_growth_2024/perm_001.csv",
            ],
            "env": None,
        },
        {
            "script": "scripts/random_null_datasets.py",
            "args": [],
            "artifacts": [
                "output/random_samples/all_GO_positive_growth_2016/random_001.csv",
                "output/random_samples/all_GO_positive_growth_2024/random_001.csv",
            ],
            "env": None,
        },
    ]

    dwpc_steps = [
        {
            "script": "scripts/compute_dwpc_direct.py",
            "args": [],
            "artifacts": [
                "output/dwpc_direct/all_GO_positive_growth/results/dwpc_all_GO_positive_growth_2016_real.csv",
                "output/dwpc_direct/all_GO_positive_growth/results/dwpc_all_GO_positive_growth_2024_real.csv",
            ],
            "env": None,
        }
    ]

    analysis_steps = [
        {
            "script": "scripts/metapath_signature_analysis.py",
            "args": [],
            "artifacts": [
                "output/metapath_analysis/pairwise_statistics/aggregated_statistics_all_datasets.csv"
            ],
            "env": None,
        },
        {
            "script": "scripts/divergence_score_analysis.py",
            "args": [],
            "artifacts": [
                "output/metapath_analysis/divergence_scores/divergence_all_statistics.csv"
            ],
            "env": None,
        },
    ]

    stage_map = {
        "prepare-data": prepare_steps,
        "nulls": null_steps,
        "dwpc": dwpc_steps,
        "analysis": analysis_steps,
        "pipeline": prepare_steps + null_steps + dwpc_steps + analysis_steps,
    }
    return stage_map[args.stage]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run staged year-based DWPC experiments (2016 vs 2024) with null controls."
        )
    )
    parser.add_argument(
        "--stage",
        default="pipeline",
        choices=["prepare-data", "nulls", "dwpc", "analysis", "pipeline"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to invoke step scripts.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps when their expected output artifacts already exist.",
    )
    parser.add_argument(
        "--include-parents",
        action="store_true",
        help="Run Jaccard filtering with parent GO terms included.",
    )
    parser.add_argument(
        "--log-prefix",
        default="pipeline_year_experiment",
        help="Prefix for timestamped log directories under output/logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    log_dir = _init_log_dir(root, args.log_prefix)

    print(f"Repository root: {root}")
    print(f"Run logs directory: {log_dir}")
    print(f"Stage: {args.stage}")

    steps = _build_steps(args)
    for idx, step in enumerate(steps, start=1):
        _run_step(
            repo_root=root,
            python_exe=args.python_exe,
            script_path=step["script"],
            script_args=step["args"],
            artifacts=step["artifacts"],
            log_dir=log_dir,
            step_index=idx,
            resume=args.resume,
            extra_env=step["env"],
        )

    print("\nYear experiment pipeline complete.")


if __name__ == "__main__":
    main()
