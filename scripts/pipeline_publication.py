"""
Run the full publication pipeline (includes GO hierarchy analysis).

This script executes the publication-ready sequence of pipeline steps using
the per-script entry points under scripts/.
"""

from pathlib import Path
import subprocess
import sys


PIPELINE_STEPS = [
    ("scripts/load_data.py", []),
    ("scripts/percent_change_and_filtering.py", []),
    ("scripts/go_hierarchy_analysis.py", []),
    ("scripts/jaccard_similarity_and_filtering.py", ["--include-parents"]),
    ("scripts/permutation_null_datasets.py", []),
    ("scripts/random_null_datasets.py", []),
 ##   ("scripts/lookup_dwpc_api.py", []),
]


def run_step(repo_root: Path, script_path: str, args: list[str]) -> None:
    """Run a pipeline step and raise on failure."""
    script_full_path = repo_root / script_path
    cmd = [sys.executable, str(script_full_path)] + args
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Execute all publication pipeline steps."""
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Repository root: {repo_root}")

    for script_path, args in PIPELINE_STEPS:
        run_step(repo_root, script_path, args)

    print("\nPublication pipeline complete.")


if __name__ == "__main__":
    main()
