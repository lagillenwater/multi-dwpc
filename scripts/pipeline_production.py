"""
Run the full production pipeline (no GO hierarchy analysis).

This script executes the production-ready sequence of pipeline steps using the
per-script entry points under scripts/.
"""

from datetime import datetime
from pathlib import Path
import subprocess
import sys


PIPELINE_STEPS = [
    "scripts/load_data.py",
    "scripts/percent_change_and_filtering.py",
    "scripts/jaccard_similarity_and_filtering.py",
    "scripts/permutation_null_datasets.py",
    "scripts/random_null_datasets.py",
]


def init_log_dir(repo_root: Path, pipeline_name: str) -> Path:
    """Create a timestamped log directory for the pipeline run."""
    log_root = repo_root / "output" / "logs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_root / f"{pipeline_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_step(repo_root: Path, script_path: str, log_dir: Path, step_index: int) -> None:
    """Run a pipeline step and raise on failure."""
    script_full_path = repo_root / script_path
    cmd = [sys.executable, str(script_full_path)]
    log_file = log_dir / f"{step_index:02d}_{script_full_path.stem}.log"

    print(f"\nRunning: {script_full_path}")
    print(f"Log: {log_file}")

    with log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=repo_root,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_handle.write(line)
            print(line, end="")
        retcode = proc.wait()

    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd)


def main() -> None:
    """Execute all production pipeline steps."""
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Repository root: {repo_root}")

    log_dir = init_log_dir(repo_root, "pipeline_production")
    print(f"Run logs directory: {log_dir}")

    for index, step in enumerate(PIPELINE_STEPS, start=1):
        run_step(repo_root, step, log_dir, index)

    print("\nProduction pipeline complete.")


if __name__ == "__main__":
    main()
