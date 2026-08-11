"""lv_compute_replicate_summaries.py must not trust a stale on-disk manifest.

Concurrent lv-perm/lv-rand array tasks each call write_manifest() after
creating their own artifact, re-globbing the whole artifact directory and
overwriting replicate_manifest.csv wholesale. With up to 10 tasks racing,
whichever finishes last can capture the directory before a sibling task's
file exists, silently dropping that sibling from the manifest even though
its artifact file is present on disk. By the time lv-sum runs (it depends
on both arrays via afterok), every array task has finished -- so the fix is
to re-derive the manifest fresh at that point rather than trust whatever a
mid-race task last wrote.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.lv_explicit_replicates import write_manifest  # noqa: E402

_SCRIPT_PATH = REPO_ROOT / "scripts" / "experiments" / "lv_compute_replicate_summaries.py"
_spec = importlib.util.spec_from_file_location("lv_compute_replicate_summaries", _SCRIPT_PATH)
lv_compute_replicate_summaries = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lv_compute_replicate_summaries)


class ComputeReplicateSummariesManifestStalenessTests(unittest.TestCase):
    def test_processes_every_artifact_on_disk_even_when_manifest_is_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            art_dir = output_dir / "replicate_artifacts"
            art_dir.mkdir(parents=True)

            all_names = ["lv_real"] + [f"lv_permuted_{i:03d}" for i in range(1, 11)]
            for name in all_names:
                (art_dir / f"{name}.csv").touch()

            # Simulate the race: a manifest written mid-flight, missing
            # permuted_003..permuted_006 even though their files exist above.
            stale_names = [n for n in all_names if n not in {
                "lv_permuted_003", "lv_permuted_004", "lv_permuted_005", "lv_permuted_006",
            }]
            stale = write_manifest(output_dir)
            stale = stale[stale["name"].isin(stale_names)]
            stale.to_csv(output_dir / "replicate_manifest.csv", index=False)

            processed = []

            def fake_compute(output_dir, name, force=False):
                processed.append(name)
                return output_dir / "replicate_summaries" / f"summary_{name}.csv"

            with patch.object(lv_compute_replicate_summaries, "compute_summary_for_artifact", fake_compute):
                with patch.object(sys, "argv", ["lv_compute_replicate_summaries.py", "--output-dir", str(output_dir)]):
                    lv_compute_replicate_summaries.main()

            self.assertEqual(sorted(processed), sorted(all_names))


if __name__ == "__main__":
    unittest.main()
