import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.year_replicate_analysis import load_summary_bank  # noqa: E402


class YearReplicateAnalysisTests(unittest.TestCase):
    def test_load_summary_bank_uses_year_manifest_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            summary_path = workspace_dir / "replicate_summaries" / "summary_2016_real.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "domain": "year",
                        "name": "all_GO_positive_growth_2016_real",
                        "control": "real",
                        "replicate": 0,
                        "year": 2016,
                        "go_id": "GO:1",
                        "metapath": "BPpG<rG",
                        "mean_score": 1.0,
                    }
                ]
            ).to_csv(summary_path, index=False)
            pd.DataFrame(
                [
                    {
                        "domain": "year",
                        "name": "all_GO_positive_growth_2016_real",
                        "control": "real",
                        "replicate": 0,
                        "year": 2016,
                        "source_path": "input.csv",
                        "result_path": "results.csv",
                        "summary_path": str(summary_path),
                    }
                ]
            ).to_csv(workspace_dir / "replicate_manifest.csv", index=False)

            summary_df = load_summary_bank(workspace_dir)

        self.assertEqual(len(summary_df), 1)
        self.assertEqual(int(summary_df.loc[0, "year"]), 2016)
        self.assertEqual(summary_df.loc[0, "metapath"], "BPpG<rG")

    def test_load_summary_bank_requires_year_in_manifest_when_manifest_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            summary_path = workspace_dir / "replicate_summaries" / "summary_2016_real.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "domain": "year",
                        "name": "all_GO_positive_growth_2016_real",
                        "control": "real",
                        "replicate": 0,
                        "year": 2016,
                        "go_id": "GO:1",
                        "metapath": "BPpG<rG",
                        "mean_score": 1.0,
                    }
                ]
            ).to_csv(summary_path, index=False)
            pd.DataFrame(
                [
                    {
                        "domain": "year",
                        "name": "all_GO_positive_growth_2016_real",
                        "control": "real",
                        "replicate": 0,
                        "source_path": "input.csv",
                        "result_path": "results.csv",
                        "summary_path": str(summary_path),
                    }
                ]
            ).to_csv(workspace_dir / "replicate_manifest.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Manifest missing required columns"):
                load_summary_bank(workspace_dir)


if __name__ == "__main__":
    unittest.main()
