import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lv_replicate_analysis import load_summary_bank  # noqa: E402


class LvReplicateAnalysisTests(unittest.TestCase):
    def test_load_summary_bank_uses_manifest_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            summary_path = output_dir / "replicate_summaries" / "summary_lv_real.csv"
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "domain": "lv",
                        "name": "lv_real",
                        "control": "real",
                        "replicate": 0,
                        "lv_id": "LV1",
                        "target_set_id": "TS1",
                        "target_set_label": "set1",
                        "node_type": "Gene",
                        "metapath": "GpBP",
                        "mean_score": 1.0,
                    }
                ]
            ).to_csv(summary_path, index=False)

            pd.DataFrame(
                [
                    {
                        "domain": "lv",
                        "name": "lv_real",
                        "control": "real",
                        "replicate": 0,
                        "source_path": "input.csv",
                        "result_path": "",
                        "summary_path": str(summary_path),
                    }
                ]
            ).to_csv(output_dir / "replicate_manifest.csv", index=False)

            summary_df = load_summary_bank(output_dir)

        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.loc[0, "metapath"], "GpBP")
        self.assertEqual(summary_df.loc[0, "control"], "real")

    def test_manifest_missing_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            pd.DataFrame(
                [
                    {
                        "domain": "lv",
                        "name": "lv_real",
                        "control": "real",
                        "replicate": 0,
                        "summary_path": "summary.csv",
                    }
                ]
            ).to_csv(output_dir / "replicate_manifest.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Manifest missing required columns"):
                load_summary_bank(output_dir)


if __name__ == "__main__":
    unittest.main()
