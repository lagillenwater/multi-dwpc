import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import year_statistics  # noqa: E402


class YearStatisticsTests(unittest.TestCase):
    def test_standardize_year_result_frame_renames_api_columns(self):
        api_like = pd.DataFrame(
            {
                "neo4j_source_id": ["GO:1"],
                "metapath_abbreviation": ["BPpG<rG"],
                "dwpc": [0.5],
            }
        )

        standardized = year_statistics._standardize_year_result_frame(api_like)

        self.assertIn("go_id", standardized.columns)
        self.assertIn("metapath", standardized.columns)
        self.assertEqual(standardized.loc[0, "go_id"], "GO:1")
        self.assertEqual(standardized.loc[0, "metapath"], "BPpG<rG")

    def test_build_aggregated_year_statistics_and_detect_supported_statistics(self):
        datasets = {
            "2016_real": pd.DataFrame(
                {
                    "go_id": ["GO:1", "GO:1", "GO:1"],
                    "metapath": ["BPpG<rG", "BPpG<rG", "BPpG<rG"],
                    "dwpc": [0.0, 2.0, 4.0],
                    "p_value": [0.0, 0.2, 0.4],
                    "dgp_nonzero_sd": [0.0, 1.0, 2.0],
                }
            )
        }

        aggregated = year_statistics.build_aggregated_year_statistics(datasets)
        row = aggregated.iloc[0]
        supported = year_statistics.detect_supported_statistics(aggregated)

        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(row["mean_dwpc"], 2.0)
        self.assertAlmostEqual(row["mean_dwpc_nonzero"], 3.0)
        self.assertAlmostEqual(row["median_dwpc"], 2.0)
        self.assertAlmostEqual(row["median_dwpc_nonzero"], 3.0)
        self.assertAlmostEqual(row["mean_pvalue"], 0.2)
        self.assertAlmostEqual(row["mean_pvalue_nonzero"], 0.3)
        self.assertAlmostEqual(row["mean_std"], 1.0)
        self.assertAlmostEqual(row["mean_std_nonzero"], 1.5)
        self.assertEqual(row["n_total"], 3)
        self.assertEqual(row["n_nonzero"], 2)
        self.assertEqual(
            supported,
            [
                "mean_dwpc",
                "mean_dwpc_nonzero",
                "median_dwpc",
                "median_dwpc_nonzero",
                "mean_pvalue",
                "mean_pvalue_nonzero",
                "median_pvalue",
                "median_pvalue_nonzero",
                "mean_std",
                "mean_std_nonzero",
                "median_std",
                "median_std_nonzero",
            ],
        )

    def test_load_labeled_year_result_files_excludes_default_metapaths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "go_id": ["GO:1", "GO:1", "GO:1"],
                    "metapath": ["BPpG", "GpBP", "BPpG<rG"],
                    "dwpc": [1.0, 2.0, 3.0],
                }
            ).to_csv(data_dir / "example.csv", index=False)

            loaded = year_statistics.load_labeled_year_result_files(
                data_dir,
                {"example": "example.csv"},
            )

        self.assertIn("example", loaded)
        self.assertEqual(loaded["example"]["metapath"].tolist(), ["BPpG<rG"])


if __name__ == "__main__":
    unittest.main()
