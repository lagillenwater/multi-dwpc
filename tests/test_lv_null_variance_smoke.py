import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class LvNullVarianceSmokeTests(unittest.TestCase):
    def test_lv_null_variance_script_runs_on_tiny_manifest_and_summary_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "lv_output"
            summary_dir = output_dir / "replicate_summaries"
            analysis_dir = tmp_path / "analysis"
            summary_dir.mkdir(parents=True, exist_ok=True)

            manifest_rows = []
            summary_specs = [
                ("lv_real", "real", 0, 1.0),
                ("lv_permuted_001", "permuted", 1, 0.8),
                ("lv_permuted_002", "permuted", 2, 0.9),
                ("lv_random_001", "random", 1, 0.4),
                ("lv_random_002", "random", 2, 0.5),
            ]

            for name, control, replicate, score in summary_specs:
                summary_path = summary_dir / f"summary_{name}.csv"
                pd.DataFrame(
                    [
                        {
                            "domain": "lv",
                            "name": name,
                            "control": control,
                            "replicate": replicate,
                            "lv_id": "LV1",
                            "target_id": "TS1",
                            "target_name": "set1",
                            "node_type": "Gene",
                            "metapath": "GpBP",
                            "mean_score": score,
                        }
                    ]
                ).to_csv(summary_path, index=False)
                manifest_rows.append(
                    {
                        "domain": "lv",
                        "name": name,
                        "control": control,
                        "replicate": replicate,
                        "source_path": str(output_dir / "replicate_artifacts" / f"{name}.csv"),
                        "result_path": "",
                        "summary_path": str(summary_path),
                    }
                )

            pd.DataFrame(manifest_rows).to_csv(output_dir / "replicate_manifest.csv", index=False)

            cmd = [
                sys.executable,
                "scripts/experiments/lv_null_variance_experiment.py",
                "--output-dir",
                str(output_dir),
                "--analysis-output-dir",
                str(analysis_dir),
                "--b-values",
                "1,2",
                "--seeds",
                "11,22",
            ]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

            feature_path = analysis_dir / "feature_variance_summary.csv"
            overall_path = analysis_dir / "overall_variance_summary.csv"
            plot_path = analysis_dir / "variance_overall_by_group.png"

            self.assertTrue(feature_path.exists())
            self.assertTrue(overall_path.exists())
            self.assertTrue(plot_path.exists())

            feature_df = pd.read_csv(feature_path)
            overall_df = pd.read_csv(overall_path)
            self.assertIn("diff_var", feature_df.columns)
            self.assertIn("diff_std", feature_df.columns)
            self.assertIn("mean_diff_var", overall_df.columns)
            self.assertEqual(set(overall_df["control"]), {"permuted", "random"})


if __name__ == "__main__":
    unittest.main()
