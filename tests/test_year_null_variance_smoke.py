import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class YearNullVarianceSmokeTests(unittest.TestCase):
    def test_year_null_variance_script_runs_on_tiny_summary_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            summaries_dir = tmp_path / "replicate_summaries"
            output_dir = tmp_path / "output"
            summaries_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            for year, real_score in [(2016, 1.0), (2024, 1.5)]:
                rows.append(
                    {
                        "domain": "year",
                        "name": f"{year}_real",
                        "control": "real",
                        "replicate": 0,
                        "year": year,
                        "go_id": "GO:1",
                        "metapath": "BPpG<rG",
                        "mean_score": real_score,
                    }
                )
                for control, scores in {
                    "permuted": [0.8, 0.9],
                    "random": [0.4, 0.5],
                }.items():
                    for replicate, score in enumerate(scores, start=1):
                        rows.append(
                            {
                                "domain": "year",
                                "name": f"{year}_{control}_{replicate:03d}",
                                "control": control,
                                "replicate": replicate,
                                "year": year,
                                "go_id": "GO:1",
                                "metapath": "BPpG<rG",
                                "mean_score": score + (0.1 if year == 2024 else 0.0),
                            }
                        )

            pd.DataFrame(rows).to_csv(summaries_dir / "summary_smoke.csv", index=False)

            cmd = [
                sys.executable,
                "scripts/experiments/year_null_variance_experiment.py",
                "--summaries-dir",
                str(summaries_dir),
                "--output-dir",
                str(output_dir),
                "--b-values",
                "1,2",
                "--seeds",
                "11,22",
            ]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

            exp_dir = output_dir / "year_null_variance_experiment"
            feature_path = exp_dir / "feature_variance_summary.csv"
            overall_path = exp_dir / "overall_variance_summary.csv"
            plot_path = exp_dir / "variance_overall_by_group.png"

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
