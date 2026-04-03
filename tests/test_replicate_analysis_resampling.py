import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lv_replicate_analysis import build_b_seed_runs as build_lv_b_seed_runs  # noqa: E402
from src.replicate_analysis import build_b_seed_runs as build_generic_b_seed_runs  # noqa: E402
from src.year_replicate_analysis import build_b_seed_runs as build_year_b_seed_runs  # noqa: E402


class ReplicateAnalysisResamplingTests(unittest.TestCase):
    def test_generic_build_b_seed_runs_expected_diff_for_b_equals_all_replicates(self):
        rows = []
        for year, real_score, perm_scores, rand_scores in [
            (2016, 10.0, (6.0, 8.0), (3.0, 5.0)),
            (2024, 20.0, (15.0, 17.0), (9.0, 11.0)),
        ]:
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
            for rep, score in enumerate(perm_scores, start=1):
                rows.append(
                    {
                        "domain": "year",
                        "name": f"{year}_permuted_{rep:03d}",
                        "control": "permuted",
                        "replicate": rep,
                        "year": year,
                        "go_id": "GO:1",
                        "metapath": "BPpG<rG",
                        "mean_score": score,
                    }
                )
            for rep, score in enumerate(rand_scores, start=1):
                rows.append(
                    {
                        "domain": "year",
                        "name": f"{year}_random_{rep:03d}",
                        "control": "random",
                        "replicate": rep,
                        "year": year,
                        "go_id": "GO:1",
                        "metapath": "BPpG<rG",
                        "mean_score": score,
                    }
                )

        runs_df = build_generic_b_seed_runs(
            pd.DataFrame(rows),
            b_values=[2],
            seeds=[11, 22],
            join_keys=["year", "go_id", "metapath"],
            replicate_pool_keys=["year", "control"],
        )

        self.assertEqual(len(runs_df), 8)
        subset = runs_df[
            (runs_df["year"] == 2016)
            & (runs_df["control"] == "permuted")
            & (runs_df["go_id"] == "GO:1")
            & (runs_df["metapath"] == "BPpG<rG")
        ].copy()
        self.assertEqual(set(subset["seed"].tolist()), {11, 22})
        self.assertTrue((subset["null_mean_score"] == 7.0).all())
        self.assertTrue((subset["real_mean_score"] == 10.0).all())
        self.assertTrue((subset["diff"] == 3.0).all())

    def test_generic_build_b_seed_runs_raises_when_b_exceeds_available_replicates(self):
        summary_df = pd.DataFrame(
            [
                {
                    "domain": "year",
                    "name": "2016_real",
                    "control": "real",
                    "replicate": 0,
                    "year": 2016,
                    "go_id": "GO:1",
                    "metapath": "BPpG<rG",
                    "mean_score": 1.0,
                },
                {
                    "domain": "year",
                    "name": "2016_permuted_001",
                    "control": "permuted",
                    "replicate": 1,
                    "year": 2016,
                    "go_id": "GO:1",
                    "metapath": "BPpG<rG",
                    "mean_score": 0.5,
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "Requested max B=2"):
            build_generic_b_seed_runs(
                summary_df,
                b_values=[2],
                seeds=[11],
                join_keys=["year", "go_id", "metapath"],
                replicate_pool_keys=["year", "control"],
            )

    def test_year_and_lv_wrappers_match_generic_builder(self):
        year_df = pd.DataFrame(
            [
                {
                    "domain": "year",
                    "name": "2016_real",
                    "control": "real",
                    "replicate": 0,
                    "year": 2016,
                    "go_id": "GO:1",
                    "metapath": "BPpG<rG",
                    "mean_score": 1.0,
                },
                {
                    "domain": "year",
                    "name": "2016_permuted_001",
                    "control": "permuted",
                    "replicate": 1,
                    "year": 2016,
                    "go_id": "GO:1",
                    "metapath": "BPpG<rG",
                    "mean_score": 0.4,
                },
                {
                    "domain": "year",
                    "name": "2016_permuted_002",
                    "control": "permuted",
                    "replicate": 2,
                    "year": 2016,
                    "go_id": "GO:1",
                    "metapath": "BPpG<rG",
                    "mean_score": 0.6,
                },
            ]
        )
        expected_year = build_generic_b_seed_runs(
            year_df,
            b_values=[2],
            seeds=[11],
            join_keys=["year", "go_id", "metapath"],
            replicate_pool_keys=["year", "control"],
        ).reset_index(drop=True)
        actual_year = build_year_b_seed_runs(year_df, [2], [11]).reset_index(drop=True)
        self.assertTrue(actual_year.equals(expected_year))

        lv_df = pd.DataFrame(
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
                    "mean_score": 2.0,
                },
                {
                    "domain": "lv",
                    "name": "lv_permuted_001",
                    "control": "permuted",
                    "replicate": 1,
                    "lv_id": "LV1",
                    "target_set_id": "TS1",
                    "target_set_label": "set1",
                    "node_type": "Gene",
                    "metapath": "GpBP",
                    "mean_score": 1.0,
                },
                {
                    "domain": "lv",
                    "name": "lv_permuted_002",
                    "control": "permuted",
                    "replicate": 2,
                    "lv_id": "LV1",
                    "target_set_id": "TS1",
                    "target_set_label": "set1",
                    "node_type": "Gene",
                    "metapath": "GpBP",
                    "mean_score": 1.2,
                },
            ]
        )
        expected_lv = build_generic_b_seed_runs(
            lv_df,
            b_values=[2],
            seeds=[11],
            join_keys=["lv_id", "target_set_id", "target_set_label", "node_type", "metapath"],
            replicate_pool_keys=["control"],
        ).reset_index(drop=True)
        actual_lv = build_lv_b_seed_runs(lv_df, [2], [11]).reset_index(drop=True)
        self.assertTrue(actual_lv.equals(expected_lv))


if __name__ == "__main__":
    unittest.main()
