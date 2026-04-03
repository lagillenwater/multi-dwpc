import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.replicate_analysis import summarize_rank_stability  # noqa: E402


def _toy_rank_df() -> pd.DataFrame:
    rows = []
    ranks_by_seed = {
        11: [("M1", 1), ("M2", 2), ("M3", 3)],
        22: [("M2", 1), ("M1", 2), ("M3", 3)],
        33: [("M1", 1), ("M3", 2), ("M2", 3)],
    }
    for seed, ranking in ranks_by_seed.items():
        for metapath, rank in ranking:
            rows.append(
                {
                    "year": 2016,
                    "control": "permuted",
                    "b": 2,
                    "go_id": "GO:1",
                    "seed": seed,
                    "metapath": metapath,
                    "metapath_rank": rank,
                }
            )
    return pd.DataFrame(rows)


class ReplicateAnalysisRankStabilityTests(unittest.TestCase):
    def test_summarize_rank_stability_includes_rbo_and_topk_columns(self):
        rank_df = _toy_rank_df()
        pairwise_df, entity_df, overall_df = summarize_rank_stability(
            rank_df,
            outer_keys=["year", "control", "b", "go_id"],
            replicate_col="seed",
            feature_col="metapath",
            rank_col="metapath_rank",
            top_k=[1, "all"],
            rbo_p=0.9,
        )

        self.assertIn("rbo", pairwise_df.columns)
        self.assertIn("mean_rbo", entity_df.columns)
        self.assertIn("mean_rbo", overall_df.columns)
        self.assertIn("topk_jaccard_1", pairwise_df.columns)
        self.assertIn("topk_jaccard_all", pairwise_df.columns)
        self.assertIn("mean_topk_jaccard_1", entity_df.columns)
        self.assertIn("mean_topk_jaccard_all", entity_df.columns)
        self.assertEqual(int(entity_df.loc[0, "n_pairs"]), 3)

    def test_summarize_rank_stability_disables_rbo_when_requested(self):
        rank_df = _toy_rank_df()
        pairwise_df, entity_df, overall_df = summarize_rank_stability(
            rank_df,
            outer_keys=["year", "control", "b", "go_id"],
            replicate_col="seed",
            feature_col="metapath",
            rank_col="metapath_rank",
            top_k=10,
            rbo_p=None,
        )

        self.assertNotIn("rbo", pairwise_df.columns)
        self.assertNotIn("mean_rbo", entity_df.columns)
        self.assertNotIn("mean_rbo", overall_df.columns)
        self.assertIn("mean_spearman_rho", overall_df.columns)
        self.assertIn("mean_topk_jaccard_10", overall_df.columns)


if __name__ == "__main__":
    unittest.main()
