"""Fixed-bin stratified SRSWOR contract for generate_promiscuity_matched_samples.

Universe g0..g5 has promiscuity 0..5 respectively (via dummy annotation edges).
n_bins=2 splits it into bin0={g0,g1,g2} and bin1={g3,g4,g5} by rank. The single
real edge s1->g2 (promiscuity 2, bin0) must only ever be replaced by a bin0
candidate (g0 or g1) -- never a bin1 gene, even though g3/g4 fall inside the old
sliding-window tolerance (+/-2) around promiscuity 2.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.bipartite_nulls import generate_promiscuity_matched_samples

UNIVERSE = ["g0", "g1", "g2", "g3", "g4", "g5"]


def _edges():
    edge_df = pd.DataFrame({"src": ["s1"], "tgt": ["g2"]})
    dummy_rows = (
        [("d1", "g1")]
        + [("s1", "g2"), ("d1", "g2")]
        + [(f"d{i}", "g3") for i in range(1, 4)]
        + [(f"d{i}", "g4") for i in range(1, 5)]
        + [(f"d{i}", "g5") for i in range(1, 6)]
    )
    all_annotations = pd.DataFrame(dummy_rows, columns=["src", "tgt"])
    return edge_df, all_annotations


def _sample(seed):
    edge_df, all_annotations = _edges()
    return generate_promiscuity_matched_samples(
        edge_df=edge_df,
        all_annotations_df=all_annotations,
        source_col="src",
        target_col="tgt",
        target_universe=UNIVERSE,
        n_bins=2,
        random_state=seed,
    )


def test_excludes_real_gene():
    out = _sample(seed=1)
    assert len(out) == 1
    assert out["tgt"].iloc[0] != "g2"


def test_substitute_never_crosses_the_fixed_bin_boundary():
    # g3/g4 sit inside the old +/-2 tolerance window around g2's promiscuity (2)
    # but belong to bin1, not g2's bin0. Fixed-bin SRSWOR must never draw them.
    drawn = {_sample(seed=s)["tgt"].iloc[0] for s in range(30)}
    assert drawn.issubset({"g0", "g1"})
    assert drawn == {"g0", "g1"}  # both bin0 candidates get drawn across seeds


def test_raises_when_bin_candidate_pool_too_small():
    edge_df = pd.DataFrame({"src": ["s1"], "tgt": ["g0"]})
    all_annotations = pd.DataFrame({"src": ["s1"], "tgt": ["g0"]})
    tiny_universe = ["g0"]  # bin 0 has no candidates once real gene g0 is excluded
    with pytest.raises(ValueError):
        generate_promiscuity_matched_samples(
            edge_df=edge_df,
            all_annotations_df=all_annotations,
            source_col="src",
            target_col="tgt",
            target_universe=tiny_universe,
            n_bins=1,
            random_state=1,
        )
