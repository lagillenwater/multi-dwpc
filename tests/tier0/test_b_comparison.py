import pandas as pd

from scripts.experiments.tier0_b_comparison import (
    ROW_COMPOSITION_CAVEAT,
    build_comparison,
    render_markdown,
)


def _synthetic_curve_df(n_values: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "b": [10, 10],
            "length_bucket": ["L=2", "L>=3"],
            "n": n_values,
            "n_excluded_nan": [1, 2],
            "median_active_strata": [1.0, 2.0],
            "min_active_strata": [1, 1],
            "spearman_rho_analytical_vs_mc_highb": [0.9, 0.95],
            "jaccard_selected_analytical_vs_mc_highb": [1.0, 1.0],
            "max_abs_relative_std_error": [0.1, 0.2],
        }
    )


def test_build_comparison_tags_and_stacks_rows_without_mutating_inputs():
    promiscuity_df = _synthetic_curve_df([8, 23])
    metaedge_df = _synthetic_curve_df([10, 23])

    combined = build_comparison(promiscuity_df, metaedge_df)

    assert len(combined) == 4
    assert set(combined["strategy"]) == {"promiscuity", "metaedge_degree"}
    # Inputs must not be mutated in place (build_comparison copies before tagging).
    assert "strategy" not in promiscuity_df.columns
    assert "strategy" not in metaedge_df.columns

    prom_rows = combined[combined["strategy"] == "promiscuity"]
    assert sorted(prom_rows["n"].tolist()) == [8, 23]


def test_build_comparison_preserves_row_composition_difference():
    """The whole point of Finding 4: two strategies over the same B/bucket
    can legitimately report different n counts (different rows excluded as
    NaN) -- build_comparison must not silently align/merge them into one
    row, which would hide that.
    """
    promiscuity_df = _synthetic_curve_df([8, 23])
    metaedge_df = _synthetic_curve_df([10, 23])

    combined = build_comparison(promiscuity_df, metaedge_df)
    l2 = combined[(combined["b"] == 10) & (combined["length_bucket"] == "L=2")]
    assert set(l2["n"]) == {8, 10}


def test_render_markdown_includes_caveat_note_and_per_b_sections():
    promiscuity_df = _synthetic_curve_df([8, 23])
    metaedge_df = _synthetic_curve_df([10, 23])
    combined = build_comparison(promiscuity_df, metaedge_df)

    md = render_markdown(combined)

    assert ROW_COMPOSITION_CAVEAT in md
    assert "## B=10" in md
    assert "median_active_strata" in md
    assert "min_active_strata" in md


def test_render_markdown_handles_multiple_b_values():
    promiscuity_df = pd.concat(
        [_synthetic_curve_df([8, 23]).assign(b=10), _synthetic_curve_df([9, 24]).assign(b=30)],
        ignore_index=True,
    )
    metaedge_df = pd.concat(
        [_synthetic_curve_df([10, 23]).assign(b=10), _synthetic_curve_df([10, 24]).assign(b=30)],
        ignore_index=True,
    )
    combined = build_comparison(promiscuity_df, metaedge_df)

    md = render_markdown(combined)
    assert "## B=10" in md
    assert "## B=30" in md
