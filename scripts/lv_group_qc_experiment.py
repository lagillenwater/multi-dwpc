#!/usr/bin/env python3
"""Build exploratory LVQC packets for LV dual-null experiments."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if Path.cwd().name == "scripts":
    REPO_ROOT = Path("..").resolve()
else:
    REPO_ROOT = Path.cwd()

sys.path.insert(0, str(REPO_ROOT))
from src.bipartite_nulls import calculate_target_membership_counts  # noqa: E402
from src.replicate_analysis import rank_features  # noqa: E402


def _load_csv(path: Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    suffix = path.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    if required_columns:
        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _iqr(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = set_a | set_b
    if not union:
        return np.nan
    return float(len(set_a & set_b) / len(union))


def _safe_p90(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    return float(values.quantile(0.90))


def _reverse_metapath_abbrev(metapath: str) -> str:
    node_abbrevs = {"G", "BP", "CC", "MF", "PW", "A", "D", "C", "SE", "S", "PC"}
    edge_abbrevs = {"p", "i", "c", "r", ">", "<", "a", "d", "u", "e", "b", "t", "l"}
    tokens: list[str] = []
    pos = 0
    while pos < len(metapath):
        if pos + 2 <= len(metapath) and metapath[pos : pos + 2] in node_abbrevs:
            tokens.append(metapath[pos : pos + 2])
            pos += 2
        elif metapath[pos] in node_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        elif metapath[pos] in edge_abbrevs:
            tokens.append(metapath[pos])
            pos += 1
        else:
            pos += 1
    direction_map = {">": "<", "<": ">"}
    reversed_tokens = [direction_map.get(token, token) for token in reversed(tokens)]
    return "".join(reversed_tokens)


def _build_metapath_length_map(stats_df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in stats_df.itertuples(index=False):
        mp = str(row.metapath)
        length = int(row.length)
        out[mp] = length
        out[_reverse_metapath_abbrev(mp)] = length
    return out


def _real_base_edges(output_dir: Path) -> pd.DataFrame:
    top_genes = _load_csv(output_dir / "lv_top_genes.csv", ["lv_id", "gene_identifier"])
    lv_targets = _load_csv(output_dir / "lv_targets.csv", ["lv_id", "target_id"])
    return (
        top_genes.merge(lv_targets[["lv_id", "target_id"]], on="lv_id", how="inner")[["lv_id", "target_id", "gene_identifier"]]
        .drop_duplicates()
        .sort_values(["lv_id", "gene_identifier"])
        .reset_index(drop=True)
    )


def _descriptor_panel(
    output_dir: Path,
    analysis_dir: Path,
    stats_path: Path,
    gap_b: int,
    score_gap_max_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_manifest = _load_csv(output_dir / "feature_manifest.csv", ["target_set_id", "metapath"])
    real_scores = _load_csv(output_dir / "real_feature_scores.csv", ["lv_id", "target_set_id", "real_mean"])
    target_sets = _load_csv(output_dir / "target_sets.csv", ["target_set_id", "target_set_label", "target_id"])
    lv_map = _load_csv(output_dir / "lv_target_map.csv", ["lv_id", "target_set_id"])
    runs_df = _load_csv(
        analysis_dir / "all_runs_long.csv",
        ["control", "b", "seed", "lv_id", "target_set_id", "metapath", "diff"],
    )
    stats_df = _load_csv(stats_path, ["metapath", "length"])
    length_map = _build_metapath_length_map(stats_df)
    feature_manifest = feature_manifest.copy()
    feature_manifest["metapath_length"] = feature_manifest["metapath"].map(length_map)

    base_edges = _real_base_edges(output_dir)
    gene_prom = calculate_target_membership_counts(
        base_edges,
        source_col="lv_id",
        target_col="gene_identifier",
    ).rename(columns={"gene_identifier": "gene_identifier"})
    target_prom = calculate_target_membership_counts(
        target_sets,
        source_col="target_set_id",
        target_col="target_id",
    ).rename(columns={"target_id": "target_id"})

    descriptor_rows = []
    for row in (
        lv_map.merge(
            target_sets[["target_set_id", "target_set_label"]].drop_duplicates(),
            on="target_set_id",
            how="left",
        )
        .drop_duplicates()
        .itertuples(index=False)
    ):
        gene_subset = base_edges[base_edges["lv_id"].astype(str) == str(row.lv_id)].copy()
        gene_subset = gene_subset.merge(gene_prom, on="gene_identifier", how="left")
        target_subset = target_sets[target_sets["target_set_id"].astype(str) == str(row.target_set_id)].copy()
        target_subset = target_subset.merge(target_prom, on="target_id", how="left")
        feature_subset = feature_manifest[feature_manifest["target_set_id"].astype(str) == str(row.target_set_id)].copy()
        real_subset = real_scores[
            (real_scores["lv_id"].astype(str) == str(row.lv_id))
            & (real_scores["target_set_id"].astype(str) == str(row.target_set_id))
        ].copy()
        descriptor_row = {
            "group_id": f"{row.lv_id}__{row.target_set_id}",
            "lv_id": str(row.lv_id),
            "target_set_id": str(row.target_set_id),
            "target_set_label": str(row.target_set_label),
            "n_genes": int(gene_subset["gene_identifier"].nunique()),
            "target_set_size": int(target_subset["target_id"].nunique()),
            "n_candidate_metapaths": int(feature_subset["metapath"].nunique()),
            "gene_promiscuity_median": float(gene_subset["promiscuity"].median()) if not gene_subset.empty else np.nan,
            "gene_promiscuity_iqr": _iqr(gene_subset["promiscuity"]),
            "gene_promiscuity_p90": _safe_p90(gene_subset["promiscuity"]),
            "gene_promiscuity_max": float(gene_subset["promiscuity"].max()) if not gene_subset.empty else np.nan,
            "target_promiscuity_median": float(target_subset["promiscuity"].median()) if not target_subset.empty else np.nan,
            "target_promiscuity_iqr": _iqr(target_subset["promiscuity"]),
            "target_promiscuity_p90": _safe_p90(target_subset["promiscuity"]),
            "target_promiscuity_max": float(target_subset["promiscuity"].max()) if not target_subset.empty else np.nan,
            "score_sparsity": float(np.isclose(real_subset["real_mean"].astype(float), 0.0, atol=1e-8).mean())
            if not real_subset.empty
            else np.nan,
        }
        for length in (1, 2, 3):
            descriptor_row[f"n_metapaths_len_{length}"] = int(
                (feature_subset["metapath_length"].astype("Int64") == length).sum()
            )
        descriptor_rows.append(descriptor_row)
    descriptor_df = pd.DataFrame(descriptor_rows)

    gap_rows = []
    score_gap_rows = []
    gap_subset = runs_df[runs_df["b"].astype(int) == int(gap_b)].copy()
    for key, group in gap_subset.groupby(["control", "lv_id", "target_set_id"], sort=True):
        control, lv_id, target_set_id = key
        mean_diff = (
            group.groupby("metapath", as_index=False)["diff"]
            .mean()
            .sort_values(["diff", "metapath"], ascending=[False, True])
            .reset_index(drop=True)
        )
        group_diff_std = float(mean_diff["diff"].astype(float).std(ddof=1)) if len(mean_diff) > 1 else np.nan
        top5_gap = np.nan
        top10_gap = np.nan
        if len(mean_diff) >= 6:
            top5_gap = float(mean_diff.loc[4, "diff"] - mean_diff.loc[5, "diff"])
        if len(mean_diff) >= 11:
            top10_gap = float(mean_diff.loc[9, "diff"] - mean_diff.loc[10, "diff"])
        gap_rows.append(
            {
                "lv_id": str(lv_id),
                "target_set_id": str(target_set_id),
                "control": str(control),
                "top5_gap": top5_gap,
                "top10_gap": top10_gap,
            }
        )
        max_k = min(int(score_gap_max_k), max(len(mean_diff) - 1, 0))
        for k in range(1, max_k + 1):
            upper = mean_diff.loc[k - 1]
            lower = mean_diff.loc[k]
            score_gap_rows.append(
                {
                    "lv_id": str(lv_id),
                    "target_set_id": str(target_set_id),
                    "control": str(control),
                    "b": int(gap_b),
                    "rank_upper": int(k),
                    "rank_lower": int(k + 1),
                    "metapath_upper": str(upper["metapath"]),
                    "metapath_lower": str(lower["metapath"]),
                    "diff_upper": float(upper["diff"]),
                    "diff_lower": float(lower["diff"]),
                    "score_gap": float(upper["diff"] - lower["diff"]),
                    "group_diff_std": group_diff_std,
                    "standardized_score_gap": (
                        float((upper["diff"] - lower["diff"]) / group_diff_std)
                        if pd.notna(group_diff_std) and abs(group_diff_std) > 1e-12
                        else np.nan
                    ),
                }
            )
    gap_df = pd.DataFrame(gap_rows)
    score_gap_df = pd.DataFrame(score_gap_rows)
    if not gap_df.empty:
        gap_wide = gap_df.pivot(
            index=["lv_id", "target_set_id"],
            columns="control",
            values=["top5_gap", "top10_gap"],
        )
        gap_wide.columns = [f"{metric}_{control}" for metric, control in gap_wide.columns]
        gap_wide = gap_wide.reset_index()
        descriptor_df = descriptor_df.merge(gap_wide, on=["lv_id", "target_set_id"], how="left")

    return descriptor_df, gap_df, score_gap_df


def _random_qc(output_dir: Path, requested_tolerance: int) -> pd.DataFrame:
    manifest = _load_csv(
        output_dir / "replicate_manifest.csv",
        ["control", "replicate", "source_path"],
    )
    lv_map = _load_csv(output_dir / "lv_target_map.csv", ["lv_id", "target_set_id"])
    rows = []
    for meta in manifest[manifest["control"].astype(str) == "random"].itertuples(index=False):
        artifact = _load_csv(Path(meta.source_path))
        if "real_promiscuity" not in artifact.columns or "sampled_promiscuity" not in artifact.columns:
            raise ValueError(
                f"Random-control artifact is missing match metadata columns: {meta.source_path}"
            )
        if "target_set_id" not in artifact.columns:
            artifact = artifact.merge(lv_map, on="lv_id", how="left")
        artifact["abs_diff"] = np.abs(
            artifact["real_promiscuity"].astype(float) - artifact["sampled_promiscuity"].astype(float)
        )
        grouped = artifact.groupby(["lv_id", "target_set_id"], as_index=False).agg(
            random_match_mae=("abs_diff", "mean"),
            random_match_p90_absdiff=("abs_diff", lambda x: float(pd.Series(x).quantile(0.90))),
            random_exact_match_rate=("abs_diff", lambda x: float(np.mean(np.asarray(x) == 0))),
            random_within_tolerance_rate=(
                "abs_diff",
                lambda x: float(np.mean(np.asarray(x) <= int(requested_tolerance))),
            ),
            random_exceeds_tolerance_rate=(
                "abs_diff",
                lambda x: float(np.mean(np.asarray(x) > int(requested_tolerance))),
            ),
        )
        grouped["replicate"] = int(meta.replicate)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    qc_df = pd.concat(rows, ignore_index=True)
    return (
        qc_df.groupby(["lv_id", "target_set_id"], as_index=False)
        .agg(
            n_random_replicates=("replicate", "nunique"),
            random_match_mae=("random_match_mae", "mean"),
            random_match_p90_absdiff=("random_match_p90_absdiff", "mean"),
            random_exact_match_rate=("random_exact_match_rate", "mean"),
            random_within_tolerance_rate=("random_within_tolerance_rate", "mean"),
            random_exceeds_tolerance_rate=("random_exceeds_tolerance_rate", "mean"),
        )
        .sort_values(["lv_id", "target_set_id"])
        .reset_index(drop=True)
    )


def _rbo_score(rank_a: list[str], rank_b: list[str], p: float) -> float:
    """Compute extrapolated finite-list rank-biased overlap for two ranked lists."""
    if not rank_a or not rank_b:
        return np.nan
    depth = min(len(rank_a), len(rank_b))
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    overlap_sum = 0.0
    overlap_at_depth = 0.0
    for d in range(1, depth + 1):
        seen_a.add(rank_a[d - 1])
        seen_b.add(rank_b[d - 1])
        overlap_at_depth = len(seen_a & seen_b) / float(d)
        overlap_sum += overlap_at_depth * (p ** (d - 1))
    return float(((1.0 - p) * overlap_sum) + (overlap_at_depth * (p ** depth)))


def _permuted_qc(output_dir: Path) -> pd.DataFrame:
    manifest = _load_csv(
        output_dir / "replicate_manifest.csv",
        ["control", "replicate", "source_path"],
    )
    lv_map = _load_csv(output_dir / "lv_target_map.csv", ["lv_id", "target_set_id"])
    real_edges = _real_base_edges(output_dir)
    real_sets = {
        str(lv_id): set(group["gene_identifier"].astype(str).tolist())
        for lv_id, group in real_edges.groupby("lv_id", sort=True)
    }
    real_source_sizes = {k: len(v) for k, v in real_sets.items()}
    real_target_degree = real_edges["gene_identifier"].astype(str).value_counts().sort_index()

    rows = []
    per_lv_rep_sets: dict[tuple[str, str], list[set[str]]] = {}
    for meta in manifest[manifest["control"].astype(str) == "permuted"].itertuples(index=False):
        artifact = _load_csv(Path(meta.source_path), ["lv_id", "gene_identifier"])
        if "target_set_id" not in artifact.columns:
            artifact = artifact.merge(lv_map, on="lv_id", how="left")
        perm_target_degree = artifact["gene_identifier"].astype(str).value_counts().sort_index()
        target_degree_match = int(perm_target_degree.equals(real_target_degree))
        for (lv_id, target_set_id), group in artifact.groupby(["lv_id", "target_set_id"], sort=True):
            lv_id = str(lv_id)
            target_set_id = str(target_set_id)
            perm_set = set(group["gene_identifier"].astype(str).tolist())
            real_set = real_sets[lv_id]
            source_degree_match = int(len(perm_set) == real_source_sizes[lv_id])
            overlap = len(real_set & perm_set) / float(len(real_set)) if real_set else np.nan
            rows.append(
                {
                    "lv_id": lv_id,
                    "target_set_id": target_set_id,
                    "replicate": int(meta.replicate),
                    "perm_source_degree_match": source_degree_match,
                    "perm_target_degree_match": target_degree_match,
                    "perm_edge_overlap_with_real": overlap,
                }
            )
            per_lv_rep_sets.setdefault((lv_id, target_set_id), []).append(perm_set)
    if not rows:
        return pd.DataFrame()
    qc_df = pd.DataFrame(rows)
    pairwise_rows = []
    for key, rep_sets in per_lv_rep_sets.items():
        lv_id, target_set_id = key
        pairwise_vals = []
        for a, b in itertools.combinations(rep_sets, 2):
            pairwise_vals.append(_jaccard(a, b))
        pairwise_rows.append(
            {
                "lv_id": lv_id,
                "target_set_id": target_set_id,
                "perm_pairwise_overlap_mean": float(np.nanmean(pairwise_vals)) if pairwise_vals else np.nan,
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows)
    out = (
        qc_df.groupby(["lv_id", "target_set_id"], as_index=False)
        .agg(
            n_permuted_replicates=("replicate", "nunique"),
            perm_source_degree_match_rate=("perm_source_degree_match", "mean"),
            perm_target_degree_match_rate=("perm_target_degree_match", "mean"),
            perm_edge_overlap_with_real_mean=("perm_edge_overlap_with_real", "mean"),
            perm_edge_overlap_with_real_p90=(
                "perm_edge_overlap_with_real",
                lambda x: float(pd.Series(x).quantile(0.90)),
            ),
        )
        .sort_values(["lv_id", "target_set_id"])
        .reset_index(drop=True)
    )
    return out.merge(pairwise_df, on=["lv_id", "target_set_id"], how="left")


def _calibration_envelope(descriptor_df: pd.DataFrame) -> pd.DataFrame:
    key_vars = [
        "n_genes",
        "target_set_size",
        "n_candidate_metapaths",
        "gene_promiscuity_median",
        "gene_promiscuity_iqr",
        "gene_promiscuity_p90",
        "gene_promiscuity_max",
        "target_promiscuity_median",
        "target_promiscuity_iqr",
        "target_promiscuity_p90",
        "target_promiscuity_max",
        "score_sparsity",
    ]
    rows = []
    for var in key_vars:
        series = pd.to_numeric(descriptor_df[var], errors="coerce")
        rows.append(
            {
                "variable": var,
                "median": float(series.median()),
                "p05": float(series.quantile(0.05)),
                "p95": float(series.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _descriptor_deviation_table(
    descriptor_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
) -> pd.DataFrame:
    key_vars = envelope_df["variable"].astype(str).tolist()
    envelope = envelope_df.set_index("variable").to_dict(orient="index")
    rows = []
    for row in descriptor_df.itertuples(index=False):
        for var in key_vars:
            value = float(getattr(row, var))
            median = float(envelope[var]["median"])
            p05 = float(envelope[var]["p05"])
            p95 = float(envelope[var]["p95"])
            scale = p95 - p05
            if abs(scale) < 1e-12:
                deviation = np.nan
            else:
                deviation = (value - median) / scale
            rows.append(
                {
                    "group_id": row.group_id,
                    "lv_id": row.lv_id,
                    "target_set_id": row.target_set_id,
                    "variable": var,
                    "value": value,
                    "median": median,
                    "p05": p05,
                    "p95": p95,
                    "deviation": deviation,
                    "out_of_envelope": bool((value < p05) or (value > p95)),
                }
            )
    return pd.DataFrame(rows)


def _load_or_build_rank_df(analysis_dir: Path) -> pd.DataFrame:
    rank_path = analysis_dir / "metapath_rank_table.csv"
    if rank_path.exists():
        return _load_csv(
            rank_path,
            ["control", "b", "seed", "lv_id", "target_set_id", "metapath", "metapath_rank"],
        )
    runs_df = _load_csv(
        analysis_dir / "all_runs_long.csv",
        ["control", "b", "seed", "lv_id", "target_set_id", "metapath", "diff"],
    )
    return rank_features(
        runs_df,
        rank_group_keys=["control", "b", "seed", "lv_id", "target_set_id"],
        feature_col="metapath",
        score_col="diff",
        rank_col="metapath_rank",
    )


def _within_null_stability_summary(
    rank_df: pd.DataFrame,
    rbo_p: float,
) -> pd.DataFrame:
    rows = []
    for key, group in rank_df.groupby(["control", "b", "lv_id", "target_set_id"], sort=True):
        control, b, lv_id, target_set_id = key
        by_seed: dict[int, pd.DataFrame] = {}
        for seed, seed_df in group.groupby("seed", sort=True):
            ordered = seed_df.sort_values("metapath_rank")
            by_seed[int(seed)] = ordered[["metapath", "metapath_rank"]].copy()
        seed_ids = sorted(by_seed.keys())
        spearman_vals = []
        rbo_vals = []
        for idx, seed_a in enumerate(seed_ids):
            for seed_b in seed_ids[idx + 1 :]:
                df_a = by_seed[seed_a].rename(columns={"metapath_rank": "rank_a"})
                df_b = by_seed[seed_b].rename(columns={"metapath_rank": "rank_b"})
                merged = df_a.merge(df_b, on="metapath", how="inner")
                if len(merged) >= 2:
                    spearman_vals.append(
                        _spearman_from_rank_arrays(merged["rank_a"], merged["rank_b"])
                    )
                list_a = df_a.sort_values("rank_a")["metapath"].astype(str).tolist()
                list_b = df_b.sort_values("rank_b")["metapath"].astype(str).tolist()
                rbo_vals.append(_rbo_score(list_a, list_b, p=rbo_p))
        rows.append(
            {
                "control": str(control),
                "b": int(b),
                "lv_id": str(lv_id),
                "target_set_id": str(target_set_id),
                "n_pairs": int(len(spearman_vals)),
                "mean_spearman_rho": float(np.nanmean(spearman_vals)) if spearman_vals else np.nan,
                "mean_rbo": float(np.nanmean(rbo_vals)) if rbo_vals else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["control", "lv_id", "target_set_id", "b"]).reset_index(drop=True)


def _spearman_from_rank_arrays(rank_a: pd.Series, rank_b: pd.Series) -> float:
    if len(rank_a) < 2 or len(rank_b) < 2:
        return np.nan
    return float(rank_a.rank(method="average").corr(rank_b.rank(method="average"), method="pearson"))


def _between_null_agreement_summary(
    rank_df: pd.DataFrame,
    rbo_p: float,
) -> pd.DataFrame:
    rows = []
    for key, group in rank_df.groupby(["b", "lv_id", "target_set_id"], sort=True):
        b, lv_id, target_set_id = key
        control_means = (
            group.groupby(["control", "metapath"], as_index=False)["metapath_rank"]
            .mean()
            .rename(columns={"metapath_rank": "mean_rank"})
        )
        if set(control_means["control"].astype(str)) != {"permuted", "random"}:
            continue
        perm_df = control_means[control_means["control"].astype(str) == "permuted"][["metapath", "mean_rank"]]
        rand_df = control_means[control_means["control"].astype(str) == "random"][["metapath", "mean_rank"]]
        merged = perm_df.merge(rand_df, on="metapath", suffixes=("_permuted", "_random"))
        if len(merged) < 2:
            continue
        row = {
            "b": int(b),
            "lv_id": str(lv_id),
            "target_set_id": str(target_set_id),
            "mean_spearman_rho": _spearman_from_rank_arrays(
                merged["mean_rank_permuted"],
                merged["mean_rank_random"],
            ),
            "mean_rbo": _rbo_score(
                merged.sort_values("mean_rank_permuted")["metapath"].astype(str).tolist(),
                merged.sort_values("mean_rank_random")["metapath"].astype(str).tolist(),
                p=rbo_p,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["lv_id", "target_set_id", "b"]).reset_index(drop=True)


def _within_null_pairwise_diagnostics(
    rank_df: pd.DataFrame,
    rbo_p: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    prefix_rows = []
    for key, group in rank_df.groupby(["control", "b", "lv_id", "target_set_id"], sort=True):
        control, b, lv_id, target_set_id = key
        by_seed: dict[int, pd.DataFrame] = {}
        for seed, seed_df in group.groupby("seed", sort=True):
            ordered = seed_df.sort_values("metapath_rank")
            by_seed[int(seed)] = ordered[["metapath", "metapath_rank"]].copy()
        seed_ids = sorted(by_seed.keys())
        for idx, seed_a in enumerate(seed_ids):
            for seed_b in seed_ids[idx + 1 :]:
                df_a = by_seed[seed_a].rename(columns={"metapath_rank": "rank_a"})
                df_b = by_seed[seed_b].rename(columns={"metapath_rank": "rank_b"})
                merged = df_a.merge(df_b, on="metapath", how="inner")
                if merged.empty:
                    continue
                list_a = df_a.sort_values("rank_a")["metapath"].astype(str).tolist()
                list_b = df_b.sort_values("rank_b")["metapath"].astype(str).tolist()
                pair_rows.append(
                    {
                        "control": str(control),
                        "b": int(b),
                        "lv_id": str(lv_id),
                        "target_set_id": str(target_set_id),
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "n_common_metapaths": int(len(merged)),
                        "spearman_rho": _spearman_from_rank_arrays(
                            merged["rank_a"],
                            merged["rank_b"],
                        )
                        if len(merged) >= 2
                        else np.nan,
                        "rbo": _rbo_score(list_a, list_b, p=rbo_p),
                    }
                )
                max_k = min(len(list_a), len(list_b))
                prefix_a: set[str] = set()
                prefix_b: set[str] = set()
                for k in range(1, max_k + 1):
                    prefix_a.add(list_a[k - 1])
                    prefix_b.add(list_b[k - 1])
                    prefix_rows.append(
                        {
                            "control": str(control),
                            "b": int(b),
                            "lv_id": str(lv_id),
                            "target_set_id": str(target_set_id),
                            "seed_a": int(seed_a),
                            "seed_b": int(seed_b),
                            "k": int(k),
                            "prefix_jaccard": _jaccard(prefix_a, prefix_b),
                        }
                    )
    return (
        pd.DataFrame(pair_rows).sort_values(
            ["control", "lv_id", "target_set_id", "b", "seed_a", "seed_b"]
        ).reset_index(drop=True),
        pd.DataFrame(prefix_rows).sort_values(
            ["control", "lv_id", "target_set_id", "b", "seed_a", "seed_b", "k"]
        ).reset_index(drop=True),
    )


def _between_null_rank_scatter_table(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in rank_df.groupby(["b", "lv_id", "target_set_id", "metapath"], sort=True):
        b, lv_id, target_set_id, metapath = key
        mean_ranks = (
            group.groupby("control", as_index=False)["metapath_rank"]
            .mean()
            .rename(columns={"metapath_rank": "mean_rank"})
        )
        if set(mean_ranks["control"].astype(str)) != {"permuted", "random"}:
            continue
        perm_rank = float(
            mean_ranks.loc[mean_ranks["control"].astype(str) == "permuted", "mean_rank"].iloc[0]
        )
        rand_rank = float(
            mean_ranks.loc[mean_ranks["control"].astype(str) == "random", "mean_rank"].iloc[0]
        )
        rows.append(
            {
                "b": int(b),
                "lv_id": str(lv_id),
                "target_set_id": str(target_set_id),
                "metapath": str(metapath),
                "mean_rank_permuted": perm_rank,
                "mean_rank_random": rand_rank,
            }
        )
    return pd.DataFrame(rows).sort_values(["lv_id", "target_set_id", "b", "metapath"]).reset_index(drop=True)

def _metric_snapshot(
    descriptor_df: pd.DataFrame,
    random_qc_df: pd.DataFrame,
    perm_qc_df: pd.DataFrame,
    within_df: pd.DataFrame,
    between_df: pd.DataFrame,
    score_gap_df: pd.DataFrame,
    snapshot_b: int,
    near_tie_threshold: float,
) -> pd.DataFrame:
    within_subset = within_df[within_df["b"].astype(int) == int(snapshot_b)].copy()
    within_wide = (
        within_subset.pivot(
            index=["lv_id", "target_set_id"],
            columns="control",
            values=["mean_spearman_rho", "mean_rbo"],
        )
        .sort_index(axis=1)
    )
    within_wide.columns = [f"{metric}_{control}" for metric, control in within_wide.columns]
    within_wide = within_wide.reset_index()

    between_subset = between_df[between_df["b"].astype(int) == int(snapshot_b)].copy()
    between_subset = between_subset.rename(
        columns={
            "mean_spearman_rho": "between_null_mean_spearman_rho",
            "mean_rbo": "between_null_mean_rbo",
        }
    )

    score_subset = score_gap_df[score_gap_df["b"].astype(int) == int(snapshot_b)].copy()
    near_tie_df = (
        score_subset.assign(
            near_tie=lambda df: df["standardized_score_gap"].astype(float) < float(near_tie_threshold)
        )
        .groupby(["lv_id", "target_set_id", "control"], as_index=False)
        .agg(
            mean_standardized_top10_gap=(
                "standardized_score_gap",
                lambda x: float(pd.Series(x).astype(float).head(10).mean()),
            ),
            near_tie_count_top10=("near_tie", lambda x: int(pd.Series(x).head(10).sum())),
            near_tie_count_top20=("near_tie", lambda x: int(pd.Series(x).head(20).sum())),
        )
    )
    near_tie_wide = near_tie_df.pivot(
        index=["lv_id", "target_set_id"],
        columns="control",
        values=["mean_standardized_top10_gap", "near_tie_count_top10", "near_tie_count_top20"],
    )
    near_tie_wide.columns = [f"{metric}_{control}" for metric, control in near_tie_wide.columns]
    near_tie_wide = near_tie_wide.reset_index()

    summary_df = descriptor_df.merge(random_qc_df, on=["lv_id", "target_set_id"], how="left")
    summary_df = summary_df.merge(perm_qc_df, on=["lv_id", "target_set_id"], how="left")
    summary_df = summary_df.merge(within_wide, on=["lv_id", "target_set_id"], how="left")
    summary_df = summary_df.merge(
        between_subset[["lv_id", "target_set_id", "between_null_mean_spearman_rho", "between_null_mean_rbo"]],
        on=["lv_id", "target_set_id"],
        how="left",
    )
    summary_df = summary_df.merge(near_tie_wide, on=["lv_id", "target_set_id"], how="left")
    summary_df["snapshot_b"] = int(snapshot_b)
    return summary_df.sort_values(["lv_id", "target_set_id"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/lv_experiment")
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Defaults to <output-dir>/lv_rank_stability_experiment",
    )
    parser.add_argument(
        "--qc-output-dir",
        default=None,
        help="Defaults to <output-dir>/lv_group_qc_experiment",
    )
    parser.add_argument("--metapath-stats-path", default="data/metapath-dwpc-stats.tsv")
    parser.add_argument("--gap-b", type=int, default=5)
    parser.add_argument("--score-gap-max-k", type=int, default=20)
    parser.add_argument("--rbo-p", type=float, default=0.9)
    parser.add_argument("--diagnostic-b", type=int, default=5)
    parser.add_argument("--random-promiscuity-tolerance", type=int, default=2)
    parser.add_argument("--near-tie-threshold", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = (
        Path(args.analysis_dir)
        if args.analysis_dir
        else output_dir / "lv_rank_stability_experiment"
    )
    qc_output_dir = (
        Path(args.qc_output_dir)
        if args.qc_output_dir
        else output_dir / "lv_group_qc_experiment"
    )
    qc_output_dir.mkdir(parents=True, exist_ok=True)

    descriptor_df, gap_df, score_gap_df = _descriptor_panel(
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        stats_path=Path(args.metapath_stats_path),
        gap_b=int(args.gap_b),
        score_gap_max_k=int(args.score_gap_max_k),
    )
    random_qc_df = _random_qc(
        output_dir=output_dir,
        requested_tolerance=int(args.random_promiscuity_tolerance),
    )
    perm_qc_df = _permuted_qc(output_dir=output_dir)
    rank_df = _load_or_build_rank_df(analysis_dir)
    entity_df = _within_null_stability_summary(
        rank_df=rank_df,
        rbo_p=float(args.rbo_p),
    )
    between_null_df = _between_null_agreement_summary(
        rank_df=rank_df,
        rbo_p=float(args.rbo_p),
    )
    pairwise_diag_df, prefix_diag_df = _within_null_pairwise_diagnostics(
        rank_df=rank_df,
        rbo_p=float(args.rbo_p),
    )
    between_scatter_df = _between_null_rank_scatter_table(rank_df)
    envelope_df = _calibration_envelope(descriptor_df)
    descriptor_deviation_df = _descriptor_deviation_table(descriptor_df, envelope_df)
    metric_snapshot_df = _metric_snapshot(
        descriptor_df=descriptor_df,
        random_qc_df=random_qc_df,
        perm_qc_df=perm_qc_df,
        within_df=entity_df,
        between_df=between_null_df,
        score_gap_df=score_gap_df,
        snapshot_b=int(args.gap_b),
        near_tie_threshold=float(args.near_tie_threshold),
    )

    descriptor_df.to_csv(qc_output_dir / "descriptor_panel.csv", index=False)
    gap_df.to_csv(qc_output_dir / "gap_summary.csv", index=False)
    score_gap_df.to_csv(qc_output_dir / "score_separation_table.csv", index=False)
    random_qc_df.to_csv(qc_output_dir / "random_match_qc.csv", index=False)
    perm_qc_df.to_csv(qc_output_dir / "permuted_null_qc.csv", index=False)
    envelope_df.to_csv(qc_output_dir / "calibration_envelope.csv", index=False)
    descriptor_deviation_df.to_csv(qc_output_dir / "descriptor_deviation.csv", index=False)
    entity_df.to_csv(qc_output_dir / "within_null_stability_summary.csv", index=False)
    between_null_df.to_csv(qc_output_dir / "between_null_agreement.csv", index=False)
    pairwise_diag_df.to_csv(qc_output_dir / "within_null_pairwise_diagnostics.csv", index=False)
    prefix_diag_df.to_csv(qc_output_dir / "within_null_prefix_overlap.csv", index=False)
    between_scatter_df.to_csv(qc_output_dir / "between_null_rank_scatter.csv", index=False)
    metric_snapshot_df.to_csv(qc_output_dir / "lv_qc_metric_snapshot.csv", index=False)
    metric_snapshot_df.to_csv(qc_output_dir / "lv_qc_summary.csv", index=False)

    print(f"Saved QC descriptor panel: {qc_output_dir / 'descriptor_panel.csv'}")
    print(f"Saved random-null QC: {qc_output_dir / 'random_match_qc.csv'}")
    print(f"Saved permuted-null QC: {qc_output_dir / 'permuted_null_qc.csv'}")
    print(f"Saved score separation table: {qc_output_dir / 'score_separation_table.csv'}")
    print(f"Saved calibration envelope: {qc_output_dir / 'calibration_envelope.csv'}")
    print(f"Saved descriptor deviation table: {qc_output_dir / 'descriptor_deviation.csv'}")
    print(f"Saved within-null stability summary: {qc_output_dir / 'within_null_stability_summary.csv'}")
    print(f"Saved between-null agreement summary: {qc_output_dir / 'between_null_agreement.csv'}")
    print(f"Saved within-null pairwise diagnostics: {qc_output_dir / 'within_null_pairwise_diagnostics.csv'}")
    print(f"Saved within-null prefix overlap: {qc_output_dir / 'within_null_prefix_overlap.csv'}")
    print(f"Saved between-null rank scatter table: {qc_output_dir / 'between_null_rank_scatter.csv'}")
    print(f"Saved LVQC metric snapshot: {qc_output_dir / 'lv_qc_metric_snapshot.csv'}")
    print(f"Saved LVQC summary: {qc_output_dir / 'lv_qc_summary.csv'}")


if __name__ == "__main__":
    main()
