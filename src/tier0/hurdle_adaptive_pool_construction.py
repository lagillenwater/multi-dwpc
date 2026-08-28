# src/tier0/hurdle_adaptive_pool_construction.py
"""Hurdle+adaptive stratified-SRSWOR pool construction (design doc S1/S2).

Both strategies satisfy the harness PoolFn contract
``(lv_id, feature_idx, substrate_dir) -> (scores, pools, counts, observed)``
and differ only in the stratification key: leave-target-out raw-DWPC
capacity (S1, primary) or first-hop metaedge degree (S2, ablation). The
partition is a function of (metapath, target, graph) only; the tested gene
set enters solely through self-exclusion in pools_from_bins and the
feasibility fallback, both shared with the MC arm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dwpc_direct import HetMat
from src.lv_precompute import _map_target_positions
from src.tier0._pool_assembly import pools_from_bins
from src.tier0.capacity import CapacityProvider
from src.tier0.hurdle_adaptive_bins import hurdle_adaptive_bins, merge_deficient_strata
from src.tier0.metaedge_degree_pool_construction import MetaedgeDegreePoolStrategy

PIPELINE_DAMPING = 0.5


class _HurdleAdaptiveStrategyBase:
    def __init__(self, data_dir: Path, min_stratum_size: int = 50):
        self.data_dir = Path(data_dir)
        self.min_stratum_size = min_stratum_size
        self.merge_log: list[dict] = []
        self._hetmat = None
        self._gene_order_checked: set[str] = set()

    # -- test-override points -------------------------------------------
    def _keys(self, metapath: str, lv_id: str, substrate_dir: Path) -> np.ndarray:
        raise NotImplementedError

    def _hetmat_gene_ids(self) -> np.ndarray:
        return self._get_hetmat().get_nodes("Gene")["identifier"].to_numpy()

    # -------------------------------------------------------------------
    def _get_hetmat(self) -> HetMat:
        if self._hetmat is None:
            self._hetmat = HetMat(
                data_dir=self.data_dir, damping=PIPELINE_DAMPING,
                use_disk_cache=True, write_disk_cache=False,
            )
        return self._hetmat

    def _check_gene_order(self, gene_ids: np.ndarray, substrate_dir: Path) -> None:
        # Same guard, same reason as MetaedgeDegreePoolStrategy._check_gene_order:
        # keys are indexed by hetmat Gene position; a reordered substrate would
        # silently scramble every stratum.
        key = str(substrate_dir)
        if key in self._gene_order_checked:
            return
        if not np.array_equal(np.asarray(gene_ids), self._hetmat_gene_ids()):
            raise ValueError(
                f"gene_ids.npy order in {substrate_dir} does not match this "
                "hetmat's Gene node position order; stratification keys cannot "
                "be safely aligned to substrate rows."
            )
        self._gene_order_checked.add(key)

    def __call__(self, lv_id: str, feature_idx: int, substrate_dir: Path):
        substrate_dir = Path(substrate_dir)
        gene_ids = np.load(substrate_dir / "gene_ids.npy", allow_pickle=True)
        gene_scores = np.load(substrate_dir / "gene_feature_scores.npy")
        top_genes = pd.read_csv(substrate_dir / "lv_top_genes.csv")
        real_scores = pd.read_csv(substrate_dir / "real_feature_scores.csv")
        manifest = pd.read_csv(substrate_dir / "feature_manifest.csv")

        self._check_gene_order(gene_ids, substrate_dir)

        scores = gene_scores[:, feature_idx].astype(float)
        metapath = manifest[manifest["feature_idx"] == feature_idx].iloc[0]["metapath"]

        keys = self._keys(metapath, lv_id, substrate_dir)
        if len(keys) != len(gene_ids):
            raise ValueError(
                f"key vector length {len(keys)} != gene universe {len(gene_ids)}"
            )
        bins = hurdle_adaptive_bins(keys, self.min_stratum_size)

        this_lv = set(top_genes[top_genes["lv_id"] == lv_id]["gene_identifier"])
        real_row_idx = np.flatnonzero(np.isin(gene_ids, list(this_lv)))
        pools, counts = pools_from_bins(bins, real_row_idx, int(bins.max()) + 1)
        pools, counts, merges = merge_deficient_strata(pools, counts)
        for from_s, into_s in merges:
            self.merge_log.append(
                dict(lv_id=lv_id, feature_idx=int(feature_idx),
                     from_stratum=int(from_s), into_stratum=int(into_s))
            )

        row = real_scores[
            (real_scores["lv_id"] == lv_id)
            & (real_scores["feature_idx"] == feature_idx)
        ].iloc[0]
        return scores, pools, counts, float(row["real_mean"])


class CapacityHurdleAdaptiveStrategy(_HurdleAdaptiveStrategyBase):
    """S1: leave-target-out raw-DWPC capacity key."""

    def __init__(self, data_dir: Path, min_stratum_size: int = 50):
        super().__init__(data_dir, min_stratum_size)
        self._provider = None
        self._target_pos_by_lv: dict[str, dict[str, int]] = {}

    def _target_position(self, lv_id: str, substrate_dir: Path) -> int:
        key = str(substrate_dir)
        if key not in self._target_pos_by_lv:
            lv_targets = pd.read_csv(Path(substrate_dir) / "lv_targets.csv")
            self._target_pos_by_lv[key] = _map_target_positions(
                hetmat=self._get_hetmat(), lv_targets=lv_targets
            )
        return self._target_pos_by_lv[key][lv_id]

    def _keys(self, metapath: str, lv_id: str, substrate_dir: Path) -> np.ndarray:
        if self._provider is None:
            self._provider = CapacityProvider(self._get_hetmat(), PIPELINE_DAMPING)
        return self._provider.capacity(metapath, self._target_position(lv_id, substrate_dir))


class MetaedgeDegreeHurdleAdaptiveStrategy(_HurdleAdaptiveStrategyBase):
    """S2 (ablation): first-hop metaedge degree under the same binning."""

    def __init__(self, data_dir: Path, min_stratum_size: int = 50):
        super().__init__(data_dir, min_stratum_size)
        self._degree_source = MetaedgeDegreePoolStrategy(data_dir=Path(data_dir))

    def _keys(self, metapath: str, lv_id: str, substrate_dir: Path) -> np.ndarray:
        return self._degree_source._degree_for_first_hop(metapath)
