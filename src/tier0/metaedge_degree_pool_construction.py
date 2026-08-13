"""Metaedge-degree stratified-SRSWOR pool construction: replaces LV-membership
promiscuity (near-zero resolution at LV scale -- only 102 of 20,945 genes
have any nonzero promiscuity across 3 LVs, and all 48 rows sampled under it
collapsed to a single stratum, per
output/tier0_offline_recompute/summary.md's Caveats section) with degree in
the metapath's own first-hop edge type as the stratification key.

Does NOT reuse src.dwpc_direct.parse_metapath -- that function cannot parse
metapaths using directional-arrow notation (G<r/Gr>), which covers 21% of
this substrate's metapaths (regulates is a `forward`-only edge type, not
`both`, in data/metagraph.json). Uses hetmatpy's own metapath parser
instead (via src.dwpc_direct.HetMat._hetmatpy), and resolves the on-disk
edge-matrix filename by checking existence directly rather than assuming a
naming convention -- verified against all 453 real metapaths in the
substrate during design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.bipartite_nulls import _assign_rank_bins
from src.dwpc_direct import HetMat
from src.tier0._pool_assembly import pools_from_bins

N_BINS = 10


class MetaedgeDegreePoolStrategy:
    """Callable with the same (lv_id, feature_idx, substrate_dir) -> (scores,
    pools, counts, observed) signature as pool_construction.build_pools_and_counts
    -- a drop-in replacement wherever that function is passed as pool_fn.

    Caches loaded adjacency matrices and computed bins per first-hop
    metaedge filename, since many metapaths for the same LV share a first
    hop (17 distinct first-hop types across the current 3-LV substrate's
    453 metapaths, per the design doc's empirical check).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._hetmat = HetMat(self.data_dir)
        self._bins_by_file: dict[str, np.ndarray] = {}
        self._gene_order_checked: set[str] = set()

    def _check_gene_order(self, gene_ids: np.ndarray, substrate_dir: Path) -> None:
        """Guard the assumption `__call__` otherwise makes silently:
        `bin_of_row` (from `_bins_for_first_hop`) is indexed by this
        hetmat's own Gene node position order, while `gene_ids`/
        `gene_feature_scores` row order comes from the substrate. If a
        future substrate reordered or subsetted genes relative to the
        hetmat, `bin_of_row[i]` would silently refer to the wrong gene for
        every row, scrambling every stratum with no error. Checked once per
        substrate_dir (cached) rather than on every call, since Gene.tsv
        doesn't change between calls for the same substrate/hetmat pair.
        """
        key = str(substrate_dir)
        if key in self._gene_order_checked:
            return
        hetmat_gene_ids = self._hetmat.get_nodes("Gene")["identifier"].to_numpy()
        if not np.array_equal(np.asarray(gene_ids), hetmat_gene_ids):
            raise ValueError(
                f"gene_ids.npy order in {substrate_dir} does not match the "
                f"Gene node position order in this hetmat ({self.data_dir}); "
                "bin_of_row is indexed by hetmat Gene position and cannot "
                "be safely aligned to gene_ids/gene_feature_scores row "
                "order. Verify this substrate wasn't built against a "
                "reordered or subsetted gene universe."
            )
        self._gene_order_checked.add(key)

    def _resolve_first_hop(self, metapath: str) -> tuple[str, str]:
        """Returns (filename, axis) where axis is 'row' if the metapath's
        starting gene is the matrix's row/source axis, 'col' if it's the
        column/target axis. Tries the edge's own (source, kind, target)
        filename first, then its inverse's -- whichever actually exists on
        disk is the canonical stored form.
        """
        mp_obj = self._hetmat._hetmatpy.metagraph.metapath_from_abbrev(metapath)
        e0 = mp_obj.edges[0]
        for edge, is_inverse in ((e0, False), (e0.inverse, True)):
            fname = f"{edge.source.abbrev}{edge.kind_abbrev}{edge.target.abbrev}.sparse.npz"
            if (self.data_dir / "edges" / fname).exists():
                axis = "col" if is_inverse else "row"
                return fname, axis
        raise ValueError(f"could not resolve first-hop matrix file for metapath {metapath!r}")

    def _bins_for_first_hop(self, metapath: str) -> np.ndarray:
        fname, axis = self._resolve_first_hop(metapath)
        cache_key = f"{fname}:{axis}"
        if cache_key in self._bins_by_file:
            return self._bins_by_file[cache_key]

        matrix = sparse.load_npz(self.data_dir / "edges" / fname).tocsr()
        if axis == "row":
            degree = np.asarray(matrix.sum(axis=1)).flatten()
        else:
            degree = np.asarray(matrix.sum(axis=0)).flatten()

        bins = _assign_rank_bins(pd.Series(degree), N_BINS).to_numpy()
        self._bins_by_file[cache_key] = bins
        return bins

    def __call__(
        self, lv_id: str, feature_idx: int, substrate_dir: Path
    ) -> tuple[np.ndarray, list[np.ndarray], list[int], float]:
        substrate_dir = Path(substrate_dir)
        gene_ids = np.load(substrate_dir / "gene_ids.npy", allow_pickle=True)
        gene_scores = np.load(substrate_dir / "gene_feature_scores.npy")
        top_genes = pd.read_csv(substrate_dir / "lv_top_genes.csv")
        real_scores = pd.read_csv(substrate_dir / "real_feature_scores.csv")
        feature_manifest = pd.read_csv(substrate_dir / "feature_manifest.csv")

        self._check_gene_order(gene_ids, substrate_dir)

        scores = gene_scores[:, feature_idx].astype(float)

        metapath = feature_manifest[
            feature_manifest["feature_idx"] == feature_idx
        ].iloc[0]["metapath"]
        bin_of_row = self._bins_for_first_hop(metapath)

        this_lv_genes = set(top_genes[top_genes["lv_id"] == lv_id]["gene_identifier"])
        real_row_idx = np.flatnonzero(np.isin(gene_ids, list(this_lv_genes)))
        pools, counts = pools_from_bins(bin_of_row, real_row_idx, N_BINS)

        row = real_scores[
            (real_scores["lv_id"] == lv_id) & (real_scores["feature_idx"] == feature_idx)
        ].iloc[0]
        observed = float(row["real_mean"])

        return scores, pools, counts, observed
