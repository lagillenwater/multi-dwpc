# Capacity-stratified hurdle+adaptive analytical null — validation design

**As of** 2026-08-28.
**Branch** `fix/random-null-stratified-srswor` (extends the Tier 0 validation harness on this branch).
**Status** OPEN — validation task; integration into the query path is a separate follow-on task.

## Context and decision trail

The Tier 0 program validates HetNetEX-MD's `exact_resampling_moments` as a
deterministic replacement for the Monte-Carlo stratified-SRSWOR gene-subset
null. Three findings drive this design:

1. **Promiscuity stratification was vacuous** (2026-08-13 whole-branch
   review): near-constant key at LV scale plus `_assign_rank_bins`'s
   positional tie-breaking produced Entrez-ID-block strata; median null z was
   22.2 with zero rows within 0.5 of the 1.65 threshold, so analytical-vs-MC
   concordance was guaranteed rather than tested.
2. **Metaedge-degree stratification was the first informative test** (median
   z 1.42, six rows within 0.5 of threshold; formula held — Jaccard converges
   to 1.0 by B≈1000, moments within 0.1–0.6% of B=60,000 MC). Practical pass
   rate at z ≥ 1.65 moved from 29/34 (85%) to 17/36 (47%).
3. **Two residual weaknesses in metaedge-degree rank-binning**: (a) on sparse
   edge types (CbG 92% zero-degree, CdG 86%, CuG 85%) 8–9 of 10 rank bins are
   arbitrary partitions of the zero block while the entire nonzero tail
   (degree 1–516 in CbG) is lumped into 1–2 bins — under-correcting degree
   exactly where correction matters; (b) first-hop degree is a proxy that
   cannot see later hops of the metapath.

Decisions from the 2026-08-28 brainstorm (this session):

- The null to integrate and ship is HetNetEX-MD's exact resampling moments
  (the validated object). The configuration-model HetNetEX line (5 unpushed
  commits on `pr-pipeline-updates-and-webtool-reqs`) is dropped as a backend;
  its adapter/app wiring will be reworked onto this backend in the
  integration task.
- Before integration, validate an upgraded stratification: **hurdle +
  adaptive binning**, primarily on a **leave-target-out metapath capacity**
  key, with first-hop degree under the same binning as an ablation arm.
- HetNetEX-MD is consumed pinned to the canonical upstream
  (`tghosh30/HetNetEx-MD@26f5ba8`, to which the `lagillenwater` fork is
  currently identical); the fork is the contribution vehicle only.

## Binding definitions (unchanged from the Tier 0 harness)

- Per-gene score for feature (lv_id, metapath m with target node t):
  `x_g = arcsinh(DWPC_m(g, t) / raw_mean)` (`src/lv_precompute.py`).
- Observed statistic: `T = (1/K) * sum_{g in S} x_g` over the LV's real gene
  set S of size K.
- Null: stratified SRSWOR — draw `counts[s]` genes from `pools[s]` per
  stratum s, real genes excluded from their own candidate pools
  (`src/tier0/_pool_assembly.pools_from_bins` self-exclusion semantics).
- Moments via `exact_resampling_moments` only (per
  `src/tier0/hetnetex_md_import.py` policy); z = (T_obs − mu)/sigma and
  p = Phi_bar(z) derived locally; the library's p-values are never surfaced.

## New stratification strategies

Both strategies share one binning contract, replacing `_assign_rank_bins`:

- **Hurdle stratum**: all genes with key exactly 0 form one stratum. For the
  capacity key this is an exact exchangeability class (zero capacity implies
  identically zero score for every target of that metapath).
- **Value-respecting adaptive bins over positive keys**: strata are contiguous
  in key value; genes with equal key are never split across strata (this
  eliminates the positional-tie-break artifact class by construction). Bins
  are built greedily from distinct key values in ascending order, closing a
  bin once it holds at least `min_stratum_size` genes (default 50, a
  parameter of the strategy).
- **Feasibility fallback**: at pool-construction time, if a stratum's
  candidate pool (after self-exclusion) is smaller than its real-gene count,
  merge that stratum into its lower-key neighbor (the next-higher neighbor
  when the deficient stratum is the lowest) and record the merge in the
  row's output. This is part of the null's definition and
  applies identically to the analytical and MC arms.
- **Partition properties**: fixed per feature — a function of the metapath,
  the feature's target node, and the graph only; never of the gene set under
  test. This preserves the fixed-partition premise of the exact moment
  formulas.

**S1 — `capacity_hurdle_adaptive` (primary).** Key for gene g and feature
(m, t): `c_g = sum_{t' != t} DWPC_m(g, t')` on the **raw** (untransformed)
DWPC scale — the row sum of the metapath's DWPC matrix minus the target
column's entry. Captures every hop with exactly the damping the statistic
uses, while excluding the feature's own target so the key remains a generic
"reach along this metapath" covariate, not the outcome.

**S2 — `metaedge_degree_hurdle_adaptive` (ablation).** First-hop
metaedge-specific degree (the existing key of
`src/tier0/metaedge_degree_pool_construction.py`) under the same
hurdle+adaptive binning. Isolates how much change comes from the binning
upgrade versus the multi-hop key.

## Capacity computation

- Source: `data/dwpc_cache/dwpc_{metapath}_d0.50.npz` (45 GB locally; only
  the sampled rows' metapaths are needed for this task).
- Per metapath: one O(nnz) row-sum pass; per feature: subtract the target
  column's raw entries. Cached per metapath alongside the harness's existing
  per-first-hop caching.
- No new pipeline stage: for the validation this is computed inside the
  strategy; the integration task decides where it lands in production
  (natural home: one extra line in `precompute_gene_feature_scores`, which
  already holds each matrix in CSC form).

## Compute plan

- **Local**: development, unit tests, and a small synthetic smoke run that
  exercises the real code path (synthetic scores + synthetic capacity keys,
  a handful of rows) — shape and sign checked before any cluster time.
- **Alpine**: the full validation run (concordance, B-sweep, both controls,
  diagnostics) as a batch job on the standing account/partition/QOS
  (amc-general / acpu / cpu-normal). Deployment is git-only: push the
  branch, pull on Alpine, submit — no direct file copies.
- **Verify inputs on Alpine before submitting**: the DWPC cache for the
  sampled metapaths and the `end_to_end_2026_4_23` substrate must exist on
  the Alpine checkout/scratch (scratch purges periodically; a prior run's
  cache may be gone). If missing, regenerating or staging them is its own
  approved step, not a silent side effect of the job.
- **Job logs are pulled back** beside the outputs under
  `output/tier0_capacity_hurdle/` when the run finishes — the log is part of
  the run's record.
- Every Alpine-side non-read-only command and every `git push` is shown
  before running and waits for explicit approval.

## Validation experiments

All reuse the existing three-arm harness (`scripts/experiments/tier0_*.py`);
strategy is already a parameter, and the MC reference draws from the same
`pools`/`counts` as the analytical arm, so both arms see identical partitions
by construction. Seeding follows the harness convention (`random_state=0`,
hash-derived per-row seeds). Outputs to `output/tier0_capacity_hurdle/`.

1. **Three-arm concordance + B-convergence** on the same stratified
   48-row subsample (L=2 / L>=3, floor-pinned crossing): analytical vs
   high-B MC (B = 10,000) and analytical vs original pipeline z, with the
   B-sweep (10, 30, 100, 300, 1000, 3000, 10,000) for S1 and S2.
2. **Negative control (calibration)**: for each sampled row, draw 200
   stratum-matched random gene sets, score them with the analytical z;
   pooled across rows, the fraction with z ≥ 1.65 must sit inside a 95%
   binomial interval around 0.05, and a QQ plot against N(0,1) is produced.
3. **Positive control (planted signal)**: for each sampled row, draw a base
   set from the null, then replace a fraction f of the drawn genes with the
   highest-`x_g` genes available in the same stratum's pool — the stratum
   profile is preserved exactly while target-specific signal is injected.
   Sweep f in {0.1, 0.25, 0.5}; report the minimum f recovered (z ≥ 1.65)
   per row.
4. **Diagnostics**: near-threshold density (rows with |z − 1.65| ≤ 0.5 —
   the vacuousness lesson: a concordance result is only as informative as
   this count); pass-rate table across promiscuity, metaedge_degree (10
   rank bins), S2, and S1 on the same rows; count and identity of fallback
   merges; wall-time per row for pool construction and moments.

## Success criteria

- **Formula concordance**: Spearman rho (analytical vs B=10,000 MC) ≥ 0.99
  per length bucket; max |relative sd error| ≤ 5%; Jaccard at z ≥ 1.65 equal
  to 1.0 at B = 10,000, except for rows whose analytical z lies within two
  MC standard errors of the threshold — such flips are MC sampling noise on
  genuinely near-threshold rows (the rows this design exists to have) and
  are recorded, not counted as failures.
- **Calibration**: negative-control tail fraction within the 95% binomial
  interval around 0.05; no systematic QQ deviation in the upper tail.
- **Power**: every row recovers the planted signal at f = 0.5 under S1; the
  distribution of minimum recovered f is reported for S1 vs S2. If S1 fails
  recovery that S2 passes on the same rows, that is evidence of
  over-conditioning: S2 becomes the shipping strategy and the decision is
  recorded here with the data.
- **Informativeness**: near-threshold density > 0 under the shipping
  strategy (else the concordance claim is flagged as weak, per the
  promiscuity lesson, and the subsample is re-drawn to include near-threshold
  rows before any go decision).
- **Runtime sanity**: analytical per-row cost within the same order as the
  current metaedge-degree strategy (sub-millisecond moments; capacity prep
  amortized per metapath).

## Out of scope

- Query-path/app integration and the upstream PR (follow-on task; branch cut
  from `upstream/main` per the 2026-08-28 brainstorm).
- The PROCESS/SPEC/STATE documentation system and Tier 0 write-up
  (follow-on docs task; note `docs/` is currently gitignored on the main
  checkout's branch — the prior Tier 0 specs are untracked and must be
  rescued into version control there).
- Batch pipeline Monte-Carlo nulls (`scripts/permutation_null_datasets.py`,
  `scripts/random_null_datasets.py`) — remain as-is.
- Network-permutation null family; `network_null_moments` /
  `aggregate_network_null` remain excluded per the import-module policy.

## Decisions

- **2026-08-28** — initial design, from the brainstorm that reviewed the
  Himmelstein et al. 2023 degree-grouping method (exact degree grouping with
  gamma-hurdle zeros and empirical-P fallback) and measured on this repo's
  data: exact-degree matching is computationally free (0.15 ms/row vs
  0.08 ms/row at 10 bins; finer strata touch smaller pools) but statistically
  infeasible for hubs (GpBP: 106 genes with unique degree; GiG: degree-8611
  hub with an empty exact pool), motivating hurdle+adaptive binning; the
  capacity key was chosen over joint or weighted-calibration designs because
  it subsumes first-hop degree, needs no kernel changes, and keeps the
  partition fixed per feature.
