# Methods Overview: LV Pipeline and 2016-vs-2024 GO-Term Analysis

This document summarizes the implemented analysis methods in this repository as manuscript-style methods text, with emphasis on null-model design, replicate counts, and statistical definitions.

## 1. Latent Variable (LV) Multi-DWPC Pipeline

### 1.1 LV inputs and target mapping
We analyzed selected latent variables (LV603, LV246, LV57) from a MultiPLIER loading matrix and mapped genes to the Hetionet gene universe (`data/nodes/Gene.tsv`). For each LV, genes were ranked by loading and the top fraction was retained (`top_fraction = 0.005`, i.e., top 0.5%, with at least one gene per LV).

Each LV was mapped to a predefined biological target set:

- LV603 -> neutrophil biological process terms
- LV246 -> adipose tissue anatomy terms
- LV57 -> hypothyroidism disease term

### 1.2 Metapath feature construction
For each target set, Gene-to-target metapaths were selected from `metapath-dwpc-stats.tsv` with maximum length 3 (direct 1-edge metapaths excluded by default). Let:

- $g$ index genes,
- $t$ index target nodes in a target set $T_f$,
- $m_f$ be the metapath for feature $f$,
- $\mu_{m_f}$ be the metapath-specific raw DWPC mean from `metapath-dwpc-stats.tsv`.

The transformed pair score is:

$$x_{g,t,m_f} = \operatorname{arcsinh}\left(\frac{\mathrm{DWPC}_{g,t,m_f}}{\mu_{m_f}}\right)$$

The per-gene feature score is the target-average transformed score:

$$S_{g,f} = \frac{1}{|T_f|}\sum_{t\in T_f} x_{g,t,m_f}$$

For each LV $l$ with top-gene set $G_l$, the observed LV-feature mean is:

$$R_{l,f} = \frac{1}{|G_l|}\sum_{g\in G_l} S_{g,f}$$

### 1.3 Degree-matched null models
Gene degrees were computed from Hetionet sparse edge matrices and discretized into degree quantile bins (default `n_degree_bins = 10`). Two null models were used for each LV-feature combination.

1. Random null (`null_type = random`): for each LV, genes were sampled within degree bins from the global pool excluding that LV's real genes in the same bin.
2. Permuted null (`null_type = permuted`): within each degree bin, pooled real LV genes were permuted and reallocated across LVs according to each LV's per-bin slot counts.

For each replicate $b$, a null LV-feature mean $N^{(b)}_{l,f}$ was computed using the same feature matrix $S_{g,f}$.

### 1.4 Adaptive replicate counts and empirical p-values
Null sampling used adaptive stopping with defaults:

- $B_{\min}=200$,
- $B_{\max}=1000$,
- batch size $=100$,
- stop early if $p < 0.005$ or $p > 0.20$ after $B_{\min}$.

Empirical p-values were estimated as:

$$p_{l,f} = \frac{1 + \sum_{b=1}^{B} \mathbf{1}\!\left(N^{(b)}_{l,f} \ge R_{l,f}\right)}{1 + B}$$

where $B$ is the effective replicate count (`n_eff`) for that LV-feature-null combination.

### 1.5 Effect-size summaries and final support calls
For each replicate and feature, a paired effect size over LV gene slots was computed:

$$d^{(b)}_{l,f} = \frac{\overline{\Delta^{(b)}_{l,f}}}{s\!\left(\Delta^{(b)}_{l,f}\right)},\quad
\Delta^{(b)}_{l,f} = \{S_{g,f}^{\text{real}} - S_{g,f}^{\text{null}(b)}\}_{g\in G_l}$$

The reported null effect size per null type was the median over replicates (`d_median`), yielding $d_{\text{perm}}$ and $d_{\text{rand}}$. Mean-shift summaries were:

$$\mathrm{diff}_{\text{perm}} = R_{l,f} - \bar{N}_{l,f,\text{perm}}, \quad
\mathrm{diff}_{\text{rand}} = R_{l,f} - \bar{N}_{l,f,\text{rand}}$$

Conservative summaries were:

$$\min d = \min(d_{\text{perm}}, d_{\text{rand}}), \quad
\min\mathrm{diff} = \min(\mathrm{diff}_{\text{perm}}, \mathrm{diff}_{\text{rand}})$$

Benjamini-Hochberg FDR correction was applied separately to permutation and random empirical p-values within each `(lv_id, target_set_id)` group. A feature was marked `supported` when:

- `p_perm_fdr < 0.05`,
- `p_rand_fdr < 0.05`,
- `diff_perm > 0`,
- `diff_rand > 0`.

### 1.6 LV subgraph metapath selection and multi-gene plotting
LV subgraph plotting used a two-step selection process: metapath selection, then multi-gene selection within each selected metapath.

Metapaths were selected from `lv_metapath_results.csv` as follows:

1. Restrict to rows with `supported == True`; if no rows were supported, use all rows.
2. Within each `(lv_id, target_set_id)`, rank by descending `min_d`, descending `min_diff`, and ascending $(p_{\mathrm{perm,fdr}} + p_{\mathrm{rand,fdr}})$.
3. Keep the top `top_metapaths` rows per `(lv_id, target_set_id)` (default `10`).

For each selected metapath, all LV gene-target pairs were rescored with real DWPC and ranked by descending `dwpc`. The top `top_pairs` pairs per `(lv_id, target_set_id, metapath)` were retained (default `10`), and top path instances were enumerated per pair (`top_paths`, default `5`).

Final multi-gene network panels were constructed per `(lv_id, target_set_id, metapath, target_id)`. Genes were prioritized by:

- minimum `pair_rank` (higher-ranked DWPC pairs first),
- maximum path score,
- number of retained paths.

Up to `plot_max_genes` genes were plotted per panel (default `10`), requiring at least `plot_min_genes` genes (default `2`). An optional path-length constraint required at least `plot_shared_intermediates_min` intermediate nodes per path (`0` by default, i.e., no additional intermediate-node filtering).

## 2. 2016-vs-2024 GO-Term DWPC Analysis

### 2.1 Cohort construction and growth-focused filtering
The year-comparison workflow begins with Hetionet 2016 BP-Gene associations and 2024 GO BP annotations. GO-gene pairs were classified per GO term as:

- stable: present in both 2016 and 2024,
- added: present in 2024 only.

For each GO term, percent change in gene count was:

$$\mathrm{pct\_change} = \frac{n_{2024} - n_{2016}}{n_{2016}}$$

We retained positive-growth GO terms and applied IQR-based filtering on baseline gene count and percent change:

$$[\;Q_1 - 1.5\,\mathrm{IQR},\; Q_3 + 1.5\,\mathrm{IQR}\;]$$

with an additional minimum baseline size (`min_genes = 10`).

### 2.2 Redundancy pruning by Jaccard overlap
GO-term redundancy was reduced using pairwise gene-set Jaccard similarity:

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}$$

Pairs above threshold (`J > 0.1`) were processed greedily (highest $J$ first); the term with smaller absolute percent change was removed.

### 2.3 Null dataset generation and replicate counts
Two null controls were generated for each year-specific filtered GO-term dataset.

1. Permuted GO-label null (`perm_001`): gene IDs were globally shuffled and reassigned to GO terms while preserving GO-term sizes and total association count.
2. Promiscuity-controlled random null (`random_001`): for each real GO-gene pair, a replacement gene was sampled from other GO terms with matched annotation promiscuity (default tolerance $\pm 2$, expanded as needed).

Current pipeline configuration generates one replicate of each null type per year:

- 2016: `perm_001`, `random_001`
- 2024: `perm_001`, `random_001`

Together with real datasets, this yields six datasets for downstream comparisons.

### 2.4 DWPC computation
DWPC values were computed directly from HetMat sparse matrices (`scripts/compute_dwpc_direct.py`) with damping $w=0.5$. Metapaths were selected as BP-to-Gene metapaths with length $\le 3$ (excluding direct BPpG), then reversed to Gene-to-BP for matrix extraction. API-compatible transformation was:

$$\operatorname{DWPC}^{*}=\operatorname{arcsinh}\left(\frac{\operatorname{DWPC}_{\text{raw}}}{\mu_m}\right)$$

### 2.5 Metapath signature analysis (paired Wilcoxon framework)
For each dataset, DWPC values were aggregated by (GO term, metapath), then paired across shared GO terms to compare:

- `2016_real` vs `2024_real`
- `2016_real` vs `2016_perm`
- `2016_real` vs `2016_random`
- `2024_real` vs `2024_perm`
- `2024_real` vs `2024_random`

For each metapath-statistic combination ($n \ge 10$ paired GO terms), a two-sided Wilcoxon signed-rank test was applied. A rank-biserial effect proxy was computed as:

$$r_{\text{rb}}=\frac{n_{+}-n_{-}}{n}$$

where $n_{+}$ and $n_{-}$ are counts of positive and negative paired differences. Benjamini-Hochberg FDR correction was applied globally and per statistic.

### 2.6 Divergence score analysis (paired t-test and effect size)
For each metapath-statistic and year-control pairing, divergence was quantified on paired GO-term summaries:

$$\Delta_i = x^{\text{real}}_i - x^{\text{ctrl}}_i$$

$$d=\frac{\overline{\Delta}}{s(\Delta)}$$

where $d$ is paired Cohen's $d$. Paired $t$-tests were used for significance ($n \ge 10$ GO terms). Per-year effects were averaged across permutation and random controls:

$$d_{2016}=\frac{d_{2016,\text{perm}}+d_{2016,\text{rand}}}{2}, \quad
d_{2024}=\frac{d_{2024,\text{perm}}+d_{2024,\text{rand}}}{2}$$

A conservative cross-year score was:

$$d_{\text{combined}}=\min(d_{2016}, d_{2024})$$

Combined per-year p-values were summarized via geometric mean of control-specific p-values:

$$p_{2016}=\sqrt{p_{2016,\text{perm}}\,p_{2016,\text{rand}}}, \quad
p_{2024}=\sqrt{p_{2024,\text{perm}}\,p_{2024,\text{rand}}}$$

Metapaths were flagged as jointly significant when both year-specific p-values were below 0.05.

### 2.7 Side-by-side path-instance plot selection (2016 vs 2024)
Side-by-side path-instance plots were generated from `path_instances_<year>.csv` and required exactly two years in the plotting stage. For each selected `(metapath, GO term)` key, both years were rendered in a shared two-panel figure.

Selection of `(metapath, GO term)` keys followed one of three modes:

1. **Unfiltered mode:** include all available keys (optionally restricted to a user-specified metapath).
2. **Paired-difference ranking (`--rank-by-paired-diff`):**
   - load paired real-vs-control GO-term statistics for each year from `paired_go_terms_<year>_vs_perm.csv` and `paired_go_terms_<year>_vs_random.csv`,
   - compute a per-key ranking value from the mean of permutation and random differences (signed or absolute),
   - normalize metapath orientation to BP-first form before grouping,
   - keep top `paired_top_n` GO terms per metapath (default `10`), unioned across years.
3. **Path-change ranking (`--rank-by-path-change`):**
   - for each shared `(metapath, GO term)`, compare the two years using path-instance identity sets (`path_nodes_ids`),
   - score change by either $1-$Jaccard overlap or absolute path-count difference,
   - keep top `path_change_top_n` GO terms per metapath (default `10`).

An optional node-alignment setting (`--align-nodes`) fixed layer-wise node positions across year panels so visual differences reflected path composition changes rather than independent layout variation.
