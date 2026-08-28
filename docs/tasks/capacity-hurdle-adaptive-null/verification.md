# Capacity-stratified hurdle+adaptive null — verification against spec success criteria

**As of** 2026-08-28. Scores every criterion in `design.md`'s "Success criteria"
section against the committed artifacts under `output/tier0_capacity_hurdle/`
(commit `9525647`). All commands run with `conda activate multi_dwpc &&`.

## Run record

- **Alpine campaign**: prewarm job `31770079`, validation job `31770080`, both
  `COMPLETED`. Run from the `/projects` checkout at commit `b415be0`. Logs
  committed at `output/tier0_capacity_hurdle/logs/cap-prewarm-31770079.out`
  and `.../cap-validate-31770080.out`.
- **Substrate**: Alpine-resident
  `output/e2e_full_stratsrswor_full_20260811_135731/lv_experiment`, verified
  same-frame as the local `output/end_to_end_2026_4_23/lv_experiment (1)`:
  identical `feature_manifest.csv` and `lv_top_genes.csv` md5 checksums;
  `gene_feature_scores.npy` identical shape and sum, with nnz differing by 97
  of ~1.94M entries — consistent with float32 rounding jitter, not a
  different substrate. This identity check is recorded in the campaign
  commit message (`9525647`) and reproduced here as the binding evidence for
  "same data" claims below.
- **Env caveats**: two campaign attempts failed before the clean run,
  both from a stale Alpine conda env that had drifted from
  `environment.yml`'s own declarations:
  1. `hetnetex_md` was not the editable install pinned to submodule commit
     `26f5ba8` — fixed by reinstalling per `environment.yml`.
  2. `tabulate` was missing from the env — fixed by installing it per
     `environment.yml`.
  Both fixes brought the env into line with what `environment.yml` already
  specified; no new dependency was introduced. The clean run (jobs
  `31770079`/`31770080`) is the first attempt after both fixes.

---

## 1. Formula concordance

**Criterion**: Spearman rho (analytical vs B=10,000 MC) >= 0.99 per length
bucket; max relative sd error <= 5%; Jaccard at z >= 1.65 equal to 1.0,
except rows within two MC standard errors of the 1.65 threshold (recorded,
not counted as failures).

```
conda activate multi_dwpc && python -c "
import pandas as pd
for s in ['capacity_hurdle_adaptive','metaedge_degree_hurdle_adaptive']:
    df = pd.read_csv(f'output/tier0_capacity_hurdle/b_convergence/{s}/curve_data.csv')
    sub = df[df.b==10000][['length_bucket','spearman_rho_analytical_vs_mc_highb','max_abs_relative_std_error','jaccard_selected_analytical_vs_mc_highb','n','n_excluded_nan']]
    print(s); print(sub.to_string(index=False)); print()
"
```

Output:

```
capacity_hurdle_adaptive
length_bucket  spearman_rho_analytical_vs_mc_highb  max_abs_relative_std_error  jaccard_selected_analytical_vs_mc_highb  n  n_excluded_nan
          L=2                                  1.0                    0.010955                                      1.0 10               8
         L>=3                                  1.0                    0.039359                                      1.0 24               6

metaedge_degree_hurdle_adaptive
length_bucket  spearman_rho_analytical_vs_mc_highb  max_abs_relative_std_error  jaccard_selected_analytical_vs_mc_highb  n  n_excluded_nan
          L=2                                  1.0                    0.010423                                      1.0 10               8
         L>=3                                  1.0                    0.018045                                      1.0 24               6
```

Both strategies: rho = 1.0 in both length buckets (>= 0.99, pass); max
relative sd error 1.0-3.9% (<= 5%, pass); Jaccard = 1.0 in both buckets
(pass).

**Row-level flip check** (`rows_at_max_b.csv`, all 48 sampled rows, B=10,000;
MC z-standard-error approximated as `sqrt((1 + z^2/2) / B)`):

```
conda activate multi_dwpc && python -c "
import pandas as pd, numpy as np
for s in ['capacity_hurdle_adaptive','metaedge_degree_hurdle_adaptive']:
    df = pd.read_csv(f'output/tier0_capacity_hurdle/b_convergence/{s}/rows_at_max_b.csv')
    flips = df[df.selected_analytical != df.selected_mc_highb]
    print(s, 'n rows', len(df), 'flips', len(flips))
    se = np.sqrt((1 + df.z_mc_highb**2/2)/df.b_mc_highb)
    near = df[np.abs(df.z_mc_highb - 1.65) <= 2*se]
    print('  rows within 2*SE of 1.65:', len(near))
"
```

Output:

```
capacity_hurdle_adaptive n rows 48 flips 0
  rows within 2*SE of 1.65: 0
metaedge_degree_hurdle_adaptive n rows 48 flips 0
  rows within 2*SE of 1.65: 0
```

Zero selection flips between analytical and B=10,000 MC for either strategy,
and zero rows fall within 2 MC standard errors of the 1.65 threshold at this
B — so the "except rows within 2 SE" carve-out is vacuously satisfied (no
flips exist to carve out).

**Caveat — zero-variance exclusions**: 14 of 48 sampled rows (8 at L=2, 6 at
L>=3, identical rows for both strategies since both are excluded on
`std_analytical`/`std_mc_highb` being NaN, i.e. zero-variance null
distributions) are excluded from the rho/Jaccard/sd-error computation. This
matches `n_valid=34` in `pass_rates.csv` (48 - 14 = 34). This is the same
zero-variance-null caveat flagged in the project's scientific-concerns
memory; the concordance claim above is over the 34 rows with a well-defined
null variance, not all 48 sampled rows.

**Verdict: PASS** for both S1 and S2 on rho, sd error, and Jaccard, on the
34 rows with nonzero null variance.

---

## 2. Calibration

**Criterion**: negative-control tail fraction at z >= 1.65 lies inside the
95% binomial interval around 0.05; no systematic QQ deviation in the upper
tail.

```
conda activate multi_dwpc && python -c "
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
for s in ['capacity_hurdle_adaptive', 'metaedge_degree_hurdle_adaptive']:
    z = pd.read_csv(f'output/tier0_capacity_hurdle/negative_control_{s}.csv')['z_draw'].dropna()
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(z, dist='norm', plot=ax)
    ax.set_title(f'Negative-control z vs N(0,1): {s} (n={len(z)})')
    fig.savefig(f'output/tier0_capacity_hurdle/qq_negative_{s}.png', dpi=150, bbox_inches='tight')
    print(s, 'upper-tail 0.05-quantile z:', np.quantile(z, 0.95))
"
```

Output:

```
capacity_hurdle_adaptive upper-tail 0.05-quantile z: 1.6483883615535564
metaedge_degree_hurdle_adaptive upper-tail 0.05-quantile z: 1.6894024723174346
```

`negative_summary_*.csv`:

```
strategy,n_draws,n_tail,tail_fraction,ci_low,ci_high,nominal
capacity_hurdle_adaptive,6800,340,0.05,0.04494026254369027,0.05544988244603849,0.050000000000000044
metaedge_degree_hurdle_adaptive,6800,357,0.0525,0.047318255919706835,0.058069560658313646,0.050000000000000044
```

`capacity_hurdle_adaptive`: tail fraction 340/6800 = 0.0500, 95% CI
[0.0449, 0.0554] — nominal 0.05 sits exactly at the observed fraction, well
inside the CI. `metaedge_degree_hurdle_adaptive`: tail fraction 357/6800 =
0.0525, 95% CI [0.0473, 0.0581] — 0.05 inside the CI. Empirical 95th
percentiles (1.648 and 1.689) both bracket the nominal 1.65 threshold with
no systematic upper-tail skew.

QQ plots saved:
- `output/tier0_capacity_hurdle/qq_negative_capacity_hurdle_adaptive.png`
- `output/tier0_capacity_hurdle/qq_negative_metaedge_degree_hurdle_adaptive.png`

**Verdict: PASS** for both S1 and S2.

---

## 3. Power

**Criterion**: every row recovers the planted signal at f = 0.5 under S1;
distribution of minimum recovered f reported for S1 vs S2. If S1 fails
recovery that S2 passes on the same rows, S2 becomes the shipping strategy.

```
conda activate multi_dwpc && python -c "
import pandas as pd
for s in ['capacity_hurdle_adaptive','metaedge_degree_hurdle_adaptive']:
    df = pd.read_csv(f'output/tier0_capacity_hurdle/positive_control_{s}.csv')
    f05 = df[df.fraction==0.5]
    print(s, 'rows at f=0.5:', len(f05), 'all recovered:', f05.recovered.all(), 'n not recovered:', (~f05.recovered).sum())
    rec = df[df.recovered].groupby(['lv_id','feature_idx']).fraction.min()
    all_rows = df.groupby(['lv_id','feature_idx']).size().index
    never = set(all_rows) - set(rec.index)
    print('  n rows total:', len(all_rows), 'never recovered (any f):', len(never))
    print('  min-f distribution (recovered rows):')
    print(rec.value_counts().sort_index())
"
```

Output:

```
capacity_hurdle_adaptive rows at f=0.5: 34 all recovered: True n not recovered: 0
  n rows total: 34 never recovered (any f): 0
  min-f distribution (recovered rows):
fraction
0.1    34
Name: count, dtype: int64

metaedge_degree_hurdle_adaptive rows at f=0.5: 34 all recovered: True n not recovered: 0
  n rows total: 34 never recovered (any f): 0
  min-f distribution (recovered rows):
fraction
0.1    34
Name: count, dtype: int64
```

S1 (`capacity_hurdle_adaptive`): all 34 rows recover at f = 0.5 — spec's
minimum requirement is met. Every row also recovers at the smallest tested
fraction, f = 0.1 (identical min-f distribution for S2). S1 fails no row
that S2 passes: the fallback-decision condition ("if S1 fails recovery that
S2 passes on the same rows") does not fire — there is no S1 failure to
compare against. The data fits the rule's default case cleanly (no
ambiguity requiring a dual reading).

**Disclosure — nominal f is a lower bound, not the effective replaced
fraction**: `plant_signal` computes `m = min(ceil(fraction * k_r), k_r)`
*per stratum*, so every active stratum with a nonzero count has at least
one member replaced regardless of how small the nominal fraction is. Under
S1, the median row has 18+ active strata, most with singleton counts, so
the effective replaced fraction at nominal f = 0.1 is roughly
`n_active_strata / K` — several times the nominal value — and the f-sweep
is largely degenerate rather than a granular power curve. Concretely, in
the committed `positive_control_capacity_hurdle_adaptive.csv`, 22 of 34
rows have byte-identical `z_planted` across all three tested fractions
(5 of 34 for S2), and the minimum S1 `z_planted` at f = 0.1 is 3.54 (S2:
2.04) — i.e. even the "smallest" nominal fraction already plants a
strong, easily-recovered signal for most rows. Replacement also removes
the lowest-scoring drawn members and inserts the highest-scoring
candidates, the most favorable reading of "replace a fraction f" of the
pool. None of this affects the criterion actually being scored: the
f = 0.5 recovery requirement passes robustly because the true effective
fraction is always >= the nominal fraction, and the S1-vs-S2 fallback rule
above is unaffected. It does mean the "every row recovers at f = 0.1"
observation is not a granular power result and should not be read as
evidence of sensitivity down to a 10% replacement rate — it reflects a
per-stratum ceiling behavior specified by the planting procedure itself,
not a campaign artifact or implementation bug.

**Decision-rule branch applied**: default branch — S1 ships (S2 fallback
not triggered).

**Verdict: PASS** for S1 (and S2, though S2 is not needed on this
criterion). **Shipping strategy: S1 (`capacity_hurdle_adaptive`)**.

---

## 4. Informativeness

**Criterion**: `n_near_threshold` > 0 for the shipping strategy (else the
concordance claim is flagged as weak per the promiscuity lesson).

Full pass-rate table (`pass_rates.csv`), regenerating and superseding the
previously-unpersisted 17/36 (47%) `metaedge_degree` figure:

```
strategy,n_valid,n_pass,pass_rate,n_near_threshold
promiscuity,34,29,0.8529411764705882,0
metaedge_degree,36,17,0.4722222222222222,6
capacity_hurdle_adaptive,34,5,0.14705882352941177,2
metaedge_degree_hurdle_adaptive,34,7,0.20588235294117646,2
```

- `promiscuity`: 29/34 pass (85%), `n_near_threshold` = 0 — reconfirms the
  2026-08-13 vacuousness finding (no rows near threshold, so its high
  concordance was never a real test).
- `metaedge_degree`: 17/36 pass (47%), `n_near_threshold` = 6 — regenerates
  the previously-unpersisted headline number.
- `capacity_hurdle_adaptive` (S1, shipping): 5/34 pass (14.7%),
  `n_near_threshold` = 2 (> 0, pass).
- `metaedge_degree_hurdle_adaptive` (S2): 7/34 pass (20.6%),
  `n_near_threshold` = 2 (> 0).

**Verdict: PASS** for the shipping strategy S1 — `n_near_threshold` = 2 > 0.
Note the near-threshold density for both hurdle+adaptive strategies (2) is
lower than for plain `metaedge_degree` (6); the concordance test above is
informative but thinner than the metaedge-degree baseline. This is
consistent with hurdle+adaptive binning sharpening pool composition, which
also lowers raw pass rate substantially (29/34 -> 5/34 for the capacity
line, 17/36 -> 7/34 for the metaedge-degree line) — flagged as a scientific
concern for the integration task to interpret (a null that is harder to
clear is not itself a defect, but the magnitude of the pass-rate drop
warrants review before shipping to production thresholds).

---

## 5. Runtime

**Criterion**: analytical per-row cost within the same order as the current
metaedge-degree strategy (sub-millisecond moments; capacity prep amortized
per metapath).

```
conda activate multi_dwpc && python -c "
import pandas as pd
df = pd.read_csv('output/tier0_capacity_hurdle/runtime_benchmark_capacity_hurdle_adaptive.csv')
g = df.groupby('b')[['t_poolfn_ms','t_analytical_ms','t_mc_ms','speedup']].median()
print(g)
print('n rows:', len(df), 'unique (lv,feat):', df[[\"lv_id\",\"feature_idx\"]].drop_duplicates().shape[0])
"
```

Output:

```
       t_poolfn_ms  t_analytical_ms      t_mc_ms      speedup
b
1000     37.292573         0.481093   107.513169   214.031893
10000    37.292573         0.481093  1074.409591  2144.821029
n rows: 96 unique (lv,feat): 48
```

Analytical moment computation: 0.48 ms/row median (sub-millisecond, as
required), independent of B. Pool construction (`t_poolfn_ms`, one-time,
amortized per metapath/feature): 37.3 ms median. MC reference: 107.5 ms/row
at B=1,000, 1074.4 ms/row at B=10,000. Median speedup: 214x at B=1,000,
2,145x at B=10,000 — consistent with the design doc's preliminary numbers
(210x / 1,865x on a different sample).

**Gap**: only `runtime_benchmark_capacity_hurdle_adaptive.csv` was produced
by the campaign (per the validation log, the benchmark step targets the
shipping candidate S1 only, not S2). This is sufficient to score this
criterion (it applies to the shipping strategy), but there is no committed
S2 runtime figure for direct comparison — noted as a minor gap, not a
failure, since S2 was not chosen to ship.

**Verdict: PASS** for S1.

---

## 6. Fallback merges

**Criterion**: count and identity of feasibility-fallback stratum merges,
per strategy.

```
for f in output/tier0_capacity_hurdle/b_convergence/*/merge_log.csv; do echo "$f"; cat "$f"; echo; done
```

Output: all four `merge_log.csv` files (`promiscuity`, `metaedge_degree`,
`capacity_hurdle_adaptive`, `metaedge_degree_hurdle_adaptive`) contain only
the header row (`lv_id,feature_idx,from_stratum,into_stratum`) — zero
fallback merges occurred in any strategy on the 48-row validation sample.

**Clarification — the two legacy strategies have no merge machinery**:
"zero merges" is a real observation only for the two hurdle+adaptive
strategies, which implement the feasibility-fallback merge logic. The
sweep writes each strategy's log via `getattr(pool_fn, "merge_log", [])`,
and `promiscuity`/`metaedge_degree` have no `merge_log` attribute at all —
their empty CSVs are structural (the attribute defaults to `[]`), not a
measured zero from an exercised code path. The feasibility fallback exists
in the S1/S2 implementations but was never exercised by this sample; this
is expected given the sample's `min_stratum_size` default of 50 and the
substrate's stratum pool sizes, but means the fallback path itself is
unexercised by this validation run (noted as a residual gap for the
integration task's own test suite, which should exercise it with unit
tests rather than relying on this sample).

---

## Overall verdict

| Criterion | S1 (`capacity_hurdle_adaptive`) | S2 (`metaedge_degree_hurdle_adaptive`) |
|---|---|---|
| Formula concordance | PASS | PASS |
| Calibration | PASS | PASS |
| Power (f=0.5) | PASS | PASS |
| Informativeness | PASS (n_near_threshold=2) | PASS (n_near_threshold=2) |
| Runtime | PASS | not separately benchmarked |
| Fallback merges | 0 (unexercised) | 0 (unexercised) |

**Shipping strategy: S1, `capacity_hurdle_adaptive`.** The S2-fallback
condition in the Power criterion never fires (S1 fails no recoverable row),
so the default rule applies and S1 ships. All five scored success criteria
pass for S1 on the committed campaign artifacts.

**Concerns carried forward** (see task-11-report.md for the full list):
1. Pass rate at z >= 1.65 drops sharply under hurdle+adaptive binning
   (29/34 promiscuity, 17/36 metaedge_degree -> 5/34 capacity_hurdle_adaptive,
   7/34 metaedge_degree_hurdle_adaptive) — a large shift in what counts as
   "significant" that the integration task should interpret against
   production thresholds, not silently inherit.
2. 14/48 sampled rows are zero-variance and excluded from concordance
   scoring; the concordance claim is over 34/48 rows, not the full sample.
3. Fallback-merge machinery is untested by this validation run (zero
   merges observed); recommend unit-test coverage in the integration task.
4. No S2 runtime benchmark was produced for direct comparison (not needed
   for the shipping decision, but a gap if S2 is revisited later).
