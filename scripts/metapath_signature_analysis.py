# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.005265, "end_time": "2025-12-03T15:50:07.724096", "exception": false, "start_time": "2025-12-03T15:50:07.718831", "status": "completed"}
# # 3. Metapath Signature Analysis
#
# ## Research Question
#
# Which metapaths show **stable signal patterns** where:
# 1. 2016 real GO annotations differ significantly from 2016 null distributions (permuted and random)
# 2. 2024 real GO annotations differ significantly from 2024 null distributions
# 3. 2016 and 2024 real annotations do NOT significantly differ from each other
#
# ## Objective
#
# Test 12 GO-term-aggregated statistics to identify which statistic most clearly reveals these patterns:
# - Mean and median of DWPC (all values and non-zero only)
# - Mean and median of p-values (all values and non-zero only)
# - Mean and median of standard deviations (all values and non-zero only)

# %% papermill={"duration": 3.598487, "end_time": "2025-12-03T15:50:11.326857", "exception": false, "start_time": "2025-12-03T15:50:07.728370", "status": "completed"}
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

print("Notebook 3: Pairwise Statistical Comparisons of GO-Term-Aggregated Statistics")

# %% papermill={"duration": 0.006075, "end_time": "2025-12-03T15:50:11.336346", "exception": false, "start_time": "2025-12-03T15:50:11.330271", "status": "completed"}
repo_root = Path(__file__).resolve().parent.parent
BASE_NAME = 'all_GO_positive_growth'
data_dir = repo_root / 'output' / 'dwpc_direct' / BASE_NAME / 'results'
output_dir = repo_root / 'output' / 'metapath_analysis' / 'pairwise_statistics'
output_dir.mkdir(parents=True, exist_ok=True)

files_config = {
    '2016_real': f'dwpc_{BASE_NAME}_2016_real.csv',
    '2016_perm': f'dwpc_{BASE_NAME}_2016_perm_001.csv',
    '2016_random': f'dwpc_{BASE_NAME}_2016_random_001.csv',
    '2024_real': f'dwpc_{BASE_NAME}_2024_real.csv',
    '2024_perm': f'dwpc_{BASE_NAME}_2024_perm_001.csv',
    '2024_random': f'dwpc_{BASE_NAME}_2024_random_001.csv'
}

EXCLUDE_METAPATHS = ['BPpG', 'GpBP']

print(f"Loading {len(files_config)} datasets")
print(f"Output directory: {output_dir}")

# %% papermill={"duration": 135.779422, "end_time": "2025-12-03T15:52:27.118424", "exception": false, "start_time": "2025-12-03T15:50:11.339002", "status": "completed"}
datasets = {}
for label, filename in files_config.items():
    filepath = data_dir / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        df = df.rename(columns={
            'metapath_abbreviation': 'metapath',
            'neo4j_source_id': 'go_id'
        })
        df = df[~df['metapath'].isin(EXCLUDE_METAPATHS)]
        datasets[label] = df
        print(f"Loaded {label}: {len(df):,} rows")
    else:
        print(f"WARNING: {label} file not found")

print("\nAggregating data by GO term and metapath...")

has_pvalue = any('p_value' in df.columns for df in datasets.values())
has_std = any('dgp_nonzero_sd' in df.columns for df in datasets.values())

agg_results = []
for label, df in datasets.items():
    grouped = df.groupby(['go_id', 'metapath'])

    for (go_term, metapath), group in grouped:
        dwpc_all = group['dwpc'].dropna()
        dwpc_nonzero = dwpc_all[dwpc_all > 0]
        if 'p_value' in group.columns:
            pval_all = group['p_value'].dropna()
            pval_nonzero = pval_all[pval_all > 0] if len(pval_all) > 0 else pd.Series(dtype=float)
        else:
            pval_all = pd.Series(dtype=float)
            pval_nonzero = pd.Series(dtype=float)

        if 'dgp_nonzero_sd' in group.columns:
            std_all = group['dgp_nonzero_sd'].dropna()
            std_nonzero = std_all[std_all > 0] if len(std_all) > 0 else pd.Series(dtype=float)
        else:
            std_all = pd.Series(dtype=float)
            std_nonzero = pd.Series(dtype=float)

        agg_results.append({
            'go_id': go_term,
            'metapath': metapath,
            'dataset': label,
            'mean_dwpc': dwpc_all.mean() if len(dwpc_all) > 0 else np.nan,
            'mean_dwpc_nonzero': dwpc_nonzero.mean() if len(dwpc_nonzero) > 0 else np.nan,
            'median_dwpc': dwpc_all.median() if len(dwpc_all) > 0 else np.nan,
            'median_dwpc_nonzero': dwpc_nonzero.median() if len(dwpc_nonzero) > 0 else np.nan,
            'mean_pvalue': pval_all.mean() if len(pval_all) > 0 else np.nan,
            'mean_pvalue_nonzero': pval_nonzero.mean() if len(pval_nonzero) > 0 else np.nan,
            'median_pvalue': pval_all.median() if len(pval_all) > 0 else np.nan,
            'median_pvalue_nonzero': pval_nonzero.median() if len(pval_nonzero) > 0 else np.nan,
            'mean_std': std_all.mean() if len(std_all) > 0 else np.nan,
            'mean_std_nonzero': std_nonzero.mean() if len(std_nonzero) > 0 else np.nan,
            'median_std': std_all.median() if len(std_all) > 0 else np.nan,
            'median_std_nonzero': std_nonzero.median() if len(std_nonzero) > 0 else np.nan,
            'n_total': len(dwpc_all),
            'n_nonzero': len(dwpc_nonzero)
        })

agg_df = pd.DataFrame(agg_results)
agg_df.to_csv(output_dir / 'aggregated_statistics_all_datasets.csv', index=False)
print(f"\nAggregated {len(agg_df):,} (GO term, metapath, dataset) combinations")
print(f"Unique GO terms: {agg_df['go_id'].nunique()}")
print(f"Unique metapaths: {agg_df['metapath'].nunique()}")


# %% papermill={"duration": 0.007154, "end_time": "2025-12-03T15:52:27.128673", "exception": false, "start_time": "2025-12-03T15:52:27.121519", "status": "completed"}
def perform_paired_comparison(agg_df, stat_col, dataset1, dataset2, comparison_name, min_n=10):
    """
    Perform Wilcoxon signed-rank test for one statistic comparing two datasets.
    
    This test is robust to outliers and tests whether the median difference is zero.

    Parameters
    ----------
    agg_df : DataFrame
        Aggregated data with GO term-level statistics
    stat_col : str
        Column name of statistic to test (e.g., 'mean_dwpc')
    dataset1, dataset2 : str
        Dataset labels to compare
    comparison_name : str
        Name for this comparison
    min_n : int
        Minimum number of paired GO terms required

    Returns
    -------
    list of dict
        Test results per metapath
    """
    results = []
    metapaths = sorted(agg_df['metapath'].unique())

    for metapath in metapaths:
        df1 = agg_df[(agg_df['dataset'] == dataset1) &
                     (agg_df['metapath'] == metapath)][['go_id', stat_col]]
        df2 = agg_df[(agg_df['dataset'] == dataset2) &
                     (agg_df['metapath'] == metapath)][['go_id', stat_col]]

        paired = df1.merge(df2, on='go_id', suffixes=('_1', '_2'))
        paired = paired.dropna()

        if len(paired) < min_n:
            continue

        vals1 = paired[f'{stat_col}_1'].values
        vals2 = paired[f'{stat_col}_2'].values
        
        diff = vals1 - vals2
        
        try:
            wilcoxon_stat, p_val = stats.wilcoxon(vals1, vals2, alternative='two-sided')
        except ValueError:
            continue
        
        n_pos = (diff > 0).sum()
        n_neg = (diff < 0).sum()
        rank_biserial = (n_pos - n_neg) / len(diff) if len(diff) > 0 else 0

        results.append({
            'metapath': metapath,
            'statistic': stat_col,
            'comparison': comparison_name,
            'n_pairs': len(paired),
            'mean_dataset1': vals1.mean(),
            'mean_dataset2': vals2.mean(),
            'median_dataset1': np.median(vals1),
            'median_dataset2': np.median(vals2),
            'mean_diff': diff.mean(),
            'median_diff': np.median(diff),
            'wilcoxon_statistic': wilcoxon_stat,
            'p_value': p_val,
            'rank_biserial': rank_biserial
        })

    return results

print("Wilcoxon signed-rank test function defined")

# %% papermill={"duration": 111.787166, "end_time": "2025-12-03T15:54:18.918813", "exception": false, "start_time": "2025-12-03T15:52:27.131647", "status": "completed"}
comparisons_config = [
    ('2016_real', '2024_real', 'real_2016_vs_2024'),
    ('2016_real', '2016_perm', '2016_real_vs_perm'),
    ('2016_real', '2016_random', '2016_real_vs_random'),
    ('2024_real', '2024_perm', '2024_real_vs_perm'),
    ('2024_real', '2024_random', '2024_real_vs_random')
]

statistics = [
    'mean_dwpc', 'mean_dwpc_nonzero',
    'median_dwpc', 'median_dwpc_nonzero',
]
if has_pvalue:
    statistics += [
        'mean_pvalue', 'mean_pvalue_nonzero',
        'median_pvalue', 'median_pvalue_nonzero',
    ]
if has_std:
    statistics += [
        'mean_std', 'mean_std_nonzero',
        'median_std', 'median_std_nonzero',
    ]

print("Running pairwise comparisons using Wilcoxon signed-rank test...")
print(f"  {len(statistics)} statistics × {len(comparisons_config)} comparisons")
print(f"  Expected ~{len(statistics) * len(comparisons_config) * 50} total tests")
print(f"  Note: Wilcoxon test is robust to outliers and tests median differences\n")

all_results = []
for stat in statistics:
    print(f"Testing: {stat}")
    for ds1, ds2, comp_name in comparisons_config:
        if ds1 in datasets and ds2 in datasets:
            results = perform_paired_comparison(agg_df, stat, ds1, ds2, comp_name)
            all_results.extend(results)
            print(f"  {comp_name}: {len(results)} metapaths tested")

results_df = pd.DataFrame(all_results)
print(f"\nCompleted {len(results_df):,} Wilcoxon signed-rank tests")

# %% papermill={"duration": 0.037356, "end_time": "2025-12-03T15:54:18.960658", "exception": false, "start_time": "2025-12-03T15:54:18.923302", "status": "completed"}
print("Applying global FDR correction...")
all_pvals = results_df['p_value'].dropna().values
if len(all_pvals) > 0:
    _, p_fdr_global, _, _ = multipletests(all_pvals, method='fdr_bh')
    results_df.loc[results_df['p_value'].notna(), 'p_fdr_global'] = p_fdr_global
else:
    results_df['p_fdr_global'] = np.nan

for stat in statistics:
    stat_mask = results_df['statistic'] == stat
    stat_pvals = results_df.loc[stat_mask, 'p_value'].dropna()
    
    if len(stat_pvals) > 0:
        _, p_fdr_stat, _, _ = multipletests(stat_pvals.values, method='fdr_bh')
        results_df.loc[stat_mask & results_df['p_value'].notna(), 'p_fdr_per_statistic'] = p_fdr_stat
    else:
        results_df.loc[stat_mask, 'p_fdr_per_statistic'] = np.nan

results_df.to_csv(output_dir / 'pairwise_tests_fdr_corrected.csv', index=False)
print(f"Saved: pairwise_tests_fdr_corrected.csv")

n_sig_global = (results_df['p_fdr_global'] < 0.05).sum()
n_sig_perstat = (results_df['p_fdr_per_statistic'] < 0.05).sum()
print(f"\nGlobal FDR < 0.05: {n_sig_global}/{len(results_df)} tests")
print(f"Per-statistic FDR < 0.05: {n_sig_perstat}/{len(results_df)} tests")

# %% papermill={"duration": 0.428584, "end_time": "2025-12-03T15:54:19.393334", "exception": false, "start_time": "2025-12-03T15:54:18.964750", "status": "completed"}
print("Filtering metapaths meeting all criteria...")

candidate_results = []
for stat in statistics:
    for metapath in results_df['metapath'].unique():
        subset = results_df[(results_df['statistic'] == stat) &
                           (results_df['metapath'] == metapath)]

        temporal = subset[subset['comparison'] == 'real_2016_vs_2024']
        perm_2016 = subset[subset['comparison'] == '2016_real_vs_perm']
        random_2016 = subset[subset['comparison'] == '2016_real_vs_random']
        perm_2024 = subset[subset['comparison'] == '2024_real_vs_perm']
        random_2024 = subset[subset['comparison'] == '2024_real_vs_random']

        if len(temporal) == 0 or len(perm_2016) == 0 or len(random_2016) == 0:
            continue
        if len(perm_2024) == 0 or len(random_2024) == 0:
            continue

        p_temporal = temporal.iloc[0]['p_fdr_per_statistic']
        p_2016_perm = perm_2016.iloc[0]['p_fdr_per_statistic']
        p_2016_rand = random_2016.iloc[0]['p_fdr_per_statistic']
        p_2024_perm = perm_2024.iloc[0]['p_fdr_per_statistic']
        p_2024_rand = random_2024.iloc[0]['p_fdr_per_statistic']

        criterion_1 = p_temporal >= 0.05
        criterion_2 = (p_2016_perm < 0.05) or (p_2016_rand < 0.05)
        criterion_3 = (p_2024_perm < 0.05) or (p_2024_rand < 0.05)

        meets_all = criterion_1 and criterion_2 and criterion_3

        candidate_results.append({
            'statistic': stat,
            'metapath': metapath,
            'meets_all_criteria': meets_all,
            'criterion_1_stable': criterion_1,
            'criterion_2_biology_2016': criterion_2,
            'criterion_3_biology_2024': criterion_3,
            'p_2016_vs_2024': p_temporal,
            'p_2016_vs_perm': p_2016_perm,
            'p_2016_vs_random': p_2016_rand,
            'p_2024_vs_perm': p_2024_perm,
            'p_2024_vs_random': p_2024_rand
        })

candidates_df = pd.DataFrame(candidate_results)
candidates_df.to_csv(output_dir / 'candidate_metapaths_all_statistics.csv', index=False)

passing_df = candidates_df[candidates_df['meets_all_criteria']].copy()
passing_df.to_csv(output_dir / 'metapaths_meeting_all_criteria.csv', index=False)

print(f"\nTotal (statistic, metapath) combinations tested: {len(candidates_df)}")
print(f"Meeting ALL 3 criteria: {len(passing_df)}")

# %% papermill={"duration": 0.008097, "end_time": "2025-12-03T15:54:19.403731", "exception": false, "start_time": "2025-12-03T15:54:19.395634", "status": "completed"}
print("="*80)
print("SUMMARY: Metapaths Meeting All Criteria by Statistic")
print("="*80)

summary_by_stat = passing_df.groupby('statistic').size().sort_values(ascending=False)
print("\nNumber of metapaths passing for each statistic:")
print(summary_by_stat.to_string())

if len(summary_by_stat) > 0:
    best_stat = summary_by_stat.index[0]
    print(f"\nBest statistic: {best_stat} ({summary_by_stat.iloc[0]} metapaths)")

    best_metapaths = passing_df[passing_df['statistic'] == best_stat]['metapath'].tolist()
    print(f"\nMetapaths identified by {best_stat}:")
    for mp in sorted(best_metapaths):
        print(f"  - {mp}")
else:
    print("\nNo metapaths meet all criteria for any statistic.")

# %% papermill={"duration": 0.012072, "end_time": "2025-12-03T15:54:19.419937", "exception": false, "start_time": "2025-12-03T15:54:19.407865", "status": "completed"}
if len(passing_df) > 0:
    print("\n" + "="*80)
    print("DETAILED RESULTS: Metapaths Meeting All Criteria")
    print("="*80)

    for stat in passing_df['statistic'].unique():
        stat_pass = passing_df[passing_df['statistic'] == stat]
        print(f"\n\n{stat.upper()} ({len(stat_pass)} metapaths)")
        print("-" * 80)

        for _, row in stat_pass.iterrows():
            print(f"\n{row['metapath']}:")
            print(f"  2016 vs 2024:    p = {row['p_2016_vs_2024']:.4f}  [NOT significant - STABLE]")
            print(f"  2016 vs perm:    p = {row['p_2016_vs_perm']:.4f}")
            print(f"  2016 vs random:  p = {row['p_2016_vs_random']:.4f}")
            print(f"  2024 vs perm:    p = {row['p_2024_vs_perm']:.4f}")
            print(f"  2024 vs random:  p = {row['p_2024_vs_random']:.4f}")

# %% papermill={"duration": 0.011212, "end_time": "2025-12-03T15:54:19.435152", "exception": false, "start_time": "2025-12-03T15:54:19.423940", "status": "completed"}
report_lines = []
report_lines.append("PAIRWISE STATISTICAL COMPARISONS REPORT")
report_lines.append("="*80)
report_lines.append(f"Date: {pd.Timestamp.now()}")
report_lines.append(f"\nObjective: Identify metapaths that are:")
report_lines.append("  1. Stable across years (2016 vs 2024 NOT significant)")
report_lines.append("  2. Different from permuted controls")
report_lines.append("  3. Different from random controls")
report_lines.append(f"\nTotal tests performed: {len(results_df):,}")
report_lines.append(f"Statistics tested: {len(statistics)}")
report_lines.append(f"Comparisons per statistic: {len(comparisons_config)}")
report_lines.append(f"Unique metapaths: {results_df['metapath'].nunique()}")
report_lines.append(f"\nGlobal FDR correction applied across all {len(results_df)} tests")
report_lines.append(f"\n{len(passing_df)} (statistic, metapath) combinations meet all criteria")

if len(summary_by_stat) > 0:
    report_lines.append(f"\n\nBest identifying statistic: {best_stat}")
    report_lines.append(f"Metapaths identified: {', '.join(sorted(best_metapaths))}")

report_text = "\n".join(report_lines)
with open(output_dir / 'pairwise_comparison_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print(f"\n\nReport saved: {output_dir / 'pairwise_comparison_report.txt'}")

# %% papermill={"duration": 0.107215, "end_time": "2025-12-03T15:54:19.546524", "exception": false, "start_time": "2025-12-03T15:54:19.439309", "status": "completed"}
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

print("\n1. Verifying paired structure...")
for comp in comparisons_config:
    ds1, ds2, comp_name = comp
    if ds1 in datasets and ds2 in datasets:
        go1 = set(agg_df[agg_df['dataset'] == ds1]['go_id'].unique())
        go2 = set(agg_df[agg_df['dataset'] == ds2]['go_id'].unique())
        overlap = go1 & go2
        print(f"  {comp_name}: {len(overlap)} shared GO terms")

print("\n2. Sample size distribution...")
print(f"  Minimum n_pairs: {results_df['n_pairs'].min()}")
print(f"  Median n_pairs: {results_df['n_pairs'].median():.0f}")
print(f"  Maximum n_pairs: {results_df['n_pairs'].max()}")

print("\n3. FDR correction validation...")
raw_sig = (results_df['p_value'] < 0.05).sum()
fdr_sig = (results_df['p_fdr_global'] < 0.05).sum()
print(f"  Raw p < 0.05: {raw_sig}/{len(results_df)}")
print(f"  FDR p < 0.05: {fdr_sig}/{len(results_df)}")
print(f"  FDR reduced false positives by {raw_sig - fdr_sig}")

print("\nValidation complete.")

# %% papermill={"duration": 0.014217, "end_time": "2025-12-03T15:54:19.563334", "exception": false, "start_time": "2025-12-03T15:54:19.549117", "status": "completed"}
summary_stats = {
    'Total tests performed': len(results_df),
    'Unique metapaths tested': results_df['metapath'].nunique(),
    'Statistics tested': len(statistics),
    'Comparisons per statistic': len(comparisons_config),
    'Tests significant (raw p < 0.05)': (results_df['p_value'] < 0.05).sum(),
    'Tests significant (FDR < 0.05)': (results_df['p_fdr_global'] < 0.05).sum(),
    'Metapath-statistic combos meeting all criteria': len(passing_df),
    'Unique metapaths meeting criteria': passing_df['metapath'].nunique() if len(passing_df) > 0 else 0
}

summary_df = pd.DataFrame([summary_stats]).T
summary_df.columns = ['Value']
summary_df.to_csv(output_dir / 'summary_statistics.csv')

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(summary_df.to_string())

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
print("  - aggregated_statistics_all_datasets.csv")
print("  - pairwise_tests_fdr_corrected.csv")
print("  - candidate_metapaths_all_statistics.csv")
print("  - metapaths_meeting_all_criteria.csv")
print("  - pairwise_comparison_report.txt")
print("  - summary_statistics.csv")
