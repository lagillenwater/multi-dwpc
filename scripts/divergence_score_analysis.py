# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.004061, "end_time": "2025-12-17T19:09:11.141196", "exception": false, "start_time": "2025-12-17T19:09:11.137135", "status": "completed"}
# # 4. Divergence Score Analysis
#
# ## Research Question
#
# Which metapaths show the **strongest biological signal** in both 2016 and 2024, as measured by
# divergence from control distributions?
#
# ## Approach
#
# Compute effect sizes (Cohen's d) and statistical significance for each metapath comparing real data
# to control distributions (permuted and random).
#
# ### Statistics Analyzed
# - **DWPC**: mean, mean_nonzero, median, median_nonzero
# - **-log10(p-value)**: mean, mean_nonzero, median, median_nonzero
#
# All statistics are oriented so that **higher values = stronger signal**.
#
# ### Methodology
#
# 1. **Zero Fraction Analysis**: Compare sparsity between real and control data
# 2. **Effect Size (Cohen's d)**: Standardized difference between real and control
# 3. **Statistical Significance**: Paired t-test (p < 0.05)
# 4. **Combined Analysis**: Identify metapaths significant in **both** 2016 and 2024

# %% papermill={"duration": 9.231772, "end_time": "2025-12-17T19:09:20.379936", "exception": false, "start_time": "2025-12-17T19:09:11.148164", "status": "completed"}
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

import warnings
warnings.filterwarnings('ignore')

print("Notebook 4: Directional Divergence Score Analysis")

# %% [markdown]
# ## CLI Arguments
#
# Use --plots-only to regenerate figures from divergence_all_statistics.csv

parser = argparse.ArgumentParser(description="Divergence score analysis")
parser.add_argument(
    "--plots-only",
    action="store_true",
    help="Skip analysis and regenerate plots from divergence_all_statistics.csv",
)
args = parser.parse_args()

def run_additional_visualizations(div_df, statistics, output_dir, repo_root, base_name):
    print("\n" + "="*80)
    print("ADDITIONAL VISUALIZATIONS")
    print("="*80)

    primary_stat = 'mean_dwpc_nonzero' if 'mean_dwpc_nonzero' in statistics else 'mean_dwpc'
    primary_df = div_df[div_df['statistic'] == primary_stat].copy()

    def save_figure(fig, filename):
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 1) Metapath fingerprint heatmap
    try:
        heat_df = div_df.pivot(index='metapath', columns='statistic', values='combined_effect')
        heat_df = heat_df.loc[heat_df.max(axis=1).sort_values(ascending=False).index]
        sig_df = div_df.pivot(index='metapath', columns='statistic', values='both_significant')
        sig_df = sig_df.reindex(index=heat_df.index, columns=heat_df.columns)

        fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(heat_df))))
        sns.heatmap(
            heat_df,
            ax=ax,
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Combined effect (min of years)'},
            linewidths=0.2,
            linecolor='white'
        )
        for i in range(sig_df.shape[0]):
            for j in range(sig_df.shape[1]):
                if bool(sig_df.iloc[i, j]):
                    ax.scatter(j + 0.5, i + 0.5, s=10, c='black')
        ax.set_title(f'Metapath fingerprint heatmap ({primary_stat})')
        save_figure(fig, 'metapath_effect_heatmap.png')
        print("Saved: metapath_effect_heatmap.png")
    except Exception as exc:
        print(f"Heatmap skipped: {exc}")

    # 3) Quadrant plot with density
    try:
        n_stats = len(statistics)
        n_cols = 2
        n_rows = int(np.ceil(n_stats / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for idx, stat in enumerate(statistics):
            ax = axes[idx]
            stat_df = div_df[div_df['statistic'] == stat]
            if stat_df.empty:
                ax.axis('off')
                continue
            ax.hexbin(
                stat_df['2016_effect_size'],
                stat_df['2024_effect_size'],
                gridsize=25,
                cmap='Blues',
                mincnt=1,
                alpha=0.8
            )
            ax.scatter(
                stat_df['2016_effect_size'],
                stat_df['2024_effect_size'],
                s=10,
                c='white',
                alpha=0.5,
                edgecolors='none'
            )
            ax.axhline(0, color='gray', linewidth=0.8)
            ax.axvline(0, color='gray', linewidth=0.8)
            lim = max(
                abs(stat_df['2016_effect_size']).max(),
                abs(stat_df['2024_effect_size']).max()
            )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.3)
            ax.set_title(f'{stat}: 2016 vs 2024')
            ax.set_xlabel('2016 effect size')
            ax.set_ylabel('2024 effect size')

            top = stat_df.sort_values('combined_effect', ascending=False).head(5)
            for _, row in top.iterrows():
                ax.annotate(
                    row['metapath'],
                    (row['2016_effect_size'], row['2024_effect_size']),
                    fontsize=6,
                    alpha=0.8
                )

        for idx in range(n_stats, len(axes)):
            axes[idx].axis('off')

        save_figure(fig, 'quadrant_density_plots.png')
        print("Saved: quadrant_density_plots.png")
    except Exception as exc:
        print(f"Quadrant density plot skipped: {exc}")

    # 4) Metapath network map
    def parse_metanodes(metapath):
        node_abbrevs = ["BP", "CC", "MF", "PW", "SE", "PC", "G", "A", "D", "C", "S"]
        node_abbrevs = sorted(node_abbrevs, key=len, reverse=True)
        nodes = []
        i = 0
        while i < len(metapath):
            matched = False
            for ab in node_abbrevs:
                if metapath.startswith(ab, i):
                    nodes.append(ab)
                    i += len(ab)
                    matched = True
                    break
            if not matched:
                i += 1
        return nodes

    try:
        edge_stats = {}
        for _, row in primary_df.iterrows():
            nodes = parse_metanodes(row['metapath'])
            for u, v in zip(nodes, nodes[1:]):
                key = tuple(sorted((u, v)))
                acc = edge_stats.get(key, [0.0, 0])
                acc[0] += row['combined_effect']
                acc[1] += 1
                edge_stats[key] = acc

        if edge_stats:
            edges = []
            for (u, v), (s, n) in edge_stats.items():
                avg = s / n if n else 0.0
                edges.append((u, v, avg))

            nodes = sorted({n for e in edges for n in e[:2]})
            base_order = ["G", "BP", "PW", "MF", "CC", "A", "D", "C", "SE", "S", "PC"]
            nodes = [n for n in base_order if n in nodes] + [n for n in nodes if n not in base_order]

            positions = {}
            positions["G"] = (-1.2, 0.0)
            positions["BP"] = (1.2, 0.0)
            others = [n for n in nodes if n not in ("G", "BP")]
            angles = np.linspace(0, 2 * np.pi, len(others), endpoint=False)
            for n, ang in zip(others, angles):
                positions[n] = (0.95 * np.cos(ang), 0.95 * np.sin(ang))

            max_abs = max(abs(e[2]) for e in edges) if edges else 1.0
            fig, ax = plt.subplots(figsize=(8, 8))
            for u, v, avg in edges:
                x1, y1 = positions[u]
                x2, y2 = positions[v]
                width = 0.5 + 3.5 * (abs(avg) / max_abs)
                color = 'green' if avg >= 0 else 'red'
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.6)

            for n in nodes:
                x, y = positions[n]
                ax.scatter([x], [y], s=600, c='white', edgecolors='black', zorder=3)
                ax.text(x, y, n, ha='center', va='center', fontsize=10, zorder=4)

            ax.set_title(f'Metapath network map ({primary_stat})')
            ax.axis('off')
            save_figure(fig, 'metapath_network_map.png')
            print("Saved: metapath_network_map.png")
        else:
            print("Metapath network map skipped: no edges found.")
    except Exception as exc:
        print(f"Metapath network map skipped: {exc}")

    # 5) Rank stability plot
    try:
        rank_df = primary_df[['metapath', '2016_effect_size', '2024_effect_size', 'combined_effect']].copy()
        rank_df['rank_2016'] = rank_df['2016_effect_size'].rank(ascending=False, method='average')
        rank_df['rank_2024'] = rank_df['2024_effect_size'].rank(ascending=False, method='average')
        n = len(rank_df)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(rank_df['rank_2016'], rank_df['rank_2024'], s=30, alpha=0.7)
        ax.plot([1, n], [1, n], 'k--', alpha=0.4)
        ax.set_xlabel('2016 rank (higher effect size = rank 1)')
        ax.set_ylabel('2024 rank (higher effect size = rank 1)')
        ax.set_title(f'Rank stability: {primary_stat}')

        rank_df['rank_diff'] = (rank_df['rank_2016'] - rank_df['rank_2024']).abs()
        top_labels = rank_df.sort_values('rank_diff', ascending=False).head(8)
        for _, row in top_labels.iterrows():
            ax.annotate(row['metapath'], (row['rank_2016'], row['rank_2024']), fontsize=6, alpha=0.8)

        save_figure(fig, 'rank_stability.png')
        print("Saved: rank_stability.png")
    except Exception as exc:
        print(f"Rank stability plot skipped: {exc}")

    # 7) Ridge plots for top metapaths
    try:
        base_dir = repo_root / 'output' / 'dwpc_direct' / base_name / 'results'
        dataset_files = {
            '2016_real': base_dir / f'dwpc_{base_name}_2016_real.csv',
            '2016_perm': base_dir / f'dwpc_{base_name}_2016_perm_001.csv',
            '2016_random': base_dir / f'dwpc_{base_name}_2016_random_001.csv',
            '2024_real': base_dir / f'dwpc_{base_name}_2024_real.csv',
            '2024_perm': base_dir / f'dwpc_{base_name}_2024_perm_001.csv',
            '2024_random': base_dir / f'dwpc_{base_name}_2024_random_001.csv',
        }

        if all(p.exists() for p in dataset_files.values()):
            top_metapaths = primary_df.sort_values('combined_effect', ascending=False)['metapath'].head(5).tolist()

            def collect_dwpc_values(path, metapaths, chunksize=200000):
                store = {mp: [] for mp in metapaths}
                for chunk in pd.read_csv(path, usecols=['metapath', 'dwpc'], chunksize=chunksize):
                    chunk = chunk[chunk['metapath'].isin(metapaths)]
                    if chunk.empty:
                        continue
                    for mp, group in chunk.groupby('metapath'):
                        store[mp].append(group['dwpc'].values)
                return {mp: (np.concatenate(vals) if vals else np.array([])) for mp, vals in store.items()}

            values = {label: collect_dwpc_values(path, top_metapaths) for label, path in dataset_files.items()}

            fig, axes = plt.subplots(len(top_metapaths), 2, figsize=(12, 2.2 * len(top_metapaths)), sharex='col')
            if len(top_metapaths) == 1:
                axes = np.array([axes])

            colors = {'real': '#1f77b4', 'perm': '#ff7f0e', 'random': '#2ca02c'}
            offset_step = 1.2
            max_points = 20000

            for i, metapath in enumerate(top_metapaths):
                for j, year in enumerate([2016, 2024]):
                    ax = axes[i, j]
                    year_vals = []
                    for kind in ['real', 'perm', 'random']:
                        label = f'{year}_{kind}'
                        vals = values[label].get(metapath, np.array([]))
                        if len(vals) == 0:
                            continue
                        if len(vals) > max_points:
                            vals = np.random.choice(vals, max_points, replace=False)
                        year_vals.append(vals)
                    if not year_vals:
                        ax.axis('off')
                        continue

                    x_min = min(v.min() for v in year_vals)
                    x_max = max(v.max() for v in year_vals)
                    xs = np.linspace(x_min, x_max, 200)

                    for k, kind in enumerate(['real', 'perm', 'random']):
                        label = f'{year}_{kind}'
                        vals = values[label].get(metapath, np.array([]))
                        if len(vals) < 5:
                            continue
                        if len(vals) > max_points:
                            vals = np.random.choice(vals, max_points, replace=False)
                        kde = stats.gaussian_kde(vals)
                        ys = kde(xs)
                        ys = ys / (ys.max() if ys.max() > 0 else 1.0)
                        offset = k * offset_step
                        ax.fill_between(xs, offset, ys + offset, alpha=0.4, color=colors[kind])
                        ax.plot(xs, ys + offset, color=colors[kind], linewidth=1)

                    ax.set_yticks([0, offset_step, 2 * offset_step])
                    ax.set_yticklabels(['real', 'perm', 'random'])
                    if j == 0:
                        ax.set_ylabel(metapath, fontsize=8)
                    if i == 0:
                        ax.set_title(f'{year}')
                    if i == len(top_metapaths) - 1:
                        ax.set_xlabel('DWPC')

            save_figure(fig, 'ridge_top_metapaths.png')
            print("Saved: ridge_top_metapaths.png")
        else:
            print("Ridge plots skipped: missing direct DWPC outputs.")
    except Exception as exc:
        print(f"Ridge plots skipped: {exc}")

    # 8) Consistency band plot
    try:
        diff = primary_df['2016_effect_size'] - primary_df['2024_effect_size']
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(diff, primary_df['combined_effect'], s=30, alpha=0.7)
        ax.axvspan(-0.1, 0.1, color='gray', alpha=0.2, label='consistency band')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('2016 effect - 2024 effect')
        ax.set_ylabel('Combined effect (min of years)')
        ax.set_title(f'Consistency vs strength ({primary_stat})')

        top = primary_df.sort_values('combined_effect', ascending=False).head(8)
        for _, row in top.iterrows():
            ax.annotate(
                row['metapath'],
                (row['2016_effect_size'] - row['2024_effect_size'], row['combined_effect']),
                fontsize=6,
                alpha=0.8
            )

        save_figure(fig, 'consistency_band_plot.png')
        print("Saved: consistency_band_plot.png")
    except Exception as exc:
        print(f"Consistency band plot skipped: {exc}")

# %% papermill={"duration": 0.010338, "end_time": "2025-12-17T19:09:20.392297", "exception": false, "start_time": "2025-12-17T19:09:20.381959", "status": "completed"}
# Handle path resolution whether running from repo root or notebooks directory
import os

cwd = Path.cwd()
if cwd.name == 'notebooks':
    repo_root = cwd.parent
else:
    repo_root = cwd

BASE_NAME = 'all_GO_positive_growth'
data_dir = repo_root / 'output' / 'dwpc_direct' / BASE_NAME / 'results'
output_dir = repo_root / 'output' / 'metapath_analysis' / 'divergence_scores'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Working directory: {cwd}")
print(f"Data directory: {data_dir}")
print(f"Data exists: {data_dir.exists()}")

files_config = {
    '2016_real': f'dwpc_{BASE_NAME}_2016_real.csv',
    '2016_perm': f'dwpc_{BASE_NAME}_2016_perm_001.csv',
    '2016_random': f'dwpc_{BASE_NAME}_2016_random_001.csv',
    '2024_real': f'dwpc_{BASE_NAME}_2024_real.csv',
    '2024_perm': f'dwpc_{BASE_NAME}_2024_perm_001.csv',
    '2024_random': f'dwpc_{BASE_NAME}_2024_random_001.csv'
}

if args.plots_only:
    stats_path = output_dir / 'divergence_all_statistics.csv'
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run full analysis first."
        )
    div_df = pd.read_csv(stats_path)
    STATISTICS = sorted(div_df['statistic'].dropna().unique().tolist())
    run_additional_visualizations(div_df, STATISTICS, output_dir, repo_root, BASE_NAME)
    sys.exit(0)

EXCLUDE_METAPATHS = ['BPpG', 'GpBP']

print(f"Output directory: {output_dir}")

# %% papermill={"duration": 4.785854, "end_time": "2025-12-17T19:09:25.184702", "exception": false, "start_time": "2025-12-17T19:09:20.398848", "status": "completed"}
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

# %% [markdown] papermill={"duration": 0.003252, "end_time": "2025-12-17T19:09:25.189565", "exception": false, "start_time": "2025-12-17T19:09:25.186313", "status": "completed"}
# ## Zero Value Analysis
#
# Understanding the role of zeros in DWPC data:
# - **Sparse metapaths**: Many gene pairs have no paths (DWPC = 0)
# - **Zero fraction signal**: Real data may have systematically fewer zeros than controls
# - This analysis shows whether the zero/nonzero pattern contributes to biological signal

# %% papermill={"duration": 10.050573, "end_time": "2025-12-17T19:09:35.242808", "exception": false, "start_time": "2025-12-17T19:09:25.192235", "status": "completed"}
print("Analyzing zero fractions across datasets and metapaths...\n")

# Compute zero fractions for each metapath and dataset
zero_results = []
for label, df in datasets.items():
    for metapath in df['metapath'].unique():
        mp_dwpc = df[df['metapath'] == metapath]['dwpc']
        n_total = len(mp_dwpc)
        n_zeros = (mp_dwpc == 0).sum()
        zero_frac = n_zeros / n_total if n_total > 0 else np.nan
        
        zero_results.append({
            'metapath': metapath,
            'dataset': label,
            'n_total': n_total,
            'n_zeros': n_zeros,
            'zero_fraction': zero_frac
        })

zero_df = pd.DataFrame(zero_results)

# Pivot to compare real vs control
zero_pivot = zero_df.pivot_table(
    index='metapath', 
    columns='dataset', 
    values='zero_fraction'
).reset_index()

# Calculate differences (control - real): positive means real has fewer zeros
zero_pivot['2016_perm_diff'] = zero_pivot['2016_perm'] - zero_pivot['2016_real']
zero_pivot['2016_random_diff'] = zero_pivot['2016_random'] - zero_pivot['2016_real']
zero_pivot['2024_perm_diff'] = zero_pivot['2024_perm'] - zero_pivot['2024_real']
zero_pivot['2024_random_diff'] = zero_pivot['2024_random'] - zero_pivot['2024_real']

# Average difference per year
zero_pivot['2016_zero_diff'] = (zero_pivot['2016_perm_diff'] + zero_pivot['2016_random_diff']) / 2
zero_pivot['2024_zero_diff'] = (zero_pivot['2024_perm_diff'] + zero_pivot['2024_random_diff']) / 2

print("Zero Fraction Analysis (positive diff = real has fewer zeros than control):")
print("="*80)

display_cols = ['metapath', '2016_real', '2016_perm', '2016_zero_diff', '2024_real', '2024_perm', '2024_zero_diff']
zero_summary = zero_pivot[display_cols].sort_values('2016_zero_diff', ascending=False)
zero_summary.columns = ['Metapath', '2016 Real', '2016 Perm', '2016 Diff', '2024 Real', '2024 Perm', '2024 Diff']

print(zero_summary.to_string(index=False, float_format='%.3f'))

# %% papermill={"duration": 1.12904, "end_time": "2025-12-17T19:09:36.374752", "exception": false, "start_time": "2025-12-17T19:09:35.245712", "status": "completed"}
# Visualize zero fraction differences
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Zero fraction in real vs control (2016)
ax1 = axes[0]
ax1.scatter(zero_pivot['2016_real'], zero_pivot['2016_perm'], 
            c='steelblue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
ax1.set_xlabel('Zero Fraction (2016 Real)')
ax1.set_ylabel('Zero Fraction (2016 Permuted)')
ax1.set_title('2016: Real vs Permuted Zero Fractions\n(points above line = real has fewer zeros)')
ax1.legend()

# Label metapaths with large differences
for _, row in zero_pivot.nlargest(5, '2016_zero_diff').iterrows():
    ax1.annotate(row['metapath'], (row['2016_real'], row['2016_perm']), fontsize=7)

# Plot 2: Zero difference 2016 vs 2024
ax2 = axes[1]
colors = ['green' if (x > 0 and y > 0) else 'gray' for x, y in 
          zip(zero_pivot['2016_zero_diff'], zero_pivot['2024_zero_diff'])]
ax2.scatter(zero_pivot['2016_zero_diff'], zero_pivot['2024_zero_diff'],
            c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax2.set_xlabel('2016 Zero Diff (control - real)')
ax2.set_ylabel('2024 Zero Diff (control - real)')
ax2.set_title('Consistency of Zero Fraction Signal\n(green = real has fewer zeros in both years)')

# Label top metapaths
for _, row in zero_pivot.nlargest(5, '2016_zero_diff').iterrows():
    ax2.annotate(row['metapath'], (row['2016_zero_diff'], row['2024_zero_diff']), fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / 'zero_fraction_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved: zero_fraction_analysis.png")

# Summary statistics
n_fewer_zeros_both = ((zero_pivot['2016_zero_diff'] > 0.01) & (zero_pivot['2024_zero_diff'] > 0.01)).sum()
print(f"\nMetapaths where real has >1% fewer zeros than control in BOTH years: {n_fewer_zeros_both}")

# %% papermill={"duration": 0.036044, "end_time": "2025-12-17T19:09:36.416910", "exception": false, "start_time": "2025-12-17T19:09:36.380866", "status": "completed"}
# Save zero fraction analysis
zero_pivot.to_csv(output_dir / 'zero_fraction_analysis.csv', index=False)
print(f"Saved: zero_fraction_analysis.csv")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION: Zero Fractions and Effect Sizes")
print("="*60)
print("""
When comparing mean_dwpc vs mean_dwpc_nonzero effect sizes:

- If d(all) > d(nonzero): The zero/nonzero pattern carries signal.
  Real data has fewer zeros, so including zeros captures this.
  
- If d(all) < d(nonzero): Nonzero values alone are more discriminating.
  Zeros add noise to the comparison.
  
- If they're equal: Zeros are rare or similar between real/control.

For sparse metapaths (high zero fraction), the difference between
'all' and 'nonzero' statistics can be substantial.
""")

# %% papermill={"duration": 174.686949, "end_time": "2025-12-17T19:12:31.110201", "exception": false, "start_time": "2025-12-17T19:09:36.423252", "status": "completed"}
print("Aggregating data by GO term and metapath...")

has_pvalue = any('p_value' in df.columns for df in datasets.values())

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

        # Convert p-values to -log10(p) so higher = more significant (same direction as DWPC)
        neglog_pval_all = -np.log10(pval_all.clip(lower=1e-300)) if len(pval_all) > 0 else pd.Series(dtype=float)
        neglog_pval_nonzero = -np.log10(pval_nonzero.clip(lower=1e-300)) if len(pval_nonzero) > 0 else pd.Series(dtype=float)

        agg_results.append({
            'go_id': go_term,
            'metapath': metapath,
            'dataset': label,
            'mean_dwpc': dwpc_all.mean() if len(dwpc_all) > 0 else np.nan,
            'mean_dwpc_nonzero': dwpc_nonzero.mean() if len(dwpc_nonzero) > 0 else np.nan,
            'median_dwpc': dwpc_all.median() if len(dwpc_all) > 0 else np.nan,
            'median_dwpc_nonzero': dwpc_nonzero.median() if len(dwpc_nonzero) > 0 else np.nan,
            'mean_neglog10p': neglog_pval_all.mean() if len(neglog_pval_all) > 0 else np.nan,
            'mean_neglog10p_nonzero': neglog_pval_nonzero.mean() if len(neglog_pval_nonzero) > 0 else np.nan,
            'median_neglog10p': neglog_pval_all.median() if len(neglog_pval_all) > 0 else np.nan,
            'median_neglog10p_nonzero': neglog_pval_nonzero.median() if len(neglog_pval_nonzero) > 0 else np.nan,
            'n_total': len(dwpc_all),
            'n_nonzero': len(dwpc_nonzero)
        })

agg_df = pd.DataFrame(agg_results)
print(f"Aggregated {len(agg_df):,} (GO term, metapath, dataset) combinations")

# Statistics to analyze (excluding std)
STATISTICS = [
    'mean_dwpc', 'mean_dwpc_nonzero',
    'median_dwpc', 'median_dwpc_nonzero',
]
if has_pvalue:
    STATISTICS += [
        'mean_neglog10p', 'mean_neglog10p_nonzero',
        'median_neglog10p', 'median_neglog10p_nonzero'
    ]
print(f"Statistics to analyze: {len(STATISTICS)}")

# Export paired GO-term values (real vs perm/random)
def build_paired_go_term_values(agg_df, year, control_label, stats):
    real_label = f"{year}_real"
    ctrl_label = f"{year}_{control_label}"
    real = agg_df[agg_df['dataset'] == real_label].copy()
    ctrl = agg_df[agg_df['dataset'] == ctrl_label].copy()

    if real.empty or ctrl.empty:
        return pd.DataFrame()

    paired = real.merge(ctrl, on=['go_id', 'metapath'], suffixes=('_real', '_ctrl'))

    for stat in stats:
        paired[f'{stat}_diff'] = paired[f'{stat}_real'] - paired[f'{stat}_ctrl']

    return paired


bp_path = repo_root / "data" / "nodes" / "Biological Process.tsv"
bp_map = None
if bp_path.exists():
    bp_df = pd.read_csv(bp_path, sep="\t")
    bp_map = dict(zip(bp_df["identifier"], bp_df["name"]))

for year, control in [("2016", "perm"), ("2016", "random"), ("2024", "perm"), ("2024", "random")]:
    paired = build_paired_go_term_values(agg_df, year, control, STATISTICS)
    if paired.empty:
        print(f"Skipping paired GO-term export for {year} vs {control}: no data.")
        continue

    if bp_map is not None and "go_name" not in paired.columns:
        paired.insert(1, "go_name", paired["go_id"].map(bp_map))

    base_cols = ["go_id", "metapath"]
    if "go_name" in paired.columns:
        base_cols.insert(1, "go_name")

    stat_cols = []
    for stat in STATISTICS:
        stat_cols.extend([f"{stat}_real", f"{stat}_ctrl", f"{stat}_diff"])

    count_cols = []
    for col in ["n_total", "n_nonzero"]:
        real_col = f"{col}_real"
        ctrl_col = f"{col}_ctrl"
        if real_col in paired.columns and ctrl_col in paired.columns:
            count_cols.extend([real_col, ctrl_col])

    out_cols = base_cols + stat_cols + count_cols
    out_path = output_dir / f"paired_go_terms_{year}_vs_{control}.csv"
    paired[out_cols].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# %% [markdown] papermill={"duration": 0.005895, "end_time": "2025-12-17T19:12:31.121739", "exception": false, "start_time": "2025-12-17T19:12:31.115844", "status": "completed"}
# ## Effect Size and Significance Computation
#
# For each metapath, we compute:
# - **Cohen's d**: Standardized effect size (mean difference / pooled SD)
# - **p-value**: From paired t-test comparing real vs control DWPC values

# %% papermill={"duration": 0.018426, "end_time": "2025-12-17T19:12:31.146437", "exception": false, "start_time": "2025-12-17T19:12:31.128011", "status": "completed"}
def compute_divergence_score(real_values, control_values):
    """
    Compute divergence statistics between real and control distributions.
    
    Parameters
    ----------
    real_values : array-like
        Values from real dataset
    control_values : array-like
        Values from control dataset (paired by GO term)
    
    Returns
    -------
    dict
        Contains effect_size (Cohen's d), t_stat, p_value, mean_diff, n_pairs
    """
    mask = ~(np.isnan(real_values) | np.isnan(control_values))
    real = np.array(real_values)[mask]
    control = np.array(control_values)[mask]
    
    n = len(real)
    if n < 10:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'mean_diff': np.nan,
            'n_pairs': n
        }
    
    # Paired t-test
    diff = real - control
    t_stat, p_value = stats.ttest_rel(real, control)
    
    # Cohen's d for paired samples (signed)
    effect_size = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'mean_diff': diff.mean(),
        'n_pairs': n
    }


print("Divergence score function defined (Cohen's d + paired t-test)")

# %% papermill={"duration": 5.848829, "end_time": "2025-12-17T19:12:37.013545", "exception": false, "start_time": "2025-12-17T19:12:31.164716", "status": "completed"}
print("Computing divergence scores for each metapath and statistic...\n")

comparisons = [
    ('2016_real', '2016_perm', '2016_vs_perm'),
    ('2016_real', '2016_random', '2016_vs_random'),
    ('2024_real', '2024_perm', '2024_vs_perm'),
    ('2024_real', '2024_random', '2024_vs_random'),
]

metapaths = sorted(agg_df['metapath'].unique())
all_divergence_results = []

for stat_col in STATISTICS:
    print(f"Processing: {stat_col}")
    
    for metapath in metapaths:
        mp_data = agg_df[agg_df['metapath'] == metapath]
        
        scores = {'metapath': metapath, 'statistic': stat_col}
        
        for real_ds, ctrl_ds, comp_name in comparisons:
            real = mp_data[mp_data['dataset'] == real_ds][['go_id', stat_col]]
            ctrl = mp_data[mp_data['dataset'] == ctrl_ds][['go_id', stat_col]]

            paired = real.merge(ctrl, on='go_id', suffixes=('_real', '_ctrl'))
            
            if len(paired) < 10:
                continue
                
            result = compute_divergence_score(
                paired[f'{stat_col}_real'].values,
                paired[f'{stat_col}_ctrl'].values
            )
            
            scores[f'{comp_name}_effect_size'] = result['effect_size']
            scores[f'{comp_name}_t_stat'] = result['t_stat']
            scores[f'{comp_name}_p_value'] = result['p_value']
            scores[f'{comp_name}_n_pairs'] = result['n_pairs']
        
        all_divergence_results.append(scores)

div_df = pd.DataFrame(all_divergence_results)
print(f"\nComputed divergence scores for {len(div_df)} (metapath, statistic) combinations")

# %% papermill={"duration": 0.02078, "end_time": "2025-12-17T19:12:37.036960", "exception": false, "start_time": "2025-12-17T19:12:37.016180", "status": "completed"}
print("Computing combined scores per year...\n")

# Mean effect sizes per year (average of perm and random comparisons)
div_df['2016_effect_size'] = div_df[['2016_vs_perm_effect_size', '2016_vs_random_effect_size']].mean(axis=1)
div_df['2024_effect_size'] = div_df[['2024_vs_perm_effect_size', '2024_vs_random_effect_size']].mean(axis=1)

# Combined effect size using min (both years must show effect)
div_df['combined_effect'] = div_df[['2016_effect_size', '2024_effect_size']].min(axis=1)

# Mean p-values per year (using geometric mean for p-values)
div_df['2016_p_value'] = np.sqrt(
    div_df['2016_vs_perm_p_value'] * div_df['2016_vs_random_p_value']
)
div_df['2024_p_value'] = np.sqrt(
    div_df['2024_vs_perm_p_value'] * div_df['2024_vs_random_p_value']
)

# Significance flags (p < 0.05)
div_df['2016_significant'] = div_df['2016_p_value'] < 0.05
div_df['2024_significant'] = div_df['2024_p_value'] < 0.05
div_df['both_significant'] = div_df['2016_significant'] & div_df['2024_significant']

# Direction consistency
div_df['same_direction'] = (div_df['2016_effect_size'] > 0) == (div_df['2024_effect_size'] > 0)

print("Summary by statistic:")
summary = div_df.groupby('statistic').agg({
    'both_significant': 'sum',
    'combined_effect': ['mean', 'max']
}).round(3)
summary.columns = ['n_significant_both', 'mean_effect', 'max_effect']
print(summary)

# %% [markdown] papermill={"duration": 0.009081, "end_time": "2025-12-17T19:12:37.049291", "exception": false, "start_time": "2025-12-17T19:12:37.040210", "status": "completed"}
# ## Interpreting Results
#
# ### Effect Size (Cohen's d)
# - **Positive d**: Real DWPC > control (more connectivity than expected)
# - **Negative d**: Real DWPC < control (less connectivity than expected)
# - **Magnitude**: |d| ~ 0.2 small, 0.5 medium, 0.8 large
#
# ### Statistical Significance (p-value)
# - **p < 0.05**: Statistically significant difference between real and control
# - Combined p-value uses geometric mean of permutation and random control comparisons

# %% papermill={"duration": 0.807377, "end_time": "2025-12-17T19:12:37.867223", "exception": false, "start_time": "2025-12-17T19:12:37.059846", "status": "completed"}
# Create a figure for each statistic showing all metapaths
n_stats = len(STATISTICS)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, stat in enumerate(STATISTICS):
    ax = axes[idx]
    stat_df = div_df[div_df['statistic'] == stat]
    
    scatter = ax.scatter(
        stat_df['2016_effect_size'], 
        stat_df['2024_effect_size'],
        c=stat_df['both_significant'].map({True: 'green', False: 'gray'}),
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.3)
    
    # Label significant metapaths
    sig_stat = stat_df[stat_df['both_significant']]
    for _, row in sig_stat.iterrows():
        ax.annotate(row['metapath'], (row['2016_effect_size'], row['2024_effect_size']),
                   fontsize=6, alpha=0.8)
    
    n_sig = stat_df['both_significant'].sum()
    ax.set_title(f'{stat}\n(n={n_sig} significant)', fontsize=10)
    ax.set_xlabel("2016 d", fontsize=8)
    ax.set_ylabel("2024 d", fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'all_statistics_effect_sizes.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved: all_statistics_effect_sizes.png")

# %% papermill={"duration": 0.984783, "end_time": "2025-12-17T19:12:38.854722", "exception": false, "start_time": "2025-12-17T19:12:37.869939", "status": "completed"}
# Bar chart showing significant metapaths for each statistic
sig_df = div_df[div_df['both_significant']].copy()
print(f"Total (metapath, statistic) combinations significant in both years: {len(sig_df)}")

if len(sig_df) > 0:
    # Group by statistic and show bar chart for each
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, stat in enumerate(STATISTICS):
        ax = axes[idx]
        stat_sig = sig_df[sig_df['statistic'] == stat].sort_values('combined_effect', ascending=True)
        
        if len(stat_sig) > 0:
            colors = ['green' if x > 0 else 'red' for x in stat_sig['combined_effect']]
            ax.barh(range(len(stat_sig)), stat_sig['combined_effect'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(stat_sig)))
            ax.set_yticklabels(stat_sig['metapath'], fontsize=8)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title(f'{stat}\n({len(stat_sig)} significant)', fontsize=10)
        ax.set_xlabel("Combined Effect Size (d)", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'significant_metapaths_by_statistic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSaved: significant_metapaths_by_statistic.png")
else:
    print("No metapaths are significant in both years for any statistic.")

# %% papermill={"duration": 0.017213, "end_time": "2025-12-17T19:12:38.883608", "exception": false, "start_time": "2025-12-17T19:12:38.866395", "status": "completed"}
print("="*80)
print("ALL RESULTS BY STATISTIC AND METAPATH")
print("="*80)

display_cols = [
    'statistic',
    'metapath',
    '2016_effect_size',
    '2024_effect_size', 
    'combined_effect',
    '2016_p_value',
    '2024_p_value',
    'both_significant'
]

all_results = div_df.sort_values(['statistic', 'combined_effect'], ascending=[True, False])[display_cols].copy()
all_results.columns = ['Statistic', 'Metapath', '2016 d', '2024 d', 'Combined d', '2016 p', '2024 p', 'Sig Both']

# Show top 5 per statistic
for stat in STATISTICS:
    print(f"\n{stat}:")
    print("-" * 60)
    stat_results = all_results[all_results['Statistic'] == stat].head(10)
    print(stat_results.to_string(index=False, float_format='%.3f'))

# %% papermill={"duration": 0.019192, "end_time": "2025-12-17T19:12:38.906507", "exception": false, "start_time": "2025-12-17T19:12:38.887315", "status": "completed"}
print("\n" + "="*80)
print("SIGNIFICANCE ANALYSIS BY STATISTIC")
print("="*80)

for stat in STATISTICS:
    stat_df = div_df[div_df['statistic'] == stat]
    sig_both = stat_df[stat_df['both_significant']]
    
    print(f"\n{stat}:")
    print(f"  Significant in 2016 only: {(stat_df['2016_significant'] & ~stat_df['2024_significant']).sum()}")
    print(f"  Significant in 2024 only: {(~stat_df['2016_significant'] & stat_df['2024_significant']).sum()}")
    print(f"  Significant in BOTH: {len(sig_both)}")
    
    if len(sig_both) > 0:
        sorted_sig = sig_both.sort_values('combined_effect', ascending=False)
        metapaths_str = ', '.join(sorted_sig['metapath'].tolist())
        print(f"    Metapaths: {metapaths_str}")

# %% [markdown]
# ## Additional Visualizations
#
# 1) Metapath fingerprint heatmap
# 3) Quadrant plot with density
# 4) Metapath network map
# 5) Rank stability plot
# 7) Ridge plots for top metapaths
# 8) Consistency band plot

# %%
run_additional_visualizations(div_df, STATISTICS, output_dir, repo_root, BASE_NAME)

# %% papermill={"duration": 0.026964, "end_time": "2025-12-17T19:12:38.941252", "exception": false, "start_time": "2025-12-17T19:12:38.914288", "status": "completed"}
div_df.to_csv(output_dir / 'divergence_all_statistics.csv', index=False)
print(f"Saved: divergence_all_statistics.csv")

# Significant in both years
sig_both = div_df[div_df['both_significant']].sort_values(['statistic', 'combined_effect'], ascending=[True, False])
sig_both.to_csv(output_dir / 'significant_both_years_all_statistics.csv', index=False)
print(f"Saved: significant_both_years_all_statistics.csv")

print(f"\nTotal (metapath, statistic) combinations significant in both years: {len(sig_both)}")
print(f"Unique metapaths significant for at least one statistic: {sig_both['metapath'].nunique()}")

# %% papermill={"duration": 0.010412, "end_time": "2025-12-17T19:12:38.958736", "exception": false, "start_time": "2025-12-17T19:12:38.948324", "status": "completed"}
print("\n" + "="*80)
print("COMPARISON WITH NOTEBOOK 3 RESULTS")
print("="*80)

nb3_path = repo_root / 'output' / 'metapath_analysis' / 'pairwise_statistics' / 'metapaths_meeting_all_criteria.csv'

if nb3_path.exists():
    nb3_results = pd.read_csv(nb3_path)
    
    print("\nComparison by statistic:")
    print("-" * 60)
    
    for stat in STATISTICS:
        nb3_stat = set(nb3_results[nb3_results['statistic'] == stat]['metapath'])
        nb4_stat = set(div_df[(div_df['statistic'] == stat) & div_df['both_significant']]['metapath'])
        
        overlap = nb3_stat & nb4_stat
        only_nb3 = nb3_stat - nb4_stat
        only_nb4 = nb4_stat - nb3_stat
        
        print(f"\n{stat}:")
        print(f"  NB3 (stability pattern): {len(nb3_stat)}")
        print(f"  NB4 (significant both): {len(nb4_stat)}")
        print(f"  Overlap: {len(overlap)}")
        if overlap:
            print(f"    {sorted(overlap)}")
else:
    print("\nNotebook 3 results not found.")

# %% papermill={"duration": 0.012126, "end_time": "2025-12-17T19:12:38.978125", "exception": false, "start_time": "2025-12-17T19:12:38.965999", "status": "completed"}
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

summary_stats = {
    'Total (metapath, statistic) combinations': len(div_df),
    'Statistics analyzed': len(STATISTICS),
    'Unique metapaths': div_df['metapath'].nunique(),
    'Significant in 2016 (p < 0.05)': div_df['2016_significant'].sum(),
    'Significant in 2024 (p < 0.05)': div_df['2024_significant'].sum(),
    'Significant in BOTH years': div_df['both_significant'].sum(),
    'Unique metapaths sig. in both (any stat)': div_df[div_df['both_significant']]['metapath'].nunique()
}

for k, v in summary_stats.items():
    print(f"{k}: {v}")

print("\n" + "-"*40)
print("Significant combinations by statistic:")
print("-"*40)
for stat in STATISTICS:
    n = div_df[(div_df['statistic'] == stat) & div_df['both_significant']].shape[0]
    print(f"  {stat}: {n}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {output_dir}")
