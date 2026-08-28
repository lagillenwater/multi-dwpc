# B-convergence: promiscuity vs metaedge_degree vs capacity_hurdle_adaptive vs metaedge_degree_hurdle_adaptive

**Note on `n` / `n_excluded_nan`:** these columns are not directly comparable counts across strategies -- they can differ in *which* rows are excluded, not just how many. A given (lv_id, feature_idx) row's analytical null can be zero-variance (NaN z-score, excluded from `n`) under one stratification scheme and non-degenerate under another, because each strategy assigns different candidate pools to the same row. Read `n` per strategy as "how many rows fed that strategy's metrics", not as evidence that strategies are scored over an identical row population.

## B=10

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |   8 |               10 |                    1   |                   1 |                              0.952381 |                                  1        |                     0.389021 |
| promiscuity                     | L>=3            |  23 |                7 |                    1   |                   1 |                              0.913043 |                                  1        |                     0.885918 |
| metaedge_degree                 | L=2             |  10 |                8 |                    7   |                   5 |                              0.951515 |                                  1        |                     0.576667 |
| metaedge_degree                 | L>=3            |  23 |                7 |                    8   |                   4 |                              0.938735 |                                  0.75     |                     1.15444  |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                              0.963636 |                                  0.666667 |                     0.604437 |
| capacity_hurdle_adaptive        | L>=3            |  23 |                7 |                   18.5 |                   2 |                              0.942688 |                                  0.75     |                     1.05791  |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                              0.987879 |                                  1        |                     0.717886 |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                              0.98087  |                                  0.8      |                     0.902932 |

## B=30

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |   9 |                9 |                    1   |                   1 |                              0.95     |                                  1        |                     0.515051 |
| promiscuity                     | L>=3            |  24 |                6 |                    1   |                   1 |                              0.969565 |                                  1        |                     0.283243 |
| metaedge_degree                 | L=2             |  10 |                8 |                    7   |                   5 |                              0.963636 |                                  1        |                     0.264079 |
| metaedge_degree                 | L>=3            |  24 |                6 |                    8   |                   4 |                              0.998261 |                                  0.933333 |                     0.269873 |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                              0.975758 |                                  1        |                     0.28804  |
| capacity_hurdle_adaptive        | L>=3            |  23 |                7 |                   18.5 |                   2 |                              0.960474 |                                  1        |                     0.511125 |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                              1        |                                  1        |                     0.330795 |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                              0.991304 |                                  1        |                     0.331169 |

## B=100

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |  10 |                8 |                    1   |                   1 |                              0.987879 |                                  1        |                    0.273762  |
| promiscuity                     | L>=3            |  24 |                6 |                    1   |                   1 |                              0.982609 |                                  1        |                    0.175964  |
| metaedge_degree                 | L=2             |  11 |                7 |                    7   |                   5 |                              0.972727 |                                  1        |                    0.535856  |
| metaedge_degree                 | L>=3            |  24 |                6 |                    8   |                   4 |                              0.991304 |                                  0.933333 |                    0.164939  |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                              0.987879 |                                  1        |                    0.154371  |
| capacity_hurdle_adaptive        | L>=3            |  24 |                6 |                   18.5 |                   2 |                              0.98     |                                  0.75     |                    0.273315  |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                              0.987879 |                                  1        |                    0.0946211 |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                              0.993043 |                                  1        |                    0.211608  |

## B=300

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |  10 |                8 |                    1   |                   1 |                              0.987879 |                                  1        |                    0.122856  |
| promiscuity                     | L>=3            |  24 |                6 |                    1   |                   1 |                              0.964348 |                                  1        |                    0.192383  |
| metaedge_degree                 | L=2             |  12 |                6 |                    7   |                   5 |                              0.993007 |                                  1        |                    0.196782  |
| metaedge_degree                 | L>=3            |  24 |                6 |                    8   |                   4 |                              0.991304 |                                  0.933333 |                    0.0906403 |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                              0.987879 |                                  1        |                    0.074638  |
| capacity_hurdle_adaptive        | L>=3            |  24 |                6 |                   18.5 |                   2 |                              0.993913 |                                  1        |                    0.125116  |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                              1        |                                  1        |                    0.0943886 |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                              0.994783 |                                  1        |                    0.0926959 |

## B=1000

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |  10 |                8 |                    1   |                   1 |                              1        |                                         1 |                    0.0473874 |
| promiscuity                     | L>=3            |  24 |                6 |                    1   |                   1 |                              0.992174 |                                         1 |                    0.0915617 |
| metaedge_degree                 | L=2             |  12 |                6 |                    7   |                   5 |                              1        |                                         1 |                    0.196587  |
| metaedge_degree                 | L>=3            |  24 |                6 |                    8   |                   4 |                              0.99913  |                                         1 |                    0.0488982 |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                              1        |                                         1 |                    0.03889   |
| capacity_hurdle_adaptive        | L>=3            |  24 |                6 |                   18.5 |                   2 |                              0.995652 |                                         1 |                    0.0378698 |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                              1        |                                         1 |                    0.0486585 |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                              1        |                                         1 |                    0.0781066 |

## B=3000

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |  10 |                8 |                    1   |                   1 |                               1       |                                         1 |                    0.0674693 |
| promiscuity                     | L>=3            |  24 |                6 |                    1   |                   1 |                               0.99913 |                                         1 |                    0.0980451 |
| metaedge_degree                 | L=2             |  12 |                6 |                    7   |                   5 |                               1       |                                         1 |                    0.152956  |
| metaedge_degree                 | L>=3            |  24 |                6 |                    8   |                   4 |                               0.99913 |                                         1 |                    0.0457103 |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                               1       |                                         1 |                    0.0198278 |
| capacity_hurdle_adaptive        | L>=3            |  24 |                6 |                   18.5 |                   2 |                               1       |                                         1 |                    0.0284886 |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                               1       |                                         1 |                    0.017902  |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                               0.99913 |                                         1 |                    0.0328727 |

## B=10000

| strategy                        | length_bucket   |   n |   n_excluded_nan |   median_active_strata |   min_active_strata |   spearman_rho_analytical_vs_mc_highb |   jaccard_selected_analytical_vs_mc_highb |   max_abs_relative_std_error |
|:--------------------------------|:----------------|----:|-----------------:|-----------------------:|--------------------:|--------------------------------------:|------------------------------------------:|-----------------------------:|
| promiscuity                     | L=2             |  10 |                8 |                    1   |                   1 |                               1       |                                         1 |                    0.018236  |
| promiscuity                     | L>=3            |  24 |                6 |                    1   |                   1 |                               1       |                                         1 |                    0.030179  |
| metaedge_degree                 | L=2             |  12 |                6 |                    7   |                   5 |                               1       |                                         1 |                    0.0307065 |
| metaedge_degree                 | L>=3            |  24 |                6 |                    8   |                   4 |                               0.99913 |                                         1 |                    0.0210697 |
| capacity_hurdle_adaptive        | L=2             |  10 |                8 |                   17.5 |                   1 |                               1       |                                         1 |                    0.0109552 |
| capacity_hurdle_adaptive        | L>=3            |  24 |                6 |                   18.5 |                   2 |                               1       |                                         1 |                    0.0393589 |
| metaedge_degree_hurdle_adaptive | L=2             |  10 |                8 |                   10   |                   2 |                               1       |                                         1 |                    0.0104232 |
| metaedge_degree_hurdle_adaptive | L>=3            |  24 |                6 |                   11   |                   2 |                               1       |                                         1 |                    0.018045  |

## Pass rates and near-threshold density

| strategy                        |   n_valid |   n_pass |   pass_rate |   n_near_threshold |
|:--------------------------------|----------:|---------:|------------:|-------------------:|
| promiscuity                     |        34 |       29 |    0.852941 |                  0 |
| metaedge_degree                 |        36 |       17 |    0.472222 |                  6 |
| capacity_hurdle_adaptive        |        34 |        5 |    0.147059 |                  2 |
| metaedge_degree_hurdle_adaptive |        34 |        7 |    0.205882 |                  2 |

## Fallback merges

- **promiscuity**: 0 fallback merge(s)
- **metaedge_degree**: 0 fallback merge(s)
- **capacity_hurdle_adaptive**: 0 fallback merge(s)
- **metaedge_degree_hurdle_adaptive**: 0 fallback merge(s)

