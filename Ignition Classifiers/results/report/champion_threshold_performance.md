| champion               | model_id       | protocol                 | threshold_policy   |   threshold_mean |   tp |   fp |   tn |   fn |   sensitivity |   specificity |   precision |
|:-----------------------|:---------------|:-------------------------|:-------------------|-----------------:|-----:|-----:|-----:|-----:|--------------:|--------------:|------------:|
| interpolation_champion | xgb_paper_sqrt | interpolation_stratified | mcc                |         0.792046 | 8034 |  423 | 2889 | 2115 |      0.791605 |     0.872283  |    0.949982 |
| interpolation_champion | xgb_paper_sqrt | interpolation_stratified | f1                 |         0.368789 | 9658 | 1723 | 1589 |  491 |      0.951621 |     0.479771  |    0.848607 |
| interpolation_champion | xgb_paper_sqrt | interpolation_stratified | balanced_accuracy  |         0.837934 | 7774 |  327 | 2985 | 2375 |      0.765987 |     0.901268  |    0.959635 |
| interpolation_champion | xgb_paper_sqrt | interpolation_stratified | youden_j           |         0.837934 | 7774 |  327 | 2985 | 2375 |      0.765987 |     0.901268  |    0.959635 |
| extrapolation_champion | xgb_baseline   | extrapolation_grouped    | mcc                |         0.809915 | 7399 | 1379 | 1933 | 2750 |      0.729037 |     0.583635  |    0.842903 |
| extrapolation_champion | xgb_baseline   | extrapolation_grouped    | f1                 |         0.284961 | 9984 | 3116 |  196 |  165 |      0.983742 |     0.0591787 |    0.762137 |
| extrapolation_champion | xgb_baseline   | extrapolation_grouped    | balanced_accuracy  |         0.858301 | 6738 | 1119 | 2193 | 3411 |      0.663908 |     0.662138  |    0.857579 |
| extrapolation_champion | xgb_baseline   | extrapolation_grouped    | youden_j           |         0.858301 | 6738 | 1119 | 2193 | 3411 |      0.663908 |     0.662138  |    0.857579 |
