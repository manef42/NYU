| champion               | model_id           | protocol                 | threshold_policy   |   threshold_mean |    tp |   fp |   tn |   fn |   sensitivity |   specificity |   precision |
|:-----------------------|:-------------------|:-------------------------|:-------------------|-----------------:|------:|-----:|-----:|-----:|--------------:|--------------:|------------:|
| interpolation_champion | xgb_all_unweighted | interpolation_stratified | mcc                |         0.814944 |  7996 |  330 | 2985 | 2192 |      0.784845 |      0.900452 |    0.960365 |
| interpolation_champion | xgb_all_unweighted | interpolation_stratified | f1                 |         0.357602 |  9655 | 1715 | 1600 |  533 |      0.947684 |      0.482655 |    0.849164 |
| interpolation_champion | xgb_all_unweighted | interpolation_stratified | balanced_accuracy  |         0.854504 |  7796 |  239 | 3076 | 2392 |      0.765214 |      0.927903 |    0.970255 |
| interpolation_champion | xgb_all_unweighted | interpolation_stratified | youden_j           |         0.854504 |  7796 |  239 | 3076 | 2392 |      0.765214 |      0.927903 |    0.970255 |
| extrapolation_champion | xgb_all_unweighted | extrapolation_grouped    | mcc                |         0.827769 |  7068 | 1368 | 1947 | 3120 |      0.693757 |      0.58733  |    0.837838 |
| extrapolation_champion | xgb_all_unweighted | extrapolation_grouped    | f1                 |         0.139965 | 10040 | 3177 |  138 |  148 |      0.985473 |      0.041629 |    0.759628 |
| extrapolation_champion | xgb_all_unweighted | extrapolation_grouped    | balanced_accuracy  |         0.906665 |  6142 |  962 | 2353 | 4046 |      0.602866 |      0.709804 |    0.864583 |
| extrapolation_champion | xgb_all_unweighted | extrapolation_grouped    | youden_j           |         0.906665 |  6142 |  962 | 2353 | 4046 |      0.602866 |      0.709804 |    0.864583 |
