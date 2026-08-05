# Reproducible champion selection

> Interpolation and extrapolation champions answer different scientific questions; neither result substitutes for the other.

## Interpolation Champion

Selected `xgb_paper_sqrt`. Highest eligible roc_auc within tolerance 0.01; tie-breakers applied in declared order: higher_pr_auc, lower_uncertainty, simpler_model. Refit hyperparameters use: modal outer-fold selected configuration with lexical tie resolution.

- ROC-AUC: 0.9076 ± 0.0102
- PR-AUC: 0.9687 ± 0.0039
- Protocol: `interpolation_stratified`
- Exact hyperparameters: `{"colsample_bytree": 0.6, "gamma": 0.0, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1, "n_estimators": 800, "objective_variant": "logistic", "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 1.0}`

| Rank | Candidate | Family | ROC-AUC | PR-AUC | SD |
|---:|---|---|---:|---:|---:|
| 1 | xgb_paper_sqrt | xgboost | 0.9076 | 0.9687 | 0.0102 |
| 2 | xgb_baseline | xgboost | 0.9058 | 0.9682 | 0.0091 |
| 3 | xgb_weighted_sqrt | xgboost | 0.9057 | 0.9679 | 0.0088 |
| 4 | xgb_class_only | xgboost | 0.9055 | 0.9680 | 0.0083 |
| 5 | xgb_weighted_inverse | xgboost | 0.9010 | 0.9661 | 0.0088 |
| 6 | xgb_paper_inverse | xgboost | 0.9008 | 0.9664 | 0.0101 |
| 7 | mlp_baseline | mlp | 0.8708 | 0.9509 | 0.0162 |
| 8 | mlp_weighted_sqrt | mlp | 0.8661 | 0.9542 | 0.0118 |
| 9 | mlp_focal_sqrt | mlp | 0.8609 | 0.9494 | 0.0163 |
| 10 | svm_weighted_sqrt | svm | 0.8608 | 0.9518 | 0.0125 |
| 11 | svm_weighted_inverse | svm | 0.8567 | 0.9502 | 0.0119 |
| 12 | mlp_weighted_inverse | mlp | 0.8539 | 0.9474 | 0.0149 |
| 13 | dt_baseline | decision_tree | 0.8531 | 0.9372 | 0.0192 |
| 14 | dt_weighted_sqrt | decision_tree | 0.8481 | 0.9367 | 0.0164 |
| 15 | mlp_focal_inverse | mlp | 0.8466 | 0.9445 | 0.0148 |
| 16 | dt_weighted_inverse | decision_tree | 0.8376 | 0.9342 | 0.0124 |
| 17 | knn_weighted_sqrt | knn | 0.8325 | 0.9340 | 0.0122 |
| 18 | knn_distance_sqrt | knn | 0.8319 | 0.9337 | 0.0114 |
| 19 | knn_weighted_inverse | knn | 0.8271 | 0.9325 | 0.0161 |
| 20 | knn_distance_inverse | knn | 0.8251 | 0.9299 | 0.0177 |
| 21 | knn_baseline | knn | 0.8224 | 0.9251 | 0.0164 |
| 22 | svm_baseline | svm | 0.8222 | 0.9315 | 0.0160 |
| 23 | svm_linear_sqrt | svm | 0.7579 | 0.9032 | 0.0159 |
| 24 | svm_linear_inverse | svm | 0.7555 | 0.9004 | 0.0163 |
| 25 | xgb_focal_inverse | xgboost | 0.4987 | 0.7539 | 0.0032 |
| 26 | xgb_focal_sqrt | xgboost | 0.4960 | 0.7533 | 0.0111 |

## Extrapolation Champion

Selected `xgb_baseline`. Highest eligible roc_auc within tolerance 0.01; tie-breakers applied in declared order: higher_pr_auc, lower_uncertainty, simpler_model. Refit hyperparameters use: modal outer-fold selected configuration with lexical tie resolution.

- ROC-AUC: 0.7183 ± 0.0444
- PR-AUC: 0.8767 ± 0.0298
- Protocol: `extrapolation_grouped`
- Exact hyperparameters: `{"colsample_bytree": 0.6, "gamma": 0.1, "learning_rate": 0.01, "max_depth": 2, "min_child_weight": 4, "n_estimators": 400, "objective_variant": "logistic", "reg_alpha": 0.5, "reg_lambda": 1.0, "subsample": 0.6}`

| Rank | Candidate | Family | ROC-AUC | PR-AUC | SD |
|---:|---|---|---:|---:|---:|
| 1 | xgb_baseline | xgboost | 0.7183 | 0.8767 | 0.0444 |
| 2 | xgb_class_only | xgboost | 0.7036 | 0.8697 | 0.0447 |
| 3 | xgb_paper_sqrt | xgboost | 0.7004 | 0.8650 | 0.0365 |
| 4 | xgb_paper_inverse | xgboost | 0.6934 | 0.8617 | 0.0394 |
| 5 | svm_weighted_inverse | svm | 0.6934 | 0.8787 | 0.0456 |
| 6 | mlp_weighted_sqrt | mlp | 0.6928 | 0.8607 | 0.0630 |
| 7 | xgb_weighted_sqrt | xgboost | 0.6842 | 0.8555 | 0.0455 |
| 8 | mlp_baseline | mlp | 0.6814 | 0.8562 | 0.0635 |
| 9 | svm_linear_inverse | svm | 0.6810 | 0.8644 | 0.0808 |
| 10 | xgb_weighted_inverse | xgboost | 0.6797 | 0.8526 | 0.0385 |
| 11 | svm_weighted_sqrt | svm | 0.6792 | 0.8675 | 0.0433 |
| 12 | mlp_focal_sqrt | mlp | 0.6741 | 0.8569 | 0.0497 |
| 13 | svm_linear_sqrt | svm | 0.6724 | 0.8550 | 0.0924 |
| 14 | mlp_weighted_inverse | mlp | 0.6692 | 0.8530 | 0.0534 |
| 15 | knn_weighted_sqrt | knn | 0.6688 | 0.8471 | 0.0571 |
| 16 | knn_baseline | knn | 0.6651 | 0.8478 | 0.0760 |
| 17 | svm_baseline | svm | 0.6641 | 0.8580 | 0.0687 |
| 18 | knn_distance_sqrt | knn | 0.6633 | 0.8461 | 0.0594 |
| 19 | mlp_focal_inverse | mlp | 0.6572 | 0.8487 | 0.0756 |
| 20 | knn_distance_inverse | knn | 0.6396 | 0.8296 | 0.0435 |
| 21 | knn_weighted_inverse | knn | 0.6309 | 0.8262 | 0.0531 |
| 22 | dt_weighted_inverse | decision_tree | 0.6298 | 0.8210 | 0.0996 |
| 23 | dt_weighted_sqrt | decision_tree | 0.6263 | 0.8146 | 0.0750 |
| 24 | dt_baseline | decision_tree | 0.6117 | 0.8105 | 0.0711 |
| 25 | xgb_focal_inverse | xgboost | 0.4997 | 0.7554 | 0.0010 |
| 26 | xgb_focal_sqrt | xgboost | 0.4802 | 0.7516 | 0.0386 |

