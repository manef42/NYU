# Reproducible FSR champion selection

> The interpolation champion answers "how well can we predict a new row of a paper we already know?" and the extrapolation champion answers "how well can we predict a brand-new paper?". Neither number substitutes for the other.

## Interpolation Champion

Selected `rf_baseline` under data condition `rd`. Lowest eligible RMSE within 2% of the best candidate; tie-breakers applied in declared order: higher_r2, lower_uncertainty, no_augmentation, simpler_model, fewer_features. Refit hyper-parameters use: modal outer-fold selected configuration with lexical tie resolution.

- RMSE: 0.0063 ± 0.0005
- MAE: 0.0021 ± 0.0002
- R2: 0.8790 ± 0.0178
- Protocol: `interpolation_random`
- Exact hyper-parameters: `{"max_depth": 10, "max_features": 0.8, "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 500}`

| Rank | Candidate | Condition | RMSE | MAE | R2 | Features |
|---:|---|---|---:|---:|---:|---|
| 1 | rf_baseline | rd | 0.0063 | 0.0021 | 0.8790 | all |
| 2 | xgb_baseline | rd_bt | 0.0062 | 0.0022 | 0.8783 | all |
| 3 | rf_baseline | rd_bt | 0.0063 | 0.0021 | 0.8780 | all |
| 4 | xgb_absoluteerror_sqrt | rd | 0.0063 | 0.0020 | 0.8752 | all |
| 5 | xgb_baseline | rd | 0.0064 | 0.0023 | 0.8727 | all |
| 6 | rf_paper_sqrt | rd_bt | 0.0064 | 0.0020 | 0.8712 | all |
| 7 | rf_paper_sqrt | rd | 0.0065 | 0.0020 | 0.8685 | all |
| 8 | xgb_log_target_inverse | rd_bt | 0.0065 | 0.0023 | 0.8673 | all |
| 9 | xgb_absoluteerror_sqrt | rd_bt | 0.0065 | 0.0021 | 0.8678 | all |
| 10 | xgb_paper_inverse | rd_bt | 0.0065 | 0.0023 | 0.8670 | all |
| 11 | xgb_log_target_sqrt | rd_bt | 0.0066 | 0.0022 | 0.8622 | all |
| 12 | xgb_log_target_inverse | rd | 0.0066 | 0.0024 | 0.8645 | all |
| 13 | xgb_log_target_sqrt | rd | 0.0066 | 0.0023 | 0.8629 | all |
| 14 | xgb_absoluteerror_inverse | rd | 0.0066 | 0.0023 | 0.8655 | all |
| 15 | xgb_absoluteerror_inverse | rd_bt | 0.0066 | 0.0023 | 0.8648 | all |
| 16 | xgb_paper_sqrt | rd_bt | 0.0066 | 0.0023 | 0.8643 | all |
| 17 | xgb_paper_sqrt | rd | 0.0067 | 0.0024 | 0.8631 | all |
| 18 | xgb_paper_inverse | rd | 0.0067 | 0.0024 | 0.8601 | all |
| 19 | xgb_pseudohuber_inverse | rd_bt | 0.0068 | 0.0023 | 0.8541 | all |
| 20 | xgb_pseudohuber_sqrt | rd | 0.0068 | 0.0023 | 0.8561 | all |
| 21 | xgb_pseudohuber_sqrt | rd_bt | 0.0068 | 0.0023 | 0.8535 | all |
| 22 | xgb_pseudohuber_inverse | rd | 0.0068 | 0.0024 | 0.8504 | all |
| 23 | rf_paper_inverse | rd | 0.0070 | 0.0023 | 0.8480 | all |
| 24 | rf_paper_inverse | rd_bt | 0.0070 | 0.0022 | 0.8465 | all |
| 25 | xgb_paper_bagging_sqrt | rd_bt | 0.0079 | 0.0031 | 0.8031 | all |
| 26 | xgb_paper_bagging_sqrt | rd | 0.0080 | 0.0032 | 0.7997 | all |
| 27 | knn_uniform_sqrt | rd_bt | 0.0090 | 0.0033 | 0.7472 | all |
| 28 | knn_distance_inverse | rd_bt | 0.0092 | 0.0035 | 0.7391 | all |
| 29 | knn_distance | rd_bt | 0.0092 | 0.0033 | 0.7385 | all |
| 30 | knn_distance_sqrt | rd_bt | 0.0093 | 0.0034 | 0.7348 | all |
| 31 | knn_uniform | rd | 0.0093 | 0.0035 | 0.7288 | all |
| 32 | knn_uniform | rd_bt | 0.0093 | 0.0034 | 0.7308 | all |
| 33 | knn_uniform_sqrt | rd | 0.0094 | 0.0035 | 0.7260 | all |
| 34 | knn_distance_sqrt | rd | 0.0094 | 0.0035 | 0.7245 | all |
| 35 | knn_distance_inverse | rd | 0.0095 | 0.0037 | 0.7194 | all |
| 36 | knn_uniform_inverse | rd | 0.0096 | 0.0037 | 0.7162 | all |
| 37 | knn_uniform_inverse | rd_bt | 0.0096 | 0.0036 | 0.7183 | all |
| 38 | knn_distance | rd | 0.0096 | 0.0035 | 0.7080 | all |
| 39 | mlp_log_target_sqrt | rd_bt | 0.0102 | 0.0051 | 0.6777 | all |
| 40 | mlp_log_target_inverse | rd | 0.0102 | 0.0058 | 0.6758 | all |
| 41 | mlp_log_target_inverse | rd_bt | 0.0104 | 0.0059 | 0.6538 | all |
| 42 | mlp_log_target_sqrt | rd | 0.0106 | 0.0053 | 0.6334 | all |
| 43 | svr_rbf | rd_bt | 0.0106 | 0.0073 | 0.6471 | all |
| 44 | svr_rbf_inverse | rd | 0.0106 | 0.0073 | 0.6477 | all |
| 45 | svr_rbf_inverse | rd_bt | 0.0107 | 0.0072 | 0.6440 | all |
| 46 | svr_rbf | rd | 0.0109 | 0.0074 | 0.6212 | all |
| 47 | svr_rbf_sqrt | rd_bt | 0.0109 | 0.0073 | 0.6286 | all |
| 48 | svr_rbf_sqrt | rd | 0.0112 | 0.0074 | 0.6140 | all |
| 49 | svr_linear | rd_bt | 0.0143 | 0.0077 | 0.3734 | all |
| 50 | svr_linear | rd | 0.0146 | 0.0078 | 0.3357 | all |

## Extrapolation Champion

Selected `rf_baseline` under data condition `rd`. Lowest eligible RMSE within 2% of the best candidate; tie-breakers applied in declared order: higher_r2, lower_uncertainty, smaller_generalization_gap, physics_only, no_augmentation, simpler_model. Refit hyper-parameters use: modal outer-fold selected configuration with lexical tie resolution.

- RMSE: 0.0122 ± 0.0041
- MAE: 0.0048 ± 0.0011
- R2: 0.4576 ± 0.2215
- Protocol: `extrapolation_grouped`
- Exact hyper-parameters: `{"max_depth": 12, "max_features": 0.5, "min_samples_leaf": 8, "min_samples_split": 2, "n_estimators": 500}`

| Rank | Candidate | Condition | RMSE | MAE | R2 | Features |
|---:|---|---|---:|---:|---:|---|
| 1 | rf_baseline | rd | 0.0122 | 0.0048 | 0.4576 | all |
| 2 | rf_baseline | rd_bt | 0.0122 | 0.0049 | 0.4477 | all |
| 3 | xgb_paper_sqrt | rd_bt | 0.0126 | 0.0051 | 0.4411 | all |
| 4 | xgb_absoluteerror_inverse | rd_bt | 0.0126 | 0.0050 | 0.4223 | all |
| 5 | xgb_absoluteerror_sqrt | rd | 0.0127 | 0.0051 | 0.4128 | all |
| 6 | xgb_absoluteerror_inverse | rd | 0.0127 | 0.0051 | 0.4039 | all |
| 7 | xgb_log_target_sqrt | rd | 0.0128 | 0.0052 | 0.4142 | all |
| 8 | rf_paper_sqrt | rd | 0.0128 | 0.0050 | 0.4028 | all |
| 9 | rf_paper_sqrt | rd_bt | 0.0129 | 0.0050 | 0.3876 | all |
| 10 | xgb_baseline | rd | 0.0129 | 0.0055 | 0.3964 | all |
| 11 | xgb_paper_sqrt | rd | 0.0130 | 0.0053 | 0.4021 | all |
| 12 | xgb_absoluteerror_sqrt | rd_bt | 0.0130 | 0.0050 | 0.3929 | all |
| 13 | rf_paper_inverse | rd_bt | 0.0131 | 0.0049 | 0.3347 | all |
| 14 | xgb_pseudohuber_inverse | rd | 0.0132 | 0.0053 | 0.3612 | all |
| 15 | xgb_pseudohuber_inverse | rd_bt | 0.0132 | 0.0053 | 0.3614 | all |
| 16 | xgb_pseudohuber_sqrt | rd_bt | 0.0133 | 0.0055 | 0.3551 | all |
| 17 | rf_paper_inverse | rd | 0.0134 | 0.0050 | 0.3319 | all |
| 18 | xgb_paper_inverse | rd | 0.0134 | 0.0054 | 0.3404 | all |
| 19 | xgb_log_target_sqrt | rd_bt | 0.0134 | 0.0054 | 0.3387 | all |
| 20 | xgb_log_target_inverse | rd_bt | 0.0135 | 0.0054 | 0.3480 | all |
| 21 | xgb_pseudohuber_sqrt | rd | 0.0135 | 0.0057 | 0.3440 | all |
| 22 | xgb_paper_bagging_sqrt | rd | 0.0135 | 0.0059 | 0.3273 | all |
| 23 | xgb_baseline | rd_bt | 0.0135 | 0.0056 | 0.3420 | all |
| 24 | xgb_paper_bagging_sqrt | rd_bt | 0.0135 | 0.0059 | 0.3207 | all |
| 25 | knn_distance_inverse | rd_bt | 0.0136 | 0.0060 | 0.2945 | all |
| 26 | knn_uniform_inverse | rd_bt | 0.0140 | 0.0061 | 0.2521 | all |
| 27 | xgb_paper_inverse | rd_bt | 0.0140 | 0.0055 | 0.2670 | all |
| 28 | knn_distance_sqrt | rd_bt | 0.0141 | 0.0061 | 0.2402 | all |
| 29 | xgb_log_target_inverse | rd | 0.0141 | 0.0059 | 0.2682 | all |
| 30 | knn_distance_inverse | rd | 0.0143 | 0.0062 | 0.2047 | all |
| 31 | knn_uniform_inverse | rd | 0.0143 | 0.0062 | 0.2047 | all |
| 32 | knn_distance | rd | 0.0146 | 0.0062 | 0.2208 | all |
| 33 | knn_uniform | rd | 0.0146 | 0.0062 | 0.2208 | all |
| 34 | knn_uniform_sqrt | rd_bt | 0.0149 | 0.0062 | 0.0711 | all |
| 35 | knn_distance | rd_bt | 0.0149 | 0.0064 | 0.1659 | all |
| 36 | knn_distance_sqrt | rd | 0.0152 | 0.0064 | 0.0567 | all |
| 37 | knn_uniform_sqrt | rd | 0.0152 | 0.0064 | 0.0567 | all |
| 38 | knn_uniform | rd_bt | 0.0153 | 0.0064 | 0.1455 | all |
| 39 | svr_rbf_sqrt | rd | 0.0153 | 0.0089 | 0.1573 | all |
| 40 | svr_rbf_sqrt | rd_bt | 0.0155 | 0.0093 | 0.1455 | all |
| 41 | svr_rbf_inverse | rd | 0.0156 | 0.0100 | 0.1154 | all |
| 42 | svr_rbf_inverse | rd_bt | 0.0158 | 0.0102 | 0.0808 | all |
| 43 | svr_rbf | rd_bt | 0.0160 | 0.0093 | 0.0695 | all |
| 44 | svr_rbf | rd | 0.0161 | 0.0095 | 0.0505 | all |

Rejected:
- mlp_log_target_sqrt [rd]: mean R2 -0.7998119046362822 is below -0.25
- svr_linear [rd]: RMSE fold coefficient of variation 0.458 exceeds 0.45; mean R2 -1.0240694422077508 is below -0.25
- mlp_log_target_inverse [rd]: RMSE fold coefficient of variation 0.567 exceeds 0.45; mean R2 -1.3325248205043485 is below -0.25
- mlp_log_target_sqrt [rd_bt]: RMSE fold coefficient of variation 0.463 exceeds 0.45; mean R2 -0.6961545152191372 is below -0.25
- mlp_log_target_inverse [rd_bt]: RMSE fold coefficient of variation 0.466 exceeds 0.45; mean R2 -1.1078266851328094 is below -0.25
- svr_linear [rd_bt]: RMSE fold coefficient of variation 0.475 exceeds 0.45; mean R2 -3.093429239857993 is below -0.25

