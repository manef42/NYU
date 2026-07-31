# Reproducible champion selection

> Interpolation and extrapolation champions answer different scientific questions; neither result substitutes for the other.

## Interpolation Champion

Selected `xgb_all_unweighted`. Highest eligible roc_auc within tolerance 0.01; tie-breakers applied in declared order: higher_pr_auc, lower_uncertainty, simpler_model, fewer_features. Refit hyperparameters use: modal outer-fold selected configuration with lexical tie resolution.

- ROC-AUC: 0.9133 ± 0.0071
- PR-AUC: 0.9717 ± 0.0028
- Protocol: `interpolation_stratified`
- Exact hyperparameters: `{"colsample_bytree": 0.6, "gamma": 0.0, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 2, "n_estimators": 800, "objective_variant": "logistic", "reg_alpha": 0.5, "reg_lambda": 5.0, "subsample": 0.6}`

| Rank | Candidate | ROC-AUC | PR-AUC | SD | Features |
|---:|---|---:|---:|---:|---|
| 1 | xgb_all_unweighted | 0.9133 | 0.9717 | 0.0071 | all |
| 2 | xgb_all_class_paper_weighted | 0.9127 | 0.9713 | 0.0072 | all |
| 3 | xgb_physics_unweighted | 0.8973 | 0.9658 | 0.0127 | physics |
| 4 | xgb_physics_class_paper_weighted_monotone_o2 | 0.8950 | 0.9647 | 0.0133 | physics |
| 5 | mlp_all | 0.8786 | 0.9575 | 0.0120 | all |
| 6 | xgb_physics_paper_bagging | 0.8758 | 0.9592 | 0.0124 | physics |
| 7 | svm_all | 0.8640 | 0.9521 | 0.0106 | all |
| 8 | mlp_physics | 0.8572 | 0.9497 | 0.0139 | physics |
| 9 | decision_tree_all | 0.8492 | 0.9368 | 0.0188 | all |
| 10 | svm_physics | 0.8424 | 0.9441 | 0.0121 | physics |
| 11 | decision_tree_physics | 0.8344 | 0.9311 | 0.0193 | physics |
| 12 | knn_physics | 0.8294 | 0.9292 | 0.0158 | physics |
| 13 | knn_all | 0.8248 | 0.9319 | 0.0101 | all |
| 14 | xgb_physics_focal_class_paper_weighted | 0.4832 | 0.7521 | 0.0310 | physics |
| 15 | xgb_all_focal_class_paper_weighted | 0.4814 | 0.7495 | 0.0383 | all |

## Extrapolation Champion

Selected `xgb_all_unweighted`. Highest eligible roc_auc within tolerance 0.01; tie-breakers applied in declared order: higher_pr_auc, lower_uncertainty, physics_only, simpler_model. Refit hyperparameters use: modal outer-fold selected configuration with lexical tie resolution.

- ROC-AUC: 0.7201 ± 0.0455
- PR-AUC: 0.8830 ± 0.0276
- Protocol: `extrapolation_grouped`
- Exact hyperparameters: `{"colsample_bytree": 0.6, "gamma": 0.0, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 2, "n_estimators": 800, "objective_variant": "logistic", "reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.6}`

| Rank | Candidate | ROC-AUC | PR-AUC | SD | Features |
|---:|---|---:|---:|---:|---|
| 1 | xgb_all_unweighted | 0.7201 | 0.8830 | 0.0455 | all |
| 2 | xgb_physics_unweighted | 0.7225 | 0.8812 | 0.0448 | physics |
| 3 | xgb_all_class_paper_weighted | 0.7190 | 0.8823 | 0.0465 | all |
| 4 | xgb_physics_paper_bagging | 0.7176 | 0.8764 | 0.0551 | physics |
| 5 | xgb_physics_class_paper_weighted_monotone_o2 | 0.7132 | 0.8734 | 0.0390 | physics |
| 6 | knn_physics | 0.7025 | 0.8671 | 0.0427 | physics |
| 7 | svm_physics | 0.6988 | 0.8754 | 0.0498 | physics |
| 8 | mlp_physics | 0.6921 | 0.8633 | 0.0572 | physics |
| 9 | mlp_all | 0.6800 | 0.8668 | 0.0566 | all |
| 10 | svm_all | 0.6600 | 0.8624 | 0.0648 | all |
| 11 | decision_tree_physics | 0.6402 | 0.8218 | 0.0824 | physics |
| 12 | decision_tree_all | 0.6354 | 0.8186 | 0.0880 | all |
| 13 | knn_all | 0.6145 | 0.8253 | 0.0831 | all |
| 14 | xgb_all_focal_class_paper_weighted | 0.4904 | 0.7526 | 0.0343 | all |
| 15 | xgb_physics_focal_class_paper_weighted | 0.4539 | 0.7416 | 0.0667 | physics |

