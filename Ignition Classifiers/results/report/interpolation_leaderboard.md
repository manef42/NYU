| model_id                                     | model_family   | feature_set   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   fold_count |
|:---------------------------------------------|:---------------|:--------------|---------------:|--------------:|--------------:|-------------:|-------------:|-------------:|
| xgb_all_unweighted                           | xgboost        | all           |       0.913307 |    0.0070564  |      0.97174  |   0.00284895 |     0.104477 |           15 |
| xgb_all_class_paper_weighted                 | xgboost        | all           |       0.912703 |    0.00723809 |      0.971302 |   0.00316095 |     0.1138   |           15 |
| xgb_physics_unweighted                       | xgboost        | physics       |       0.897297 |    0.0127224  |      0.965758 |   0.00538873 |     0.113259 |           15 |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |       0.894988 |    0.0132565  |      0.964661 |   0.0055566  |     0.125499 |           15 |
| mlp_all                                      | mlp            | all           |       0.878635 |    0.0119518  |      0.957513 |   0.00609407 |     0.146601 |           15 |
| xgb_physics_paper_bagging                    | xgboost        | physics       |       0.875823 |    0.0123982  |      0.959195 |   0.00580236 |     0.122141 |           15 |
| svm_all                                      | svm            | all           |       0.864018 |    0.0106071  |      0.952073 |   0.00492777 |     0.176767 |           15 |
| mlp_physics                                  | mlp            | physics       |       0.857223 |    0.0139173  |      0.94966  |   0.00691213 |     0.160021 |           15 |
| decision_tree_all                            | decision_tree  | all           |       0.849214 |    0.0188265  |      0.936786 |   0.0115832  |     0.156869 |           15 |
| svm_physics                                  | svm            | physics       |       0.842372 |    0.0120846  |      0.944123 |   0.00657943 |     0.181871 |           15 |
| decision_tree_physics                        | decision_tree  | physics       |       0.834353 |    0.0192574  |      0.931135 |   0.0131942  |     0.165195 |           15 |
| knn_physics                                  | knn            | physics       |       0.829386 |    0.0158381  |      0.929194 |   0.0100293  |     0.178705 |           15 |
| knn_all                                      | knn            | all           |       0.824839 |    0.0101103  |      0.931856 |   0.00624867 |     0.180929 |           15 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |       0.483181 |    0.0309676  |      0.752119 |   0.0089109  |     0.752159 |           15 |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |       0.481381 |    0.0383142  |      0.749464 |   0.0117161  |     0.745651 |           15 |
