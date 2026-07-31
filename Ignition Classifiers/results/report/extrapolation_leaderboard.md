| model_id                                     | model_family   | feature_set   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   fold_count |
|:---------------------------------------------|:---------------|:--------------|---------------:|--------------:|--------------:|-------------:|-------------:|-------------:|
| xgb_physics_unweighted                       | xgboost        | physics       |       0.722525 |     0.0447671 |      0.881231 |    0.0309045 |     0.17481  |           15 |
| xgb_all_unweighted                           | xgboost        | all           |       0.720102 |     0.0455493 |      0.88301  |    0.0276044 |     0.183428 |           15 |
| xgb_all_class_paper_weighted                 | xgboost        | all           |       0.718996 |     0.0464987 |      0.882279 |    0.0242816 |     0.195623 |           15 |
| xgb_physics_paper_bagging                    | xgboost        | physics       |       0.717619 |     0.0550644 |      0.876369 |    0.0361217 |     0.177973 |           15 |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |       0.713239 |     0.0390041 |      0.873419 |    0.030319  |     0.187964 |           15 |
| knn_physics                                  | knn            | physics       |       0.702463 |     0.0427423 |      0.8671   |    0.0283317 |     0.214185 |           15 |
| svm_physics                                  | svm            | physics       |       0.698753 |     0.0497803 |      0.875361 |    0.0305303 |     0.202227 |           15 |
| mlp_physics                                  | mlp            | physics       |       0.692088 |     0.0572007 |      0.863304 |    0.0370558 |     0.236533 |           15 |
| mlp_all                                      | mlp            | all           |       0.67998  |     0.0566343 |      0.866786 |    0.0294759 |     0.237017 |           15 |
| svm_all                                      | svm            | all           |       0.660016 |     0.0647971 |      0.862352 |    0.0387732 |     0.206518 |           15 |
| decision_tree_physics                        | decision_tree  | physics       |       0.640245 |     0.0823774 |      0.821841 |    0.049635  |     0.228469 |           15 |
| decision_tree_all                            | decision_tree  | all           |       0.635421 |     0.0879657 |      0.81864  |    0.0475856 |     0.227964 |           15 |
| knn_all                                      | knn            | all           |       0.614507 |     0.0831158 |      0.825324 |    0.0525389 |     0.252298 |           15 |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |       0.490376 |     0.0342793 |      0.752557 |    0.0183788 |     0.728139 |           15 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |       0.453868 |     0.0666781 |      0.741609 |    0.0254859 |     0.737194 |           15 |
