| model_id                                     | model_family   | feature_set   |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   brier_mean |   fold_count |
|:---------------------------------------------|:---------------|:--------------|---------------:|--------------:|--------------:|-------------:|-------------:|-------------:|
| xgb_all_class_paper_weighted                 | xgboost        | all           |       0.914887 |    0.00189236 |      0.97256  |  0.00168613  |     0.115116 |            3 |
| xgb_all_unweighted                           | xgboost        | all           |       0.913243 |    0.00624916 |      0.972035 |  0.00213751  |     0.103029 |            3 |
| xgb_physics_unweighted                       | xgboost        | physics       |       0.894827 |    0.00694222 |      0.96448  |  0.00255297  |     0.113083 |            3 |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |       0.8943   |    0.00212165 |      0.964541 |  0.00105549  |     0.126737 |            3 |
| svm_all                                      | svm            | all           |       0.876109 |    0.00385591 |      0.957617 |  0.0016056   |     0.149172 |            3 |
| mlp_all                                      | mlp            | all           |       0.87276  |    0.0129968  |      0.958091 |  0.00424704  |     0.149399 |            3 |
| xgb_physics_paper_bagging                    | xgboost        | physics       |       0.86442  |    0.0145669  |      0.954017 |  0.00617795  |     0.127843 |            3 |
| decision_tree_all                            | decision_tree  | all           |       0.862391 |    0.0150706  |      0.942308 |  0.00556737  |     0.150761 |            3 |
| mlp_physics                                  | mlp            | physics       |       0.843704 |    0.00427855 |      0.947034 |  0.000955877 |     0.171493 |            3 |
| svm_physics                                  | svm            | physics       |       0.842528 |    0.00573807 |      0.9444   |  0.000475684 |     0.166166 |            3 |
| decision_tree_physics                        | decision_tree  | physics       |       0.839289 |    0.0242674  |      0.932558 |  0.0200834   |     0.163872 |            3 |
| knn_all                                      | knn            | all           |       0.826535 |    0.00983181 |      0.932213 |  0.00352524  |     0.180312 |            3 |
| knn_physics                                  | knn            | physics       |       0.82365  |    0.00817455 |      0.930167 |  0.00499863  |     0.182206 |            3 |
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |       0.499981 |    0.00223422 |      0.755078 |  0.000624769 |     0.753607 |            3 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |       0.496617 |    0.00549946 |      0.753546 |  0.00267787  |     0.754717 |            3 |
