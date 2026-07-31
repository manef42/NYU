| model_id                                     | model_family   | feature_set   |   extrapolation_grouped |   interpolation_stratified |   roc_auc_drop |
|:---------------------------------------------|:---------------|:--------------|------------------------:|---------------------------:|---------------:|
| xgb_all_focal_class_paper_weighted           | xgboost        | all           |                0.490376 |                   0.481381 |    -0.00899487 |
| xgb_physics_focal_class_paper_weighted       | xgboost        | physics       |                0.453868 |                   0.483181 |     0.0293131  |
| knn_physics                                  | knn            | physics       |                0.702463 |                   0.829386 |     0.126923   |
| svm_physics                                  | svm            | physics       |                0.698753 |                   0.842372 |     0.143619   |
| xgb_physics_paper_bagging                    | xgboost        | physics       |                0.717619 |                   0.875823 |     0.158204   |
| mlp_physics                                  | mlp            | physics       |                0.692088 |                   0.857223 |     0.165135   |
| xgb_physics_unweighted                       | xgboost        | physics       |                0.722525 |                   0.897297 |     0.174771   |
| xgb_physics_class_paper_weighted_monotone_o2 | xgboost        | physics       |                0.713239 |                   0.894988 |     0.181749   |
| xgb_all_unweighted                           | xgboost        | all           |                0.720102 |                   0.913307 |     0.193204   |
| xgb_all_class_paper_weighted                 | xgboost        | all           |                0.718996 |                   0.912703 |     0.193708   |
| decision_tree_physics                        | decision_tree  | physics       |                0.640245 |                   0.834353 |     0.194107   |
| mlp_all                                      | mlp            | all           |                0.67998  |                   0.878635 |     0.198655   |
| svm_all                                      | svm            | all           |                0.660016 |                   0.864018 |     0.204002   |
| knn_all                                      | knn            | all           |                0.614507 |                   0.824839 |     0.210331   |
| decision_tree_all                            | decision_tree  | all           |                0.635421 |                   0.849214 |     0.213793   |
