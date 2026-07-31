# Candidate Design — Version 3

## Context

This document justifies the design of the 28 candidates across 5 model families.

---

## Design Principles

### 1. Baseline first
Every family includes an unweighted, unmodified baseline (`paper_weight: none`,
`class_weight: false`). This is the reference point against which all other
candidates are compared.

### 2. Isolating imbalance correction mechanisms
The database presents **two independent sources of imbalance**:

- **Paper imbalance** — some papers contribute disproportionately more rows than
  others, introducing sampling bias toward specific experimental conditions.
- **Class imbalance** — the ignition/extinction ratio is not balanced, causing
  models to be biased toward the majority class.

These two problems are corrected by orthogonal mechanisms:
- `paper_weight` corrects sampling bias by down-weighting over-represented papers.
- `class_weight` corrects label bias by re-scaling loss contributions per class.

For XGBoost (the primary model), candidates isolate each mechanism independently
(`paper_only`, `class_only`) before combining them, allowing clean attribution of
performance gains. For the remaining families, only the combined weighting is tested
since they serve as comparison baselines.

### 3. Two paper weighting strategies: `sqrt` vs `inverse`
Both strategies are available in the pipeline (`fable_common.py: paper_weights()`).

| Strategy | Formula | Behavior |
|---|---|---|
| `sqrt` | 1 / √count | Moderate correction, preserves some influence of larger papers |
| `inverse` | 1 / count | Aggressive correction, fully equalizes paper contributions |

`sqrt` is the standard default and works well under moderate paper imbalance.
`inverse` is tested alongside it to assess whether a stronger correction is
beneficial given the specific imbalance structure of the physics database.
The best strategy is determined empirically from the nested CV results.

### 4. Focal loss stands alone
Focal loss (Lin et al., 2017) and class weighting are **both imbalance-handling
mechanisms** — they address the same problem through different routes:

- **Class weight** — re-scales loss by class frequency at the dataset level.
- **Focal loss** — dynamically down-weights easy/confident samples during training,
  regardless of class.

Combining them risks **double-correcting** for imbalance and over-penalizing the
majority class. Focal loss candidates therefore use `class_weight: false` and rely
solely on focal loss for imbalance correction. Paper weighting is retained since it
corrects a different type of bias (sampling, not label).

`focal_gamma` is fixed at **2.0**, the standard value from the original paper,
appropriate for moderate class imbalance. Tuning γ is deferred to a future ablation.

### 5. Structural variants tested with both weighting strategies
Specialized structural variants (pruning for Decision Tree, distance weighting for
KNN, RBF kernel for SVM) are each tested with both `sqrt` and `inverse` paper
weighting. This ensures that the best structural configuration is identified
independently of the weighting strategy choice.

---

## Candidate Summary

### XGBoost (8 candidates)
XGBoost is the primary model and receives the most granular treatment.

| candidate_id | paper_weight | class_weight | objective |
|---|---|---|---|
| xgb_baseline | none | false | logistic |
| xgb_paper_sqrt | sqrt | false | logistic |
| xgb_paper_inverse | inverse | false | logistic |
| xgb_class_only | none | true | logistic |
| xgb_weighted_sqrt | sqrt | true | logistic |
| xgb_weighted_inverse | inverse | true | logistic |
| xgb_focal_sqrt | sqrt | false | focal γ=2 |
| xgb_focal_inverse | inverse | false | focal γ=2 |

The first six candidates form a 2×3 ablation grid isolating paper weight strategy
(none / sqrt / inverse) and class weight (off / on). The last two test focal loss
as a standalone imbalance correction with each paper weighting strategy.

### Decision Tree (5 candidates)
Pruning (`max_depth`, `min_samples_leaf` tuned via nested search) is the key
structural variant, controlling overfitting on the small physics dataset.

| candidate_id | paper_weight | class_weight | special |
|---|---|---|---|
| dt_baseline | none | false | — |
| dt_weighted_sqrt | sqrt | true | — |
| dt_weighted_inverse | inverse | true | — |
| dt_pruned_sqrt | sqrt | true | pruned |
| dt_pruned_inverse | inverse | true | pruned |

### KNN (5 candidates)
Distance-based sample weighting (`weights: distance`) is the key structural variant,
giving closer neighbors stronger influence during prediction.

| candidate_id | paper_weight | class_weight | special |
|---|---|---|---|
| knn_baseline | none | false | — |
| knn_weighted_sqrt | sqrt | true | — |
| knn_weighted_inverse | inverse | true | — |
| knn_distance_sqrt | sqrt | true | distance weights |
| knn_distance_inverse | inverse | true | distance weights |

### MLP (5 candidates)
Mirrors the XGBoost focal logic at a reduced scale (no paper/class isolation split
since MLP is a comparison baseline). Focal loss is the key structural variant.

| candidate_id | paper_weight | class_weight | special |
|---|---|---|---|
| mlp_baseline | none | false | — |
| mlp_weighted_sqrt | sqrt | true | — |
| mlp_weighted_inverse | inverse | true | — |
| mlp_focal_sqrt | sqrt | false | focal γ=2 |
| mlp_focal_inverse | inverse | false | focal γ=2 |

### SVM (5 candidates)
The RBF kernel is tested as the key structural variant. The default kernel (linear)
is implicitly covered by the baseline and weighted candidates.

| candidate_id | paper_weight | class_weight | special |
|---|---|---|---|
| svm_baseline | none | false | — |
| svm_weighted_sqrt | sqrt | true | — |
| svm_weighted_inverse | inverse | true | — |
| svm_rbf_sqrt | sqrt | true | RBF kernel |
| svm_rbf_inverse | inverse | true | RBF kernel |

---

## Total: 28 candidates

| Family | Count |
|---|---|
| XGBoost | 8 |
| Decision Tree | 5 |
| KNN | 5 |
| MLP | 5 |
| SVM | 5 |
| **Total** | **28** |

The asymmetry in XGBoost (8 vs 5) is intentional: as the primary model, it receives
a full ablation of weighting mechanisms. The remaining four families serve as
structured comparison baselines and do not require the same granularity.
