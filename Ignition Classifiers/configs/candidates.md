# Candidate Design — Version 5

## Context

The pipeline uses a physics-only database (`Microgravity_Database.csv`).
All features are physics-based; there is no `feature_set` distinction.
This document justifies the design of the 26 candidates across 5 model families.

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
- **Class imbalance** — the ignition/no-ignition ratio is not balanced, causing
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
appropriate for moderate class imbalance.

### 5. Structural variants tested with both weighting strategies
Specialized structural variants (distance weighting for KNN, linear kernel for SVM)
are each tested with both `sqrt` and `inverse` paper weighting. This ensures that
the best structural configuration is identified independently of the weighting
strategy choice.

### 6. Decision Tree — no structural variant needed
The Decision Tree search space already tunes `max_depth`, `min_samples_leaf`, and
`criterion` for every candidate. There is no structural variant that is not already
covered by the hyperparameter search, so DT is kept at 3 candidates (baseline,
sqrt, inverse). Adding a fixed-depth "pruned" candidate would be redundant — it
would simply constrain the tuner to values it already explores freely.

### 7. SVM kernel choice — linear vs RBF
The SVM implementation defaults to an RBF kernel for all candidates. `svm_linear`
candidates explicitly test a **linear kernel**, which is a genuinely different
hypothesis: that the ignition decision boundary is linearly separable in the physics
feature space. This is scientifically plausible given that many physics relationships
(e.g. oxygen threshold, pressure effects) are approximately linear in the observed
feature ranges.

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

### Decision Tree (3 candidates)
No structural variant is added since all meaningful hyperparameters (`max_depth`,
`min_samples_leaf`, `criterion`) are already tuned by the nested search for every
candidate.

| candidate_id | paper_weight | class_weight |
|---|---|---|
| dt_baseline | none | false |
| dt_weighted_sqrt | sqrt | true |
| dt_weighted_inverse | inverse | true |

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
Mirrors the XGBoost focal logic at a reduced scale. Focal loss is the key structural
variant, used without class weighting to avoid double-correcting for imbalance.

| candidate_id | paper_weight | class_weight | special |
|---|---|---|---|
| mlp_baseline | none | false | — |
| mlp_weighted_sqrt | sqrt | true | — |
| mlp_weighted_inverse | inverse | true | — |
| mlp_focal_sqrt | sqrt | false | focal γ=2 |
| mlp_focal_inverse | inverse | false | focal γ=2 |

### SVM (5 candidates)
The linear kernel is tested as the key structural variant against the RBF default.
A linear SVM tests whether the ignition boundary is linearly separable in the
physics feature space.

| candidate_id | paper_weight | class_weight | special |
|---|---|---|---|
| svm_baseline | none | false | — |
| svm_weighted_sqrt | sqrt | true | — |
| svm_weighted_inverse | inverse | true | — |
| svm_linear_sqrt | sqrt | true | linear kernel |
| svm_linear_inverse | inverse | true | linear kernel |

---

## Total: 26 candidates

| Family | Count |
|---|---|
| XGBoost | 8 |
| Decision Tree | 3 |
| KNN | 5 |
| MLP | 5 |
| SVM | 5 |
| **Total** | **26** |

The asymmetry in XGBoost (8 vs 3–5) is intentional: as the primary model, it
receives a full ablation of weighting mechanisms. The remaining four families serve
as structured comparison baselines.
