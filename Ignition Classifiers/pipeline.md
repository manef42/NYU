# 🔥 How the Ignition Classifier Pipeline Works
### A plain-English guide for ML beginners

---

## 🧠 The Big Picture

The goal is simple: **given a set of experimental conditions
(oxygen level, pressure, material, gravity, etc.), predict whether
a material will ignite — Yes or No.**

This is a **binary classification** problem. The twist is that the
data comes from ~93 scientific papers, and we want our model to
work on *brand-new papers it has never seen before* — not just
shuffle the same papers around.

The pipeline is split across several Python files ("modules"), each
doing one job:

| File | Job |
|---|---|
| `fable_common.py` | Load & clean the database, engineer features |
| `fable_splits.py` | Generate all train/test splits (frozen, never touched again) |
| `fable_models.py` | Define all model families and their wrappers |
| `fable_search.py` | Tune hyperparameters *inside* each training fold |
| `fable_evaluate.py` | Run the full benchmark and save all results |
| `fable_refit.py` | Re-train the winning model on all data |
| `fable_report.py` | Generate tables, plots, and the final report |
| `fable_select.py` | Apply the selection policy to pick the champion model |
| `fable_predict.py` | Use the final model to predict on new data |

---

## STEP 1 — Load and Clean the Data (`fable_common.py`)

### What happens
The raw database is an Excel/CSV file with ~5,000 rows. Each row is
one combustion experiment from a published paper.

### Feature Engineering
The raw columns are messy (e.g. `"94 W"`, `"0.21 atm"`, `"3×2×1 mm"`).
The code parses every cell into a clean number:

- **Pressure** → always converted to **kPa**
  - `"1 atm"` → `101.325 kPa`
  - `"0.5 MPa"` → `500 kPa`
- **Flow velocity** → always **mm/s**
  - `"5 cm/s"` → `50 mm/s`
- **Gravity** → fraction of Earth gravity (**g**)
  - `"9.81 m/s²"` → `1.0 g`
  - `"microgravity"` → `0.000001 g`
- **Dimensions** → split into up to 3 individual measurements in **mm**,
  plus min, max, mean, count
- **Oxygen** → converted to a fraction between 0 and 1
  - `"21%"` → `0.21`
- **Ignition energy** → computed as:

$$E_{\text{ignition}} = P_{\text{ignition}} \times t_{\text{ignition}}$$

- **Log gravity** → $\log_{10}(\text{gravity\_g})$ is added as a feature
  because gravity spans many orders of magnitude and a log scale
  makes it easier for models to learn from

### Post-ignition leakage columns are removed
Flame Spread Rate, Flame Length, HRR, Smoke — these only exist
*after* ignition happens, so including them would be cheating.

### The two feature sets
Every model is run twice — once with **all features**, once with
only **physics features** (no apparatus-specific columns like
rig name or facility). This tests whether the model learned real
physics or just memorized the lab setup.

| Feature Set | Columns included |
|---|---|
| `physics` | O₂ fraction, pressure, flow, gravity, log-gravity, fuel/gas thermal properties, geometry, diluent, flow direction |
| `all` | Everything above + ignition power/time/energy, sample dimensions, internal geometry, facility, rig |

**Total engineered numeric features: 37**
**Total categorical features: 7**

---

## STEP 2 — Generate Frozen Splits (`fable_splits.py`)

### Why freeze the splits?
If you re-generate splits every time you run, different runs are
not comparable. Freezing them once means every model sees exactly
the same train/test rows.

### The 4 split protocols

#### Protocol A: `interpolation_holdout`
A standard 80/20 random train/test split, **stratified** (keeps
the same ~76% ignition / 24% no-ignition ratio in both halves).
This is the *easiest* protocol — the model will likely have seen
similar experiments at training time.

#### Protocol B: `interpolation_stratified`
5-fold cross-validation with stratification. The data is split into
5 equal chunks; in each fold, 4 chunks train and 1 tests.
This is run with **3 different random seeds**, giving 15 outer folds.

#### Protocol C: `extrapolation_grouped`
**The scientifically important one.** All rows from the same paper
are kept *together* — either all in train or all in test.
This mimics the real question: *can the model predict ignition in
a paper it has never read?*
Uses `StratifiedGroupKFold` with 5 folds × 3 seeds = **15 outer folds**.

#### Protocol D: `lopo` (Leave-One-Paper-Out)
The strictest test. One paper is completely held out, the model
trains on the other 92, then predicts on the held-out paper.
This is repeated for every paper → **93 outer folds**.
There is no randomness: fold `k` always holds out paper `k`.

### Integrity checks
After generation, every split is validated:
- No row appears in both train and test
- Every row appears in exactly one partition
- For grouped/LOPO: no paper appears in both train and test

---

## STEP 3 — Define the Models (`fable_models.py`)

There are **5 model families**, each run with 2–3 candidate
configurations = **15 total candidates**.

---

### Model 1: Decision Tree 🌳

A flowchart of yes/no questions on the features.

**Search space (hyperparameters to tune):**

| Parameter | Values tried |
|---|---|
| `max_depth` | 3, 5, 7, 10, None (unlimited) |
| `min_samples_leaf` | 1, 2, 5, 10, 20 |
| `criterion` | `gini`, `entropy`, `log_loss` |

**Criterion formulas:**

Gini impurity for a node with class proportions $p_0, p_1$:

$$\text{Gini} = 1 - p_0^2 - p_1^2$$

Entropy (information gain):

$$H = -p_0 \log_2 p_0 - p_1 \log_2 p_1$$

**Candidates:** `decision_tree_all`, `decision_tree_physics`

---

### Model 2: XGBoost 🚀

An ensemble of many shallow decision trees built *sequentially*,
where each new tree tries to fix the mistakes of the previous ones.

**Core idea:**

$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(x)$$

where $\eta$ is the learning rate and $f_t$ is the new tree.

**Search space:**

| Parameter | Values tried | What it controls |
|---|---|---|
| `n_estimators` | 200, 400, 600, 800 | Number of trees |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1 | How big each step is |
| `max_depth` | 2, 3, 4, 5, 6 | How deep each tree can be |
| `min_child_weight` | 1, 2, 4, 8 | Min samples to split a node |
| `subsample` | 0.6, 0.8, 1.0 | % of rows used per tree |
| `colsample_bytree` | 0.6, 0.8, 1.0 | % of features used per tree |
| `reg_lambda` | 0.5, 1.0, 2.0, 5.0 | L2 regularization (penalizes large weights) |
| `reg_alpha` | 0.0, 0.1, 0.5 | L1 regularization (encourages sparse weights) |
| `gamma` | 0.0, 0.1, 0.5 | Min gain required to make a split |

**XGBoost has 7 candidate variants:**

| Candidate ID | Feature set | Loss function | Weighting |
|---|---|---|---|
| `xgb_all_unweighted` | all | logistic | none |
| `xgb_physics_unweighted` | physics | logistic | none |
| `xgb_all_class_paper_weighted` | all | logistic | class + paper |
| `xgb_physics_class_paper_weighted_monotone_o2` | physics | logistic | class + paper + O₂ monotone |
| `xgb_all_focal_class_paper_weighted` | all | **focal loss** | class + paper |
| `xgb_physics_focal_class_paper_weighted` | physics | **focal loss** | class + paper |
| `xgb_physics_paper_bagging` | physics | logistic | class + paper + 10 bootstrap ensembles |

**What is focal loss?**
Standard logistic loss treats every sample equally. Focal loss
down-weights easy examples so the model focuses on hard borderline cases:

$$\mathcal{L}_{\text{focal}} = -(1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's probability for the correct class and
$\gamma = 2.0$. When the model is already confident ($p_t \approx 1$),
$(1-p_t)^2 \approx 0$ and the loss is nearly zero — hard examples
get all the gradient signal.

**What is the monotone O₂ constraint?**
For the physics candidate, we enforce that increasing oxygen
concentration must never decrease $P(\text{ignition})$.
This is physically obvious — more oxygen → easier to ignite.
It is enforced by telling XGBoost: `monotone_constraints = +1`
on the oxygen feature.

**What is paper bagging?**
Instead of training one XGBoost model, we train **10 models**,
each on a different bootstrap resample of the *papers* (not rows).
The final prediction is the average of all 10:

$$\hat{p} = \frac{1}{10} \sum_{k=1}^{10} \hat{p}_k$$

This reduces variance caused by the specific papers that happen
to be in the training set.

---

### Model 3: K-Nearest Neighbors (KNN) 🗺️

To predict for a new experiment, find the K most similar
experiments in the training set and take a majority vote.

**Distance formula — Manhattan ($p=1$):**

$$d(a, b) = \sum_i |a_i - b_i|$$

**Distance formula — Euclidean ($p=2$):**

$$d(a, b) = \sqrt{\sum_i (a_i - b_i)^2}$$

**Search space:**

| Parameter | Values tried | What it controls |
|---|---|---|
| `n_neighbors` | 5, 9, 15, 25, 40, 60 | How many neighbors to consult |
| `weights` | `uniform`, `distance` | Uniform: all K neighbors vote equally. Distance: closer neighbors vote more |
| `p` | 1, 2 | 1 = Manhattan distance, 2 = Euclidean |
| `leaf_size` | 15, 30, 60 | Speed/memory trade-off for the KD-tree |

**Candidates:** `knn_all`, `knn_physics`

---

### Model 4: Multi-Layer Perceptron (MLP) 🧬

A neural network with hidden layers. Each neuron computes:

$$z = \text{activation}\!\left(\sum_j w_j x_j + b\right)$$

**Search space:**

| Parameter | Values tried | What it controls |
|---|---|---|
| `hidden_layer_sizes` | (32,), (64,), (64,32), (128,64) | Network architecture |
| `alpha` | 1e-5, 1e-4, 1e-3, 1e-2 | L2 weight regularization |
| `learning_rate_init` | 0.0001, 0.0003, 0.001, 0.003 | Step size for gradient descent |
| `activation` | `relu`, `tanh` | Non-linearity applied at each neuron |
| `batch_size` | 32, 64, 128 | Samples per gradient update |

The network uses **early stopping** (stops training when validation
loss doesn't improve for 25 consecutive epochs, max 600 epochs).

**Candidates:** `mlp_all`, `mlp_physics`

---

### Model 5: Support Vector Machine (SVM) 🗡️

Finds the widest possible margin between igniting and
non-igniting experiments in a high-dimensional feature space.

**Kernel trick (RBF):**

$$K(x, x') = \exp\!\left(-\gamma \|x - x'\|^2\right)$$

This maps data into a space where it is linearly separable
without ever computing the high-dimensional coordinates explicitly.

Since SVM doesn't naturally output probabilities, a
**Platt sigmoid calibration** is wrapped around it (3-fold CV):

$$P(\text{ignition}) = \frac{1}{1 + e^{A \cdot f(x) + B}}$$

**Search space:**

| Parameter | Values tried | What it controls |
|---|---|---|
| `C` | 0.1, 0.3, 1.0, 3.0, 10.0, 30.0 | Penalty for misclassification (larger = tighter fit) |
| `gamma` | `scale`, 0.01, 0.03, 0.1, 0.3 | Width of the RBF kernel |
| `shrinking` | True, False | Optimization speed heuristic |

**Candidates:** `svm_all`, `svm_physics`

---

## STEP 4 — Sample Weighting (`fable_common.py`)

### The problem
Some papers contribute 300 rows, others only 4. Without
correction, models effectively memorize the large papers.

### Paper weighting strategies
Each row gets a weight based on how many rows its paper contributes.
If paper $p$ has $n_p$ rows, the weight for each row is:

| Strategy | Formula | Effect |
|---|---|---|
| `none` | $w = 1$ | All rows equal |
| `inverse` | $w = 1 / n_p$ | Large papers heavily down-weighted |
| `sqrt` | $w = 1 / \sqrt{n_p}$ | Moderate down-weighting (used by most candidates) |
| `log` | $w = 1 / \ln(1 + n_p)$ | Gentle down-weighting |
| `effective` | $w = (1-\beta) / (1-\beta^{n_p})$ | Class Balanced Loss formula |

All weights are normalized so their mean = 1.

### Class weighting
Because 76% of rows are "ignition = Yes", the model can score 76%
accuracy by always predicting Yes. Class weighting multiplies the
paper weight by an additional factor:

$$w_{\text{class}} = \frac{N}{2 \cdot N_k}$$

where $N_k$ is the count of the row's class. This makes ignition
and non-ignition examples contribute equally.

---

## STEP 5 — Hyperparameter Tuning (Nested Search) (`fable_search.py`)

### The core idea
For every outer training fold, we need to pick the best
hyperparameters — **without peeking at the test fold**.
We do this by splitting the outer training set into **inner folds**
and trying many configurations.

### The inner loop

For each outer fold:
1. Generate 3 inner `StratifiedGroupKFold` splits of the training data
   (papers are kept together here too, for the grouped/LOPO protocols)
2. Sample **40 random configurations** from the search space
3. For each configuration:
   - Train on 2 inner folds, predict on the 3rd → repeat 3 times
   - Record the **out-of-fold ROC-AUC** for each inner fold
4. Pick the configuration with the highest mean inner ROC-AUC
5. Train a fresh model with those hyperparameters on the **full** outer training fold
6. Predict on the outer test fold → these are the official predictions

### Threshold optimization
Standard classifiers output a probability $\hat{p} \in [0,1]$.
To get a hard Yes/No prediction, you need a threshold $\tau$:

$$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \geq \tau \\ 0 & \text{otherwise} \end{cases}$$

Instead of always using $\tau = 0.5$, the pipeline finds the
**optimal threshold for 4 different objectives**, using only the
inner out-of-fold predictions (never the test set):

| Threshold name | What it maximizes |
|---|---|
| `mcc` | Matthews Correlation Coefficient |
| `f1` | F1 score (harmonic mean of precision & recall) |
| `balanced_accuracy` | Mean of sensitivity and specificity |
| `youden_j` | Sensitivity + Specificity − 1 |

**Matthews Correlation Coefficient:**

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

This ranges from −1 to +1 and is considered the most honest single
metric for binary classification under class imbalance.

### LOPO special case
For a Leave-One-Paper-Out fold, running 40 configurations × 3 inner
folds is wasteful because we already know which hyperparameters work
from the grouped CV. Instead:
- Look at all `extrapolation_grouped` outer folds whose training set
  **did not include** the held-out paper
- Take the **most frequently selected** (modal) hyperparameters
- Freeze those and train with 1 configuration, 1 iteration

---

## STEP 6 — Metrics (`fable_search.py`, `fable_evaluate.py`)

Every fold records these metrics at the chosen thresholds:

| Metric | Formula | Plain English |
|---|---|---|
| **ROC-AUC** | Area under TPR vs FPR curve | How well does the model *rank* igniting vs non-igniting? 1.0 = perfect, 0.5 = random |
| **PR-AUC** | Area under Precision vs Recall curve | Better metric when classes are imbalanced |
| **Brier score** | $\frac{1}{n}\sum(\hat{p}_i - y_i)^2$ | Mean squared error of the probabilities (lower = better) |
| **MCC** | See above | Best overall single number |
| **F1** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Balance between catching ignitions and not over-predicting |
| **Balanced accuracy** | $\frac{1}{2}(TPR + TNR)$ | Mean recall across both classes |
| **Sensitivity** | $TP / (TP + FN)$ | Fraction of real ignitions caught |
| **Specificity** | $TN / (TN + FP)$ | Fraction of real non-ignitions correctly rejected |

---

## STEP 7 — Statistical Validation (`fable_evaluate.py`)

After all folds are done, the pipeline runs extra statistical checks.

### Bootstrap confidence intervals
To know how uncertain the metrics are, the pipeline does 2,000
bootstrap resamples:
1. Randomly resample with replacement (for grouped protocols, resample
   whole papers, not individual rows)
2. Recompute ROC-AUC, PR-AUC, Brier score on the resampled set
3. Repeat 2,000 times → take the 2.5th and 97.5th percentiles as the
   **95% confidence interval**

### Wilcoxon signed-rank test
To check if one model is *genuinely* better than another, a
paired statistical test is run on the fold-level metrics:
- For each pair of models A and B, compute $\delta = \text{AUC}_A - \text{AUC}_B$ on every shared fold
- Run Wilcoxon signed-rank test: $H_0$ = the median difference is zero
- A p-value < 0.05 suggests the difference is statistically significant

---

## STEP 8 — Model Selection (`fable_select.py`)

A candidate wins if it scores best on the **primary metric** for
its target protocol. Ties are broken in this order:

**For extrapolation (the scientifically important one):**
1. Highest ROC-AUC on `extrapolation_grouped`
2. Tie? → Higher PR-AUC
3. Still tied? → Lower uncertainty (narrower confidence interval)
4. Still tied? → Physics-only feature set preferred
5. Still tied? → Simpler model preferred

**Simplicity order:** Decision Tree < KNN < SVM < XGBoost < MLP

A candidate also passes a **robustness check**: it must achieve
reasonable ROC-AUC on the LOPO protocol too (not just
grouped CV). Candidates that only look good on one protocol are rejected.

**Instability filter:** Any candidate whose fold-level ROC-AUC has
a standard deviation > 0.15 is disqualified (too inconsistent to trust).

---

## STEP 9 — Refit and Deploy (`fable_refit.py`, `fable_predict.py`)

Once the champion model is selected:
1. It is retrained on **all available data** (no held-out test set)
   using the modal hyperparameters from the grouped outer folds
2. Saved as a `.joblib` file
3. `fable_predict.py` loads this file and can score new experiments:
   input a CSV row → output $P(\text{ignition})$ + a hard Yes/No per threshold

---

## 🗺️ Full Pipeline Diagram

```
Raw database (CSV)
       │
       ▼
fable_common.py ──► Clean + engineer 44 features ──► Deduplicate rows
       │
       ▼
fable_splits.py ──► Generate 4 frozen protocols (ONCE, never re-run)
       │           ├─ interpolation_holdout    (3 × 1 fold  =  3 folds)
       │           ├─ interpolation_stratified (3 × 5 folds = 15 folds)
       │           ├─ extrapolation_grouped    (3 × 5 folds = 15 folds)
       │           └─ lopo                     (1 × 93 folds = 93 folds)
       │
       ▼
fable_evaluate.py ── For each of 15 candidates × each outer fold:
       │           ├─ fable_search.py: 40 configs × 3 inner folds → best HP
       │           ├─ fable_models.py: train final model on outer train set
       │           ├─ Predict on outer test set → probabilities
       │           ├─ Optimize 4 thresholds on inner OOF predictions
       │           └─ Record all metrics + predictions
       │
       ▼
fable_evaluate.py ── Post-processing:
       │           ├─ 2,000-iteration bootstrap CIs
       │           ├─ Wilcoxon paired tests (all model pairs)
       │           └─ Per-paper breakdown
       │
       ▼
fable_select.py ──► Apply selection policy → pick 1 extrapolation champion
       │
       ▼
fable_refit.py ──► Train champion on all data → save model
       │
       ▼
fable_predict.py ──► Score new experiments
```

---

## 🔑 Key Takeaways for an ML Beginner

1. **Train/test split is not enough** — when data comes from papers,
   you must keep whole papers together or your metrics are misleadingly optimistic.

2. **Nested cross-validation** means the hyperparameter search is
   also validated fairly — tuning on the test set would inflate your scores.

3. **Threshold matters** — 0.5 is rarely the best threshold.
   This pipeline picks the threshold that maximizes the metric you care about.

4. **More data ≠ more information** — a paper with 300 rows
   is not 300× more informative than one with 4 rows.
   Sample weighting corrects for this.

5. **One number is never enough** — reporting ROC-AUC, PR-AUC,
   MCC, Brier score, and per-paper metrics together paints a
   much more honest picture than a single accuracy score.
```