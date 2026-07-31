# How the Ignition Classifier Pipeline Works
### A plain-English guide for ML beginners

---

## The big picture

The goal is simple: given a set of experimental conditions such as oxygen level, pressure,
material, gravity, and geometry, predict whether a material will ignite. This is a binary
classification problem.

The dataset is organized by paper, so the key challenge is to evaluate whether a model generalizes
to new papers rather than memorizing familiar experimental campaigns. The pipeline keeps the workflow
reproducible by freezing the data splits, fixing the feature list, and persisting every evaluation
artifact.

---

## Step 1 — Load and clean the data (`fable_common.py`)

The raw database is a CSV file with a category row followed by the real header row. The loader parses
units, normalizes text, validates required source columns, and drops six columns that must never be
used as features: Material of Sample, Rig Name, Internal Geometry, Internal Dimensions, Facility,
and Info.

Every model uses the same fixed physics feature list: 40 numeric features and 5 categorical features.
There are no feature sets anymore, no `log10_gravity_g`, no `internal_dim_*` variables, and none of
the six dropped columns above. The feature list covers fuel and gas properties, flow, pressure,
oxygen, gravity, ignition conditions, geometry-derived measurements, and five categorical descriptors
for geometry, diluent, flow direction, gravity regime, and ignition method.

The engineered feature count is therefore 45 total features: 40 numeric and 5 categorical.
Missing numeric values are handled by the downstream models, and categorical values are encoded with
unknown-safe one-hot pipelines.

The target is ignition, recorded as yes or no. The loader also removes duplicates using the full
feature list plus the paper identity, so repeated rows do not inflate the evidence count.

### A few feature examples

Here are some of the engineered numeric variables used by every model:

- `oxygen_fraction`
- `pressure_kpa`
- `flow_velocity_mm_s`
- `gravity_g`
- `ignition_energy_j`
- `half_thickness_m`
- `sample_dim_mean_mm`
- `core_diameter_mm`
- `insulation_thickness_mm`

The categorical variables are:

- `geometry_cat`
- `diluent_cat`
- `flow_direction_cat`
- `gravity_regime_cat`
- `ignition_method_cat`

### Example formula

Ignition energy is computed from ignition power and ignition time:

\[
E_{\text{ignition}} = P_{\text{ignition}} \times t_{\text{ignition}}
\]

This gives one derived feature that helps summarize how much energy was delivered during ignition.

---

## Step 2 — Generate frozen splits (`fable_splits.py`)

The split files are created once and then treated as fixed inputs for the rest of the pipeline. This
is important because different random splits would make model comparisons unfair.

There are four split protocols:

- `interpolation_holdout`: a stratified holdout split for a quick interpolation check.
- `interpolation_stratified`: repeated stratified row-level cross-validation.
- `extrapolation_grouped`: repeated grouped cross-validation where whole papers stay together.
- `lopo`: leave-one-paper-out, which holds out one paper at a time.

Together these protocols cover both easy interpolation and the harder question of transfer to unseen
papers. The grouped protocols ensure that the same paper never appears in both train and test.

### Why grouped splits matter

If rows from the same paper appear in both training and testing, the score can look unrealistically
good because the model has already seen very similar experiments. Grouped splits make the benchmark
much closer to the real scientific question: can the model handle a new paper?

---

## Step 3 — Define the candidates (`configs/candidates.yaml`)

The pipeline evaluates 26 candidates across five model families: XGBoost, decision tree, KNN, MLP,
and SVM. Each candidate is a fixed combination of family, weighting policy, and any family-specific
settings.

The candidate IDs are:

- XGBoost: `xgb_baseline`, `xgb_paper_sqrt`, `xgb_paper_inverse`, `xgb_class_only`, `xgb_weighted_sqrt`, `xgb_weighted_inverse`, `xgb_focal_sqrt`, `xgb_focal_inverse`
- Decision tree: `dt_baseline`, `dt_weighted_sqrt`, `dt_weighted_inverse`
- KNN: `knn_baseline`, `knn_weighted_sqrt`, `knn_weighted_inverse`, `knn_distance_sqrt`, `knn_distance_inverse`
- MLP: `mlp_baseline`, `mlp_weighted_sqrt`, `mlp_weighted_inverse`, `mlp_focal_sqrt`, `mlp_focal_inverse`
- SVM: `svm_baseline`, `svm_weighted_sqrt`, `svm_weighted_inverse`, `svm_linear_sqrt`, `svm_linear_inverse`

That fixed candidate list is what the evaluation step searches over. Older IDs such as
`xgb_all_unweighted` and `knn_physics` do not belong to the current pipeline.

### What “candidate” means

A candidate is not just a model family. It is a specific modeling recipe: for example, an XGBoost
model with inverse paper weighting, or an SVM with a linear kernel and square-root paper weighting.
This lets the benchmark compare not only algorithms, but also weighting choices and objective
variants.

---

## Step 4 — Nested search and scoring (`fable_search.py`, `fable_evaluate.py`)

For every outer fold, the pipeline tunes hyperparameters only inside the training data for that fold.
This is nested search: the outer test fold stays untouched until the very end.

The main search objective is ROC-AUC, and the pipeline also records PR-AUC, Brier score, MCC, F1,
balanced accuracy, sensitivity, specificity, and precision. Four separate decision thresholds are
frozen from inner out-of-fold predictions: MCC, F1, balanced accuracy, and Youden J.

### Threshold rule

A classifier produces a probability \(\hat{p}\). To convert it into a yes/no prediction, the pipeline
uses a threshold \(\tau\):

\[
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge \tau \\
0 & \text{if } \hat{p} < \tau
\end{cases}
\]

Instead of forcing one universal threshold such as 0.5, the pipeline keeps four different threshold
policies because different scientific goals reward different trade-offs.

### LOPO special handling

LOPO is treated specially. Instead of re-tuning everything from scratch for each held-out paper, the
pipeline reuses the modal hyperparameters discovered by the grouped outer folds when the same paper
was absent from training. This reduces redundancy while preserving the no-leakage rule.

---

## Step 5 — Weighting and model behavior

The pipeline uses paper weighting to avoid letting large papers dominate the learning signal. The
common paper strategies in the candidate registry are `none`, `sqrt`, and `inverse`, and class
weighting can also be turned on for selected candidates.

If paper \(p\) contributes \(n_p\) rows, a typical square-root paper weighting rule is:

\[
w_p = \frac{1}{\sqrt{n_p}}
\]

This reduces the influence of very large papers without making every paper contribute exactly the
same amount.

### Family differences

Different model families handle weighting differently:

- XGBoost and decision trees can consume sample weights directly.
- Calibrated SVM can use supported fit-time weighting.
- KNN and MLP cannot use sample weights in the same way, so the pipeline uses deterministic weighted resampling for them.

These family-specific mechanics are part of what makes the benchmark fair across very different model
types.

---

## Step 6 — Selection and refit (`fable_select.py`, `fable_refit.py`)

The selection step applies a fixed policy to the persisted evaluation artifacts and chooses one
champion for interpolation and one for extrapolation. It does not train any new models.

The refit step then retrains the chosen configuration on all labeled data for deployment. This
creates the deployable artifacts after the evaluation phase is complete.

### Why there are two champions

Interpolation and extrapolation answer different questions. A model can be very strong when test rows
look similar to rows already seen during training, yet weaker when a completely new paper is held out.
That is why the pipeline keeps separate champions rather than forcing a single winner for both tasks.

---

## Step 7 — Per-family output directories

Evaluation artifacts are stored by family rather than under older `model_*` directory names. The
per-family directory layout is based on the keys in `MODEL_FAMILIES`:

- `results/evaluation/xgb/`
- `results/evaluation/svm/`
- `results/evaluation/knn/`
- `results/evaluation/dt/`
- `results/evaluation/mlp/`

These directories hold family-level outputs such as summary metrics, predictions, integrity checks,
and comparison files. This structure is also what `fable_select.py` and the updated `fable_report.py`
now expect.

---

## Step 8 — Report generation (`fable_report.py`)

The report script builds publication tables and figures from the saved evaluation artifacts. It reads
from the per-family output directories, summarizes the selected champions, merges integrity logs, and
creates the report README and plots.

The generated report README uses `Microgravity_Database.csv` in the project folder and points to
family-based integrity files such as `../evaluation/<family>/integrity_checks.json`.

---

## Pipeline map

```text
Raw database (CSV)
       │
       ▼
fable_common.py ──► Clean data and engineer 45 fixed features
       │
       ▼
fable_splits.py ──► Create 4 frozen split protocols
       │
       ▼
fable_evaluate.py ──► Tune and score 26 candidates across 5 families
       │
       ▼
results/evaluation/<family>/
       ├─ summary metrics
       ├─ predictions and thresholds
       ├─ integrity checks
       └─ comparisons
       │
       ▼
fable_select.py ──► Apply the selection policy
       │
       ▼
fable_refit.py ──► Refit the selected champions
       │
       ▼
fable_report.py ──► Build tables, plots, and the final report
```

---

## Why the pipeline is structured this way

The design keeps the science honest. By freezing splits, holding out papers together, and evaluating
many candidates under the same conditions, the pipeline can compare models fairly.

The fixed physics feature list means every candidate sees the same inputs, which makes the benchmark
cleaner and easier to interpret. The per-family output directories also make the persisted artifacts
simpler to inspect and reuse.

---

## A simple mental model

Think of the pipeline as a funnel. First it cleans the raw database, then it generates safe split
protocols, then it searches the 26 candidates, then it selects the best champions, and finally it
refits and reports the result.

That order matters because each later stage depends on the previous one being frozen and reproducible.