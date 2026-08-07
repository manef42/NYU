# Ignition Classifier Pipeline — Full Guide


If you only need commands, use [`README.md`](README.md). This document explains **what each stage does**, **why it exists**, and **what files it produces**.

---

## 1. Big picture

We predict **binary ignition** from experimental conditions in `Microgravity_Database.csv`.

The scientific risk is **leakage**: if rows from the same paper appear in both train and test, scores look too good because the model has already seen a very similar campaign. The pipeline therefore evaluates two questions separately:

1. **Interpolation** — can we predict similar row-level conditions?
2. **Extrapolation** — can we transfer to a held-out paper?

Everything after data loading is built to keep those questions honest: freeze splits once, tune only inside each outer training fold, score the outer test fold once, select champions from saved metrics (no re-training), then optionally refit for deployment.

```text
Microgravity_Database.csv
        │
        ▼
 fable_common.py          load, clean, engineer 37 features (library)
        │
        ▼
 fable_splits.py          freeze 4 split protocols
        │
        ▼
 fable_evaluate.py        for each candidate × outer fold:
        │                   └─ fable_search.py (nested tune on train)
        │                   └─ fable_models.py (fit on train, score test)
        ▼
 results/evaluation/<family>/
        │
        ▼
 fable_select.py          apply selection_policy.yaml → selection.json
        │
        ▼
 fable_refit.py           train deployable interpolation + extrapolation champions
        │
        ▼
 fable_report.py          tables, figures, XGBoost explainability
        │
        ▼
 fable_predict.py         (later) score new CSVs with an artifact
```

---

## 2. `fable_common.py` — data loading and shared utilities

**Not a CLI.** Imported by every other script.

### Responsibilities

1. **Read the CSV** (`read_raw`)
   - Skips the first (category) row; second row is the header.
   - Tries encodings: `utf-8`, `utf-8-sig`, `cp1252`, `latin-1`.

2. **Drop disallowed columns** before engineering:
   - Material of Sample, Rig Name, Internal Geometry, Internal Dimensions, Facility, Info.

3. **Build paper identity**
   - Prefer canonical DOI → `paper_id = "doi::<normalized>"`.
   - Else canonical citation → `"citation::<normalized>"`.
   - Never use raw DOI strings as CV groups.

4. **Engineer a fixed feature manifest** (`DATA_VERSION = "fable-data-v5"`)

   **32 numeric features** (physics quantities; SI / natural units as stored in the DB):

   | Group | Features |
   |---|---|
   | Fuel | `fuel_density_kg_m3`, `fuel_k_w_mk`, `fuel_cp_j_kgk`, `fuel_pyrolysis_t_k`, `fuel_alpha_m2_s`, `fuel_volumetric_heat_capacity_j_m3k` |
   | Core | `core_density_kg_m3`, `core_k_w_mk`, `core_cp_j_kgk`, `core_volumetric_heat_capacity_j_m3k` |
   | Gas / transport | `oxygen`, `gas_molar_mass`, `gas_cp_j_kgk`, `gas_k_w_mk`, `gas_density_kg_m3`, `gas_alpha_m2_s`, `gas_nu_m2_s`, `reynolds`, `peclet`, `prandtl`, `thermal_diffusion_time_s` |
   | Conditions | `pressure_pa`, `flow_velocity_m_s`, `flow_speed_abs_m_s`, `gravity_g`, `ignition_power_w`, `ignition_time_s` |
   | Geometry | `half_thickness_m`, `characteristic_length_m`, `sample_dim_1_m`, `sample_dim_2_m`, `sample_dim_3_m` |

   **5 categorical features:**

   - `geometry_cat` — Wire / Flat / Cylindrical / Spherical (mapped from free text)
   - `diluent_cat` — diluent string (missing → `"Unknown"`)
   - `flow_direction_cat` — Coflow / Counterflow / Quiescent from signed flow
   - `gravity_regime_cat` — Microgravity / Partial / Earth / Hyper·Unknown from `gravity_g`
   - `ignition_method_cat` — Open Flame / Radiative Heater / Discharge / Wire·Coil

   There is **no** `feature_set` switch and no post-outcome flame/spread/HRR/smoke features.

5. **Target**
   - Column `Ignition` → `ignition_binary` ∈ {0, 1}.
   - Missing/unrecognized labels raise when `require_target=True` (training/eval). They are allowed for inference.

6. **Deduplicate** on full feature list + label + `paper_id` (keeps first).

7. **Validation report** — row counts, SHA-256, missingness, label counts, feature availability.

8. **Weighting helpers**
   - `paper_weights(papers, strategy)` with `none`, `sqrt` (\(1/\sqrt{n_p}\)), `inverse` (\(1/n_p\)), plus `effective` / `log`.
   - `combined_weights(...)` multiplies paper weights by optional balanced class weights, then renormalizes.

9. **Torch / Slurm helpers** — device selection, CUDA cache clear, `SLURM_CPUS_PER_TASK` worker count.

10. **`MODEL_FAMILIES`** — maps output folder names to family strings, in Slurm array order:

    ```text
    xgb → xgboost
    svm → svm
    knn → knn
    dt  → decision_tree
    mlp → mlp
    ```

### Key API

```python
df, report = load_data("Microgravity_Database.csv", require_target=True)
numeric, categorical = feature_lists()  # 32 + 5
```

---

## 3. `fable_models.py` — model families and preprocessing

**Not a CLI.** Defines how each family is built and fitted.

### `MODEL_REGISTRY`

Each family has:

- A search space (sampled by `fable_search.py`)
- Whether it accepts `sample_weight`
- Whether numerics are scaled
- Whether numerics use native missingness (XGBoost only)

| Family | Sample weight | Scale numerics | Native NaN | Notes |
|---|---|---|---|---|
| `xgboost` | Yes | No | Yes | Logistic or focal objective; CPU hist |
| `decision_tree` | Yes | No | No (median impute) | Depth / leaf / criterion tuned |
| `knn` | No → resample | Yes | No | `n_neighbors`, `weights`, `p`, … |
| `mlp` | No → resample | Yes | No | PyTorch MLP; optional focal loss |
| `svm` | Yes (calibrated path) | Yes | No | RBF default; linear via fixed `kernel` |

### `FableModel`

Public interface used everywhere:

```text
model = make_model(candidate_dict, params, seed)
model.fit(X_df, y, papers)
proba = model.predict_proba(X_df)   # class-1 probability in [0, 1]
```

Preprocessing (fit on training fold only):

- Numeric: optional median impute + optional `StandardScaler`
- Categorical: constant `"Unknown"` + `OneHotEncoder(handle_unknown="ignore")`

Weighting policy:

- Families with `supports_sample_weight=True` pass weights into `.fit(..., sample_weight=...)`.
- KNN / MLP **cannot** take sample weights → deterministic probability-proportional bootstrap of the training fold (`_resample`). No family silently ignores the candidate’s weighting flags.

Optional XGBoost extras (candidate YAML / fixed params): focal objective, monotone oxygen (legacy hook), paper bagging.

---

## 4. `fable_splits.py` — freeze all partitions

**CLI.** Creates immutable train/test memberships used by every later stage.

### Command

```bash
python fable_splits.py \
  --data Microgravity_Database.csv \
  --out results/splits \
  --n-seeds 3 \
  --n-group-folds 5 \
  --n-row-folds 5 \
  --base-seed 42
```

| Flag | Default | Meaning |
|---|---:|---|
| `--data` | required | Path to CSV |
| `--out` | required | Output directory |
| `--n-seeds` | 3 | How many repeated seeds (`base`, `base+1`, …) |
| `--n-group-folds` | 5 | Folds for grouped extrapolation CV |
| `--n-row-folds` | 5 | Folds for stratified interpolation CV |
| `--base-seed` | 42 | First seed |

### Four protocols

| Protocol | What it does |
|---|---|
| `interpolation_holdout` | Stratified 80/20 holdout per seed (quick check) |
| `interpolation_stratified` | `StratifiedKFold` over rows, repeated over seeds |
| `extrapolation_grouped` | `StratifiedGroupKFold` with `groups=paper_id` |
| `lopo` | Leave-One-Paper-Out: each unique paper is a test fold once |

### Integrity checks

For every `split_id`:

- No row in both train and test
- Train ∪ test covers all rows
- For grouped protocols and LOPO: **no paper** in both partitions

### Outputs

```text
results/splits/
├── split_assignments.csv
├── split_assignments.parquet
├── split_assignments.joblib
├── splits_metadata.json          # SHA-256, counts, protocol sizes
└── data_validation_report.json
```

Each assignment row: `split_id`, `protocol`, `seed`, `fold`, `partition` (`train`|`test`), `row_id`, `paper_id`.

**Intern tip:** regenerate splits only when the dataset or feature/identity logic changes. Reusing the same split files keeps model comparisons fair.

---

## 5. `configs/` — candidates and selection policy

### `candidates.yaml` (version 5)

Lists all **26** candidates. Each entry has:

- `candidate_id`
- `model_family`
- `paper_weight` (`none` | `sqrt` | `inverse`)
- `class_weight` (bool)
- optional `fixed_params` (e.g. `objective_variant: focal`, `weights: distance`, `kernel: linear`)

Design rationale (baselines, ablation of paper vs class weight, focal without double-counting class weight, linear SVM, distance KNN): see [`configs/candidates.md`](configs/candidates.md).

### `selection_policy.yaml`

Consumed only by `fable_select.py`. Highlights:

- Candidates must pass integrity, complete folds, valid probabilities.
- Instability gate: fold ROC-AUC SD ≤ `0.15`.
- Tie tolerance on primary metric: `0.01`.
- **Interpolation champion:** primary protocol `interpolation_stratified`, metric `roc_auc`.
- **Extrapolation champion:** primary protocol `extrapolation_grouped`, metric `roc_auc`, plus mandatory successful `lopo` robustness.
- Tie-breakers in order: higher PR-AUC → lower uncertainty → simpler family (`decision_tree` < `knn` < `svm` < `xgboost` < `mlp`).
- Refit hyperparameters: **modal** outer-fold selected config, lexical tie-break.
- Four operational thresholds are kept; none is declared universally best.

---

## 6. `fable_search.py` — nested search and thresholds

**Not a CLI.** Called once per outer training fold inside `fable_evaluate.py`.

### Nested search (`nested_search`)

Given only the **outer train** rows:

1. Build **inner** CV splits:
   - Interpolation protocols → `StratifiedKFold`
   - Extrapolation / LOPO → paper-disjoint `StratifiedGroupKFold` (must keep both classes in every fold)
2. Sample up to `iterations` unique hyperparameter configs from the family’s search space (plus candidate `fixed_params`).
3. Fit each config on each inner train fold; score inner validation (parallel for most families; SVM uses 1 job).
4. Rank configs by mean inner **ROC-AUC** (then PR-AUC, then lower Brier).
5. Freeze **four thresholds** from the winning config’s full inner OOF probabilities.

### Threshold policies (`optimize_thresholds`)

For probability \(\hat{p}\) and threshold \(\tau\):

\[
\hat{y} = \mathbf{1}[\hat{p} \ge \tau]
\]

| Name | Objective on validation OOF |
|---|---|
| `mcc` | Maximize Matthews correlation |
| `f1` | Maximize F1 |
| `balanced_accuracy` | Maximize \((sens + spec)/2\) |
| `youden_j` | Maximize \(sens + spec - 1\) |

Ties prefer thresholds closer to 0.5. These thresholds are then applied to the **outer** test fold without re-optimizing on test labels.

### Metrics recorded

ROC-AUC, PR-AUC, Brier, MCC, F1, balanced accuracy, sensitivity, specificity, precision.

---

## 7. `fable_evaluate.py` — the heavy benchmark runner

**CLI.** Only place that runs the full candidate × protocol × fold loop.

### Command

```bash
python fable_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation \
  --search-iterations 40 \
  --inner-group-folds 3 \
  --bootstrap-iterations 2000
```

| Flag | Default | Meaning |
|---|---:|---|
| `--data` | required | Labeled CSV |
| `--splits` | required | Directory from `fable_splits.py` |
| `--config` | required | `candidates.yaml` |
| `--out` | required | Root or family directory (see below) |
| `--search-iterations` | 40 | Random configs per outer train fold |
| `--inner-group-folds` | 3 | Inner CV folds |
| `--bootstrap-iterations` | 2000 | Used when writing bootstrap CIs |
| `--model` | all families | Restrict to one family |
| `--candidate-id` | all in family | Restrict to one YAML candidate |
| `--base-seed` | 42 | Reserved / documented seed |

### Behavior per candidate × outer fold

1. Load train/test row IDs from frozen assignments.
2. Run `nested_search` on **train only**.
3. Fit `make_model(...)` on full outer train with selected params.
4. Predict outer test probabilities; apply the four frozen thresholds.
5. Record fold metrics, per-paper metrics, search history, inner OOF preds.
6. Catch failures into `integrity_checks.json` (do not silently invent scores).

When `--model` is omitted, outputs go to `results/evaluation/<family_key>/`.  
When `--model` is set, outputs go **directly** to `--out` (so pass `--out results/evaluation/xgb`).

### Family outputs written

- `candidate_manifest.csv`
- `summary_metrics.csv` — mean/std metrics by candidate × protocol
- `bootstrap_intervals.csv`
- `paired_model_comparisons.csv`
- `selected_hyperparameters.csv`
- `per_paper_metrics.csv`
- `search_history.csv`
- `inner_oof_predictions.csv`
- `outer_fold_predictions.parquet`
- `integrity_checks.json`

### Cost

This step dominates runtime: many outer folds × 26 candidates × nested search. Prefer HPC array jobs or start with `--model` / fewer `--search-iterations` while debugging.

---

## 8. `fable_select.py` — choose champions without training

**CLI.**

```bash
python fable_select.py \
  --evaluation results/evaluation \
  --policy configs/selection_policy.yaml \
  --out results/selection.json
```

### What it does

1. Concatenate `summary_metrics.csv`, bootstrap intervals, paired comparisons, manifests, and hyperparameters from each family directory under `results/evaluation/`.
2. Merge `integrity_checks.json` protocol statuses.
3. Filter ineligible candidates (failed integrity, incomplete folds, invalid probs, unstable SD, missing LOPO for extrapolation).
4. Rank remaining candidates per selection rule; mark one winner each for interpolation and extrapolation.
5. Attach **modal** hyperparameters from outer folds of the primary protocol.
6. Write:
   - `results/selection.json` — machine-readable
   - `results/selection.md` — human-readable ranking tables

**Does not fit any model.**

---

## 9. `fable_refit.py` — deployable artifacts

**CLI.**

```bash
python fable_refit.py \
  --data Microgravity_Database.csv \
  --selection results/selection.json \
  --champion interpolation \
  --out artifacts/interpolation_champion

python fable_refit.py \
  --data Microgravity_Database.csv \
  --selection results/selection.json \
  --champion extrapolation \
  --out artifacts/extrapolation_champion
```

| Flag | Meaning |
|---|---|
| `--champion` | `interpolation` or `extrapolation` |
| `--out` | Artifact directory |
| `--random-state` | Default 42 |

### Steps

1. Read the chosen candidate + exact hyperparameters from `selection.json`.
2. Build full-data OOF predictions under the matching protocol:
   - Interpolation: stratified 5-fold × 3 seeds
   - Extrapolation: `StratifiedGroupKFold(5)` × 3 seeds (asserts no paper leakage)
3. Optimize the four thresholds from those OOF probabilities.
4. Fit the final model on **all** labeled rows.
5. Write the artifact package.

### Artifact contents

```text
artifacts/<champion>_champion/
├── model.joblib
├── model_card.json
├── thresholds.json
├── feature_manifest.json
├── training_data_fingerprint.json
├── refit_metadata.json
└── out_of_fold_predictions.csv
```

---

## 10. `fable_report.py` — tables, plots, explainability

**CLI.**

```bash
python fable_report.py \
  --data Microgravity_Database.csv \
  --evaluation results/evaluation \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report
```

### Produces (among others)

- Dataset integrity summary
- Interpolation / holdout / extrapolation leaderboards (CSV + Markdown)
- Interpolation–extrapolation drop table
- Champion threshold performance table
- Figures: ROC/PR uncertainty, five-family comparison, drop plot, paired deltas, LOPO histogram, calibration, threshold bars
- XGBoost SHAP ranking + permutation importance (per champion subfolder when possible)
- `results/report/README.md` and `report_manifest.json`

Explainability prefers the champion if it is XGBoost; otherwise the best eligible XGBoost on that leaderboard. Failures are recorded in `explainability_status.json` rather than crashing the whole report when possible.

---

## 11. `fable_predict.py` — inference

**CLI.**

```bash
python fable_predict.py \
  --input new_conditions.csv \
  --output predictions.csv \
  --champion extrapolation
# optional: --artifact /path/to/artifacts/extrapolation_champion
```

### Requirements in the artifact directory

`model.joblib`, `model_card.json`, `thresholds.json`.

### Output columns

- `row_id`, `paper_id`, `paper_label`
- `ignition_probability`
- `prediction_<threshold>` and `threshold_<threshold>` for each of the four policies
- Model / fingerprint metadata (`champion`, `model_id`, `model_family`, training dataset SHA-256, …)

Input loading uses `require_target=False` and does not drop duplicates (preserves one prediction per input row).

---

## 12. SLURM scripts (cluster workflow)

All scripts `cd` to `SLURM_SUBMIT_DIR` and run inside Singularity with a project overlay. Update overlay/image/account paths for your account.

| Script | Job | What it runs |
|---|---|---|
| `run_classifiers_single.slurm` | `ignition-splits` | `fable_splits.py` (3 seeds, 5+5 folds) |
| `run_classifiers_array.slurm` | array `0–4` | `fable_evaluate.py --model {xgb,svm,knn,dt,mlp}` → `results/evaluation/$MODEL` with 15 search iterations |
| `run_mlp_candidates_parallel.slurm` | array `0–4` | One MLP `candidate_id` per task → `results/evaluation/mlp/<id>/` |
| `run_classifiers_finalize.slurm` | `ignition-finalize` | `select` → both `refit`s → `report` |

Suggested order:

```bash
mkdir -p logs
sbatch run_classifiers_single.slurm
# wait for splits
sbatch run_classifiers_array.slurm
# optional MLP parallel path; then ensure family-level mlp/ metrics exist for select
sbatch run_classifiers_finalize.slurm
```

---

## 13. Leakage and honesty checklist

Use this when reviewing a PR or a new experiment:

- [ ] Splits were generated once and reused for all candidates being compared.
- [ ] Grouped / LOPO folds have zero paper overlap (enforced in `fable_splits.py`).
- [ ] Hyperparameters and thresholds for each outer fold were chosen using **only** that fold’s training rows.
- [ ] Outer test metrics were not used to pick hyperparameters.
- [ ] Interpolation and extrapolation champions are reported as answering **different** questions.
- [ ] Deployed `artifacts/` models are not described as nested-CV scores.
- [ ] Feature list matches `fable-data-v5` (32 numeric + 5 categorical); no post-outcome columns.
6. **Build the production copy** — refit on all labeled data (`fable_refit`).
7. **Write the report and serve predictions** — `fable_report` / `fable_predict`.

That order is the science. Changing it (especially peeking at outer test data during tuning) breaks the benchmark.
