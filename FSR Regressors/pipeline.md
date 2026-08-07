# FSR Regression Pipeline — Full Guide


If you only need commands, use [`README.md`](README.md). This document explains **what each stage does**, **why it exists**, and **what it writes**.

---

## 1. Big picture

We predict **continuous flame spread rate (FSR)** from literature experiments in `Microgravity_Database.csv` (or an Excel export of the same schema).

The scientific risk is **leakage**: if the same paper’s rows appear in both train and test, scores look too good because the model has already seen a very similar campaign. The pipeline therefore answers two questions separately and never conflates them:

1. **Interpolation** (`interpolation_random`) — new rows from campaigns already partly seen.
2. **Extrapolation** (`extrapolation_grouped`) — entirely held-out papers (primary scientific result).

Every expensive stage is restartable: freeze splits once, evaluate candidates as independent shards, merge, select champions from tables only, then refit for deployment.

```text
Microgravity_Database.csv / .xlsm
        │
        ▼
 fsr_common.py            load, detect roles, engineer feature manifest (library)
        │
        ▼
 fsr_splits.py            freeze interpolation + extrapolation holdouts
        │
        ▼
 fsr_evaluate.py          for each candidate × outer split:
        │                   └─ fsr_search.py (nested tune on train)
        │                   └─ fsr_models.py (fit rd and rd_bt, score test)
        │                 → results/evaluation/<family>/<candidate_id>/
        ▼
 fsr_aggregate.py         merge shards → results/evaluation/merged/
        │
        ▼
 fsr_select.py            selection_policy.yaml → selection.json
        │
        ▼
 fsr_refit.py             deployable interpolation + extrapolation champions
        │
        ▼
 fsr_report.py            numbered report folders + figures
        │
        ▼
 fsr_predict.py           (later) score new CSVs / Excel with an artifact
```

---

## 2. `fsr_common.py` — data loading and shared utilities

**Not a CLI.** Imported by every other script. Data version: `fsr-data-v2`. Internal target column name: `fsr`.

### Responsibilities

1. **Resolve and read the database** (`locate_data_file`, `read_database`)
   - Accepts CSV/TXT or Excel (`.xlsm` / `.xlsx` / `.xls`).
   - Tries encodings and header layouts (category banner for CSV; two-level Excel headers).
   - Chooses the parse that best exposes an FSR-like column.

2. **Drop disallowed columns** before modelling:
   - Material of Sample, Rig Name, Internal Geometry, Internal Dimensions, Facility, Info.

3. **Detect column roles** (not a hard-coded feature list)
   - **Target:** column whose name looks like FSR / flame spread rate and has the most numeric values.
   - **Group:** most complete citation-like column (article / paper / MLA / DOI / …).
   - **Leakage:** duplicate FSR fields, post-outcome columns (flame length, HRR, smoke, extinction, …), notes, and paper fingerprints (authors, DOI, article) — excluded from features.
   - **Features:** remaining columns → numeric vs categorical.
     - Numeric parse only if the cell **begins** with a number (`"101.3 kPa"` → number; `"N2"` / `"CO2"` stay categorical).

4. **Paper identity**
   - Prefer canonical DOI → `paper_id = "doi::<normalized>"`.
   - Else canonical citation → `"citation::<normalized>"`.

5. **Feature manifest**
   - Each feature tagged `kind` (`numeric` / `categorical`) and `role` (`physics` / `apparatus`).
   - `feature_lists(manifest, feature_set=...)` still accepts `feature_set` for API compatibility but **returns all features** (this database is physics-only; role filtering is a no-op).

6. **Weights and augmentation**
   - `paper_weights(papers, strategy)` with `none`, `sqrt`, `inverse`, `log`.
   - `bootstrap_augment(...)` adds extra training rows drawn with replacement; each copy inherits its source paper ID.

7. **Metrics** (`regression_metrics`): R², RMSE, MAE, MAPE, MBE, NRMSE.

8. **GPU / Slurm helpers** — Torch device, XGBoost `cuda` vs `cpu`, `SLURM_CPUS_PER_TASK`.

9. **Preprocessing contract** (`build_preprocessor`) — median impute / optional scale for numerics; constant + one-hot for categoricals; XGBoost can skip numeric impute (native missing).

### Key API

```python
df, report = load_data("Microgravity_Database.csv", require_target=True)
# df has row_id, paper_id, paper_label, features, and fsr
```

---

## 3. `fsr_models.py` — model families and preprocessing

**Not a CLI.** Defines `MODEL_REGISTRY` and `FSRModel`.

### Families

| Family | Sample weight | Scale numerics | Native NaN | Accelerator note |
|---|---|---|---|---|
| `xgboost` | Yes | No | Yes | CUDA hist when GPU visible |
| `random_forest` | Yes | No | No | CPU, multi-core |
| `knn` | No → resample | Yes | No | cuML if available, else sklearn |
| `mlp` | Yes (weighted MSE) | Yes | No | PyTorch GPU or NCCL DDP |
| `svr` | No → resample | Yes | No | sklearn SVR |

Legacy numeric filter for whole-family runs: `MODEL_ID_TO_FAMILY = (xgboost, random_forest, knn, mlp, svr)` → `--model-id 0..4`.

### `FSRModel` / `make_model`

```text
model = make_model(candidate_dict, params, feature_manifest)
model.fit(X_df, y, papers)
y_hat = model.predict(X_df)   # always in original FSR units
```

Extras driven by the candidate YAML:

- `target_transform: signed_log1p` — train on compressed target; invert on predict.
- `paper_bagging` (XGBoost only) — average models trained on paper-level bootstrap samples.
- `monotone_oxygen` (XGBoost only, optional) — monotone constraints on oxygen-like features.
- `fixed_params` — e.g. XGB objective variant (`squarederror` / `absoluteerror` / `pseudohuber`), RF `criterion`, KNN `weights`, SVR `kernel`.

Weight-blind families (KNN, SVR) use deterministic probability-proportional resampling so paper weighting is never silently ignored.

---

## 4. `fsr_splits.py` — freeze holdouts

**CLI.** Materializes both protocols once for every downstream family/candidate.

### Command

```bash
python fsr_splits.py \
  --data Microgravity_Database.csv \
  --out results/splits \
  --n-repeats 10 \
  --test-size 0.2 \
  --split-candidates 50 \
  --target-bins 6 \
  --base-seed 42
```

| Flag | Default | Meaning |
|---|---:|---|
| `--n-repeats` | 10 | Repeated holdouts per protocol |
| `--test-size` | 0.2 | Fraction of rows (or papers) in test |
| `--split-candidates` | 50 | Grouped partitions screened per repeat |
| `--target-bins` | 6 | Quantile bins for stratification / balance |
| `--base-seed` | 42 | First seed |

### Protocols

| Protocol | Mechanism |
|---|---|
| `interpolation_random` | Target-quantile–stratified row holdout when feasible; papers may appear on both sides |
| `extrapolation_grouped` | Many `GroupShuffleSplit` candidates; keep the most **target-balanced** paper-disjoint split |

Balance screening reduces nuisance R² swings from unlucky extreme papers in the test set without relaxing the no-paper-split rule.

### Integrity

For every `split_id`: full row coverage, non-empty partitions, no row overlap; for grouped protocol, **no paper** in both train and test.

### Outputs

```text
results/splits/
├── split_assignments.csv|.parquet
├── split_diagnostics.csv
├── splits_metadata.json
└── data_validation_report.json
```

---

## 5. `configs/` — candidates and selection policy

### `candidates.yaml` (version 3)

- `bootstrap_rows: 1000` — size of `rd_bt` augmentation.
- **34** candidates (10 XGB + 7 RF + 6 KNN + 5 MLP + 6 SVR).

Design notes: [`configs/candidates.md`](configs/candidates.md). Prefer the YAML if that markdown’s counts lag behind.

### `selection_policy.yaml`

Consumed only by `fsr_select.py`:

- Integrity / complete folds / finite predictions required.
- Reject if RMSE fold CV > `0.45` or mean R² < `-0.25`.
- Primary metric: **RMSE** (lower better); tie band = best RMSE × `(1 + 0.02)`.
- Champions chosen among `(candidate_id, augmentation)` pairs for `rd` and `rd_bt`.
- **Interpolation:** protocol `interpolation_random`; tie-breakers include higher R², lower uncertainty, prefer `rd`, simpler family, fewer features.
- **Extrapolation:** protocol `extrapolation_grouped`; also prefers smaller generalization gap and “physics_only” when that field differs (currently all candidates see the same full feature list).
- Refit hyperparameters: **modal** outer-fold selected config, lexical tie-break.
- Reporting policy states extrapolation is the primary scientific result.

---

## 6. `fsr_search.py` — nested search

**Not a CLI.** Called once per outer training partition inside `fsr_evaluate.py`.

1. Build **inner** folds from outer **train** only:
   - Extrapolation → paper-disjoint `GroupKFold`
   - Interpolation → shuffled `KFold`
2. Sample up to `iterations` distinct hyperparameter configs from the family search space (+ candidate `fixed_params`).
3. Fit each config × fold (joblib; CPU families parallelize; GPU families keep one fit per worker).
4. Rank by mean inner **RMSE**, then MAE, then higher R².
5. Return selected params, CV metrics, search history, and inner OOF predictions.

No thresholds (this is regression, not classification).

---

## 7. `fsr_evaluate.py` — the heavy benchmark runner

**CLI.** Sole place that runs nested tune + outer scoring. One run = one shard.

### Command

```bash
python fsr_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation/xgboost/xgb_baseline \
  --candidate-id xgb_baseline \
  --search-iterations 40 \
  --inner-folds 5 \
  --augmentations rd,rd_bt \
  --bootstrap-iterations 2000
```

| Flag | Meaning |
|---|---|
| `--candidate-id` | Preferred: exactly one YAML candidate (Slurm array mode) |
| `--model-id` | Legacy: all candidates in one family (`0`…`4`) |
| `--augmentations` | Comma list, default `rd,rd_bt` |
| `--protocols` | Optional filter to one protocol |
| `--search-jobs` | joblib workers for the config × fold grid |
| `--bootstrap-rows` | Override YAML `bootstrap_rows` |

### Per outer split

1. Assert no paper leakage on extrapolation splits.
2. `nested_search` on train only.
3. For each augmentation:
   - `rd` — fit on real train rows.
   - `rd_bt` — fit on real train + bootstrap copies (papers inherited); **test never augmented**.
4. Score untouched outer test; record fold metrics, predictions, search history.
5. Optionally stash explainability models for tree families.
6. Record failures in `integrity_checks.json` instead of silently omitting them.

### Shard outputs (under `--out`)

Include `fold_metrics.csv`, `outer_fold_predictions.csv`, `selected_hyperparameters.csv`, `search_history.csv`, `candidate_manifest.csv`, `integrity_checks.json`, `evaluation_metadata.json`, and derived tables written for that shard.

**Fingerprint guard:** dataset SHA-256 must match `splits_metadata.json`.

---

## 8. `fsr_aggregate.py` — merge shards

**CLI.**

```bash
python fsr_aggregate.py \
  --evaluation results/evaluation \
  --out results/evaluation/merged \
  --bootstrap-iterations 2000
```

### Discovery

Looks for shards as:

1. Legacy `model_*` directories, or
2. New layout `results/evaluation/<family>/<candidate_id>/` (directories that contain `fold_metrics.csv`).

### Derived tables written to `--out`

- Merged raw frames (predictions, fold metrics, histories, …)
- `summary_metrics.csv`
- `per_paper_metrics.csv`
- `paired_model_comparisons.csv` (paired tests over shared folds)
- `bootstrap_intervals.csv` (paper-cluster bootstrap)
- `generalization_gap.csv` (interpolation vs extrapolation)
- `augmentation_comparison.csv` (`rd` vs `rd_bt`)
- `metric_stability.csv`
- Merged `integrity_checks.json` + `evaluation_metadata.json`
- Copied `explainability_models/`

**Select and report read this merged directory**, not individual shards.

---

## 9. `fsr_select.py` — choose champions without training

**CLI.**

```bash
python fsr_select.py \
  --evaluation results/evaluation/merged \
  --policy configs/selection_policy.yaml \
  --out results/selection.json
```

Ranks eligible `(model_id, augmentation)` rows per protocol, applies rejection rules and tie-breakers, attaches modal hyperparameters, writes:

- `results/selection.json`
- `results/selection.md`

Does **not** fit any model.

---

## 10. `fsr_refit.py` — deployable artifacts

**CLI.**

```bash
python fsr_refit.py \
  --data Microgravity_Database.csv \
  --selection results/selection.json \
  --evaluation results/evaluation/merged \
  --champion extrapolation \
  --out artifacts/extrapolation_champion
```

### Steps

1. Read candidate, exact hyperparameters, and winning **augmentation** from selection.
2. Build full-data OOF predictions under the matching protocol (repeated KFold vs GroupKFold), applying `rd_bt` during OOF fits when that condition won.
3. Fit the final model on all labeled rows (again with `rd_bt` if selected).
4. Package the artifact.

### Artifact contents

```text
artifacts/<champion>_champion/
├── model.joblib
├── model_card.json
├── feature_manifest.json
├── training_data_fingerprint.json
├── refit_metadata.json
└── out_of_fold_predictions.csv
```

---

## 11. `fsr_report.py` — publication pack

**CLI.**

```bash
python fsr_report.py \
  --data Microgravity_Database.csv \
  --evaluation results/evaluation/merged \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report
# optional: --no-shap
```

### Output layout

```text
results/report/
├── 00_overview/
├── 01_comparison/
├── 02_diagnostics/
├── 03_error_analysis/
├── 04_hard_papers/
├── 05_learning_curves/          # when enough fold data exists
├── 06_per_paper/
├── 07_explainability/           # permutation + SHAP for tree champions when available
├── README.md
└── report_manifest.json
```

Built only from persisted evaluation + artifact files (no re-tuning).

---

## 12. `fsr_predict.py` — inference

**CLI.**

```bash
python fsr_predict.py \
  --input new_conditions.csv \
  --output predictions.csv \
  --champion extrapolation
```

Requires `model.joblib`, `model_card.json`, and `feature_manifest.json` in the artifact directory. Loads input with `require_target=False` (keeps rows without FSR). Writes `predicted_fsr` plus metadata; includes `observed_fsr` when the target column is present.

---

## 13. SLURM scripts

| Script | Job | Role |
|---|---|---|
| `run_regressors_splits.slurm` | `fsr-splits` | `fsr_splits.py` (default 10 repeats) |
| `run_candidates_array.slurm` | `fsr-candidates` array `0–33` | One `fsr_evaluate.py --candidate-id …` per task |
| `run_regressors_finalize.slurm` | `fsr-finalize` | aggregate → select → both refits → report |

Typical chain:

```bash
mkdir -p logs
SPLITS=$(sbatch --parsable run_regressors_splits.slurm)
ARRAY=$(sbatch --parsable --dependency=afterok:${SPLITS} run_candidates_array.slurm)
sbatch --dependency=afterok:${ARRAY} run_regressors_finalize.slurm
```

Override data / overlay / image / search budget with `FSR_DATA`, `FSR_OVERLAY`, `FSR_IMAGE`, `FSR_N_REPEATS`, `FSR_SEARCH_ITERATIONS`, `FSR_INNER_FOLDS`, `FSR_BOOTSTRAP_ITERATIONS`.

**Note:** default `FSR_DATA` differs slightly across scripts (folder CSV vs repo-root CSV). Prefer setting `FSR_DATA` explicitly to the same file used for splits.

---

## 14. Leakage and honesty checklist

- [ ] Same frozen splits for every candidate being compared.
- [ ] Extrapolation folds have zero paper overlap.
- [ ] Hyperparameters chosen only inside each outer train partition.
- [ ] Outer test rows never used for tuning, imputation vocabularies, or bootstrap augmentation.
- [ ] Interpolation and extrapolation champions reported as different questions.
- [ ] `rd_bt` described as re-weighting observed training conditions, not new experiments.
- [ ] Deployed `artifacts/` not described as nested-CV scores.
- [ ] Dataset SHA-256 matches between splits and evaluate.

