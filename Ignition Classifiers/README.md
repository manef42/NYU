# Ignition Classifiers (FABLE)

Intern-facing entry point for the **binary ignition classification** pipeline.

**Goal:** given experimental conditions (oxygen, pressure, material/fuel properties, gravity, flow, geometry, ignition method, …), predict whether a sample will ignite (`Yes` / `No`).

This folder is self-contained: Python scripts, configs, SLURM launchers, the dataset CSV, and (when generated) evaluation artifacts. Start here, then read [`pipeline.md`](pipeline.md) for a deeper walkthrough of every stage.

---

## What this project is (and is not)

| This project **is** | This project **is not** |
|---|---|
| A leakage-controlled benchmark of 26 candidates across 5 model families | A single “best model forever” claim |
| Two scientific questions: **interpolation** vs **extrapolation** | Proof of causal physics mechanisms |
| Reproducible splits, search histories, and deployable refit artifacts | An excuse to mix train/test papers |

- **Interpolation** — stratified row-level splits. Test rows can come from papers that also appear in training. Answers: “How well do we predict similar conditions already represented in the database?”
- **Extrapolation** — whole papers are held out (`StratifiedGroupKFold` + LOPO). Answers: “How well do we transfer to an unseen paper / campaign?”

Never present interpolation scores as evidence of unseen-paper generalization.

---

## Repository layout

```text
Ignition Classifiers/
├── Microgravity_Database.csv     # Source database (category row + header + data)
├── requirements.txt              # Python dependencies
├── configs/
│   ├── candidates.yaml           # 26 candidates (family × weighting × variants)
│   ├── candidates.md             # Design rationale for those candidates
│   └── selection_policy.yaml     # How champions are chosen (no training)
├── fable_common.py               # Load / clean / engineer features / weights
├── fable_models.py               # Model families, preprocessing, fit/predict
├── fable_splits.py               # Freeze all CV / holdout / LOPO assignments
├── fable_search.py               # Nested hyperparameter search + thresholds
├── fable_evaluate.py             # Outer-fold benchmark runner (heavy)
├── fable_select.py               # Policy-based champion selection
├── fable_refit.py                # Train deployable champions on all labeled data
├── fable_report.py               # Tables, figures, explainability
├── fable_predict.py              # Inference on new CSVs using refit artifacts
├── run_classifiers_single.slurm  # HPC: create splits
├── run_classifiers_array.slurm   # HPC: evaluate one family per array task
├── run_mlp_candidates_parallel.slurm  # HPC: one MLP candidate per array task
├── run_classifiers_finalize.slurm     # HPC: select → refit → report
├── results/                      # Generated (often gitignored / large)
└── artifacts/                    # Generated deployable models
```

Supporting docs: [`pipeline.md`](pipeline.md) (full stage-by-stage guide), [`configs/candidates.md`](configs/candidates.md) (why each candidate exists).

---

## Quick start (local)

Python **3.11+** recommended. From the repo root:

```bash
cd "Ignition Classifiers"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch` is required (MLP). On machines without a GPU, PyTorch runs on CPU automatically.

### Full pipeline (order matters)

```bash
# 1) Freeze splits once
python fable_splits.py \
  --data Microgravity_Database.csv \
  --out results/splits \
  --n-seeds 3 \
  --n-group-folds 5 \
  --n-row-folds 5

# 2) Nested evaluate all 26 candidates (expensive)
python fable_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation \
  --search-iterations 40 \
  --inner-group-folds 3

# 3) Select champions (no training)
python fable_select.py \
  --evaluation results/evaluation \
  --policy configs/selection_policy.yaml \
  --out results/selection.json

# 4) Refit deployable models
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

# 5) Publication report
python fable_report.py \
  --data Microgravity_Database.csv \
  --evaluation results/evaluation \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report
```

### Faster / partial evaluation (optional)

Evaluate one family only (writes directly into `--out`, so point `--out` at that family’s folder):

```bash
python fable_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation/xgb \
  --model xgb \
  --search-iterations 15 \
  --inner-group-folds 3
```

Evaluate a single candidate (useful for MLP parallel jobs):

```bash
python fable_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation/mlp/mlp_baseline \
  --model mlp \
  --candidate-id mlp_baseline \
  --search-iterations 15 \
  --inner-group-folds 3
```

`--model` accepts family keys (`xgb`, `svm`, `knn`, `dt`, `mlp`) or full names (`xgboost`, `decision_tree`, …).

**Important for selection:** `fable_select.py` and `fable_report.py` load CSVs from:

```text
results/evaluation/xgb/
results/evaluation/svm/
results/evaluation/knn/
results/evaluation/dt/
results/evaluation/mlp/
```

If you run MLP (or any family) into per-candidate subfolders, consolidate the standard CSVs/`integrity_checks.json` into the family directory before select/report, or run that family with `--out results/evaluation/<family>` and no `--candidate-id`.

### Inference

After refit artifacts exist:

```bash
python fable_predict.py \
  --input /path/to/new_conditions.csv \
  --output predictions.csv \
  --champion extrapolation
```

Use `--artifact /path/to/artifact_dir` to override `artifacts/<champion>_champion`. The input CSV must follow the same schema as the database (category row + header). Target labels are optional for inference.

---

## Script cheat sheet

| Script | Role | Trains models? |
|---|---|---|
| `fable_common.py` | Shared library: load CSV, engineer **32 numeric + 5 categorical** features, paper IDs, weights | No (library) |
| `fable_models.py` | Shared library: XGBoost / DT / KNN / MLP / SVM wrappers + preprocessing | Library used by trainers |
| `fable_splits.py` | Write frozen split assignments for 4 protocols | No |
| `fable_search.py` | Nested random search + 4 threshold policies on an outer **train** fold | Yes (inner only) |
| `fable_evaluate.py` | Loop candidates × outer folds; persist metrics, preds, integrity | Yes |
| `fable_select.py` | Apply `selection_policy.yaml`; pick interpolation & extrapolation champions | No |
| `fable_refit.py` | Rebuild chosen config on **all** labeled rows; write `artifacts/` | Yes |
| `fable_report.py` | Leaderboards, plots, SHAP/permutation for XGBoost, report README | No (reads artifacts) |
| `fable_predict.py` | Score new rows with a refit champion | No (loads artifact) |

Details for each script: [`pipeline.md`](pipeline.md).

---

## Candidates (26)

Defined in `configs/candidates.yaml` (design notes in `configs/candidates.md`):

| Family | Count | IDs |
|---|---:|---|
| XGBoost | 8 | `xgb_baseline`, `xgb_paper_sqrt`, `xgb_paper_inverse`, `xgb_class_only`, `xgb_weighted_sqrt`, `xgb_weighted_inverse`, `xgb_focal_sqrt`, `xgb_focal_inverse` |
| Decision tree | 3 | `dt_baseline`, `dt_weighted_sqrt`, `dt_weighted_inverse` |
| KNN | 5 | `knn_baseline`, `knn_weighted_sqrt`, `knn_weighted_inverse`, `knn_distance_sqrt`, `knn_distance_inverse` |
| MLP | 5 | `mlp_baseline`, `mlp_weighted_sqrt`, `mlp_weighted_inverse`, `mlp_focal_sqrt`, `mlp_focal_inverse` |
| SVM | 5 | `svm_baseline`, `svm_weighted_sqrt`, `svm_weighted_inverse`, `svm_linear_sqrt`, `svm_linear_inverse` |

Each candidate is a **recipe**: model family + paper-weight strategy (`none` / `sqrt` / `inverse`) + optional class weighting + optional fixed params (focal loss, distance KNN, linear SVM, …).

---

## Data assumptions (must know)

- Default file: `Microgravity_Database.csv` in this folder.
- File layout: **row 1 = category labels**, **row 2 = real header**, data starts on row 3. The loader uses `skiprows=1`.
- Encodings tried: UTF-8, UTF-8-BOM, CP1252, Latin-1.
- Target for train/eval: `Ignition` → yes/no (`require_target=True`). Inference may omit it.
- Paper identity: normalized DOI (`doi::…`), else normalized citation (`citation::…`). Raw DOI strings are never used as groups.
- Dropped on load (never features): Material of Sample, Rig Name, Internal Geometry, Internal Dimensions, Facility, Info.
- Post-outcome columns excluded from features: Flame Length, FSR, HRR, Smoke/Aerosols.
- Data version stamp: `fable-data-v5` (`DATA_VERSION` in `fable_common.py`).
- Features: **32 numeric + 5 categorical** (same list for every candidate). See `pipeline.md`.

Missing numerics: median impute for sklearn-style models; XGBoost keeps native missing handling. Categoricals: constant `"Unknown"` impute + unknown-safe one-hot.

---

## Outputs

```text
results/
├── splits/
│   ├── split_assignments.csv|.parquet|.joblib
│   ├── splits_metadata.json
│   └── data_validation_report.json
├── evaluation/
│   ├── xgb/ | svm/ | knn/ | dt/ | mlp/
│   │   ├── candidate_manifest.csv
│   │   ├── summary_metrics.csv
│   │   ├── bootstrap_intervals.csv
│   │   ├── paired_model_comparisons.csv
│   │   ├── selected_hyperparameters.csv
│   │   ├── per_paper_metrics.csv
│   │   ├── search_history.csv
│   │   ├── inner_oof_predictions.csv
│   │   ├── outer_fold_predictions.parquet
│   │   └── integrity_checks.json
│   └── (optional) mlp/<candidate_id>/ … from parallel SLURM
├── selection.json
├── selection.md
└── report/                    # tables, PNGs, explainability, README

artifacts/
├── interpolation_champion/    # model.joblib, thresholds, model card, …
└── extrapolation_champion/
```

**Evaluation artifacts** = unbiased comparison evidence (not production models).  
**Refit artifacts** = production models (not unbiased CV evidence).

Large generated files may be ignored by Git; keep the scripts and configs versioned.

---

## HPC (NYU Greene / Singularity)

SLURM scripts assume:

- Overlay: `/scratch/me3144/ignition_hpc/ignition_python.ext3`
- Image: `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- Account: `torch_pr_1163_tandon_advanced`

Submit **from** the `Ignition Classifiers` directory (`SLURM_SUBMIT_DIR` becomes the project root):

```bash
mkdir -p logs

# 1) Splits (~30 min allocation)
sbatch run_classifiers_single.slurm

# 2) One Slurm array task per family: xgb, svm, knn, dt, mlp
#    Wait until splits exist. Uses --search-iterations 15.
sbatch run_classifiers_array.slurm

# Optional: parallelize MLP candidates separately (writes under evaluation/mlp/<id>/)
sbatch run_mlp_candidates_parallel.slurm

# 3) After all family outputs are in place under results/evaluation/<family>/
sbatch run_classifiers_finalize.slurm
```

Logs land in `logs/`. Adjust `#SBATCH` account, overlay path, and image if your cluster allocation differs.

---

## Reproducibility

- Default base seed: **42**. Repeated splits use consecutive seeds (`42`, `43`, `44` when `--n-seeds 3`).
- Splits store dataset SHA-256, prevalence, paper counts, and exact row/paper memberships.
- Search configs and histories are persisted per fold.
- Weighting, resampling, XGBoost, MLP init, and report resampling use recorded deterministic seeds.

---

## Common pitfalls for new contributors

1. **Wrong run order** — splits → evaluate → select → refit → report. Do not skip splits.
2. **Confusing the two champions** — interpolation ≠ extrapolation. Use the champion that matches the scientific question.
3. **Family vs candidate `--out` paths** — select/report only read family directories listed above.
4. **Editing features mid-benchmark** — change `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` only with a new `DATA_VERSION` and regenerate splits + evaluation.
5. **Calling evaluation metrics “the deployed model”** — only `artifacts/*_champion` are deployable.
6. **Treating LOPO as the sole ranking metric** — LOPO is mandatory robustness for the extrapolation champion; primary selection uses `extrapolation_grouped` ROC-AUC (see `selection_policy.yaml`).

---

## Where to go next

1. Read [`pipeline.md`](pipeline.md) end-to-end once.
2. Skim [`configs/candidates.md`](configs/candidates.md) and [`configs/selection_policy.yaml`](configs/selection_policy.yaml).
3. Run `fable_splits.py` locally to confirm the environment and inspect `results/splits/`.
4. Run a **single-family** evaluation with fewer `--search-iterations` before attempting the full 26-candidate job.
