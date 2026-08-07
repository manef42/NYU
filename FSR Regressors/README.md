# FSR Regressors

Intern-facing entry point for the **flame spread rate (FSR) regression** pipeline.

**Goal:** given experimental conditions from the microgravity combustion literature database, predict continuous **flame spread rate** (`fsr`, typically in m/s).

This folder is self-contained: Python scripts, configs, SLURM launchers, the dataset CSV, and (when generated) evaluation / deployable artifacts. Start here, then read [`pipeline.md`](pipeline.md) for a stage-by-stage walkthrough of every script.

---

## What this project is (and is not)

| This project **is** | This project **is not** |
|---|---|
| A leakage-controlled benchmark of **34 candidates** across **5** model families | A single “best model forever” claim |
| Two scientific questions: **interpolation** vs **extrapolation** | Proof of causal combustion mechanisms |
| Restartable, shardable jobs with merge → select → refit → report | An excuse to mix train/test papers |
| Evaluation of real data (`rd`) vs bootstrap-augmented training (`rd_bt`) | Creation of new physics via resampling |

- **Interpolation** (`interpolation_random`) — repeated target-stratified **row-level** 80/20 holdouts. Rows from the same paper may appear in both train and test. Optimistic baseline: “How well do we predict a new row from a campaign we have partly seen?”
- **Extrapolation** (`extrapolation_grouped`) — repeated **paper-disjoint** holdouts (`GroupShuffleSplit`, balance-screened). **Primary scientific result:** “How well do we transfer to a brand-new paper / campaign?”

Never present random-split / interpolation scores as unseen-paper generalization.

---

## Repository layout

```text
FSR Regressors/
├── Microgravity_Database.csv     # Default local dataset (category banner + header + data)
├── requirements.txt
├── configs/
│   ├── candidates.yaml           # 34 candidates + bootstrap_rows
│   ├── candidates.md             # Design rationale
│   └── selection_policy.yaml     # How champions are chosen (no training)
├── fsr_common.py                 # Load / detect roles / weights / metrics / GPU helpers
├── fsr_models.py                 # XGBoost, RF, KNN, MLP, SVR wrappers
├── fsr_splits.py                 # Freeze interpolation + extrapolation holdouts
├── fsr_search.py                 # Nested hyperparameter search on one outer train fold
├── fsr_evaluate.py               # Benchmark runner (one candidate or one family)
├── fsr_aggregate.py              # Merge shards + derived tables
├── fsr_select.py                 # Policy-based champion selection
├── fsr_refit.py                  # Deployable champions on all labeled rows
├── fsr_report.py                 # Numbered report folders + figures
├── fsr_predict.py                # Inference with a refit artifact
├── run_regressors_splits.slurm
├── run_candidates_array.slurm    # Array 0–33: one candidate per task
├── run_regressors_finalize.slurm # aggregate → select → refit → report
├── results/                      # Generated
└── artifacts/                    # Generated deployable models
```

---

## Quick start (local)

Python **3.11+** recommended:

```bash
cd "FSR Regressors"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch` is required (MLP). GPU is optional; XGBoost / cuML / PyTorch fall back to CPU.

### Full pipeline (order matters)

Default data path below is the CSV **in this folder**. Excel (`.xlsm` / `.xlsx`) also works if you pass that path.

```bash
# 1) Freeze splits once (10 repeats × 2 protocols by default)
python fsr_splits.py \
  --data Microgravity_Database.csv \
  --out results/splits \
  --n-repeats 10 \
  --test-size 0.2 \
  --split-candidates 50 \
  --target-bins 6

# 2) Evaluate candidates (expensive). Example: one candidate shard
python fsr_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation/xgboost/xgb_baseline \
  --candidate-id xgb_baseline \
  --search-iterations 40 \
  --inner-folds 5 \
  --augmentations rd,rd_bt

# Or evaluate a whole family (legacy --model-id):
#   0=xgboost, 1=random_forest, 2=knn, 3=mlp, 4=svr
python fsr_evaluate.py \
  --data Microgravity_Database.csv \
  --splits results/splits \
  --config configs/candidates.yaml \
  --out results/evaluation/random_forest \
  --model-id 1 \
  --search-iterations 40 \
  --inner-folds 5 \
  --augmentations rd,rd_bt

# Prefer looping all 34 candidates into family/candidate shards (matches Slurm):
# for each candidate_id → --out results/evaluation/<family>/<candidate_id> --candidate-id ...

# 3) Merge shards + derived tables
python fsr_aggregate.py \
  --evaluation results/evaluation \
  --out results/evaluation/merged \
  --bootstrap-iterations 2000

# 4) Select champions (no training)
python fsr_select.py \
  --evaluation results/evaluation/merged \
  --policy configs/selection_policy.yaml \
  --out results/selection.json

# 5) Refit deployable models
python fsr_refit.py \
  --data Microgravity_Database.csv \
  --selection results/selection.json \
  --evaluation results/evaluation/merged \
  --champion interpolation \
  --out artifacts/interpolation_champion

python fsr_refit.py \
  --data Microgravity_Database.csv \
  --selection results/selection.json \
  --evaluation results/evaluation/merged \
  --champion extrapolation \
  --out artifacts/extrapolation_champion

# 6) Publication report
python fsr_report.py \
  --data Microgravity_Database.csv \
  --evaluation results/evaluation/merged \
  --selection results/selection.json \
  --artifacts artifacts \
  --out results/report
```

`fsr_evaluate.py` refuses to run if the dataset SHA-256 differs from the one stored in `results/splits/splits_metadata.json`.

### Inference

```bash
python fsr_predict.py \
  --input /path/to/new_conditions.csv \
  --output predictions.csv \
  --champion extrapolation
```

Optional: `--artifact /path/to/artifacts/extrapolation_champion`. Output includes `predicted_fsr`, optional `observed_fsr`, paper IDs, and model / fingerprint metadata.

---

## Script cheat sheet

| Script | Role | Trains? |
|---|---|---|
| `fsr_common.py` | Shared library: load Excel/CSV, detect target/group/leakage, feature manifest, weights, metrics, GPU helpers | Library |
| `fsr_models.py` | Shared library: five families + preprocessing + target transforms | Library |
| `fsr_splits.py` | Write frozen `interpolation_random` + `extrapolation_grouped` assignments | No |
| `fsr_search.py` | Nested random search on one outer **train** partition | Yes (inner only) |
| `fsr_evaluate.py` | Candidate × split × (`rd`/`rd_bt`) benchmark; writes a shard | Yes |
| `fsr_aggregate.py` | Discover shards, merge tables, bootstrap CIs, gaps, stability | No |
| `fsr_select.py` | Apply `selection_policy.yaml` → interpolation & extrapolation champions | No |
| `fsr_refit.py` | Fit chosen config (+ optional `rd_bt`) on all labeled rows | Yes |
| `fsr_report.py` | Numbered report directories, diagnostics, explainability | Reads artifacts |
| `fsr_predict.py` | Score new files with a refit champion | Loads artifact |

Full detail: [`pipeline.md`](pipeline.md).

---

## Candidates (34)

Defined in `configs/candidates.yaml` (rationale in `configs/candidates.md` — note the YAML is authoritative if counts differ):

| Family | Count | Themes |
|---|---:|---|
| XGBoost | 10 | Baseline, paper weights, absolute / Pseudo-Huber loss, `signed_log1p` target, paper bagging |
| Random forest | 7 | Baseline, paper weights, absolute-error criterion, log target |
| KNN | 6 | Uniform vs distance × paper weights |
| MLP | 5 | Baseline, paper weights, log target |
| SVR | 6 | Linear vs RBF × paper weights |

Each candidate is a **recipe**: family + `paper_weight` (`none` / `sqrt` / `inverse`) + optional `target_transform`, `fixed_params`, `paper_bagging`.

Augmentation is **not** a separate YAML candidate: every evaluated candidate is scored under both `rd` and `rd_bt` (1000 bootstrap training rows by default). Selection treats `(candidate, augmentation)` as the unit.

---

## Data assumptions (must know)

- Default local file: `Microgravity_Database.csv` in this folder. Loader also accepts Excel and resolves paths via `locate_data_file`.
- CSV layout: category banner row + real header; Excel may use a two-level header. The parse that exposes an FSR-like column wins.
- Target: auto-detected FSR column → internal name `fsr`. Training drops rows without a numeric target.
- Paper ID: normalized DOI (`doi::…`) else normalized citation (`citation::…`).
- Dropped on load: Material of Sample, Rig Name, Internal Geometry, Internal Dimensions, Facility, Info.
- Leakage columns auto-excluded (duplicate FSR fields, flame length, HRR, smoke, ignition yes/no, notes, author/DOI fingerprints, …).
- Remaining columns → numeric vs categorical by leading-number parsing (`"101.3 kPa"` → number; `"N2"` stays categorical).
- Data version: `fsr-data-v2`.
- Feature roles (`physics` / `apparatus`) are still tagged in the manifest for reporting, but **`feature_lists` no longer filters by role** — every candidate sees the full physics database feature list.

Numeric: median impute (XGBoost keeps native NaNs) + optional scaling. Categorical: constant impute + unknown-safe one-hot.

---

## Outputs

```text
results/
├── splits/
│   ├── split_assignments.csv|.parquet
│   ├── split_diagnostics.csv
│   ├── splits_metadata.json
│   └── data_validation_report.json
├── evaluation/
│   ├── <family>/<candidate_id>/   # shards from evaluate / Slurm array
│   └── merged/                    # aggregate output used by select/report
│       ├── fold_metrics.csv, summary_metrics.csv, …
│       ├── outer_fold_predictions.parquet
│       ├── bootstrap_intervals.csv, generalization_gap.csv, …
│       ├── integrity_checks.json
│       └── explainability_models/
├── selection.json
├── selection.md
└── report/
    ├── 00_overview/ … 07_explainability/
    ├── README.md
    └── report_manifest.json

artifacts/
├── interpolation_champion/   # model.joblib, model_card.json, …
└── extrapolation_champion/
```

**Evaluation / merged artifacts** = unbiased comparison evidence.  
**Refit artifacts** = production models (not nested-CV scores).

---

## HPC (NYU Greene / Singularity)

Scripts assume (overridable with env vars):

| Variable | Typical default |
|---|---|
| `FSR_DATA` | Folder or repo `Microgravity_Database.csv` (check each `.slurm`) |
| `FSR_OVERLAY` | `/scratch/me3144/ignition_hpc/ignition_python.ext3` |
| `FSR_IMAGE` | `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` |
| `FSR_N_REPEATS` | `10` |
| `FSR_SEARCH_ITERATIONS` | `40` |
| `FSR_INNER_FOLDS` | `5` |
| `FSR_BOOTSTRAP_ITERATIONS` | `2000` |

Submit **from** the `FSR Regressors` directory:

```bash
mkdir -p logs

SPLITS=$(sbatch --parsable run_regressors_splits.slurm)
ARRAY=$(sbatch --parsable --dependency=afterok:${SPLITS} run_candidates_array.slurm)
sbatch --dependency=afterok:${ARRAY} run_regressors_finalize.slurm
```

- `run_candidates_array.slurm` — `#SBATCH --array=0-33`, one task per candidate → `results/evaluation/<family>/<candidate_id>/`
- `run_regressors_finalize.slurm` — `fsr_aggregate` → `fsr_select` → both `fsr_refit` → `fsr_report`

---

## Reproducibility

- Seed **42** everywhere; repeats use consecutive seeds.
- Splits store dataset SHA-256, target/group columns, feature manifest, exact memberships.
- Evaluate aborts on dataset fingerprint mismatch.
- Search histories, selected hyperparameters, and integrity failures are persisted per shard.

---

## Common pitfalls for new contributors

1. **Wrong order** — splits → evaluate shards → **aggregate** → select → refit → report. Select/report need `results/evaluation/merged/`.
2. **Confusing champions** — interpolation ≠ extrapolation. Extrapolation is the primary scientific claim.
3. **Calling `rd_bt` “new data”** — bootstrap copies existing training rows (same papers); it does not invent physics.
4. **Mixing datasets between splits and evaluate** — SHA-256 must match.
5. **Presenting merged metrics as the deployed model** — only `artifacts/*_champion` are deployable.
6. **Stale README mental model** — families are XGBoost / RF / KNN / MLP / SVR (not decision-tree-only); parallelism is **per candidate** (`0–33`), not only four family GPUs.

---

## Where to go next

1. Read [`pipeline.md`](pipeline.md) once end-to-end.
2. Skim [`configs/candidates.yaml`](configs/candidates.yaml) and [`configs/selection_policy.yaml`](configs/selection_policy.yaml).
3. Run `fsr_splits.py` locally and inspect `results/splits/`.
4. Run **one** `--candidate-id` evaluation with fewer `--search-iterations` before launching the full array.
