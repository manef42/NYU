# FSR regression pipeline (paper-aware, GPU-parallel)

This directory is the reconstruction of the single-script study in `../FSR Regression 3`
as one reproducible, leakage-controlled, Slurm-parallel pipeline for predicting **flame
spread rate (FSR)** from the microgravity-combustion literature database. The science is
unchanged - four model families, two evaluation strategies, bootstrap augmentation - but
every stage is now a separate, restartable script that writes immutable artifacts, and the
expensive stage fans out over GPUs as a Slurm array.

Two deliberately different questions are answered and never conflated:

- **Interpolation** (`interpolation_random`) uses target-stratified row-level holdouts.
  Test rows may come from papers that are also represented in training, so the model only
  has to interpolate inside campaigns it has already partly seen. This is the optimistic
  baseline.
- **Extrapolation** (`extrapolation_grouped`) holds out complete canonical papers with a
  balance-aware `GroupShuffleSplit`. No paper is ever split across partitions, so this is
  the only evidence about transfer to a brand-new paper, campaign or rig. **This is the
  primary scientific result.**

## Scripts

| Script | Role |
| --- | --- |
| `fsr_common.py` | Dataset loading (Excel or CSV), canonical paper identity, automatic target/leakage/feature-role detection, preprocessing contract, paper weighting, bootstrap augmentation, metrics, GPU/Slurm helpers. |
| `fsr_splits.py` | Materializes and validates all outer splits once (row coverage, partition emptiness, paper leakage) plus per-repeat balance diagnostics. |
| `fsr_models.py` | Model registry and one wrapper: GPU XGBoost, cuML-or-scikit-learn KNN, decision tree, and a PyTorch MLP with single-GPU or NCCL `DistributedDataParallel` training. |
| `fsr_search.py` | Nested random search on one outer training partition, parallelized over the (configuration x inner fold) grid with joblib. |
| `fsr_evaluate.py` | The sole benchmark runner. `--model-id` restricts it to one family so the Slurm array can run the four families concurrently; writes a self-contained shard. |
| `fsr_aggregate.py` | Owns every derived table (fold summaries, per-paper breakdown, paired tests, bootstrap intervals, generalization gap, augmentation effect, stability) and merges the array shards. |
| `fsr_select.py` | Applies `configs/selection_policy.yaml` and picks the interpolation and extrapolation champions. Trains nothing. |
| `fsr_refit.py` | Rebuilds a champion on all labelled rows, recomputes honest out-of-fold predictions, and packages a deployable artifact with a model card. |
| `fsr_report.py` | Publication tables and figures, permutation importance and SHAP, built exclusively from persisted artifacts. |
| `fsr_predict.py` | Inference for new conditions with a refitted champion. |

Slurm entry points: `run_regressors_splits.slurm`, `run_regressors_array.slurm`
(`--array=0-3`), `run_regressors_finalize.slurm`.

## Data assumptions

The default dataset is `../Microgravity_Database.xlsm`; `../Microgravity_Database_reduced.csv`
also works. The workbook stores a two-level "section / field" header and the CSV export
stores a category banner above the real header, so both layouts are attempted and the parse
that actually exposes a flame-spread-rate column wins. Text cells are normalized, the
common missing tokens are mapped to `NaN`, and unit-laden cells such as `"101.3 kPa"` or
`"305 mm diameter"` are parsed to numbers - but only when the cell *begins* with a number,
so labels such as `N2` and `CO2` stay categorical instead of silently becoming `2`.

Column roles are detected, not hard-coded:

- the target is the FSR-like column with the most numeric values;
- the grouping column is the most complete citation-like column, and the canonical paper ID
  is a normalized true DOI with normalized citation identity as fallback, so the same paper
  written two ways is one group;
- duplicate target representations, post-experiment outcomes (flame length, HRR, smoke,
  extinction), free-text notes and paper fingerprints (authors, DOI, article) are dropped
  before modelling;
- remaining features are split into numeric and categorical, and each is tagged `physics`
  (thermophysical, flow, gravity, geometry, gas composition) or `apparatus` (rig, facility,
  internal duct geometry, ignition hardware). The `all` feature set uses both; the
  `physics` feature set drops apparatus descriptors, which is the transferable subset.

Numeric features are median-imputed (XGBoost keeps its native missing-value handling) and
standardized for distance- and gradient-based models; categoricals use constant imputation
and unknown-safe one-hot encoding so categories that appear only in held-out papers cannot
break inference.

## Protocol and leakage controls

`fsr_splits.py` writes exact row and paper assignments once, and every model family
benchmarks the same partitions. The grouped holdout screens many candidate paper partitions
and keeps the one whose train and test target distributions agree best, which removes the
nuisance variance that makes single-shot grouped R2 swing between repeats without weakening
the constraint that no paper is ever split.

`fsr_evaluate.py` tunes hyper-parameters **inside each outer training partition only**.
Extrapolation folds use paper-disjoint `GroupKFold` inner folds, interpolation folds use
shuffled `KFold` inner folds, and imputation, scaling, one-hot vocabularies, weighting and
fitting therefore never see an outer test row. Each selected configuration is then refit
under two data conditions:

- `rd` - the real training rows;
- `rd_bt` - the real training rows plus `bootstrap_rows` rows drawn with replacement from
  the training partition (Rivera et al., Sec. 3.3). Each copy inherits the paper of its
  source row, and test rows are never augmented, so augmentation cannot leak unseen-paper
  information.

Paper weighting (`none`, `sqrt`, `inverse`, `log`) counteracts papers that contribute many
correlated rows. XGBoost, decision trees and the PyTorch MLP consume sample weights at fit
time; KNN cannot, so it receives a deterministic probability-proportional resample of its
training fold. No model silently ignores the requested weighting policy.

Failed candidate/protocol combinations are recorded in `integrity_checks.json` and excluded
from every result table rather than partially reported.

## Parallelization and GPU use

- **Stage level**: split generation, the per-family benchmark, and finalization are separate
  jobs, so a failure never forces a full re-run.
- **Family level**: `run_regressors_array.slurm` runs `--array=0-3`, one GPU each, mapping
  `0=XGBoost, 1=KNN, 2=Decision Tree, 3=MLP`. Each task writes
  `results/evaluation/model_<task>` and `fsr_aggregate.py` merges the shards.
- **Search level**: the (configuration x inner fold) grid is executed through joblib. CPU
  families spread folds across the allocated cores; GPU families keep one fit at a time per
  worker and the array script starts CUDA MPS so concurrent GPU work shares the device
  predictably.
- **Model level**: XGBoost uses the CUDA histogram builder when a GPU is visible, KNN uses
  cuML when it is installed and compatible, and the MLP trains in PyTorch on one GPU or
  across every visible GPU with NCCL `DistributedDataParallel`.

Every accelerator path degrades to CPU automatically, so the exact same commands reproduce
results off the cluster.

## Installation

Python 3.11 or newer is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cluster run

```bash
cd "FSR Regressors"
SPLITS=$(sbatch --parsable run_regressors_splits.slurm)
ARRAY=$(sbatch --parsable --dependency=afterok:${SPLITS} run_regressors_array.slurm)
sbatch --dependency=afterok:${ARRAY} run_regressors_finalize.slurm
```

Override the defaults with `FSR_DATA`, `FSR_N_REPEATS`, `FSR_SEARCH_ITERATIONS`,
`FSR_INNER_FOLDS`, `FSR_BOOTSTRAP_ITERATIONS`, `FSR_OVERLAY` and `FSR_IMAGE`.

## Local run (same pipeline, no Slurm)

```bash
cd "FSR Regressors"

python fsr_splits.py \
  --data ../Microgravity_Database.xlsm \
  --out results/splits \
  --n-repeats 10 --test-size 0.2 --split-candidates 50

for MODEL_ID in 0 1 2 3; do
  python fsr_evaluate.py \
    --data ../Microgravity_Database.xlsm \
    --splits results/splits \
    --config configs/candidates.yaml \
    --out "results/evaluation/model_${MODEL_ID}" \
    --search-iterations 40 --inner-folds 5 \
    --augmentations rd,rd_bt --model-id "${MODEL_ID}"
done

python fsr_aggregate.py --evaluation results/evaluation --out results/evaluation/merged

python fsr_select.py \
  --evaluation results/evaluation/merged \
  --policy configs/selection_policy.yaml \
  --out results/selection.json

python fsr_refit.py --data ../Microgravity_Database.xlsm \
  --selection results/selection.json --evaluation results/evaluation/merged \
  --champion interpolation --out artifacts/interpolation_champion
python fsr_refit.py --data ../Microgravity_Database.xlsm \
  --selection results/selection.json --evaluation results/evaluation/merged \
  --champion extrapolation --out artifacts/extrapolation_champion

python fsr_report.py --data ../Microgravity_Database.xlsm \
  --evaluation results/evaluation/merged --selection results/selection.json \
  --artifacts artifacts --out results/report
```

This is intentionally expensive: every candidate is tuned inside every outer training fold,
both protocols and both data conditions are evaluated, and uncertainty uses paper-cluster
bootstrap resampling. Runtime scales with paper count, candidate count, repeats and search
iterations.

## Reproducibility

The seed is 42 everywhere; repeated splits use consecutive seeds. Split files store the
dataset SHA-256, the target and grouping columns, the feature manifest and exact
memberships, and `fsr_evaluate.py` refuses to run against a dataset whose fingerprint
differs from the one used to build the splits. Candidate spaces live in `fsr_models.py`,
sampled configurations and full search histories are persisted, and model, resampling,
XGBoost, PyTorch and report seeds are recorded.

## Output map

```text
results/
├── splits/                    # assignments, diagnostics, metadata, data validation report
├── evaluation/model_<task>/   # one shard per model family (Slurm array task)
├── evaluation/merged/         # merged predictions, metrics, CIs, integrity, comparisons
├── selection.json             # machine-readable champion decisions
├── selection.md               # ranked human-readable decisions
└── report/                    # publication CSV/Markdown tables and PNG figures
artifacts/
├── interpolation_champion/    # deployable model, card, manifest, fingerprint, OOF predictions
└── extrapolation_champion/
```

Generated results, models and logs are ignored by Git; only `.gitkeep` placeholders are
versioned.

## Inference

```bash
python fsr_predict.py \
  --input ../new_conditions.csv \
  --output predictions.csv \
  --champion extrapolation
```

Use `--artifact /path/to/artifact` to override the artifact directory. The output has one
row per input row with the predicted FSR, the observed FSR when present, paper identity, and
model/fingerprint metadata.

## What changed relative to `../FSR Regression 3`

The reconstruction preserves the study design and adds the following:

1. One 2,000-line script became ten focused modules plus three Slurm entry points, so the
   heavy stage is restartable and shardable.
2. The benchmark fans out over four GPUs as a Slurm array; searches parallelize over the
   (configuration x fold) grid; XGBoost, KNN and the MLP now use the GPU.
3. The scikit-learn MLP was replaced by a PyTorch MLP that trains on one GPU or on every
   visible GPU with NCCL DDP, and that honours sample weights through a weighted MSE.
4. Outer splits are materialized once, validated for coverage and paper leakage, and shared
   by every family instead of being rebuilt inside each repeat.
5. Hyper-parameters are tuned inside each protocol's own training partition. The original
   script tuned once with grouped CV and reused those settings for the random-split
   condition, which let information from the random split's test rows influence its own
   model selection.
6. Paper identity is canonical (normalized DOI with citation fallback) rather than raw
   citation text, and near-duplicate rows are removed, so grouped splits cannot be defeated
   by two spellings of one paper.
7. Numeric detection only accepts cells that begin with a number, which keeps chemical
   labels such as `N2` and `CO2` categorical.
8. Candidates are declarative (`configs/candidates.yaml`): feature set, paper weighting,
   target transform, XGBoost objective variant, monotone oxygen constraint and paper-cluster
   bagging. Champion choice is a declarative policy (`configs/selection_policy.yaml`) with
   explicit rejection rules and tie-breakers instead of a single hard-coded metric.
9. Bootstrap augmentation is a first-class evaluated condition (`rd` versus `rd_bt`) whose
   copies inherit the paper of their source row.
10. Uncertainty uses paper-cluster bootstrap intervals and paired Wilcoxon comparisons over
    identical outer folds, and refit artifacts ship a model card with data fingerprints and
    documented limitations.

Evaluation artifacts are unbiased comparison evidence, not trained production models. Refit
artifacts are trained production models, not unbiased evaluation evidence. Random-split
results must never be presented as unseen-paper generalization.
