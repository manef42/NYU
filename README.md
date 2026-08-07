# NYU Microgravity Combustion ML

Entry point for this repository’s **main** branch.

The project builds reproducible, leakage-controlled machine-learning pipelines on a curated microgravity combustion literature database. Work is organized into focused folders; start with the task you care about, then read that folder’s `README.md` and `pipeline.md`.

---

## What’s in this repo

| Path | What it is | Start here |
|---|---|---|
| [`Ignition Classifiers/`](Ignition%20Classifiers/) | Binary **ignition** classification (Yes/No) across 26 candidates and 5 model families | [`Ignition Classifiers/README.md`](Ignition%20Classifiers/README.md) → [`pipeline.md`](Ignition%20Classifiers/pipeline.md) |
| [`FSR Regressors/`](FSR%20Regressors/) | Continuous **flame spread rate (FSR)** regression across 34 candidates and 5 model families | [`FSR Regressors/README.md`](FSR%20Regressors/README.md) → [`pipeline.md`](FSR%20Regressors/pipeline.md) |
| [`metrics/`](metrics/) | Exploratory dataset figures and summary plots (`generate_metrics.py`) | Run scripts in that folder after skimming outputs |
| `Microgravity_Database_converted.xlsx` | Shared Excel export of the literature database (pipelines also ship their own CSV copies) | Prefer the CSV inside each pipeline folder unless you need Excel |

Each modeling folder is largely **self-contained**: its own scripts, configs, requirements, SLURM launchers, local CSV, and (when generated) `results/` + `artifacts/`.

---

## Two scientific questions (both pipelines)

Both pipelines deliberately separate:

1. **Interpolation** — predict new rows that may come from papers already partly seen in training (optimistic baseline).
2. **Extrapolation** — hold out entire papers / campaigns (primary evidence for transfer to unseen work).

Never quote interpolation scores as unseen-paper generalization. Details and run commands live in each folder’s docs.

---

## New here? Suggested path

1. Skim this page to pick **classification** vs **regression**.
2. Open that folder’s **README** (install + how to run).
3. Read its **pipeline.md** (what every script does).
4. Run **splits only** first to validate your environment, then a single model/candidate before a full cluster job.

### Quick links

```text
Ignition Classifiers/README.md     # classify ignition
Ignition Classifiers/pipeline.md

FSR Regressors/README.md           # regress flame spread rate
FSR Regressors/pipeline.md
```

---

## Shared conventions

- Python **3.11+** recommended; each pipeline has its own `requirements.txt` and virtualenv instructions.
- Default random seed is **42**; splits and evaluation artifacts are meant to be frozen and reused.
- **Evaluation** outputs are comparison evidence; **refit** `artifacts/` are deployable models — do not conflate them.
- Cluster runs use NYU Greene / Singularity SLURM scripts inside each folder (accounts, overlays, and images are documented there and may need local adjustment).

---

## Repository layout (top level)

```text
.
├── README.md                          # you are here
├── Ignition Classifiers/              # ignition Yes/No pipeline
├── FSR Regressors/                    # flame spread rate pipeline
├── metrics/                           # dataset exploration plots
└── Microgravity_Database_converted.xlsx
```

For anything beyond this map — candidate lists, leakage controls, HPC submit order, inference — use the folder README for that pipeline.
