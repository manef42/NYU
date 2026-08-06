# Flame-spread-rate regression report

Two questions are answered separately and must never be conflated. The random
row split measures interpolation inside papers the model has already partly seen.
The paper-grouped split holds out entire publications and is the only evidence
about transfer to a brand-new campaign or rig.

## Selected champions

- Interpolation: `rf_baseline` (`rd`),
  R² 0.8790 ± 0.0178,
  RMSE 0.0063 ± 0.0005
- Extrapolation: `rf_baseline` (`rd`),
  R² 0.4576 ± 0.2215,
  RMSE 0.0122 ± 0.0041

## Dataset

`/home/me3144/FSR Regressors/Microgravity_Database.csv` → 2531 rows with a usable
flame spread rate across 66 canonical papers; target
column `FSR (m/s)`, grouping column
`Article (MLA)`; SHA-256 `6e09a99e4ea69148e69151bd46149be4140d181b2b8294106927ea9499002fc8`.

## Integrity

Evaluation integrity overall: `True`. Failed candidate/protocol
combinations are excluded from every table and documented in
`../evaluation/integrity_checks.json`.

## Output folders

| Folder | Contents |
|--------|----------|
| `00_overview/` | Dataset summary, leaderboards, gap/stability/bootstrap tables |
| `01_comparison/` | Candidate bar chart, strip plot, gap plot, augmentation plot |
| `02_diagnostics/` | Per-family pred-vs-true, error-vs-true, residuals, error histogram |
| `03_error_analysis/` | Error vs top-10 features, O₂ × pressure heatmap |
| `04_hard_papers/` | Top-10 hardest papers table + bar chart |
| `05_learning_curves/` | Training-size vs RMSE and R² per protocol |
| `06_per_paper/` | Per-paper RMSE/MAE/R² histograms and boxplots |
| `07_explainability/` | Feature importance, permutation importance, SHAP |

Evaluation artifacts are unbiased comparison evidence, not deployable models.
Refit artifacts are deployable models, not unbiased evaluation evidence.
