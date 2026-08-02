# FSR Regression Benchmark Design

## Philosophy

The benchmark is designed to compare fundamentally different regression paradigms while keeping the comparison interpretable.

Rather than creating dozens of arbitrary model variants, each candidate tests one clear hypothesis:

- Does paper weighting improve generalization?
- Does a more robust loss improve performance?
- Does predicting a transformed target help?
- Which machine learning paradigm best models flame spread rate?

The objective is not only to identify the best-performing model, but also to understand *why* a particular approach generalizes better to unseen papers.

---

# Model Families

Five complementary regression families are benchmarked.

| Family | Learning principle |
|---------|--------------------|
| XGBoost | Gradient-boosted decision trees |
| Random Forest | Bagged decision trees |
| KNN | Instance-based learning |
| MLP | Neural network function approximation |
| SVR | Kernel-based regression |

Together these cover the main classes of machine learning algorithms commonly applied to structured scientific datasets.

---

# Paper Weighting

Three paper-weighting strategies are evaluated.

| Strategy | Motivation |
|----------|------------|
| none | Standard empirical risk minimization. Every sample contributes equally. |
| sqrt | Moderately reduces the influence of papers containing many experiments while retaining most of their information. |
| inverse | Gives each paper approximately equal influence regardless of the number of experiments it contains. |

Since Leave-One-Paper-Out (LOPO) evaluation is the primary measure of extrapolation performance, reducing the dominance of large experimental campaigns may improve generalization to unseen publications.

---

# XGBoost Variants

XGBoost is the primary model family because gradient-boosted trees generally provide state-of-the-art performance on heterogeneous tabular scientific datasets.

Three additional ideas are investigated.

## Robust loss functions

Three regression objectives are compared.

- squarederror
- absoluteerror
- pseudohuber

The objective is to evaluate robustness to experimental noise and outliers.

Absolute Error reduces sensitivity to large residuals.

Pseudo-Huber combines the smooth optimization properties of squared error with the robustness of absolute error.

---

## Target transformation

Some flame spread rate distributions are highly skewed.

Two models therefore predict

signed_log1p(FSR)

instead of the raw target.

This investigates whether compressing large values improves optimization and generalization.

---

## Paper Bagging

One additional XGBoost candidate trains multiple models using paper-level bootstrap sampling.

Predictions are averaged across the ensemble.

The goal is to reduce dependence on individual laboratories or experimental campaigns and improve robustness under LOPO evaluation.

---

# Random Forest

Random Forest serves as the natural ensemble-tree baseline.

Unlike XGBoost, it relies solely on bootstrap aggregation rather than sequential boosting.

This comparison isolates the benefit of gradient boosting over classical bagging while keeping the underlying tree representation similar.

---

# K-Nearest Neighbors

KNN represents local, instance-based regression.

Two neighbor weighting schemes are evaluated.

- uniform
- distance

Distance weighting gives closer experiments greater influence and is expected to perform better when nearby operating conditions are more representative than distant ones.

---

# Multi-Layer Perceptron

The MLP represents continuous nonlinear function approximation.

Besides the baseline and paper-weighting variants, two log-target models are included.

Neural networks often optimize more effectively on compressed target distributions when the response spans multiple orders of magnitude.

---

# Support Vector Regression

SVR provides a fundamentally different learning strategy based on kernel methods.

Two kernels are compared.

## Linear kernel

Evaluates whether the engineered physics-based features are already approximately linearly predictive.

A strong performance would indicate that feature engineering captures much of the underlying combustion physics.

## RBF kernel

Introduces nonlinear decision functions through a Gaussian kernel.

This evaluates whether additional nonlinear relationships remain after feature engineering.

---

# Overall Benchmark Structure

The benchmark contains 30 candidates.

| Family | Candidates |
|---------|-----------:|
| XGBoost | 10 |
| Random Forest | 3 |
| KNN | 6 |
| MLP | 5 |
| SVR | 6 |
| **Total** | **30** |

This benchmark simultaneously evaluates:

- algorithm family,
- paper-weighting strategy,
- robustness to outliers,
- target transformation,
- local versus global regression,
- ensemble versus neural approaches,
- kernel-based versus tree-based learning.

The resulting comparison is intended to provide both the highest-performing predictive model and scientific insight into which inductive biases best capture the underlying physics governing flame spread rate.