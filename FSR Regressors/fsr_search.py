"""Leakage-safe nested hyper-parameter search for one outer training partition.

Every configuration is scored only on inner folds carved out of the outer
*training* rows, and the inner folds respect the protocol under test: paper-
disjoint ``GroupKFold`` folds for the extrapolation question and shuffled
``KFold`` folds for the interpolation baseline. Imputation, scaling, one-hot
vocabularies, weighting and model fitting therefore never see an outer test row.

The (configuration x inner fold) grid is embarrassingly parallel and is executed
through joblib: CPU families spread folds across the allocated cores, while GPU
families keep one fit at a time per worker so that the accelerator is shared
predictably (CUDA MPS makes concurrent GPU fits safe when the Slurm script
enables it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold

from fsr_common import METRIC_KEYS, regression_metrics
from fsr_models import MODEL_REGISTRY, make_model

# Families that run exclusively on CPU and are safe to parallelize across cores.
# GPU families (xgboost, mlp) keep parallel_jobs=1 to avoid CUDA contention.
CPU_PARALLEL_FAMILIES = ("knn", "random_forest", "svr")
SELECTION_ORDER = ("RMSE", "MAE", "R2")


@dataclass
class SearchResult:
    selected_params: dict[str, Any]
    cv_metrics: dict[str, float]
    history: pd.DataFrame
    inner_predictions: pd.DataFrame


def sample_configurations(candidate: dict[str, Any], iterations: int,
                          seed: int) -> list[dict[str, Any]]:
    """Deterministically draw distinct configurations from the candidate space."""
    spec = MODEL_REGISTRY[candidate["model_family"]]
    space = candidate.get("search_space", spec.search_space)
    if not space:
        return [{}]
    rng = np.random.default_rng(seed)
    configurations: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts, budget = 0, max(1, int(iterations)) * 100
    while len(configurations) < max(1, int(iterations)) and attempts < budget:
        configuration = {key: values[int(rng.integers(len(values)))]
                         for key, values in sorted(space.items())}
        token = json.dumps(configuration, sort_keys=True, default=str)
        if token not in seen:
            seen.add(token)
            configurations.append(configuration)
        attempts += 1
    return configurations or [{}]


def inner_splits(X: pd.DataFrame, papers: pd.Series, protocol: str, folds: int,
                 seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Protocol-appropriate inner folds over the outer training partition."""
    folds = max(2, int(folds))
    if protocol.startswith("extrapolation"):
        available = int(pd.Series(papers).nunique())
        if available < 2:
            raise ValueError("Grouped inner folds need at least two training papers")
        splitter = GroupKFold(n_splits=min(folds, available))
        splits = list(splitter.split(X, groups=papers))
        for train, validation in splits:
            if set(papers.iloc[train]) & set(papers.iloc[validation]):
                raise ValueError("Grouped inner split leaked a paper")
        return splits
    return list(KFold(n_splits=min(folds, len(X)), shuffle=True,
                      random_state=seed).split(X))


def _fit_inner_fold(candidate: dict[str, Any], params: dict[str, Any],
                    manifest: list[dict[str, str]], X: pd.DataFrame, y: np.ndarray,
                    papers: pd.Series, configuration_id: int, fold: int,
                    train: np.ndarray, validation: np.ndarray
                    ) -> tuple[int, int, np.ndarray, dict[str, float]]:
    model = make_model(candidate, params, manifest)
    model.fit(X.iloc[train], y[train], papers.iloc[train])
    prediction = model.predict(X.iloc[validation])
    return (configuration_id, fold, prediction,
            regression_metrics(y[validation], prediction))


def nested_search(candidate: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
                  papers: pd.Series, manifest: list[dict[str, str]], protocol: str,
                  iterations: int, folds: int, seed: int,
                  n_jobs: int = -1) -> SearchResult:
    """Tune inside one outer training partition and return the selected settings."""
    X = X.reset_index(drop=True)
    papers = pd.Series(papers).reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    splits = inner_splits(X, papers, protocol, folds, seed)
    configurations = sample_configurations(candidate, iterations, seed)
    family = candidate["model_family"]
    parallel_jobs = n_jobs if family in CPU_PARALLEL_FAMILIES else 1
    jobs = [(configuration_id, fold, params, train, validation)
            for configuration_id, params in enumerate(configurations)
            for fold, (train, validation) in enumerate(splits)]
    results = Parallel(n_jobs=parallel_jobs, backend="threading")(
        delayed(_fit_inner_fold)(candidate, params, manifest, X, y, papers,
                                 configuration_id, fold, train, validation)
        for configuration_id, fold, params, train, validation in jobs)

    out_of_fold = {index: np.full(len(y), np.nan) for index in range(len(configurations))}
    fold_metrics: dict[int, list[tuple[int, dict[str, float]]]] = {
        index: [] for index in range(len(configurations))}
    for configuration_id, fold, prediction, metrics in results:
        out_of_fold[configuration_id][splits[fold][1]] = prediction
        fold_metrics[configuration_id].append((fold, metrics))

    history_rows = []
    for configuration_id, params in enumerate(configurations):
        if np.isnan(out_of_fold[configuration_id]).any():
            raise ValueError("Inner out-of-fold predictions are incomplete")
        parameters = json.dumps(params, sort_keys=True, default=str)
        records = []
        for fold, metrics in sorted(fold_metrics[configuration_id], key=lambda item: item[0]):
            records.append(metrics)
            history_rows.append({"configuration_id": configuration_id, "inner_fold": str(fold),
                                 "parameters": parameters, **metrics})
        means = pd.DataFrame(records).mean(numeric_only=True).to_dict()
        history_rows.append({"configuration_id": configuration_id, "inner_fold": "mean",
                             "parameters": parameters, **means})
    history = pd.DataFrame(history_rows)
    means = history[history["inner_fold"] == "mean"].copy()
    means = means.sort_values(list(SELECTION_ORDER), ascending=[True, True, False],
                              kind="stable")
    best = int(means.iloc[0]["configuration_id"])
    cv_metrics = {key: float(means.iloc[0][key]) for key in METRIC_KEYS
                  if key in means.columns}
    inner_predictions = pd.DataFrame({
        "row_id": X["row_id"].to_numpy(), "paper_id": papers.to_numpy(),
        "y_true": y, "y_pred": out_of_fold[best]})
    return SearchResult(configurations[best], cv_metrics, history, inner_predictions)
