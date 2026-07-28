"""Refit a selected champion on all labelled rows and package a deployable model.

Evaluation artifacts are unbiased comparison evidence, not deployable models.
This script produces the complement: the champion configuration is rebuilt on
every labelled row, honest out-of-fold predictions are recomputed under the
matching protocol for the model card, and the fitted pipeline is written together
with its feature manifest, data fingerprint and known limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from fsr_common import (DATA_VERSION, RANDOM_STATE, TARGET, bootstrap_augment,
                        configure_torch, empty_cuda_cache, load_data,
                        regression_metrics)
from fsr_models import make_model

OOF_SEEDS = 3
OOF_FOLDS = 5


def _candidate(selection: dict, champion: str) -> tuple[dict, dict]:
    key = f"{champion}_champion"
    if key not in selection:
        raise ValueError(f"Selection file has no {key}")
    chosen = selection[key]
    configuration = chosen["candidate_configuration"]
    candidate = {
        "candidate_id": chosen["candidate_id"],
        "model_family": chosen["model_family"],
        "feature_set": chosen["feature_set"],
        "paper_weight": configuration.get("paper_weight") or "none",
        "target_transform": configuration.get("target_transform") or "identity",
        "monotone_oxygen": bool(configuration.get("monotone_oxygen") or False),
        "paper_bagging": int(configuration.get("paper_bagging") or 0),
        "fixed_params": {},
    }
    return candidate, chosen


def _folds(champion: str, df: pd.DataFrame, papers: pd.Series, seed: int
           ) -> list[tuple[np.ndarray, np.ndarray]]:
    if champion == "interpolation":
        return list(KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=seed).split(df))
    splits = min(OOF_FOLDS, int(papers.nunique()))
    try:
        splitter = GroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    except TypeError:  # scikit-learn versions without a shuffled GroupKFold
        splitter = GroupKFold(n_splits=splits)
    return list(splitter.split(df, groups=papers))


def _out_of_fold(df: pd.DataFrame, candidate: dict, params: dict,
                 manifest: list[dict[str, str]], champion: str, augmentation: str,
                 bootstrap_rows: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    y = df[TARGET].to_numpy(dtype=float)
    papers = df["paper_id"].reset_index(drop=True)
    sums = np.zeros(len(df))
    counts = np.zeros(len(df), dtype=int)
    records: list[dict[str, Any]] = []
    for offset in range(OOF_SEEDS):
        split_seed = seed + offset
        for fold, (train, validation) in enumerate(
                _folds(champion, df, papers, split_seed)):
            if champion == "extrapolation" and (
                    set(papers.iloc[train]) & set(papers.iloc[validation])):
                raise ValueError("Paper leakage in the extrapolation refit OOF split")
            fit_X = df.iloc[train].reset_index(drop=True)
            fit_y = y[train]
            fit_papers = papers.iloc[train].reset_index(drop=True)
            if augmentation == "rd_bt":
                fit_X, fit_y, source = bootstrap_augment(
                    fit_X, fit_y, bootstrap_rows, split_seed * 100 + fold)
                fit_papers = fit_papers.iloc[source].reset_index(drop=True)
            model = make_model(candidate, params, manifest)
            model.fit(fit_X, fit_y, fit_papers)
            prediction = model.predict(df.iloc[validation])
            sums[validation] += prediction
            counts[validation] += 1
            records.extend({
                "row_id": df.iloc[index]["row_id"], "paper_id": papers.iloc[index],
                "seed": split_seed, "fold": fold, "y_true": float(y[index]),
                "y_pred": float(prediction[local]),
            } for local, index in enumerate(validation))
            empty_cuda_cache()
    if np.any(counts == 0):
        raise RuntimeError("Refit out-of-fold predictions do not cover every row")
    return sums / counts, pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--champion", choices=["interpolation", "extrapolation"],
                        required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evaluation", help="Evaluation directory used for provenance.")
    parser.add_argument("--bootstrap-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    configure_torch()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    candidate, chosen = _candidate(selection, args.champion)
    params = chosen["exact_hyperparameters"]
    augmentation = chosen.get("augmentation", "rd")
    df, report = load_data(args.data, require_target=True)
    manifest = report["feature_manifest"]
    bootstrap_rows = args.bootstrap_rows
    evaluation_metadata: dict[str, Any] = {}
    if args.evaluation:
        path = Path(args.evaluation) / "evaluation_metadata.json"
        if path.is_file():
            evaluation_metadata = json.loads(path.read_text(encoding="utf-8"))
            shards = evaluation_metadata.get("shard_metadata") or [evaluation_metadata]
            recorded = [shard.get("bootstrap_rows") for shard in shards
                        if shard.get("bootstrap_rows") is not None]
            if bootstrap_rows is None and recorded:
                bootstrap_rows = int(recorded[0])
    if bootstrap_rows is None:
        bootstrap_rows = 1000

    oof_prediction, oof_records = _out_of_fold(
        df, candidate, params, manifest, args.champion, augmentation, bootstrap_rows,
        args.random_state)
    oof_records.to_csv(out / "out_of_fold_predictions.csv", index=False)
    oof_metrics = regression_metrics(df[TARGET].to_numpy(dtype=float), oof_prediction)

    y = df[TARGET].to_numpy(dtype=float)
    fit_X, fit_y = df.reset_index(drop=True), y
    fit_papers = df["paper_id"].reset_index(drop=True)
    if augmentation == "rd_bt":
        fit_X, fit_y, source = bootstrap_augment(fit_X, fit_y, bootstrap_rows,
                                                 args.random_state)
        fit_papers = fit_papers.iloc[source].reset_index(drop=True)
    model = make_model(candidate, params, manifest)
    model.fit(fit_X, fit_y, fit_papers)
    joblib.dump(model, out / "model.joblib", compress=3)
    empty_cuda_cache()

    fingerprint = {
        "dataset_sha256": report["dataset_sha256"], "data_version": DATA_VERSION,
        "training_row_ids_sha256": hashlib.sha256(
            "\n".join(df["row_id"]).encode()).hexdigest(),
        "training_rows": int(len(df)), "training_papers": int(df["paper_id"].nunique()),
        "target_column": report["target_column"], "group_column": report["group_column"],
    }
    (out / "training_data_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8")
    (out / "feature_manifest.json").write_text(json.dumps({
        "data_version": DATA_VERSION, "selected_feature_set": candidate["feature_set"],
        "model_features": model.features, "features": manifest}, indent=2), encoding="utf-8")
    card = {
        "artifact_version": "fsr-model-v1", "champion": args.champion,
        "model_id": chosen["candidate_id"], "model_family": chosen["model_family"],
        "feature_set": chosen["feature_set"], "augmentation": augmentation,
        "hyperparameters": params,
        "weighting_policy": {
            "paper_weight": candidate["paper_weight"],
            "unsupported_fit_weight_strategy": "deterministic weighted resampling"},
        "target_transform": candidate["target_transform"],
        "bootstrap_rows": bootstrap_rows if augmentation == "rd_bt" else 0,
        "training_rows": int(len(df)), "training_papers": int(df["paper_id"].nunique()),
        "data_fingerprint": fingerprint,
        "full_data_out_of_fold_metrics": oof_metrics,
        "evaluation_summary": {
            "selected_champion": chosen["metrics"],
            "interpolation": selection["interpolation_champion"]["metrics"],
            "extrapolation": selection["extrapolation_champion"]["metrics"]},
        "intended_use": ("Point prediction of flame spread rate for conditions matching "
                         "the documented database schema; use the champion that matches "
                         "the scientific question."),
        "known_limitations": [
            "The database aggregates heterogeneous literature campaigns with missing fields.",
            "Random-split scores do not establish unseen-paper generalization.",
            "Extrapolation estimates apply to campaigns represented by the source database.",
            "Bootstrap augmentation re-weights observed conditions and creates no new physics.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str),
                                         encoding="utf-8")
    metadata = {
        "champion": args.champion, "candidate": candidate, "hyperparameters": params,
        "augmentation": augmentation, "bootstrap_rows": bootstrap_rows,
        "random_state": args.random_state,
        "oof_protocol": (f"shuffled KFold({OOF_FOLDS}) repeated over {OOF_SEEDS} seeds"
                         if args.champion == "interpolation"
                         else f"GroupKFold({OOF_FOLDS}) over paper clusters repeated over "
                              f"{OOF_SEEDS} seeds"),
        "out_of_fold_metrics": oof_metrics,
        "evaluation_provenance": {
            "dataset_sha256": report["dataset_sha256"],
            "merged_shards": evaluation_metadata.get("merged_shards", []),
        },
    }
    (out / "refit_metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                             encoding="utf-8")
    print(json.dumps({"artifact": str(out), "model_id": chosen["candidate_id"],
                      "augmentation": augmentation,
                      "out_of_fold_RMSE": oof_metrics["RMSE"]}, indent=2))


if __name__ == "__main__":
    main()
