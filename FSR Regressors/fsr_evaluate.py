"""The sole FSR benchmark runner: nested tuning on immutable outer splits.

One invocation evaluates one or more candidates on every persisted split and
writes a self-contained shard of results. ``--model-id`` restricts the run to a
single model family, which is how the Slurm array fans the four families out over
four GPUs; ``fsr_aggregate.py`` merges the shards afterwards.

For every outer split the pipeline

1. tunes hyper-parameters only on that split's training rows
   (``fsr_search.nested_search``, protocol-appropriate inner folds),
2. refits the selected configuration once per data condition - ``rd`` (real data)
   and ``rd_bt`` (real data plus bootstrap-resampled training rows) - and
3. scores the untouched outer test rows.

Bootstrap augmentation copies training rows only, and each copy inherits the
paper of its source row, so neither augmentation nor weighting can leak held-out
paper information.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from fsr_aggregate import write_derived_tables
from fsr_common import (DATA_VERSION, RANDOM_STATE, TARGET, bootstrap_augment,
                        configure_torch, empty_cuda_cache, load_data,
                        regression_metrics)
from fsr_models import MODEL_ID_TO_FAMILY, MODEL_REGISTRY, make_model
from fsr_search import nested_search

AUGMENTATIONS = ("rd", "rd_bt")
EXPLAINABLE_FAMILIES = ("xgboost", "decision_tree")


def _load_candidates(path: str | Path) -> tuple[list[dict[str, Any]], int, int]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    identifiers = [candidate.get("candidate_id") for candidate in candidates]
    if not candidates or len(identifiers) != len(set(identifiers)) or any(
            not value for value in identifiers):
        raise ValueError("Candidate config must contain unique non-empty candidate_id values")
    for candidate in candidates:
        family = candidate.get("model_family")
        if family not in MODEL_REGISTRY:
            raise ValueError(f"Candidate {candidate['candidate_id']!r} uses unknown "
                             f"model family {family!r}")
    return (candidates, int(payload.get("random_state", RANDOM_STATE)),
            int(payload.get("bootstrap_rows", 1000)))


def _split_indices(split: pd.DataFrame, lookup: dict[str, int]
                   ) -> tuple[np.ndarray, np.ndarray]:
    train_ids = split.loc[split["partition"] == "train", "row_id"]
    test_ids = split.loc[split["partition"] == "test", "row_id"]
    unknown = (set(train_ids) | set(test_ids)) - set(lookup)
    if unknown:
        raise ValueError(f"Split references {len(unknown)} row IDs absent from the dataset")
    return (np.array([lookup[value] for value in train_ids], dtype=int),
            np.array([lookup[value] for value in test_ids], dtype=int))


def _condition_training_data(X: pd.DataFrame, y: np.ndarray, papers: pd.Series,
                            augmentation: str, bootstrap_rows: int, seed: int
                            ) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    if augmentation == "rd":
        return X.reset_index(drop=True), np.asarray(y, dtype=float), papers.reset_index(
            drop=True)
    if augmentation != "rd_bt":
        raise ValueError(f"Unknown augmentation condition {augmentation!r}")
    if bootstrap_rows <= 0:
        raise ValueError("Bootstrap augmentation requested with a non-positive row count")
    X = X.reset_index(drop=True)
    papers = papers.reset_index(drop=True)
    augmented_X, augmented_y, source = bootstrap_augment(X, y, bootstrap_rows, seed)
    return augmented_X, augmented_y, papers.iloc[source].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-iterations", type=int, default=40)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--search-jobs", type=int, default=-1,
                        help="joblib workers for the inner (configuration x fold) grid.")
    parser.add_argument("--bootstrap-rows", type=int, default=None,
                        help="Rows added by bootstrap augmentation (default: config value).")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000,
                        help="Resampling iterations for the shard confidence intervals.")
    parser.add_argument("--augmentations", default=",".join(AUGMENTATIONS))
    parser.add_argument("--protocols", default="",
                        help="Optional comma-separated protocol filter.")
    parser.add_argument("--model-id", type=int, choices=range(len(MODEL_ID_TO_FAMILY)),
                        help="Run one family: " + ", ".join(
                            f"{index}={family}" for index, family
                            in enumerate(MODEL_ID_TO_FAMILY)))
    args = parser.parse_args()
    configure_torch()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    augmentations = [name.strip() for name in args.augmentations.split(",") if name.strip()]
    unknown = set(augmentations) - set(AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unknown augmentation conditions: {sorted(unknown)}")

    candidates, base_seed, config_bootstrap_rows = _load_candidates(args.config)
    bootstrap_rows = (config_bootstrap_rows if args.bootstrap_rows is None
                      else args.bootstrap_rows)
    if bootstrap_rows <= 0 and "rd_bt" in augmentations:
        augmentations = [name for name in augmentations if name != "rd_bt"]
    indexed = list(enumerate(candidates))
    if args.model_id is not None:
        family = MODEL_ID_TO_FAMILY[args.model_id]
        indexed = [(number, candidate) for number, candidate in indexed
                   if candidate["model_family"] == family]
        if not indexed:
            raise ValueError(f"No configured candidates for model family {family!r}")

    df, data_report = load_data(args.data, require_target=True)
    split_dir = Path(args.splits)
    assignments = pd.read_parquet(split_dir / "split_assignments.parquet")
    split_metadata = json.loads((split_dir / "splits_metadata.json").read_text("utf-8"))
    if split_metadata["dataset_sha256"] != data_report["dataset_sha256"]:
        raise ValueError("Dataset fingerprint differs from the dataset used for the splits")
    manifest = split_metadata["feature_manifest"]
    if args.protocols:
        wanted = {name.strip() for name in args.protocols.split(",") if name.strip()}
        missing = wanted - set(assignments["protocol"].unique())
        if missing:
            raise ValueError(f"Requested protocols are not present in the splits: {missing}")
        assignments = assignments[assignments["protocol"].isin(wanted)]
    lookup = {row_id: index for index, row_id in enumerate(df["row_id"])}
    y_all = df[TARGET].to_numpy(dtype=float)
    split_groups = list(assignments.groupby("split_id", sort=False))

    manifest_rows, prediction_rows, metric_rows = [], [], []
    parameter_rows, history_frames, inner_frames = [], [], []
    explainability_rows, explained = [], set()
    failures: list[dict[str, Any]] = []
    status: dict[str, dict[str, Any]] = {}
    expected_folds = assignments.groupby("protocol")["split_id"].nunique().to_dict()

    for position, (candidate_number, candidate) in enumerate(indexed, start=1):
        candidate_id = candidate["candidate_id"]
        manifest_rows.append({
            **{key: value for key, value in candidate.items()
               if key not in {"fixed_params", "search_space"}},
            "fixed_params": json.dumps(candidate.get("fixed_params", {}), sort_keys=True),
            "accelerator": MODEL_REGISTRY[candidate["model_family"]].accelerator,
            "weighting_policy": json.dumps({
                "paper_weight": candidate.get("paper_weight", "none"),
                "unsupported_fit_weight_strategy": "deterministic_weighted_resampling",
            }, sort_keys=True),
        })
        for split_number, (split_id, split) in enumerate(split_groups, start=1):
            protocol = str(split["protocol"].iloc[0])
            seed = int(split["seed"].iloc[0])
            repeat = int(split["repeat"].iloc[0])
            key = f"{candidate_id}::{protocol}"
            status.setdefault(key, {
                "candidate_id": candidate_id, "protocol": protocol,
                "expected_folds": int(expected_folds.get(protocol, 0)),
                "completed_folds": 0, "valid_predictions": True, "passed": True,
                "errors": []})
            try:
                train, test = _split_indices(split, lookup)
                train_papers = set(df.iloc[train]["paper_id"])
                test_papers = set(df.iloc[test]["paper_id"])
                if protocol.startswith("extrapolation") and train_papers & test_papers:
                    raise ValueError("Outer grouped split contains paper leakage")
                X_train = df.iloc[train].reset_index(drop=True)
                X_test = df.iloc[test].reset_index(drop=True)
                y_train, y_test = y_all[train], y_all[test]
                papers_train = X_train["paper_id"]
                search_seed = base_seed + candidate_number * 100_000 + split_number
                search = nested_search(
                    candidate, X_train, y_train, papers_train, manifest, protocol,
                    args.search_iterations, args.inner_folds, search_seed,
                    n_jobs=args.search_jobs)
                selected = json.dumps(search.selected_params, sort_keys=True, default=str)
                for augmentation in augmentations:
                    fit_X, fit_y, fit_papers = _condition_training_data(
                        X_train, y_train, papers_train, augmentation, bootstrap_rows,
                        search_seed)
                    model = make_model(candidate, search.selected_params, manifest)
                    model.fit(fit_X, fit_y, fit_papers)
                    prediction = model.predict(X_test)
                    if not np.all(np.isfinite(prediction)):
                        status[key]["valid_predictions"] = False
                        raise ValueError("Model produced non-finite predictions")
                    metrics = regression_metrics(y_test, prediction)
                    metric_rows.append({
                        "split_id": split_id, "protocol": protocol, "seed": seed,
                        "repeat": repeat, "augmentation": augmentation,
                        "model_id": candidate_id, "model_family": candidate["model_family"],
                        "feature_set": candidate.get("feature_set", "all"),
                        "n_train_rows": int(len(fit_X)), "n_test_rows": int(len(test)),
                        "n_train_papers": int(len(train_papers)),
                        "n_test_papers": int(len(test_papers)),
                        **{key_: value for key_, value in metrics.items() if key_ != "n"},
                        **{f"cv_{name}": value for name, value in search.cv_metrics.items()},
                        "selected_hyperparameters": selected,
                    })
                    prediction_rows.append(pd.DataFrame({
                        "row_id": X_test["row_id"].to_numpy(),
                        "paper_id": X_test["paper_id"].to_numpy(),
                        "paper_label": X_test["paper_label"].to_numpy(),
                        "split_id": split_id, "protocol": protocol, "seed": seed,
                        "repeat": repeat, "augmentation": augmentation,
                        "model_id": candidate_id, "model_family": candidate["model_family"],
                        "feature_set": candidate.get("feature_set", "all"),
                        "y_true": y_test, "y_pred": prediction}))
                    if (augmentation == "rd" and protocol == "extrapolation_grouped"
                            and candidate["model_family"] in EXPLAINABLE_FAMILIES
                            and candidate_id not in explained):
                        directory = out / "explainability_models"
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / f"{candidate_id}.joblib"
                        joblib.dump(model, path, compress=3)
                        explainability_rows.append({
                            "model_id": candidate_id,
                            "model_family": candidate["model_family"],
                            "split_id": split_id,
                            "model_path": str(path.relative_to(out)),
                            "held_out_row_ids": json.dumps(
                                X_test["row_id"].tolist()),
                            "held_out_paper_ids": json.dumps(sorted(test_papers))})
                        explained.add(candidate_id)
                parameter_rows.append({
                    "split_id": split_id, "protocol": protocol, "seed": seed,
                    "repeat": repeat, "model_id": candidate_id,
                    "selected_hyperparameters": selected,
                    **{f"cv_{name}": value for name, value in search.cv_metrics.items()}})
                history_frames.append(search.history.assign(
                    split_id=split_id, protocol=protocol, seed=seed, repeat=repeat,
                    model_id=candidate_id))
                inner_frames.append(search.inner_predictions.assign(
                    split_id=split_id, protocol=protocol, seed=seed, repeat=repeat,
                    model_id=candidate_id))
                status[key]["completed_folds"] += 1
                print(f"[{position}/{len(indexed)}] {candidate_id} "
                      f"[{split_number}/{len(split_groups)}] {split_id}: complete",
                      flush=True)
            except Exception as exc:
                status[key]["passed"] = False
                status[key]["errors"].append(f"{split_id}: {exc}")
                failures.append({
                    "candidate_id": candidate_id, "protocol": protocol,
                    "split_id": split_id, "error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc()})
                print(f"[{position}/{len(indexed)}] {candidate_id} "
                      f"[{split_number}/{len(split_groups)}] {split_id}: FAILED: {exc}",
                      flush=True)
            finally:
                empty_cuda_cache()

    for record in status.values():
        record["complete"] = record["completed_folds"] == record["expected_folds"]
        record["passed"] = bool(record["passed"] and record["complete"]
                                and record["valid_predictions"])
    eligible = {(record["candidate_id"], record["protocol"])
                for record in status.values() if record["passed"]}

    def _keep(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        mask = [(model_id, protocol) in eligible for model_id, protocol
                in zip(frame["model_id"], frame["protocol"])]
        return frame.loc[mask].reset_index(drop=True)

    predictions = _keep(pd.concat(prediction_rows, ignore_index=True)
                        if prediction_rows else pd.DataFrame())
    fold_metrics = _keep(pd.DataFrame(metric_rows))
    predictions.to_csv(out / "outer_fold_predictions.csv", index=False)
    if len(predictions):
        predictions.to_parquet(out / "outer_fold_predictions.parquet", index=False)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out / "candidate_manifest.csv", index=False)
    _keep(pd.DataFrame(parameter_rows)).to_csv(out / "selected_hyperparameters.csv",
                                               index=False)
    history = _keep(pd.concat(history_frames, ignore_index=True)
                    if history_frames else pd.DataFrame())
    history.to_csv(out / "search_history.csv", index=False)
    inner = _keep(pd.concat(inner_frames, ignore_index=True)
                  if inner_frames else pd.DataFrame())
    inner.to_csv(out / "inner_oof_predictions.csv", index=False)
    pd.DataFrame(explainability_rows).to_csv(out / "explainability_index.csv", index=False)
    counts = write_derived_tables(out, predictions, fold_metrics,
                                  args.bootstrap_iterations, base_seed)
    integrity = {
        "passed": not failures, "dataset_hash_matches_splits": True,
        "protocol_status": list(status.values()), "failures": failures,
        "failed_result_policy": ("Failed candidate/protocol combinations are excluded "
                                 "from every result table.")}
    (out / "integrity_checks.json").write_text(json.dumps(integrity, indent=2),
                                               encoding="utf-8")
    metadata = {
        "data_version": DATA_VERSION, "dataset_sha256": data_report["dataset_sha256"],
        "target_column": data_report["target_column"],
        "group_column": data_report["group_column"],
        "row_count": data_report["row_count"], "paper_count": data_report["paper_count"],
        "candidate_count": len(indexed),
        "model_id": args.model_id,
        "model_family_filter": (MODEL_ID_TO_FAMILY[args.model_id]
                                if args.model_id is not None else None),
        "search_iterations": args.search_iterations, "inner_folds": args.inner_folds,
        "search_jobs": args.search_jobs, "augmentations": augmentations,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_iterations": args.bootstrap_iterations,
        "protocols": sorted(assignments["protocol"].unique().tolist()),
        "tuning_policy": ("Hyper-parameters are tuned inside each outer training "
                          "partition only; extrapolation uses paper-disjoint GroupKFold "
                          "inner folds and interpolation uses shuffled KFold folds."),
        "augmentation_policy": ("Bootstrap rows are drawn with replacement from the "
                               "training partition only and inherit the paper of their "
                               "source row; test rows are never augmented."),
        "derived_table_rows": counts,
        "feature_manifest": manifest,
    }
    (out / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if failures:
        (out / "evaluation_failures.json").write_text(json.dumps(failures, indent=2),
                                                      encoding="utf-8")
    print(json.dumps({"output": str(out), "integrity_passed": not failures,
                      "eligible_candidate_protocols": len(eligible),
                      "fold_metric_rows": int(len(fold_metrics))}, indent=2))


if __name__ == "__main__":
    main()
