"""Evaluate model families across frozen outer splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fable_common import MODEL_FAMILIES, configure_torch, load_data
from fable_models import make_model
from fable_search import classification_metrics, nested_search

THRESHOLD_POLICIES = ("mcc", "f1", "balanced_accuracy", "youden_j")
SUMMARY_METRICS = (
    "roc_auc", "pr_auc", "brier", "mcc", "f1", "balanced_accuracy",
    "sensitivity", "specificity", "precision",
)


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _resolve_family(model: str | None) -> tuple[str, str] | None:
    if model is None:
        return None
    normalized = model.strip().lower()
    if normalized in MODEL_FAMILIES:
        return normalized, MODEL_FAMILIES[normalized]
    for output_key, family in MODEL_FAMILIES.items():
        if normalized == family.lower():
            return output_key, family
    raise ValueError(
        f"Unknown model family {model!r}; expected one of {sorted(MODEL_FAMILIES)} "
        f"or {sorted(MODEL_FAMILIES.values())}")


def _load_assignments(splits: Path) -> pd.DataFrame:
    parquet = splits / "split_assignments.parquet"
    csv = splits / "split_assignments.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No split assignments found in {splits}")


def _load_candidates(config_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", payload)
    if not isinstance(candidates, list):
        raise ValueError(f"Expected a candidate list in {config_path}")
    return candidates


def _subset_by_row_id(frame: pd.DataFrame, row_ids: list[str]) -> pd.DataFrame:
    subset = frame.loc[row_ids]
    if isinstance(subset, pd.Series):
        subset = subset.to_frame().T
    return subset.reset_index(drop=True)


def _bootstrap_interval(values: list[float], seed: int, iterations: int = 2000) -> tuple[float, float]:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if data.size == 0:
        return float("nan"), float("nan")
    if data.size == 1:
        point = float(data[0])
        return point, point
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.choice(data, size=data.size, replace=True)
        samples[index] = float(np.mean(sample))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _candidate_manifest(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "model_id": candidate["candidate_id"],
            "model_family": candidate["model_family"],
            "paper_weight": candidate.get("paper_weight", "none"),
            "class_weight": bool(candidate.get("class_weight", False)),
            "fixed_params": json.dumps(_clean_json(candidate.get("fixed_params", {})), sort_keys=True),
        })
    return pd.DataFrame(rows)


def _family_candidates(candidates: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if candidate["model_family"] == family]


def _summary_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame(columns=[
            "candidate_id", "model_id", "model_family", "protocol", "fold_count",
            *[f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in ("mean", "std")],
        ])

    rows = []
    for (candidate_id, protocol), frame in fold_metrics.groupby(["candidate_id", "protocol"], sort=False):
        row = {
            "candidate_id": candidate_id,
            "model_id": candidate_id,
            "model_family": frame["model_family"].iloc[0],
            "protocol": protocol,
            "fold_count": int(len(frame)),
        }
        for metric in SUMMARY_METRICS:
            series = frame[metric].astype(float)
            row[f"{metric}_mean"] = float(series.mean())
            row[f"{metric}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame(columns=["candidate_id", "model_id", "model_family", "protocol", "metric", "ci_low", "ci_high", "mean"])

    rows = []
    for (candidate_id, protocol), frame in fold_metrics.groupby(["candidate_id", "protocol"], sort=False):
        for metric in ("roc_auc", "pr_auc", "brier"):
            values = frame[metric].astype(float).tolist()
            seed = _stable_seed(candidate_id, protocol, metric)
            lower, upper = _bootstrap_interval(values, seed)
            rows.append({
                "candidate_id": candidate_id,
                "model_id": candidate_id,
                "model_family": frame["model_family"].iloc[0],
                "protocol": protocol,
                "metric": metric,
                "ci_low": lower,
                "ci_high": upper,
                "mean": float(np.mean([value for value in values if np.isfinite(value)])) if values else float("nan"),
            })
    return pd.DataFrame(rows)


def _paired_comparisons(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame(columns=["protocol", "model_a", "model_b", "family_a", "family_b", "delta_roc_auc", "delta_pr_auc", "paired_fold_count"])

    rows = []
    for protocol, frame in fold_metrics.groupby("protocol", sort=False):
        models = sorted(frame["candidate_id"].unique())
        for model_a, model_b in combinations(models, 2):
            left = frame[frame["candidate_id"] == model_a][["split_id", "roc_auc", "pr_auc"]]
            right = frame[frame["candidate_id"] == model_b][["split_id", "roc_auc", "pr_auc"]]
            joined = left.merge(right, on="split_id", suffixes=("_a", "_b"))
            if joined.empty:
                continue
            rows.append({
                "protocol": protocol,
                "model_a": model_a,
                "model_b": model_b,
                "family_a": frame[frame["candidate_id"] == model_a]["model_family"].iloc[0],
                "family_b": frame[frame["candidate_id"] == model_b]["model_family"].iloc[0],
                "delta_roc_auc": float((joined["roc_auc_a"] - joined["roc_auc_b"]).mean()),
                "delta_pr_auc": float((joined["pr_auc_a"] - joined["pr_auc_b"]).mean()),
                "paired_fold_count": int(len(joined)),
            })
    return pd.DataFrame(rows)


def _per_paper_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["candidate_id", "model_id", "model_family", "protocol", "split_id", "paper_id", *SUMMARY_METRICS])

    rows = []
    for (candidate_id, protocol, split_id, paper_id), frame in predictions.groupby(["candidate_id", "protocol", "split_id", "paper_id"], sort=False):
        metrics = classification_metrics(frame["true_label"].to_numpy(), frame["predicted_probability"].to_numpy(), 0.5)
        rows.append({
            "candidate_id": candidate_id,
            "model_id": candidate_id,
            "model_family": frame["model_family"].iloc[0],
            "protocol": protocol,
            "split_id": split_id,
            "paper_id": paper_id,
            **{metric: metrics[metric] for metric in SUMMARY_METRICS},
        })
    return pd.DataFrame(rows)


def _evaluate_candidate(candidate: dict[str, Any],df: pd.DataFrame,assignments: pd.DataFrame,
                        search_iterations: int,inner_group_folds: int,candidate_number: int,candidate_total: int,total_folds: int,) -> tuple[
                            list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
                            list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
                            dict[str, Any]]:
    indexed = df.set_index("row_id", drop=False)
    fold_metrics: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_paper_rows: list[dict[str, Any]] = []
    search_history_rows: list[dict[str, Any]] = []
    inner_oof_rows: list[dict[str, Any]] = []
    protocol_status: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "protocol": None,
        "complete": False,
        "valid_probabilities": True,
        "errors": [],
    }
    completed_folds = 0
    for protocol, protocol_frame in assignments.groupby("protocol", sort=False):
        protocol_status["protocol"] = protocol
        protocol_errors: list[str] = []
        protocol_valid = True
        protocol_fold_count = 0

        for (split_id, seed, fold), split_frame in protocol_frame.groupby(["split_id", "seed", "fold"], sort=False):
            train_ids = split_frame[split_frame["partition"] == "train"]["row_id"].tolist()
            test_ids = split_frame[split_frame["partition"] == "test"]["row_id"].tolist()
            train_df = _subset_by_row_id(indexed, train_ids)
            test_df = _subset_by_row_id(indexed, test_ids)
            try:
                search = nested_search(
                    candidate, train_df, train_df["ignition_binary"].to_numpy(),
                    train_df["paper_id"].reset_index(drop=True), protocol,
                    iterations=search_iterations, inner_folds=inner_group_folds,
                    seed=int(seed), n_jobs=-1)
                selected_params = _clean_json(search.selected_params)
                selected_rows.append({
                    "candidate_id": candidate["candidate_id"],
                    "model_id": candidate["candidate_id"],
                    "model_family": candidate["model_family"],
                    "protocol": protocol,
                    "split_id": split_id,
                    "seed": int(seed),
                    "fold": int(fold),
                    "selected_hyperparameters": json.dumps(selected_params, sort_keys=True),
                })

                final_model = make_model(candidate, search.selected_params, int(seed))
                final_model.fit(
                    train_df, train_df["ignition_binary"].to_numpy(),
                    train_df["paper_id"].reset_index(drop=True))
                probability = np.asarray(final_model.predict_proba(test_df), dtype=float)
                if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
                    protocol_valid = False
                    raise ValueError("Model produced invalid probabilities")

                metrics = classification_metrics(test_df["ignition_binary"].to_numpy(), probability, 0.5)
                fold_metrics.append({
                    "candidate_id": candidate["candidate_id"],
                    "model_id": candidate["candidate_id"],
                    "model_family": candidate["model_family"],
                    "protocol": protocol,
                    "split_id": split_id,
                    "seed": int(seed),
                    "fold": int(fold),
                    **{metric: metrics[metric] for metric in SUMMARY_METRICS},
                })

                for local_index, record in enumerate(test_df.itertuples(index=False)):
                    prediction_row = {
                        "candidate_id": candidate["candidate_id"],
                        "model_id": candidate["candidate_id"],
                        "model_family": candidate["model_family"],
                        "protocol": protocol,
                        "split_id": split_id,
                        "seed": int(seed),
                        "fold": int(fold),
                        "row_id": record.row_id,
                        "paper_id": record.paper_id,
                        "true_label": int(record.ignition_binary),
                        "predicted_probability": float(probability[local_index]),
                    }
                    for policy in THRESHOLD_POLICIES:
                        threshold = float(search.thresholds[policy])
                        prediction_row[f"threshold_{policy}"] = threshold
                        prediction_row[f"prediction_{policy}"] = int(probability[local_index] >= threshold)
                    prediction_rows.append(prediction_row)

                for paper_id, paper_frame in test_df.assign(predicted_probability=probability).groupby("paper_id", sort=False):
                    paper_metrics = classification_metrics(
                        paper_frame["ignition_binary"].to_numpy(),
                        paper_frame["predicted_probability"].to_numpy(), 0.5)
                    per_paper_rows.append({
                        "candidate_id": candidate["candidate_id"],
                        "model_id": candidate["candidate_id"],
                        "model_family": candidate["model_family"],
                        "protocol": protocol,
                        "split_id": split_id,
                        "seed": int(seed),
                        "fold": int(fold),
                        "paper_id": paper_id,
                        **{metric: paper_metrics[metric] for metric in SUMMARY_METRICS},
                    })

                search_history = search.history.copy()
                search_history["candidate_id"] = candidate["candidate_id"]
                search_history["model_id"] = candidate["candidate_id"]
                search_history["model_family"] = candidate["model_family"]
                search_history["protocol"] = protocol
                search_history["split_id"] = split_id
                search_history_rows.extend(search_history.to_dict(orient="records"))

                inner_oof = search.inner_predictions.copy()
                inner_oof["candidate_id"] = candidate["candidate_id"]
                inner_oof["model_id"] = candidate["candidate_id"]
                inner_oof["model_family"] = candidate["model_family"]
                inner_oof["protocol"] = protocol
                inner_oof["split_id"] = split_id
                inner_oof_rows.extend(inner_oof.to_dict(orient="records"))

                protocol_fold_count += 1
                completed_folds += 1
                print(
                    f"[{candidate_number}/{candidate_total}] "
                    f"{candidate['candidate_id']} "
                    f"[{completed_folds}/{total_folds}] "
                    f"{protocol}__seed-{seed}__fold-{fold}: complete",
                    flush=True,
                )
            except Exception as exc:
                protocol_errors.append(f"{split_id}: {exc}")

        protocol_status.update({
            "complete": bool(protocol_fold_count > 0 and not protocol_errors),
            "valid_probabilities": bool(protocol_valid),
            "errors": protocol_errors,
            "fold_count": protocol_fold_count,
        })

    return fold_metrics, selected_rows, prediction_rows, per_paper_rows, search_history_rows, inner_oof_rows, protocol_status


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_family_outputs(out_dir: Path, candidates: list[dict[str, Any]], fold_rows: list[dict[str, Any]],
                          selected_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]],
                          per_paper_rows: list[dict[str, Any]], search_history_rows: list[dict[str, Any]],
                          inner_oof_rows: list[dict[str, Any]], protocol_status_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_frame = _candidate_manifest(candidates)
    fold_frame = pd.DataFrame(fold_rows)
    selected_frame = pd.DataFrame(selected_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    per_paper_frame = pd.DataFrame(per_paper_rows)
    search_history_frame = pd.DataFrame(search_history_rows)
    inner_oof_frame = pd.DataFrame(inner_oof_rows)

    summary = _summary_table(fold_frame)
    intervals = _bootstrap_table(fold_frame)
    paired = _paired_comparisons(fold_frame)
    integrity = {
        "passed": bool(all(status.get("complete", False) and status.get("valid_probabilities", False)
                           and not status.get("errors") for status in protocol_status_rows)),
        "protocol_status": protocol_status_rows,
    }

    _write_csv(candidate_frame, out_dir / "candidate_manifest.csv")
    _write_csv(summary, out_dir / "summary_metrics.csv")
    _write_csv(intervals, out_dir / "bootstrap_intervals.csv")
    _write_csv(paired, out_dir / "paired_model_comparisons.csv")
    _write_csv(selected_frame, out_dir / "selected_hyperparameters.csv")
    _write_csv(per_paper_frame, out_dir / "per_paper_metrics.csv")
    _write_csv(search_history_frame, out_dir / "search_history.csv")
    _write_csv(inner_oof_frame, out_dir / "inner_oof_predictions.csv")
    prediction_frame.to_parquet(out_dir / "outer_fold_predictions.parquet", index=False)
    (out_dir / "integrity_checks.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-iterations", type=int, default=40)
    parser.add_argument("--inner-group-folds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--model", help="Optional family key or family name to evaluate")
    parser.add_argument(
    "--candidate-id",
    help="Optional exact candidate_id to evaluate; must match one YAML candidate.",
)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    configure_torch()
    data, _ = load_data(args.data, require_target=True)
    assignments = _load_assignments(Path(args.splits))
    candidates = _load_candidates(Path(args.config))
    family_selection = _resolve_family(args.model)

    if family_selection is None:
        families = [(output_key, family) for output_key, family in MODEL_FAMILIES.items()]
        direct_output = False
    else:
        families = [family_selection]
        direct_output = True

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for output_key, family in families:
        family_candidates = _family_candidates(candidates, family)
        if args.candidate_id is not None:
            family_candidates = [
                candidate for candidate in family_candidates
                if candidate["candidate_id"] == args.candidate_id
            ]
            if len(family_candidates) != 1:
                raise ValueError(
                    f"Expected exactly one {family!r} candidate with "
                    f"candidate_id={args.candidate_id!r}; found {len(family_candidates)}."
                )        
        if not family_candidates:
            if args.candidate_id is not None:
                raise ValueError(
                    f"No candidate matched model={args.model!r}, "
                    f"candidate_id={args.candidate_id!r}."
                )
            continue

        fold_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        per_paper_rows: list[dict[str, Any]] = []
        search_history_rows: list[dict[str, Any]] = []
        inner_oof_rows: list[dict[str, Any]] = []
        protocol_status_rows: list[dict[str, Any]] = []
        total_folds = int(assignments[["split_id", "seed", "fold"]].drop_duplicates().shape[0])
        
        
        for candidate_number, candidate in enumerate(family_candidates, start=1):
            candidate_fold_rows, candidate_selected_rows, candidate_prediction_rows, candidate_per_paper_rows, candidate_search_history_rows, candidate_inner_oof_rows, protocol_status = _evaluate_candidate(
                candidate, data, assignments, args.search_iterations, args.inner_group_folds,candidate_number,len(family_candidates),total_folds,)
            fold_rows.extend(candidate_fold_rows)
            selected_rows.extend(candidate_selected_rows)
            prediction_rows.extend(candidate_prediction_rows)
            per_paper_rows.extend(candidate_per_paper_rows)
            search_history_rows.extend(candidate_search_history_rows)
            inner_oof_rows.extend(candidate_inner_oof_rows)
            protocol_status_rows.append(protocol_status)

        target_out = out_root if direct_output else out_root / output_key
        _write_family_outputs(
            target_out, family_candidates, fold_rows, selected_rows,
            prediction_rows, per_paper_rows, search_history_rows,
            inner_oof_rows, protocol_status_rows)

        print(json.dumps({
            "family": family,
            "output": str(target_out),
            "candidates": len(family_candidates),
            "fold_rows": len(fold_rows),
            "passed": bool(all(status.get("complete", False) and status.get("valid_probabilities", False)
                               and not status.get("errors") for status in protocol_status_rows)),
        }, indent=2))


if __name__ == "__main__":
    main()