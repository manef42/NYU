"""Derived result tables and the Slurm-array shard merge step.

``fsr_evaluate.py`` writes one shard per model family (one Slurm array task).
This module owns every table that is computed *from* those shards - fold
summaries, per-paper breakdowns, paired tests, bootstrap intervals, the
interpolation/extrapolation generalization gap and the augmentation effect - so
that a single shard and the merged whole are summarized by exactly the same code.

Run as a script it merges ``<evaluation>/model_*`` shards into one evaluation
directory and recomputes all derived tables over the union.
"""
from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from fsr_common import METRIC_KEYS, regression_metrics

KEYS = ("model_id", "model_family", "feature_set", "protocol", "augmentation")
PAIRED_METRICS = ("RMSE", "MAE", "R2")
SHARD_FRAMES = {
    "outer_fold_predictions.csv": ["model_id", "protocol", "augmentation", "split_id",
                                   "row_id"],
    "fold_metrics.csv": ["model_id", "protocol", "augmentation", "split_id"],
    "candidate_manifest.csv": ["candidate_id"],
    "selected_hyperparameters.csv": ["model_id", "protocol", "split_id"],
    "search_history.csv": ["model_id", "protocol", "split_id", "configuration_id",
                           "inner_fold"],
    "inner_oof_predictions.csv": ["model_id", "protocol", "split_id", "row_id"],
}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a persisted table, returning an empty frame for missing or empty files."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every fold metric per candidate condition."""
    if fold_metrics.empty:
        return pd.DataFrame()
    numeric = [column for column in fold_metrics.select_dtypes(include="number").columns
               if column not in {"seed", "repeat", "n_test_rows", "n_test_papers",
                                 "n_train_rows", "n_train_papers"}]
    rows = []
    for values, group in fold_metrics.groupby(list(KEYS), dropna=False):
        record = dict(zip(KEYS, values))
        record.update({
            "fold_count": int(len(group)),
            "test_row_total": int(group["n_test_rows"].sum()),
            "test_paper_total": int(group["n_test_papers"].sum()),
        })
        for metric in numeric:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
            record[f"{metric}_median"] = float(group[metric].median())
            record[f"{metric}_min"] = float(group[metric].min())
            record[f"{metric}_max"] = float(group[metric].max())
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["protocol", "augmentation", "RMSE_mean"])


def per_paper(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-paper error distribution: average metrics hide catastrophic papers."""
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    keys = list(KEYS) + ["paper_id", "paper_label"]
    for values, group in predictions.groupby(keys, dropna=False):
        metrics = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        rows.append({**dict(zip(keys, values)), "n_predictions": int(len(group)),
                     "n_unique_rows": int(group["row_id"].nunique()), **metrics})
    return pd.DataFrame(rows).sort_values(["protocol", "augmentation", "RMSE"],
                                          ascending=[True, True, False])


def paired(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Exploratory paired comparison of two candidates over identical outer folds."""
    if fold_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (protocol, augmentation), frame in fold_metrics.groupby(
            ["protocol", "augmentation"]):
        models = sorted(frame["model_id"].unique())
        indexed = {model: frame[frame["model_id"] == model].set_index("split_id")
                   for model in models}
        for left, right in combinations(models, 2):
            common = indexed[left].index.intersection(indexed[right].index)
            record = {"protocol": protocol, "augmentation": augmentation,
                      "model_a": left, "model_b": right, "paired_fold_count": len(common)}
            for metric in PAIRED_METRICS:
                differences = (indexed[left].loc[common, metric] -
                               indexed[right].loc[common, metric]).to_numpy(dtype=float)
                differences = differences[np.isfinite(differences)]
                record[f"delta_{metric}"] = (float(np.mean(differences))
                                             if len(differences) else np.nan)
                try:
                    record[f"wilcoxon_p_{metric}"] = (
                        float(wilcoxon(differences).pvalue)
                        if len(differences) and np.any(differences != 0) else 1.)
                except ValueError:
                    record[f"wilcoxon_p_{metric}"] = np.nan
            record["caveat"] = ("Exploratory paired test over identical outer folds; "
                                "multiplicity is not adjusted.")
            rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_intervals(predictions: pd.DataFrame, iterations: int,
                        seed: int) -> pd.DataFrame:
    """Resample rows (interpolation) or whole paper clusters (extrapolation)."""
    if predictions.empty or iterations <= 0:
        return pd.DataFrame()
    rows = []
    for values, group in predictions.groupby(list(KEYS), dropna=False):
        record_keys = dict(zip(KEYS, values))
        grouped = str(record_keys["protocol"]).startswith("extrapolation")
        collapsed = group.groupby(["row_id", "paper_id"], as_index=False).agg(
            y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
        estimate = regression_metrics(collapsed["y_true"].to_numpy(),
                                      collapsed["y_pred"].to_numpy())
        rng = np.random.default_rng(seed + abs(hash(str(values))) % 100_000)
        units = (collapsed["paper_id"].unique() if grouped
                 else np.arange(len(collapsed)))
        samples: dict[str, list[float]] = {metric: [] for metric in PAIRED_METRICS}
        for _ in range(int(iterations)):
            drawn = rng.choice(units, len(units), replace=True)
            if grouped:
                sample = collapsed[collapsed["paper_id"].isin(set(drawn))]
                repeats = pd.Series(drawn).value_counts()
                sample = sample.loc[sample.index.repeat(
                    sample["paper_id"].map(repeats).to_numpy())]
            else:
                sample = collapsed.iloc[drawn]
            metrics = regression_metrics(sample["y_true"].to_numpy(),
                                         sample["y_pred"].to_numpy())
            for metric in PAIRED_METRICS:
                if np.isfinite(metrics[metric]):
                    samples[metric].append(metrics[metric])
        for metric in PAIRED_METRICS:
            values_list = samples[metric]
            low, high = (np.quantile(values_list, [.025, .975])
                         if values_list else (np.nan, np.nan))
            rows.append({**record_keys, "metric": metric, "estimate": estimate[metric],
                         "ci_low": float(low), "ci_high": float(high),
                         "confidence_level": .95, "iterations": int(iterations),
                         "resampling_unit": ("canonical_paper" if grouped
                                             else "unique_row_id"),
                         "repeat_aggregation": "mean prediction per row before resampling"})
    return pd.DataFrame(rows)


def generalization_gap(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Extrapolation RMSE minus interpolation RMSE, repeat by repeat."""
    if fold_metrics.empty:
        return pd.DataFrame()
    columns = ["model_id", "model_family", "feature_set", "augmentation", "repeat"]
    pivot = fold_metrics.pivot_table(index=columns, columns="protocol", values="RMSE")
    if not {"interpolation_random", "extrapolation_grouped"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["interpolation_random", "extrapolation_grouped"])
    pivot = pivot.rename(columns={"interpolation_random": "random_RMSE",
                                  "extrapolation_grouped": "grouped_RMSE"}).reset_index()
    pivot["generalization_gap"] = pivot["grouped_RMSE"] - pivot["random_RMSE"]
    summary = pivot.groupby(["model_id", "model_family", "feature_set", "augmentation"])[
        ["random_RMSE", "grouped_RMSE", "generalization_gap"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index().sort_values("grouped_RMSE_mean")


def augmentation_effect(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap-augmentation effect on RMSE, holding the outer fold fixed."""
    if fold_metrics.empty or fold_metrics["augmentation"].nunique() < 2:
        return pd.DataFrame()
    columns = ["model_id", "model_family", "feature_set", "protocol", "repeat"]
    pivot = fold_metrics.pivot_table(index=columns, columns="augmentation", values="RMSE")
    if not {"rd", "rd_bt"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["rd", "rd_bt"]).reset_index()
    pivot["delta_RMSE_bt_minus_rd"] = pivot["rd_bt"] - pivot["rd"]
    summary = pivot.groupby(["model_id", "model_family", "feature_set", "protocol"])[
        ["rd", "rd_bt", "delta_RMSE_bt_minus_rd"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index().sort_values(["protocol", "rd_mean"])


def stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Repeat-to-repeat variability of every headline metric (diagnostic)."""
    if fold_metrics.empty:
        return pd.DataFrame()
    rows = []
    for values, group in fold_metrics.groupby(list(KEYS), dropna=False):
        record = dict(zip(KEYS, values))
        for metric in METRIC_KEYS:
            if metric not in group:
                continue
            series = group[metric].astype(float)
            record.update({
                f"{metric}_mean": float(series.mean()),
                f"{metric}_std": float(series.std(ddof=1)),
                f"{metric}_median": float(series.median()),
                f"{metric}_min": float(series.min()),
                f"{metric}_max": float(series.max()),
            })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["protocol", "augmentation", "RMSE_mean"])


def write_derived_tables(out: Path, predictions: pd.DataFrame,
                         fold_metrics: pd.DataFrame, bootstrap_iterations: int,
                         seed: int) -> dict[str, int]:
    """Persist every derived table for one shard or for the merged evaluation."""
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary_metrics": summarize(fold_metrics),
        "per_paper_metrics": per_paper(predictions),
        "paired_model_comparisons": paired(fold_metrics),
        "bootstrap_intervals": bootstrap_intervals(predictions, bootstrap_iterations, seed),
        "generalization_gap": generalization_gap(fold_metrics),
        "augmentation_comparison": augmentation_effect(fold_metrics),
        "metric_stability": stability(fold_metrics),
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return {name: int(len(frame)) for name, frame in tables.items()}


def _merge_integrity(shards: list[Path]) -> dict[str, Any]:
    status, failures = [], []
    for shard in shards:
        path = shard / "integrity_checks.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        status.extend(payload.get("protocol_status", []))
        failures.extend(payload.get("failures", []))
    return {"passed": not failures, "protocol_status": status, "failures": failures,
            "failed_result_policy": ("Failed candidate/protocol combinations are excluded "
                                     "from every result table.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True,
                        help="Directory that contains the per-family model_* shards.")
    parser.add_argument("--out", required=True, help="Merged evaluation directory.")
    parser.add_argument("--shard-glob", default="model_*")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluation, out = Path(args.evaluation), Path(args.out)
    shards = sorted(path for path in evaluation.glob(args.shard_glob) if path.is_dir())
    if not shards:
        raise FileNotFoundError(
            f"No evaluation shards matching {args.shard_glob!r} under {evaluation}")
    out.mkdir(parents=True, exist_ok=True)
    merged: dict[str, pd.DataFrame] = {}
    for name, subset in SHARD_FRAMES.items():
        frames = [read_table(shard / name) for shard in shards]
        frames = [frame for frame in frames if len(frame)]
        if not frames:
            merged[name] = pd.DataFrame()
            continue
        frame = pd.concat(frames, ignore_index=True)
        columns = [column for column in subset if column in frame.columns]
        if columns:
            frame = frame.drop_duplicates(subset=columns, keep="first")
        merged[name] = frame.reset_index(drop=True)
        frame.to_csv(out / name, index=False)
    predictions = merged["outer_fold_predictions.csv"]
    if len(predictions):
        predictions.to_parquet(out / "outer_fold_predictions.parquet", index=False)
    if len(merged["inner_oof_predictions.csv"]):
        merged["inner_oof_predictions.csv"].to_parquet(
            out / "inner_oof_predictions.parquet", index=False)
    fold_metrics = merged["fold_metrics.csv"]
    counts = write_derived_tables(out, predictions, fold_metrics,
                                  args.bootstrap_iterations, args.seed)
    explainability = []
    for shard in shards:
        frame = read_table(shard / "explainability_index.csv")
        for _, row in frame.iterrows():
            source = shard / str(row["model_path"])
            if not source.is_file():
                continue
            destination = out / "explainability_models" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            explainability.append({**row.to_dict(),
                                   "model_path": str(destination.relative_to(out))})
    pd.DataFrame(explainability).to_csv(out / "explainability_index.csv", index=False)
    integrity = _merge_integrity(shards)
    (out / "integrity_checks.json").write_text(json.dumps(integrity, indent=2),
                                               encoding="utf-8")
    metadata_shards = [json.loads((shard / "evaluation_metadata.json").read_text("utf-8"))
                       for shard in shards if (shard / "evaluation_metadata.json").is_file()]
    (out / "evaluation_metadata.json").write_text(json.dumps({
        "merged_shards": [shard.name for shard in shards],
        "shard_metadata": metadata_shards,
        "bootstrap_iterations": args.bootstrap_iterations,
        "derived_table_rows": counts,
        "integrity_passed": integrity["passed"],
    }, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"merged": str(out), "shards": len(shards),
                      "fold_metric_rows": int(len(fold_metrics)),
                      "prediction_rows": int(len(predictions)),
                      "integrity_passed": integrity["passed"],
                      "derived_table_rows": counts}, indent=2))


if __name__ == "__main__":
    main()
