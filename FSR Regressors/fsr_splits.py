"""Generate immutable interpolation and paper-grouped extrapolation splits.

Both evaluation strategies of the flame-spread-rate study are materialized once,
validated for row coverage and paper leakage, and written to disk so that every
downstream model family in the Slurm array benchmarks exactly the same
partitions:

* ``interpolation_random``   -- repeated target-stratified 80/20 row holdouts
  (the optimistic baseline: rows of a paper may appear on both sides).
* ``extrapolation_grouped``  -- repeated balance-aware ``GroupShuffleSplit``
  holdouts where every paper stays entirely in train or entirely in test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from fsr_common import (DATA_VERSION, RANDOM_STATE, TARGET, distribution_distance,
                        load_data, quantile_bins, write_validation_report)

PROTOCOLS = ("interpolation_random", "extrapolation_grouped")


def _balance_score(y_train: np.ndarray, y_test: np.ndarray, test_size: float,
                   full_std: float, train_histogram: np.ndarray | None = None,
                   test_histogram: np.ndarray | None = None) -> float:
    """Lower is better: similar target moments and a test fraction close to target."""
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    scale = full_std if np.isfinite(full_std) and full_std > 1e-12 else 1.
    score = abs(np.nanmean(y_train) - np.nanmean(y_test)) / scale
    score += .5 * abs(np.nanmedian(y_train) - np.nanmedian(y_test)) / scale
    score += .25 * abs(np.nanstd(y_train) - np.nanstd(y_test)) / scale
    score += 2. * abs(len(y_test) / max(len(y_train) + len(y_test), 1) - test_size)
    if train_histogram is not None and test_histogram is not None:
        score += 1.5 * distribution_distance(train_histogram, test_histogram)
    return float(score)


def _random_holdout(y: np.ndarray, test_size: float, seed: int, target_bins: int
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Row-level holdout, stratified by target quantile bins when feasible."""
    indices = np.arange(len(y))
    bins = quantile_bins(y, max_bins=target_bins)
    stratify = None
    if bins is not None:
        labels = pd.Series(bins).dropna().astype(int)
        counts = labels.value_counts()
        if len(labels) == len(y) and len(counts) >= 2 and counts.min() >= 2:
            stratify = labels.to_numpy()
    train, test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify)
    return train, test, {"stratified": stratify is not None}


def _grouped_holdout(groups: np.ndarray, y: np.ndarray, test_size: float, seed: int,
                     candidates: int, target_bins: int
                     ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pick the most target-balanced paper-disjoint holdout among many seeds.

    A plain single-shot ``GroupShuffleSplit`` can land a handful of extreme papers
    entirely in the test partition, which makes R2 swing wildly between repeats
    for reasons that have nothing to do with model quality. Screening many
    candidate paper partitions and keeping the one whose train and test target
    distributions agree best removes that nuisance variance while preserving the
    hard constraint that no paper is ever split.
    """
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    full_std = float(np.nanstd(y))
    per_paper = pd.DataFrame({"paper": groups, "y": y}).groupby(
        "paper", as_index=False).agg(y_mean=("y", "mean"))
    bins = quantile_bins(per_paper["y_mean"].to_numpy(), max_bins=target_bins)
    if bins is not None and np.isfinite(bins).any():
        per_paper["bin"] = pd.Series(bins).round().astype("Int64")
        max_bin = int(per_paper["bin"].dropna().max())
    else:
        per_paper["bin"] = pd.Series([pd.NA] * len(per_paper), dtype="Int64")
        max_bin = -1
    best: tuple[np.ndarray, np.ndarray, dict] | None = None
    best_score = np.inf
    for attempt in range(max(1, int(candidates))):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed + 1000 * attempt)
        train, test = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        if max_bin >= 0:
            train_bins = per_paper.loc[per_paper["paper"].isin(np.unique(groups[train])),
                                       "bin"].dropna().astype(int).to_numpy()
            test_bins = per_paper.loc[per_paper["paper"].isin(np.unique(groups[test])),
                                      "bin"].dropna().astype(int).to_numpy()
            train_histogram = np.bincount(train_bins, minlength=max_bin + 1)
            test_histogram = np.bincount(test_bins, minlength=max_bin + 1)
            bin_distance = distribution_distance(train_histogram, test_histogram)
        else:
            train_histogram = test_histogram = None
            bin_distance = 0.
        score = _balance_score(y[train], y[test], test_size, full_std,
                               train_histogram, test_histogram) + bin_distance
        if score < best_score:
            best_score = score
            best = (train, test, {"balance_score": float(score),
                                  "target_bin_distance": float(bin_distance)})
    assert best is not None
    return best


def _record(df: pd.DataFrame, protocol: str, seed: int, repeat: int,
            train: np.ndarray, test: np.ndarray) -> list[dict]:
    split_id = f"{protocol}__seed-{seed}__repeat-{repeat}"
    row_ids = df["row_id"].to_numpy()
    paper_ids = df["paper_id"].to_numpy()
    return [{"split_id": split_id, "protocol": protocol, "seed": seed, "repeat": repeat,
             "partition": partition, "row_id": row_ids[index], "paper_id": paper_ids[index]}
            for partition, indices in (("train", train), ("test", test))
            for index in indices]


def validate(assignments: pd.DataFrame, df: pd.DataFrame) -> dict:
    expected = set(df["row_id"])
    failures, checked = [], 0
    for split_id, split in assignments.groupby("split_id", sort=False):
        train = split[split["partition"] == "train"]
        test = split[split["partition"] == "test"]
        if set(train["row_id"]) & set(test["row_id"]):
            failures.append(f"{split_id}: row overlap between partitions")
        if set(train["row_id"]) | set(test["row_id"]) != expected:
            failures.append(f"{split_id}: incomplete row coverage")
        if len(test) == 0 or len(train) == 0:
            failures.append(f"{split_id}: empty partition")
        if split["protocol"].iloc[0] == "extrapolation_grouped":
            leaked = set(train["paper_id"]) & set(test["paper_id"])
            if leaked:
                failures.append(f"{split_id}: paper leakage ({len(leaked)} papers)")
        checked += 1
    if failures:
        raise ValueError("Split integrity validation failed:\n" + "\n".join(failures))
    return {"passed": True, "splits_checked": checked, "failures": []}


def generate(df: pd.DataFrame, n_repeats: int, test_size: float, base_seed: int,
             split_candidates: int, target_bins: int
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].to_numpy(dtype=float)
    groups = df["paper_id"].to_numpy()
    rows: list[dict] = []
    diagnostics: list[dict] = []
    for repeat in range(1, n_repeats + 1):
        seed = base_seed + repeat - 1
        random_train, random_test, random_meta = _random_holdout(
            y, test_size, seed, target_bins)
        rows.extend(_record(df, "interpolation_random", seed, repeat,
                            random_train, random_test))
        grouped_train, grouped_test, grouped_meta = _grouped_holdout(
            groups, y, test_size, seed, split_candidates, target_bins)
        rows.extend(_record(df, "extrapolation_grouped", seed, repeat,
                            grouped_train, grouped_test))
        diagnostics.append({
            "repeat": repeat, "seed": seed,
            "random_stratified": bool(random_meta["stratified"]),
            "random_train_rows": len(random_train), "random_test_rows": len(random_test),
            "grouped_train_rows": len(grouped_train), "grouped_test_rows": len(grouped_test),
            "grouped_train_papers": int(pd.Series(groups[grouped_train]).nunique()),
            "grouped_test_papers": int(pd.Series(groups[grouped_test]).nunique()),
            "grouped_balance_score": grouped_meta["balance_score"],
            "grouped_target_bin_distance": grouped_meta["target_bin_distance"],
            "grouped_train_y_mean": float(np.mean(y[grouped_train])),
            "grouped_test_y_mean": float(np.mean(y[grouped_test])),
            "grouped_train_y_std": float(np.std(y[grouped_train])),
            "grouped_test_y_std": float(np.std(y[grouped_test])),
        })
    assignments = pd.DataFrame(rows)
    validate(assignments, df)
    return assignments, pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--base-seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--split-candidates", type=int, default=50,
                        help="Candidate grouped partitions screened per repeat.")
    parser.add_argument("--target-bins", type=int, default=6)
    args = parser.parse_args()
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be positive")
    if not 0 < args.test_size < .9:
        raise ValueError("--test-size must be between 0 and 0.9")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, report = load_data(args.data, require_target=True)
    if df["paper_id"].nunique() < 3:
        raise ValueError("Paper-grouped evaluation needs at least three papers")
    assignments, diagnostics = generate(
        df, args.n_repeats, args.test_size, args.base_seed,
        args.split_candidates, args.target_bins)
    assignments.to_csv(out / "split_assignments.csv", index=False)
    assignments.to_parquet(out / "split_assignments.parquet", index=False)
    diagnostics.to_csv(out / "split_diagnostics.csv", index=False)
    metadata = {
        "data_version": DATA_VERSION,
        "dataset_sha256": report["dataset_sha256"],
        "target_column": report["target_column"],
        "group_column": report["group_column"],
        "row_count": int(len(df)),
        "paper_count": int(df["paper_id"].nunique()),
        "n_repeats": args.n_repeats,
        "test_size": args.test_size,
        "base_seed": args.base_seed,
        "split_candidates": args.split_candidates,
        "target_bins": args.target_bins,
        "protocols": list(PROTOCOLS),
        "protocol_split_counts": assignments.groupby("protocol")["split_id"].nunique().to_dict(),
        "feature_manifest": report["feature_manifest"],
        "integrity": validate(assignments, df),
    }
    (out / "splits_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_validation_report(report, out / "data_validation_report.json")
    print(json.dumps({key: value for key, value in metadata.items()
                      if key != "feature_manifest"}, indent=2, default=str))


if __name__ == "__main__":
    main()
