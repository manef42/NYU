"""Select the interpolation and extrapolation champions from evaluation artifacts.

This script trains nothing. It reads the merged evaluation tables, rejects
candidates that are incomplete, invalid or unstable, and then applies the
declared policy to pick one champion per scientific question. A candidate here is
a (configuration, data-condition) pair, so the bootstrap-augmentation decision is
made by the same auditable rule as the model choice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

TIE_COLUMNS = {
    "higher_r2": ("R2_mean", False),
    "lower_rmse": ("RMSE_mean", True),
    "lower_uncertainty": ("uncertainty", True),
    "lower_mae": ("MAE_mean", True),
    "simpler_model": ("simplicity_rank", True),
    "fewer_features": ("feature_rank", True),
    "physics_only": ("feature_rank", True),
    "no_augmentation": ("augmentation_rank", True),
    "smaller_generalization_gap": ("generalization_gap_mean", True),
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _rank(summary: pd.DataFrame, intervals: pd.DataFrame, gaps: pd.DataFrame,
          manifest: pd.DataFrame, status: dict[tuple[str, str], dict],
          policy: dict[str, Any], rule: dict[str, Any]
          ) -> tuple[list[dict[str, Any]], list[str]]:
    protocol = rule["primary_protocol"]
    conditions = policy.get("augmentation_policy", {}).get("compare_conditions", ["rd"])
    table = summary[(summary["protocol"] == protocol) &
                    (summary["augmentation"].isin(conditions))].copy()
    if table.empty:
        raise RuntimeError(f"No evaluated candidates for protocol {protocol!r}")
    instability = policy["instability"]
    rejected, keep = [], []
    for _, row in table.iterrows():
        reasons = []
        record = status.get((row["model_id"], protocol), {})
        if policy.get("required_integrity_pass", True) and record.get("errors"):
            reasons.append("failed protocol integrity checks")
        if policy.get("required_complete_folds", True) and not record.get("complete", False):
            reasons.append("could not be evaluated on every required outer fold")
        if policy.get("required_valid_predictions", True) and not record.get(
                "valid_predictions", False):
            reasons.append("produced non-finite predictions")
        mean, deviation = row.get("RMSE_mean"), row.get("RMSE_std")
        if pd.isna(mean) or pd.isna(deviation):
            reasons.append("has an undefined RMSE mean or fold standard deviation")
        elif mean > 0 and deviation / mean > instability["maximum_rmse_coefficient_of_variation"]:
            reasons.append(
                f"RMSE fold coefficient of variation {deviation / mean:.3f} exceeds "
                f"{instability['maximum_rmse_coefficient_of_variation']}")
        minimum_r2 = policy.get("minimum_r2")
        if minimum_r2 is not None and (pd.isna(row.get("R2_mean")) or
                                       row["R2_mean"] < minimum_r2):
            reasons.append(f"mean R2 {row.get('R2_mean')!r} is below {minimum_r2}")
        label = f"{row['model_id']} [{row['augmentation']}]"
        if reasons:
            rejected.append(f"{label}: " + "; ".join(reasons))
        else:
            keep.append(label)
    table["label"] = table["model_id"] + " [" + table["augmentation"] + "]"
    table = table[table["label"].isin(keep)].copy()
    if table.empty:
        raise RuntimeError(f"No eligible candidates for {protocol}. Rejections: {rejected}")

    widths = intervals[(intervals["protocol"] == protocol) &
                       (intervals["metric"] == "RMSE")].copy()
    if len(widths):
        widths["uncertainty"] = widths["ci_high"] - widths["ci_low"]
        table = table.merge(widths[["model_id", "augmentation", "uncertainty"]],
                            on=["model_id", "augmentation"], how="left")
    else:
        table["uncertainty"] = np.nan
    if len(gaps):
        table = table.merge(
            gaps[["model_id", "augmentation", "generalization_gap_mean"]],
            on=["model_id", "augmentation"], how="left")
    else:
        table["generalization_gap_mean"] = np.nan
    table = table.merge(
        manifest[["candidate_id", "model_family", "fixed_params", "paper_weight"]]
        .rename(columns={"candidate_id": "model_id"}),
        on=["model_id", "model_family"], how="left")
    simplicity = {name: index for index, name in enumerate(policy["simplicity_order"])}
    table["simplicity_rank"] = table["model_family"].map(simplicity).fillna(999)
    table["feature_rank"] = table["feature_set"].map({"physics": 0, "all": 1}).fillna(9)
    table["augmentation_rank"] = table["augmentation"].map({"rd": 0, "rd_bt": 1}).fillna(9)

    best = float(table["RMSE_mean"].min())
    tolerance = best * (1 + float(policy["tie_tolerance_fraction"]))
    table["within_primary_tie"] = table["RMSE_mean"] <= tolerance
    unknown = set(rule["tie_breakers"]) - set(TIE_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown selection tie-breakers: {sorted(unknown)}")
    columns = [TIE_COLUMNS[name][0] for name in rule["tie_breakers"]] + ["label"]
    ascending = [TIE_COLUMNS[name][1] for name in rule["tie_breakers"]] + [True]
    tied = table[table["within_primary_tie"]].sort_values(
        columns, ascending=ascending, kind="stable")
    champion = tied.iloc[0]["label"]
    table["selected"] = table["label"] == champion
    table = table.sort_values(["selected", "RMSE_mean"], ascending=[False, True])
    return table.where(pd.notna(table), None).to_dict(orient="records"), rejected


def _modal_parameters(parameters: pd.DataFrame, model_id: str, protocol: str) -> dict:
    values = parameters[(parameters["model_id"] == model_id) &
                        (parameters["protocol"] == protocol)]["selected_hyperparameters"]
    if values.empty:
        raise RuntimeError(f"No selected hyper-parameters recorded for {model_id}/{protocol}")
    counts = values.value_counts()
    modal = sorted(counts[counts == counts.max()].index)[0]
    return json.loads(modal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluation, out = Path(args.evaluation), Path(args.out)
    policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    summary = pd.read_csv(evaluation / "summary_metrics.csv")
    intervals = pd.read_csv(evaluation / "bootstrap_intervals.csv")
    gaps = pd.read_csv(evaluation / "generalization_gap.csv")
    augmentation = pd.read_csv(evaluation / "augmentation_comparison.csv")
    paired = pd.read_csv(evaluation / "paired_model_comparisons.csv")
    manifest = pd.read_csv(evaluation / "candidate_manifest.csv")
    parameters = pd.read_csv(evaluation / "selected_hyperparameters.csv")
    integrity = json.loads((evaluation / "integrity_checks.json").read_text("utf-8"))
    status = {(item["candidate_id"], item["protocol"]): item
              for item in integrity["protocol_status"]}

    selections: dict[str, Any] = {}
    markdown = [
        "# Reproducible FSR champion selection", "",
        "> The interpolation champion answers \"how well can we predict a new row of a "
        "paper we already know?\" and the extrapolation champion answers \"how well can we "
        "predict a brand-new paper?\". Neither number substitutes for the other.", "",
    ]
    for name in ("interpolation_champion", "extrapolation_champion"):
        rule = policy["selections"][name]
        ranking, rejected = _rank(summary, intervals, gaps, manifest, status, policy, rule)
        winner = next(row for row in ranking if row["selected"])
        manifest_row = manifest[manifest["candidate_id"] == winner["model_id"]].iloc[0].to_dict()
        selected_params = _modal_parameters(parameters, winner["model_id"],
                                            rule["primary_protocol"])
        fixed = json.loads(manifest_row.get("fixed_params") or "{}")
        reason = (
            f"Lowest eligible {rule['primary_metric']} within "
            f"{policy['tie_tolerance_fraction']:.0%} of the best candidate; tie-breakers "
            f"applied in declared order: {', '.join(rule['tie_breakers'])}. Refit "
            f"hyper-parameters use: {policy['refit_hyperparameter_rule']}.")
        selections[name] = {
            "candidate_id": winner["model_id"], "model_family": winner["model_family"],
            "feature_set": winner["feature_set"], "augmentation": winner["augmentation"],
            "primary_protocol": rule["primary_protocol"], "metrics": winner,
            "exact_hyperparameters": {**fixed, **selected_params},
            "candidate_configuration": manifest_row, "reason": reason,
            "ranking": ranking, "rejected_candidates": rejected,
            "paired_comparisons": paired[
                (paired["protocol"] == rule["primary_protocol"]) &
                ((paired["model_a"] == winner["model_id"]) |
                 (paired["model_b"] == winner["model_id"]))
            ].where(lambda frame: pd.notna(frame), None).to_dict("records"),
        }
        markdown += [
            f"## {name.replace('_', ' ').title()}", "",
            f"Selected `{winner['model_id']}` under data condition "
            f"`{winner['augmentation']}`. {reason}", "",
            f"- RMSE: {winner['RMSE_mean']:.4f} ± {winner['RMSE_std']:.4f}",
            f"- MAE: {winner['MAE_mean']:.4f} ± {winner['MAE_std']:.4f}",
            f"- R2: {winner['R2_mean']:.4f} ± {winner['R2_std']:.4f}",
            f"- Protocol: `{rule['primary_protocol']}`",
            f"- Exact hyper-parameters: "
            f"`{json.dumps(selections[name]['exact_hyperparameters'], sort_keys=True)}`", "",
            "| Rank | Candidate | Condition | RMSE | MAE | R2 | Features |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
        for rank, row in enumerate(ranking, start=1):
            markdown.append(
                f"| {rank} | {row['model_id']} | {row['augmentation']} | "
                f"{row['RMSE_mean']:.4f} | {row['MAE_mean']:.4f} | "
                f"{row['R2_mean']:.4f} | {row['feature_set']} |")
        if rejected:
            markdown += ["", "Rejected:"] + [f"- {item}" for item in rejected]
        markdown.append("")

    payload = _clean({
        "policy": policy, **selections,
        "augmentation_comparison": augmentation.where(
            pd.notna(augmentation), None).to_dict("records"),
        "generalization_gap": gaps.where(pd.notna(gaps), None).to_dict("records"),
        "warning": ("Random-split results are an interpolation baseline and must never be "
                    "presented as unseen-paper generalization."),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    out.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({name: {"candidate_id": selections[name]["candidate_id"],
                             "augmentation": selections[name]["augmentation"]}
                      for name in selections}, indent=2))


if __name__ == "__main__":
    main()
