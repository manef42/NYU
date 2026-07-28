"""Build publication tables and figures exclusively from persisted artifacts.

Nothing is re-tuned here. Every number comes from the merged evaluation
directory, the champion selection file, and the refit artifacts, so the report
can be regenerated at any time without touching a GPU-hour of training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fsr_aggregate import read_table
from fsr_common import RANDOM_STATE, load_data, regression_metrics, slug

sns.set_theme(style="whitegrid", context="paper")
PROTOCOL_COLORS = {"interpolation_random": "#0072B2", "extrapolation_grouped": "#D55E00"}
CONDITION_COLORS = {"rd": "#0072B2", "rd_bt": "#E69F00"}
FAMILY_ORDER = ("decision_tree", "xgboost", "knn", "mlp")


def _shorten(label: str, limit: int = 46) -> str:
    """Auto-detected database headers are long; axis labels stay readable."""
    label = str(label)
    return label if len(label) <= limit else label[:limit - 3].rstrip() + "..."


def _save_table(frame: pd.DataFrame, name: str, out: Path) -> None:
    frame.to_csv(out / f"{name}.csv", index=False)
    try:
        text = frame.to_markdown(index=False)
    except ImportError:
        text = frame.to_string(index=False)
    (out / f"{name}.md").write_text(text + "\n", encoding="utf-8")


def _read_predictions(evaluation: Path) -> pd.DataFrame:
    parquet = evaluation / "outer_fold_predictions.parquet"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    return pd.read_csv(evaluation / "outer_fold_predictions.csv")


def _leaderboard(summary: pd.DataFrame, protocol: str) -> pd.DataFrame:
    columns = ["model_id", "model_family", "feature_set", "augmentation",
               "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std",
               "MAPE_mean", "NRMSE_mean", "MBE_mean", "fold_count"]
    frame = summary[summary["protocol"] == protocol]
    available = [column for column in columns if column in frame.columns]
    return frame[available].sort_values("RMSE_mean").reset_index(drop=True)


def _comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    """Mean +/- SD RMSE for every candidate under all protocol/condition pairs."""
    frame = summary.copy()
    frame["condition"] = frame["protocol"] + " / " + frame["augmentation"]
    order = (frame[frame["protocol"] == "extrapolation_grouped"]
             .sort_values("RMSE_mean")["model_id"].drop_duplicates().tolist())
    order += [model for model in frame["model_id"].unique() if model not in order]
    conditions = [condition for condition in
                  ["interpolation_random / rd", "interpolation_random / rd_bt",
                   "extrapolation_grouped / rd", "extrapolation_grouped / rd_bt"]
                  if condition in set(frame["condition"])]
    palette = ["#4C9AFF", "#80C2FF", "#E07A5F", "#F4A261"]
    positions = np.arange(len(order))
    width = .8 / max(1, len(conditions))
    fig, ax = plt.subplots(figsize=(min(20., max(9., .7 * len(order))), 6.4))
    for index, condition in enumerate(conditions):
        subset = frame[frame["condition"] == condition].set_index("model_id").reindex(order)
        offset = (index - (len(conditions) - 1) / 2) * width
        ax.bar(positions + offset, subset["RMSE_mean"], width=width,
               yerr=subset["RMSE_std"], capsize=3, label=condition,
               color=palette[index % len(palette)], edgecolor="black", linewidth=.5)
    ax.set_xticks(positions, order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Outer-fold RMSE (mean ± SD across repeats)")
    ax.set_title("FSR candidate comparison: interpolation baseline versus unseen papers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _strip_plot(fold_metrics: pd.DataFrame, path: Path) -> None:
    """Per-repeat extrapolation RMSE so run-to-run variability stays visible."""
    frame = fold_metrics[(fold_metrics["protocol"] == "extrapolation_grouped") &
                         (fold_metrics["augmentation"] == "rd")]
    if frame.empty:
        return
    order = frame.groupby("model_id")["RMSE"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(order)), 4.6))
    for index, model_id in enumerate(order):
        values = frame.loc[frame["model_id"] == model_id, "RMSE"].to_numpy(dtype=float)
        jitter = (np.random.default_rng(RANDOM_STATE + index).random(len(values)) - .5) * .18
        ax.scatter(np.full(len(values), index, dtype=float) + jitter, values,
                   color="#E07A5F", edgecolor="black", s=40, zorder=3,
                   label="per repeat" if index == 0 else None)
        if len(values):
            ax.hlines(values.mean(), index - .25, index + .25, colors="black",
                      linewidth=2, label="mean" if index == 0 else None)
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Extrapolation RMSE (real data condition)")
    ax.set_title("Per-repeat unseen-paper RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _gap_plot(gaps: pd.DataFrame, path: Path) -> None:
    frame = gaps[gaps["augmentation"] == "rd"].sort_values("generalization_gap_mean")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, .35 * len(frame))))
    colors = np.where(frame["generalization_gap_mean"] > 0, "#D55E00", "#0072B2")
    ax.barh(frame["model_id"], frame["generalization_gap_mean"],
            xerr=frame["generalization_gap_std"], color=colors, capsize=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("RMSE on unseen papers minus RMSE on random rows")
    ax.set_title("Generalization gap (positive = paper-specific overfitting)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _augmentation_plot(augmentation: pd.DataFrame, path: Path) -> None:
    if augmentation.empty:
        return
    frame = augmentation.copy()
    order = (frame[frame["protocol"] == "extrapolation_grouped"]
             .sort_values("rd_mean")["model_id"].drop_duplicates().tolist())
    order += [model for model in frame["model_id"].unique() if model not in order]
    protocols = [protocol for protocol in PROTOCOL_COLORS
                 if protocol in set(frame["protocol"])]
    fig, axes = plt.subplots(1, max(1, len(protocols)),
                             figsize=(6.5 * max(1, len(protocols)), 4.6), squeeze=False)
    positions = np.arange(len(order))
    for axis, protocol in zip(axes[0], protocols):
        subset = frame[frame["protocol"] == protocol].set_index("model_id").reindex(order)
        axis.bar(positions - .2, subset["rd_mean"], width=.4, yerr=subset["rd_std"],
                 capsize=3, label="real data", color=CONDITION_COLORS["rd"],
                 edgecolor="black", linewidth=.5)
        axis.bar(positions + .2, subset["rd_bt_mean"], width=.4, yerr=subset["rd_bt_std"],
                 capsize=3, label="real data + bootstrap", color=CONDITION_COLORS["rd_bt"],
                 edgecolor="black", linewidth=.5)
        axis.set_xticks(positions, order, rotation=30, ha="right", fontsize=8)
        axis.set_title(protocol.replace("_", " "))
        axis.set_ylabel("RMSE (mean ± SD)")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _diagnostic_plots(frame: pd.DataFrame, label: str, name: str, out: Path) -> None:
    """Predicted-versus-experimental, residual and error-histogram panels."""
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(y_true, y_pred, alpha=.45, edgecolor="black", linewidth=.3, s=26)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    pad = .05 * (high - low + 1e-9)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "r--", lw=1.4, label="1:1 line")
    ax.set(xlim=(low - pad, high + pad), ylim=(low - pad, high + pad),
           xlabel="Experimental FSR", ylabel="Predicted FSR")
    ax.set_title(f"{label}\n$R^2$={metrics['R2']:.3f}  RMSE={metrics['RMSE']:.3f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"pred_vs_true_{name}.png", dpi=300)
    plt.close(fig)

    residual = y_pred - y_true
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(y_pred, residual, alpha=.45, edgecolor="black", linewidth=.3, s=26)
    ax.axhline(0, color="red", ls="--", lw=1.4)
    ax.set(xlabel="Predicted FSR", ylabel="Residual (predicted - experimental)",
           title=f"{label}: residuals")
    fig.tight_layout()
    fig.savefig(out / f"residuals_{name}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.hist(residual, bins=30, color="steelblue", edgecolor="black", alpha=.85)
    ax.axvline(0, color="red", ls="--", lw=1.4)
    ax.axvline(float(residual.mean()), color="orange", lw=1.6,
               label=f"mean bias = {residual.mean():.3f}")
    ax.set(xlabel="Prediction error (predicted - experimental)", ylabel="Count",
           title=f"{label}: error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"error_hist_{name}.png", dpi=300)
    plt.close(fig)


def _per_paper_plots(per_paper: pd.DataFrame, label: str, name: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(per_paper["RMSE"].dropna(), bins=20, color="indianred",
                 edgecolor="black", alpha=.85)
    axes[0].set(title="Per-paper RMSE", xlabel="RMSE", ylabel="Number of papers")
    axes[1].hist(per_paper["MAE"].dropna(), bins=20, color="slateblue",
                 edgecolor="black", alpha=.85)
    axes[1].set(title="Per-paper MAE", xlabel="MAE", ylabel="Number of papers")
    fig.suptitle(f"{label}: per-paper error distributions on unseen papers")
    fig.tight_layout()
    fig.savefig(out / f"per_paper_rmse_mae_hist_{name}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    values = per_paper["R2"].dropna()
    if len(values):
        ax.hist(values, bins=20, color="darkgreen", edgecolor="black", alpha=.85)
    ax.set(title=f"{label}: per-paper $R^2$", xlabel="$R^2$ on one held-out paper",
           ylabel="Number of papers")
    fig.tight_layout()
    fig.savefig(out / f"per_paper_r2_hist_{name}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 5))
    ax.boxplot(per_paper["RMSE"].dropna(), patch_artist=True,
               boxprops=dict(facecolor="lightcoral"))
    ax.set(ylabel="Per-paper RMSE", title=f"{label}\nper-paper RMSE spread")
    fig.tight_layout()
    fig.savefig(out / f"per_paper_rmse_boxplot_{name}.png", dpi=300)
    plt.close(fig)


def _importance_plot(names: list[str], values: np.ndarray, title: str, path: Path,
                     errors: np.ndarray | None = None, top: int = 20) -> None:
    order = np.argsort(values)[::-1][:top][::-1]
    selected = [_shorten(names[index]) for index in order]
    fig, ax = plt.subplots(figsize=(8, max(4, .38 * len(selected))))
    ax.barh(selected, values[order],
            xerr=None if errors is None else errors[order], color="#0072B2", capsize=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _permutation_importance(model, held_out: pd.DataFrame, out: Path, label: str,
                           repeats: int = 10) -> pd.DataFrame:
    """Held-out-paper permutation importance measured as an RMSE increase."""
    y = held_out["y_true"].to_numpy(dtype=float)
    frame = held_out.drop(columns=["y_true"])
    baseline = regression_metrics(y, model.predict(frame))["RMSE"]
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for feature in model.features:
        increases = []
        for _ in range(repeats):
            shuffled = frame.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            increases.append(
                regression_metrics(y, model.predict(shuffled))["RMSE"] - baseline)
        rows.append({"feature": feature, "importance_mean": float(np.mean(increases)),
                     "importance_std": float(np.std(increases, ddof=1)),
                     "held_out_rmse": baseline})
    importance = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    _save_table(importance, "permutation_importance", out)
    _importance_plot(importance["feature"].tolist(),
                     importance["importance_mean"].to_numpy(),
                     f"Permutation importance on unseen papers ({label})",
                     out / "permutation_importance.png",
                     importance["importance_std"].to_numpy())
    return importance


def _shap(model, held_out: pd.DataFrame, out: Path, label: str) -> dict:
    import shap
    features = held_out.drop(columns=["y_true"])
    transformed = np.asarray(model.preprocessor_.transform(features[model.features]),
                             dtype=np.float32)
    names = list(model.get_feature_names_out())
    values = np.mean([np.asarray(shap.TreeExplainer(estimator).shap_values(transformed))
                      for estimator in model.estimators_], axis=0)
    display_names = [_shorten(name) for name in names]
    ranking = pd.DataFrame({"feature": names,
                            "mean_absolute_shap": np.abs(values).mean(axis=0)}
                           ).sort_values("mean_absolute_shap", ascending=False)
    _save_table(ranking, "shap_ranking", out)
    shap.summary_plot(values, transformed, feature_names=display_names, show=False,
                      max_display=20)
    plt.title(f"SHAP on unseen papers ({label})")
    plt.tight_layout()
    plt.savefig(out / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    shap.summary_plot(values, transformed, feature_names=display_names, plot_type="bar",
                      show=False, max_display=20)
    plt.title(f"Mean absolute SHAP ({label})")
    plt.tight_layout()
    plt.savefig(out / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    return {"top_features": ranking.head(10).to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip the SHAP figures (kept for quick report reruns).")
    args = parser.parse_args()
    evaluation = Path(args.evaluation)
    artifacts = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, data_report = load_data(args.data, require_target=True)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    summary = pd.read_csv(evaluation / "summary_metrics.csv")
    fold_metrics = pd.read_csv(evaluation / "fold_metrics.csv")
    predictions = _read_predictions(evaluation)
    per_paper = pd.read_csv(evaluation / "per_paper_metrics.csv")
    gaps = pd.read_csv(evaluation / "generalization_gap.csv")
    augmentation = pd.read_csv(evaluation / "augmentation_comparison.csv")
    stability = pd.read_csv(evaluation / "metric_stability.csv")
    intervals = pd.read_csv(evaluation / "bootstrap_intervals.csv")
    integrity = json.loads((evaluation / "integrity_checks.json").read_text("utf-8"))
    cards = {}
    for champion in ("interpolation", "extrapolation"):
        path = artifacts / f"{champion}_champion" / "model_card.json"
        if path.is_file():
            cards[champion] = json.loads(path.read_text(encoding="utf-8"))

    _save_table(pd.DataFrame([{
        "rows": data_report["row_count"], "papers": data_report["paper_count"],
        "raw_rows": data_report["raw_row_count"],
        "rows_without_target": data_report["missing_target_count"],
        "duplicates_removed": data_report["duplicate_count"],
        "target_column": data_report["target_column"],
        "group_column": data_report["group_column"],
        "dataset_sha256": data_report["dataset_sha256"],
        "evaluation_integrity_passed": integrity["passed"],
    }]), "dataset_integrity_summary", out)
    _save_table(_leaderboard(summary, "interpolation_random"),
                "interpolation_leaderboard", out)
    _save_table(_leaderboard(summary, "extrapolation_grouped"),
                "extrapolation_leaderboard", out)
    _save_table(gaps, "generalization_gap", out)
    _save_table(augmentation, "augmentation_comparison", out)
    _save_table(stability, "metric_stability", out)
    _save_table(intervals, "bootstrap_intervals", out)

    _comparison_plot(summary, out / "model_comparison_bars.png")
    _strip_plot(fold_metrics, out / "per_repeat_group_rmse_strip.png")
    _gap_plot(gaps, out / "interpolation_extrapolation_gap.png")
    _augmentation_plot(augmentation, out / "augmentation_comparison.png")

    best_per_family = (summary[(summary["protocol"] == "extrapolation_grouped") &
                               (summary["augmentation"] == "rd")]
                       .sort_values("RMSE_mean").groupby("model_family", as_index=False)
                       .first())
    family_labels = {}
    for _, row in best_per_family.iterrows():
        frame = predictions[(predictions["model_id"] == row["model_id"]) &
                            (predictions["protocol"] == "extrapolation_grouped") &
                            (predictions["augmentation"] == "rd")]
        if frame.empty:
            continue
        label = f"{row['model_family']} ({row['model_id']})"
        name = slug(row["model_family"])
        family_labels[row["model_family"]] = row["model_id"]
        _diagnostic_plots(frame, label, name, out)
        family_paper = per_paper[(per_paper["model_id"] == row["model_id"]) &
                                 (per_paper["protocol"] == "extrapolation_grouped") &
                                 (per_paper["augmentation"] == "rd")]
        if len(family_paper):
            _save_table(family_paper, f"per_paper_metrics_{name}", out)
            _per_paper_plots(family_paper, label, name, out)

    explainability_status: dict = {"requested_shap": not args.no_shap}
    index = read_table(evaluation / "explainability_index.csv")
    for _, row in index.iterrows():
        model = joblib.load(evaluation / str(row["model_path"]))
        importance = model.feature_importances()
        if importance is None:
            continue
        names = list(model.get_feature_names_out())
        frame = pd.DataFrame({"feature": names, "importance": importance}).sort_values(
            "importance", ascending=False)
        family = slug(str(row["model_family"]))
        _save_table(frame, f"feature_importance_{family}", out)
        _importance_plot(names, np.asarray(importance),
                         f"{row['model_id']}: impurity feature importance",
                         out / f"feature_importance_{family}.png")

    champion_id = selection["extrapolation_champion"]["candidate_id"]
    champion_family = selection["extrapolation_champion"]["model_family"]
    preferred = index[index["model_id"] == champion_id]
    if preferred.empty:
        preferred = index[index["model_family"] == champion_family]
    explained = preferred if len(preferred) else index
    if len(explained):
        row = explained.iloc[0]
        model = joblib.load(evaluation / str(row["model_path"]))
        held_out_ids = set(json.loads(row["held_out_row_ids"]))
        held_out = data[data["row_id"].isin(held_out_ids)].copy()
        held_out["y_true"] = held_out["fsr"].to_numpy(dtype=float)
        label = f"{row['model_id']}"
        importance = _permutation_importance(model, held_out, out, label)
        explainability_status.update({
            "permutation_importance_model": str(row["model_id"]),
            "held_out_papers": int(held_out["paper_id"].nunique()),
            "held_out_rows": int(len(held_out)),
            "top_permutation_features": importance.head(10).to_dict("records"),
        })
        if not args.no_shap:
            try:
                explainability_status.update(
                    {"shap": True, **_shap(model, held_out, out, label)})
            except Exception as exc:
                explainability_status.update({"shap": False, "shap_error": str(exc)})
                (out / "explainability_status.json").write_text(
                    json.dumps(explainability_status, indent=2, default=str),
                    encoding="utf-8")
                raise RuntimeError(f"Required SHAP explanation failed: {exc}") from exc
    else:
        explainability_status["error"] = "No persisted held-out-paper model was available"
    (out / "explainability_status.json").write_text(
        json.dumps(explainability_status, indent=2, default=str), encoding="utf-8")

    interpolation = selection["interpolation_champion"]
    extrapolation = selection["extrapolation_champion"]
    readme = f"""# Flame-spread-rate regression report

Two questions are answered separately and must never be conflated. The random
row split measures interpolation inside papers the model has already partly seen.
The paper-grouped split holds out entire publications and is the only evidence
about transfer to a brand-new campaign or rig.

## Selected champions

- Interpolation: `{interpolation['candidate_id']}` (`{interpolation['augmentation']}`),
  RMSE {interpolation['metrics']['RMSE_mean']:.4f} ± {interpolation['metrics']['RMSE_std']:.4f}
- Extrapolation: `{extrapolation['candidate_id']}` (`{extrapolation['augmentation']}`),
  RMSE {extrapolation['metrics']['RMSE_mean']:.4f} ± {extrapolation['metrics']['RMSE_std']:.4f}

## Dataset

`{data_report['dataset_path']}` -> {data_report['row_count']} rows with a usable
flame spread rate across {data_report['paper_count']} canonical papers; target
column `{data_report['target_column']}`, grouping column
`{data_report['group_column']}`; SHA-256 `{data_report['dataset_sha256']}`.

## Integrity

Evaluation integrity overall: `{integrity['passed']}`. Failed candidate/protocol
combinations are excluded from every table and documented in
`../evaluation/integrity_checks.json`.

## Figures

- `model_comparison_bars.png` - candidate RMSE under all protocol/condition pairs.
- `per_repeat_group_rmse_strip.png` - unseen-paper RMSE for every repeat.
- `interpolation_extrapolation_gap.png` - generalization gap per candidate.
- `augmentation_comparison.png` - bootstrap-augmentation effect per protocol.
- `pred_vs_true_*`, `residuals_*`, `error_hist_*` - per-family diagnostics on unseen papers.
- `per_paper_*` - per-paper error distributions on unseen papers.
- `feature_importance_*`, `permutation_importance.*`, `shap_*` - explainability.

Evaluation artifacts are unbiased comparison evidence, not deployable models.
Refit artifacts are deployable models, not unbiased evaluation evidence.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "report_manifest.json").write_text(json.dumps({
        "interpolation_champion": interpolation["candidate_id"],
        "extrapolation_champion": extrapolation["candidate_id"],
        "model_cards": cards,
        "family_representatives": family_labels,
        "explainability": explainability_status,
        "tables": sorted(path.name for path in out.glob("*.csv")),
        "figures": sorted(path.name for path in out.glob("*.png")),
    }, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"report": str(out),
                      "figures": len(list(out.glob("*.png"))),
                      "tables": len(list(out.glob("*.csv")))}, indent=2))


if __name__ == "__main__":
    main()
