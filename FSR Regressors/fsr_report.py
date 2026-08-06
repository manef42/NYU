"""Build publication tables and figures exclusively from persisted artifacts.

Nothing is re-tuned here. Every number comes from the merged evaluation
directory, the champion selection file, and the refit artifacts, so the report
can be regenerated at any time without touching a compute-hour of training.

Output layout
-------------
results/report/
  00_overview/          dataset summary, leaderboards, champion README
  01_comparison/        candidate bar chart, strip plot, gap plot, augmentation plot
  02_diagnostics/       per-family pred-vs-true, residuals, error-hist, error-vs-true
  03_error_analysis/    error-vs-feature (top-10 + mandatory), O2/pressure heatmap
  04_hard_papers/       hardest-papers table + figure
  05_learning_curves/   training-size vs RMSE and R2
  06_per_paper/         per-paper RMSE/MAE/R2 histograms and boxplots
  07_explainability/    feature importance, permutation importance, SHAP
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

# Columns that must appear in the error-vs-feature plots regardless of rank
MANDATORY_FEATURE_PLOTS = [
    "Pressure (Pa)",
    "Oxygen Concentration",
    "Flow Velocity (m/s)",
]
TOP_FEATURE_PLOT_TOTAL = 10   # total slots (mandatory fill from the bottom if not in top-N)

# Canonical column names used for the O2 / pressure heatmap
O2_COL = "Oxygen Concentration"
PRESS_COL = "Pressure (Pa)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _shorten(label: str, limit: int = 46) -> str:
    label = str(label)
    return label if len(label) <= limit else label[:limit - 3].rstrip() + "..."


def _save_table(frame: pd.DataFrame, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{name}.csv", index=False)
    try:
        text = frame.to_markdown(index=False)
    except ImportError:
        text = frame.to_string(index=False)
    (out / f"{name}.md").write_text(text + "\n", encoding="utf-8")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _read_predictions(evaluation: Path) -> pd.DataFrame:
    parquet = evaluation / "outer_fold_predictions.parquet"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    return pd.read_csv(evaluation / "outer_fold_predictions.csv")


def _leaderboard(summary: pd.DataFrame, protocol: str) -> pd.DataFrame:
    columns = ["model_id", "model_family", "feature_set", "augmentation",
               "R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
               "MAE_mean", "MAE_std", "MAPE_mean", "NRMSE_mean", "MBE_mean", "fold_count"]
    frame = summary[summary["protocol"] == protocol]
    available = [c for c in columns if c in frame.columns]
    return frame[available].sort_values("R2_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 01_comparison
# ---------------------------------------------------------------------------

def _comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.copy()
    frame["condition"] = frame["protocol"] + " / " + frame["augmentation"]
    order = (frame[frame["protocol"] == "extrapolation_grouped"]
             .sort_values("RMSE_mean")["model_id"].drop_duplicates().tolist())
    order += [m for m in frame["model_id"].unique() if m not in order]
    conditions = [c for c in
                  ["interpolation_random / rd", "interpolation_random / rd_bt",
                   "extrapolation_grouped / rd", "extrapolation_grouped / rd_bt"]
                  if c in set(frame["condition"])]
    palette = ["#4C9AFF", "#80C2FF", "#E07A5F", "#F4A261"]
    positions = np.arange(len(order))
    width = .8 / max(1, len(conditions))
    fig, ax = plt.subplots(figsize=(min(20., max(9., .7 * len(order))), 6.4))
    for i, condition in enumerate(conditions):
        subset = frame[frame["condition"] == condition].set_index("model_id").reindex(order)
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(positions + offset, subset["RMSE_mean"], width=width,
               yerr=subset["RMSE_std"], capsize=3, label=condition,
               color=palette[i % len(palette)], edgecolor="black", linewidth=.5)
    ax.set_xticks(positions, order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Outer-fold RMSE (mean ± SD across repeats)")
    ax.set_title("FSR candidate comparison: interpolation baseline versus unseen papers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, path)


def _strip_plot(fold_metrics: pd.DataFrame, path: Path) -> None:
    frame = fold_metrics[(fold_metrics["protocol"] == "extrapolation_grouped") &
                         (fold_metrics["augmentation"] == "rd")]
    if frame.empty:
        return
    order = frame.groupby("model_id")["RMSE"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(order)), 4.6))
    for i, model_id in enumerate(order):
        values = frame.loc[frame["model_id"] == model_id, "RMSE"].to_numpy(dtype=float)
        jitter = (np.random.default_rng(RANDOM_STATE + i).random(len(values)) - .5) * .18
        ax.scatter(np.full(len(values), i, dtype=float) + jitter, values,
                   color="#E07A5F", edgecolor="black", s=40, zorder=3,
                   label="per repeat" if i == 0 else None)
        if len(values):
            ax.hlines(values.mean(), i - .25, i + .25, colors="black",
                      linewidth=2, label="mean" if i == 0 else None)
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Extrapolation RMSE (real data condition)")
    ax.set_title("Per-repeat unseen-paper RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, path)


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
    _save_fig(fig, path)


def _augmentation_plot(augmentation: pd.DataFrame, path: Path) -> None:
    if augmentation.empty:
        return
    frame = augmentation.copy()
    order = (frame[frame["protocol"] == "extrapolation_grouped"]
             .sort_values("rd_mean")["model_id"].drop_duplicates().tolist())
    order += [m for m in frame["model_id"].unique() if m not in order]
    protocols = [p for p in PROTOCOL_COLORS if p in set(frame["protocol"])]
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
    _save_fig(fig, path)


# ---------------------------------------------------------------------------
# 02_diagnostics
# ---------------------------------------------------------------------------

def _diagnostic_plots(frame: pd.DataFrame, label: str, name: str, out: Path) -> None:
    """pred-vs-true, error-vs-true, residuals and error histogram."""
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    residual = y_pred - y_true
    metrics = regression_metrics(y_true, y_pred)

    # predicted vs actual
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(y_true, y_pred, alpha=.45, edgecolor="black", linewidth=.3, s=26)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    pad = .05 * (high - low + 1e-9)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "r--", lw=1.4, label="1:1 line")
    ax.set(xlim=(low - pad, high + pad), ylim=(low - pad, high + pad),
           xlabel="Experimental FSR (m/s)", ylabel="Predicted FSR (m/s)")
    ax.set_title(f"{label}\n$R^2$={metrics['R2']:.3f}  RMSE={metrics['RMSE']:.4f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_fig(fig, out / f"pred_vs_true_{name}.png")

    # error vs true FSR
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(y_true, residual, alpha=.45, edgecolor="black", linewidth=.3, s=26,
               color="#E07A5F")
    ax.axhline(0, color="red", ls="--", lw=1.4)
    ax.set(xlabel="Experimental FSR (m/s)",
           ylabel="Error  (predicted − experimental)",
           title=f"{label}: error vs true FSR")
    fig.tight_layout()
    _save_fig(fig, out / f"error_vs_true_{name}.png")

    # residuals vs predicted
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(y_pred, residual, alpha=.45, edgecolor="black", linewidth=.3, s=26)
    ax.axhline(0, color="red", ls="--", lw=1.4)
    ax.set(xlabel="Predicted FSR (m/s)", ylabel="Residual (predicted − experimental)",
           title=f"{label}: residuals")
    fig.tight_layout()
    _save_fig(fig, out / f"residuals_{name}.png")

    # error histogram
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.hist(residual, bins=30, color="steelblue", edgecolor="black", alpha=.85)
    ax.axvline(0, color="red", ls="--", lw=1.4)
    ax.axvline(float(residual.mean()), color="orange", lw=1.6,
               label=f"mean bias = {residual.mean():.4f}")
    ax.set(xlabel="Prediction error (predicted − experimental)", ylabel="Count",
           title=f"{label}: error distribution")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, out / f"error_hist_{name}.png")


# ---------------------------------------------------------------------------
# 03_error_analysis
# ---------------------------------------------------------------------------

def _feature_plot_list(importance: pd.DataFrame, data_cols: list[str]) -> list[str]:
    """Return TOP_FEATURE_PLOT_TOTAL feature names: top-N by importance + mandatory extras."""
    ranked = importance["feature"].tolist()
    # keep only features that actually exist as raw data columns
    ranked = [f for f in ranked if f in data_cols]
    top = ranked[:TOP_FEATURE_PLOT_TOTAL]
    extras = [f for f in MANDATORY_FEATURE_PLOTS
              if f in data_cols and f not in top]
    slots_left = TOP_FEATURE_PLOT_TOTAL - len(top)
    top = top + extras[:slots_left]
    # if mandatory pushed us short on top-N, backfill from ranked list
    for f in ranked:
        if len(top) >= TOP_FEATURE_PLOT_TOTAL:
            break
        if f not in top:
            top.append(f)
    return top[:TOP_FEATURE_PLOT_TOTAL]


def _error_vs_feature_plots(frame: pd.DataFrame, data: pd.DataFrame,
                             importance: pd.DataFrame, label: str,
                             name: str, out: Path) -> None:
    """One scatter plot of error vs each of the 10 selected features."""
    if "row_id" not in frame.columns or "row_id" not in data.columns:
        return
    merged = frame.merge(data[["row_id"] + [c for c in data.columns if c != "row_id"]],
                         on="row_id", how="left")
    merged["error"] = merged["y_pred"] - merged["y_true"]
    features = _feature_plot_list(importance, data.columns.tolist())
    for feat in features:
        if feat not in merged.columns:
            continue
        col_vals = pd.to_numeric(merged[feat], errors="coerce")
        if col_vals.isna().all():
            continue
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.scatter(col_vals, merged["error"], alpha=.4, edgecolor="black",
                   linewidth=.3, s=24, color="#4C9AFF")
        ax.axhline(0, color="red", ls="--", lw=1.4)
        ax.set(xlabel=_shorten(feat), ylabel="Error (predicted − experimental)",
               title=f"{label}\nerror vs {_shorten(feat, 40)}")
        fig.tight_layout()
        _save_fig(fig, out / f"error_vs_feature_{name}_{slug(feat)}.png")


def _o2_pressure_heatmap(frame: pd.DataFrame, data: pd.DataFrame,
                          label: str, name: str, out: Path) -> None:
    """2-D binned mean absolute error on O2 vs Pressure grid."""
    if "row_id" not in frame.columns or "row_id" not in data.columns:
        return
    needed = ["row_id", O2_COL, PRESS_COL]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        return
    merged = frame.merge(data[needed], on="row_id", how="left")
    merged["abs_error"] = (merged["y_pred"] - merged["y_true"]).abs()
    o2 = pd.to_numeric(merged[O2_COL], errors="coerce")
    pr = pd.to_numeric(merged[PRESS_COL], errors="coerce")
    valid = merged[o2.notna() & pr.notna()].copy()
    if len(valid) < 9:
        return
    valid["o2_bin"] = pd.cut(pd.to_numeric(valid[O2_COL], errors="coerce"), bins=8)
    valid["pr_bin"] = pd.cut(pd.to_numeric(valid[PRESS_COL], errors="coerce"), bins=8)
    pivot = valid.pivot_table(values="abs_error", index="pr_bin", columns="o2_bin",
                               aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(pivot.astype(float), ax=ax, cmap="YlOrRd", linewidths=.4,
                cbar_kws={"label": "Mean absolute error (m/s)"},
                fmt=".4f", annot=pivot.shape[0] * pivot.shape[1] <= 64)
    ax.set_xlabel("Oxygen Concentration")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title(f"{label}: mean |error| — O₂ × Pressure")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    _save_fig(fig, out / f"heatmap_o2_pressure_{name}.png")


# ---------------------------------------------------------------------------
# 04_hard_papers
# ---------------------------------------------------------------------------

def _hard_papers_plot(per_paper: pd.DataFrame, data: pd.DataFrame,
                       label: str, name: str, out: Path, top_n: int = 10) -> None:
    """Table + horizontal bar chart of the hardest held-out papers."""
    frame = per_paper.copy()
    # attach geometry / material from the raw data if possible
    paper_col = "paper_id" if "paper_id" in frame.columns else None
    geo_col = next((c for c in ["Geometry of Sample", "geometry"] if c in data.columns), None)
    mat_col = next((c for c in ["Material", "Fuel", "fuel"] if c in data.columns), None)

    if paper_col and (geo_col or mat_col):
        agg_cols = {}
        if geo_col:
            agg_cols[geo_col] = lambda x: x.mode().iloc[0] if len(x) else ""
        if mat_col:
            agg_cols[mat_col] = lambda x: x.mode().iloc[0] if len(x) else ""
        meta = data.groupby("paper_id").agg(agg_cols).reset_index()
        frame = frame.merge(meta, on="paper_id", how="left")

    show_cols = [c for c in ["paper_id", "RMSE", "MAE", "R2", "row_count",
                               geo_col, mat_col] if c and c in frame.columns]
    hard = (frame.sort_values("RMSE", ascending=False).head(top_n)[show_cols]
            .rename(columns={"row_count": "Rows", geo_col: "Geometry",
                              mat_col: "Material"}))
    _save_table(hard, f"hardest_papers_{name}", out)

    fig, ax = plt.subplots(figsize=(9, max(4, .45 * top_n)))
    labels = hard["paper_id"].astype(str).str[:40].tolist()
    rmse_vals = hard["RMSE"].to_numpy(dtype=float)
    mae_vals = hard["MAE"].to_numpy(dtype=float) if "MAE" in hard.columns else None
    y = np.arange(len(labels))
    ax.barh(y, rmse_vals[::-1], color="#D55E00", alpha=.85, label="RMSE", edgecolor="black")
    if mae_vals is not None:
        ax.barh(y, mae_vals[::-1], color="#F4A261", alpha=.7, label="MAE", edgecolor="black")
    ax.set_yticks(y, labels[::-1], fontsize=8)
    ax.set_xlabel("Error (m/s)")
    ax.set_title(f"{label}: top-{top_n} hardest papers (extrapolation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out / f"hardest_papers_{name}.png")


# ---------------------------------------------------------------------------
# 05_learning_curves
# ---------------------------------------------------------------------------

def _learning_curves(fold_metrics: pd.DataFrame, model_id: str,
                      label: str, name: str, out: Path) -> None:
    """Plot training-size vs RMSE and R2 across outer folds."""
    needed = {"model_id", "protocol", "augmentation", "RMSE", "R2"}
    size_col = next((c for c in ["train_size", "n_train", "training_size"]
                     if c in fold_metrics.columns), None)
    if size_col is None or not needed.issubset(fold_metrics.columns):
        return
    frame = fold_metrics[(fold_metrics["model_id"] == model_id) &
                         (fold_metrics["augmentation"] == "rd")].copy()
    if frame.empty:
        return
    protocols = [p for p in ["interpolation_random", "extrapolation_grouped"]
                 if p in frame["protocol"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric in zip(axes, ["RMSE", "R2"]):
        for protocol in protocols:
            sub = frame[frame["protocol"] == protocol].sort_values(size_col)
            if sub.empty:
                continue
            color = PROTOCOL_COLORS.get(protocol, "gray")
            ax.scatter(sub[size_col], sub[metric], color=color, s=40,
                       edgecolor="black", linewidth=.4, zorder=3,
                       label=protocol.replace("_", " "))
            # smooth trend
            if len(sub) >= 3:
                z = np.polyfit(sub[size_col].to_numpy(float),
                               sub[metric].to_numpy(float), 1)
                xs = np.linspace(sub[size_col].min(), sub[size_col].max(), 100)
                ax.plot(xs, np.polyval(z, xs), color=color, lw=1.2, ls="--")
        ax.set_xlabel("Training set size (rows)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs training size")
        ax.legend(fontsize=8)
    fig.suptitle(f"{label}: learning curves", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out / f"learning_curves_{name}.png")


# ---------------------------------------------------------------------------
# 06_per_paper
# ---------------------------------------------------------------------------

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
    _save_fig(fig, out / f"per_paper_rmse_mae_hist_{name}.png")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    values = per_paper["R2"].dropna()
    if len(values):
        ax.hist(values, bins=20, color="darkgreen", edgecolor="black", alpha=.85)
    ax.set(title=f"{label}: per-paper $R^2$", xlabel="$R^2$ on one held-out paper",
           ylabel="Number of papers")
    fig.tight_layout()
    _save_fig(fig, out / f"per_paper_r2_hist_{name}.png")

    fig, ax = plt.subplots(figsize=(4.6, 5))
    ax.boxplot(per_paper["RMSE"].dropna(), patch_artist=True,
               boxprops=dict(facecolor="lightcoral"))
    ax.set(ylabel="Per-paper RMSE", title=f"{label}\nper-paper RMSE spread")
    fig.tight_layout()
    _save_fig(fig, out / f"per_paper_rmse_boxplot_{name}.png")


# ---------------------------------------------------------------------------
# 07_explainability
# ---------------------------------------------------------------------------

def _importance_plot(names: list[str], values: np.ndarray, title: str, path: Path,
                     errors: np.ndarray | None = None, top: int = 20) -> None:
    order = np.argsort(values)[::-1][:top][::-1]
    selected = [_shorten(names[i]) for i in order]
    fig, ax = plt.subplots(figsize=(8, max(4, .38 * len(selected))))
    ax.barh(selected, values[order],
            xerr=None if errors is None else errors[order], color="#0072B2", capsize=3)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, path)


def _permutation_importance(model, held_out: pd.DataFrame, out: Path,
                             label: str, repeats: int = 10) -> pd.DataFrame:
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
        rows.append({"feature": feature,
                     "importance_mean": float(np.mean(increases)),
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
    transformed = np.asarray(
        model.preprocessor_.transform(features[model.features]), dtype=np.float32)
    names = list(model.get_feature_names_out())
    values = np.mean(
        [np.asarray(shap.TreeExplainer(estimator).shap_values(transformed))
         for estimator in model.estimators_], axis=0)
    display_names = [_shorten(n) for n in names]
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
    shap.summary_plot(values, transformed, feature_names=display_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title(f"Mean absolute SHAP ({label})")
    plt.tight_layout()
    plt.savefig(out / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    return {"top_features": ranking.head(10).to_dict("records")}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    evaluation = Path(args.evaluation)
    artifacts = Path(args.artifacts)
    root = Path(args.out)

    # sub-folders
    d_overview   = root / "00_overview"
    d_comparison = root / "01_comparison"
    d_diag       = root / "02_diagnostics"
    d_error      = root / "03_error_analysis"
    d_hard       = root / "04_hard_papers"
    d_lc         = root / "05_learning_curves"
    d_pp         = root / "06_per_paper"
    d_expl       = root / "07_explainability"
    for d in (d_overview, d_comparison, d_diag, d_error, d_hard, d_lc, d_pp, d_expl):
        d.mkdir(parents=True, exist_ok=True)

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

    # -- 00_overview ----------------------------------------------------------
    _save_table(pd.DataFrame([{
        "rows": data_report["row_count"],
        "papers": data_report["paper_count"],
        "raw_rows": data_report["raw_row_count"],
        "rows_without_target": data_report["missing_target_count"],
        "duplicates_removed": data_report["duplicate_count"],
        "target_column": data_report["target_column"],
        "group_column": data_report["group_column"],
        "dataset_sha256": data_report["dataset_sha256"],
        "evaluation_integrity_passed": integrity["passed"],
    }]), "dataset_integrity_summary", d_overview)
    _save_table(_leaderboard(summary, "interpolation_random"),
                "interpolation_leaderboard", d_overview)
    _save_table(_leaderboard(summary, "extrapolation_grouped"),
                "extrapolation_leaderboard", d_overview)
    _save_table(gaps, "generalization_gap", d_overview)
    _save_table(augmentation, "augmentation_comparison", d_overview)
    _save_table(stability, "metric_stability", d_overview)
    _save_table(intervals, "bootstrap_intervals", d_overview)

    # -- 01_comparison --------------------------------------------------------
    _comparison_plot(summary, d_comparison / "model_comparison_bars.png")
    _strip_plot(fold_metrics, d_comparison / "per_repeat_group_rmse_strip.png")
    _gap_plot(gaps, d_comparison / "interpolation_extrapolation_gap.png")
    _augmentation_plot(augmentation, d_comparison / "augmentation_comparison.png")

    # -- per-family loop: 02_diagnostics, 03_error_analysis, 04_hard, 06_per_paper --
    best_per_family = (
        summary[(summary["protocol"] == "extrapolation_grouped") &
                (summary["augmentation"] == "rd")]
        .sort_values("RMSE_mean").groupby("model_family", as_index=False).first())
    family_labels: dict = {}
    # placeholder importance (overwritten after explainability block runs)
    current_importance: pd.DataFrame = pd.DataFrame(columns=["feature", "importance_mean"])

    for _, row in best_per_family.iterrows():
        frame = predictions[
            (predictions["model_id"] == row["model_id"]) &
            (predictions["protocol"] == "extrapolation_grouped") &
            (predictions["augmentation"] == "rd")]
        if frame.empty:
            continue
        label = f"{row['model_family']} ({row['model_id']})"
        name = slug(row["model_family"])
        family_labels[row["model_family"]] = row["model_id"]

        _diagnostic_plots(frame, label, name, d_diag)

        family_paper = per_paper[
            (per_paper["model_id"] == row["model_id"]) &
            (per_paper["protocol"] == "extrapolation_grouped") &
            (per_paper["augmentation"] == "rd")]
        if len(family_paper):
            _save_table(family_paper, f"per_paper_metrics_{name}", d_pp)
            _per_paper_plots(family_paper, label, name, d_pp)
            _hard_papers_plot(family_paper, data, label, name, d_hard)

        _learning_curves(fold_metrics, row["model_id"], label, name, d_lc)

    # -- 07_explainability + 03_error_analysis --------------------------------
    explainability_status: dict = {"requested_shap": not args.no_shap}
    index = read_table(evaluation / "explainability_index.csv")

    for _, row in index.iterrows():
        model = joblib.load(evaluation / str(row["model_path"]))
        importance_vals = model.feature_importances()
        if importance_vals is None:
            continue
        names = list(model.get_feature_names_out())
        imp_frame = pd.DataFrame({"feature": names, "importance": importance_vals})\
            .sort_values("importance", ascending=False)
        family = slug(str(row["model_family"]))
        _save_table(imp_frame, f"feature_importance_{family}", d_expl)
        _importance_plot(names, np.asarray(importance_vals),
                         f"{row['model_id']}: impurity feature importance",
                         d_expl / f"feature_importance_{family}.png")

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
        label = str(row["model_id"])
        name = slug(champion_family)

        current_importance = _permutation_importance(model, held_out, d_expl, label)
        explainability_status.update({
            "permutation_importance_model": label,
            "held_out_papers": int(held_out["paper_id"].nunique()),
            "held_out_rows": int(len(held_out)),
            "top_permutation_features": current_importance.head(10).to_dict("records"),
        })

        # error-vs-feature and O2/pressure heatmap use permutation importance ranking
        champ_preds = predictions[
            (predictions["model_id"] == champion_id) &
            (predictions["protocol"] == "extrapolation_grouped") &
            (predictions["augmentation"] == "rd")]
        if not champ_preds.empty:
            _error_vs_feature_plots(champ_preds, data, current_importance,
                                     label, name, d_error)
            _o2_pressure_heatmap(champ_preds, data, label, name, d_error)

        if not args.no_shap:
            try:
                explainability_status.update(
                    {"shap": True, **_shap(model, held_out, d_expl, label)})
            except Exception as exc:
                explainability_status.update({"shap": False, "shap_error": str(exc)})
                (d_expl / "explainability_status.json").write_text(
                    json.dumps(explainability_status, indent=2, default=str),
                    encoding="utf-8")
                raise RuntimeError(f"Required SHAP explanation failed: {exc}") from exc
    else:
        explainability_status["error"] = "No persisted held-out-paper model was available"

    (d_expl / "explainability_status.json").write_text(
        json.dumps(explainability_status, indent=2, default=str), encoding="utf-8")

    # -- README ---------------------------------------------------------------
    interpolation = selection["interpolation_champion"]
    extrapolation = selection["extrapolation_champion"]
    readme = f"""# Flame-spread-rate regression report

Two questions are answered separately and must never be conflated. The random
row split measures interpolation inside papers the model has already partly seen.
The paper-grouped split holds out entire publications and is the only evidence
about transfer to a brand-new campaign or rig.

## Selected champions

- Interpolation: `{interpolation['candidate_id']}` (`{interpolation['augmentation']}`),
  R² {interpolation['metrics']['R2_mean']:.4f} ± {interpolation['metrics']['R2_std']:.4f},
  RMSE {interpolation['metrics']['RMSE_mean']:.4f} ± {interpolation['metrics']['RMSE_std']:.4f}
- Extrapolation: `{extrapolation['candidate_id']}` (`{extrapolation['augmentation']}`),
  R² {extrapolation['metrics']['R2_mean']:.4f} ± {extrapolation['metrics']['R2_std']:.4f},
  RMSE {extrapolation['metrics']['RMSE_mean']:.4f} ± {extrapolation['metrics']['RMSE_std']:.4f}

## Dataset

`{data_report['dataset_path']}` → {data_report['row_count']} rows with a usable
flame spread rate across {data_report['paper_count']} canonical papers; target
column `{data_report['target_column']}`, grouping column
`{data_report['group_column']}`; SHA-256 `{data_report['dataset_sha256']}`.

## Integrity

Evaluation integrity overall: `{integrity['passed']}`. Failed candidate/protocol
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
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    (root / "report_manifest.json").write_text(json.dumps({
        "interpolation_champion": interpolation["candidate_id"],
        "extrapolation_champion": extrapolation["candidate_id"],
        "model_cards": cards,
        "family_representatives": family_labels,
        "explainability": explainability_status,
        "folders": [d.name for d in sorted(root.iterdir()) if d.is_dir()],
    }, indent=2, default=str), encoding="utf-8")
    total_figs = sum(1 for _ in root.rglob("*.png"))
    total_tabs = sum(1 for _ in root.rglob("*.csv"))
    print(json.dumps({"report": str(root),
                      "figures": total_figs,
                      "tables": total_tabs}, indent=2))


if __name__ == "__main__":
    main()
