"""Generate the report's figures from persisted evaluation artifacts.

The script intentionally performs no model fitting.  Every plotted value is
read from a version-controlled result table so the figures remain traceable.
Run from the repository root with:

    python Report/generate_report_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Report" / "figures"
BLUE = "#243B6B"
RED = "#B7352D"
GOLD = "#D8A321"
GREY = "#667085"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def dataset_composition() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45))
    axes[0].bar(["Ignition", "No ignition"], [3383, 1104], color=[BLUE, GREY])
    axes[0].set_ylabel("Deduplicated observations")
    axes[0].set_title("Ignition-classification subset")
    for i, value in enumerate([3383, 1104]):
        axes[0].text(i, value + 70, f"{value:,}", ha="center", va="bottom")

    axes[1].bar(["Ignition task", "FSR task"], [87, 66], color=[BLUE, GOLD])
    axes[1].set_ylabel("Canonical papers")
    axes[1].set_title("Source coverage by task")
    for i, value in enumerate([87, 66]):
        axes[1].text(i, value + 1.5, str(value), ha="center", va="bottom")
    fig.tight_layout(w_pad=2.0)
    save(fig, "dataset_composition")


def protocol_comparison() -> None:
    ignition = pd.read_csv(
        ROOT / "Ignition Classifiers" / "results" / "report"
        / "interpolation_extrapolation_drop.csv"
    )
    ignition = ignition[ignition["model_id"] == "xgb_baseline"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    axes[0].bar(
        ["Row-level\ninterpolation", "Paper-grouped\nextrapolation"],
        [
            ignition["interpolation_stratified"],
            ignition["extrapolation_grouped"],
        ],
        color=[BLUE, RED],
    )
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("ROC--AUC")
    axes[0].set_title("Ignition: XGBoost baseline")
    axes[0].axhline(0.5, color="black", linewidth=0.7, linestyle=":")

    fsr_interpolation = pd.read_csv(
        ROOT / "FSR Regressors" / "results" / "report" / "00_overview"
        / "interpolation_leaderboard.csv"
    )
    fsr_extrapolation = pd.read_csv(
        ROOT / "FSR Regressors" / "results" / "report" / "00_overview"
        / "extrapolation_leaderboard.csv"
    )
    i = fsr_interpolation.query(
        "model_id == 'rf_baseline' and augmentation == 'rd'"
    ).iloc[0]
    e = fsr_extrapolation.query(
        "model_id == 'rf_baseline' and augmentation == 'rd'"
    ).iloc[0]
    axes[1].bar(
        ["Row-level\ninterpolation", "Paper-grouped\nextrapolation"],
        [i["R2_mean"], e["R2_mean"]],
        yerr=[i["R2_std"], e["R2_std"]],
        capsize=3,
        color=[BLUE, RED],
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel(r"$R^2$ (mean $\pm$ fold SD)")
    axes[1].set_title("FSR: random-forest baseline")
    fig.tight_layout(w_pad=2.0)
    save(fig, "protocol_comparison")


def feature_rankings() -> None:
    ignition = pd.read_csv(
        ROOT / "Ignition Classifiers" / "results" / "report"
        / "xgboost_shap_ranking.csv"
    ).head(10)
    ignition["feature"] = (
        ignition["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("_", " ")
    )

    fsr = pd.read_csv(
        ROOT / "FSR Regressors" / "results" / "report" / "07_explainability"
        / "shap_ranking.csv"
    ).head(10)
    fsr["feature"] = fsr["feature"].str.replace("_", " ")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9))
    for ax, data, title, color in [
        (axes[0], ignition, "Ignition classifier", BLUE),
        (axes[1], fsr, "FSR regressor", GOLD),
    ]:
        data = data.iloc[::-1]
        ax.barh(data["feature"], data["mean_absolute_shap"], color=color)
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout(w_pad=1.7)
    save(fig, "shap_rankings")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    dataset_composition()
    protocol_comparison()
    feature_rankings()


if __name__ == "__main__":
    main()
