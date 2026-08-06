from pathlib import Path
from matplotlib.lines import Line2D

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_CSV = "metric_stability.csv"
OUTPUT_DIR = Path("plots")
TOP_N = 12

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.92,
    rc={
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)

PURPLE = "#57068C"
BLUE = "#0072B2"
GRAY = "#A7A9AC"
DARK = "#1F2937"

FAMILY_COLORS = {
    "random_forest": "#009E73",
    "xgboost": "#0072B2",
    "knn": "#E69F00",
    "svr": "#D55E00",
    "mlp": "#CC79A7",
}

# ============================================================
# Load and validate data
# ============================================================
df = pd.read_csv(INPUT_CSV)

required_columns = {
    "model_id", "model_family", "protocol", "augmentation",
    "R2_mean", "R2_std", "RMSE_mean", "RMSE_std",
}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["run_label"] = (
    df["model_id"].str.replace("_", " ", regex=False)
    + "\n("
    + df["augmentation"].astype(str)
    + ")"
)

extra = df.loc[df["protocol"].eq("extrapolation_grouped")].copy()
interp = df.loc[df["protocol"].eq("interpolation_random")].copy()

if extra.empty or interp.empty:
    raise ValueError(
        "Expected both 'extrapolation_grouped' and 'interpolation_random' protocols."
    )

# Retain one row for each model/augmentation candidate.
extra = (
    extra.sort_values("R2_mean", ascending=False)
    .drop_duplicates(["model_id", "augmentation"], keep="first")
    .copy()
)
interp = (
    interp.sort_values("R2_mean", ascending=False)
    .drop_duplicates(["model_id", "augmentation"], keep="first")
    .copy()
)

# ============================================================
# Ranking plot with mean ± standard deviation shown at every point
# ============================================================
def save_ranked_model_plot(
    data,
    metric_mean,
    metric_std,
    metric_display_name,
    protocol_display_name,
    filename,
    higher_is_better,
    top_n=12,
):
    if higher_is_better:
        top = data.nlargest(top_n, metric_mean).copy()
        top = top.sort_values(metric_mean, ascending=False).reset_index(drop=True)
    else:
        top = data.nsmallest(top_n, metric_mean).copy()
        top = top.sort_values(metric_mean, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    y_positions = list(range(len(top)))
    colors = [PURPLE] + [GRAY] * (len(top) - 1)

    ax.errorbar(
        top[metric_mean],
        y_positions,
        xerr=top[metric_std],
        fmt="none",
        ecolor=DARK,
        elinewidth=1.4,
        capsize=3.5,
        zorder=1,
    )

    ax.scatter(
        top[metric_mean],
        y_positions,
        s=135,
        c=colors,
        edgecolor="white",
        linewidth=1.1,
        zorder=2,
    )

    for y, (_, row) in enumerate(top.iterrows()):
        if metric_mean == "R2_mean":
            value_text = f"{row[metric_mean]:.3f} ± {row[metric_std]:.3f}"
        else:
            value_text = f"{row[metric_mean]:.5f} ± {row[metric_std]:.5f}"

        ax.annotate(
            value_text,
            xy=(row[metric_mean], y),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=9.2,
            color=PURPLE if y == 0 else DARK,
            weight="bold" if y == 0 else "normal",
        )

    lower = (top[metric_mean] - top[metric_std]).min()
    upper = (top[metric_mean] + top[metric_std]).max()
    span = max(upper - lower, abs(upper) * 0.1, 0.01)
    ax.set_xlim(lower - 0.06 * span, upper + 0.42 * span)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(top["run_label"])
    ax.invert_yaxis()
    ax.axvline(0, color=DARK, linestyle="--", linewidth=1.1)

    if metric_mean == "R2_mean":
        x_label = r"Mean $R^2$"
    else:
        x_label = "Mean RMSE"

    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_title(
        f"{protocol_display_name}",
        loc="left",
        weight="bold",
        pad=15,
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Combined interpolation/extrapolation accuracy–stability plot
# The x-axis contains two identical R² scales, one per shaded region.
# ============================================================
def save_combined_accuracy_stability_plot(extra_data, interp_data, filename):
    extra_valid = extra_data.loc[
        extra_data["R2_mean"].between(0, 1, inclusive="both")
    ].copy()
    interp_valid = interp_data.loc[
        interp_data["R2_mean"].between(0, 1, inclusive="both")
    ].copy()

    if extra_valid.empty or interp_valid.empty:
        raise ValueError("Both protocols need at least one result with 0 ≤ R² ≤ 1.")

    # Both panels share a common RMSE axis. Each region has its own 0–1 R² scale.
    extra_valid["plot_x"] = extra_valid["R2_mean"]
    interp_valid["plot_x"] = interp_valid["R2_mean"] + 1.25

    fig, ax = plt.subplots(figsize=(13.33, 7.5))

    # Background regions: left = extrapolation, right = interpolation.
    ax.axvspan(-0.06, 1.06, color="#EDE3F4", alpha=0.95, zorder=0)
    ax.axvspan(1.19, 2.31, color="#E4F0FA", alpha=0.95, zorder=0)
    ax.axvline(1.125, color=DARK, linestyle="--", linewidth=1.1, zorder=1)

    for data, marker in [(extra_valid, "o"), (interp_valid, "o")]:
        for family, group in data.groupby("model_family"):
            ax.scatter(
                group["plot_x"],
                group["RMSE_mean"],
                s=88,
                marker=marker,
                color=FAMILY_COLORS.get(family, GRAY),
                alpha=0.78,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    # Highlight and label only the best-R² result from each protocol.
    best_extra = extra_valid.loc[extra_valid["R2_mean"].idxmax()]
    best_interp = interp_valid.loc[interp_valid["R2_mean"].idxmax()]

    for best, protocol_name, marker in [
        (best_extra, "Extrapolation", "o"),
        (best_interp, "Interpolation", "o"),
    ]:
        ax.scatter(
            best["plot_x"],
            best["RMSE_mean"],
            s=235,
            marker=marker,
            color=PURPLE,
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        ax.annotate(
            f"{best['model_id'].replace('_', ' ')}\n"
            f"({best['augmentation']})\n"
            f"$R^2$ = {best['R2_mean']:.3f}\n"
            f"RMSE = {best['RMSE_mean']:.5f}",
            xy=(best["plot_x"], best["RMSE_mean"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9.5,
            color=PURPLE,
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="white",
                edgecolor=PURPLE,
                alpha=0.96,
            ),
            arrowprops=dict(arrowstyle="-", color=PURPLE, linewidth=1.1),
            zorder=6,
        )

    r2_ticks = [0, 0.25, 0.50, 0.75, 1.00]
    ax.set_xticks(r2_ticks + [tick + 1.25 for tick in r2_ticks])
    ax.set_xticklabels([f"{tick:.2f}" for tick in r2_ticks] * 2)
    ax.set_xlim(-0.06, 2.31)

    y_top = ax.get_ylim()[1]
    ax.text(
        0.50,
        y_top,
        "Grouped extrapolation",
        ha="center",
        va="bottom",
        fontsize=13,
        weight="bold",
        color=PURPLE,
    )
    ax.text(
        1.75,
        y_top,
        "Random interpolation",
        ha="center",
        va="bottom",
        fontsize=13,
        weight="bold",
        color=BLUE,
    )

    family_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=9,
            label=family.replace("_", " ").title(),
        )
        for family, color in FAMILY_COLORS.items()
        if family in set(extra_valid["model_family"]) | set(interp_valid["model_family"])
    ]
    protocol_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color="black",
            markersize=8,
            label="Grouped extrapolation",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color="black",
            markersize=8,
            label="Random interpolation",
        ),
    ]

    legend_1 = ax.legend(
        handles=family_handles,
        title="Model family",
        loc="upper right",
        frameon=True,
        fontsize=9,
    )
    ax.add_artist(legend_1)


    ax.set_xlabel(r"Mean $R^2$")
    ax.set_ylabel("Mean RMSE")
    ax.set_title(
        r"Accuracy–Error Tradeoff: Grouped Extrapolation vs. Random Interpolation",
        loc="left",
        weight="bold",
        pad=22,
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# R² ranking figures
# ============================================================
save_ranked_model_plot(
    data=extra,
    metric_mean="R2_mean",
    metric_std="R2_std",
    metric_display_name=r"",
    protocol_display_name="Grouped Extrapolation",
    filename="r2_grouped_extrapolation_model_ranking.png",
    higher_is_better=True,
    top_n=TOP_N,
)

save_ranked_model_plot(
    data=interp,
    metric_mean="R2_mean",
    metric_std="R2_std",
    metric_display_name=r"",
    protocol_display_name="Random Interpolation",
    filename="r2_random_interpolation_model_ranking.png",
    higher_is_better=True,
    top_n=TOP_N,
)

# ============================================================
# Combined accuracy–stability figure
# ============================================================
save_combined_accuracy_stability_plot(
    extra_data=extra,
    interp_data=interp,
    filename="accuracy_stability_extrapolation_vs_interpolation.png",
)

# ============================================================
# RMSE ranking figures
# ============================================================
save_ranked_model_plot(
    data=extra,
    metric_mean="RMSE_mean",
    metric_std="RMSE_std",
    metric_display_name="Grouped-Extrapolation RMSE",
    protocol_display_name="Grouped Extrapolation",
    filename="rmse_grouped_extrapolation_model_ranking.png",
    higher_is_better=False,
    top_n=TOP_N,
)

save_ranked_model_plot(
    data=interp,
    metric_mean="RMSE_mean",
    metric_std="RMSE_std",
    metric_display_name="Random-Interpolation RMSE",
    protocol_display_name="Random Interpolation",
    filename="rmse_random_interpolation_model_ranking.png",
    higher_is_better=False,
    top_n=TOP_N,
)

best_extra_r2 = extra.loc[extra["R2_mean"].idxmax()]
best_interp_r2 = interp.loc[interp["R2_mean"].idxmax()]
best_extra_rmse = extra.loc[extra["RMSE_mean"].idxmin()]
best_interp_rmse = interp.loc[interp["RMSE_mean"].idxmin()]

print("\nBEST GROUPED-EXTRAPOLATION R² MODEL")
print(f"{best_extra_r2['model_id']} ({best_extra_r2['augmentation']}): "
      f"R² = {best_extra_r2['R2_mean']:.3f} ± {best_extra_r2['R2_std']:.3f}; "
      f"RMSE = {best_extra_r2['RMSE_mean']:.5f} ± {best_extra_r2['RMSE_std']:.5f}")

print("\nBEST RANDOM-INTERPOLATION R² MODEL")
print(f"{best_interp_r2['model_id']} ({best_interp_r2['augmentation']}): "
      f"R² = {best_interp_r2['R2_mean']:.3f} ± {best_interp_r2['R2_std']:.3f}; "
      f"RMSE = {best_interp_r2['RMSE_mean']:.5f} ± {best_interp_r2['RMSE_std']:.5f}")

print("\nLOWEST GROUPED-EXTRAPOLATION RMSE MODEL")
print(f"{best_extra_rmse['model_id']} ({best_extra_rmse['augmentation']}): "
      f"RMSE = {best_extra_rmse['RMSE_mean']:.5f} ± {best_extra_rmse['RMSE_std']:.5f}")

print("\nLOWEST RANDOM-INTERPOLATION RMSE MODEL")
print(f"{best_interp_rmse['model_id']} ({best_interp_rmse['augmentation']}): "
      f"RMSE = {best_interp_rmse['RMSE_mean']:.5f} ± {best_interp_rmse['RMSE_std']:.5f}")

print(f"\nSaved five figures in: {OUTPUT_DIR.resolve()}")