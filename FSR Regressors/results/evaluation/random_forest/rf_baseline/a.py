from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

OUTER_PREDICTIONS_CSV = HERE / "outer_fold_predictions.csv"
INNER_PREDICTIONS_CSV = HERE / "inner_oof_predictions.csv"

EXTRAPOLATION_PNG = HERE / "pred_vs_true_random_forest_extrapolation.png"
INTERPOLATION_RAW_PNG = HERE / "pred_vs_true_random_forest_interpolation_raw.png"
INTERPOLATION_AVG_PNG = HERE / "pred_vs_true_random_forest_interpolation_averaged.png"


# ---------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------
def plot_pred_vs_true(df, output_path, title):
    """
    Create a predicted-vs-true FSR plot with a 1:1 reference line.

    Required columns in df:
        y_true
        y_pred
    """
    plot_df = df[["y_true", "y_pred"]].copy()

    plot_df["y_true"] = pd.to_numeric(plot_df["y_true"], errors="coerce")
    plot_df["y_pred"] = pd.to_numeric(plot_df["y_pred"], errors="coerce")
    plot_df = plot_df.dropna()

    if plot_df.empty:
        raise ValueError(f"No valid y_true/y_pred values available for: {title}")

    y_true = plot_df["y_true"].to_numpy()
    y_pred = plot_df["y_pred"].to_numpy()

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_pred - y_true))
    bias = np.mean(y_pred - y_true)

    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())

    span = upper - lower
    padding = 0.05 * span if span > 0 else 1.0
    limits = (lower - padding, upper + padding)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=250)

    ax.scatter(
        y_true,
        y_pred,
        s=24,
        alpha=0.55,
        color="#1f77b4",
        edgecolor="none",
        label=f"Predictions, n = {len(plot_df):,}",
        rasterized=True,
    )

    ax.plot(
        limits,
        limits,
        linestyle="--",
        linewidth=1.7,
        color="black",
        label="1:1 line",
        zorder=3,
    )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("True FSR")
    ax.set_ylabel("Predicted FSR")
    ax.set_title(title, pad=12)

    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="upper left", frameon=True)

    metrics_text = (
        f"$R^2$ = {r2:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"Bias = {bias:.4f}"
    )

    ax.text(
        0.98,
        0.04,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.92,
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output_path.name}")
    print(f"  Number of prediction records: {len(plot_df):,}")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  Bias: {bias:.4f}")


# ---------------------------------------------------------------------
# Read prediction records
# ---------------------------------------------------------------------
if not OUTER_PREDICTIONS_CSV.exists():
    raise FileNotFoundError(
        f"Could not find {OUTER_PREDICTIONS_CSV.name} in:\n{HERE}"
    )

if not INNER_PREDICTIONS_CSV.exists():
    raise FileNotFoundError(
        f"Could not find {INNER_PREDICTIONS_CSV.name} in:\n{HERE}"
    )

outer_df = pd.read_csv(OUTER_PREDICTIONS_CSV)
inner_df = pd.read_csv(INNER_PREDICTIONS_CSV)

required_columns = {"row_id", "paper_id", "y_true", "y_pred"}

missing_outer = required_columns - set(outer_df.columns)
missing_inner = required_columns - set(inner_df.columns)

if missing_outer:
    raise KeyError(
        f"{OUTER_PREDICTIONS_CSV.name} is missing columns: "
        f"{sorted(missing_outer)}"
    )

if missing_inner:
    raise KeyError(
        f"{INNER_PREDICTIONS_CSV.name} is missing columns: "
        f"{sorted(missing_inner)}"
    )

print("Outer-fold prediction columns:")
print(list(outer_df.columns))

print("\nInner-OOF prediction columns:")
print(list(inner_df.columns))

if "protocol" in outer_df.columns:
    print("\nOuter-fold protocols:")
    print(sorted(outer_df["protocol"].dropna().astype(str).unique()))

if "protocol" in inner_df.columns:
    print("\nInner-OOF protocols:")
    print(sorted(inner_df["protocol"].dropna().astype(str).unique()))


# ---------------------------------------------------------------------
# 1. Extrapolation plot
#
# outer_fold_predictions.csv contains predictions from the outer
# validation folds, i.e. the final held-out evaluation records.
# ---------------------------------------------------------------------
plot_pred_vs_true(
    df=outer_df,
    output_path=EXTRAPOLATION_PNG,
    title="Random Forest — Predicted vs True (Extrapolation)",
)


# ---------------------------------------------------------------------
# 2. Interpolation plot: raw inner out-of-fold records
#
# Each row is an inner-CV validation prediction. A physical row can
# appear repeatedly because it may occur in multiple repeats, seeds,
# outer folds, or augmentation configurations.
# ---------------------------------------------------------------------
plot_pred_vs_true(
    df=inner_df,
    output_path=INTERPOLATION_RAW_PNG,
    title="Random Forest — Predicted vs True (Interpolation, Raw OOF)",
)


# ---------------------------------------------------------------------
# 3. Interpolation plot: one averaged prediction per physical row
#
# This avoids visually overweighting rows that appear many times in the
# inner-CV prediction records.
# ---------------------------------------------------------------------
group_columns = ["row_id", "paper_id", "y_true"]

inner_averaged_df = (
    inner_df
    .groupby(group_columns, as_index=False)
    .agg(
        y_pred=("y_pred", "mean"),
        n_oof_predictions=("y_pred", "size"),
    )
)

print(
    "\nInterpolation averaging:"
    f"\n  Raw inner-OOF records: {len(inner_df):,}"
    f"\n  Unique physical rows: {len(inner_averaged_df):,}"
    f"\n  Mean predictions per row: "
    f"{inner_averaged_df['n_oof_predictions'].mean():.2f}"
)

plot_pred_vs_true(
    df=inner_averaged_df,
    output_path=INTERPOLATION_AVG_PNG,
    title="Random Forest — Predicted vs True (Interpolation, Averaged OOF)",
)

print("\nDone.")