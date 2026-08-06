from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = Path("permutation_importance.csv")
OUTPUT_PNG = Path("permutation_importance_random_forest.png")
TOP_N = 10

PURPLE = "#57068C"
DARK = "#1F2937"
GRAY = "#A7A9AC"

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.95,
    rc={
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)

# ============================================================
# Load importance table
# ============================================================
df = pd.read_csv(INPUT_CSV)

print("Available columns:")
print(df.columns.tolist())

def find_column(candidates, available_columns):
    lower_to_original = {col.lower(): col for col in available_columns}

    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates}.\n"
        f"Available columns are: {available_columns.tolist()}"
    )

feature_col = find_column(
    ["feature", "feature_name", "variable", "predictor"],
    df.columns,
)

importance_mean_col = find_column(
    [
        "importance_mean",
        "mean_importance",
        "permutation_importance_mean",
        "importance",
        "mean",
    ],
    df.columns,
)

try:
    importance_std_col = find_column(
        [
            "importance_std",
            "std_importance",
            "permutation_importance_std",
            "std",
            "importance_sd",
        ],
        df.columns,
    )
except ValueError:
    importance_std_col = None
    print("\nNo importance standard-deviation column found.")
    print("Plotting mean permutation importance without error bars.")

# ============================================================
# Prepare top features
# ============================================================
plot_df = df[[feature_col, importance_mean_col]].copy()

if importance_std_col is not None:
    plot_df["importance_std"] = df[importance_std_col]
else:
    plot_df["importance_std"] = 0.0

plot_df.columns = ["feature", "importance_mean", "importance_std"]

plot_df = (
    plot_df.dropna(subset=["feature", "importance_mean"])
    .nlargest(TOP_N, "importance_mean")
    .sort_values("importance_mean", ascending=True)
    .reset_index(drop=True)
)

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(13.33, 7.5))

colors = [
    PURPLE if value > 0 else GRAY
    for value in plot_df["importance_mean"]
]

ax.barh(
    plot_df["feature"],
    plot_df["importance_mean"],
    xerr=plot_df["importance_std"],
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    error_kw={
        "ecolor": DARK,
        "elinewidth": 1.3,
        "capsize": 3.5,
    },
)

ax.axvline(
    0,
    color=DARK,
    linestyle="--",
    linewidth=1.2,
)

for y, (_, row) in enumerate(plot_df.iterrows()):
    direction = 1 if row["importance_mean"] >= 0 else -1


x_min = min(
    (plot_df["importance_mean"] - plot_df["importance_std"]).min(),
    0,
)
x_max = max(
    (plot_df["importance_mean"] + plot_df["importance_std"]).max(),
    0,
)
x_span = max(x_max - x_min, 1e-6)

ax.set_xlim(
    x_min - 0.12 * x_span,
    x_max + 0.35 * x_span,
)

ax.set_xlabel(
    "Permutation importance (RMSE mean ± std)",
)
ax.set_ylabel("")
ax.set_title(
    "rf_baseline: Top 10 features by permutation importance",
    loc="left",
    weight="bold",
    pad=15,
)

plt.tight_layout()
fig.savefig(OUTPUT_PNG, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure: {OUTPUT_PNG.resolve()}")