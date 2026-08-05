import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


# --------------------------------------------------------
# Load and filter
# --------------------------------------------------------
df = pd.read_csv("a.csv")

df = df[
    df["champion"].isin(
        ["interpolation_champion", "extrapolation_champion"]
    )
].copy()

# Youden J duplicates balanced accuracy in this dataset
df = df[df["threshold_policy"] != "youden_j"].copy()


# --------------------------------------------------------
# Labels, colors, and markers
# --------------------------------------------------------
labels = {
    "mcc": "MCC",
    "f1": "F1",
    "balanced_accuracy": "Balanced Accuracy",
}

# More-separated NYU-inspired violet palette
policy_colors = {
    "mcc": "#3D0066",                # Deep violet
    "f1": "#8E24AA",                 # Bright purple
    "balanced_accuracy": "#D6A8FF",  # Light lavender
}

# Shape identifies champion type
champion_markers = {
    "interpolation_champion": "o",   # Circle
    "extrapolation_champion": "^",   # Triangle
}

champion_names = {
    "interpolation_champion": "Interpolation: xgb_paper_sqrt",
    "extrapolation_champion": "Extrapolation: xgb_baseline",
}

# Annotation offsets
# Negative x-offset = label to the left of the point
# Positive x-offset = label to the right of the point
offsets = {
    # MCC: slightly farther to the upper-right
    ("interpolation_champion", "mcc"): (22, 12),
    ("extrapolation_champion", "mcc"): (22, 12),

    # F1: close to the upper-left
    ("interpolation_champion", "f1"): (-14, 12),
    ("extrapolation_champion", "f1"): (-14, 12),

    # Balanced Accuracy: upper-left, matching the F1 placement
    ("interpolation_champion", "balanced_accuracy"): (-14, 12),
    ("extrapolation_champion", "balanced_accuracy"): (-14, 12),
}


# --------------------------------------------------------
# Figure
# --------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(10, 9))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")


# --------------------------------------------------------
# Plot points and annotations
# --------------------------------------------------------
for _, row in df.iterrows():
    x = row["sensitivity"]
    y = row["specificity"]
    champion = row["champion"]
    policy = row["threshold_policy"]

    color = policy_colors[policy]
    marker = champion_markers[champion]

    ax.scatter(
        x,
        y,
        s=240,
        marker=marker,
        color=color,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.95,
        zorder=4,
    )

    annotation_text = (
        f"{labels[policy]}\n"
        f"Threshold = {row['threshold_mean']:.3f}\n"
        f"({x:.2f}, {y:.2f})"
    )

    # F1 and Balanced Accuracy labels go left; MCC labels go right
    label_on_left = policy in ["f1", "balanced_accuracy"]

    ax.annotate(
        annotation_text,
        xy=(x, y),
        xytext=offsets.get((champion, policy), (10, 10)),
        textcoords="offset points",
        ha="right" if label_on_left else "left",
        va="center",
        multialignment="center",
        fontsize=9.5,
        color="#222222",
        linespacing=1.3,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=color,
            alpha=0.95,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=1.0,
            alpha=0.75,
        ),
        zorder=5,
    )


# --------------------------------------------------------
# Axes and cosmetics
# --------------------------------------------------------
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal", adjustable="box")

ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

ax.set_xlabel(
    "Sensitivity (Recall)",
    fontsize=13,
    weight="bold",
    labelpad=10,
)

ax.set_ylabel(
    "Specificity",
    fontsize=13,
    weight="bold",
    labelpad=10,
)

ax.set_title(
    "Threshold Operating Points",
    fontsize=17,
    weight="bold",
    pad=16,
)

ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
ax.set_axisbelow(True)

# Reference diagonal
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    linewidth=1.2,
    alpha=0.55,
    zorder=1,
)

# Cleaner frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# --------------------------------------------------------
# Legends
# --------------------------------------------------------
champion_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label=champion_names["interpolation_champion"],
        markerfacecolor="#555555",
        markeredgecolor="black",
        color="#555555",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        linestyle="None",
        label=champion_names["extrapolation_champion"],
        markerfacecolor="#555555",
        markeredgecolor="black",
        color="#555555",
        markersize=11,
    ),
]

policy_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label="MCC",
        markerfacecolor=policy_colors["mcc"],
        markeredgecolor="black",
        color=policy_colors["mcc"],
        markersize=9,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label="F1",
        markerfacecolor=policy_colors["f1"],
        markeredgecolor="black",
        color=policy_colors["f1"],
        markersize=9,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        label="Balanced Accuracy",
        markerfacecolor=policy_colors["balanced_accuracy"],
        markeredgecolor="black",
        color=policy_colors["balanced_accuracy"],
        markersize=9,
    ),
]

champion_legend = ax.legend(
    handles=champion_handles,
    title="Champions",
    loc="lower left",
    fontsize=10,
    title_fontsize=11,
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    edgecolor="#000000",
)

ax.add_artist(champion_legend)

ax.legend(
    handles=policy_handles,
    title="Threshold Policy",
    loc="upper left",
    fontsize=10,
    title_fontsize=11,
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    edgecolor="#B0B0B0",
)

plt.tight_layout()
plt.show()