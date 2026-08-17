#!/usr/bin/env python3
"""Generate the vector-first figures used in the Methodology chapter.

Each figure is exported as PDF (the format included by LaTeX) and as a
high-resolution PNG fallback.  All geometry is defined in physical inches so
that labels remain at least 7 pt at the final report width.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent

# Colour-blind-safe palette; meaning is reinforced by labels and line styles.
NAVY = "#243B6B"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#6F4E9C"
SLATE = "#4B5563"
LIGHT = "#F3F5F7"
GRID = "#D6DAE1"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.4,
        "axes.edgecolor": SLATE,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    """Save publication PDF and 600-dpi review raster."""
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "generate_method_charts.py"},
    )
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    print(f"saved {stem}.pdf and {stem}.png")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=NAVY,
        fontsize=9,
        fontweight="bold",
    )


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    face: str = WHITE,
    edge: str = NAVY,
    fontsize: float = 7.5,
    weight: str = "normal",
    radius: float = 0.05,
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SLATE,
    style: str = "-|>",
    linewidth: float = 1.0,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=linewidth,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=connectionstyle,
        )
    )


def fig_jose_mapping() -> None:
    """Map prior model families to the present benchmark without dense prose."""
    fig, ax = plt.subplots(figsize=(7.25, 3.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [0.025, 0.27, 0.51, 0.82]
    widths = [0.235, 0.23, 0.30, 0.155]
    headers = ["Rivera/Jose family", "Present benchmark", "Scientific role", "Task"]
    for x, width, title in zip(columns, widths, headers):
        ax.add_patch(Rectangle((x, 0.88), width, 0.085, facecolor=NAVY, edgecolor=WHITE, linewidth=0.7))
        ax.text(x + 0.01, 0.922, title, color=WHITE, ha="left", va="center", fontsize=7.3, fontweight="bold")

    rows = [
        ("Gradient boosting", "XGBoost", "Strong tabular baseline; boosting", "Both", BLUE),
        ("Decision tree", "Decision tree", "Interpretable nonlinear baseline", "Ignition", PURPLE),
        ("—", "Random forest", "Bagging control added in this work", "FSR", GREEN),
        ("$k$-nearest neighbours", "$k$-NN", "Local similarity in condition space", "Both", ORANGE),
        ("Multilayer perceptron", "MLP", "Smooth nonlinear approximation", "Both", VERMILION),
        ("Support-vector regression", "SVM / SVR", "Linear versus radial kernels", "Both", SKY),
        ("Linear and shrinkage models", "Not retained", "Weak prior on heterogeneous FSR table", "—", SLATE),
    ]
    top = 0.855
    row_h = 0.105
    for idx, (prior, current, role, task, colour) in enumerate(rows):
        y = top - (idx + 1) * row_h
        fill = "#F7F8FA" if idx % 2 == 0 else WHITE
        ax.add_patch(Rectangle((0.025, y), 0.95, row_h, facecolor=fill, edgecolor=GRID, linewidth=0.35))
        ax.add_patch(Rectangle((0.025, y), 0.008, row_h, facecolor=colour, edgecolor="none"))
        texts = [prior, current, role, task]
        for x, text in zip(columns, texts):
            ax.text(
                x + 0.012,
                y + row_h / 2,
                text,
                ha="left",
                va="center",
                color=NAVY if text != "Not retained" else VERMILION,
                fontsize=7.25,
                fontweight="bold" if x == columns[1] else "normal",
            )

    ax.text(
        0.025,
        0.012,
        "Protocol extension: nested selection; paper-disjoint testing; source/class weighting; robust losses and target transforms.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=SLATE,
    )
    save(fig, "method_jose_model_mapping")


def fig_nested_evaluation() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Outer loop
    ax.add_patch(
        FancyBboxPatch(
            (0.18, 0.28),
            9.62,
            4.35,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#FAFBFC",
            edgecolor=NAVY,
            linewidth=1.2,
        )
    )
    ax.text(0.4, 4.42, "OUTER LOOP — unbiased procedure assessment", color=NAVY, fontsize=8, fontweight="bold")

    box(ax, (0.55, 2.00), (1.55, 1.12), "Frozen outer split", face="#E7EEF7", edge=BLUE, weight="bold")
    box(ax, (2.55, 2.00), (1.70, 1.12), "Outer training\npartition", face="#E7F4EF", edge=GREEN)
    box(ax, (6.00, 2.00), (1.70, 1.12), "Refit selected\nprocedure", face="#E7F4EF", edge=GREEN)
    box(ax, (8.05, 2.00), (1.38, 1.12), "Score outer\ntest once", face="#FCEBE6", edge=VERMILION, weight="bold")
    arrow(ax, (2.10, 2.56), (2.55, 2.56))
    arrow(ax, (4.25, 2.56), (4.67, 2.56))
    arrow(ax, (5.60, 2.56), (6.00, 2.56))
    arrow(ax, (7.70, 2.56), (8.05, 2.56))

    # Inner loop inset
    ax.add_patch(
        FancyBboxPatch(
            (4.65, 1.10),
            0.98,
            2.92,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            facecolor="#EEF5FA",
            edgecolor=BLUE,
            linewidth=1.0,
        )
    )
    ax.text(5.14, 3.72, "INNER CV", ha="center", color=BLUE, fontsize=7.4, fontweight="bold")
    for i in range(3):
        y = 3.18 - i * 0.55
        ax.add_patch(Rectangle((4.83, y), 0.42, 0.20, facecolor=GREEN, edgecolor="none"))
        ax.add_patch(Rectangle((5.25, y), 0.20, 0.20, facecolor=ORANGE, edgecolor="none"))
    ax.text(5.14, 1.55, "Tune + freeze\nrecipe / threshold", ha="center", va="center", color=NAVY, fontsize=6.8)

    # Outer test branch is visually isolated.
    arrow(ax, (1.33, 2.00), (8.74, 1.15), color=VERMILION, connectionstyle="arc3,rad=0.16")
    ax.text(7.3, 0.68, "outer test remains untouched", color=VERMILION, fontsize=7.1, fontstyle="italic")

    ax.text(
        0.5,
        0.48,
        "Fold-local operations: imputation · scaling · category vocabulary · weighting · hyperparameter and threshold selection",
        color=SLATE,
        fontsize=7.1,
    )
    save(fig, "method_nested_evaluation")


def fig_interp_extrap() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.18), sharex=True, sharey=True)
    rng = np.random.default_rng(17)
    centres = np.array([[0.9, 2.4], [2.2, 1.0], [3.5, 2.3], [4.7, 1.1], [1.0, 0.55]])
    counts = [8, 7, 10, 6, 4]
    colours = [BLUE, PURPLE, GREEN, ORANGE, SKY]

    for panel, ax in enumerate(axes):
        ax.set_xlim(0.15, 5.45)
        ax.set_ylim(0.05, 3.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#FAFBFC")
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.8)

        for p, (centre, count, paper_colour) in enumerate(zip(centres, counts, colours)):
            points = centre + rng.normal(size=(count, 2)) * np.array([0.30, 0.19])
            if panel == 0:
                test = np.arange(count) % 3 == 0
            else:
                test = np.full(count, p == 2)
            ax.scatter(
                points[~test, 0],
                points[~test, 1],
                s=24,
                marker="o",
                facecolor=paper_colour,
                edgecolor=WHITE,
                linewidth=0.45,
                zorder=3,
            )
            ax.scatter(
                points[test, 0],
                points[test, 1],
                s=26,
                marker="s",
                facecolor=WHITE,
                edgecolor=VERMILION,
                linewidth=1.2,
                zorder=4,
            )
            ax.text(centre[0], centre[1] + 0.38, f"Paper {p + 1}", ha="center", color=SLATE, fontsize=6.5)

    axes[0].set_title("Row-level interpolation", color=NAVY, fontweight="bold", pad=7)
    axes[1].set_title("Paper-level extrapolation", color=NAVY, fontweight="bold", pad=7)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    axes[0].text(0.5, -0.12, "Paper overlap permitted", transform=axes[0].transAxes, ha="center", color=SLATE, fontsize=7.3)
    axes[1].text(0.5, -0.12, "Zero paper overlap", transform=axes[1].transAxes, ha="center", color=VERMILION, fontsize=7.3, fontweight="bold")

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor=WHITE, markersize=6, label="Training row"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=WHITE, markeredgecolor=VERMILION, markersize=6, label="Test row"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.23, wspace=0.08)
    save(fig, "method_interp_vs_extrap")


def fig_weighting() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), gridspec_kw={"width_ratios": [1.35, 1]})
    n = np.logspace(0, 3, 250)
    curves = [
        ("None", n, SLATE, "-"),
        (r"Square root, $w_p=n_p^{-1/2}$", np.sqrt(n), BLUE, "--"),
        (r"Inverse, $w_p=n_p^{-1}$", np.ones_like(n), VERMILION, ":"),
    ]
    ax = axes[0]
    for label, influence, colour, style in curves:
        ax.plot(n, influence, label=label, color=colour, linestyle=style, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Rows contributed by paper, $n_p$")
    ax.set_ylabel(r"Total paper influence, $n_p w_p$")
    ax.grid(True, which="both", color=GRID, linewidth=0.45)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b")
    ax.text(0.02, 0.93, "Interpretation", color=NAVY, fontsize=8.5, fontweight="bold", va="top")
    notes = [
        (0.72, SLATE, "None", "Dense campaigns dominate"),
        (0.46, BLUE, "Square root", "Moderate source correction"),
        (0.20, VERMILION, "Inverse", "Approximately equal paper totals"),
    ]
    for y, colour, title, note in notes:
        ax.add_patch(Rectangle((0.02, y - 0.06), 0.035, 0.15, facecolor=colour, edgecolor="none"))
        ax.text(0.09, y + 0.04, title, color=NAVY, fontweight="bold", fontsize=7.7, va="center")
        ax.text(0.09, y - 0.035, note, color=SLATE, fontsize=7.1, va="center")
    ax.text(0.02, 0.02, "Strategy is selected inside the\nnested training procedure.", color=NAVY, fontsize=7.2, fontstyle="italic")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.96, wspace=0.30)
    save(fig, "method_paper_weights")


def candidate_figure(stem: str, total: int, families: list[tuple[str, int, str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 3.28))
    names = [f[0] for f in families][::-1]
    counts = np.array([f[1] for f in families][::-1])
    colours = [f[3] for f in families][::-1]
    y = np.arange(len(families))
    ax.barh(y, counts, color=colours, height=0.56, edgecolor=WHITE, linewidth=0.6)
    ax.set_yticks(y, names, color=NAVY, fontweight="bold")
    ax.set_xlim(0, max(counts) * 2.95)
    ax.set_xlabel("Number of pre-specified candidates")
    ax.set_xticks(range(0, int(max(counts)) + 1, 2))
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, (count, family) in enumerate(zip(counts, families[::-1])):
        _, _, hypothesis, _ = family
        ax.text(count - 0.18, i, str(count), ha="right", va="center", color=WHITE, fontsize=7.4, fontweight="bold")
        ax.text(count + 0.28, i, hypothesis, ha="left", va="center", color=SLATE, fontsize=7.05)
    ax.text(0.0, 1.04, f"{total} candidates · five model families", transform=ax.transAxes, color=NAVY, fontsize=8.7, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.18, top=0.90)
    save(fig, stem)


def fig_ignition_candidates() -> None:
    candidate_figure(
        "method_ignition_candidates",
        26,
        [
            ("XGBoost", 8, "paper × class weighting; focal loss", BLUE),
            ("Decision tree", 3, "baseline; combined weighting", PURPLE),
            ("$k$-NN", 5, "uniform vs distance neighbours", ORANGE),
            ("MLP", 5, "combined weighting; focal loss", VERMILION),
            ("SVM", 5, "radial vs linear kernel", GREEN),
        ],
    )


def fig_fsr_candidates() -> None:
    candidate_figure(
        "method_fsr_candidates",
        34,
        [
            ("XGBoost", 10, "loss; target scale; weights; paper bagging", BLUE),
            ("Random forest", 7, "bagging; robust split; target scale", GREEN),
            ("$k$-NN", 6, "uniform vs distance neighbours", ORANGE),
            ("MLP", 5, "weights; signed-log target", VERMILION),
            ("SVR", 6, "radial vs linear kernel", PURPLE),
        ],
    )


def pipeline_figure(
    stem: str,
    task_label: str,
    feature_label: str,
    tune_label: str,
    score_label: str,
    protocol_rows: list[tuple[str, str, str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 4.05))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    stages = [
        (0.2, "Curate", feature_label, "#E7EEF7", BLUE),
        (2.65, "Freeze", "outer assignments\nand integrity checks", "#F0ECF7", PURPLE),
        (5.10, "Tune", tune_label, "#E8F3F8", SKY),
        (7.55, "Evaluate", score_label, "#FCEBE6", VERMILION),
        (10.00, "Select", "aggregate evidence;\nrefit champion", "#E7F4EF", GREEN),
    ]
    for x, heading, detail, face, edge in stages:
        box(ax, (x, 4.95), (1.95, 1.42), f"{heading}\n{detail}", face=face, edge=edge, fontsize=7.15, weight="bold")
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(ax, (left[0] + 1.95, 5.66), (right[0], 5.66))

    ax.text(0.2, 6.72, task_label, color=NAVY, fontsize=9.3, fontweight="bold")
    ax.text(0.2, 4.55, "FROZEN OUTER PROTOCOLS", color=NAVY, fontsize=7.4, fontweight="bold")

    y_positions = [3.38, 2.08, 0.78]
    for y, (name, assignment, test, note) in zip(y_positions, protocol_rows):
        ax.add_patch(
            FancyBboxPatch(
                (0.2, y),
                11.75,
                0.98,
                boxstyle="round,pad=0.012,rounding_size=0.04",
                facecolor="#FAFBFC",
                edgecolor=GRID,
                linewidth=0.8,
            )
        )
        ax.add_patch(Rectangle((0.2, y), 0.08, 0.98, facecolor=GREEN if "Interpolation" in name else VERMILION if "Extrapolation" in name else ORANGE, edgecolor="none"))
        ax.text(0.48, y + 0.64, name, color=NAVY, fontsize=7.5, fontweight="bold", va="center")
        ax.text(0.48, y + 0.30, assignment, color=SLATE, fontsize=6.8, va="center")
        ax.text(4.45, y + 0.50, test, color=NAVY, fontsize=7.0, va="center")
        ax.text(8.25, y + 0.50, note, color=SLATE, fontsize=6.8, va="center")
        ax.plot([4.15, 4.15], [y + 0.16, y + 0.82], color=GRID, lw=0.7)
        ax.plot([7.95, 7.95], [y + 0.16, y + 0.82], color=GRID, lw=0.7)

    ax.text(4.45, 4.48, "EVALUATION UNIT", color=SLATE, fontsize=6.4, fontweight="bold")
    ax.text(8.25, 4.48, "ROLE", color=SLATE, fontsize=6.4, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.98)
    save(fig, stem)


def fig_ignition_pipeline() -> None:
    pipeline_figure(
        "method_ignition_pipeline",
        "Ignition classification — executable evaluation design",
        "32 numeric + 5\ncategorical features",
        "inner ROC–AUC;\nfreeze 4 thresholds",
        "outer ROC/PR–AUC,\nBrier + decisions",
        [
            ("Interpolation", "stratified 5-fold × 3 seeds", "held-out rows", "within-campaign prediction"),
            ("Extrapolation", "paper-grouped 5-fold × 3 seeds", "held-out papers", "primary transfer evidence"),
            ("LOPO robustness", "each of 87 papers held out once", "one complete paper", "eligibility gate; pooled metrics"),
        ],
    )


def fig_fsr_pipeline() -> None:
    pipeline_figure(
        "method_fsr_pipeline",
        "Flame-spread-rate regression — executable evaluation design",
        "29 numeric + 4\ncategorical fields",
        "inner RMSE;\nthen MAE and $R^2$",
        "outer RMSE, MAE,\n$R^2$ and bias",
        [
            ("Interpolation", "10 quantile-balanced 80/20 splits", "held-out rows", "2,024 train / 507 test"),
            ("Extrapolation", "10 balance-screened paper splits", "14 held-out papers", "52 training papers; primary evidence"),
            ("Training ablation", "real data (rd) vs rd + 1,000 resamples", "same untouched test", "copies retain source-paper identity"),
        ],
    )


if __name__ == "__main__":
    fig_jose_mapping()
    fig_nested_evaluation()
    fig_interp_extrap()
    fig_weighting()
    fig_ignition_candidates()
    fig_ignition_pipeline()
    fig_fsr_candidates()
    fig_fsr_pipeline()
    print("done")
