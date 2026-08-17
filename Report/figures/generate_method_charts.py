#!/usr/bin/env python3
"""Publication figures for Report/chapters/methodology.tex."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent

NAVY = "#243B6B"
VIOLET = "#57068C"
TEAL = "#1B7A7A"
GOLD = "#B8860B"
SLATE = "#4A5568"
PAPER = "#F7F5F0"
BOX = "#FFFFFF"
TRAIN = "#2E6B4F"
TEST = "#A33B3B"
INNER = "#3D6FA8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 180,
})


def rounded(ax, x, y, w, h, text, fc=BOX, ec=NAVY, tc=NAVY, lw=1.4, fs=8.2, weight="medium", wrap=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        color=tc, fontsize=fs,
        fontweight=("bold" if weight == "bold" else "normal"), wrap=True,
        multialignment="center",
    )
    return patch


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0,
    ))


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", path.name)


def fig_nested_evaluation():
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("Nested evaluation contract (both pipelines)", color=NAVY, fontsize=13, fontweight="bold", pad=8)

    rounded(ax, 0.25, 2.05, 1.9, 1.15, "Frozen outer\nsplit", fc="#E8EEF6", fs=9, weight="bold")
    rounded(ax, 2.55, 3.35, 2.35, 1.25, "Inner CV on\nouter train only\n(hyperparameters)", fc="#DCEAF7", ec=INNER, fs=8.4)
    rounded(ax, 2.55, 0.55, 2.35, 1.25, "Thresholds / recipe\nfrozen from inner\nout-of-fold scores", fc="#F4E8D4", ec=GOLD, fs=8.4)
    rounded(ax, 5.35, 2.05, 2.45, 1.15, "Refit selected config\non full outer train", fc="#E4F0E8", ec=TRAIN, fs=8.6)
    rounded(ax, 8.25, 2.05, 2.0, 1.15, "Score outer\ntest once", fc="#F8E4E4", ec=TEST, fs=8.8, weight="bold")
    rounded(ax, 10.45, 2.05, 1.35, 1.15, "Repeat\nouter folds", fc="#EEE8F5", ec=VIOLET, fs=8.6)

    arrow(ax, 2.15, 2.62, 2.55, 3.85)
    arrow(ax, 2.15, 2.62, 2.55, 1.18)
    arrow(ax, 4.90, 3.85, 5.35, 2.85)
    arrow(ax, 4.90, 1.18, 5.35, 2.40)
    arrow(ax, 7.80, 2.62, 8.25, 2.62)
    arrow(ax, 10.25, 2.62, 10.45, 2.62)

    ax.text(3.72, 4.75, "never sees outer test", color=INNER, fontsize=8, ha="center", style="italic")
    ax.text(9.25, 1.70, "honest generalization score", color=TEST, fontsize=8, ha="center", style="italic")
    ax.text(6.0, 0.18, "Imputation, scaling, one-hot vocabularies, weights, and thresholds are fit on training partitions only.",
            color=SLATE, fontsize=8, ha="center")
    save(fig, "method_nested_evaluation.png")


def fig_interp_extrap():
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35))

    papers = [
        ((1.2, 2.6), 8, 0.55),
        ((2.6, 1.3), 6, 0.42),
        ((3.9, 2.8), 10, 0.62),
        ((5.2, 1.5), 5, 0.38),
        ((1.0, 0.7), 4, 0.35),
    ]

    def draw_panel(ax, title, hold_paper=False):
        ax.set_xlim(0.1, 6.3)
        ax.set_ylim(0.1, 3.9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#D0D4DC")
        ax.set_title(title, color=NAVY, fontsize=11, fontweight="bold", pad=8)
        ax.set_facecolor(PAPER)
        local = np.random.default_rng(7)
        for i, ((cx, cy), n, spread) in enumerate(papers):
            xs = cx + local.normal(0, spread, n)
            ys = cy + local.normal(0, spread * 0.55, n)
            if hold_paper:
                is_test = i == 2
                color = TEST if is_test else TRAIN
                marker = "s" if is_test else "o"
                ax.scatter(xs, ys, c=color, s=38, marker=marker, alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
                if is_test:
                    ax.add_patch(Circle((cx, cy), 0.95, fill=False, ls="--", ec=TEST, lw=1.3))
                    ax.text(cx, cy + 1.05, "held-out paper", color=TEST, fontsize=8, ha="center")
            else:
                test_mask = local.random(n) < 0.28
                ax.scatter(xs[~test_mask], ys[~test_mask], c=TRAIN, s=38, marker="o", alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
                ax.scatter(xs[test_mask], ys[test_mask], c=TEST, s=38, marker="s", alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)
            ax.text(cx, cy - 0.62 if i != 4 else cy + 0.55, f"P{i+1}", color=SLATE, fontsize=7.5, ha="center")

        legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=TRAIN, markersize=8, label="Train rows"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TEST, markersize=8, label="Test rows"),
        ]
        ax.legend(handles=legend, loc="upper right", frameon=True, fontsize=8, edgecolor="#E2E6EE")

    draw_panel(axes[0], "Interpolation: rows held out\n(same papers may appear on both sides)")
    draw_panel(axes[1], "Extrapolation: entire papers held out\n(zero paper overlap; primary scientific test)")
    fig.suptitle("Two evaluation questions, not one score", color=NAVY, fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "method_interp_vs_extrap.png")


def fig_jose_mapping():
    fig, ax = plt.subplots(figsize=(10.8, 5.15))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Model-family mapping from Rivera/Jose to this report", color=NAVY, fontsize=13, fontweight="bold", pad=6)

    rounded(ax, 0.25, 5.15, 11.5, 0.75,
            "Rivera et al. (Jose): ~1,200 wire FSR rows  ·  random 10-fold CV  ·  numerical + categorical features  ·  bootstrap / noise / GAN augmentation",
            fc="#EEE8F5", ec=VIOLET, fs=8.3)

    headers = [
        (0.25, "Jose family"),
        (3.55, "This report"),
        (6.85, "Why retained / changed"),
        (9.85, "Where used"),
    ]
    for x, t in headers:
        ax.text(x, 4.85, t, color=SLATE, fontsize=8, fontweight="bold")

    rows = [
        ("Gradient boosting", "XGBoost (primary)", "Strongest tabular prior (Shwartz-Ziv)", "Ignition + FSR"),
        ("Decision tree", "Decision tree", "Transparent nonlinear baseline from Jose", "Ignition"),
        ("—", "Random forest", "Bagging counterpart to boosting; robustness under source shift", "FSR"),
        ("k-NN", "k-NN", "Local similarity in experimental condition space", "Ignition + FSR"),
        ("MLP", "MLP", "Flexible function approximation; test if depth helps tables", "Ignition + FSR"),
        ("SVR", "SVM / SVR", "Kernel baseline; linear vs RBF tests feature linearity", "Ignition + FSR"),
        ("Linear / elastic net /\nSGD / Bayesian / kernel ridge", "Not retained", "Jose: linear families lagged nonlinear models on FSR", "—"),
    ]
    y = 4.35
    for i, (a, b, c, d) in enumerate(rows):
        fc = "#F3F6FB" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(Rectangle((0.2, y - 0.42), 11.6, 0.52, facecolor=fc, edgecolor="none"))
        ax.text(0.30, y - 0.16, a, color=NAVY, fontsize=8, va="center")
        ax.text(3.55, y - 0.16, b, color=TEAL if b != "Not retained" else TEST, fontsize=8, va="center", fontweight="bold")
        ax.text(6.85, y - 0.16, c, color=SLATE, fontsize=7.6, va="center")
        ax.text(9.85, y - 0.16, d, color=NAVY, fontsize=8, va="center")
        y -= 0.54

    ax.text(6.0, 0.18, "Beyond Jose: nested interpolation vs paper-grouped extrapolation, paper/class weights, focal loss, signed-log targets, and real-data vs bootstrap conditions.",
            color=SLATE, fontsize=8, ha="center")
    save(fig, "method_jose_model_mapping.png")


def _family_panel(ax, title, families, note):
    ax.set_xlim(0, 10)
    ymax = 0.55 + 0.72 * len(families)
    ax.set_ylim(0, ymax + 0.35)
    ax.axis("off")
    ax.set_title(title, color=NAVY, fontsize=12, fontweight="bold", loc="left")
    y = ymax - 0.15
    palette = {
        "XGBoost": "#3D6FA8",
        "Decision tree": "#6B5B95",
        "Random forest": "#2E6B4F",
        "k-NN": "#C47B2B",
        "MLP": "#8B3A4A",
        "SVM": "#1B7A7A",
        "SVR": "#1B7A7A",
    }
    for fam, n, hyp in families:
        color = palette[fam]
        ax.add_patch(FancyBboxPatch((0.1, y - 0.55), 1.85, 0.52, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor=color, edgecolor=color, linewidth=0.6))
        ax.text(1.02, y - 0.29, f"{fam}\n{n}", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.add_patch(FancyBboxPatch((2.1, y - 0.55), 7.7, 0.52, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor="#F4F7FB", edgecolor=color, linewidth=0.9))
        ax.text(2.25, y - 0.29, hyp, color=NAVY, fontsize=8, va="center")
        y -= 0.68
    ax.text(0.1, 0.08, note, color=SLATE, fontsize=7.6)


def fig_ignition_candidates():
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    families = [
        ("XGBoost", "8 candidates", "2×3 ablation of paper weight × class weight, plus focal loss without class weight (no double correction)."),
        ("Decision tree", "3 candidates", "Baseline vs combined class+paper weights. Depth, leaf size, and split criterion already tuned."),
        ("k-NN", "5 candidates", "Uniform neighbours vs distance weighting; each with sqrt/inverse paper weights."),
        ("MLP", "5 candidates", "Baseline, combined weights, and focal loss (γ=2) without class weight."),
        ("SVM", "5 candidates", "RBF default vs linear kernel; linear tests whether ignition is linearly separable in physics space."),
    ]
    _family_panel(ax, "Ignition portfolio — 26 candidates across five families", families,
                  "Primary model is XGBoost because it receives the full imbalance ablation. Remaining families are structured comparison baselines aligned with Jose.")
    save(fig, "method_ignition_candidates.png")


def fig_fsr_candidates():
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    families = [
        ("XGBoost", "10 candidates", "Paper weights; squared / absolute / pseudo-Huber losses; signed-log target; paper-level bagging."),
        ("Random forest", "7 candidates", "Bagged-tree counterpart: baseline, paper weights, absolute-error criterion, signed-log target."),
        ("k-NN", "6 candidates", "Uniform vs distance neighbours × none/sqrt/inverse paper weighting."),
        ("MLP", "5 candidates", "Baseline and paper weights; two signed-log models because FSR spans orders of magnitude."),
        ("SVR", "6 candidates", "Linear vs RBF kernels × none/sqrt/inverse paper weighting."),
    ]
    _family_panel(ax, "FSR portfolio — 34 candidates across five families", families,
                  "Each candidate tests one hypothesis (weighting, loss, target scale, locality, kernel). Jose’s linear families are omitted; random forest is added as the bagging control.")
    save(fig, "method_fsr_candidates.png")


def fig_ignition_pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 6.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    ax.set_title("Ignition classification pipeline", color=NAVY, fontsize=13, fontweight="bold")

    stages = [
        (0.3, 5.55, "1. Load & engineer\n32 numeric + 5 categorical\nphysics features", "#E8EEF6"),
        (3.2, 5.55, "2. Freeze splits\nstratified 5×3\ngrouped 5×3 + LOPO", "#EEE8F5"),
        (6.1, 5.55, "3. Nested search\ninner ROC–AUC rank\n40 configs / fold", "#DCEAF7"),
        (9.0, 5.55, "4. Outer test\nfour frozen τ\nMCC / F1 / BA / Youden", "#F8E4E4"),
        (1.75, 3.35, "5. Select champions\nROC–AUC, SD ≤ 0.15\nLOPO required for extrapolation", "#F4E8D4"),
        (6.55, 3.35, "6. Refit artifacts\nfull labeled cohort\ninference only, not evidence", "#E4F0E8"),
    ]
    for x, y, t, fc in stages:
        rounded(ax, x, y, 2.7, 1.45, t, fc=fc, fs=8.3, weight="medium")

    arrow(ax, 3.0, 6.28, 3.2, 6.28)
    arrow(ax, 5.9, 6.28, 6.1, 6.28)
    arrow(ax, 8.8, 6.28, 9.0, 6.28)
    arrow(ax, 10.35, 5.55, 8.0, 4.80)
    arrow(ax, 4.45, 3.35, 4.45, 5.55)
    arrow(ax, 4.45, 4.08, 6.55, 4.08)

    # protocol boxes
    rounded(ax, 0.3, 0.35, 3.6, 2.45,
            "Interpolation\nStratifiedKFold, 15 outer tests\nPapers may appear on both sides\nPrimary metric: ROC-AUC",
            fc="#E8F3EC", ec=TRAIN, fs=8.4)
    rounded(ax, 4.2, 0.35, 3.6, 2.45,
            "Extrapolation\nStratifiedGroupKFold, 15 tests\nZero paper overlap\nPrimary metric: ROC-AUC",
            fc="#FBECEC", ec=TEST, fs=8.4)
    rounded(ax, 8.1, 0.35, 3.6, 2.45,
            "LOPO robustness\nEach of 87 papers held out once\nRequired for extrapolation eligibility\nPooled + defined-paper metrics",
            fc="#F4EFE4", ec=GOLD, fs=8.4)
    save(fig, "method_ignition_pipeline.png")


def fig_fsr_pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 6.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    ax.set_title("Flame-spread-rate regression pipeline", color=NAVY, fontsize=13, fontweight="bold")

    stages = [
        (0.3, 5.55, "1. Load & detect\n29 numeric + 4 categorical\nphysics fields (fsr-data-v2)", "#E8EEF6"),
        (3.2, 5.55, "2. Freeze holdouts\n10× 80/20 repeats\nrows vs papers", "#EEE8F5"),
        (6.1, 5.55, "3. Nested search\ninner RMSE rank\nthen MAE, then R²", "#DCEAF7"),
        (9.0, 5.55, "4. Dual fit\nrd and rd_bt\ntest never augmented", "#F8E4E4"),
        (1.75, 3.35, "5. Aggregate + select\nRMSE, CV ≤ 0.45, R² > −0.25\n2% RMSE tie band", "#F4E8D4"),
        (6.55, 3.35, "6. Refit artifacts\nwinning (candidate, rd/rd_bt)\nfull 2,531 FSR rows", "#E4F0E8"),
    ]
    for x, y, t, fc in stages:
        rounded(ax, x, y, 2.7, 1.45, t, fc=fc, fs=8.3, weight="medium")

    arrow(ax, 3.0, 6.28, 3.2, 6.28)
    arrow(ax, 5.9, 6.28, 6.1, 6.28)
    arrow(ax, 8.8, 6.28, 9.0, 6.28)
    arrow(ax, 10.35, 5.55, 8.0, 4.80)
    arrow(ax, 4.45, 3.35, 4.45, 5.55)
    arrow(ax, 4.45, 4.08, 6.55, 4.08)

    rounded(ax, 0.3, 0.35, 3.6, 2.45,
            "Interpolation\nQuantile-stratified row holdout\n2,024 train / 507 test rows\nPrimary metric: RMSE",
            fc="#E8F3EC", ec=TRAIN, fs=8.4)
    rounded(ax, 4.2, 0.35, 3.6, 2.45,
            "Extrapolation\nPaper GroupShuffleSplit\n52 / 14 papers; target-balanced\nPrimary metric: RMSE",
            fc="#FBECEC", ec=TEST, fs=8.4)
    rounded(ax, 8.1, 0.35, 3.6, 2.45,
            "Bootstrap condition rd_bt\n+1,000 resampled train rows\nCopies inherit paper IDs\nNot new experiments",
            fc="#F4EFE4", ec=GOLD, fs=8.4)
    save(fig, "method_fsr_pipeline.png")


def fig_weighting():
    fig, ax = plt.subplots(figsize=(10.6, 3.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    ax.set_title("Paper-weighting strategies used by both pipelines", color=NAVY, fontsize=13, fontweight="bold")
    items = [
        ("none", "Every row equal", "Standard empirical risk.\nLarge campaigns dominate."),
        ("sqrt", r"$w_p = 1/\sqrt{n_p}$", "Moderate correction.\nDefault scientific compromise."),
        ("inverse", r"$w_p = 1/n_p$", "Each paper ≈ equal influence.\nAggressive source equalization."),
    ]
    colors = ["#4A5568", TEAL, VIOLET]
    for i, ((name, formula, blurb), color) in enumerate(zip(items, colors)):
        x = 0.35 + i * 3.9
        rounded(ax, x, 0.35, 3.55, 2.55, "", fc="#F7F8FB", ec=color, lw=1.6)
        ax.text(x + 1.78, 2.45, name, color=color, ha="center", fontsize=12, fontweight="bold")
        ax.text(x + 1.78, 1.85, formula, color=NAVY, ha="center", fontsize=10)
        ax.text(x + 1.78, 1.05, blurb, color=SLATE, ha="center", fontsize=8.3)
    save(fig, "method_paper_weights.png")


if __name__ == "__main__":
    fig_nested_evaluation()
    fig_interp_extrap()
    fig_jose_mapping()
    fig_ignition_candidates()
    fig_fsr_candidates()
    fig_ignition_pipeline()
    fig_fsr_pipeline()
    fig_weighting()
    print("done")
