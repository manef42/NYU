#!/usr/bin/env python3
"""
generate_metrics_merged.py
===========================
Merged metrics generator for the Microgravity Flammability Database.

Combines the architectural improvements from the upgraded schema script
(folder organisation, CLI, PlotWriter class, dimensionless numbers,
ignition probability heatmap, material-level FSR, co-availability matrix,
source diversity / ESS) with plots that existed only in the original script
(range + gap strip helper, per-paper O2/pressure span range-plots,
gravity-regime bars, fuel property space scatter, paper-size vs ignition-
fraction bubble, diluent-coloured O2 vs pressure scatter).

Folder layout written to --output (default: same directory as this file):
    01_overview/          - dataset cards, paper contributions, completeness
    02_parameter_coverage/ - histograms, hexbin, dimensionless coverage, gaps
    03_ignition_metrics/  - outcome balance, rates by group, probability map
    04_fsr_metrics/       - FSR distributions, scatter, boxplots, associations
    05_data_quality/      - co-availability, correlation, duplication, ESS

Usage:
    python generate_metrics_merged.py
    python generate_metrics_merged.py --input path/to/db.csv --clean
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "Microgravity_Database.csv"

OUTPUT_FOLDERS = {
    "overview":  "01_overview",
    "coverage":  "02_parameter_coverage",
    "ignition":  "03_ignition_metrics",
    "fsr":       "04_fsr_metrics",
    "quality":   "05_data_quality",
}

# ── Column map (upgraded schema) ──────────────────────────────────────────────
COL = {
    "paper":            "Article (MLA)",
    "authors":          "Authors",
    "doi":              "DOI",
    "geometry":         "Geometry of Sample",
    "material":         "Material of sample",
    "half_thickness":   "half_thickness",
    "length":           "characteristic_length_m",
    "fuel_density":     "fuel_density_kg_m3",
    "fuel_k":           "fuel_k_W_mK",
    "fuel_cp":          "fuel_cp_J_kgK",
    "fuel_pyrolysis_t": "fuel_pyrolysis_T_K",
    "fuel_alpha":       "fuel_alpha_m2_s",
    "o2":               "Oxygen Concentration",
    "diluent":          "diluent",
    "gas_density":      "gas_density_kg_m3",
    "reynolds":         "Reynolds",
    "peclet":           "Peclet",
    "prandtl":          "Prandtl",
    "diffusion_time":   "thermal_diffusion_time",
    "pressure":         "Pressure (Pa)",
    "flow":             "Flow Velocity (m/s)",
    "rig":              "Rig Name",
    "internal_geometry":"Internal Geometry",
    "gravity":          "Gravity",
    "facility":         "Facility",
    "ignition_method":  "Ignition Method",
    "ignition_power":   "Ignition Power (W)",
    "ignition_time":    "Ignition Time (s)",
    "ignition":         "Ignition",
    "flame_length":     "Flame Length",
    "fsr":              "FSR (m/s)",
    "hrr":              "HRR",
    "smoke":            "Smoke/ Areosols",
}

NUMERIC_KEYS = [
    "half_thickness", "length", "fuel_density", "fuel_k", "fuel_cp",
    "fuel_pyrolysis_t", "fuel_alpha", "o2", "gas_density", "reynolds",
    "peclet", "prandtl", "diffusion_time", "pressure", "flow", "gravity",
    "ignition_power", "ignition_time", "hrr",
]

# ── Style ────────────────────────────────────────────────────────────────────
IGNITION_COLORS = {"Yes": "#24935b", "No": "#d1495b"}
GRAVITY_ORDER   = ["Microgravity (<=0.01 g)", "Partial gravity",
                   "Earth gravity", "Hypergravity"]
GRAVITY_COLORS  = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
PALETTE         = ["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
DILUENT_COLORS  = {"N2": "#2563eb", "CO2": "#dc2626", "Ar": "#10b981",
                   "He": "#f59e0b"}

plt.rcParams.update({
    "figure.dpi":       130,
    "savefig.dpi":      160,
    "font.size":        9.5,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.grid":        True,
    "grid.alpha":       0.22,
    "figure.facecolor": "white",
    "axes.facecolor":   "#fbfdff",
})


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path, default=DEFAULT_INPUT,
                   help="Path to Microgravity_Database.csv")
    p.add_argument("--output", type=Path, default=BASE_DIR,
                   help="Parent directory for output folders")
    p.add_argument("--clean",  action="store_true",
                   help="Delete existing output folders before writing")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def clean_text(series: pd.Series) -> pd.Series:
    return (series.astype("string").str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}))


def parse_flame_length_m(value: object) -> float:
    """Parse mixed-unit Flame Length field → metres."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split()
    if len(parts) != 2:
        return np.nan
    try:
        mag = float(parts[0])
    except ValueError:
        return np.nan
    return mag * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}.get(parts[1], np.nan)


def load_database(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")

    df = pd.read_csv(path, header=1, encoding="cp1252",
                     low_memory=False).dropna(how="all")

    missing = sorted(set(COL.values()) - set(df.columns))
    if missing:
        raise ValueError(
            "Schema mismatch — missing columns:\n  " + "\n  ".join(missing))

    # Numeric columns
    for key in NUMERIC_KEYS:
        df[key] = (pd.to_numeric(df[COL[key]], errors="coerce")
                   .replace([np.inf, -np.inf], np.nan))
    df["fsr"]           = (pd.to_numeric(df[COL["fsr"]], errors="coerce")
                           .replace([np.inf, -np.inf], np.nan))
    df["flame_length_m"] = df[COL["flame_length"]].map(parse_flame_length_m)

    # Text columns
    df["paper"]             = clean_text(df[COL["paper"]])
    df["authors"]           = clean_text(df[COL["authors"]])
    df["material"]          = clean_text(df[COL["material"]])
    df["geometry"]          = clean_text(df[COL["geometry"]]).str.title()
    df["facility"]          = clean_text(df[COL["facility"]]).str.title()
    df["diluent"]           = clean_text(df[COL["diluent"]])
    df["rig"]               = clean_text(df[COL["rig"]])
    df["internal_geometry"] = clean_text(df[COL["internal_geometry"]]).str.title()
    df["ignition_method"]   = clean_text(df[COL["ignition_method"]]).str.title()
    df["ignition"]          = clean_text(df[COL["ignition"]]).str.title()
    df.loc[~df["ignition"].isin(["Yes", "No"]), "ignition"] = pd.NA
    df["ignition_binary"]   = df["ignition"].map({"No": 0.0, "Yes": 1.0})

    # Derived columns
    df["pressure_kpa"]      = df["pressure"] / 1_000.0
    df["fsr_mm_s"]          = df["fsr"] * 1_000.0
    df["abs_flow"]          = df["flow"].abs()
    df["ignition_energy_j"] = df["ignition_power"] * df["ignition_time"]
    df["gravity_regime"]    = pd.cut(
        df["gravity"],
        [-np.inf, 0.01, 0.9, 1.1, np.inf],
        labels=GRAVITY_ORDER,
        include_lowest=True,
    )

    # Short paper id
    first_author = (df["authors"].fillna(df["paper"])
                    .str.split(",").str[0].str.strip())
    codes, _ = pd.factorize(df["paper"], sort=True)
    df["paper_id"] = [f"P{c+1:02d} {a[:18]}"
                      for c, a in zip(codes, first_author)]
    return df


# ═════════════════════════════════════════════════════════════════════════════
# PLOT WRITER
# ═════════════════════════════════════════════════════════════════════════════
class PlotWriter:
    def __init__(self, root: Path, clean: bool = False) -> None:
        self.root  = root.resolve()
        self.saved: list[Path] = []
        if clean:
            for p in self.root.glob("*.png"):
                p.unlink()
            for folder in OUTPUT_FOLDERS.values():
                target = self.root / folder
                if target.exists():
                    shutil.rmtree(target)
        for folder in OUTPUT_FOLDERS.values():
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, category: str, filename: str) -> None:
        path = self.root / OUTPUT_FOLDERS[category] / filename
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.saved.append(path)
        print(f"  saved  {path.relative_to(self.root)}")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def annotate_bars(ax: plt.Axes, fmt: str = ",.0f") -> None:
    for container in ax.containers:
        labels = [format(bar.get_height(), fmt) for bar in container]
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def safe_rate_table(df: pd.DataFrame, group: str,
                    min_n: int = 20) -> pd.DataFrame:
    tbl = (df.groupby(group, observed=True)["ignition_binary"]
           .agg(["mean", "count"]))
    return tbl[tbl["count"] >= min_n].sort_values("mean")


def range_and_gaps(
    vals: pd.Series,
    label: str,
    unit: str,
    writer: PlotWriter,
    category: str,
    filename: str,
    log: bool = False,
    bins: int = 40,
    top_gaps: int = 8,
) -> None:
    """Histogram + coverage strip with largest untested gaps highlighted."""
    v = pd.to_numeric(vals, errors="coerce").dropna()
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    u = np.sort(v.unique())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1.3]})

    b = (np.logspace(np.log10(u.min()), np.log10(u.max()), bins)
         if log else bins)
    if log:
        ax1.set_xscale("log")
    ax1.hist(v, bins=b, color="#2563eb", edgecolor="white", alpha=0.85)
    ax1.set_ylabel("Rows")
    ax1.set_title(
        f"Range of {label}: {u.min():.4g} – {u.max():.4g} {unit}  "
        f"({len(u):,} unique values, n={len(v):,})")

    ax2.eventplot(u, orientation="horizontal", colors="#2563eb",
                  lineoffsets=0.5, linelengths=0.8, linewidths=0.8)
    if log:
        ax2.set_xscale("log")
        gaps = np.diff(np.log10(u))
    else:
        gaps = np.diff(u)
    if len(gaps):
        idx = np.argsort(gaps)[::-1][:top_gaps]
        for i in idx:
            ax2.axvspan(u[i], u[i + 1], color="red", alpha=0.25)
        i0 = idx[0]
        mid = np.sqrt(u[i0] * u[i0 + 1]) if log else (u[i0] + u[i0 + 1]) / 2
        ax2.text(mid, 1.08, f"largest gap:\n{u[i0]:.4g} – {u[i0+1]:.4g}",
                 ha="center", fontsize=8, color="darkred")
    ax2.set_ylim(0, 1.5)
    ax2.set_yticks([])
    ax2.set_xlabel(f"{label} {unit}")
    ax2.set_title(
        f"Coverage strip – {top_gaps} largest untested gaps (red)", fontsize=10)
    fig.tight_layout()
    writer.save(fig, category, filename)


# ═════════════════════════════════════════════════════════════════════════════
# 01 OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
def overview_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    n        = len(df)
    papers   = df["paper"].nunique()
    n_fsr    = df["fsr"].notna().sum()
    psizes   = df.groupby("paper", observed=True).size().sort_values(ascending=False)

    # ── 01 Dataset overview card ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    stats = [
        ("Experimental rows",         f"{n:,}"),
        ("Papers",                     f"{papers}"),
        ("Distinct materials",         f"{df['material'].nunique()}"),
        ("Facilities / rigs",          f"{df['facility'].nunique()} / {df['rig'].nunique()}"),
        ("Ignition rate",              f"{100 * df['ignition_binary'].mean():.1f}%"),
        ("Numeric FSR targets",        f"{n_fsr:,}  ({100*n_fsr/n:.1f}%)"),
        ("O₂ range",                  f"{df['o2'].min():.3f} – {df['o2'].max():.3f}"),
        ("Pressure range",             f"{df['pressure_kpa'].min():.1f} – {df['pressure_kpa'].max():,.0f} kPa"),
        ("Gravity range",              f"{df['gravity'].min():.3g} – {df['gravity'].max():.3g} g"),
        ("Flow range",                 f"{df['flow'].min():.2f} – {df['flow'].max():.2f} m/s"),
        ("FSR range",                  f"{df['fsr_mm_s'].min():.3g} – {df['fsr_mm_s'].max():.0f} mm/s"),
        ("Median rows / paper",        f"{psizes.median():.0f}"),
    ]
    for i, (label, val) in enumerate(stats):
        row, col = divmod(i, 3)
        x = 0.03 + col * 0.33
        y = 0.88 - row * 0.22
        ax.text(x, y,        label, weight="bold", color="#334155",
                transform=ax.transAxes)
        ax.text(x, y - 0.08, val,   fontsize=15,   color=PALETTE[col],
                transform=ax.transAxes)
    ax.set_title("Microgravity Flammability Database — Overview",
                 fontsize=17, pad=20)
    writer.save(fig, "overview", "01_dataset_overview.png")

    # ── 02 Points per paper (all) ─────────────────────────────────────────
    pids   = df.groupby("paper_id", observed=True).size().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(8, len(pids) * 0.22)))
    pids.plot.barh(ax=ax, color="#2563eb")
    ax.set(xlabel="Rows", ylabel="", title=f"Rows per Paper (all {papers} papers)")
    ax.tick_params(axis="y", labelsize=7)
    writer.save(fig, "overview", "02_rows_per_paper_all.png")

    # ── 03 Top-25 papers ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    psizes.head(25).sort_values().plot.barh(ax=ax, color="#f59e0b")
    ax.set(xlabel="Rows", title="Top 25 Papers by Row Count")
    writer.save(fig, "overview", "03_top25_papers.png")

    # ── 04 Pareto / concentration ─────────────────────────────────────────
    cumulative = psizes.cumsum() / n * 100
    n50 = int((cumulative < 50).sum() + 1)
    n80 = int((cumulative < 80).sum() + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative,
            marker="o", ms=3, color="#0f766e")
    for share, count in [(50, n50), (80, n80)]:
        ax.axhline(share, color="#dc2626", ls="--", alpha=.5)
        ax.axvline(count, color="#dc2626", ls=":", alpha=.5,
                   label=f"{count} papers → {share}%")
    ax.set(xlabel="Papers (largest first)", ylabel="Cumulative share (%)",
           title="Dataset Concentration Across Sources")
    ax.legend()
    writer.save(fig, "overview", "04_pareto_concentration.png")

    # ── 05 Paper-size histogram ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(psizes.values, bins=30, color="#0891b2", edgecolor="white")
    ax.axvline(psizes.median(), color="r", ls="--",
               label=f"median = {psizes.median():.0f}")
    ax.axvline(psizes.mean(),   color="b", ls="--",
               label=f"mean = {psizes.mean():.1f}")
    ax.set(xlabel="Rows in paper", ylabel="Number of papers",
           title="Distribution of Paper Sizes")
    ax.legend()
    writer.save(fig, "overview", "05_paper_size_histogram.png")

    # ── 06 Field completeness ─────────────────────────────────────────────
    fields = {
        "Paper": "paper", "Material": "material", "Geometry": "geometry",
        "Fuel density": "fuel_density", "Oxygen": "o2", "Pressure": "pressure",
        "Flow velocity": "flow", "Gravity": "gravity", "Facility": "facility",
        "Rig": "rig", "Ignition method": "ignition_method",
        "Ignition power": "ignition_power", "Ignition time": "ignition_time",
        "Ignition outcome": "ignition", "FSR target": "fsr",
        "Flame length": "flame_length_m", "HRR": "hrr",
        "Smoke": COL["smoke"],
    }
    comp = pd.Series(
        {lbl: 100 * df[col].notna().mean()
         for lbl, col in fields.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#dc2626" if v < 25 else "#f59e0b" if v < 75 else "#0f766e"
              for v in comp]
    ax.barh(comp.index, comp.values, color=colors)
    for i, v in enumerate(comp):
        ax.text(v + 0.7, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set(xlim=(0, 108), xlabel="Rows populated (%)",
           title="Field Completeness")
    writer.save(fig, "overview", "06_field_completeness.png")

    # ── 07 Target availability by paper (top 30) ─────────────────────────
    tgt = (df.groupby("paper_id", observed=True)
           .agg(rows=("paper", "size"),
                fsr=("fsr", "count"),
                ignition=("ignition", "count")))
    tgt["FSR available"]      = 100 * tgt["fsr"]      / tgt["rows"]
    tgt["Ignition available"] = 100 * tgt["ignition"] / tgt["rows"]
    tgt = (tgt.sort_values("rows", ascending=False)
           .head(30).sort_values("rows"))
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(tgt))
    ax.barh(y, tgt["Ignition available"], color="#94a3b8", label="Ignition")
    ax.barh(y, tgt["FSR available"],      color="#7c3aed", label="Numeric FSR")
    ax.set_yticks(y, tgt.index, fontsize=7)
    ax.set(xlabel="Target availability within paper (%)",
           title="Outcome Availability by Major Data Source")
    ax.legend()
    writer.save(fig, "overview", "07_target_availability_by_paper.png")

    # ── 08 Paper size vs ignition fraction bubble (from old script) ───────
    agg = df.groupby(COL["paper"]).agg(
        n=("ignition_binary", "size"),
        ign_frac=("ignition_binary", "mean"),
        o2_span=(COL["o2"], lambda s: s.max() - s.min()),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        agg["n"], agg["ign_frac"] * 100,
        s=30 + 600 * agg["o2_span"].fillna(0),
        alpha=0.55, c=agg["o2_span"], cmap="viridis",
        edgecolors="k", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="O₂ span explored (mole fraction)")
    ax.set_xscale("log")
    ax.set(xlabel="Rows in paper (log)", ylabel="% Ignition = Yes",
           title="Paper Size vs Ignition Fraction (bubble = O₂ span explored)")
    writer.save(fig, "overview", "08_paper_size_vs_ignition_fraction.png")


# ═════════════════════════════════════════════════════════════════════════════
# 02 PARAMETER COVERAGE
# ═════════════════════════════════════════════════════════════════════════════
def coverage_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    # ── 01 Experimental composition (6-panel) ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (col, title) in zip(axes.flat, [
        ("facility",         "Facility"),
        ("geometry",         "Sample geometry"),
        ("diluent",          "Diluent"),
        ("internal_geometry","Rig internal geometry"),
        ("ignition_method",  "Ignition method"),
        ("gravity_regime",   "Gravity regime"),
    ]):
        vc = (df[col].astype("string").fillna("Unknown")
              .value_counts().head(10).sort_values())
        ax.barh(vc.index, vc.values, color="#0891b2")
        ax.set_title(title)
        ax.set_xlabel("Rows")
    fig.suptitle("Experimental Composition", fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "coverage", "01_experimental_composition.png")

    # ── 02 Core parameter histograms (6-panel) ───────────────────────────
    specs = [
        ("o2",           "Oxygen mole fraction",       False),
        ("pressure_kpa", "Pressure (kPa)",             True),
        ("gravity",      "Gravity (g)",                False),
        ("flow",         "Flow velocity (m/s)",        False),
        ("length",       "Characteristic length (m)",  True),
        ("half_thickness","Half-thickness (m)",        True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (col, label, log) in zip(axes.flat, specs):
        vals = df[col].dropna()
        vals = vals[vals > 0] if log else vals
        bins = (np.logspace(np.log10(vals.min()), np.log10(vals.max()), 35)
                if log else 35)
        ax.hist(vals, bins=bins, color="#2563eb", alpha=.82, edgecolor="white")
        if log:
            ax.set_xscale("log")
        ax.set_title(f"{label}\nn={len(vals):,}, unique={vals.nunique():,}")
        ax.set(xlabel=label, ylabel="Rows")
    fig.suptitle("Core Experimental Parameter Coverage", fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "coverage", "02_core_parameter_distributions.png")

    # ── 03 O₂ × Pressure hexbin density ──────────────────────────────────
    sub = df.dropna(subset=["o2", "pressure_kpa"])
    sub = sub[sub["pressure_kpa"] > 0]
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(sub["o2"], sub["pressure_kpa"], gridsize=42,
                   yscale="log", mincnt=1, norm=LogNorm(), cmap="viridis")
    fig.colorbar(hb, ax=ax, label="Rows per cell (log)")
    ax.set(xlabel="Oxygen mole fraction", ylabel="Pressure (kPa, log)",
           title="Joint Coverage Density: O₂ × Pressure")
    writer.save(fig, "coverage", "03_oxygen_pressure_hexbin.png")

    # ── 04 O₂ vs Pressure — coloured by ignition ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for lab, col in IGNITION_COLORS.items():
        s = sub[sub["ignition"] == lab]
        ax.scatter(s["o2"], s["pressure_kpa"], s=12, alpha=0.45, c=col,
                   label=f"Ignition={lab} (n={len(s):,})", edgecolors="none")
    ax.set_yscale("log")
    ax.set(xlabel="Oxygen mole fraction", ylabel="Pressure (kPa, log)",
           title="O₂ vs Pressure — coloured by Ignition outcome")
    ax.legend()
    writer.save(fig, "coverage", "04_o2_pressure_by_ignition.png")

    # ── 05 O₂ vs Pressure — coloured by gravity regime ───────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for regime, color in zip(GRAVITY_ORDER, GRAVITY_COLORS):
        s = sub[sub["gravity_regime"] == regime]
        ax.scatter(s["o2"], s["pressure_kpa"], s=12, alpha=0.45, c=color,
                   label=f"{regime} (n={len(s):,})", edgecolors="none")
    ax.set_yscale("log")
    ax.set(xlabel="Oxygen mole fraction", ylabel="Pressure (kPa, log)",
           title="O₂ vs Pressure — coloured by Gravity Regime")
    ax.legend(fontsize=8)
    writer.save(fig, "coverage", "05_o2_pressure_by_gravity.png")

    # ── 06 O₂ vs Pressure — coloured by diluent ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for dil, color in DILUENT_COLORS.items():
        s = sub[sub["diluent"] == dil]
        ax.scatter(s["o2"], s["pressure_kpa"], s=12, alpha=0.5, c=color,
                   label=f"{dil} (n={len(s):,})", edgecolors="none")
    ax.set_yscale("log")
    ax.set(xlabel="Oxygen mole fraction", ylabel="Pressure (kPa, log)",
           title="O₂ vs Pressure — coloured by Diluent Gas")
    ax.legend()
    writer.save(fig, "coverage", "06_o2_pressure_by_diluent.png")

    # ── 07 O₂ vs Gravity — coloured by ignition ──────────────────────────
    sub2 = df.dropna(subset=["o2", "gravity"])
    fig, ax = plt.subplots(figsize=(10, 7))
    for lab, col in IGNITION_COLORS.items():
        s = sub2[sub2["ignition"] == lab]
        ax.scatter(s["gravity"], s["o2"], s=12, alpha=0.45, c=col,
                   label=f"Ignition={lab}", edgecolors="none")
    ax.set(xlabel="Gravity (g)", ylabel="Oxygen mole fraction",
           title="O₂ vs Gravity — coloured by Ignition outcome")
    ax.legend()
    writer.save(fig, "coverage", "07_o2_vs_gravity_by_ignition.png")

    # ── 08 Gravity × Flow by facility ────────────────────────────────────
    sub3 = df.dropna(subset=["gravity", "flow", "facility"])
    fig, ax = plt.subplots(figsize=(10, 7))
    top_fac = sub3["facility"].value_counts().head(7).index
    for color, fac in zip(PALETTE, top_fac):
        s = sub3[sub3["facility"] == fac]
        ax.scatter(s["gravity"], s["flow"], s=16, alpha=.45, color=color,
                   label=f"{fac} (n={len(s):,})")
    ax.axhline(0, color="black", lw=.8)
    ax.set(xlabel="Gravity (g)", ylabel="Signed flow velocity (m/s)",
           title="Gravity × Flow Coverage by Facility")
    ax.legend(fontsize=8)
    writer.save(fig, "coverage", "08_gravity_flow_by_facility.png")

    # ── 09 Dimensionless transport coverage (Re, Pe, Pr) ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, col, label in zip(axes,
                               ["reynolds", "peclet", "prandtl"],
                               ["Reynolds", "Péclet", "Prandtl"]):
        vals = df[col].dropna()
        vals = vals[vals > 0]
        ax.hist(vals,
                bins=np.logspace(np.log10(vals.min()), np.log10(vals.max()), 40),
                color="#7c3aed", edgecolor="white")
        ax.set_xscale("log")
        ax.set(title=f"{label}  (n={len(vals):,})",
               xlabel=f"{label} number (log)")
    fig.suptitle("Dimensionless Transport-Regime Coverage",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "coverage", "09_dimensionless_transport_coverage.png")

    # ── 10 Gravity regime bar + ignition share ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    vc = (df["gravity_regime"].value_counts()
          .reindex(GRAVITY_ORDER).fillna(0))
    ax1.bar(GRAVITY_ORDER, vc.values, color=GRAVITY_COLORS, edgecolor="k")
    for i, v in enumerate(vc):
        ax1.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    ax1.set(ylabel="Rows", title="Rows by Gravity Regime")
    ax1.tick_params(axis="x", rotation=12)
    ct = (pd.crosstab(df["gravity_regime"], df["ignition"],
                      normalize="index") * 100)
    ct.reindex(GRAVITY_ORDER)[["Yes", "No"]].plot.bar(
        stacked=True, ax=ax2,
        color=[IGNITION_COLORS["Yes"], IGNITION_COLORS["No"]], edgecolor="k")
    ax2.set(ylabel="% of rows", title="Ignition Share by Gravity Regime")
    ax2.tick_params(axis="x", rotation=12)
    ax2.legend(title="Ignition", loc="lower right")
    fig.tight_layout()
    writer.save(fig, "coverage", "10_gravity_regime_counts_and_ignition.png")

    # ── 11 Per-paper O₂ span (top-30 papers) ──────────────────────────────
    top30 = (df.groupby("paper_id", observed=True).size()
             .nlargest(30).index)
    rng = (df.groupby("paper_id", observed=True)["o2"]
           .agg(["min", "max", "count"])
           .reindex(top30).dropna().sort_values("count"))
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (name, row) in enumerate(rng.iterrows()):
        ax.plot([row["min"], row["max"]], [i, i], lw=3,
                color="#2563eb", alpha=0.7)
        ax.plot([row["min"], row["max"]], [i, i], "o", ms=4, color="#2563eb")
    ax.set_yticks(range(len(rng)))
    ax.set_yticklabels(rng.index, fontsize=7)
    ax.set(xlabel="Oxygen mole fraction",
           title="O₂ Range Explored per Paper (top 30)")
    writer.save(fig, "coverage", "11_o2_span_per_paper.png")

    # ── 12 Per-paper pressure span (top-30 papers) ────────────────────────
    rng = (df.groupby("paper_id", observed=True)["pressure_kpa"]
           .agg(["min", "max", "count"])
           .reindex(top30).dropna().sort_values("count"))
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (name, row) in enumerate(rng.iterrows()):
        ax.plot([row["min"], row["max"]], [i, i], lw=3,
                color="#dc2626", alpha=0.7)
        ax.plot([row["min"], row["max"]], [i, i], "o", ms=4, color="#dc2626")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rng)))
    ax.set_yticklabels(rng.index, fontsize=7)
    ax.set(xlabel="Pressure (kPa, log)",
           title="Pressure Range Explored per Paper (top 30)")
    writer.save(fig, "coverage", "12_pressure_span_per_paper.png")

    # ── 13 Fuel property space ────────────────────────────────────────────
    sub4 = df.dropna(subset=["fuel_density", "fuel_k"])
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(sub4["fuel_density"], sub4["fuel_k"],
                    s=14, alpha=0.5,
                    c=sub4["fuel_cp"], cmap="cividis")
    fig.colorbar(sc, ax=ax, label="Fuel cp [J/kg-K]")
    ax.set_yscale("log")
    ax.set(xlabel="Fuel density [kg/m³]",
           ylabel="Fuel conductivity [W/m-K] (log)",
           title="Fuel Property Space: density vs conductivity (coloured by cp)")
    writer.save(fig, "coverage", "13_fuel_property_space.png")

    # ── 14 Material test-space breadth ────────────────────────────────────
    mat = df.groupby("material", observed=True).agg(
        rows=("paper", "size"),
        papers=("paper", "nunique"),
        o2_span=("o2", lambda x: x.max() - x.min()),
        pressure_ratio=("pressure_kpa",
                        lambda x: x.max() / x.min() if x.min() > 0 else np.nan),
    )
    mat = mat[mat["rows"] >= 20].sort_values("rows").tail(25)
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(mat["o2_span"], mat["pressure_ratio"],
                    s=30 + 20 * mat["papers"],
                    c=mat["rows"], cmap="plasma", norm=LogNorm(),
                    alpha=.8, edgecolor="white")
    for name, row in mat.tail(12).iterrows():
        ax.annotate(str(name)[:28], (row["o2_span"], row["pressure_ratio"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_yscale("log")
    ax.set(xlabel="O₂ span", ylabel="Pressure range ratio (max/min, log)",
           title="Material Test-Space Breadth")
    fig.colorbar(sc, ax=ax, label="Rows (log); bubble size = papers")
    writer.save(fig, "coverage", "14_material_test_space_breadth.png")

    # ── 15-22 Individual range-and-gap plots ──────────────────────────────
    gap_specs = [
        ("fuel_density", "Fuel Density",               "[kg/m³]",  False, "15_fuel_density_range_gaps.png"),
        ("fuel_k",       "Fuel Thermal Conductivity",  "[W/m-K]",  True,  "16_fuel_conductivity_range_gaps.png"),
        ("fuel_cp",      "Fuel Specific Heat cp",      "[J/kg-K]", False, "17_fuel_cp_range_gaps.png"),
        ("o2",           "Oxygen Concentration",       "[mol fr]", False, "18_oxygen_range_gaps.png"),
        ("pressure_kpa", "Pressure",                   "[kPa]",    True,  "19_pressure_range_gaps.png"),
        ("gravity",      "Gravity Level",              "[g]",      False, "20_gravity_range_gaps.png"),
        ("flow",         "Flow Velocity",              "[m/s]",    False, "21_flow_velocity_range_gaps.png"),
        ("fsr_mm_s",     "Flame Spread Rate (FSR)",    "[mm/s]",   True,  "22_fsr_range_gaps.png"),
    ]
    for col, label, unit, log, fname in gap_specs:
        range_and_gaps(df[col], label, unit, writer, "coverage", fname, log=log)

    # ── 23 Gap summary table ──────────────────────────────────────────────
    lines = []
    for col, label, unit, log, _ in gap_specs:
        u = pd.to_numeric(df[col], errors="coerce").dropna()
        u = u[u > 0] if log else u
        u = np.sort(u.unique())
        if len(u) < 2:
            continue
        g = np.diff(np.log10(u)) if log else np.diff(u)
        for i in np.argsort(g)[-3:][::-1]:
            score = 10 ** g[i] if log else g[i]
            desc  = f"×{score:.2g}" if log else f"Δ={score:.3g}"
            lines.append((label, u[i], u[i + 1], unit, desc))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.set_title("Largest One-Dimensional Sampling Gaps", fontsize=15)
    hdr_y = 0.95
    for x, txt in zip([.02, .28, .50, .68, .84],
                      ["Variable", "Lower bound", "Upper bound", "Unit", "Gap"]):
        ax.text(x, hdr_y, txt, weight="bold", transform=ax.transAxes)
    for i, (label, lo, hi, unit, desc) in enumerate(lines):
        y = hdr_y - 0.048 * (i + 1)
        ax.text(.02, y, label,           transform=ax.transAxes)
        ax.text(.28, y, f"{lo:.5g}",    transform=ax.transAxes, family="monospace")
        ax.text(.50, y, f"{hi:.5g}",    transform=ax.transAxes, family="monospace")
        ax.text(.68, y, unit,            transform=ax.transAxes, color="#475569")
        ax.text(.84, y, desc,            transform=ax.transAxes, color="#dc2626")
    writer.save(fig, "coverage", "23_sampling_gaps_summary.png")


# ═════════════════════════════════════════════════════════════════════════════
# 03 IGNITION METRICS
# ═════════════════════════════════════════════════════════════════════════════
def ignition_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    counts = df["ignition"].value_counts().reindex(["Yes", "No"]).fillna(0)

    # ── 01 Outcome balance pie + per-paper histogram ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].pie(
        counts,
        labels=[f"{x}: {int(v):,}\n({100*v/counts.sum():.1f}%)"
                for x, v in counts.items()],
        colors=[IGNITION_COLORS[x] for x in counts.index],
        startangle=90, wedgeprops={"edgecolor": "white"})
    axes[0].set_title("Pooled ignition outcomes")
    per_paper = df.groupby("paper", observed=True)["ignition_binary"].mean().dropna()
    axes[1].hist(per_paper * 100, bins=np.arange(0, 105, 5),
                 color="#7c3aed", edgecolor="white")
    axes[1].axvline(100 * per_paper.mean(), color="#dc2626", ls="--",
                    label=f"paper-balanced mean={100*per_paper.mean():.1f}%")
    axes[1].set(xlabel="Ignition rate within paper (%)", ylabel="Papers",
                title="Source-level heterogeneity")
    axes[1].legend(fontsize=8)
    fig.suptitle("Ignition Outcome Balance: Rows vs Sources",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "ignition", "01_outcome_balance.png")

    # ── 02-04 Rate by facility / geometry / ignition method ───────────────
    for num, (group, title) in enumerate(
        [("facility", "Facility"),
         ("geometry", "Sample Geometry"),
         ("ignition_method", "Ignition Method")], start=2
    ):
        table = safe_rate_table(df, group).sort_values("mean")
        fig, ax = plt.subplots(
            figsize=(10, max(5, .55 * len(table) + 1.5)))
        colors = plt.cm.RdYlGn(table["mean"].to_numpy())
        ax.barh(table.index.astype(str), 100 * table["mean"], color=colors)
        for i, (_, row) in enumerate(table.iterrows()):
            ax.text(100 * row["mean"] + 1, i,
                    f"{100*row['mean']:.1f}%  n={int(row['count']):,}",
                    va="center", fontsize=8)
        ax.set(xlim=(0, 115), xlabel="Ignition rate (%)",
               title=f"Ignition Rate by {title}  (groups with n ≥ 20)")
        writer.save(fig, "ignition", f"{num:02d}_ignition_rate_by_{group}.png")

    # ── 05 Pooled vs paper-balanced rate by facility ──────────────────────
    rows = []
    for fac, part in df.groupby("facility", observed=True):
        src = part.groupby("paper", observed=True)["ignition_binary"].mean().dropna()
        if len(part) >= 30 and len(src) >= 2:
            rows.append((fac, part["ignition_binary"].mean(),
                         src.mean(), len(part), len(src)))
    balance = pd.DataFrame(
        rows, columns=["facility", "pooled", "paper_balanced", "rows", "papers"]
    ).sort_values("pooled")
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(balance))
    ax.plot(100 * balance["pooled"],         y, "o", ms=8,
            label="Pooled rows",             color="#2563eb")
    ax.plot(100 * balance["paper_balanced"], y, "s", ms=7,
            label="Each paper weighted equally", color="#f59e0b")
    for _, row in balance.iterrows():
        ax.plot([100 * row["pooled"], 100 * row["paper_balanced"]],
                [y[balance.index.get_loc(_)],
                 y[balance.index.get_loc(_)]],
                color="#94a3b8", zorder=0)
    ax.set_yticks(y, [f"{r.facility} (p={r.papers}, n={r.rows})"
                      for r in balance.itertuples()])
    ax.set(xlabel="Ignition rate (%)",
           title="Facility Ignition Rate: Pooled vs Paper-Balanced")
    ax.legend()
    writer.save(fig, "ignition", "05_pooled_vs_paper_balanced.png")

    # ── 06 Ignition probability heatmap in O₂ × Pressure space ───────────
    valid = df.dropna(subset=["o2", "pressure_kpa", "ignition_binary"])
    valid = valid[valid["pressure_kpa"] > 0].copy()
    valid["o2_bin"] = pd.cut(
        valid["o2"],
        bins=np.linspace(valid["o2"].min(), valid["o2"].max(), 18))
    valid["p_bin"]  = pd.cut(np.log10(valid["pressure_kpa"]), bins=14)
    rates   = valid.pivot_table(
        index="p_bin", columns="o2_bin",
        values="ignition_binary", aggfunc="mean", observed=True)
    counts2 = valid.pivot_table(
        index="p_bin", columns="o2_bin",
        values="ignition_binary", aggfunc="count", observed=True)
    rates = rates.where(counts2 >= 5)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(rates.to_numpy(), origin="lower", aspect="auto",
                   vmin=0, vmax=1, cmap="RdYlGn")
    xcent = [itv.mid for itv in rates.columns]
    ycent = [10 ** itv.mid for itv in rates.index]
    ax.set_xticks(np.arange(len(xcent))[::2],
                  [f"{x:.2f}" for x in xcent[::2]])
    ax.set_yticks(np.arange(len(ycent))[::2],
                  [f"{y:.1f}" for y in ycent[::2]])
    ax.set(xlabel="O₂ mole-fraction bin",
           ylabel="Pressure bin centre (kPa, log-spaced)",
           title="Empirical Ignition Probability: O₂ × Pressure (cells with n ≥ 5)")
    fig.colorbar(im, ax=ax, label="Observed ignition probability")
    writer.save(fig, "ignition", "06_oxygen_pressure_ignition_heatmap.png")

    # ── 07 Igniter operating envelope ────────────────────────────────────
    energy = df.dropna(subset=["ignition_power", "ignition_time", "ignition"])
    fig, ax = plt.subplots(figsize=(10, 7))
    for outcome, color in IGNITION_COLORS.items():
        s = energy[energy["ignition"] == outcome]
        ax.scatter(s["ignition_time"], s["ignition_power"],
                   s=20, alpha=.5, color=color,
                   label=f"{outcome} (n={len(s):,})")
    ax.set(xlabel="Ignition time (s)", ylabel="Ignition power (W)",
           title="Igniter Operating Envelope and Outcome")
    ax.legend()
    writer.save(fig, "ignition", "07_ignition_power_time_envelope.png")

    # ── 08 Ignition rate by material (top materials) ──────────────────────
    table = safe_rate_table(df, "material", min_n=30)
    table = table.sort_values("count", ascending=False).head(25).sort_values("mean")
    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(table))
    rc = plt.cm.RdYlGn(table["mean"].to_numpy(copy=True))
    ax.barh(y, 100 * table["mean"], color=rc)
    ax.set_yticks(y, [str(n)[:48] for n in table.index], fontsize=7)
    for i, (_, row) in enumerate(table.iterrows()):
        ax.text(100 * row["mean"] + 1, i,
                f"{100*row['mean']:.0f}%  (n={int(row['count'])})",
                va="center", fontsize=7)
    ax.set(xlim=(0, 118), xlabel="Ignition rate (%)",
           title="Ignition Rate for Best-Represented Materials  (n ≥ 30)")
    writer.save(fig, "ignition", "08_ignition_rate_by_material.png")


# ═════════════════════════════════════════════════════════════════════════════
# 04 FSR METRICS
# ═════════════════════════════════════════════════════════════════════════════
def fsr_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    # ── 01 FSR parsing readiness ──────────────────────────────────────────
    raw_present = df[COL["fsr"]].notna()
    numeric     = df["fsr"].notna()
    positive    = df["fsr"] > 0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["Raw present", "Numeric SI", "Positive targets"],
           [raw_present.sum(), numeric.sum(), positive.sum()],
           color=["#94a3b8", "#7c3aed", "#0f766e"])
    annotate_bars(ax)
    ax.set(ylabel="Rows", title="FSR Target Parsing and Regression Readiness")
    ax.tick_params(axis="x", rotation=10)
    n_excluded = int((raw_present & ~numeric).sum())
    ax.text(.98, .94, f"Excluded descriptive/ratio values: {n_excluded}",
            ha="right", va="top", transform=ax.transAxes, color="#dc2626")
    writer.save(fig, "fsr", "01_fsr_target_readiness.png")

    fs = df[df["fsr_mm_s"] > 0].copy()

    # ── 02 FSR distribution: row-level vs paper-balanced ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(fs["fsr_mm_s"],
                 bins=np.logspace(
                     np.log10(fs["fsr_mm_s"].min()),
                     np.log10(fs["fsr_mm_s"].max()), 45),
                 color="#7c3aed", edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set(xlabel="FSR (mm/s, log)", ylabel="Rows",
                title="Positive FSR distribution")
    paper_med = fs.groupby("paper", observed=True)["fsr_mm_s"].median()
    axes[1].hist(paper_med,
                 bins=np.logspace(
                     np.log10(paper_med.min()),
                     np.log10(paper_med.max()), 28),
                 color="#0891b2", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Median FSR per paper (mm/s, log)", ylabel="Papers",
                title="Paper-balanced target distribution")
    fig.suptitle("FSR Distribution: Row-Level vs Source-Level",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "fsr", "02_fsr_distribution_row_vs_paper.png")

    # ── 03 FSR vs O₂ and Pressure ────────────────────────────────────────
    sub = fs.dropna(subset=["o2", "pressure_kpa"])
    sub = sub[sub["pressure_kpa"] > 0]
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(sub["o2"], sub["fsr_mm_s"],
                    c=sub["pressure_kpa"], norm=LogNorm(),
                    cmap="plasma", s=18, alpha=.55)
    ax.set_yscale("log")
    ax.set(xlabel="Oxygen mole fraction", ylabel="FSR (mm/s, log)",
           title="Flame Spread Rate vs Oxygen and Pressure")
    fig.colorbar(sc, ax=ax, label="Pressure (kPa, log)")
    writer.save(fig, "fsr", "03_fsr_vs_oxygen_pressure.png")

    # ── 04 FSR vs Gravity and signed Flow ────────────────────────────────
    sub = fs.dropna(subset=["gravity", "flow"])
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(sub["gravity"], sub["fsr_mm_s"],
                    c=sub["flow"], cmap="coolwarm", s=18, alpha=.55)
    ax.set_yscale("log")
    ax.set(xlabel="Gravity (g)", ylabel="FSR (mm/s, log)",
           title="Flame Spread Rate vs Gravity and Signed Flow")
    fig.colorbar(sc, ax=ax, label="Flow velocity (m/s)")
    writer.save(fig, "fsr", "04_fsr_vs_gravity_flow.png")

    # ── 05 FSR vs Fuel density ────────────────────────────────────────────
    sub = fs.dropna(subset=["fuel_density"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(sub["fuel_density"], sub["fsr_mm_s"],
               s=14, alpha=0.5, color="#2563eb", edgecolors="none")
    ax.set_yscale("log")
    ax.set(xlabel="Fuel density [kg/m³]", ylabel="FSR (mm/s, log)",
           title="FSR vs Fuel Density")
    writer.save(fig, "fsr", "05_fsr_vs_fuel_density.png")

    # ── 06 FSR vs dimensionless transport (Re, Pe) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col, label in [(axes[0], "reynolds", "Reynolds"),
                           (axes[1], "peclet", "Péclet")]:
        sub = fs.dropna(subset=[col])
        sub = sub[sub[col] > 0]
        ax.scatter(sub[col], sub["fsr_mm_s"],
                   s=15, alpha=.4, color="#2563eb")
        ax.set_xscale("log")
        ax.set_yscale("log")
        corr = sub[[col, "fsr_mm_s"]].corr(method="spearman").iloc[0, 1]
        ax.set(xlabel=f"{label} number (log)", ylabel="FSR (mm/s, log)",
               title=f"FSR vs {label}  (Spearman r={corr:.2f})")
    fig.suptitle("FSR Across Dimensionless Transport Regimes",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "fsr", "06_fsr_vs_dimensionless_transport.png")

    # ── 07 FSR boxplot by geometry ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    groups, labels = [], []
    for gname, gdf in fs.groupby("geometry", observed=True):
        if len(gdf) >= 10:
            groups.append(gdf["fsr_mm_s"].values)
            labels.append(f"{gname}\n(n={len(gdf)})")
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set(ylabel="FSR (mm/s, log)",
           title="FSR Distribution by Sample Geometry (outliers hidden)")
    writer.save(fig, "fsr", "07_fsr_boxplot_by_geometry.png")

    # ── 08 FSR boxplot by gravity regime ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    groups, labels = [], []
    for regime in GRAVITY_ORDER:
        vals = fs.loc[fs["gravity_regime"] == regime, "fsr_mm_s"].dropna()
        if len(vals) >= 10:
            groups.append(vals)
            labels.append(f"{regime}\n(n={len(vals):,})")
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.set(ylabel="FSR (mm/s, log)",
           title="FSR Distribution by Gravity Regime (outliers hidden)")
    ax.tick_params(axis="x", rotation=10)
    writer.save(fig, "fsr", "08_fsr_boxplot_by_gravity_regime.png")

    # ── 09 Material-level FSR (median ± IQR) ─────────────────────────────
    mat = fs.groupby("material", observed=True)["fsr_mm_s"].agg(
        ["median", "count",
         lambda x: x.quantile(.25),
         lambda x: x.quantile(.75)])
    mat.columns = ["median", "count", "q25", "q75"]
    mat = mat[mat["count"] >= 20].sort_values("median")
    fig, ax = plt.subplots(figsize=(11, max(6, .4 * len(mat))))
    y = np.arange(len(mat))
    ax.errorbar(mat["median"], y,
                xerr=[mat["median"] - mat["q25"],
                      mat["q75"] - mat["median"]],
                fmt="o", color="#7c3aed", ecolor="#a78bfa", capsize=3)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{str(n)[:48]}  (n={int(r['count'])})"
                      for n, r in mat.iterrows()], fontsize=7)
    ax.set(xlabel="Median FSR ± IQR (mm/s, log)",
           title="Material-Level Flame Spread Summary  (n ≥ 20)")
    writer.save(fig, "fsr", "09_fsr_by_material.png")

    # ── 10 Total vs within-paper Spearman associations ────────────────────
    variables = ["o2", "pressure_kpa", "gravity", "flow",
                 "reynolds", "peclet",
                 "fuel_density", "fuel_k", "fuel_cp",
                 "length", "half_thickness"]
    total, within = [], []
    log_fsr = np.log10(fs["fsr_mm_s"])
    for col in variables:
        total.append(fs[col].corr(log_fsr, method="spearman"))
        demeaned = (fs[col] -
                    fs.groupby("paper", observed=True)[col].transform("mean"))
        log_fsr_dm = log_fsr - fs.groupby("paper", observed=True)["fsr_mm_s"].transform(
            lambda x: np.log10(x).mean())
        within.append(demeaned.corr(log_fsr_dm, method="spearman"))
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(variables))
    w = 0.38
    ax.bar(x - w / 2, total,  w, label="Across all rows",          color="#2563eb")
    ax.bar(x + w / 2, within, w, label="Within-paper (demeaned)",  color="#f59e0b")
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(x, variables, rotation=35, ha="right")
    ax.set(ylabel="Spearman correlation with log₁₀(FSR)",
           title="Total vs Within-Paper FSR Associations")
    ax.legend()
    writer.save(fig, "fsr", "10_total_vs_within_paper_associations.png")


# ═════════════════════════════════════════════════════════════════════════════
# 05 DATA QUALITY
# ═════════════════════════════════════════════════════════════════════════════
def quality_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    features = ["fuel_density", "fuel_k", "fuel_cp", "o2", "pressure", "flow",
                "gravity", "facility", "ignition_method", "reynolds", "peclet",
                "ignition_power", "ignition_time", "fsr"]

    # ── 01 Feature co-availability ────────────────────────────────────────
    avail  = df[features].notna().astype(float)
    matrix = (avail.T @ avail).to_numpy(dtype=float)
    denom  = np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    sim    = np.divide(matrix, denom,
                       out=np.zeros_like(matrix), where=denom > 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if sim[i, j] > .65 else "black")
    fig.colorbar(im, ax=ax, label="Cosine co-availability")
    ax.set_title("Feature Co-Availability (can fields be modelled together?)")
    writer.save(fig, "quality", "01_feature_coavailability.png")

    # ── 02 Spearman correlation heatmap ───────────────────────────────────
    numeric_cols = [
        "half_thickness", "length", "fuel_density", "fuel_k", "fuel_cp",
        "fuel_pyrolysis_t", "fuel_alpha", "o2", "pressure_kpa", "flow",
        "gravity", "reynolds", "peclet", "prandtl", "diffusion_time",
        "ignition_power", "ignition_time", "fsr_mm_s",
    ]
    cm = df[numeric_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=50, ha="right")
    ax.set_yticks(range(len(cm)), cm.columns)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.iloc[i, j]
            if abs(v) >= .45:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > .7 else "black")
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Numeric Feature Correlations  (|r| ≥ 0.45 labelled)")
    writer.save(fig, "quality", "02_numeric_correlation_heatmap.png")

    # ── 03 Replication / duplicate audit ─────────────────────────────────
    exact_cols = ["material", "geometry", "o2", "diluent",
                  "pressure", "flow", "gravity", "ignition_method"]
    cond  = df.groupby(exact_cols, dropna=False, observed=True).size()
    rep   = int(cond[cond > 1].sum())
    uniq  = len(cond)
    dup   = int(df.duplicated(
        subset=[COL[k] for k in ["paper", "material", "o2", "pressure",
                                 "flow", "gravity", "fsr"]],
        keep=False).sum())
    metrics = pd.Series({
        "Total rows":                      len(df),
        "Unique experimental conditions":  uniq,
        "Rows in repeated conditions":     rep,
        "Exact source/target duplicates":  dup,
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics.index, metrics.values,
           color=["#2563eb", "#0f766e", "#f59e0b", "#dc2626"])
    annotate_bars(ax)
    ax.set(ylabel="Rows / groups",
           title="Replication and Potential Duplication Audit")
    ax.tick_params(axis="x", rotation=16)
    writer.save(fig, "quality", "03_replication_duplicate_audit.png")

    # ── 04 Per-paper modelling readiness ─────────────────────────────────
    core = ["material", "geometry", "fuel_density", "fuel_k", "fuel_cp",
            "o2", "pressure", "gravity", "facility", "ignition_method",
            "reynolds", "peclet"]
    paper = df.groupby("paper_id", observed=True).agg(
        rows=("paper", "size"),
        fsr=("fsr", "count"),
        materials=("material", "nunique"),
    )
    paper["core_completeness"] = (
        df[core].notna()
        .groupby(df["paper_id"], observed=True).mean()
        .mean(axis=1) * 100)
    paper["fsr_share"] = 100 * paper["fsr"] / paper["rows"]
    paper = paper.sort_values("rows", ascending=False).head(35)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(paper["core_completeness"], paper["fsr_share"],
                    s=25 + paper["rows"] * .22,
                    c=paper["materials"], cmap="viridis",
                    alpha=.75, edgecolor="white")
    for name, row in paper.nlargest(10, "rows").iterrows():
        ax.annotate(name, (row["core_completeness"], row["fsr_share"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Mean core-feature completeness (%)",
           ylabel="Rows with numeric FSR (%)",
           title="Source Readiness for FSR Modelling  (35 largest papers)")
    fig.colorbar(sc, ax=ax, label="Distinct materials; bubble size = rows")
    writer.save(fig, "quality", "04_paper_modeling_readiness.png")

    # ── 05 Effective source diversity (ESS) ───────────────────────────────
    def ess(sizes: np.ndarray) -> float:
        w = sizes / sizes.sum()
        return float(1 / np.square(w).sum())

    all_sizes = df.groupby("paper", observed=True).size().to_numpy(float)
    fsr_sizes = (df[df["fsr"].notna()]
                 .groupby("paper", observed=True).size().to_numpy(float))
    actual    = [len(all_sizes), len(fsr_sizes)]
    effective = [ess(all_sizes),  ess(fsr_sizes)]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, actual,    w, label="Observed papers",              color="#94a3b8")
    ax.bar(x + w / 2, effective, w, label="Effective papers (row-balanced)", color="#dc2626")
    ax.set_xticks(x, ["Full database", "Numeric FSR subset"])
    ax.set(ylabel="Number of papers",
           title="Effective Source Diversity  (ESS: equal-weighted equivalent)")
    ax.legend()
    annotate_bars(ax, fmt=".1f")
    writer.save(fig, "quality", "05_effective_source_diversity.png")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main() -> int:
    args   = parse_args()
    df     = load_database(args.input.resolve())
    writer = PlotWriter(args.output, args.clean)

    print("\n── Overview ──────────────────────────────────────────")
    overview_metrics(df, writer)
    print("── Parameter coverage ────────────────────────────────")
    coverage_metrics(df, writer)
    print("── Ignition metrics ──────────────────────────────────")
    ignition_metrics(df, writer)
    print("── FSR metrics ───────────────────────────────────────")
    fsr_metrics(df, writer)
    print("── Data quality ──────────────────────────────────────")
    quality_metrics(df, writer)

    print(f"\nDone — {len(writer.saved)} PNG files written to {writer.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())