#!/usr/bin/env python3
"""Generate categorized exploratory metrics for Microgravity_Database.csv.

The upgraded database uses SI units and exposes material, rig, internal geometry,
dimensionless flow properties, ignition energy inputs, and richer outputs.  This
script validates that schema, cleans known label inconsistencies, and writes PNG
figures into five topic folders beside the database.

Run from any directory:
    python metrics/generate_metrics.py
    python metrics/generate_metrics.py --clean
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "Microgravity_Database.csv"
OUTPUT_FOLDERS = {
    "overview": "01_overview",
    "coverage": "02_parameter_coverage",
    "ignition": "03_ignition_metrics",
    "fsr": "04_fsr_metrics",
    "quality": "05_data_quality",
}

COL = {
    "paper": "Article (MLA)",
    "authors": "Authors",
    "doi": "DOI",
    "geometry": "Geometry of Sample",
    "material": "Material of sample",
    "half_thickness": "half_thickness",
    "length": "characteristic_length_m",
    "fuel_density": "fuel_density_kg_m3",
    "fuel_k": "fuel_k_W_mK",
    "fuel_cp": "fuel_cp_J_kgK",
    "fuel_pyrolysis_t": "fuel_pyrolysis_T_K",
    "fuel_alpha": "fuel_alpha_m2_s",
    "o2": "Oxygen Concentration",
    "diluent": "diluent",
    "gas_density": "gas_density_kg_m3",
    "reynolds": "Reynolds",
    "peclet": "Peclet",
    "prandtl": "Prandtl",
    "diffusion_time": "thermal_diffusion_time",
    "pressure": "Pressure (Pa)",
    "flow": "Flow Velocity (m/s)",
    "rig": "Rig Name",
    "internal_geometry": "Internal Geometry",
    "gravity": "Gravity",
    "facility": "Facility",
    "ignition_method": "Ignition Method",
    "ignition_power": "Ignition Power (W)",
    "ignition_time": "Ignition Time (s)",
    "ignition": "Ignition",
    "flame_length": "Flame Length",
    "fsr": "FSR (m/s)",
    "hrr": "HRR",
    "smoke": "Smoke/ Areosols",
}

NUMERIC_KEYS = [
    "half_thickness", "length", "fuel_density", "fuel_k", "fuel_cp",
    "fuel_pyrolysis_t", "fuel_alpha", "o2", "gas_density", "reynolds",
    "peclet", "prandtl", "diffusion_time", "pressure", "flow", "gravity",
    "ignition_power", "ignition_time", "hrr",
]

IGNITION_COLORS = {"Yes": "#24935b", "No": "#d1495b"}
GRAVITY_ORDER = ["Microgravity (<=0.01 g)", "Partial gravity", "Earth gravity", "Hypergravity"]
GRAVITY_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
PALETTE = ["#2563eb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 9.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfdff",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="upgraded CSV path")
    parser.add_argument("--output", type=Path, default=BASE_DIR, help="parent output directory")
    parser.add_argument("--clean", action="store_true", help="remove old categorized output first")
    return parser.parse_args()


def clean_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def parse_length_m(value: object) -> float:
    """Parse the mixed-unit Flame Length field to metres without guessing bare units."""
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
        magnitude = float(parts[0])
    except ValueError:
        return np.nan
    return magnitude * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}.get(parts[1], np.nan)


def load_database(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    df = pd.read_csv(path, header=1, encoding="cp1252", low_memory=False).dropna(how="all")
    missing = sorted(set(COL.values()) - set(df.columns))
    if missing:
        raise ValueError("Upgraded database schema mismatch; missing columns: " + ", ".join(missing))

    for key in NUMERIC_KEYS:
        df[key] = pd.to_numeric(df[COL[key]], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["fsr"] = pd.to_numeric(df[COL["fsr"]], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["flame_length_m"] = df[COL["flame_length"]].map(parse_length_m)

    df["paper"] = clean_text(df[COL["paper"]])
    df["authors"] = clean_text(df[COL["authors"]])
    df["material"] = clean_text(df[COL["material"]])
    df["geometry"] = clean_text(df[COL["geometry"]]).str.title()
    df["facility"] = clean_text(df[COL["facility"]]).str.title()
    df["diluent"] = clean_text(df[COL["diluent"]])
    df["rig"] = clean_text(df[COL["rig"]])
    df["internal_geometry"] = clean_text(df[COL["internal_geometry"]]).str.title()
    df["ignition_method"] = clean_text(df[COL["ignition_method"]]).str.title()
    df["ignition"] = clean_text(df[COL["ignition"]]).str.title()
    df.loc[~df["ignition"].isin(["Yes", "No"]), "ignition"] = pd.NA
    df["ignition_binary"] = df["ignition"].map({"No": 0.0, "Yes": 1.0})

    df["pressure_kpa"] = df["pressure"] / 1_000.0
    df["fsr_mm_s"] = df["fsr"] * 1_000.0
    df["abs_flow"] = df["flow"].abs()
    df["ignition_energy_j"] = df["ignition_power"] * df["ignition_time"]
    df["gravity_regime"] = pd.cut(
        df["gravity"], [-np.inf, 0.01, 0.9, 1.1, np.inf], labels=GRAVITY_ORDER,
        include_lowest=True,
    )

    first_author = df["authors"].fillna(df["paper"]).str.split(",").str[0].str.strip()
    paper_codes, _ = pd.factorize(df["paper"], sort=True)
    df["paper_id"] = [f"P{code + 1:02d} {author[:18]}" for code, author in zip(paper_codes, first_author)]
    return df


class PlotWriter:
    def __init__(self, root: Path, clean: bool = False) -> None:
        self.root = root.resolve()
        self.saved: list[Path] = []
        if clean:
            # Flat PNGs are legacy outputs from the pre-upgrade schema.  New
            # outputs always live in categorized folders.
            for path in self.root.glob("*.png"):
                path.unlink()
            for folder in OUTPUT_FOLDERS.values():
                path = self.root / folder
                if path.exists():
                    shutil.rmtree(path)
        for folder in OUTPUT_FOLDERS.values():
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def save(self, fig: plt.Figure, category: str, filename: str) -> None:
        path = self.root / OUTPUT_FOLDERS[category] / filename
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.saved.append(path)
        print(f"saved {path.relative_to(self.root)}")


def annotate_bars(ax: plt.Axes, fmt: str = ",.0f") -> None:
    for container in ax.containers:
        labels = [format(bar.get_height(), fmt) for bar in container]
        ax.bar_label(container, labels=labels, padding=2, fontsize=8)


def safe_rate_table(df: pd.DataFrame, group: str, min_n: int = 20) -> pd.DataFrame:
    table = df.groupby(group, observed=True)["ignition_binary"].agg(["mean", "count"])
    return table[table["count"] >= min_n].sort_values("mean")


def overview_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    n, papers = len(df), df["paper"].nunique()
    numeric_fsr = df["fsr"].notna().sum()
    paper_sizes = df.groupby("paper", observed=True).size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6.5)); ax.axis("off")
    stats = [
        ("Experimental rows", f"{n:,}"), ("Papers", f"{papers}"),
        ("Distinct materials", f"{df['material'].nunique()}"), ("Facilities / rigs", f"{df['facility'].nunique()} / {df['rig'].nunique()}"),
        ("Ignition rate", f"{100 * df['ignition_binary'].mean():.1f}%"), ("Numeric FSR targets", f"{numeric_fsr:,} ({100*numeric_fsr/n:.1f}%)"),
        ("O2 range", f"{df['o2'].min():.3f} - {df['o2'].max():.3f}"), ("Pressure range", f"{df['pressure_kpa'].min():.1f} - {df['pressure_kpa'].max():,.0f} kPa"),
        ("Gravity range", f"{df['gravity'].min():.3g} - {df['gravity'].max():.3g} g"), ("Flow range", f"{df['flow'].min():.2f} - {df['flow'].max():.2f} m/s"),
        ("FSR range", f"{df['fsr_mm_s'].min():.3g} - {df['fsr_mm_s'].max():.0f} mm/s"), ("Median rows / paper", f"{paper_sizes.median():.0f}"),
    ]
    for i, (label, value) in enumerate(stats):
        row, col = divmod(i, 3); x, y = 0.03 + col * 0.33, 0.88 - row * 0.21
        ax.text(x, y, label, weight="bold", color="#334155", transform=ax.transAxes)
        ax.text(x, y - 0.075, value, fontsize=16, color=PALETTE[col], transform=ax.transAxes)
    ax.set_title("Upgraded Microgravity Database — Dataset Overview", fontsize=17, pad=20)
    writer.save(fig, "overview", "01_dataset_overview.png")

    top = paper_sizes.head(25).sort_values()
    fig, ax = plt.subplots(figsize=(11, 8))
    top.plot.barh(ax=ax, color="#2563eb")
    ax.set(xlabel="Rows", ylabel="", title="Largest Paper Contributions (top 25)")
    annotate_bars(ax)
    writer.save(fig, "overview", "02_largest_paper_contributions.png")

    cumulative = paper_sizes.cumsum() / n * 100
    n50 = int((cumulative < 50).sum() + 1); n80 = int((cumulative < 80).sum() + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", ms=3, color="#0f766e")
    for share, count in [(50, n50), (80, n80)]:
        ax.axhline(share, color="#dc2626", ls="--", alpha=.5)
        ax.axvline(count, color="#dc2626", ls=":", alpha=.5, label=f"{count} papers produce {share}%")
    ax.set(xlabel="Papers, largest contribution first", ylabel="Cumulative share of rows (%)", title="Dataset Concentration Across Sources")
    ax.legend()
    writer.save(fig, "overview", "03_paper_pareto_concentration.png")

    fields = {
        "Paper": "paper", "Material": "material", "Geometry": "geometry", "Fuel properties": "fuel_density",
        "Oxygen": "o2", "Pressure": "pressure", "Flow velocity": "flow", "Gravity": "gravity",
        "Facility": "facility", "Rig": "rig", "Ignition method": "ignition_method", "Ignition power": "ignition_power",
        "Ignition time": "ignition_time", "Ignition outcome": "ignition", "FSR target": "fsr",
        "Flame length": "flame_length_m", "HRR": "hrr", "Smoke": COL["smoke"],
    }
    completeness = pd.Series({label: 100 * df[col].notna().mean() for label, col in fields.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#dc2626" if v < 25 else "#f59e0b" if v < 75 else "#0f766e" for v in completeness]
    ax.barh(completeness.index, completeness.values, color=colors)
    for i, value in enumerate(completeness): ax.text(value + .7, i, f"{value:.1f}%", va="center", fontsize=8)
    ax.set(xlim=(0, 108), xlabel="Rows populated (%)", title="Field Completeness in the Upgraded Schema")
    writer.save(fig, "overview", "04_field_completeness.png")

    targets = df.groupby("paper_id", observed=True).agg(rows=("paper", "size"), fsr=("fsr", "count"), ignition=("ignition", "count"))
    targets["FSR available"] = 100 * targets["fsr"] / targets["rows"]
    targets["Ignition available"] = 100 * targets["ignition"] / targets["rows"]
    targets = targets.sort_values("rows", ascending=False).head(30).sort_values("rows")
    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(targets)); ax.barh(y, targets["Ignition available"], color="#94a3b8", label="Ignition")
    ax.barh(y, targets["FSR available"], color="#7c3aed", label="Numeric FSR")
    ax.set_yticks(y, targets.index, fontsize=7)
    ax.set(xlabel="Target availability within paper (%)", title="Outcome Availability by Major Data Source")
    ax.legend()
    writer.save(fig, "overview", "05_target_availability_by_paper.png")


def coverage_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (col, title) in zip(axes.flat, [
        ("facility", "Facility"), ("geometry", "Sample geometry"), ("diluent", "Diluent"),
        ("internal_geometry", "Rig internal geometry"), ("ignition_method", "Ignition method"), ("gravity_regime", "Gravity regime"),
    ]):
        vc = df[col].astype("string").fillna("Unknown").value_counts().head(10).sort_values()
        ax.barh(vc.index, vc.values, color="#0891b2"); ax.set_title(title); ax.set_xlabel("Rows")
    fig.suptitle("Experimental Composition in the Upgraded Database", fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "coverage", "01_experimental_composition.png")

    specs = [
        ("o2", "Oxygen mole fraction", False), ("pressure_kpa", "Pressure (kPa)", True),
        ("gravity", "Gravity (g)", False), ("flow", "Flow velocity (m/s)", False),
        ("length", "Characteristic length (m)", True), ("half_thickness", "Half-thickness (m)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (col, label, log) in zip(axes.flat, specs):
        values = df[col].dropna(); values = values[values > 0] if log else values
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 35) if log else 35
        ax.hist(values, bins=bins, color="#2563eb", alpha=.82, edgecolor="white")
        if log: ax.set_xscale("log")
        ax.set_title(f"{label}\nn={len(values):,}, unique={values.nunique():,}")
        ax.set_xlabel(label); ax.set_ylabel("Rows")
    fig.suptitle("Core Experimental Parameter Coverage", fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "coverage", "02_core_parameter_distributions.png")

    sub = df.dropna(subset=["o2", "pressure_kpa"]); sub = sub[sub["pressure_kpa"] > 0]
    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(sub["o2"], sub["pressure_kpa"], gridsize=42, yscale="log", mincnt=1, norm=LogNorm(), cmap="viridis")
    fig.colorbar(hb, ax=ax, label="Rows per cell (log)")
    ax.set(xlabel="Oxygen mole fraction", ylabel="Pressure (kPa, log)", title="Joint Coverage Density: Oxygen × Pressure")
    writer.save(fig, "coverage", "03_oxygen_pressure_coverage.png")

    sub = df.dropna(subset=["gravity", "flow", "facility"])
    fig, ax = plt.subplots(figsize=(10, 7))
    facilities = sub["facility"].value_counts().head(7).index
    for color, facility in zip(PALETTE + ["#64748b"], facilities):
        s = sub[sub["facility"] == facility]
        ax.scatter(s["gravity"], s["flow"], s=16, alpha=.45, color=color, label=f"{facility} (n={len(s):,})")
    ax.axhline(0, color="black", lw=.8); ax.set(xlabel="Gravity (g)", ylabel="Signed flow velocity (m/s)", title="Gravity × Flow Coverage by Facility")
    ax.legend(fontsize=8, loc="best")
    writer.save(fig, "coverage", "04_gravity_flow_by_facility.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, col, label in zip(axes, ["reynolds", "peclet", "prandtl"], ["Reynolds", "Peclet", "Prandtl"]):
        values = df[col].dropna(); values = values[values > 0]
        ax.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), 40), color="#7c3aed", edgecolor="white")
        ax.set_xscale("log"); ax.set_title(f"{label} (n={len(values):,})"); ax.set_xlabel(f"{label} number (log)")
    fig.suptitle("Dimensionless Transport-Regime Coverage", fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "coverage", "05_dimensionless_transport_coverage.png")

    material = df.groupby("material", observed=True).agg(rows=("paper", "size"), papers=("paper", "nunique"), o2_span=("o2", lambda x: x.max()-x.min()), pressure_span=("pressure_kpa", lambda x: x.max()/x.min() if x.min() > 0 else np.nan))
    material = material[material["rows"] >= 20].sort_values("rows").tail(25)
    fig, ax = plt.subplots(figsize=(11, 8))
    sizes = 30 + 20 * material["papers"]
    sc = ax.scatter(material["o2_span"], material["pressure_span"], s=sizes, c=material["rows"], cmap="plasma", norm=LogNorm(), alpha=.8, edgecolor="white")
    for name, row in material.tail(12).iterrows(): ax.annotate(str(name)[:28], (row["o2_span"], row["pressure_span"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_yscale("log"); ax.set(xlabel="O2 span", ylabel="Pressure range ratio (max/min, log)", title="Material Test-Space Breadth")
    fig.colorbar(sc, ax=ax, label="Rows (log color); bubble size = papers")
    writer.save(fig, "coverage", "06_material_test_space_breadth.png")

    gap_specs = [("o2", "O2", False), ("pressure_kpa", "Pressure kPa", True), ("gravity", "Gravity g", False), ("flow", "Flow m/s", False), ("fsr_mm_s", "FSR mm/s", True)]
    lines = []
    for col, label, logarithmic in gap_specs:
        u = np.sort(df[col].dropna().unique()); u = u[u > 0] if logarithmic else u
        gaps = np.diff(np.log10(u)) if logarithmic else np.diff(u)
        for idx in np.argsort(gaps)[-3:][::-1]:
            score = 10 ** gaps[idx] if logarithmic else gaps[idx]
            descriptor = f"x{score:.2g}" if logarithmic else f"gap {score:.3g}"
            lines.append((label, u[idx], u[idx+1], descriptor))
    fig, ax = plt.subplots(figsize=(11, 7)); ax.axis("off")
    ax.set_title("Largest One-Dimensional Sampling Gaps", fontsize=15)
    ax.text(.02, .94, "Variable", weight="bold"); ax.text(.26, .94, "Lower bound"); ax.text(.48, .94, "Upper bound"); ax.text(.72, .94, "Gap size", weight="bold")
    for i, (label, low, high, desc) in enumerate(lines):
        y=.89-i*.052; ax.text(.02,y,label); ax.text(.26,y,f"{low:.5g}",family="monospace"); ax.text(.48,y,f"{high:.5g}",family="monospace"); ax.text(.72,y,desc,color="#dc2626")
    ax.text(.02,.03,"Log-scaled variables report multiplicative gaps; others report absolute gaps.",style="italic",color="#475569")
    writer.save(fig, "coverage", "07_largest_sampling_gaps.png")


def ignition_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    counts = df["ignition"].value_counts().reindex(["Yes", "No"]).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].pie(counts, labels=[f"{x}: {int(v):,}\n({100*v/counts.sum():.1f}%)" for x,v in counts.items()], colors=[IGNITION_COLORS[x] for x in counts.index], startangle=90, wedgeprops={"edgecolor":"white"})
    axes[0].set_title("Pooled ignition outcomes")
    per_paper = df.groupby("paper", observed=True)["ignition_binary"].mean().dropna()
    axes[1].hist(per_paper * 100, bins=np.arange(0, 105, 5), color="#7c3aed", edgecolor="white")
    axes[1].axvline(100*per_paper.mean(), color="#dc2626", ls="--", label=f"paper-balanced mean={100*per_paper.mean():.1f}%")
    axes[1].set(xlabel="Ignition rate within paper (%)", ylabel="Papers", title="Source-level outcome heterogeneity"); axes[1].legend(fontsize=8)
    fig.suptitle("Ignition Outcome Balance: Rows vs Sources", fontsize=15, weight="bold")
    fig.tight_layout()
    writer.save(fig, "ignition", "01_outcome_balance.png")

    for number, (group, title) in enumerate([("facility", "Facility"), ("geometry", "Sample Geometry"), ("ignition_method", "Ignition Method")], start=2):
        table = safe_rate_table(df, group).sort_values("mean")
        fig, ax = plt.subplots(figsize=(10, max(5, .55*len(table)+1.5)))
        colors = plt.cm.RdYlGn(table["mean"].to_numpy())
        ax.barh(table.index.astype(str), 100*table["mean"], color=colors)
        for i, (_, row) in enumerate(table.iterrows()): ax.text(100*row["mean"]+1, i, f"{100*row['mean']:.1f}%  n={int(row['count']):,}", va="center", fontsize=8)
        ax.set(xlim=(0,115), xlabel="Ignition rate (%)", title=f"Ignition Rate by {title} (groups with n >= 20)")
        writer.save(fig, "ignition", f"{number:02d}_ignition_rate_by_{group}.png")

    # Pooled rates can be dominated by large papers; compare them with equal-paper weighting.
    rows = []
    for facility, part in df.groupby("facility", observed=True):
        source_rates = part.groupby("paper", observed=True)["ignition_binary"].mean().dropna()
        if len(part) >= 30 and len(source_rates) >= 2:
            rows.append((facility, part["ignition_binary"].mean(), source_rates.mean(), len(part), len(source_rates)))
    balance = pd.DataFrame(rows, columns=["facility", "pooled", "paper_balanced", "rows", "papers"]).sort_values("pooled")
    fig, ax = plt.subplots(figsize=(10, 6)); y=np.arange(len(balance))
    ax.plot(100*balance["pooled"], y, "o", ms=8, label="Pooled rows", color="#2563eb")
    ax.plot(100*balance["paper_balanced"], y, "s", ms=7, label="Each paper weighted equally", color="#f59e0b")
    for i, row in balance.iterrows(): ax.plot([100*row["pooled"],100*row["paper_balanced"]],[i,i],color="#94a3b8",zorder=0)
    ax.set_yticks(y, [f"{x.facility} (p={x.papers}, n={x.rows})" for x in balance.itertuples()]); ax.set(xlabel="Ignition rate (%)",title="Facility Ignition Rate: Pooled vs Paper-Balanced")
    ax.legend()
    writer.save(fig, "ignition", "05_pooled_vs_paper_balanced_facility.png")

    valid = df.dropna(subset=["o2", "pressure_kpa", "ignition_binary"]); valid=valid[valid["pressure_kpa"]>0].copy()
    valid["o2_bin"] = pd.cut(valid["o2"], bins=np.linspace(valid["o2"].min(), valid["o2"].max(), 18))
    valid["p_bin"] = pd.cut(np.log10(valid["pressure_kpa"]), bins=14)
    rates = valid.pivot_table(index="p_bin", columns="o2_bin", values="ignition_binary", aggfunc="mean", observed=True)
    counts2 = valid.pivot_table(index="p_bin", columns="o2_bin", values="ignition_binary", aggfunc="count", observed=True)
    rates = rates.where(counts2 >= 5)
    fig, ax = plt.subplots(figsize=(12, 7))
    im=ax.imshow(rates.to_numpy(),origin="lower",aspect="auto",vmin=0,vmax=1,cmap="RdYlGn")
    xcent=[interval.mid for interval in rates.columns]; ycent=[10**interval.mid for interval in rates.index]
    ax.set_xticks(np.arange(len(xcent))[::2], [f"{x:.2f}" for x in xcent[::2]])
    ax.set_yticks(np.arange(len(ycent))[::2], [f"{y:.1f}" for y in ycent[::2]])
    ax.set(xlabel="Oxygen mole-fraction bin",ylabel="Pressure bin center (kPa, log-spaced)",title="Empirical Ignition Probability in O2 × Pressure Space (cell n >= 5)")
    fig.colorbar(im,ax=ax,label="Observed ignition probability")
    writer.save(fig, "ignition", "06_oxygen_pressure_ignition_map.png")

    energy=df.dropna(subset=["ignition_power","ignition_time","ignition"])
    fig, ax=plt.subplots(figsize=(10,7))
    for outcome,color in IGNITION_COLORS.items():
        s=energy[energy["ignition"]==outcome]
        ax.scatter(s["ignition_time"],s["ignition_power"],s=20,alpha=.5,color=color,label=f"{outcome} (n={len(s):,})")
    ax.set(xlabel="Ignition time (s)",ylabel="Ignition power (W)",title="Igniter Operating Envelope and Outcome"); ax.legend()
    writer.save(fig,"ignition","07_ignition_power_time_envelope.png")

    table=safe_rate_table(df,"material",min_n=30)
    table=table.sort_values("count",ascending=False).head(25).sort_values("mean")
    fig,ax=plt.subplots(figsize=(11,8)); y=np.arange(len(table))
    rate_colors = plt.cm.RdYlGn(table["mean"].to_numpy(copy=True))
    ax.barh(y,100*table["mean"],color=rate_colors)
    ax.set_yticks(y,[str(name)[:48] for name in table.index],fontsize=7)
    for i,(_,row) in enumerate(table.iterrows()): ax.text(100*row["mean"]+1,i,f"{100*row['mean']:.0f}% (n={int(row['count'])})",va="center",fontsize=7)
    ax.set(xlim=(0,118),xlabel="Ignition rate (%)",title="Ignition Rate for Best-Represented Materials")
    writer.save(fig,"ignition","08_ignition_rate_by_material.png")


def fsr_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    raw_present=df[COL["fsr"]].notna(); numeric=df["fsr"].notna(); positive=df["fsr"]>0
    labels=["Raw values present","Numeric SI values","Positive regression targets"]
    values=[raw_present.sum(),numeric.sum(),positive.sum()]
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(labels,values,color=["#94a3b8","#7c3aed","#0f766e"]); annotate_bars(ax)
    ax.set(ylabel="Rows",title="FSR Target Parsing and Regression Readiness"); ax.tick_params(axis="x",rotation=10)
    note=int((raw_present & ~numeric).sum()); ax.text(.98,.94,f"Excluded descriptive ratios: {note}",ha="right",va="top",transform=ax.transAxes,color="#dc2626")
    writer.save(fig,"fsr","01_fsr_target_readiness.png")

    fs=df[df["fsr_mm_s"]>0].copy()
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(fs["fsr_mm_s"],bins=np.logspace(np.log10(fs["fsr_mm_s"].min()),np.log10(fs["fsr_mm_s"].max()),45),color="#7c3aed",edgecolor="white")
    axes[0].set_xscale("log"); axes[0].set(xlabel="FSR (mm/s, log)",ylabel="Rows",title="Positive FSR distribution")
    paper_medians=fs.groupby("paper",observed=True)["fsr_mm_s"].median()
    axes[1].hist(paper_medians,bins=np.logspace(np.log10(paper_medians.min()),np.log10(paper_medians.max()),28),color="#0891b2",edgecolor="white")
    axes[1].set_xscale("log"); axes[1].set(xlabel="Median FSR per paper (mm/s, log)",ylabel="Papers",title="Paper-balanced target distribution")
    fig.suptitle("FSR Distribution: Row-Level vs Source-Level",fontsize=15,weight="bold"); fig.tight_layout()
    writer.save(fig,"fsr","02_fsr_distribution_row_vs_paper.png")

    sub=fs.dropna(subset=["o2","pressure_kpa"]); sub=sub[sub["pressure_kpa"]>0]
    fig,ax=plt.subplots(figsize=(10,7)); sc=ax.scatter(sub["o2"],sub["fsr_mm_s"],c=sub["pressure_kpa"],norm=LogNorm(),cmap="plasma",s=18,alpha=.55)
    ax.set_yscale("log"); ax.set(xlabel="Oxygen mole fraction",ylabel="FSR (mm/s, log)",title="Flame Spread Rate vs Oxygen and Pressure"); fig.colorbar(sc,ax=ax,label="Pressure (kPa, log)")
    writer.save(fig,"fsr","03_fsr_vs_oxygen_pressure.png")

    sub=fs.dropna(subset=["gravity","flow"])
    fig,ax=plt.subplots(figsize=(10,7)); sc=ax.scatter(sub["gravity"],sub["fsr_mm_s"],c=sub["flow"],cmap="coolwarm",s=18,alpha=.55)
    ax.set_yscale("log"); ax.set(xlabel="Gravity (g)",ylabel="FSR (mm/s, log)",title="Flame Spread Rate vs Gravity and Signed Flow"); fig.colorbar(sc,ax=ax,label="Flow velocity (m/s)")
    writer.save(fig,"fsr","04_fsr_vs_gravity_flow.png")

    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,col,label in [(axes[0],"reynolds","Reynolds"),(axes[1],"peclet","Peclet")]:
        sub=fs.dropna(subset=[col]); sub=sub[sub[col]>0]
        ax.scatter(sub[col],sub["fsr_mm_s"],s=15,alpha=.4,color="#2563eb"); ax.set_xscale("log"); ax.set_yscale("log")
        corr=sub[[col,"fsr_mm_s"]].corr(method="spearman").iloc[0,1]
        ax.set(xlabel=f"{label} number (log)",ylabel="FSR (mm/s, log)",title=f"FSR vs {label} (Spearman r={corr:.2f})")
    fig.suptitle("FSR Across Dimensionless Transport Regimes",fontsize=15,weight="bold"); fig.tight_layout()
    writer.save(fig,"fsr","05_fsr_vs_dimensionless_transport.png")

    mat=fs.groupby("material",observed=True)["fsr_mm_s"].agg(["median","count",lambda x:x.quantile(.25),lambda x:x.quantile(.75)])
    mat.columns=["median","count","q25","q75"]; mat=mat[mat["count"]>=20].sort_values("median")
    fig,ax=plt.subplots(figsize=(11,max(6,.4*len(mat))))
    y=np.arange(len(mat)); ax.errorbar(mat["median"],y,xerr=[mat["median"]-mat["q25"],mat["q75"]-mat["median"]],fmt="o",color="#7c3aed",ecolor="#a78bfa",capsize=3)
    ax.set_xscale("log"); ax.set_yticks(y,[f"{str(name)[:48]} (n={int(row['count'])})" for name,row in mat.iterrows()],fontsize=7)
    ax.set(xlabel="Median FSR and interquartile range (mm/s, log)",title="Material-Level Flame Spread Summary (n >= 20)")
    writer.save(fig,"fsr","06_fsr_by_material.png")

    groups=[]; labels=[]
    for regime in GRAVITY_ORDER:
        vals=fs.loc[fs["gravity_regime"]==regime,"fsr_mm_s"].dropna()
        if len(vals)>=10: groups.append(vals); labels.append(f"{regime}\n(n={len(vals):,})")
    fig,ax=plt.subplots(figsize=(11,6)); ax.boxplot(groups,tick_labels=labels,showfliers=False); ax.set_yscale("log")
    ax.set(ylabel="FSR (mm/s, log)",title="FSR by Gravity Regime (outliers hidden)"); ax.tick_params(axis="x",rotation=10)
    writer.save(fig,"fsr","07_fsr_by_gravity_regime.png")

    variables=["o2","pressure_kpa","gravity","flow","reynolds","peclet","fuel_density","fuel_k","fuel_cp","length","half_thickness"]
    total=[]; within=[]
    log_fsr=np.log10(fs["fsr_mm_s"])
    for col in variables:
        total.append(fs[col].corr(log_fsr,method="spearman"))
        demeaned=fs[col]-fs.groupby("paper",observed=True)[col].transform("mean")
        within.append(demeaned.corr(log_fsr-fs.groupby("paper",observed=True)["fsr_mm_s"].transform(lambda x:np.log10(x).mean()),method="spearman"))
    fig,ax=plt.subplots(figsize=(11,6)); x=np.arange(len(variables)); width=.38
    ax.bar(x-width/2,total,width,label="Across all rows",color="#2563eb"); ax.bar(x+width/2,within,width,label="Within-paper (demeaned)",color="#f59e0b")
    ax.axhline(0,color="black",lw=.8); ax.set_xticks(x,variables,rotation=35,ha="right"); ax.set(ylabel="Spearman correlation with log10(FSR)",title="Total vs Within-Paper FSR Associations"); ax.legend()
    writer.save(fig,"fsr","08_total_vs_within_paper_associations.png")


def quality_metrics(df: pd.DataFrame, writer: PlotWriter) -> None:
    features=["fuel_density","fuel_k","fuel_cp","o2","pressure","flow","gravity","facility","ignition_method","reynolds","peclet","ignition_power","ignition_time","fsr"]
    available=df[features].notna().astype(float)
    matrix=(available.T @ available).to_numpy(dtype=float)
    denom=np.sqrt(np.outer(np.diag(matrix),np.diag(matrix))); similarity=np.divide(matrix,denom,out=np.zeros_like(matrix),where=denom>0)
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(similarity,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(len(features)),features,rotation=45,ha="right"); ax.set_yticks(range(len(features)),features)
    for i in range(len(features)):
        for j in range(len(features)): ax.text(j,i,f"{similarity[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if similarity[i,j]>.65 else "black")
    fig.colorbar(im,ax=ax,label="Cosine co-availability"); ax.set_title("Feature Co-Availability (can fields be modeled together?)")
    writer.save(fig,"quality","01_feature_coavailability.png")

    numeric=["half_thickness","length","fuel_density","fuel_k","fuel_cp","fuel_pyrolysis_t","fuel_alpha","o2","pressure_kpa","flow","gravity","reynolds","peclet","prandtl","diffusion_time","ignition_power","ignition_time","fsr_mm_s"]
    cm=df[numeric].corr(method="spearman")
    fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(cm,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(cm)),cm.columns,rotation=50,ha="right"); ax.set_yticks(range(len(cm)),cm.columns)
    for i in range(len(cm)):
        for j in range(len(cm)):
            value=cm.iloc[i,j]
            if abs(value)>=.45: ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=6,color="white" if abs(value)>.7 else "black")
    fig.colorbar(im,ax=ax,label="Spearman correlation"); ax.set_title("Numeric Feature Correlations (labels shown for |r| >= 0.45)")
    writer.save(fig,"quality","02_numeric_correlation_heatmap.png")

    exact_cols=["material","geometry","o2","diluent","pressure","flow","gravity","ignition_method"]
    condition_counts=df.groupby(exact_cols,dropna=False,observed=True).size()
    repeated_rows=int(condition_counts[condition_counts>1].sum()); unique_conditions=len(condition_counts)
    duplicate_rows=int(df.duplicated(subset=[COL[k] for k in ["paper","material","o2","pressure","flow","gravity","fsr"]],keep=False).sum())
    metrics=pd.Series({"Rows":len(df),"Unique experimental conditions":unique_conditions,"Rows in repeated conditions":repeated_rows,"Rows in exact source/target duplicates":duplicate_rows})
    fig,ax=plt.subplots(figsize=(10,5)); bars=ax.bar(metrics.index,metrics.values,color=["#2563eb","#0f766e","#f59e0b","#dc2626"]); annotate_bars(ax)
    ax.set(ylabel="Rows / groups",title="Replication and Potential Duplication Audit"); ax.tick_params(axis="x",rotation=16)
    writer.save(fig,"quality","03_replication_duplicate_audit.png")

    core=["material","geometry","fuel_density","fuel_k","fuel_cp","o2","pressure","gravity","facility","ignition_method","reynolds","peclet"]
    paper=df.groupby("paper_id",observed=True).agg(rows=("paper","size"),fsr=("fsr","count"),materials=("material","nunique"),o2_unique=("o2","nunique"),pressure_unique=("pressure","nunique"))
    paper["core_completeness"]=df[core].notna().groupby(df["paper_id"],observed=True).mean().mean(axis=1)*100
    paper["fsr_share"]=100*paper["fsr"]/paper["rows"]
    paper=paper.sort_values("rows",ascending=False).head(35)
    fig,ax=plt.subplots(figsize=(10,7)); sc=ax.scatter(paper["core_completeness"],paper["fsr_share"],s=25+paper["rows"]*.22,c=paper["materials"],cmap="viridis",alpha=.75,edgecolor="white")
    for name,row in paper.nlargest(10,"rows").iterrows(): ax.annotate(name,(row["core_completeness"],row["fsr_share"]),fontsize=6,xytext=(3,3),textcoords="offset points")
    ax.set(xlabel="Mean core-feature completeness (%)",ylabel="Rows with numeric FSR (%)",title="Source Readiness for FSR Modeling (35 largest papers)")
    fig.colorbar(sc,ax=ax,label="Distinct materials; bubble size = rows")
    writer.save(fig,"quality","04_paper_modeling_readiness.png")

    # Quantify source dominance with effective sample size from source weights.
    sizes=df.groupby("paper",observed=True).size().to_numpy(dtype=float); weights=sizes/sizes.sum(); ess=1/np.square(weights).sum()
    fs_sizes=df[df["fsr"].notna()].groupby("paper",observed=True).size().to_numpy(dtype=float); fs_weights=fs_sizes/fs_sizes.sum(); fs_ess=1/np.square(fs_weights).sum()
    actual=[len(sizes),len(fs_sizes)]; effective=[ess,fs_ess]
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(2); width=.36
    ax.bar(x-width/2,actual,width,label="Observed papers",color="#94a3b8"); ax.bar(x+width/2,effective,width,label="Effective papers after row imbalance",color="#dc2626")
    ax.set_xticks(x,["Full database","Numeric FSR subset"]); ax.set(ylabel="Number of papers",title="Effective Source Diversity"); ax.legend(); annotate_bars(ax,fmt=".1f")
    writer.save(fig,"quality","05_effective_source_diversity.png")


def main() -> int:
    args=parse_args()
    df=load_database(args.input.resolve())
    writer=PlotWriter(args.output,args.clean)
    overview_metrics(df,writer)
    coverage_metrics(df,writer)
    ignition_metrics(df,writer)
    fsr_metrics(df,writer)
    quality_metrics(df,writer)
    print(f"\nDone: generated {len(writer.saved)} PNG files in {writer.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
