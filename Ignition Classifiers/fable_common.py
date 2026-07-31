"""Canonical data loading and feature engineering for ignition classification."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

DATA_VERSION = "fable-data-v5"
ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_DIM = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(µm|μm|um|mm|cm|m)?", re.I)
RANDOM_STATE = 42

# Columns that must never be used as features and are dropped on load.
# Matching is case-insensitive on the stripped header.
DROPPED_COLUMNS = (
    "Material of Sample",
    "Rig Name",
    "Internal Geometry",
    "Internal Dimensions",
    "Facility",
    "Info",
)

# Evaluation output directory name -> model family, in Slurm array order.
MODEL_FAMILIES = {
    "xgb": "xgboost",
    "svm": "svm",
    "knn": "knn",
    "dt": "decision_tree",
    "mlp": "mlp",
}


def torch_device(local_rank: int | None = None) -> torch.device:
    """Return the available CUDA device, or CPU when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank
    return torch.device(f"cuda:{rank % torch.cuda.device_count()}")


def configure_torch(local_rank: int | None = None) -> torch.device:
    """Seed PyTorch reproducibly and select the shared compute device."""
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        device = torch_device(local_rank)
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def empty_cuda_cache() -> None:
    """Release unused CUDA allocations without failing on CPU-only hosts."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def slurm_num_workers() -> int:
    """Use the CPU allocation supplied by Slurm for PyTorch data loading."""
    return max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "0")))


COLUMNS = {
    "article": "Article (MLA)", "authors": "Authors", "doi": "DOI",
    "geometry": "Geometry of Sample", "dimensions": "Dimensions of sample",
    "half_thickness": "half_thickness", "characteristic_length": "characteristic_length_m",
    "fuel_density": "fuel_density_kg_m3", "fuel_k": "fuel_k_W_mK",
    "fuel_cp": "fuel_cp_J_kgK", "fuel_pyrolysis": "fuel_pyrolysis_T_K",
    "fuel_alpha": "fuel_alpha_m2_s",
    "fuel_volumetric_cp": "fuel_volumetric_heat_capacity_J_m3K",
    "core_density": "core_density_kg_m3",
    "core_k": "core_k_W_mK", "core_cp": "core_cp_J_kgK",
    "core_volumetric_cp": "core_volumetric_heat_capacity_J_m3K",
    "oxygen": "Oxygen Concentration", "diluent": "diluent", "gas_m": "gas_M",
    "gas_cp": "gas_cp_mass", "gas_k": "gas_k", "gas_density": "gas_density_kg_m3",
    "gas_alpha": "gas_alpha_m2_s", "gas_nu": "gas_nu_m2_s",
    "reynolds": "Reynolds", "peclet": "Peclet", "prandtl": "Prandtl",
    "thermal_diffusion_time": "thermal_diffusion_time",
    "pressure": "Pressure",
    "flow": "Flow Velocity",
    "gravity": "Gravity",
    "ignition_method": "Ignition Method",
    "ignition_power": "Ignition Power", "ignition_time": "Ignition Time",
    "target": "Ignition",
}
POST_OUTCOME_COLUMNS = ("Flame Length", "FSR", "HRR", "Smoke/ Areosols")

# Numeric features — all physical quantities already in SI / natural units in the DB:
#   dimensions in m, pressure in Pa, flow/FSR in m/s, gravity in g/g_earth.
NUMERIC_FEATURES = (
    "fuel_density_kg_m3", "fuel_k_w_mk", "fuel_cp_j_kgk", "fuel_pyrolysis_t_k",
    "fuel_alpha_m2_s", "fuel_volumetric_heat_capacity_j_m3k",
    "core_density_kg_m3", "core_k_w_mk", "core_cp_j_kgk",
    "core_volumetric_heat_capacity_j_m3k",
    "oxygen_fraction", "gas_molar_mass", "gas_cp_j_kgk", "gas_k_w_mk",
    "gas_density_kg_m3", "gas_alpha_m2_s", "gas_nu_m2_s",
    "reynolds", "peclet", "prandtl", "thermal_diffusion_time_s",
    "pressure_pa", "flow_velocity_m_s", "flow_speed_abs_m_s", "gravity_g",
    "ignition_power_w", "ignition_time_s",
    "half_thickness_m", "characteristic_length_m",
    "sample_dim_1_m", "sample_dim_2_m", "sample_dim_3_m",
)
CATEGORICAL_FEATURES = (
    "geometry_cat", "diluent_cat", "flow_direction_cat", "gravity_regime_cat",
    "ignition_method_cat",
)

# Pressure: DB already stores Pa — no conversion needed (defensive parsing still supported).
PRESSURE_TO_PA = {"pa": 1., "kpa": 1e3, "mpa": 1e6, "atm": 101325., "psi": 6894.757, "bar": 1e5}

# Flow velocity: DB already stores m/s — no conversion needed (defensive parsing still supported).
FLOW_TO_M_S = {"m/s": 1., "cm/s": 1e-2, "mm/s": 1e-3}


def _text(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip())
    return np.nan if value.lower() in {"", "-", "na", "n/a", "nan", "none"} else value


def _number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = _NUM.search(str(value).replace(",", ".").replace("\u2212", "-"))
    return float(match.group()) if match else np.nan


def _factor_to_m(unit: str) -> float:
    """Convert a dimension unit string to a factor that gives metres."""
    return {"\u00b5m": 1e-6, "\u03bcm": 1e-6, "um": 1e-6,
            "mm": 1e-3, "cm": 1e-2}.get((unit or "m").lower(), 1.0)


def _dimension_text(value: Any) -> str:
    return str(value).replace(",", ".").replace("\u00d7", "x").replace("\u00d8", " diameter ")


def dimensions_m(value: Any) -> list[float]:
    """Parse a dimensions cell and return a list of values in metres."""
    if pd.isna(value):
        return []
    return [float(number) * _factor_to_m(unit)
            for number, unit in _DIM.findall(_dimension_text(value))]


def header_unit_scale(header: str, scales: dict[str, float], fallback: str) -> float:
    """Scale factor for the unit declared in a column header, e.g. "Pressure (Pa)"."""
    match = re.search(r"\(([^)]*)\)", str(header))
    unit = (match.group(1) if match else fallback).strip().lower()
    if unit not in scales:
        raise ValueError(
            f"Unrecognized unit {unit!r} in column header {header!r}; "
            f"expected one of {sorted(scales)}")
    return scales[unit]


def pressure_pa(value: Any, header_scale: float = 1.) -> float:
    """Return pressure in Pa; unitless cells use the header unit scale."""
    number, text = _number(value), str(value).lower()
    if pd.isna(number):
        return np.nan
    for unit, scale in PRESSURE_TO_PA.items():
        if unit in text:
            return number * scale
    return number * header_scale


def flow_m_s(value: Any, header_scale: float = 1.) -> float:
    """Return flow velocity in m/s; unitless cells use the header unit scale."""
    number, text = _number(value), str(value).lower()
    if pd.isna(number):
        return np.nan
    for unit, scale in FLOW_TO_M_S.items():
        if unit in text:
            return number * scale
    return number * header_scale


def gravity_g(value: Any) -> float:
    text, number = str(value).lower().replace("\u00b2", "2"), _number(value)
    if pd.isna(number):
        return 1e-6 if any(x in text for x in ("micro", "\u00b5g", "\u03bcg")) else np.nan
    if "cm/s2" in text or "cm/s^2" in text: return number / 981
    if "mm/s2" in text or "mm/s^2" in text: return number / 9810
    if "m/s2" in text or "m/s^2" in text: return number / 9.81
    return number


def watts(value: Any) -> float:
    text = str(value).lower()
    return np.nan if any(x in text for x in ("w/cm", "kw/m", "amp", "current")) else _number(value)


def _category(value: Any, mapping: dict[str, tuple[str, ...]]) -> str:
    text = str(_text(value)).lower()
    if text == "nan":
        return "Unknown"
    for label, words in mapping.items():
        if any(word in text for word in words):
            return label
    return "Other"


def canonical_doi(value: Any) -> str | float:
    """Return a normalized true DOI; non-DOI URLs fall back to citation identity."""
    value = _text(value)
    if pd.isna(value):
        return np.nan
    text = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", str(value).lower()).rstrip("/. ")
    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text, re.I)
    return match.group().rstrip(".,;") if match else np.nan


def canonical_citation(value: Any) -> str | float:
    value = _text(value)
    if pd.isna(value):
        return np.nan
    text = str(value).lower().translate(str.maketrans({"\u201c": '"', "\u201d": '"', "\u2019": "'", "\u2013": "-", "\u2014": "-"}))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def _resolve(raw: pd.DataFrame, require_target: bool) -> dict[str, str]:
    """
    Resolve database columns robustly.

    Rules:
    1. Target ("Ignition") MUST match exactly.
    2. All other columns:
       - prefer exact match;
       - otherwise accept headers that start with the expected name
         (e.g. "Pressure (Pa)", "Ignition Power (W)").
    """
    found: dict[str, str] = {}
    inference_optional = {"article", "authors", "doi", "target"}

    # Map normalized header -> original header
    normalized = {
        str(c).strip().lower(): c
        for c in raw.columns
    }

    for key, wanted in COLUMNS.items():

        wanted_norm = wanted.strip().lower()

        # ------------------------------------------------------------------
        # TARGET: require an exact match ("Ignition")
        # ------------------------------------------------------------------
        if key == "target":
            if wanted_norm in normalized:
                found[key] = normalized[wanted_norm]
                continue

            if require_target:
                raise ValueError(
                    f"Required target column '{wanted}' not found.\n"
                    f"Available columns:\n{list(raw.columns)}"
                )
            continue

        # ------------------------------------------------------------------
        # EXACT MATCH
        # ------------------------------------------------------------------
        if wanted_norm in normalized:
            found[key] = normalized[wanted_norm]
            continue

        # ------------------------------------------------------------------
        # PREFIX MATCH
        # ------------------------------------------------------------------
        matches = [
            c for c in raw.columns
            if str(c).strip().lower().startswith(wanted_norm)
        ]

        if matches:
            # Prefer the shortest match
            matches.sort(key=len)
            found[key] = matches[0]
            continue

        # ------------------------------------------------------------------
        # OPTIONAL / REQUIRED
        # ------------------------------------------------------------------
        if require_target or key not in inference_optional:
            raise ValueError(
                f"Required database column {key!r} ({wanted!r}) is missing.\n"
                f"Available columns:\n{list(raw.columns)}"
            )

    return found


def read_raw(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    errors = []
    for encoding in ENCODINGS:
        try:
            raw = pd.read_csv(path, skiprows=1, encoding=encoding)
            raw.columns = [str(c).strip() for c in raw.columns]
            return raw, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Could not decode {path} with {ENCODINGS}: {'; '.join(errors)}")


def dataset_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_lists() -> tuple[list[str], list[str]]:
    """Return the numeric and categorical physics features used by every model."""
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)


def feature_manifest() -> list[dict[str, str]]:
    return ([{"feature": name, "kind": "numeric"} for name in NUMERIC_FEATURES] +
            [{"feature": name, "kind": "categorical"} for name in CATEGORICAL_FEATURES])


def load_data(path: str | Path, require_target: bool = True, deduplicate: bool = True
              ) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw, encoding = read_raw(path)
    original_rows = len(raw)
    # Drop disallowed columns before any further processing.
    disallowed = {name.strip().lower() for name in DROPPED_COLUMNS}
    raw = raw.drop(columns=[c for c in raw.columns
                             if str(c).strip().lower() in disallowed], errors="ignore")
    blank_mask = raw.replace(r"^\s*$", np.nan, regex=True).isna().all(axis=1)
    blank_source_rows = (raw.index[blank_mask] + 3).astype(int).tolist()
    raw = raw.loc[~blank_mask].copy()
    columns = _resolve(raw, require_target=require_target)
    raw = raw.drop(columns=[c for c in raw if c.startswith("Unnamed")], errors="ignore")
    for column in raw.select_dtypes(include="object"):
        raw[column] = raw[column].map(_text)
    df = pd.DataFrame(index=raw.index)
    df["source_row_number"] = raw.index + 3
    df["row_id"] = [f"row-{i:07d}" for i in df["source_row_number"]]
    empty = pd.Series(np.nan, index=raw.index)
    article_source = raw[columns["article"]] if "article" in columns else empty
    authors_source = raw[columns["authors"]] if "authors" in columns else empty
    doi_source = raw[columns["doi"]] if "doi" in columns else empty
    df["raw_citation"] = article_source
    df["raw_authors"] = authors_source
    df["raw_doi"] = doi_source
    doi = doi_source.map(canonical_doi)
    citation = article_source.map(canonical_citation)
    missing_identity = doi.isna() & citation.isna()
    if require_target and missing_identity.any():
        rows = df.loc[missing_identity, "source_row_number"].tolist()[:20]
        raise ValueError(f"Rows lack both canonical DOI and citation identity: {rows}")
    df["paper_id"] = doi.map(lambda x: f"doi::{x}" if pd.notna(x) else np.nan)
    df["paper_id"] = df["paper_id"].fillna(citation.map(lambda x: f"citation::{x}"))
    df.loc[df["paper_id"].isna(), "paper_id"] = df.loc[
        df["paper_id"].isna(), "row_id"].map(lambda x: f"unknown::{x}")
    df["paper_label"] = df["raw_citation"].fillna(df["paper_id"]).astype(str).str.slice(0, 100)

    source_to_feature = {
        "fuel_density": "fuel_density_kg_m3", "fuel_k": "fuel_k_w_mk",
        "fuel_cp": "fuel_cp_j_kgk", "fuel_pyrolysis": "fuel_pyrolysis_t_k",
        "fuel_alpha": "fuel_alpha_m2_s",
        "fuel_volumetric_cp": "fuel_volumetric_heat_capacity_j_m3k",
        "core_density": "core_density_kg_m3",
        "core_k": "core_k_w_mk", "core_cp": "core_cp_j_kgk",
        "core_volumetric_cp": "core_volumetric_heat_capacity_j_m3k",
        "gas_m": "gas_molar_mass", "gas_cp": "gas_cp_j_kgk", "gas_k": "gas_k_w_mk",
        "gas_density": "gas_density_kg_m3", "gas_alpha": "gas_alpha_m2_s",
        "gas_nu": "gas_nu_m2_s", "reynolds": "reynolds", "peclet": "peclet",
        "prandtl": "prandtl", "thermal_diffusion_time": "thermal_diffusion_time_s",
        "half_thickness": "half_thickness_m",
        "characteristic_length": "characteristic_length_m",
    }
    for source, feature in source_to_feature.items():
        df[feature] = pd.to_numeric(raw[columns[source]], errors="coerce")

    # Oxygen is already stored as a fraction in the DB — read directly.
    df["oxygen_fraction"] = pd.to_numeric(raw[columns["oxygen"]], errors="coerce")

    # Pressure already in Pa in the DB; still handle explicit unit labels defensively.
    pressure_scale = header_unit_scale(columns["pressure"], PRESSURE_TO_PA, "pa")
    df["pressure_pa"] = raw[columns["pressure"]].map(
        lambda x: pressure_pa(x, pressure_scale))

    # Flow velocity already in m/s in the DB; still handle explicit unit labels defensively.
    flow_scale = header_unit_scale(columns["flow"], FLOW_TO_M_S, "m/s")
    df["flow_velocity_m_s"] = raw[columns["flow"]].map(lambda x: flow_m_s(x, flow_scale))
    df["flow_speed_abs_m_s"] = df["flow_velocity_m_s"].abs()

    # Gravity already in g/g_earth in the DB.
    df["gravity_g"] = raw[columns["gravity"]].map(gravity_g)

    df["ignition_power_w"] = raw[columns["ignition_power"]].map(watts)
    df["ignition_time_s"] = raw[columns["ignition_time"]].map(_number)

    # Sample dimensions: DB stores values in metres already.
    sample_dims = raw[columns["dimensions"]].map(dimensions_m)
    for i in range(3):
        df[f"sample_dim_{i + 1}_m"] = sample_dims.map(
            lambda x, i=i: x[i] if len(x) > i else np.nan)

    df["geometry_cat"] = raw[columns["geometry"]].map(lambda x: _category(
        x, {"Wire": ("wire",), "Flat": ("flat",), "Cylindrical": ("cyl",), "Spherical": ("spher",)}))
    df["ignition_method_cat"] = raw[columns["ignition_method"]].map(lambda x: _category(
        x, {"Open Flame": ("open flame", "pilot", "match"), "Radiative Heater": ("radiative", "heater"),
            "Discharge": ("discharge", "high-voltage"), "Wire / Coil": ("wire", "coil", "nicr", "electric")}))
    df["diluent_cat"] = raw[columns["diluent"]].fillna("Unknown").astype(str)
    # Zero and unreported flow are both quiescent experiments.
    df["flow_direction_cat"] = np.select(
        [df["flow_velocity_m_s"] > 0, df["flow_velocity_m_s"] < 0],
        ["Coflow", "Counterflow"], default="Quiescent")
    df["gravity_regime_cat"] = np.select(
        [df["gravity_g"] < .001, df["gravity_g"] < .95, df["gravity_g"] <= 1.05],
        ["Microgravity", "Partial", "Earth"], "Hyper / Unknown")

    missing_target_rows: list[int] = []
    if "target" in columns:
        normalized = raw[columns["target"]].astype(str).str.strip().str.lower()
        df["ignition_binary"] = normalized.map(
            {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0})
        missing_target_rows = df.loc[df["ignition_binary"].isna(), "source_row_number"].tolist()
        if require_target and missing_target_rows:
            raise ValueError(
                f"{len(missing_target_rows)} rows have missing/unrecognized target labels; "
                f"source rows include {missing_target_rows[:20]}. No rows were silently dropped."
            )
        if require_target:
            df["ignition_binary"] = df["ignition_binary"].astype(int)
    elif require_target:
        raise ValueError("Target column is required for training/evaluation")

    features = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    duplicate_mask = df.duplicated(subset=features + (["ignition_binary"] if "ignition_binary" in df else []) +
                                   ["paper_id"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if deduplicate:
        df = df.loc[~duplicate_mask].reset_index(drop=True)
    report = {
        "data_version": DATA_VERSION, "dataset_path": str(Path(path).resolve()),
        "dataset_sha256": dataset_hash(path), "encoding": encoding,
        "raw_row_count": original_rows, "row_count": len(df),
        "blank_row_count": len(blank_source_rows),
        "blank_source_rows": blank_source_rows,
        "label_counts": ({str(k): int(v) for k, v in df["ignition_binary"].value_counts().items()}
                         if "ignition_binary" in df else {}),
        "missing_target_count": len(missing_target_rows),
        "paper_count": int(df["paper_id"].nunique()), "duplicate_count": duplicate_count,
        "missingness_by_model_feature": {c: int(df[c].isna().sum()) for c in features},
        "feature_availability": {c: bool(df[c].notna().any()) for c in features},
        "post_outcome_columns_excluded": [c for c in POST_OUTCOME_COLUMNS
                                          if any(str(x).startswith(c) for x in raw.columns)],
        "feature_manifest": feature_manifest(),
    }
    return df, report


def write_validation_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def paper_weights(papers: pd.Series, strategy: str = "sqrt", beta: float = .999) -> np.ndarray:
    counts = papers.map(papers.value_counts()).to_numpy(float)
    if strategy == "none": weights = np.ones(len(papers))
    elif strategy == "inverse": weights = 1 / counts
    elif strategy == "sqrt": weights = 1 / np.sqrt(counts)
    elif strategy == "effective": weights = (1 - beta) / (1 - beta ** counts)
    elif strategy == "log": weights = 1 / (1 + np.log(counts))
    else: raise ValueError(f"Unknown paper weighting strategy: {strategy}")
    return weights / weights.mean()


def combined_weights(y: np.ndarray, papers: pd.Series, paper_strategy: str,
                     class_weight: bool) -> np.ndarray:
    weights = paper_weights(papers, paper_strategy)
    if class_weight:
        y = np.asarray(y)
        counts = np.bincount(y, minlength=2)
        class_weights = np.array([len(y) / (2 * max(counts[i], 1)) for i in range(2)])
        weights *= class_weights[y]
    return weights / weights.mean()
