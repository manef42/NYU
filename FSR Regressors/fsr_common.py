"""Canonical data loading, feature detection, and metrics for FSR regression.

Every script in this directory imports this module so that the dataset, the
canonical paper identity, the feature manifest, and the metric definitions are
identical across split generation, benchmarking, refitting, reporting, and
inference. Nothing here trains a model.
"""
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_VERSION = "fsr-data-v1"
RANDOM_STATE = 42
TARGET = "fsr"
ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
METRIC_KEYS = ("R2", "RMSE", "MAE", "MAPE", "MBE", "NRMSE")
CSV_ENGINES = {".csv", ".txt"}
EXCEL_ENGINES = {".xlsm", ".xlsx", ".xls"}
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_LEADING_NUM = re.compile(r"^[-+]?\.?\d")
SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# GPU / Slurm helpers shared by every model that can use an accelerator
# =============================================================================

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


def gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def xgboost_device() -> str:
    """XGBoost runs on the GPU when one is visible and on the CPU otherwise."""
    return "cuda" if gpu_count() else "cpu"


def slurm_num_workers() -> int:
    """Use the CPU allocation supplied by Slurm for PyTorch data loading."""
    return max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", "0")))


def slurm_cpu_count() -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    return allocated if allocated > 0 else (os.cpu_count() or 1)


# =============================================================================
# Cell-level parsing helpers (the workbook is hand-curated and unit-laden)
# =============================================================================

def clean_text(value: Any) -> Any:
    """Normalize a raw cell to a tidy string, or NaN for placeholder/empty tokens."""
    if isinstance(value, (list, tuple, dict, set)):
        return value
    if pd.isna(value):
        return np.nan
    if not isinstance(value, str):
        return value
    text = re.sub(r"\s+", " ", value.replace("\u00a0", " ").strip())
    return np.nan if text.lower() in {"", "-", "--", "n/a", "na", "nan", "none"} else text


def first_number(value: Any) -> float:
    """Extract the leading numeric token from a possibly unit-laden cell.

    Only cells that *begin* with a number (after optional comparison markers such
    as ``~`` or ``<``) are treated as measurements, so unit-laden entries such as
    ``"94 W"``, ``"101.3 kPa"`` or ``"305 mm diameter"`` parse while chemical
    labels such as ``"N2"`` or ``"CO2"`` stay categorical instead of silently
    becoming the numbers 2 and 2.
    """
    if isinstance(value, (list, tuple, dict, set)):
        return np.nan
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).replace(",", ".").replace("\u2212", "-").strip()
    text = text.lstrip("~\u2248<>=\u2264\u2265( ").strip()
    if not _LEADING_NUM.match(text):
        return np.nan
    match = _NUM.search(text)
    return float(match.group()) if match else np.nan


def slug(text: str) -> str:
    """Filesystem-safe slug used when naming per-model output files."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Sign-preserving log compression, invertible for every real value."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sign(values) * (np.expm1(np.abs(values)))


def dataset_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_data_file(user_path: str | Path) -> Path:
    """Resolve the database path independently of the working directory."""
    given = Path(user_path)
    if given.exists():
        return given
    candidates = [SCRIPT_DIR / given.name]
    candidates += [parent / given.name for parent in SCRIPT_DIR.parents]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return given


# =============================================================================
# Column-role detection (target, paper identity, leakage, numeric/categorical)
# =============================================================================

TARGET_KEYWORDS = ("fsr", "flame spread", "spread rate", "spreadrate")
GROUP_KEYWORDS = ("article", "paper", "mla", "citation", "doi", "publication",
                  "reference", "author")
LEAKAGE_KEYWORDS = ("flame length", "hrr", "heat release", "smoke", "aerosol",
                    "areosol", "ignition (yes", "burn", "extinction", "extinguish")
NOTE_KEYWORDS = ("info", "note", "comment", "remark")
APPARATUS_KEYWORDS = ("rig", "facility", "ignition", "internal", "apparatus",
                      "tower", "aircraft", "spacecraft", "sounding", "campaign",
                      "setup", "chamber", "duct", "test number")


def looks_like_target(column: str) -> bool:
    text = str(column).lower()
    return any(keyword in text for keyword in TARGET_KEYWORDS)


def detect_target_column(df: pd.DataFrame) -> str:
    """Find the Flame Spread Rate column: named like FSR and mostly numeric."""
    candidates = [c for c in df.columns if looks_like_target(c)]
    if not candidates:
        raise ValueError(
            f"No flame-spread-rate column found using keywords {TARGET_KEYWORDS}. "
            f"Available columns: {list(df.columns)}")
    return max(candidates, key=lambda c: int(df[c].map(first_number).notna().sum()))


def detect_group_column(df: pd.DataFrame) -> str:
    """Find the citation column that identifies the source paper or campaign."""
    rows = len(df)
    scored = []
    for column in df.columns:
        text = str(column).lower()
        for rank, keyword in enumerate(GROUP_KEYWORDS):
            if keyword not in text:
                continue
            distinct = int(df[column].nunique(dropna=True))
            if distinct >= max(2, .9 * rows):
                break
            completeness = float(df[column].notna().sum()) / max(rows, 1)
            scored.append((rank, -completeness, str(column)))
            break
    if not scored:
        raise ValueError(
            f"No paper/grouping column found using keywords {GROUP_KEYWORDS}. "
            f"Available columns: {list(df.columns)}")
    scored.sort()
    return scored[0][2]


def detect_leakage_columns(df: pd.DataFrame, target: str, group: str) -> list[str]:
    """Columns dropped before modelling: duplicate targets, outcomes, notes, IDs."""
    leaking = set()
    for column in df.columns:
        if column in {target, group}:
            continue
        text = str(column).lower()
        if (looks_like_target(column) or
                any(keyword in text for keyword in LEAKAGE_KEYWORDS) or
                any(keyword in text for keyword in NOTE_KEYWORDS) or
                any(keyword in text for keyword in GROUP_KEYWORDS)):
            leaking.add(column)
    return [c for c in df.columns if c in leaking]


def detect_feature_types(df: pd.DataFrame, feature_columns: list[str],
                         numeric_fraction: float = .60
                         ) -> tuple[list[str], list[str], pd.DataFrame]:
    """Split features into numeric and categorical and commit the numeric parse."""
    out = df.copy()
    numeric, categorical = [], []
    for column in feature_columns:
        series = out[column]
        available = int(series.notna().sum())
        if available == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            continue
        parsed = series.map(first_number)
        if int(parsed.notna().sum()) / available >= numeric_fraction:
            out[column] = parsed
            numeric.append(column)
        else:
            out[column] = series.map(clean_text)
            categorical.append(column)
    return numeric, categorical, out


def feature_role(column: str) -> str:
    """Physics features transfer across rigs; apparatus features describe the rig."""
    text = str(column).lower()
    return "apparatus" if any(k in text for k in APPARATUS_KEYWORDS) else "physics"


def canonical_doi(value: Any) -> Any:
    value = clean_text(value)
    if not isinstance(value, str):
        return np.nan
    text = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value.lower()).rstrip("/. ")
    match = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", text, re.I)
    return match.group().rstrip(".,;") if match else np.nan


def canonical_citation(value: Any) -> Any:
    value = clean_text(value)
    if not isinstance(value, str):
        return np.nan
    text = value.lower().translate(str.maketrans({
        "\u201c": '"', "\u201d": '"', "\u2019": "'", "\u2013": "-", "\u2014": "-"}))
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()
    return text or np.nan


# =============================================================================
# Dataset assembly
# =============================================================================

def _flatten_two_level(df: pd.DataFrame) -> pd.DataFrame:
    flat = []
    for top, sub in df.columns:
        top_text, sub_text = str(top), str(sub)
        name = top_text if (pd.isna(sub) or sub_text.startswith("Unnamed")) else sub_text
        flat.append(name.strip())
    df = df.copy()
    df.columns = flat
    return df


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~pd.Index([str(c).strip() for c in df.columns]).duplicated()]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(clean_text)
    return df.reset_index(drop=True)


def _score(df: pd.DataFrame) -> int:
    has_target = any(looks_like_target(c) for c in df.columns)
    return len(df) * df.shape[1] + (10_000_000 if has_target else 0)


def read_database(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Load the microgravity database from Excel or CSV into a flat frame.

    The workbook stores a two-level "section / field" header and the CSV export
    stores a category banner above the real header, so both layouts are tried and
    the parse that actually exposes a flame-spread-rate column wins.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    best, best_score, source = None, -1, ""
    if path.suffix.lower() in EXCEL_ENGINES:
        try:
            sheets = list(pd.ExcelFile(path, engine="openpyxl").sheet_names)
        except Exception:
            sheets = [0]
        for sheet in sheets:
            for header in ([0, 1], 0):
                try:
                    frame = pd.read_excel(path, sheet_name=sheet, header=header,
                                          engine="openpyxl")
                except Exception:
                    continue
                if isinstance(frame.columns, pd.MultiIndex):
                    frame = _flatten_two_level(frame)
                frame = _tidy(frame)
                score = _score(frame)
                if score > best_score:
                    best, best_score = frame, score
                    source = f"sheet={sheet!r}; header={header}"
    elif path.suffix.lower() in CSV_ENGINES:
        for skiprows in (1, 0):
            for encoding in ENCODINGS:
                try:
                    frame = pd.read_csv(path, skiprows=skiprows, encoding=encoding,
                                        low_memory=False)
                except (UnicodeDecodeError, ValueError):
                    continue
                frame = _tidy(frame)
                score = _score(frame)
                if score > best_score:
                    best, best_score = frame, score
                    source = f"skiprows={skiprows}; encoding={encoding}"
                break
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix!r}")
    if best is None or best.empty:
        raise ValueError(f"Could not read any usable table from {path}")
    return best, source


def load_data(path: str | Path, require_target: bool = True,
              deduplicate: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the modelling frame plus a machine-readable validation report.

    The frame always contains ``row_id``, ``paper_id``, ``paper_label``, every
    detected feature column, and (when a target is required) the numeric ``fsr``
    target. Rows with an unusable target are removed only for training and
    evaluation; inference keeps every row.
    """
    path = locate_data_file(path)
    raw, parse_source = read_database(path)
    raw_rows = len(raw)
    try:
        target_column = detect_target_column(raw)
    except ValueError:
        if require_target:
            raise
        target_column = None
    group_column = detect_group_column(raw)
    leakage_columns = detect_leakage_columns(raw, target_column, group_column)
    feature_columns = [c for c in raw.columns
                       if c not in set(leakage_columns) | {target_column, group_column}]
    numeric, categorical, parsed = detect_feature_types(raw, feature_columns)

    df = pd.DataFrame(index=parsed.index)
    df["source_row_number"] = parsed.index + 3
    df["row_id"] = [f"row-{number:07d}" for number in df["source_row_number"]]
    citation = parsed[group_column] if group_column in parsed else pd.Series(
        np.nan, index=parsed.index)
    doi_columns = [c for c in raw.columns if "doi" in str(c).lower()]
    doi = raw[doi_columns[0]].map(canonical_doi) if doi_columns else pd.Series(
        np.nan, index=parsed.index)
    normalized_citation = citation.map(canonical_citation)
    paper_id = doi.map(lambda x: f"doi::{x}" if isinstance(x, str) else np.nan)
    paper_id = paper_id.fillna(normalized_citation.map(
        lambda x: f"citation::{x}" if isinstance(x, str) else np.nan))
    df["paper_id"] = paper_id.fillna(df["row_id"].map(lambda x: f"unknown::{x}"))
    df["paper_label"] = (citation.astype("object").where(citation.notna(), df["paper_id"])
                         .astype(str).str.slice(0, 120))
    df["raw_citation"] = citation
    for column in numeric + categorical:
        df[column] = parsed[column]

    missing_target_rows: list[int] = []
    if target_column is not None and target_column in parsed:
        df[TARGET] = parsed[target_column].map(first_number)
        missing_target_rows = df.loc[df[TARGET].isna(), "source_row_number"].tolist()
        if require_target:
            df = df.loc[df[TARGET].notna()].reset_index(drop=True)
            df[TARGET] = df[TARGET].astype(float)
    elif require_target:
        raise ValueError("A flame-spread-rate target column is required")

    duplicate_mask = df.duplicated(
        subset=numeric + categorical + ([TARGET] if TARGET in df else []) + ["paper_id"],
        keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if deduplicate:
        df = df.loc[~duplicate_mask].reset_index(drop=True)

    papers = df["paper_id"]
    report = {
        "data_version": DATA_VERSION,
        "dataset_path": str(Path(path).resolve()),
        "dataset_sha256": dataset_hash(path),
        "parse_source": parse_source,
        "raw_row_count": int(raw_rows),
        "row_count": int(len(df)),
        "duplicate_count": duplicate_count,
        "missing_target_count": len(missing_target_rows),
        "target_column": target_column,
        "group_column": group_column,
        "leakage_columns_excluded": leakage_columns,
        "paper_count": int(papers.nunique()),
        "rows_per_paper": {
            "mean": float(papers.value_counts().mean()) if len(df) else np.nan,
            "median": float(papers.value_counts().median()) if len(df) else np.nan,
            "minimum": int(papers.value_counts().min()) if len(df) else 0,
            "maximum": int(papers.value_counts().max()) if len(df) else 0,
        },
        "target_summary": ({
            "mean": float(df[TARGET].mean()), "std": float(df[TARGET].std(ddof=1)),
            "minimum": float(df[TARGET].min()), "maximum": float(df[TARGET].max()),
        } if TARGET in df and len(df) else {}),
        "feature_manifest": (
            [{"feature": c, "kind": "numeric", "role": feature_role(c)} for c in numeric] +
            [{"feature": c, "kind": "categorical", "role": feature_role(c)}
             for c in categorical]),
        "missingness_by_feature": {c: int(df[c].isna().sum()) for c in numeric + categorical},
    }
    return df, report


def feature_lists(manifest: list[dict[str, str]], feature_set: str
                  ) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names for 'all' or 'physics'."""
    if feature_set not in {"all", "physics"}:
        raise ValueError("feature_set must be 'all' or 'physics'")
    numeric = [item["feature"] for item in manifest if item["kind"] == "numeric"
               and (feature_set == "all" or item["role"] == "physics")]
    categorical = [item["feature"] for item in manifest if item["kind"] == "categorical"
                   and (feature_set == "all" or item["role"] == "physics")]
    if not numeric and not categorical:
        raise ValueError(f"Feature set {feature_set!r} selected no features")
    return numeric, categorical


def write_validation_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True, default=str),
                          encoding="utf-8")


# =============================================================================
# Preprocessing, weighting, augmentation, metrics
# =============================================================================

def build_preprocessor(numeric: list[str], categorical: list[str],
                       scale_numeric: bool = True,
                       native_numeric_missing: bool = False) -> ColumnTransformer:
    """One shared preprocessing contract for every model family.

    Numeric columns are median-imputed unless the estimator handles missing
    values natively, and standardized for distance- and gradient-based models.
    Categorical columns use unknown-safe one-hot encoding so categories that
    appear only in held-out papers cannot break inference.
    """
    numeric_steps: list[tuple[str, Any]] = []
    if not native_numeric_missing:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_transformer, numeric),
        ("categorical", categorical_transformer, categorical),
    ], remainder="drop")


def paper_weights(papers: pd.Series, strategy: str = "none") -> np.ndarray:
    """Down-weight papers that contribute many correlated rows."""
    papers = pd.Series(papers).reset_index(drop=True)
    counts = papers.map(papers.value_counts()).to_numpy(float)
    if strategy == "none":
        weights = np.ones(len(papers))
    elif strategy == "inverse":
        weights = 1 / counts
    elif strategy == "sqrt":
        weights = 1 / np.sqrt(counts)
    elif strategy == "log":
        weights = 1 / (1 + np.log(counts))
    else:
        raise ValueError(f"Unknown paper weighting strategy: {strategy!r}")
    return weights / weights.mean()


def bootstrap_augment(X: pd.DataFrame, y: np.ndarray, n_extra: int,
                      random_state: int = RANDOM_STATE
                      ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Bootstrap resampling augmentation exactly as in Rivera et al., Sec. 3.3.

    ``n_extra`` rows are drawn with replacement from the training partition only
    and appended to it, so the augmentation re-weights the empirical density of
    physically admissible observations without inventing conditions and without
    ever touching held-out rows or held-out papers. The row indices of the drawn
    copies are returned so that paper identity can follow the copies.
    """
    y = np.asarray(y, dtype=float)
    if n_extra <= 0:
        return X.copy(), y.copy(), np.arange(len(X))
    rng = np.random.default_rng(random_state)
    drawn = rng.integers(0, len(X), size=int(n_extra))
    augmented = pd.concat([X, X.iloc[drawn]], axis=0, ignore_index=True)
    return augmented, np.concatenate([y, y[drawn]]), np.concatenate(
        [np.arange(len(X)), drawn])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R2, RMSE, MAE, safe MAPE, mean bias error, and normalized RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    count = len(y_true)
    if count == 0:
        return {**{key: float("nan") for key in METRIC_KEYS}, "n": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_true = float(np.mean(y_true))
    finite_variance = count >= 2 and float(np.var(y_true)) > 0
    mask = np.abs(y_true) > 1e-6
    return {
        "R2": float(r2_score(y_true, y_pred)) if finite_variance else float("nan"),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": (float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
                 if mask.any() else float("nan")),
        "MBE": float(np.mean(y_pred - y_true)),
        "NRMSE": float(rmse / mean_true) if abs(mean_true) > 1e-12 else float("nan"),
        "n": int(count),
    }


def quantile_bins(values: np.ndarray, max_bins: int = 6) -> np.ndarray | None:
    """Quantile bin labels with duplicate edges dropped, or None if impossible."""
    series = pd.Series(np.asarray(values, dtype=float)).replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(series) < 8:
        return None
    bins = int(min(max_bins, max(2, int(series.nunique()))))
    if bins < 2:
        return None
    try:
        labels = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    full = pd.Series(np.nan, index=pd.RangeIndex(len(values)), dtype=float)
    full.iloc[series.index.to_numpy()] = np.asarray(labels, dtype=float)
    return full.to_numpy()


def distribution_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Total-variation distance between two discrete histograms."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() <= 0 and right.sum() <= 0:
        return 0.
    left = left / max(left.sum(), 1e-12)
    right = right / max(right.sum(), 1e-12)
    return float(.5 * np.abs(left - right).sum())
