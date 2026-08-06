"""Predict flame spread rate with a refitted interpolation or extrapolation champion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from fsr_common import TARGET, configure_torch, empty_cuda_cache, load_data

REQUIRED_FILES = ("model.joblib", "model_card.json", "feature_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="CSV or Excel file using the database schema.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--champion", choices=["interpolation", "extrapolation"],
                        default="extrapolation")
    parser.add_argument("--artifact", help="Override the champion artifact directory.")
    args = parser.parse_args()
    configure_torch()
    root = Path(__file__).resolve().parent
    artifact = (Path(args.artifact) if args.artifact
                else root / "artifacts" / f"{args.champion}_champion")
    missing = [name for name in REQUIRED_FILES if not (artifact / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Artifact {artifact} is missing required files: {missing}")
    data, report = load_data(args.input, require_target=False, deduplicate=False)
    model = joblib.load(artifact / "model.joblib")
    card = json.loads((artifact / "model_card.json").read_text(encoding="utf-8"))
    unavailable = [feature for feature in model.features if feature not in data.columns]
    if unavailable:
        raise ValueError(f"Input is missing model feature columns: {unavailable}")
    prediction = model.predict(data)
    empty_cuda_cache()
    output = pd.DataFrame({
        "row_id": data["row_id"], "paper_id": data["paper_id"],
        "paper_label": data["paper_label"], "predicted_fsr": prediction})
    if TARGET in data.columns:
        output["observed_fsr"] = data[TARGET].to_numpy()
    output["champion"] = args.champion
    output["model_id"] = card["model_id"]
    output["model_family"] = card["model_family"]
    output["augmentation"] = card["augmentation"]
    output["artifact_version"] = card["artifact_version"]
    output["training_dataset_sha256"] = card["data_fingerprint"]["dataset_sha256"]
    output["input_dataset_sha256"] = report["dataset_sha256"]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(json.dumps({"rows": int(len(output)), "output": str(path.resolve()),
                      "model_id": card["model_id"]}, indent=2))


if __name__ == "__main__":
    main()
