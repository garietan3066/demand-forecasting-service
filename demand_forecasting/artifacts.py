"""Export a fitted forecasting model with its serving contract."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from demand_forecasting.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from demand_forecasting.modeling import FittedForecaster


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_package(fitted, destination, provenance):
    """Export into a new directory; never overwrite an existing package."""
    destination = Path(destination)

    if not fitted.known_pairs:
        raise ValueError("Cannot export a model without known series.")

    if fitted.model.booster_.feature_name() != FEATURE_COLUMNS:
        raise ValueError("Model feature order differs from the serving contract.")

    # Validate JSON serializability before creating output files.
    json.dumps(provenance, allow_nan=False)

    destination.mkdir(parents=True, exist_ok=False)
    model_path = destination / "model.txt"
    fitted.model.booster_.save_model(str(model_path))

    metadata = {
        "schema_version": 1,
        "method": "blend_50_50",
        "target": "Observed daily sales in normalized units",
        "forecast_horizon_days": 1,
        "minimum_history_days": 30,
        "training_end": fitted.training_end.date().isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": fitted.categories,
        "known_pairs": [list(pair) for pair in sorted(fitted.known_pairs)],
        "lightgbm_prediction_postprocessing": "clip_below_zero",
        "blend": {
            "lightgbm_weight": 0.5,
            "baseline_weight": 0.5,
            "baseline_feature": "mean_7",
        },
        "model_file": "model.txt",
        "model_sha256": file_sha256(model_path),
        "versions": {
            name: version(name) for name in ["lightgbm", "numpy", "pandas", "scikit-learn"]
        },
        "provenance": provenance,
    }

    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return {
        "model_sha256": metadata["model_sha256"],
        "metadata_sha256": file_sha256(metadata_path),
    }


def load_package(directory, expected_metadata_sha256):
    """Load a package whose metadata hash comes from a trusted release record."""
    directory = Path(directory).resolve()

    def read_local(name):
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Package file resolves outside its directory.")
        return path.read_bytes()

    # Verify the exact bytes before parsing metadata.
    metadata_bytes = read_local("metadata.json")
    if hashlib.sha256(metadata_bytes).hexdigest() != expected_metadata_sha256:
        raise ValueError("Metadata integrity mismatch.")

    metadata = json.loads(metadata_bytes)

    expected_contract = {
        "schema_version": 1,
        "method": "blend_50_50",
        "forecast_horizon_days": 1,
        "minimum_history_days": 30,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "lightgbm_prediction_postprocessing": "clip_below_zero",
        "model_file": "model.txt",
        "blend": {
            "lightgbm_weight": 0.5,
            "baseline_weight": 0.5,
            "baseline_feature": "mean_7",
        },
    }
    for name, expected in expected_contract.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Incompatible package contract: {name}.")

    # Start with strict compatibility. Relax only after explicit testing.
    for name in ["lightgbm", "numpy", "pandas", "scikit-learn"]:
        if metadata["versions"].get(name) != version(name):
            raise ValueError(f"Package dependency version mismatch: {name}.")

    cutoff = pd.Timestamp(metadata["training_end"])
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("Invalid package training cutoff.")

    pair_rows = metadata["known_pairs"]
    if not pair_rows or any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(type(value) is not int for value in pair)
        for pair in pair_rows
    ):
        raise ValueError("Invalid package series identifiers.")

    pairs = {tuple(pair) for pair in pair_rows}
    if len(pairs) != len(pair_rows):
        raise ValueError("Duplicate package series identifiers.")

    categories = metadata["categories"]
    expected_categories = {
        name: sorted({pair[index] for pair in pairs})
        for index, name in enumerate(CATEGORICAL_FEATURES)
    }
    if categories != expected_categories:
        raise ValueError("Category mappings do not match known series.")

    # Read once, verify once, and parse those same verified bytes.
    model_bytes = read_local("model.txt")
    if hashlib.sha256(model_bytes).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Model integrity mismatch.")

    booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))

    if booster.feature_name() != FEATURE_COLUMNS:
        raise ValueError("Loaded model feature order mismatch.")

    expected_pandas_categories = [categories[name] for name in CATEGORICAL_FEATURES]
    if booster.pandas_categorical != expected_pandas_categories:
        raise ValueError("Loaded model category order mismatch.")

    return FittedForecaster(
        model=booster,
        categories=categories,
        known_pairs=pairs,
        training_end=cutoff,
    )
