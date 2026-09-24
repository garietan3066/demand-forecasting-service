"""Export a fitted forecasting model with its serving contract."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from demand_forecasting.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


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
