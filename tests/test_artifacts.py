"""Test native model export and release metadata."""

import json

import pandas as pd
import pytest

from demand_forecasting.artifacts import export_package, file_sha256
from demand_forecasting.features import FEATURE_COLUMNS, build_training_features
from demand_forecasting.modeling import fit_forecaster


@pytest.fixture(scope="module")
def fitted():
    frame = pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(product + day % 7),
            }
            for product in [10, 20]
            for day, date in enumerate(pd.date_range("2024-01-01", periods=60))
        ]
    )
    features = build_training_features(frame)
    return fit_forecaster(features.loc[features.history_ready].copy())


def test_export_preserves_serving_contract(fitted, tmp_path):
    destination = tmp_path / "release"
    digests = export_package(fitted, destination, {"test_build": True})
    metadata = json.loads((destination / "metadata.json").read_text())

    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["categories"] == fitted.categories
    assert metadata["known_pairs"] == [[1, 10], [1, 20]]
    assert metadata["training_end"] == fitted.training_end.date().isoformat()
    assert metadata["blend"]["baseline_feature"] == "mean_7"
    assert metadata["blend"]["lightgbm_weight"] == 0.5
    assert metadata["blend"]["baseline_weight"] == 0.5
    assert digests["model_sha256"] == file_sha256(destination / "model.txt")
    assert digests["metadata_sha256"] == file_sha256(destination / "metadata.json")


def test_existing_package_cannot_be_overwritten(fitted, tmp_path):
    destination = tmp_path / "release"
    original = export_package(fitted, destination, {})

    with pytest.raises(FileExistsError):
        export_package(fitted, destination, {})

    assert file_sha256(destination / "model.txt") == original["model_sha256"]
    assert file_sha256(destination / "metadata.json") == original["metadata_sha256"]


def test_invalid_provenance_does_not_create_package(fitted, tmp_path):
    destination = tmp_path / "release"

    with pytest.raises(ValueError):
        export_package(fitted, destination, {"invalid": float("nan")})

    assert not destination.exists()
