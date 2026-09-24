"""Test native model export and release metadata."""

import json

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.artifacts import export_package, file_sha256, load_package
from demand_forecasting.ensemble import equal_weight_blend
from demand_forecasting.features import FEATURE_COLUMNS, build_training_features
from demand_forecasting.modeling import fit_forecaster, predict_forecaster
from demand_forecasting.serving import build_next_day_features


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


def prediction_history():
    return pd.DataFrame(
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


def test_loaded_model_and_blend_match_original(fitted, tmp_path):
    destination = tmp_path / "release"
    digests = export_package(fitted, destination, {})
    loaded = load_package(destination, digests["metadata_sha256"])

    features = build_next_day_features(
        prediction_history(), fitted.training_end + pd.Timedelta(days=1)
    )
    original_predictions = predict_forecaster(fitted, features)
    loaded_predictions = predict_forecaster(loaded, features)

    np.testing.assert_allclose(loaded_predictions, original_predictions, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        equal_weight_blend(features.mean_7, loaded_predictions),
        equal_weight_blend(features.mean_7, original_predictions),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("metadata.json", "Metadata integrity"),
        ("model.txt", "Model integrity"),
    ],
)
def test_modified_files_are_rejected(fitted, tmp_path, filename, message):
    destination = tmp_path / "release"
    digests = export_package(fitted, destination, {})
    path = destination / filename
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=message):
        load_package(destination, digests["metadata_sha256"])


def test_incompatible_contract_is_rejected(fitted, tmp_path):
    destination = tmp_path / "release"
    export_package(fitted, destination, {})
    path = destination / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["feature_columns"] = list(reversed(FEATURE_COLUMNS))
    path.write_text(json.dumps(metadata), encoding="utf-8")

    # Even correctly hashed metadata must satisfy the serving contract.
    with pytest.raises(ValueError, match="feature_columns"):
        load_package(destination, file_sha256(path))


def test_loaded_model_rejects_unknown_series(fitted, tmp_path):
    destination = tmp_path / "release"
    digests = export_package(fitted, destination, {})
    loaded = load_package(destination, digests["metadata_sha256"])
    features = build_next_day_features(
        prediction_history(), fitted.training_end + pd.Timedelta(days=1)
    )
    features["product_id"] = 999

    with pytest.raises(ValueError, match="Unknown store-product"):
        predict_forecaster(loaded, features)
