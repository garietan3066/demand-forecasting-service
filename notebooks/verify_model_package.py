"""Verify serving predictions against the recorded final evaluation."""

import json

import numpy as np
import pandas as pd

from demand_forecasting.artifacts import file_sha256, load_package
from demand_forecasting.constants import KEYS
from demand_forecasting.ensemble import equal_weight_blend
from demand_forecasting.modeling import predict_forecaster
from demand_forecasting.serving import build_next_day_features
from notebooks.baseline_forecast import ROOT
from notebooks.cross_validate_lightgbm import load_verified_subset

METHODS = ["mean_previous_7_days", "lightgbm", "blend_50_50"]


def main():
    build_path = ROOT / "reports/model_package_build.json"
    final_path = ROOT / "reports/final_evaluation.json"
    predictions_path = ROOT / "data/processed/final_evaluation_predictions.parquet"

    build = json.loads(build_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))

    package = (ROOT / build["package_directory"]).resolve()
    release_root = (ROOT / "artifacts/releases").resolve()
    if not package.is_relative_to(release_root):
        raise ValueError("Package directory is outside the release directory.")

    loaded = load_package(package, build["metadata_sha256"])

    metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    if metadata["model_sha256"] != build["model_sha256"]:
        raise ValueError("Build report and package model hashes differ.")
    if metadata["provenance"]["final_evaluation_report_sha256"] != file_sha256(final_path):
        raise ValueError("Final evaluation report differs from package provenance.")
    if file_sha256(predictions_path) != final["predictions_sha256"]:
        raise ValueError("Recorded evaluation predictions changed.")

    recorded = pd.read_parquet(predictions_path)
    recorded["dt"] = pd.to_datetime(recorded["dt"], errors="raise")

    training, reference = load_verified_subset()
    if (
        reference["selected_pairs"] != final["selected_pairs"]
        or reference["cache_revision"] != final["cache_revision"]
    ):
        raise ValueError("Training reference differs from final evaluation.")

    pairs = {(item["store_id"], item["product_id"]) for item in final["selected_pairs"]}
    dates = pd.date_range(final["evaluation_start"], final["evaluation_end"], freq="D")
    expected_index = pd.MultiIndex.from_tuples(
        [(store, product, date) for store, product in sorted(pairs) for date in dates],
        names=KEYS + ["dt"],
    )
    actual_index = pd.MultiIndex.from_frame(recorded[KEYS + ["dt"]])

    if (
        actual_index.has_duplicates
        or len(expected_index.difference(actual_index))
        or len(actual_index.difference(expected_index))
        or len(recorded) != final["evaluation_rows"]
    ):
        raise ValueError("Recorded prediction coverage is inconsistent.")

    if (
        loaded.known_pairs != pairs
        or loaded.training_end != pd.Timestamp(final["training_end"])
        or training.dt.max() != loaded.training_end
    ):
        raise ValueError("Package series or training cutoff differ.")

    history = training[KEYS + ["dt", "sale_amount"]].copy()
    outputs = []

    for date in dates:
        # Build predictions before exposing this day's actual sales.
        features = build_next_day_features(history, date)
        result = features[KEYS + ["dt"]].copy()
        result["mean_previous_7_days"] = features["mean_7"].to_numpy()
        result["lightgbm"] = predict_forecaster(loaded, features)
        result["blend_50_50"] = equal_weight_blend(
            result["mean_previous_7_days"], result["lightgbm"]
        )
        outputs.append(result)

        # Simulate observations arriving after this day's forecast.
        observed = recorded.loc[recorded.dt.eq(date), KEYS + ["dt", "sale_amount"]]
        history = pd.concat([history, observed], ignore_index=True)

    served = pd.concat(outputs, ignore_index=True)
    joined = recorded.merge(
        served,
        on=KEYS + ["dt"],
        suffixes=("_recorded", "_served"),
        validate="one_to_one",
    )
    if len(joined) != len(recorded):
        raise ValueError("Serving predictions do not cover every recorded row.")

    differences = {}
    for name in METHODS:
        expected = joined[f"{name}_recorded"].to_numpy(dtype=float)
        actual = joined[f"{name}_served"].to_numpy(dtype=float)

        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError("Nonfinite predictions.")

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Serving parity failed for {name}",
        )
        differences[name] = float(np.max(np.abs(actual - expected)))

    report = {
        "status": "passed",
        "package_directory": build["package_directory"],
        "metadata_sha256": build["metadata_sha256"],
        "model_sha256": build["model_sha256"],
        "verification_script_sha256": file_sha256(__file__),
        "final_evaluation_report_sha256": file_sha256(final_path),
        "reference_predictions_sha256": file_sha256(predictions_path),
        "rows_verified": len(joined),
        "days_verified": len(dates),
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 1e-12,
        "maximum_absolute_difference": differences,
        "protocol": (
            "Loaded package; serving features built from prior history; "
            "actuals appended only after each day's prediction; no retraining."
        ),
    }

    output = ROOT / "reports/model_package_verification.json"
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
