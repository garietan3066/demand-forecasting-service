"""Evaluate a fixed blend on existing development-validation predictions."""

import hashlib
import json

import pandas as pd

from demand_forecasting.constants import KEYS
from demand_forecasting.ensemble import equal_weight_blend
from demand_forecasting.metrics import metrics
from notebooks.baseline_forecast import ROOT
from notebooks.cross_validate_lightgbm import compare_metrics

BASELINE = "mean_previous_7_days"
MODEL = "lightgbm"
BLEND = "blend_50_50"
METHODS = [BASELINE, MODEL, BLEND]


def main():
    source = ROOT / "data/processed/lightgbm_cross_validation_predictions.parquet"

    reference_path = ROOT / "reports/lightgbm_cross_validation.json"

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(source)
    predictions["dt"] = pd.to_datetime(predictions["dt"], errors="raise")

    if predictions[KEYS + ["dt", "fold"]].isna().any().any():
        raise ValueError("Prediction identifiers or dates are missing.")

    if predictions.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Duplicate validation observations.")

    expected_folds = {fold["fold"] for fold in reference["folds"]}

    if set(predictions["fold"]) != expected_folds:
        raise ValueError("Saved folds differ from the reference.")

    predictions[BLEND] = equal_weight_blend(
        predictions[BASELINE],
        predictions[MODEL],
    )

    expected_pairs = {
        (item["store_id"], item["product_id"]) for item in reference["selected_pairs"]
    }

    fold_reports = []

    for fold in reference["folds"]:
        part = predictions.loc[predictions["fold"].eq(fold["fold"])].copy()

        expected_dates = pd.date_range(
            fold["validation_start"],
            fold["validation_end"],
        )

        expected_index = pd.MultiIndex.from_tuples(
            [
                (store, product, date)
                for store, product in expected_pairs
                for date in expected_dates
            ],
            names=KEYS + ["dt"],
        )

        actual_index = pd.MultiIndex.from_frame(part[KEYS + ["dt"]])

        if (
            len(part) != fold["validation_rows"]
            or len(actual_index) != len(expected_index)
            or len(actual_index.difference(expected_index)) > 0
            or len(expected_index.difference(actual_index)) > 0
        ):
            raise ValueError("Validation dates or selected series changed.")

        scores = {name: metrics(part["sale_amount"], part[name]) for name in METHODS}

        # Confirm that the original methods still reproduce their scores.
        for name in [BASELINE, MODEL]:
            compare_metrics(scores[name], fold["metrics"][name])

        fold_reports.append(
            {
                "fold": fold["fold"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "validation_rows": len(part),
                "metrics": scores,
            }
        )

    pooled = {name: metrics(predictions["sale_amount"], predictions[name]) for name in METHODS}

    for name in [BASELINE, MODEL]:
        compare_metrics(
            pooled[name],
            reference["pooled_metrics"][name],
        )

    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    report = {
        "experiment": "Fixed 50:50 blend",
        "weights": {BASELINE: 0.5, MODEL: 0.5},
        "source_predictions_sha256": source_hash,
        "cache_revision": reference["cache_revision"],
        "target": reference["target"],
        "selection_note": (
            "This experiment was proposed after inspecting development "
            "validation results. These results are not an independent "
            "final assessment. No blend-weight search was performed."
        ),
        "folds": fold_reports,
        "pooled_metrics": pooled,
    }

    (ROOT / "reports/fixed_blend_cross_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    predictions.to_parquet(
        ROOT / "data/processed/fixed_blend_predictions.parquet",
        index=False,
    )

    print("PASS: original prediction coverage and metrics match.")

    for fold in fold_reports:
        print(f"\nFold {fold['fold']}")
        for name, score in fold["metrics"].items():
            wape = score["wape"]
            display = "undefined" if wape is None else f"{wape:.2%}"
            print(f"  {name}: WAPE={display}, MAE={score['mae']:.4f}")

    print("\nPooled metrics:")
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
