"""Compare fresh LightGBM models with baselines on chronological folds."""

import json
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd

from demand_forecasting.baselines import predict_baselines
from demand_forecasting.constants import BASELINES, KEYS
from demand_forecasting.features import (
    FEATURE_COLUMNS,
    build_training_features,
)
from demand_forecasting.metrics import metrics
from demand_forecasting.modeling import (
    MODEL_PARAMETERS,
    fit_forecaster,
    predict_forecaster,
)
from demand_forecasting.validation import expanding_date_folds
from notebooks.baseline_forecast import (
    FIRST_DATE,
    LAST_DATE,
    ROOT,
    batches,
    choose_pairs,
    verified_shards,
)


def load_verified_subset():
    """Rebuild the established subset from audited training files."""
    paths, revision = verified_shards()
    print("Training file hashes verified.", flush=True)

    first_day = pd.concat(
        [batch.loc[batch.dt.eq(FIRST_DATE), KEYS] for batch in batches(paths)],
        ignore_index=True,
    )

    pairs = choose_pairs(first_day, size=200)

    reference = json.loads(
        (ROOT / "reports/baseline_cross_validation.json").read_text(encoding="utf-8")
    )

    expected_pairs = [
        (item["store_id"], item["product_id"]) for item in reference["selected_pairs"]
    ]

    if len(pairs) != 200 or pairs != expected_pairs:
        raise ValueError("Selected series differ from the baseline experiment.")

    if revision != reference["cache_revision"]:
        raise ValueError("Training data revision changed.")

    selected_index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    chunks = []

    for batch in batches(paths):
        mask = pd.MultiIndex.from_frame(batch[KEYS]).isin(selected_index)
        chunks.append(batch.loc[mask])

    selected = pd.concat(chunks, ignore_index=True)
    selected["dt"] = pd.to_datetime(selected["dt"], errors="raise")

    if (
        len(selected) != 18000
        or selected.dt.min() != pd.Timestamp(FIRST_DATE)
        or selected.dt.max() != pd.Timestamp(LAST_DATE)
    ):
        raise ValueError("Unexpected subset coverage.")

    return selected, reference


def evaluate_lightgbm(frame):
    """Train independently in each fold and score identical validation rows."""
    features = build_training_features(frame)
    baselines = predict_baselines(frame)

    # Both functions sort their outputs. Verify alignment before assigning
    # baseline predictions by position.
    pd.testing.assert_frame_equal(
        features[KEYS + ["dt"]],
        baselines[KEYS + ["dt"]],
    )

    for name in BASELINES:
        features[name] = baselines[name].to_numpy()

    fold_reports = []
    prediction_frames = []
    methods = BASELINES + ["lightgbm"]

    for fold in expanding_date_folds(features):
        training_history = features.iloc[fold.train_indices]
        training = training_history.loc[training_history["history_ready"]].copy()

        validation = features.iloc[fold.validation_indices].copy()

        if not validation["history_ready"].all():
            raise ValueError("Validation rows have incomplete features.")

        start = perf_counter()
        fitted = fit_forecaster(training)
        training_seconds = perf_counter() - start

        start = perf_counter()

        # Deliberately omit target sales from model prediction inputs.
        validation["lightgbm"] = predict_forecaster(
            fitted,
            validation[FEATURE_COLUMNS + ["dt"]],
        )

        prediction_seconds = perf_counter() - start

        fold_scores = {
            name: metrics(validation["sale_amount"], validation[name]) for name in methods
        }

        fold_reports.append(
            {
                "fold": fold.number,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                "training_history_rows": len(training_history),
                "usable_training_rows": len(training),
                "validation_rows": len(validation),
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
                "metrics": fold_scores,
            }
        )

        validation["fold"] = fold.number
        prediction_frames.append(validation[KEYS + ["dt", "sale_amount", "fold"] + methods])

        print(
            f"Fold {fold.number} finished: "
            f"{len(training):,} training rows, "
            f"{len(validation):,} validation rows.",
            flush=True,
        )

    combined = pd.concat(prediction_frames, ignore_index=True)

    if combined.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Validation observations overlap across folds.")

    pooled = {name: metrics(combined["sale_amount"], combined[name]) for name in methods}

    return {"folds": fold_reports, "pooled_metrics": pooled}, combined


def verify_baseline_parity(results, reference):
    """Fail if this experiment changed the previous comparison benchmark."""
    if len(results["folds"]) != len(reference["folds"]):
        raise ValueError("Fold count changed.")

    for actual, expected in zip(results["folds"], reference["folds"], strict=True):
        for field in [
            "fold",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "training_history_rows",
            "validation_rows",
        ]:
            if actual[field] != expected[field]:
                raise ValueError(f"Fold boundary or coverage changed: {field}")

        for name in BASELINES:
            compare_metrics(
                actual["metrics"][name],
                expected["metrics"][name],
            )

    for name in BASELINES:
        compare_metrics(
            results["pooled_metrics"][name],
            reference["pooled_metrics"][name],
        )


def compare_metrics(actual, expected):
    """Compare counts exactly and floating-point scores within tolerance."""
    if actual["rows"] != expected["rows"]:
        raise ValueError("Baseline evaluation row count changed.")

    for name in ["wape", "mae", "bias"]:
        left, right = actual[name], expected[name]

        if left is None or right is None:
            if left != right:
                raise ValueError(f"Baseline metric changed: {name}")
        elif not np.isclose(left, right, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Baseline metric changed: {name}")


def main():
    selected, reference = load_verified_subset()
    results, predictions = evaluate_lightgbm(selected)

    verify_baseline_parity(results, reference)
    print("PASS: baseline folds and metrics match the reference.")

    report = {
        "dataset": reference["dataset"],
        "source_split": "train",
        "cache_revision": reference["cache_revision"],
        "selected_pairs": reference["selected_pairs"],
        "target": "Observed daily sales in normalized units",
        "protocol": (
            "Three expanding-window folds; fresh LightGBM model per fold; "
            "rolling one-day predictions with earlier observed history. "
            "No early stopping or hyperparameter search."
        ),
        "postprocessing": "Clip negative predictions to zero.",
        "model_parameters": MODEL_PARAMETERS,
        "feature_columns": FEATURE_COLUMNS,
        "versions": {
            name: version(name)
            for name in [
                "lightgbm",
                "scikit-learn",
                "pandas",
                "numpy",
                "pyarrow",
            ]
        },
        **results,
    }

    output_dir = ROOT / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_parquet(
        output_dir / "lightgbm_cross_validation_predictions.parquet",
        index=False,
    )

    (ROOT / "reports/lightgbm_cross_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\nWAPE by fold:")
    for fold in results["folds"]:
        print(f"\nFold {fold['fold']}")
        for name, score in fold["metrics"].items():
            wape = score["wape"]
            display = "undefined" if wape is None else f"{wape:.2%}"
            print(f"  {name}: {display}")

    print("\nPooled metrics:")
    print(json.dumps(results["pooled_metrics"], indent=2))
    print("\nSaved reports/lightgbm_cross_validation.json")


if __name__ == "__main__":
    main()
