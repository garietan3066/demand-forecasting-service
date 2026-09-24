"""Compare fixed forecasting candidates on identical chronological folds."""

import json
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from demand_forecasting.baselines import predict_baselines
from demand_forecasting.constants import KEYS
from demand_forecasting.ensemble import equal_weight_blend
from demand_forecasting.features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_training_features,
)
from demand_forecasting.metrics import metrics
from demand_forecasting.modeling import (
    MODEL_PARAMETERS,
    fit_forecaster,
    predict_forecaster,
)
from demand_forecasting.validation import expanding_date_folds
from notebooks.baseline_forecast import ROOT
from notebooks.cross_validate_lightgbm import (
    compare_metrics,
    load_verified_subset,
)

BASELINE = "mean_previous_7_days"

METHODS = [
    BASELINE,
    "linear",
    "ridge",
    "logistic_ridge",
    "lightgbm",
    "blend_50_50",
]

# Weekday is categorical for the linear-family models.
LINEAR_CATEGORICAL = KEYS + ["day_of_week"]
LINEAR_NUMERIC = [name for name in NUMERIC_FEATURES if name != "day_of_week"]


def nonnegative(values):
    """Validate model output before applying the shared clipping rule."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite one-dimensional values.")

    return np.maximum(values, 0.0)


def fit_predict_linear_family(training, prediction_features):
    """Fit preprocessing and three candidates using training rows only."""
    if training.empty:
        raise ValueError("Training data is empty.")

    target = training["sale_amount"].to_numpy(dtype=float)

    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Targets must be finite and nonnegative.")

    # The existing experiment evaluates known store-product pairs only.
    known_pairs = set(training[KEYS].itertuples(index=False, name=None))
    requested_pairs = set(prediction_features[KEYS].itertuples(index=False, name=None))

    if not requested_pairs.issubset(known_pairs):
        raise ValueError("Unknown store-product pair.")

    for frame in [training, prediction_features]:
        if frame[FEATURE_COLUMNS].isna().any().any():
            raise ValueError("Missing model features.")

        if not np.isfinite(frame[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite model features.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), LINEAR_NUMERIC),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="error",
                    sparse_output=True,
                ),
                LINEAR_CATEGORICAL,
            ),
        ],
        sparse_threshold=1.0,
    )

    x_train = preprocessor.fit_transform(training[FEATURE_COLUMNS])
    x_prediction = preprocessor.transform(prediction_features[FEATURE_COLUMNS])

    linear = LinearRegression()
    linear.fit(x_train, target)

    ridge = Ridge(alpha=1.0, solver="lsqr")
    ridge.fit(x_train, target)

    positive = target > 0
    positive_count = int(positive.sum())

    if positive_count == 0:
        positive_probability = np.zeros(len(prediction_features))
        positive_amount = np.zeros(len(prediction_features))
        classifier_status = "constant_zero"
    else:
        # Conditional amount model is trained only on positive targets.
        amount_model = Ridge(alpha=1.0, solver="lsqr")
        amount_model.fit(x_train[positive], target[positive])
        positive_amount = nonnegative(amount_model.predict(x_prediction))

        if positive.all():
            positive_probability = np.ones(len(prediction_features))
            classifier_status = "constant_one"
        else:
            classifier = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                random_state=42,
            )

            # A convergence warning must not silently become a trusted score.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(x_train, positive.astype(int))

            positive_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
            positive_probability = classifier.predict_proba(x_prediction)[:, positive_column]
            classifier_status = "fitted"

    outputs = {
        "linear": nonnegative(linear.predict(x_prediction)),
        "ridge": nonnegative(ridge.predict(x_prediction)),
        "logistic_ridge": nonnegative(positive_probability * positive_amount),
    }

    metadata = {
        "training_rows": len(training),
        "positive_training_rows": positive_count,
        "zero_training_rows": int((~positive).sum()),
        "classifier_status": classifier_status,
    }

    return outputs, metadata


def main():
    selected, baseline_reference = load_verified_subset()

    model_reference = json.loads(
        (ROOT / "reports/lightgbm_cross_validation.json").read_text(encoding="utf-8")
    )
    blend_reference = json.loads(
        (ROOT / "reports/fixed_blend_cross_validation.json").read_text(encoding="utf-8")
    )

    features = build_training_features(selected)
    baselines = predict_baselines(selected)

    pd.testing.assert_frame_equal(
        features[KEYS + ["dt"]],
        baselines[KEYS + ["dt"]],
    )

    features[BASELINE] = baselines[BASELINE].to_numpy()

    fold_reports = []
    all_predictions = []

    for fold in expanding_date_folds(features):
        history = features.iloc[fold.train_indices]
        training = history.loc[history.history_ready].copy()
        validation = features.iloc[fold.validation_indices].copy()

        if not validation.history_ready.all():
            raise ValueError("Incomplete validation history.")

        if not training.dt.max() < validation.dt.min():
            raise ValueError("Training and validation time boundary is invalid.")

        # No validation target is passed to a predictor.
        prediction_features = validation[FEATURE_COLUMNS].copy()

        outputs, logistic_metadata = fit_predict_linear_family(
            training,
            prediction_features,
        )

        for name, values in outputs.items():
            validation[name] = values

        fitted = fit_forecaster(training)
        validation["lightgbm"] = predict_forecaster(
            fitted,
            validation[FEATURE_COLUMNS + ["dt"]],
        )

        validation["blend_50_50"] = equal_weight_blend(
            validation[BASELINE],
            validation["lightgbm"],
        )

        scores = {name: metrics(validation["sale_amount"], validation[name]) for name in METHODS}

        # Existing candidates must reproduce their previous scores.
        references = [
            (BASELINE, baseline_reference),
            ("lightgbm", model_reference),
            ("blend_50_50", blend_reference),
        ]

        for name, reference in references:
            expected = next(item for item in reference["folds"] if item["fold"] == fold.number)

            if (
                expected["validation_start"] != fold.validation_start.date().isoformat()
                or expected["validation_end"] != fold.validation_end.date().isoformat()
            ):
                raise ValueError("Reference validation dates changed.")

            compare_metrics(scores[name], expected["metrics"][name])

        fold_reports.append(
            {
                "fold": fold.number,
                "train_end": fold.train_end.date().isoformat(),
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                "usable_training_rows": len(training),
                "validation_rows": len(validation),
                "logistic_training": logistic_metadata,
                "metrics": scores,
            }
        )

        validation["fold"] = fold.number
        all_predictions.append(validation[KEYS + ["dt", "sale_amount", "fold"] + METHODS])

        print(f"Fold {fold.number} complete.", flush=True)

    combined = pd.concat(all_predictions, ignore_index=True)

    if combined.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Overlapping validation observations.")

    pooled = {name: metrics(combined["sale_amount"], combined[name]) for name in METHODS}

    report = {
        "dataset": baseline_reference["dataset"],
        "cache_revision": baseline_reference["cache_revision"],
        "selected_pairs": baseline_reference["selected_pairs"],
        "protocol": (
            "Fixed candidates; three expanding-window folds; "
            "rolling one-day predictions; training-only preprocessing; "
            "no hyperparameter search; published evaluation split unused."
        ),
        "target": "Observed sales in normalized units",
        "linear_preprocessing": {
            "numeric": LINEAR_NUMERIC,
            "categorical": LINEAR_CATEGORICAL,
            "scaling": "StandardScaler fitted within each training fold",
            "encoding": "OneHotEncoder, drop first, reject unknown categories",
        },
        "parameters": {
            "linear": "LinearRegression defaults",
            "ridge": {"alpha": 1.0, "solver": "lsqr"},
            "logistic_ridge": {
                "classifier": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 3000,
                    "random_state": 42,
                },
                "positive_amount": {"alpha": 1.0, "solver": "lsqr"},
            },
            "lightgbm": MODEL_PARAMETERS,
            "blend": "0.5 * LightGBM + 0.5 * seven-day average",
        },
        "postprocessing": "Clip quantity predictions below zero to zero",
        "versions": {
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm"]
        },
        "folds": fold_reports,
        "pooled_metrics": pooled,
    }

    (ROOT / "reports/model_comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    combined.to_parquet(
        ROOT / "data/processed/model_comparison_predictions.parquet",
        index=False,
    )

    table = pd.DataFrame([{"method": name, **score} for name, score in pooled.items()]).sort_values(
        "wape"
    )

    table["wape_percent"] = table["wape"] * 100

    print("\nPooled comparison — lower WAPE and MAE are better:")
    print(table[["method", "rows", "wape_percent", "mae", "bias"]].to_string(index=False))

    print("\nWAPE by fold:")
    for fold in fold_reports:
        print(f"\nFold {fold['fold']}")
        for name, score in fold["metrics"].items():
            value = score["wape"]
            text = "undefined" if value is None else f"{value:.2%}"
            print(f"  {name}: {text}")


if __name__ == "__main__":
    main()
