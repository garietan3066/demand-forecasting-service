"""Train and apply a LightGBM model using an explicit feature contract."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_forecasting.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

MODEL_PARAMETERS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 2,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass
class FittedForecaster:
    """Model and category definitions learned from one training partition."""

    model: lgb.LGBMRegressor | lgb.Booster
    categories: dict[str, list]
    known_pairs: set[tuple]
    training_end: pd.Timestamp


def feature_matrix(frame, categories):
    """Select permitted inputs and apply training-derived categories."""
    matrix = frame[FEATURE_COLUMNS].copy()

    if matrix.isna().any().any():
        raise ValueError("Model features contain missing values.")

    numeric = matrix[NUMERIC_FEATURES].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Numeric features must be finite.")

    for column in CATEGORICAL_FEATURES:
        known = categories[column]

        if not matrix[column].isin(known).all():
            raise ValueError(f"Unknown category in {column}.")

        matrix[column] = pd.Categorical(
            matrix[column],
            categories=known,
        )

    return matrix


def fit_forecaster(training_frame):
    """Fit a fresh model using only the supplied training observations."""
    if training_frame.empty:
        raise ValueError("No training observations supplied.")

    if not training_frame["history_ready"].eq(True).all():
        raise ValueError("Training rows require complete historical features.")

    dates = pd.to_datetime(training_frame["dt"], errors="raise")

    if dates.isna().any():
        raise ValueError("Training dates must not be missing.")

    categories = {
        column: sorted(training_frame[column].dropna().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }

    matrix = feature_matrix(training_frame, categories)
    target = training_frame["sale_amount"].to_numpy(dtype=float)

    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Training targets must be finite and nonnegative.")

    model = lgb.LGBMRegressor(**MODEL_PARAMETERS)
    model.fit(
        matrix,
        target,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    pairs = set(training_frame[CATEGORICAL_FEATURES].itertuples(index=False, name=None))

    return FittedForecaster(
        model=model,
        categories=categories,
        known_pairs=pairs,
        training_end=dates.max(),
    )


def predict_forecaster(fitted, prediction_frame):
    """Predict later dates without reading their target values."""
    if prediction_frame.empty:
        raise ValueError("No prediction observations supplied.")

    dates = pd.to_datetime(prediction_frame["dt"], errors="raise")

    if dates.isna().any() or not dates.gt(fitted.training_end).all():
        raise ValueError("Prediction dates must follow the training cutoff.")

    pairs = set(prediction_frame[CATEGORICAL_FEATURES].itertuples(index=False, name=None))

    if not pairs.issubset(fitted.known_pairs):
        raise ValueError("Unknown store-product pair.")

    matrix = feature_matrix(prediction_frame, fitted.categories)
    raw_predictions = np.asarray(fitted.model.predict(matrix), dtype=float)

    if not np.isfinite(raw_predictions).all():
        raise ValueError("Model returned nonfinite predictions.")

    # Sales cannot be negative. Use this rule consistently in evaluation
    # and future serving, and record it in model metadata.
    return np.maximum(raw_predictions, 0.0)
