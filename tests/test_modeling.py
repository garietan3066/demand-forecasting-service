"""Test the training/inference boundary and model input contract."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.features import build_training_features
from demand_forecasting.modeling import (
    fit_forecaster,
    predict_forecaster,
)


@pytest.fixture(scope="module")
def experiment():
    frame = pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(10 + product + day % 7),
            }
            for product in [10, 20]
            for day, date in enumerate(pd.date_range("2024-03-28", periods=90))
        ]
    )

    features = build_training_features(frame)

    train = features.loc[features.dt.lt("2024-06-12") & features.history_ready].copy()

    validation = features.loc[features.dt.ge("2024-06-12")].copy()

    fitted = fit_forecaster(train)
    return fitted, train, validation


def test_predictions_are_finite_and_nonnegative(experiment):
    fitted, _, validation = experiment
    predictions = predict_forecaster(fitted, validation)

    assert predictions.shape == (len(validation),)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_prediction_does_not_require_target_column(experiment):
    fitted, _, validation = experiment

    expected = predict_forecaster(fitted, validation)
    actual = predict_forecaster(
        fitted,
        validation.drop(columns="sale_amount"),
    )

    np.testing.assert_allclose(expected, actual, rtol=0, atol=0)


def test_unrelated_outcome_columns_do_not_affect_prediction(experiment):
    fitted, _, validation = experiment

    changed = validation.copy()
    changed["sale_amount"] = 999999.0
    changed["stock_hour6_22_cnt"] = 16

    np.testing.assert_allclose(
        predict_forecaster(fitted, validation),
        predict_forecaster(fitted, changed),
        rtol=0,
        atol=0,
    )


def test_unknown_series_is_rejected(experiment):
    fitted, _, validation = experiment

    changed = validation.copy()
    changed["product_id"] = 999

    with pytest.raises(ValueError, match="Unknown"):
        predict_forecaster(fitted, changed)

    assert 999 not in fitted.categories["product_id"]


def test_training_period_predictions_are_rejected(experiment):
    fitted, train, _ = experiment

    with pytest.raises(ValueError, match="training cutoff"):
        predict_forecaster(fitted, train.iloc[:1])


def test_each_fit_creates_a_separate_model(experiment):
    fitted, train, _ = experiment
    second = fit_forecaster(train)

    assert second.model is not fitted.model
    assert second.training_end == fitted.training_end


def test_nonfinite_prediction_features_are_rejected(experiment):
    fitted, _, validation = experiment
    changed = validation.copy()
    changed.loc[changed.index[0], "lag_1"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        predict_forecaster(fitted, changed)


def test_incomplete_training_history_is_rejected(experiment):
    _, train, _ = experiment
    changed = train.copy()
    changed.loc[changed.index[0], "history_ready"] = False

    with pytest.raises(ValueError, match="complete historical"):
        fit_forecaster(changed)
