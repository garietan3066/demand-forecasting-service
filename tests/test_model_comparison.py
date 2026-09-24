"""Check alternative models and two-stage prediction edge cases."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.features import FEATURE_COLUMNS
from notebooks.compare_models import fit_predict_linear_family


def training_frame():
    size = 70
    frame = pd.DataFrame(
        {
            "store_id": np.ones(size, dtype=int),
            "product_id": np.full(size, 10),
            "day_of_week": np.arange(size) % 7,
        }
    )

    for position, name in enumerate(FEATURE_COLUMNS):
        if name not in frame:
            frame[name] = 1.0 + (np.arange(size) % 11) * 0.1 + position * 0.01

    frame["sale_amount"] = np.where(
        np.arange(size) % 4 == 0,
        0.0,
        2.0 + (np.arange(size) % 5),
    )
    return frame


def test_candidates_return_finite_nonnegative_quantities():
    train = training_frame()
    outputs, metadata = fit_predict_linear_family(train, train[FEATURE_COLUMNS].iloc[:7])

    assert set(outputs) == {"linear", "ridge", "logistic_ridge"}
    assert metadata["classifier_status"] == "fitted"

    for values in outputs.values():
        assert values.shape == (7,)
        assert np.isfinite(values).all()
        assert (values >= 0).all()


@pytest.mark.parametrize(
    "target,status",
    [(0.0, "constant_zero"), (3.0, "constant_one")],
)
def test_single_class_training_is_handled(target, status):
    train = training_frame()
    train["sale_amount"] = target

    outputs, metadata = fit_predict_linear_family(train, train[FEATURE_COLUMNS].iloc[:7])

    assert metadata["classifier_status"] == status
    np.testing.assert_allclose(outputs["logistic_ridge"], target)


def test_prediction_target_column_is_ignored():
    train = training_frame()
    prediction = train[FEATURE_COLUMNS].iloc[:7].copy()

    before, _ = fit_predict_linear_family(train, prediction)
    prediction["sale_amount"] = 999999.0
    after, _ = fit_predict_linear_family(train, prediction)

    for name in before:
        np.testing.assert_allclose(before[name], after[name])


def test_unknown_series_is_rejected():
    train = training_frame()
    prediction = train[FEATURE_COLUMNS].iloc[:7].copy()
    prediction["product_id"] = 999

    with pytest.raises(ValueError, match="Unknown"):
        fit_predict_linear_family(train, prediction)
