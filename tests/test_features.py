"""Verify feature values, historical boundaries, and series isolation."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.features import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_training_features,
)


def panel():
    return pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(offset + day),
            }
            for product, offset in [(10, 0), (20, 100)]
            for day, date in enumerate(pd.date_range("2024-03-28", periods=90))
        ]
    )


def test_known_feature_values():
    features = build_training_features(panel())
    row = features.loc[features.product_id.eq(10)].iloc[30]

    assert row["lag_1"] == 29
    assert row["lag_7"] == 23
    assert row["lag_14"] == 16
    assert row["lag_30"] == 0
    assert row["mean_7"] == 26
    assert row["mean_14"] == 22.5
    assert row["std_7"] == pytest.approx(2.0)
    assert row["day_of_week"] == row["dt"].dayofweek


def test_history_readiness_and_series_isolation():
    features = build_training_features(panel())

    for product, offset in [(10, 0), (20, 100)]:
        series = features.loc[features.product_id.eq(product)]

        assert not series.iloc[:30]["history_ready"].any()
        assert series.iloc[30:]["history_ready"].all()
        assert np.isnan(series.iloc[0]["lag_1"])
        assert series.iloc[30]["lag_30"] == offset


def test_current_and_future_sales_do_not_change_present_features():
    original = panel()
    changed = original.copy()
    cutoff = pd.Timestamp("2024-05-15")

    changed.loc[changed.dt.ge(cutoff), "sale_amount"] = 999999.0

    before = build_training_features(original)
    after = build_training_features(changed)

    pd.testing.assert_frame_equal(
        before.loc[before.dt.le(cutoff), NUMERIC_FEATURES],
        after.loc[after.dt.le(cutoff), NUMERIC_FEATURES],
    )


def test_input_order_does_not_change_features():
    original = panel()

    expected = build_training_features(original)
    actual = build_training_features(original.sample(frac=1, random_state=42))

    pd.testing.assert_frame_equal(expected, actual)


def test_input_is_not_mutated():
    original = panel()
    snapshot = original.copy(deep=True)

    build_training_features(original)

    pd.testing.assert_frame_equal(original, snapshot)


def test_target_and_stockout_fields_are_not_predictors():
    assert "sale_amount" not in FEATURE_COLUMNS
    assert "stock_hour6_22_cnt" not in FEATURE_COLUMNS
    assert "history_ready" not in FEATURE_COLUMNS


def test_missing_day_is_rejected():
    with pytest.raises(ValueError, match="gaps"):
        build_training_features(panel().drop(index=5))
