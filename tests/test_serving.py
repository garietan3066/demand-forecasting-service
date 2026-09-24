"""Verify serving features match historical evaluation features."""

import pandas as pd
import pytest

from demand_forecasting.features import FEATURE_COLUMNS, build_training_features
from demand_forecasting.serving import build_next_day_features


def historical_panel():
    return pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(product + day % 9),
            }
            for product in [10, 20]
            for day, date in enumerate(pd.date_range("2024-01-01", periods=45))
        ]
    )


def test_serving_matches_offline_features_on_multiple_dates():
    panel = historical_panel()
    offline = build_training_features(panel)

    for forecast_date in pd.date_range("2024-01-31", periods=10):
        history = panel.loc[panel.dt.lt(forecast_date)].copy()
        actual = build_next_day_features(history, forecast_date)
        expected = offline.loc[offline.dt.eq(forecast_date), FEATURE_COLUMNS + ["dt"]].reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(actual, expected)
        assert "sale_amount" not in actual.columns


def test_shuffled_history_produces_identical_features_without_mutation():
    panel = historical_panel()
    history = panel.loc[panel.dt.lt("2024-02-01")].copy()
    snapshot = history.copy(deep=True)

    expected = build_next_day_features(history, "2024-02-01")
    actual = build_next_day_features(history.sample(frac=1, random_state=42), "2024-02-01")

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(history, snapshot)


def test_current_day_observations_are_rejected():
    history = historical_panel()

    with pytest.raises(ValueError, match="strictly before"):
        build_next_day_features(history, history.dt.max())


def test_insufficient_history_is_rejected():
    history = historical_panel()
    history = history.loc[history.dt.lt("2024-01-30")]

    with pytest.raises(ValueError, match="30 days"):
        build_next_day_features(history, "2024-01-30")


def test_stale_series_is_rejected():
    history = historical_panel()
    history = history.loc[history.dt.lt("2024-02-01")]
    history = history.loc[~(history.product_id.eq(10) & history.dt.eq("2024-01-31"))]

    with pytest.raises(ValueError, match="day before"):
        build_next_day_features(history, "2024-02-01")


def test_internal_missing_day_is_rejected():
    history = historical_panel()
    history = history.loc[history.dt.lt("2024-02-01")]
    history = history.loc[~(history.product_id.eq(10) & history.dt.eq("2024-01-15"))]

    with pytest.raises(ValueError, match="gaps"):
        build_next_day_features(history, "2024-02-01")


@pytest.mark.parametrize(
    "forecast_date",
    ["2024-02-01 12:00:00", "2024-02-01T00:00:00Z"],
)
def test_invalid_forecast_date_is_rejected(forecast_date):
    history = historical_panel()
    history = history.loc[history.dt.lt("2024-02-01")]

    with pytest.raises(ValueError, match="timezone-naive daily date"):
        build_next_day_features(history, forecast_date)
