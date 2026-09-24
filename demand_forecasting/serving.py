"""Build next-day prediction features from observed history."""

import pandas as pd

from demand_forecasting.constants import KEYS
from demand_forecasting.features import FEATURE_COLUMNS, build_training_features


def build_next_day_features(history, forecast_date):
    """Return one feature row per supplied store-product pair.

    Each series must have at least 30 consecutive daily observations,
    ending immediately before forecast_date.

    Current-day and future observations are rejected.
    """
    required = KEYS + ["dt", "sale_amount"]
    missing = set(required).difference(history.columns)
    if missing:
        raise ValueError(f"Missing history columns: {sorted(missing)}.")

    if history.empty:
        raise ValueError("History must not be empty.")

    target_date = pd.Timestamp(forecast_date)
    if (
        pd.isna(target_date)
        or target_date.tzinfo is not None
        or target_date != target_date.normalize()
    ):
        raise ValueError("Forecast date must be a timezone-naive daily date.")

    # Select only required historical fields and leave the caller's data intact.
    observed = history[required].copy()
    observed["dt"] = pd.to_datetime(observed["dt"], errors="raise")

    if observed[KEYS + ["dt", "sale_amount"]].isna().any().any():
        raise ValueError("History contains missing required values.")

    if observed.dt.dt.tz is not None or not observed.dt.eq(observed.dt.dt.normalize()).all():
        raise ValueError("History must contain timezone-naive daily dates.")

    if not observed.dt.lt(target_date).all():
        raise ValueError("History must be strictly before the forecast date.")

    # Reuse validation for duplicate dates, daily gaps, and invalid sales.
    validated = build_training_features(observed)

    coverage = validated.groupby(KEYS).dt.agg(["max", "count"])
    yesterday = target_date - pd.Timedelta(days=1)

    if not coverage["max"].eq(yesterday).all():
        raise ValueError("Every series must end the day before the forecast.")

    if not coverage["count"].ge(30).all():
        raise ValueError("Every series requires at least 30 days of history.")

    # This internal placeholder is never an observed sale or model feature.
    # All sales-derived features are shifted, so its value cannot affect
    # the target-date feature row. It is removed before returning.
    target_rows = validated[KEYS].drop_duplicates().copy()
    target_rows["dt"] = target_date
    target_rows["sale_amount"] = 0.0

    combined = pd.concat(
        [observed, target_rows[required]],
        ignore_index=True,
    )
    features = build_training_features(combined)
    result = features.loc[features.dt.eq(target_date)].copy()

    if not result.history_ready.all():
        raise ValueError("Incomplete forecast features.")

    return result[FEATURE_COLUMNS + ["dt"]].reset_index(drop=True)
