"""Historical features for rolling one-day-ahead sales forecasting."""

import pandas as pd

from demand_forecasting.baselines import predict_baselines
from demand_forecasting.constants import BASELINES, KEYS

NUMERIC_FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "mean_7",
    "mean_14",
    "std_7",
    "day_of_week",
]

CATEGORICAL_FEATURES = KEYS.copy()
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_training_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features without dropping rows or learning from future values.

    This function prepares historical training/evaluation observations.
    The input must include observed sales for each row.

    A separate serving adapter will later construct the next-day feature
    row from history without requiring the target day's sales.
    """
    # Reuse existing validation, chronological sorting, and grouping rules.
    result = predict_baselines(frame)

    if not result["dt"].eq(result["dt"].dt.normalize()).all():
        raise ValueError("Expected daily dates without time-of-day values.")

    grouped = result.groupby(KEYS)["sale_amount"]

    result["lag_1"] = result["yesterday"]
    result["lag_7"] = result["last_week"]
    result["lag_14"] = grouped.shift(14)
    result["lag_30"] = grouped.shift(30)

    result["mean_7"] = result["mean_previous_7_days"]

    result["mean_14"] = grouped.transform(
        lambda sales: sales.shift(1).rolling(14, min_periods=14).mean()
    )

    result["std_7"] = grouped.transform(
        lambda sales: sales.shift(1).rolling(7, min_periods=7).std(ddof=0)
    )

    result["day_of_week"] = result["dt"].dt.dayofweek

    # Keep incomplete initial history visible. The model runner will decide
    # which training rows are eligible and verify validation coverage.
    result["history_ready"] = result[NUMERIC_FEATURES].notna().all(axis=1)

    return result.drop(columns=BASELINES)
