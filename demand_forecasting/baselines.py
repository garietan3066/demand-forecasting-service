"""Historical-sales baselines for rolling one-day-ahead forecasting."""

import numpy as np
import pandas as pd

from demand_forecasting.constants import KEYS


def predict_baselines(frame):
    """Return baseline predictions using only earlier observations."""
    frame = frame.copy()
    frame["dt"] = pd.to_datetime(frame.dt, errors="raise")
    frame = frame.sort_values(KEYS + ["dt"]).reset_index(drop=True)

    if frame[KEYS + ["dt", "sale_amount"]].isna().any().any():
        raise ValueError("Missing required values.")

    if frame.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Duplicate series dates.")

    gaps = frame.groupby(KEYS).dt.diff().dropna()
    if not gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Daily gaps must be resolved before using row-based lags.")

    if not np.isfinite(frame.sale_amount).all() or frame.sale_amount.lt(0).any():
        raise ValueError("Sales must be finite and nonnegative.")

    grouped = frame.groupby(KEYS).sale_amount

    frame["yesterday"] = grouped.shift(1)
    frame["last_week"] = grouped.shift(7)
    frame["mean_previous_7_days"] = grouped.transform(
        lambda sales: sales.shift(1).rolling(7, min_periods=7).mean()
    )

    return frame
