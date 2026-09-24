"""Expanding-window validation with shared calendar boundaries across series."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from demand_forecasting.constants import KEYS


@dataclass(frozen=True)
class TemporalFold:
    """Positional row indices and date boundaries for one validation fold."""

    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def expanding_date_folds(
    frame: pd.DataFrame,
    n_splits: int = 3,
    validation_days: int = 14,
) -> list[TemporalFold]:
    """Split unique dates, then apply the split to every store-product series.

    This first version requires a complete daily panel: every series must
    contain one row for every date. Returned indices are positional and
    must be used with DataFrame.iloc on the original input frame.
    """
    required = KEYS + ["dt"]
    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")

    if frame[required].isna().any().any():
        raise ValueError("Series identifiers and dates must not be missing.")

    dates = pd.to_datetime(frame["dt"], errors="raise")

    if dates.isna().any():
        raise ValueError("Dates must not be missing.")

    if dates.dt.tz is not None:
        raise ValueError("Expected timezone-naive daily dates.")

    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Expected daily dates without time-of-day values.")

    keys_and_dates = frame[KEYS].copy()
    keys_and_dates["dt"] = dates

    if keys_and_dates.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Duplicate store-product-date records.")

    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    expected_dates = pd.date_range(
        unique_dates[0],
        unique_dates[-1],
        freq="D",
    )

    if not unique_dates.equals(expected_dates):
        raise ValueError("The dataset contains missing calendar dates.")

    counts = keys_and_dates.groupby(KEYS).size()

    if not counts.eq(len(unique_dates)).all():
        raise ValueError("Every series must cover the same complete daily calendar.")

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=validation_days,
        gap=0,
    )

    folds = []

    for number, (train_positions, validation_positions) in enumerate(
        splitter.split(unique_dates),
        start=1,
    ):
        train_dates = unique_dates[train_positions]
        validation_dates = unique_dates[validation_positions]

        train_indices = np.flatnonzero(dates.isin(train_dates).to_numpy())
        validation_indices = np.flatnonzero(dates.isin(validation_dates).to_numpy())

        folds.append(
            TemporalFold(
                number=number,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                validation_start=validation_dates[0],
                validation_end=validation_dates[-1],
            )
        )

    return folds
