"""Tests for calendar-based forecasting validation boundaries."""

import numpy as np
import pandas as pd
import pytest

from demand_forecasting.constants import KEYS
from demand_forecasting.validation import expanding_date_folds


def make_panel():
    """Two series sharing the audited training calendar."""
    return pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(day + product),
            }
            for product in [10, 20]
            for day, date in enumerate(pd.date_range("2024-03-28", periods=90))
        ]
    )


def test_expected_fold_boundaries():
    folds = expanding_date_folds(make_panel())

    expected = [
        ("2024-05-14", "2024-05-15", "2024-05-28"),
        ("2024-05-28", "2024-05-29", "2024-06-11"),
        ("2024-06-11", "2024-06-12", "2024-06-25"),
    ]

    assert len(folds) == 3

    for fold, (train_end, validation_start, validation_end) in zip(folds, expected, strict=True):
        assert fold.train_start == pd.Timestamp("2024-03-28")
        assert fold.train_end == pd.Timestamp(train_end)
        assert fold.validation_start == pd.Timestamp(validation_start)
        assert fold.validation_end == pd.Timestamp(validation_end)


def test_training_precedes_validation_and_windows_do_not_overlap():
    frame = make_panel()
    previous_training = set()
    seen_validation = set()

    for fold in expanding_date_folds(frame):
        train = frame.iloc[fold.train_indices]
        validation = frame.iloc[fold.validation_indices]

        current_training = set(fold.train_indices)
        current_validation = set(fold.validation_indices)

        assert train.dt.max() < validation.dt.min()
        assert current_training.isdisjoint(current_validation)
        assert previous_training.issubset(current_training)
        assert seen_validation.isdisjoint(current_validation)

        previous_training = current_training
        seen_validation.update(current_validation)


def test_each_series_uses_identical_validation_dates():
    frame = make_panel()

    for fold in expanding_date_folds(frame):
        validation = frame.iloc[fold.validation_indices]
        expected_dates = set(pd.date_range(fold.validation_start, fold.validation_end))

        assert len(validation) == 2 * 14

        for _, series in validation.groupby(KEYS):
            assert set(series.dt) == expected_dates


def test_shuffled_rows_and_nonstandard_index_are_supported():
    frame = make_panel().sample(frac=1, random_state=42)
    frame.index = np.arange(len(frame)) * 10 + 100

    for fold in expanding_date_folds(frame):
        train = frame.iloc[fold.train_indices]
        validation = frame.iloc[fold.validation_indices]

        assert train.dt.max() == fold.train_end
        assert validation.dt.min() == fold.validation_start
        assert len(validation) == 28


def test_target_values_do_not_influence_splits():
    original = make_panel()
    changed = original.copy()
    changed["sale_amount"] = 999999.0

    for before, after in zip(
        expanding_date_folds(original),
        expanding_date_folds(changed),
        strict=True,
    ):
        np.testing.assert_array_equal(before.train_indices, after.train_indices)
        np.testing.assert_array_equal(before.validation_indices, after.validation_indices)


def test_duplicate_records_are_rejected():
    frame = make_panel()
    duplicated = pd.concat([frame, frame.iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        expanding_date_folds(duplicated)


def test_missing_date_for_one_series_is_rejected():
    frame = make_panel().drop(index=5)

    with pytest.raises(ValueError, match="complete daily calendar"):
        expanding_date_folds(frame)


def test_missing_date_for_all_series_is_rejected():
    frame = make_panel()
    frame = frame.loc[frame.dt.ne(pd.Timestamp("2024-04-10"))]

    with pytest.raises(ValueError, match="missing calendar dates"):
        expanding_date_folds(frame)


def test_insufficient_history_is_rejected():
    frame = make_panel()
    frame = frame.loc[frame.dt.lt(pd.Timestamp("2024-04-17"))]

    with pytest.raises(ValueError):
        expanding_date_folds(frame)
