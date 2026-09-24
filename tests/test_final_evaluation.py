"""Test the final evaluation boundary using synthetic observations."""

import numpy as np
import pandas as pd
import pytest

from notebooks import evaluate_final as runner


def sample_data():
    frame = pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(5 + product + day % 7),
            }
            for product in [10, 20]
            for day, date in enumerate(pd.date_range("2024-01-01", periods=77))
        ]
    )
    cutoff = pd.Timestamp("2024-03-10")
    return (
        frame.loc[frame.dt.le(cutoff)].copy(),
        frame.loc[frame.dt.gt(cutoff)].copy(),
    )


def test_fit_happens_once_and_uses_training_only(monkeypatch):
    training, evaluation = sample_data()
    original = runner.fit_forecaster
    calls = []

    def checked_fit(frame):
        calls.append(frame.copy())
        return original(frame)

    monkeypatch.setattr(runner, "fit_forecaster", checked_fit)
    predictions, metadata = runner.evaluate(training, evaluation)

    assert len(calls) == 1
    assert calls[0].dt.max() == training.dt.max()
    assert len(calls[0]) == 2 * (70 - 30)
    assert metadata["evaluation_rows"] == 14
    assert np.isfinite(predictions[runner.METHODS].to_numpy()).all()
    assert (predictions[runner.METHODS].to_numpy() >= 0).all()


def test_current_and_future_targets_cannot_change_earlier_predictions():
    training, evaluation = sample_data()
    original, _ = runner.evaluate(training, evaluation)

    cutoff = evaluation.dt.min() + pd.Timedelta(days=3)
    changed = evaluation.copy()
    changed.loc[changed.dt.ge(cutoff), "sale_amount"] = 999999.0
    updated, _ = runner.evaluate(training, changed)

    pd.testing.assert_frame_equal(
        original.loc[original.dt.le(cutoff), runner.METHODS],
        updated.loc[updated.dt.le(cutoff), runner.METHODS],
    )


def test_earlier_evaluation_sales_enter_later_baseline_history():
    training, evaluation = sample_data()
    original, _ = runner.evaluate(training, evaluation)

    changed = evaluation.copy()
    first_date = evaluation.dt.min()
    changed.loc[changed.dt.eq(first_date) & changed.product_id.eq(10), "sale_amount"] += 7.0
    updated, _ = runner.evaluate(training, changed)

    next_day = first_date + pd.Timedelta(days=1)
    mask = original.dt.eq(next_day) & original.product_id.eq(10)
    difference = (
        updated.loc[mask, "mean_previous_7_days"].iloc[0]
        - original.loc[mask, "mean_previous_7_days"].iloc[0]
    )
    assert difference == pytest.approx(1.0)


def test_overlap_is_rejected():
    training, evaluation = sample_data()
    evaluation.loc[evaluation.index[0], "dt"] = training.dt.max()

    with pytest.raises(ValueError, match="cutoff"):
        runner.evaluate(training, evaluation)


def test_missing_evaluation_day_is_rejected():
    training, evaluation = sample_data()

    with pytest.raises(ValueError, match="coverage"):
        runner.evaluate(training, evaluation.iloc[1:])


def test_missing_series_is_rejected():
    training, evaluation = sample_data()

    with pytest.raises(ValueError, match="series"):
        runner.evaluate(training, evaluation.loc[evaluation.product_id.eq(10)])
