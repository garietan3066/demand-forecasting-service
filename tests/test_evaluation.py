"""Verify baseline evaluation alignment, pooling, and causality."""

import pandas as pd
import pytest

from demand_forecasting.constants import BASELINES
from demand_forecasting.evaluation import evaluate_baselines


def panel():
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


def test_fold_counts_and_known_baseline_errors():
    report, predictions = evaluate_baselines(panel())

    assert len(report["folds"]) == 3
    assert len(predictions) == 84
    assert predictions.groupby("fold").size().tolist() == [28, 28, 28]

    # Each series increases by exactly one every day.
    expected_errors = {
        "yesterday": 1.0,
        "last_week": 7.0,
        "mean_previous_7_days": 4.0,
    }

    for name, expected_mae in expected_errors.items():
        score = report["pooled_metrics"][name]
        assert score["rows"] == 84
        assert score["mae"] == pytest.approx(expected_mae)
        assert score["bias"] == pytest.approx(-expected_mae)

        expected_wape = expected_mae * len(predictions) / predictions["sale_amount"].sum()
        assert score["wape"] == pytest.approx(expected_wape)


def test_shuffling_input_preserves_results():
    original = panel()
    shuffled = original.sample(frac=1, random_state=42)

    original_report, original_predictions = evaluate_baselines(original)
    shuffled_report, shuffled_predictions = evaluate_baselines(shuffled)

    assert original_report == shuffled_report
    pd.testing.assert_frame_equal(original_predictions, shuffled_predictions)


def test_future_changes_do_not_change_earlier_predictions():
    original = panel()
    changed = original.copy()
    cutoff = pd.Timestamp("2024-06-01")

    changed.loc[changed.dt.ge(cutoff), "sale_amount"] = 99999.0

    _, before = evaluate_baselines(original)
    _, after = evaluate_baselines(changed)

    columns = ["store_id", "product_id", "dt"] + BASELINES

    # Predictions for the cutoff day must also exclude that day's actual.
    pd.testing.assert_frame_equal(
        before.loc[before.dt.le(cutoff), columns].reset_index(drop=True),
        after.loc[after.dt.le(cutoff), columns].reset_index(drop=True),
    )
