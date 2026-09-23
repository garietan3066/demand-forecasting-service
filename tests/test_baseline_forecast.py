import numpy as np
import pandas as pd
import pytest

from notebooks.baseline_forecast import choose_pairs, metrics, predict_baselines, verify_file


def sample():
    return pd.DataFrame([
        dict(store_id=1, product_id=product, dt=date, sale_amount=float(offset + day))
        for product, offset in [(1, 0), (2, 100)]
        for day, date in enumerate(pd.date_range("2024-01-01", periods=15))
    ])


def test_future_and_current_targets_cannot_change_present_prediction():
    original = sample()
    changed = original.copy()
    changed.loc[changed.dt.ge("2024-01-10"), "sale_amount"] = 9999
    columns = ["yesterday", "last_week", "mean_previous_7_days"]
    before, after = predict_baselines(original), predict_baselines(changed)
    pd.testing.assert_frame_equal(before.loc[before.dt.le("2024-01-10"), columns],
                                  after.loc[after.dt.le("2024-01-10"), columns])


def test_series_do_not_share_history_and_input_order_is_irrelevant():
    predictions = predict_baselines(sample().sample(frac=1, random_state=42))
    for product, offset in [(1, 0), (2, 100)]:
        series = predictions.loc[predictions.product_id.eq(product)]
        assert np.isnan(series.iloc[0].yesterday)
        assert series.iloc[7].last_week == offset
        assert series.iloc[7].mean_previous_7_days == offset + 3


def test_gaps_and_duplicates_fail_closed():
    with pytest.raises(ValueError, match="gaps"):
        predict_baselines(sample().drop(index=3))
    with pytest.raises(ValueError, match="Duplicate"):
        predict_baselines(pd.concat([sample(), sample().iloc[:1]]))


def test_selection_is_independent_of_sales_and_row_order():
    frame = sample()
    first = choose_pairs(frame, size=1)
    frame.sale_amount = 1e9
    assert choose_pairs(frame.sample(frac=1), size=1) == first


def test_zero_demand_wape_is_undefined_and_metrics_are_correct():
    assert metrics([0, 0], [1, 2])["wape"] is None
    assert metrics([1, 3], [2, 2]) == {"rows": 2, "wape": .5, "mae": 1., "bias": 0.}


def test_changed_training_file_is_rejected(tmp_path):
    path = tmp_path / "train.arrow"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        verify_file(path, "0" * 64)
