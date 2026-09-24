"""Verify diagnostic comparisons and observation joins."""

import pandas as pd
import pytest

from notebooks.analyze_forecast_errors import (
    attach_stockout_labels,
    compare_methods,
)


def example():
    return pd.DataFrame(
        {
            "store_id": [1, 1],
            "product_id": [10, 10],
            "dt": ["2024-06-12", "2024-06-13"],
            "sale_amount": [10.0, 20.0],
            "mean_previous_7_days": [9.0, 19.0],
            "lightgbm": [8.0, 18.0],
        }
    )


def observations():
    frame = example()[["store_id", "product_id", "dt", "sale_amount"]].copy()
    frame["stock_hour6_22_cnt"] = [0, 4]
    return frame


def test_positive_extra_error_means_model_is_worse():
    result = compare_methods(example())

    assert result["extra_absolute_error"] == pytest.approx(2.0)
    assert result["baseline"]["bias"] == pytest.approx(-1.0)
    assert result["lightgbm"]["bias"] == pytest.approx(-2.0)


def test_join_preserves_rows_and_stockout_labels():
    result = attach_stockout_labels(example(), observations())

    assert len(result) == 2
    assert result["stock_hour6_22_cnt"].tolist() == [0, 4]


def test_duplicate_observations_are_rejected():
    source = observations()
    source = pd.concat([source, source.iloc[:1]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate observation"):
        attach_stockout_labels(example(), source)


def test_missing_observations_are_rejected():
    with pytest.raises(ValueError, match="no matching"):
        attach_stockout_labels(example(), observations().iloc[:1])


def test_changed_source_sales_are_rejected():
    source = observations()
    source.loc[0, "sale_amount"] = 999.0

    with pytest.raises(ValueError, match="differ"):
        attach_stockout_labels(example(), source)
