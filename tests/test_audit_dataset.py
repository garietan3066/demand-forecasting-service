"""Small deliberately damaged records verify that the audit detects issues."""

import pandas as pd
import pyarrow as pa

from notebooks.audit_dataset import inspect_batch, summarize_series


def row():
    return dict(store_id=1, product_id=2, dt="2024-01-01", sale_amount=1.0,
                stock_hour6_22_cnt=1, holiday_flag=0, activity_flag=1, discount=1.0,
                hours_sale=[1.0] + [0.0] * 23,
                hours_stock_status=[0] * 6 + [1] + [0] * 17)


def test_hour_alignment_and_sales_totals():
    _, checks = inspect_batch(pa.Table.from_pylist([row()]))
    assert checks["daily_hourly_sales_mismatch"] == 0
    assert checks["stock_count_mismatch_ones_06_22"] == 0
    assert checks["stockout_hours_06_22"] == 1
    assert checks["stock_count_mismatch_zeros_06_22"] == 1


def test_malformed_arrays_and_values_are_reported():
    bad = row()
    bad.update(dt="invalid", sale_amount=-1.0, discount=1.5, hours_sale=[0.0],
               hours_stock_status=[2] * 24)
    _, checks = inspect_batch(pa.Table.from_pylist([bad]))
    assert checks["invalid_dates"] == 1
    assert checks["negative_daily_sales"] == 1
    assert checks["discount_outside_0_1"] == 1
    assert checks["invalid_length:hours_sale"] == 1
    assert checks["invalid_stock_flags"] == 24


def test_null_array_and_sales_mismatch_are_reported():
    bad = row()
    bad.update(sale_amount=5.0, hours_stock_status=None)
    _, checks = inspect_batch(pa.Table.from_pylist([row(), bad]))
    assert checks["null:hours_stock_status"] == 1
    assert checks["invalid_length:hours_stock_status"] == 1
    assert checks["daily_hourly_sales_mismatch"] == 1


def test_duplicate_does_not_hide_missing_day():
    first = row()
    third = dict(first, dt="2024-01-03")
    summary = summarize_series(pd.DataFrame([first, first, third]))
    assert summary["duplicate_key_rows_beyond_first"] == 1
    assert summary["internal_missing_dates"] == 1
    assert summary["series_with_internal_missing_dates"] == 1
