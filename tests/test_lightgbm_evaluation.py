"""Check fold orchestration and reference-metric verification."""

import numpy as np
import pandas as pd
import pytest

from notebooks import cross_validate_lightgbm as experiment


def panel():
    return pd.DataFrame(
        [
            {
                "store_id": 1,
                "product_id": product,
                "dt": date,
                "sale_amount": float(product + day),
            }
            for product in [10, 20]
            for day, date in enumerate(pd.date_range("2024-03-28", periods=90))
        ]
    )


def test_runner_fits_separately_and_excludes_prediction_targets(monkeypatch):
    fits = []
    predictions = []

    def fake_fit(training):
        fits.append(training.copy())
        return training["dt"].max()

    def fake_predict(training_end, prediction_frame):
        assert "sale_amount" not in prediction_frame.columns
        assert prediction_frame["dt"].gt(training_end).all()
        predictions.append(prediction_frame.copy())
        return np.zeros(len(prediction_frame))

    monkeypatch.setattr(experiment, "fit_forecaster", fake_fit)
    monkeypatch.setattr(experiment, "predict_forecaster", fake_predict)

    report, combined = experiment.evaluate_lightgbm(panel())

    assert len(fits) == 3
    assert len(predictions) == 3

    # Two series, with their first 30 days excluded from fitting.
    assert [len(frame) for frame in fits] == [36, 64, 92]
    assert [len(frame) for frame in predictions] == [28, 28, 28]

    assert [frame.dt.max() for frame in fits] == [
        pd.Timestamp("2024-05-14"),
        pd.Timestamp("2024-05-28"),
        pd.Timestamp("2024-06-11"),
    ]

    assert len(combined) == 84
    assert report["pooled_metrics"]["lightgbm"]["rows"] == 84


def test_metric_comparison_rejects_changed_scores():
    original = {"rows": 10, "wape": 0.3, "mae": 0.2, "bias": -0.1}
    changed = dict(original, wape=0.4)

    with pytest.raises(ValueError, match="metric changed"):
        experiment.compare_metrics(changed, original)


def test_metric_comparison_rejects_changed_coverage():
    original = {"rows": 10, "wape": None, "mae": 0.0, "bias": 0.0}
    changed = dict(original, rows=9)

    with pytest.raises(ValueError, match="row count"):
        experiment.compare_metrics(changed, original)
