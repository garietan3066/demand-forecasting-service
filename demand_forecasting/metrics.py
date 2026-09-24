"""Metrics for evaluating forecasts against observed sales."""

import numpy as np


def metrics(actual, predicted):
    """Calculate WAPE, MAE, and mean signed error."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)

    if not len(actual):
        return {"rows": 0, "wape": None, "mae": None, "bias": None}

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite observations and predictions.")

    error = predicted - actual
    total = float(actual.sum())

    return {
        "rows": len(actual),
        "wape": float(np.abs(error).sum() / total) if total else None,
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
    }
