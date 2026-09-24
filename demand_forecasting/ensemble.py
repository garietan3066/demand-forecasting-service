"""Fixed averaging of two aligned forecast vectors."""

import numpy as np


def equal_weight_blend(baseline, model):
    """Average finite nonnegative forecasts with identical shapes."""
    baseline = np.asarray(baseline, dtype=float)
    model = np.asarray(model, dtype=float)

    if baseline.ndim != 1 or model.ndim != 1:
        raise ValueError("Predictions must be one-dimensional.")

    if baseline.shape != model.shape:
        raise ValueError("Prediction shapes must match.")

    if not np.isfinite(baseline).all() or not np.isfinite(model).all():
        raise ValueError("Predictions must be finite.")

    if (baseline < 0).any() or (model < 0).any():
        raise ValueError("Predictions must be nonnegative.")

    return 0.5 * baseline + 0.5 * model
