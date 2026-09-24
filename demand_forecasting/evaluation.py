"""Evaluate historical-sales baselines across chronological folds."""

import pandas as pd

from demand_forecasting.baselines import predict_baselines
from demand_forecasting.constants import BASELINES, KEYS
from demand_forecasting.metrics import metrics
from demand_forecasting.validation import expanding_date_folds


def evaluate_baselines(frame):
    """Return fold metrics and rolling one-day validation predictions."""
    # predict_baselines sorts and resets the index. Split that same frame
    # so positional fold indices remain aligned with predictions.
    predictions = predict_baselines(frame)
    folds = expanding_date_folds(predictions)

    fold_results = []
    validation_frames = []

    for fold in folds:
        validation = predictions.iloc[fold.validation_indices].copy()

        if validation[BASELINES].isna().any().any():
            raise ValueError(f"Fold {fold.number} has insufficient prediction history.")

        validation["fold"] = fold.number

        scores = {name: metrics(validation["sale_amount"], validation[name]) for name in BASELINES}

        fold_results.append(
            {
                "fold": fold.number,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                "training_history_rows": len(fold.train_indices),
                "validation_rows": len(validation),
                "metrics": scores,
            }
        )

        validation_frames.append(validation[KEYS + ["dt", "sale_amount", "fold"] + BASELINES])

    combined = pd.concat(validation_frames, ignore_index=True)

    if combined.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Validation observations overlap across folds.")

    pooled = {name: metrics(combined["sale_amount"], combined[name]) for name in BASELINES}

    return {"folds": fold_results, "pooled_metrics": pooled}, combined
