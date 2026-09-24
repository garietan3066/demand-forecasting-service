"""Diagnose saved validation errors without retraining or using evaluation data."""

import json

import numpy as np
import pandas as pd

from demand_forecasting.constants import KEYS
from demand_forecasting.metrics import metrics
from notebooks.baseline_forecast import ROOT

BASELINE = "mean_previous_7_days"
MODEL = "lightgbm"
JOIN_KEYS = KEYS + ["dt"]


def compare_methods(frame):
    """Compare both methods on exactly the same observations."""
    baseline = metrics(frame["sale_amount"], frame[BASELINE])
    model = metrics(frame["sale_amount"], frame[MODEL])

    baseline_error = (frame[BASELINE] - frame["sale_amount"]).abs().sum()

    model_error = (frame[MODEL] - frame["sale_amount"]).abs().sum()

    return {
        "rows": len(frame),
        "actual_sales_total": float(frame["sale_amount"].sum()),
        "baseline": baseline,
        "lightgbm": model,
        # Positive means LightGBM is worse; negative means it improves.
        "extra_absolute_error": float(model_error - baseline_error),
    }


def attach_stockout_labels(predictions, observations):
    """Join retrospective labels without changing prediction coverage."""
    predictions = predictions.copy()
    observations = observations.copy()

    predictions["dt"] = pd.to_datetime(predictions["dt"], errors="raise")
    observations["dt"] = pd.to_datetime(observations["dt"], errors="raise")

    if predictions[JOIN_KEYS].isna().any().any():
        raise ValueError("Prediction identifiers or dates are missing.")

    if observations[JOIN_KEYS].isna().any().any():
        raise ValueError("Observation identifiers or dates are missing.")

    if predictions.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate prediction keys.")

    if observations.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate observation keys.")

    merged = predictions.merge(
        observations[JOIN_KEYS + ["sale_amount", "stock_hour6_22_cnt"]].rename(
            columns={"sale_amount": "source_sales"}
        ),
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        raise ValueError("Some predictions have no matching observation.")

    if not np.allclose(
        merged["sale_amount"],
        merged["source_sales"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Saved prediction targets differ from source sales.")

    counts = merged["stock_hour6_22_cnt"]

    if counts.isna().any() or not counts.between(0, 16).all():
        raise ValueError("Invalid stockout-hour count.")

    if not counts.eq(np.floor(counts)).all():
        raise ValueError("Stockout-hour counts must be integers.")

    return merged.drop(columns=["_merge", "source_sales"])


def main():
    processed = ROOT / "data/processed"

    predictions = pd.read_parquet(processed / "lightgbm_cross_validation_predictions.parquet")
    observations = pd.read_parquet(processed / "baseline_subset.parquet")

    reference = json.loads(
        (ROOT / "reports/lightgbm_cross_validation.json").read_text(encoding="utf-8")
    )

    # Check saved prediction coverage against the experiment report.
    if set(predictions["fold"]) != {item["fold"] for item in reference["folds"]}:
        raise ValueError("Saved prediction folds do not match the report.")

    expected_pairs = {
        (item["store_id"], item["product_id"]) for item in reference["selected_pairs"]
    }

    for fold in reference["folds"]:
        part = predictions.loc[predictions["fold"].eq(fold["fold"])]
        dates = pd.to_datetime(part["dt"], errors="raise")

        if (
            len(part) != fold["validation_rows"]
            or dates.min() != pd.Timestamp(fold["validation_start"])
            or dates.max() != pd.Timestamp(fold["validation_end"])
            or set(part[KEYS].itertuples(index=False, name=None)) != expected_pairs
        ):
            raise ValueError("Saved prediction coverage changed.")

        for method in [BASELINE, MODEL]:
            actual = metrics(part["sale_amount"], part[method])
            expected = fold["metrics"][method]

            for name in ["wape", "mae", "bias"]:
                left, right = actual[name], expected[name]

                if left is None or right is None:
                    if left != right:
                        raise ValueError("Saved metrics changed.")
                elif not np.isclose(left, right, rtol=1e-12, atol=1e-12):
                    raise ValueError("Saved metrics changed.")

    merged = attach_stockout_labels(predictions, observations)

    segments = []
    series_rows = []

    for fold_number, part in merged.groupby("fold", sort=True):
        masks = {
            "all": pd.Series(True, index=part.index),
            "no_stockout_06_22": part.stock_hour6_22_cnt.eq(0),
            "some_stockout_06_22": part.stock_hour6_22_cnt.gt(0),
        }

        for segment, mask in masks.items():
            segments.append(
                {
                    "fold": int(fold_number),
                    "segment": segment,
                    **compare_methods(part.loc[mask]),
                }
            )

        for (store_id, product_id), series in part.groupby(KEYS):
            scores = compare_methods(series)

            series_rows.append(
                {
                    "fold": int(fold_number),
                    "store_id": int(store_id),
                    "product_id": int(product_id),
                    "rows": scores["rows"],
                    "actual_sales_total": scores["actual_sales_total"],
                    "baseline_wape": scores["baseline"]["wape"],
                    "lightgbm_wape": scores["lightgbm"]["wape"],
                    "baseline_bias": scores["baseline"]["bias"],
                    "lightgbm_bias": scores["lightgbm"]["bias"],
                    "extra_absolute_error": scores["extra_absolute_error"],
                }
            )

    per_series = pd.DataFrame(series_rows)

    report = {
        "source": "Saved development cross-validation predictions",
        "interpretation": (
            "Positive extra_absolute_error means LightGBM is worse. "
            "Stockout labels are retrospective diagnostics, not predictors. "
            "Groups are observational and do not establish causality."
        ),
        "segments": segments,
        "fold_series_summary": [],
    }

    for fold_number, part in per_series.groupby("fold", sort=True):
        delta = part["extra_absolute_error"]

        report["fold_series_summary"].append(
            {
                "fold": int(fold_number),
                "series_improved": int((delta < -1e-12).sum()),
                "series_worsened": int((delta > 1e-12).sum()),
                "series_tied": int((delta.abs() <= 1e-12).sum()),
                "net_extra_absolute_error": float(delta.sum()),
            }
        )

    reports_dir = ROOT / "reports"

    (reports_dir / "forecast_error_analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    per_series.to_csv(
        reports_dir / "forecast_errors_by_series.csv",
        index=False,
    )

    print("\nFold and stockout segments:")
    for item in segments:
        baseline = item["baseline"]["wape"]
        model = item["lightgbm"]["wape"]

        baseline_text = "undefined" if baseline is None else f"{baseline:.2%}"
        model_text = "undefined" if model is None else f"{model:.2%}"

        print(
            f"Fold {item['fold']} | {item['segment']} | "
            f"rows={item['rows']} | "
            f"baseline={baseline_text} | LightGBM={model_text} | "
            f"extra absolute error={item['extra_absolute_error']:.4f}"
        )

    print("\nSeries summary:")
    print(json.dumps(report["fold_series_summary"], indent=2))

    print("\nLargest LightGBM losses in fold 1:")
    print(
        per_series.loc[per_series.fold.eq(1)]
        .nlargest(10, "extra_absolute_error")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
