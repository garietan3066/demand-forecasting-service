"""Run baseline cross-validation using verified training shards only."""

import json
from importlib.metadata import version

import pandas as pd

from demand_forecasting.constants import BASELINES, KEYS
from demand_forecasting.evaluation import evaluate_baselines
from notebooks.baseline_forecast import (
    FIRST_DATE,
    LAST_DATE,
    ROOT,
    batches,
    choose_pairs,
    verified_shards,
)


def main():
    paths, revision = verified_shards()
    print("Training file hashes verified.", flush=True)

    # Candidate IDs come from the first training date only.
    first_day = pd.concat(
        [batch.loc[batch.dt.eq(FIRST_DATE), KEYS] for batch in batches(paths)],
        ignore_index=True,
    )

    pairs = choose_pairs(first_day, size=200)

    if len(pairs) != 200:
        raise ValueError("Expected 200 selected store-product pairs.")

    # Check that we are evaluating the same subset as the earlier experiment.
    reference = json.loads((ROOT / "reports/baseline_metrics.json").read_text(encoding="utf-8"))

    expected_pairs = [
        (item["store_id"], item["product_id"]) for item in reference["selected_pairs"]
    ]

    if pairs != expected_pairs:
        raise ValueError("Selected series differ from the baseline manifest.")

    if revision != reference["cache_revision"]:
        raise ValueError("Training revision differs from the baseline manifest.")

    selected_index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    chunks = []

    for batch in batches(paths):
        mask = pd.MultiIndex.from_frame(batch[KEYS]).isin(selected_index)
        chunks.append(batch.loc[mask])

    selected = pd.concat(chunks, ignore_index=True)
    selected["dt"] = pd.to_datetime(selected["dt"], errors="raise")

    if (
        len(selected) != 200 * 90
        or selected.dt.min() != pd.Timestamp(FIRST_DATE)
        or selected.dt.max() != pd.Timestamp(LAST_DATE)
    ):
        raise ValueError("Unexpected training-subset coverage.")

    results, predictions = evaluate_baselines(selected)

    report = {
        "dataset": "Dingdong-Inc/FreshRetailNet-50K",
        "source_split": "train",
        "cache_revision": revision,
        "selected_pairs": [dict(zip(KEYS, pair)) for pair in pairs],
        "target": "Observed daily sales in the published normalized scale",
        "protocol": (
            "Three expanding-history folds with rolling one-day predictions. "
            "Earlier validation actuals are available to later predictions. "
            "No learned model is fitted in this baseline experiment."
        ),
        "versions": {
            name: version(name) for name in ["pandas", "numpy", "scikit-learn", "pyarrow"]
        },
        **results,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    (reports_dir / "baseline_cross_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    output_dir = ROOT / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        output_dir / "baseline_cross_validation_predictions.parquet",
        index=False,
    )

    for fold in results["folds"]:
        print(f"\nFold {fold['fold']}: {fold['validation_start']} to {fold['validation_end']}")
        for name in BASELINES:
            score = fold["metrics"][name]
            wape = score["wape"]
            display = "undefined" if wape is None else f"{wape:.2%}"
            print(f"  {name}: WAPE={display}, MAE={score['mae']:.4f}")

    print("\nPooled metrics:")
    print(json.dumps(results["pooled_metrics"], indent=2))
    print("\nSaved reports/baseline_cross_validation.json")


if __name__ == "__main__":
    main()
