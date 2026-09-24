"""Check reserved evaluation coverage without fitting models or scoring."""

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa

from demand_forecasting.constants import KEYS
from notebooks.baseline_forecast import ROOT, batches, verified_shards


def main():
    training_paths, revision = verified_shards()
    reference = json.loads((ROOT / "reports/model_comparison.json").read_text(encoding="utf-8"))

    if reference["cache_revision"] != revision:
        raise ValueError("Training revision differs from model selection.")

    pairs = [(item["store_id"], item["product_id"]) for item in reference["selected_pairs"]]
    if len(pairs) != 200 or len(set(pairs)) != 200:
        raise ValueError("Expected exactly 200 unique selected series.")

    selected_index = pd.MultiIndex.from_tuples(pairs, names=KEYS)

    # Restrict evaluation discovery to the audited training cache directory.
    parents = {path.parent.resolve() for path in training_paths}
    if len(parents) != 1:
        raise ValueError("Training shards span multiple cache directories.")

    cache_directory = parents.pop()
    evaluation_paths = sorted(cache_directory.glob("*-eval*.arrow"))

    if not evaluation_paths:
        raise ValueError("No evaluation Arrow files beside audited training data.")

    required = KEYS + ["dt", "sale_amount", "stock_hour6_22_cnt"]
    sources = []
    schemas = []

    for path in evaluation_paths:
        if not path.resolve().is_relative_to(cache_directory):
            raise ValueError("Evaluation file resolves outside the cache directory.")

        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()

        with pa.memory_map(str(path), "r") as source:
            schema = pa.ipc.open_stream(source).schema
            if not set(required).issubset(schema.names):
                raise ValueError(f"Missing required columns in {path.name}.")
            schemas.append({name: str(schema.field(name).type) for name in required})

        sources.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )

    if any(schema != schemas[0] for schema in schemas):
        raise ValueError("Evaluation shard schemas differ.")

    chunks = []
    total_rows = 0
    all_dates = set()

    for batch in batches(evaluation_paths):
        if batch[KEYS + ["dt"]].isna().any().any():
            raise ValueError("Evaluation identifiers or dates contain nulls.")

        dates = pd.to_datetime(batch["dt"], errors="raise")
        if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
            raise ValueError("Expected timezone-naive daily dates.")

        all_dates.update(dates.tolist())
        batch = batch.copy()
        batch["dt"] = dates
        total_rows += len(batch)

        selected = pd.MultiIndex.from_frame(batch[KEYS]).isin(selected_index)
        chunks.append(batch.loc[selected].copy())

    evaluation = pd.concat(chunks, ignore_index=True)
    if evaluation.empty:
        raise ValueError("No selected series found in evaluation data.")

    if evaluation.duplicated(KEYS + ["dt"]).any():
        raise ValueError("Duplicate selected-series evaluation dates.")

    values = evaluation["sale_amount"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Selected evaluation sales must be finite and nonnegative.")

    stock = evaluation["stock_hour6_22_cnt"].to_numpy(dtype=float)
    if (
        not np.isfinite(stock).all()
        or (stock < 0).any()
        or (stock > 16).any()
        or (stock != np.floor(stock)).any()
    ):
        raise ValueError("Invalid selected-series stockout counts.")

    training_end = max(
        pd.to_datetime(batch["dt"], errors="raise").max() for batch in batches(training_paths)
    )
    dates = pd.DatetimeIndex(sorted(all_dates))
    expected_dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        dates.max(),
        freq="D",
    )

    if len(expected_dates) == 0 or not dates.equals(expected_dates):
        raise ValueError("Evaluation dates must continue directly after training.")

    expected = pd.MultiIndex.from_tuples(
        [(store, product, date) for store, product in pairs for date in expected_dates],
        names=KEYS + ["dt"],
    )
    actual = pd.MultiIndex.from_frame(evaluation[KEYS + ["dt"]])
    missing = expected.difference(actual)
    extra = actual.difference(expected)

    if len(missing) or len(extra):
        raise ValueError(
            f"Selected-series coverage mismatch: "
            f"{len(missing)} missing rows; {len(extra)} extra rows."
        )

    report = {
        "dataset": reference["dataset"],
        "cache_revision": revision,
        "sources": sources,
        "required_column_types": schemas[0],
        "training_end": training_end.date().isoformat(),
        "evaluation_start": dates.min().date().isoformat(),
        "evaluation_end": dates.max().date().isoformat(),
        "evaluation_days": len(dates),
        "total_evaluation_rows": total_rows,
        "selected_series": len(pairs),
        "selected_rows": len(evaluation),
        "selected_pairs": reference["selected_pairs"],
        "checks_passed": True,
        "scope": (
            "Schema and date checks plus selected-series coverage and value checks; "
            "not a full audit of every evaluation series."
        ),
        "integrity_note": (
            "Hashes record the local evaluation files for subsequent verification. "
            "They are not independently authenticated publisher checksums."
        ),
        "scores_computed": False,
    }

    output = ROOT / "reports/final_evaluation_data_check.json"
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("PASS: evaluation data checks.")
    print(f"Training ends: {report['training_end']}")
    print(f"Evaluation: {report['evaluation_start']} through {report['evaluation_end']}")
    print(f"Selected series: {len(pairs)}")
    print(f"Selected evaluation rows: {len(evaluation)}")
    print("No models fitted or final scores computed.")
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
