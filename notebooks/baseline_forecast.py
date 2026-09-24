"""Reproducible rolling one-day sales baselines on audited training data."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from demand_forecasting.baselines import predict_baselines
from demand_forecasting.constants import BASELINES, KEYS
from demand_forecasting.metrics import metrics

ROOT = Path(__file__).resolve().parents[1]
FIRST_DATE = "2024-03-28"
VALIDATION_START = "2024-06-12"
LAST_DATE = "2024-06-25"


def choose_pairs(first_day, size=200):
    """Select by stable ID hashes, never by sales or validation performance."""
    pairs = set(map(tuple, first_day[KEYS].to_numpy().tolist()))
    return sorted(
        pairs,
        key=lambda pair: (
            hashlib.sha256(f"freshretail-v1:{pair[0]}:{pair[1]}".encode()).hexdigest(),
            pair,
        ),
    )[:size]


def verify_file(path, expected_hash):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_hash:
        raise ValueError(
            f"Training file integrity mismatch: {path.name}; rerun and review the audit."
        )


def verified_shards():
    audit = json.loads((ROOT / "reports/training_audit.json").read_text())
    cache = (ROOT / "data/huggingface_cache").resolve()
    paths = []
    for source in audit["sources"]:
        name = source["file"]
        if Path(name).name != name or "-train-" not in name or not name.endswith(".arrow"):
            raise ValueError("Audit manifest must contain training shard basenames only.")
        matches = [p for p in cache.rglob(name) if p.parent.name == audit["cache_revision"]]
        if len(matches) != 1 or not matches[0].resolve().is_relative_to(cache):
            raise ValueError("Missing, ambiguous, or outside-cache training shard.")
        verify_file(matches[0], source["sha256"])
        paths.append(matches[0])
    if not paths:
        raise ValueError("No audited training shards.")
    return paths, audit["cache_revision"]


def batches(paths):
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            for batch in pa.ipc.open_stream(source):
                yield (
                    pa.Table.from_batches([batch])
                    .select(KEYS + ["dt", "sale_amount", "stock_hour6_22_cnt"])
                    .to_pandas()
                )


def main():
    paths, revision = verified_shards()
    print("Verified audited training hashes; selecting IDs from first date only.", flush=True)
    first_day = pd.concat([batch.loc[batch.dt.eq(FIRST_DATE), KEYS] for batch in batches(paths)])
    pairs = choose_pairs(first_day)
    selected_index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    chunks = []
    for batch in batches(paths):
        chunks.append(batch.loc[pd.MultiIndex.from_frame(batch[KEYS]).isin(selected_index)])
    selected = pd.concat(chunks, ignore_index=True)
    predictions = predict_baselines(selected)
    if (
        len(predictions) != len(pairs) * 90
        or predictions.dt.min() != pd.Timestamp(FIRST_DATE)
        or predictions.dt.max() != pd.Timestamp(LAST_DATE)
    ):
        raise ValueError("Unexpected subset coverage.")
    validation = predictions.loc[predictions.dt.ge(VALIDATION_START)].copy()
    if validation[BASELINES].isna().any().any():
        raise ValueError("Validation history is insufficient.")
    segments = {
        "all": np.ones(len(validation), dtype=bool),
        "no_stockout_06_22": validation.stock_hour6_22_cnt.eq(0),
        "some_stockout_06_22": validation.stock_hour6_22_cnt.gt(0),
        "week_1": validation.dt.lt("2024-06-19"),
        "week_2": validation.dt.ge("2024-06-19"),
    }
    results = {
        name: {
            segment: metrics(validation.loc[mask, "sale_amount"], validation.loc[mask, name])
            for segment, mask in segments.items()
        }
        for name in BASELINES
    }
    report = {
        "cache_revision": revision,
        "selection": "200 smallest SHA256 hashes of freshretail-v1:store:product; candidates from first training date only",
        "selected_pairs": [dict(zip(KEYS, pair)) for pair in pairs],
        "history_start": FIRST_DATE,
        "validation_start": VALIDATION_START,
        "validation_end": LAST_DATE,
        "subset_rows": len(selected),
        "protocol": "Rolling one-day ahead: earlier validation actuals become available for later predictions. Not a fixed-origin 14-day forecast.",
        "target": "Observed daily sales in normalized units; not ground-truth latent demand",
        "versions": {name: version(name) for name in ["pandas", "numpy", "pyarrow"]},
        "metrics": results,
    }
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output / "baseline_subset.parquet", index=False)
    validation.to_parquet(output / "baseline_validation_predictions.parquet", index=False)
    (ROOT / "reports/baseline_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
