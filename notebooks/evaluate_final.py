"""Evaluate the frozen forecasting methods on the reserved evaluation split."""

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from demand_forecasting.constants import KEYS
from demand_forecasting.ensemble import equal_weight_blend
from demand_forecasting.features import FEATURE_COLUMNS, build_training_features
from demand_forecasting.metrics import metrics
from demand_forecasting.modeling import (
    MODEL_PARAMETERS,
    fit_forecaster,
    predict_forecaster,
)
from notebooks.baseline_forecast import ROOT, batches, verified_shards
from notebooks.cross_validate_lightgbm import load_verified_subset

METHODS = ["mean_previous_7_days", "lightgbm", "blend_50_50"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(training, evaluation):
    """Fit only on training; use strictly historical features for evaluation."""
    training = training.copy()
    evaluation = evaluation.copy()

    if training.empty or evaluation.empty:
        raise ValueError("Training and evaluation must both contain observations.")

    for frame in [training, evaluation]:
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
        if frame[KEYS + ["dt"]].isna().any().any():
            raise ValueError("Missing identifiers or dates.")

    cutoff = training.dt.max()
    if not evaluation.dt.gt(cutoff).all():
        raise ValueError("Evaluation dates must follow the training cutoff.")

    pairs = set(training[KEYS].itertuples(index=False, name=None))
    evaluation_pairs = set(evaluation[KEYS].itertuples(index=False, name=None))
    if pairs != evaluation_pairs:
        raise ValueError("Training and evaluation series must match exactly.")

    dates = pd.date_range(cutoff + pd.Timedelta(days=1), evaluation.dt.max(), freq="D")
    expected = pd.MultiIndex.from_tuples(
        [(store, product, date) for store, product in sorted(pairs) for date in dates],
        names=KEYS + ["dt"],
    )
    actual = pd.MultiIndex.from_frame(evaluation[KEYS + ["dt"]])

    if (
        actual.has_duplicates
        or len(expected.difference(actual))
        or len(actual.difference(expected))
    ):
        raise ValueError("Evaluation coverage is incomplete or duplicated.")

    # Fitting cannot access evaluation observations.
    training_features = build_training_features(training)
    eligible = training_features.loc[training_features.history_ready].copy()
    fitted = fit_forecaster(eligible)

    # Historical features are shifted within each series. Evaluation sales
    # can therefore affect later dates, but never their own or earlier dates.
    combined = pd.concat([training, evaluation], ignore_index=True)
    features = build_training_features(combined)
    future = features.loc[features.dt.gt(cutoff)].copy()

    if not future.history_ready.all():
        raise ValueError("Insufficient evaluation feature history.")

    predictions = future[KEYS + ["dt", "sale_amount"]].copy()
    predictions["mean_previous_7_days"] = future["mean_7"].to_numpy()
    predictions["lightgbm"] = predict_forecaster(fitted, future[FEATURE_COLUMNS + ["dt"]])
    predictions["blend_50_50"] = equal_weight_blend(
        predictions["mean_previous_7_days"],
        predictions["lightgbm"],
    )

    metadata = {
        "training_end": cutoff.date().isoformat(),
        "usable_training_rows": len(eligible),
        "evaluation_start": future.dt.min().date().isoformat(),
        "evaluation_end": future.dt.max().date().isoformat(),
        "evaluation_rows": len(predictions),
        "selected_series": len(pairs),
    }
    return predictions.reset_index(drop=True), metadata


def main():
    report_path = ROOT / "reports/final_evaluation.json"
    predictions_path = ROOT / "data/processed/final_evaluation_predictions.parquet"

    if report_path.exists() or predictions_path.exists():
        raise ValueError(
            "Final evaluation output already exists. "
            "Preserve and review it before any intentional rerun."
        )

    # Record a committed implementation, not an uncommitted experiment.
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("Commit reviewed changes before running final evaluation.")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    manifest_path = ROOT / "reports/final_evaluation_data_check.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads((ROOT / "reports/model_comparison.json").read_text(encoding="utf-8"))

    if manifest.get("checks_passed") is not True:
        raise ValueError("Evaluation data checks have not passed.")
    if MODEL_PARAMETERS != comparison["parameters"]["lightgbm"]:
        raise ValueError("LightGBM parameters changed after model comparison.")

    training, reference = load_verified_subset()

    if (
        manifest["cache_revision"] != reference["cache_revision"]
        or manifest["selected_pairs"] != reference["selected_pairs"]
    ):
        raise ValueError("Evaluation manifest differs from the training reference.")

    training_paths, _ = verified_shards()
    parents = {path.parent.resolve() for path in training_paths}
    if len(parents) != 1:
        raise ValueError("Expected a single training cache directory.")
    cache = parents.pop()

    evaluation_paths = []
    for source in manifest["sources"]:
        name = source["file"]
        if Path(name).name != name or "-eval" not in name or not name.endswith(".arrow"):
            raise ValueError("Invalid evaluation shard basename.")

        path = cache / name
        if not path.resolve().is_relative_to(cache):
            raise ValueError("Evaluation shard resolves outside the cache.")

        if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
            raise ValueError("Evaluation file changed since the data check.")
        evaluation_paths.append(path)

    if not evaluation_paths or len(set(evaluation_paths)) != len(evaluation_paths):
        raise ValueError("Missing or repeated evaluation shards.")

    pairs = pd.MultiIndex.from_frame(pd.DataFrame(reference["selected_pairs"])[KEYS])
    chunks = []
    total_rows = 0
    for batch in batches(evaluation_paths):
        total_rows += len(batch)
        mask = pd.MultiIndex.from_frame(batch[KEYS]).isin(pairs)
        chunks.append(batch.loc[mask].copy())

    if total_rows != manifest["total_evaluation_rows"]:
        raise ValueError("Evaluation source row count changed.")

    evaluation = pd.concat(chunks, ignore_index=True)
    predictions, metadata = evaluate(training, evaluation)

    expected_metadata = {
        "training_end": manifest["training_end"],
        "evaluation_start": manifest["evaluation_start"],
        "evaluation_end": manifest["evaluation_end"],
        "evaluation_rows": manifest["selected_rows"],
        "selected_series": manifest["selected_series"],
    }
    if any(metadata[key] != value for key, value in expected_metadata.items()):
        raise ValueError("Final evaluation coverage differs from the checked manifest.")

    scores = {name: metrics(predictions.sale_amount, predictions[name]) for name in METHODS}
    daily = [
        {
            "date": date.date().isoformat(),
            "metrics": {name: metrics(group.sale_amount, group[name]) for name in METHODS},
        }
        for date, group in predictions.groupby("dt", sort=True)
    ]

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)

    report = {
        "dataset": reference["dataset"],
        "cache_revision": reference["cache_revision"],
        "implementation_commit": commit,
        "evaluation_manifest_sha256": sha256(manifest_path),
        "selection_document_sha256": sha256(ROOT / "docs/model_selection.md"),
        "predictions_sha256": sha256(predictions_path),
        "selected_method": "blend_50_50",
        "selected_pairs": reference["selected_pairs"],
        "protocol": (
            "Rolling one-day forecasts; one training-only fit; "
            "fixed model throughout evaluation; no evaluation-driven selection."
        ),
        "target": "Observed sales in normalized units, not latent demand",
        "feature_columns": FEATURE_COLUMNS,
        "model_parameters": MODEL_PARAMETERS,
        "blend_weights": {"lightgbm": 0.5, "mean_previous_7_days": 0.5},
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow"]
        },
        **metadata,
        "pooled_metrics": scores,
        "daily_metrics": daily,
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Final evaluation complete. Selected method remains blend_50_50.")
    print(json.dumps(metadata, indent=2))
    for name, score in scores.items():
        wape = score["wape"]
        label = "undefined" if wape is None else f"{wape:.2%}"
        print(f"{name}: WAPE={label}, MAE={score['mae']:.6f}, bias={score['bias']:.6f}")
    print(f"Saved {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
