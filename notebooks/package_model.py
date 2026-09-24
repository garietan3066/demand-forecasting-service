"""Rebuild the selected training-only model and export a release package."""

import json
import subprocess

from demand_forecasting.artifacts import export_package, file_sha256
from demand_forecasting.features import FEATURE_COLUMNS, build_training_features
from demand_forecasting.modeling import MODEL_PARAMETERS, fit_forecaster
from notebooks.baseline_forecast import ROOT
from notebooks.cross_validate_lightgbm import load_verified_subset


def main():
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("Commit reviewed changes before building a release.")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    final_report_path = ROOT / "reports/final_evaluation.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))

    if final_report["selected_method"] != "blend_50_50":
        raise ValueError("Unexpected selected method.")
    if final_report["model_parameters"] != MODEL_PARAMETERS:
        raise ValueError("Model parameters differ from final evaluation.")
    if final_report["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Feature contract differs from final evaluation.")

    destination = ROOT / "artifacts/releases" / f"blend-v1-{commit[:12]}"
    build_report_path = ROOT / "reports/model_package_build.json"

    if destination.exists() or build_report_path.exists():
        raise ValueError("A package or build report already exists; review it first.")

    # Only audited training observations are loaded.
    selected, reference = load_verified_subset()
    if (
        reference["cache_revision"] != final_report["cache_revision"]
        or reference["selected_pairs"] != final_report["selected_pairs"]
    ):
        raise ValueError("Training data differs from final evaluation.")

    features = build_training_features(selected)
    training = features.loc[features.history_ready].copy()

    if (
        len(training) != final_report["usable_training_rows"]
        or training.dt.max().date().isoformat() != final_report["training_end"]
    ):
        raise ValueError("Training coverage differs from final evaluation.")

    fitted = fit_forecaster(training)

    source_files = [
        "demand_forecasting/features.py",
        "demand_forecasting/baselines.py",
        "demand_forecasting/modeling.py",
        "demand_forecasting/ensemble.py",
        "demand_forecasting/serving.py",
        "demand_forecasting/artifacts.py",
        "notebooks/package_model.py",
    ]

    provenance = {
        "implementation_commit": commit,
        "dataset": reference["dataset"],
        "cache_revision": reference["cache_revision"],
        "training_rows": len(training),
        "model_parameters": MODEL_PARAMETERS,
        "training_audit_sha256": file_sha256(ROOT / "reports/training_audit.json"),
        "final_evaluation_report_sha256": file_sha256(final_report_path),
        "source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
    }

    digests = export_package(fitted, destination, provenance)
    report = {
        "package_directory": destination.relative_to(ROOT).as_posix(),
        "implementation_commit": commit,
        "training_end": fitted.training_end.date().isoformat(),
        "training_rows": len(training),
        "known_series": len(fitted.known_pairs),
        **digests,
        "status": "Exported; verified loading and prediction parity pending",
    }

    build_report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
