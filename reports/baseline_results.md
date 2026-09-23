# First real-data baseline

## Protocol fixed before scoring

Selected 200 store-product pairs by the smallest SHA-256 hashes of `freshretail-v1:store:product`, from identifiers present on March 28, 2024. Selection does not depend on volumes, stockouts, or validation results. The manifest of selected pairs and library versions is in [baseline_metrics.json](baseline_metrics.json).

The subset contains 18,000 daily observations. Development validation covers June 12–25 (2,800 predictions). This is rolling one-day-ahead forecasting: predict each day with observations available through the preceding day. Earlier validation actuals may inform later predictions. It is not a 14-day forecast made at one fixed origin. The external evaluation split remains unopened.

## Observed-sales results

| Baseline | WAPE | MAE (normalized scale) |
|---|---:|---:|
| Yesterday's sales | 43.76% | 0.4673 |
| Same weekday last week | 45.39% | 0.4847 |
| Mean of preceding seven days | **36.33%** | **0.3879** |

The seven-day mean scores 36.33% WAPE in each weekly window after rounding. Its WAPE is 33.33% on days without stockouts in 06:00–22:00, and 40.35% on days with stockouts in that window. The latter labels are used retrospectively for scoring only, never as prediction inputs.

These are development results on a small subset and are not evidence of recovered demand accuracy, inventory savings, or generalization across the entire dataset. Low error against censored sales can still mean underestimating latent demand. The no-stockout segment covers only the stated window and is not a random sample of demand.

## Field decisions

The [publisher's data card](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) describes discount ratios and hourly out-of-stock flags but does not fully specify within-hour timing or values above one. The [official baseline README](https://github.com/Dingdong-Inc/frn-50k-baseline) does not resolve these questions either. The audit established flag polarity and count consistency, not event timing. Preserve these records. The current baseline does not use discounts or stockout flags as predictors and does not impute latent demand.

## Security and leakage verification

- Offline training-shard SHA-256 verification against the committed audit manifest, with path containment checks.
- Lag and rolling calculations group by store and product, sort dates, and shift before computing rolling statistics.
- Duplicate dates, daily gaps, invalid sales, and insufficient validation history fail explicitly.
- Tests modify current/future targets and confirm earlier predictions remain unchanged; separate tests verify series isolation and stable selection.
- Local generated data and individual predictions remain in ignored `data/processed/`.
- Dependency vulnerability results are recorded in `dependency_audit.json`; CI runs pytest and pip-audit with read-only permissions. Remote CI success must be checked separately.

## Reproduce

```powershell
.\venv\Scripts\python.exe notebooks/baseline_forecast.py
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pip_audit
```

Next: compare a LightGBM model using past sales and forecast-time calendar features against these exact baseline rows. Preserve this validation protocol and reserve the published evaluation split for a final check. Current dependencies are not fully locked; library versions are recorded for this run.
