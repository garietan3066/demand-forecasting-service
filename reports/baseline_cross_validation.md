# Baseline cross-validation results

Evaluated 200 deterministic store-product series using three expanding
history windows and rolling one-day-ahead predictions.

| Fold | Training history | Validation |
|---|---|---|
| 1 | March 28–May 14, 2024 | May 15–28, 2024 |
| 2 | March 28–May 28, 2024 | May 29–June 11, 2024 |
| 3 | March 28–June 11, 2024 | June 12–25, 2024 |

Each fold contains 2,800 validation observations; pooled results contain
8,400 observations with no overlapping validation dates.

## Results

| Baseline | Fold 1 WAPE | Fold 2 WAPE | Fold 3 WAPE | Pooled WAPE |
|---|---:|---:|---:|---:|
| Yesterday | 42.45% | 42.53% | 43.76% | 42.93% |
| Last week | 44.63% | 42.42% | 45.39% | 44.17% |
| Previous seven-day average | 35.53% | 34.88% | 36.33% | 35.60% |

The seven-day average is strongest in all three folds.
Fold 3 reproduces the earlier single-window baseline results.

Pooled WAPE is calculated from all validation errors and actuals,
not by averaging the three fold percentages.

## Interpretation and limitations

These baselines do not fit a learned model. They use historical observations
available before each prediction date.

Earlier validation actuals can inform later one-day predictions.
This is not a fixed-origin forecast of the entire validation window.

Results measure observed sales in normalized units, not latent demand.
They apply to the selected subset and development periods.

The published evaluation split was not used.

## Reproduce

From the project root:

    python -m notebooks.cross_validate_baselines

Detailed metrics, selected series, and library versions are recorded in
baseline_cross_validation.json.