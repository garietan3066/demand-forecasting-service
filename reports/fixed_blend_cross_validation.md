# Fixed 50:50 blend experiment

Evaluated the arithmetic mean of LightGBM and seven-day-average
predictions on the existing three chronological validation folds.

No retraining or blend-weight search was performed.
Prediction coverage and original method metrics matched their references.
All 114 software tests passed.

## Results

| Method | Fold 1 WAPE | Fold 2 WAPE | Fold 3 WAPE | Pooled WAPE |
|---|---:|---:|---:|---:|
| Seven-day average | 35.53% | 34.88% | 36.33% | 35.60% |
| LightGBM | 36.52% | 34.10% | 36.22% | 35.62% |
| Fixed blend | 34.58% | 33.25% | 35.29% | 34.39% |

The blend improves WAPE and MAE in every fold.

Pooled WAPE improves by approximately 1.21 percentage points,
or 3.39% relative to the seven-day average.

Pooled signed bias is approximately:
- Seven-day average: -0.0134.
- LightGBM: -0.0278.
- Blend: -0.0206.

The blend reduces absolute error but underpredicts more on average
than the seven-day baseline.

## Decision and limitations

Retain the fixed blend as the leading observed-sales candidate.
Do not tune its weight based on these results.

The experiment was proposed after reviewing development-validation
errors. These scores are therefore model-selection evidence, not an
independent final assessment.

The published evaluation split remains unused.
No latent-demand accuracy or inventory savings are established.

## Reproduce

    python -m notebooks.evaluate_fixed_blend