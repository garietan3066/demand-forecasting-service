# Initial LightGBM cross-validation

Compared a fixed LightGBM configuration with historical-sales baselines
on 200 store-product series.

Three chronological folds produced 8,400 rolling one-day validation
predictions. A fresh model was fitted in each fold. No early stopping
or hyperparameter search was used.

## Results

| Method | Fold 1 WAPE | Fold 2 WAPE | Fold 3 WAPE | Pooled WAPE |
|---|---:|---:|---:|---:|
| Seven-day average | 35.53% | 34.88% | 36.33% | 35.60% |
| LightGBM | 36.52% | 34.10% | 36.22% | 35.62% |

The baseline-parity check passed.

LightGBM wins two folds but does not improve pooled WAPE or MAE.
Its pooled signed bias is approximately -0.0278, compared with
-0.0134 for the seven-day average.

The seven-day average remains the reference method. The initial
LightGBM configuration has not demonstrated an overall advantage.

## Limitations

These are development results for observed normalized sales.
They do not establish latent-demand accuracy or inventory savings.

The published evaluation split remains unused.
The small performance difference does not establish statistical
superiority for either method.

Exact metrics, configuration, and library versions are recorded in
lightgbm_cross_validation.json.