# Final evaluation

## Procedure

The equal-weight blend was selected before final scoring.

LightGBM was fitted once on 12,000 eligible training observations,
using historical data through June 25, 2024.

Evaluation covered June 26–July 2, 2024:
200 previously selected store-product series and 1,400 observations.

The fitted model remained fixed throughout evaluation.
Predictions followed a rolling one-day-ahead protocol: earlier evaluation
sales could enter later historical features, but no evaluation observations
were used to fit the model.

## Results

| Method | WAPE | MAE | Signed bias |
|---|---:|---:|---:|
| Seven-day average | 38.16% | 0.426896 | +0.004936 |
| LightGBM | 37.28% | 0.417005 | -0.022848 |
| Equal-weight blend | 36.91% | 0.412905 | -0.008956 |

The preselected blend achieved approximately 3.3% lower absolute error
than the seven-day baseline on the same evaluation observations.

Its development WAPE was 34.39%, compared with 36.91% in final evaluation.
No model-selection or parameter changes were made in response to final scores.

The blend remains the selected method for the first release.

## Limitations

This assessment covers seven days and 200 selected series.
It does not establish performance across the full dataset, other seasons,
or other businesses.

The target is observed sales in normalized units, not original physical
quantities or ground-truth customer demand.

These results do not demonstrate inventory savings or an achieved
operational service level.

## Reproducibility

Detailed metrics, dependency versions, the implementation commit,
and integrity hashes are recorded in final_evaluation.json.

Row-level predictions remain in the ignored data/processed directory.