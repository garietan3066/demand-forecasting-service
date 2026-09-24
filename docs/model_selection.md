# Model selection for the first release

Status: Model selection frozen before final evaluation.

## Decision

Select the fixed equal-weight blend:

prediction = 0.5 * LightGBM + 0.5 * preceding-seven-day mean

The LightGBM component uses the parameters in
demand_forecasting/modeling.py.
The feature definitions remain in demand_forecasting/features.py.

No further model, feature, or blend-weight tuning is planned for this
release before final evaluation.

## Development evidence

All methods were evaluated on the same 8,400 observations from the
deterministically selected 200 store-product series.

Evaluation used three expanding-window chronological folds with
rolling one-day-ahead predictions.

| Method | Pooled WAPE |
|---|---:|
| Equal-weight blend | 34.39% |
| Logistic regression plus positive-sales Ridge | 35.03% |
| Ridge regression | 35.13% |
| Preceding-seven-day mean | 35.60% |
| LightGBM | 35.62% |
| Linear regression | 36.91% |

Source: reports/model_comparison.json.

The blend achieved the lowest WAPE in all three development folds.
Its pooled absolute error was approximately 3.4% lower than the
preceding-seven-day mean baseline.

These development results informed model selection. They are not an
independent final performance estimate or proof of statistical significance.

Logistic regression was used to estimate the probability of positive
sales. A separate Ridge model estimated the positive sales amount.
Logistic regression alone was not used as a quantity predictor.

## Frozen final-evaluation procedure

- Retain the existing 200-series selection without performance-based filtering.
- Retain the current features, LightGBM parameters, and equal blend weights.
- Fit a fresh LightGBM model using all eligible rows from the training split.
- Fit categorical mappings using training data only.
- Keep the fitted model fixed throughout final evaluation.
- Predict one day ahead using observations strictly before the target date.
- Earlier evaluation-day observations may enter subsequent days' history
  only after those dates have passed.
- Do not use evaluation observations to refit the model or select parameters.
- Evaluate the seven-day baseline, LightGBM, and selected blend on identical rows.
- Report WAPE, MAE, signed bias, observation counts, and evaluation dates.
- Report coverage problems explicitly; do not silently remove difficult series.
- Do not change the selected method based on final-evaluation scores.

Verify the evaluation data's schema, provenance, dates, and selected-series
coverage before computing final scores. Stop and document any incompatibility
rather than silently changing this procedure.

If final results expose a limitation, report it. Any subsequent modeling
changes constitute a new development cycle and require a new independent
evaluation strategy.

## Interpretation and limitations

The target is observed sales in the dataset's normalized scale.
Predictions are not original physical-unit quantities.

Sales during stockouts may understate customer demand.
This experiment does not measure accuracy against true unmet demand.

Results apply to the selected subset and evaluated periods, not automatically
to all 50,000 series or another business.

WAPE does not establish inventory savings or operational suitability.
Inventory-policy evaluation remains a separate simulation task.