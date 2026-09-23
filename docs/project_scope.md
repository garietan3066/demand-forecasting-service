# Project Scope and Acceptance Criteria

Status: Agreed implementation direction; model selection remains experimental.

## 1. Project objective

Build a reproducible forecasting service that uses historical sales and
stockout information to support inventory planning.

The project demonstrates the full workflow:
data validation, time-series evaluation, model development, PostgreSQL
integration, secure API serving, containerization, and deployment.

An inventory simulation will demonstrate how forecasts can inform ordering
decisions. It will not be presented as evidence of real-world cost savings.

## 2. Intended user and decision

The intended user is an inventory planner responsible for individual
products at individual stores.

The service provides a forecast to inform replenishment decisions.
It does not automatically place purchase orders.

Ordering decisions also require inventory position, incoming deliveries,
supplier lead time, and a policy for handling uncertainty.

## 3. Dataset

Source: Dingdong-Inc/FreshRetailNet-50K.

Dataset:
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

Paper:
https://arxiv.org/abs/2505.16319

License: CC BY 4.0. Retain attribution to the dataset authors.

The audited training split contains:
- 4,500,000 daily records.
- 50,000 store-product series.
- 90 consecutive days per series, March 28 through June 25, 2024.
- Daily sales, hourly sales, and hourly stockout indicators.

Original data remains unchanged. Local raw and processed data stays out
of Git. Source revisions and checksums are recorded.

## 4. Prediction target and horizon

### Initial target

Predict the next day's observed sales for a known store-product pair.

The prediction is expressed in the dataset's normalized sales scale.
It is not an original physical unit count or monetary amount.

### Forecast timing

A forecast is made after the previous day's observations are available,
for the following calendar day.

For historical experiments, we assume the preceding day's observations
are available before the forecast is made. The dataset does not establish
actual reporting delays, so this assumption must be documented.

### Evaluation protocol

Use rolling one-day-ahead forecasting.

Earlier validation observations may be used in later predictions only
after their observation dates have passed. They must not be used in
predictions made before those dates.

This is not a fixed-origin forecast of an entire future week or fortnight.

The initial scope covers known store-product pairs with sufficient history.
Unknown-series and insufficient-history behavior will be explicit.

## 5. Separate outputs

### Sales forecast

A prediction of observed sales. This is the first directly evaluated output.

### Demand estimate

An optional, separately evaluated estimate of purchasing requirements
hidden during stockouts.

Recovered demand is an estimate, not directly observed ground truth.
It must never be silently substituted for observed sales in reports.

### Replenishment recommendation

A simulated ordering decision based on forecasts and explicitly stated
inventory assumptions.

It is separate from the forecast itself and does not execute purchases.

## 6. Development subset

Use the existing deterministic 200-series subset for initial development.

Selection uses store-product identifiers present on the first training
date and a fixed hash rule, not validation sales or model performance.

The selected-pair manifest is recorded in reports/baseline_metrics.json.

Subset results must not be presented as performance across all 50,000
series. Any expansion of scope will be documented and evaluated separately.

## 7. Model evaluation

Use date-based expanding-window validation across all selected series.

Planned folds:

| Fold | Training dates | Validation dates |
|---|---|---|
| 1 | March 28–May 14, 2024 | May 15–28, 2024 |
| 2 | March 28–May 28, 2024 | May 29–June 11, 2024 |
| 3 | March 28–June 11, 2024 | June 12–25, 2024 |

Training rows without sufficient historical features are excluded.
Their earlier observations may still provide feature history.

A fresh model is fitted for every fold.

Baselines:
- Yesterday's sales.
- Sales from seven days earlier.
- Mean sales over the preceding seven days.

Initial machine-learning candidate:
- LightGBM, using the scikit-learn estimator interface.

Metrics:
- WAPE, with explicit handling when total actual sales is zero.
- MAE in normalized units.
- Signed bias, defined as prediction minus actual.
- Per-fold and pooled results.
- Per-series and stockout-segment diagnostics.

The same validation rows are used for every compared method.

The published evaluation split remains reserved for final assessment
after the model, features, and prediction protocol are frozen.

Development validation results may inform model selection, so they are
not an independent final assessment.

A baseline may be selected for serving if it outperforms LightGBM.
No improvement percentage is promised in advance.

## 8. Leakage-prevention requirements

- Split by calendar dates across series, not randomly by rows.
- Compute historical features within each store-product series.
- Shift observations before computing rolling statistics.
- Fit learned preprocessing only on each fold's training data.
- Use an inner training-period holdout if early stopping is required.
- Exclude target-date sales and realized target-date stockouts from inputs.
- Exclude realized future weather unless genuine forecast-time weather
  predictions are available and appropriately documented.
- Fit demand-recovery procedures only on permitted history.
- Test that changing future observations cannot change earlier features.
- Apply the same historical cutoff in offline and database-backed inference.
- Do not use the published evaluation outcomes to tune the model.

## 9. Data limitations

Sales may understate demand during stockouts.

Actual unmet demand is not directly observed. Controlled masking
experiments can test recovery methods but do not establish exact lost
demand during real stockouts.

Ninety training days do not support claims about annual seasonality.

Normalized sales cannot be interpreted as original product units without
a verified inverse transformation.

Hourly stockout flags can coexist with positive hourly sales.
Exact within-hour timing remains unresolved.

Some discount values exceed one. They are preserved pending a justified
interpretation and are excluded from the initial model.

This is fresh-retail data, not manufacturing-component demand data.

## 10. Planned architecture

Original Parquet data
  -> validated preparation
  -> reproducible feature pipeline and evaluation
  -> selected versioned model
  -> PostgreSQL historical observations
  -> authenticated FastAPI prediction service
  -> recorded forecasts
  -> separate replenishment simulation

Parquet remains the reference source for reproducible experiments.

PostgreSQL becomes the application's operational store after the modeling
contract is established.

Training and inference use the same feature definitions.

The API will retrieve historical observations rather than require callers
to construct lag features manually.

## 11. Security requirements

- Keep credentials and private configuration out of Git.
- Use parameterized database queries.
- Separate migration, ingestion, and serving database privileges.
- Restrict database access to the application network.
- Authenticate prediction requests.
- Use HTTPS for remote access.
- Enforce input, request-size, rate, and timeout limits.
- Load model artifacts only from a trusted build process.
- Verify artifact provenance and integrity.
- Avoid logging secrets and unnecessary raw payloads.
- Run tests, dependency scanning, and secret scanning in CI.
- Run containers without root privileges.
- Document backup, restore, and rollback procedures.

These are implementation requirements, not claims that every control
already exists.

## 12. Replenishment simulation boundary

The simulation will define:
- Review frequency.
- Supplier lead time.
- Starting inventory.
- Incoming orders.
- Safety-stock or quantile policy.
- Lost-sales versus backorder behavior.
- Any shelf-life or spoilage assumptions.
- The demand scenario used for evaluation.

Orders may depend only on information available at the simulated
decision time.

Results will be labeled as simulated. No actual inventory savings or
service-level improvement will be claimed without supporting evidence.

## 13. Out of scope for the first release

- Automatic purchase-order execution.
- Manufacturing bills of materials.
- Real-time streaming infrastructure.
- Cold-start forecasting without historical observations.
- Annual or long-term forecasting claims.
- Exact recovery of unobserved lost demand.
- Unverified currency savings or physical-unit ordering recommendations.
- Multi-tenant enterprise identity integration.

## 14. Completion criteria

### Data and modeling
- [ ] Data acquisition and preparation are reproducible.
- [ ] Training data has a documented audit and provenance.
- [ ] Date-based cross-validation and leakage tests pass.
- [ ] Baselines and LightGBM are compared on identical rows.
- [ ] Model choice is justified by recorded results.
- [ ] Final evaluation is performed using a frozen procedure.
- [ ] Model artifacts include feature, data, and version metadata.
- [ ] Demand recovery, if included, has separate assumptions and evaluation.

### Database and API
- [ ] Database creation is reproducible through migrations.
- [ ] Ingestion is validated and idempotent.
- [ ] Database-derived features match the reference pipeline.
- [ ] API predictions match offline predictions.
- [ ] Unknown series and insufficient history have defined behavior.
- [ ] Authentication, validation, and database permissions are tested.

### Delivery and operations
- [ ] Dependency and secret scans run in CI.
- [ ] Containers build and run from documented instructions.
- [ ] The deployed demonstration passes smoke tests.
- [ ] Backup restore and rollback procedures are tested.
- [ ] Logs and monitoring expose failures without leaking secrets.
- [ ] Replenishment outcomes are clearly labeled as simulation results.
- [ ] A new user can reproduce the documented demonstration.

## 15. Definition of success

Success means delivering an understandable, reproducible, tested service
with honest evidence about model performance and limitations.

It does not require LightGBM to win, every demand-recovery experiment to
succeed, or simulated improvements to be presented as real-world savings.