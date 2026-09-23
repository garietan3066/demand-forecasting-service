# Step-by-step development guide

## 1. Build a working inference service

- Add Pydantic request and response models in `app/schemas.py` using the contract in the README.
- In `app/main.py`, load the trusted model once during application startup. Resolve its path relative to the project, not the caller's working directory.
- Create the prediction DataFrame with exactly the training feature names and order.
- Implement `/health`, `/ready`, and `/predict`. Return a finite, nonnegative prediction with a model version; document any clipping of negative predictions.
- Start with `python -m uvicorn app.main:app --reload` and exercise the endpoint through `/docs`.

Acceptance: a valid payload produces a prediction; invalid months, negative lags, missing fields, and nonfinite values are rejected. A missing artifact produces an actionable startup error.

Suggested commit: `feat: serve one-day demand predictions with FastAPI`

## 2. Verify the inference contract

- Add pytest tests for valid requests, invalid inputs, readiness, and missing artifacts.
- Check that the API prediction agrees with direct model prediction for the same features, accounting for documented postprocessing.
- Share feature definitions between training and serving to prevent drift.
- Run `python -m pytest` before committing.

Suggested commit: `test: cover prediction contracts and model loading`

## 3. Establish forecasting evidence

- Compare LightGBM with yesterday's demand and demand from seven days earlier.
- Add several rolling chronological evaluation windows; fit only on data preceding each window.
- Preserve a final untouched test period and specify whether evaluation is rolling one-day-ahead or fixed-origin multiday.
- Save WAPE and MAE, evaluation dates, dataset details, feature order, model settings, and library versions with the model.
- Handle zero-total-demand WAPE explicitly. Report actual results even if a baseline wins.
- Compare cyclical calendar features with the existing integer features as a measured experiment.

Suggested commit: `feat: add baseline comparisons and rolling forecast evaluation`

## 4. Use realistic component demand

- Select a dataset with a usable license and document its source, date range, units, and limitations.
- Define component IDs, dates, missing-day treatment, returns, and duplicate handling.
- Construct lag features within each component and preserve chronological boundaries.
- Report aggregate and component-level errors. Sales may understate demand during stockouts; document this limitation when relevant.

Suggested commit: `feat: train and evaluate on documented component demand data`

## 5. Containerize the service

- Add a Dockerfile and `.dockerignore`; exclude the local virtual environment, Git metadata, and unnecessary data.
- Use a reproducible dependency specification and a non-root runtime user.
- Include a known model artifact and its metadata; document how to rebuild it.
- Build the image, start the container, and exercise health, readiness, and prediction endpoints.

Suggested commit: `build: containerize the forecasting API`

## 6. Automate checks and prepare the demo

- Add GitHub Actions to run tests and verify the Docker build.
- Add a sample prediction request, architecture overview, measured results, and reproducible setup instructions to the README.
- Document cold-start behavior, input assumptions, and known limitations.

Suggested commit: `ci: verify tests and container builds`

## 7. Deploy and write the portfolio case study

- Choose a host, configure the service, and verify its public endpoints.
- Measure latency with a stated workload and environment. Record model accuracy separately from service performance.
- Add the demo URL and explain what is implemented versus future work.
- Describe inventory benefits as intended use unless a simulation or actual deployment establishes savings.

Suggested commit: `docs: publish deployment instructions and measured project results`

## Git workflow for every milestone

Work in small, genuine increments. Commit completed, checked changes; do not manufacture dates or split finished work just to inflate the history.

```powershell
git status
git diff
# Run the checks appropriate to this milestone.
git add <specific-files>
git diff --cached
git commit -m "<message describing the completed change>"
git push origin HEAD
```

Use explicit file paths when staging. Never commit credentials, `.env` files, the virtual environment, or restricted datasets. For larger changes, use a `codex/` feature branch and a pull request to record the motivation and validation.
