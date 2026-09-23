# Supply Chain Demand Forecasting Service

A portfolio project connecting time-series modeling with a containerized prediction API. The intended use case is forecasting component demand to support inventory planning; inventory policy optimization is outside the current scope.

## Current status

Implemented: synthetic daily demand generation, lag features (1, 7, 14, and 30 days), day-of-week and month features, a chronological 80/20 split, LightGBM training, WAPE reporting, and model serialization.

Not implemented yet: API schemas and endpoints, automated tests, baseline comparisons, rolling validation, real-data ingestion, Docker packaging, and deployment. Files in `app/` are placeholders. This is a prototype, not yet a production-ready service.

The current experiment evaluates one-day-ahead predictions with observed historical demand. It does not evaluate a fixed-origin multiday forecast. Calendar features are currently integer values, not cyclical sine/cosine encodings.

## Run the training pipeline

From the repository root, using Python with a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python notebooks/train_model.py
```

The script prints test WAPE and writes `artifacts/model.joblib`. WAPE is the sum of absolute errors divided by the sum of actual demand. Smaller values are better; the metric is undefined when total actual demand is zero. The current synthetic experiment is a development example, not evidence of real-world inventory savings.

## Development roadmap

Complete and validate each milestone before committing it. See [DEVELOPMENT.md](DEVELOPMENT.md) for acceptance checks and suggested commit messages.

1. Define the one-day-ahead inference contract and implement FastAPI.
2. Test validation, feature consistency, and model loading.
3. Add naive baselines, rolling time-based evaluation, and saved metrics.
4. Introduce a documented real demand dataset and component identifiers.
5. Package and verify the service with Docker.
6. Add continuous integration and a reproducible demo.
7. Deploy and document measured results and limitations.

## Planned API

- `GET /health`: liveness response.
- `GET /ready`: readiness response based on successful model loading.
- `POST /predict`: validated one-day-ahead demand prediction.

The first API version will accept `lag_1`, `lag_7`, `lag_14`, `lag_30`, `day_of_week` (0–6), and `month` (1–12). Lags must be finite, nonnegative demand values. Callers must supply observations from the correct dates. A later date-and-history interface can calculate features inside the service.

Only load trusted model artifacts. Joblib artifacts must not be accepted from arbitrary API clients.
