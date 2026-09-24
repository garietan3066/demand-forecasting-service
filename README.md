See [Project scope and acceptance criteria](docs/project_scope.md)
for the prediction target, evaluation protocol, security requirements,
and completion checklist.

## Run the baseline experiment

Reusable baseline and metric calculations live in `demand_forecasting/`.
The experiment runner is `notebooks/baseline_forecast.py`.

From the project root, run:

```powershell
.\.venv-check\Scripts\python.exe -m notebooks.baseline_forecast
```

## Verified Windows development environment

Use Python 3.11.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools
.\venv\Scripts\python.exe -m pip install --require-hashes -r requirements/development-windows-py311.txt
.\venv\Scripts\python.exe -m pytest -q
```

Quality checks:

```powershell
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m ruff format --check .
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m pip_audit
```

See [dependency management](requirements/README.md) for dependency updates.

The Windows lock is tested separately from the Linux compatibility job.
A Linux deployment lock will be introduced before container deployment.

# Supply Chain Demand Forecasting Service

A portfolio project connecting time-series modeling with a containerized prediction API. The intended use case is forecasting component demand to support inventory planning; inventory policy optimization is outside the current scope.

## Current status

A reproducible 200-series baseline now compares yesterday, last week, and a seven-day moving average with chronological validation. See [baseline results](reports/baseline_results.md) and [security and leakage controls](SECURITY.md). The baseline uses observed sales, not reconstructed demand. GitHub Actions is configured to run tests and dependency vulnerability checks.

Real-data exploration now uses FreshRetailNet-50K for retail forecasting with stockout information. The full training-data audit is in [reports/training_audit.md](reports/training_audit.md). The existing synthetic model and six-feature API schemas remain prototypes; they are not yet a model or serving contract validated for this dataset. Evaluation data has not been used by the audit.

Implemented: synthetic daily demand generation, lag features (1, 7, 14, and 30 days), day-of-week and month features, a chronological 80/20 split, LightGBM training, WAPE reporting, and model serialization.

Implemented API contracts: `PredictionRequest` and `PredictionResponse` in `app/schemas.py`, with tests for valid inputs and rejection of malformed values. Demand accepts integer or fractional JSON numbers but rejects numeric strings, booleans, negative values, NaN, and infinity. Calendar values require integers, and unexpected fields are rejected. Responses require finite nonnegative demand and a nonblank model version.

Not implemented yet: API endpoints, authentication, baseline comparisons, rolling validation, real-data ingestion, Docker packaging, and deployment. `app/main.py` remains a placeholder. This is a prototype, not yet a production-ready service.

Run the schema tests with `python -m pytest tests/test_schemas.py`.

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
