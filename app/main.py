from fastapi import FastAPI

app = FastAPI(title="Demand Forecasting Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
