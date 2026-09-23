"""Validated contracts for one-day-ahead demand forecasting."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


Demand = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """Observed lags and calendar values for the day being forecast.

    Callers must align the lags to the forecast date. This feature-only
    contract cannot verify their dates or detect missing historical days.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    lag_1: Demand = Field(description="Observed demand one day before the forecast date.")
    lag_7: Demand = Field(description="Observed demand seven days before the forecast date.")
    lag_14: Demand = Field(description="Observed demand fourteen days before the forecast date.")
    lag_30: Demand = Field(description="Observed demand thirty days before the forecast date.")
    day_of_week: Annotated[int, Field(ge=0, le=6)] = Field(
        description="Forecast weekday: Monday=0 through Sunday=6."
    )
    month: Annotated[int, Field(ge=1, le=12)] = Field(
        description="Forecast month: January=1 through December=12."
    )


class PredictionResponse(BaseModel):
    """Validated service output; inference must handle negative model outputs."""

    model_config = ConfigDict(strict=True, extra="forbid")

    predicted_demand: Demand = Field(description="One-day-ahead forecast in demand units.")
    model_version: Annotated[str, Field(min_length=1, max_length=128, pattern=r"\S")] = Field(
        description="Identifier of the model artifact used for this prediction."
    )
