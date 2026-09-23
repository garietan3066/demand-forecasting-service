"""Boundary and malformed-input checks for the public prediction contract."""

import json

import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest, PredictionResponse

VALID_REQUEST = {
    "lag_1": 0,
    "lag_7": 12.5,
    "lag_14": 20,
    "lag_30": 30,
    "day_of_week": 0,
    "month": 1,
}


@pytest.mark.parametrize("weekday,month", [(0, 1), (6, 12)])
def test_valid_json_accepts_integer_and_fractional_demand(weekday, month):
    payload = dict(VALID_REQUEST, day_of_week=weekday, month=month)
    request = PredictionRequest.model_validate_json(json.dumps(payload))
    assert request.model_dump() == payload


@pytest.mark.parametrize("field", ["lag_1", "lag_7", "lag_14", "lag_30"])
@pytest.mark.parametrize(
    "value", [-1, float("nan"), float("inf"), -float("inf"), "12", True, None, []]
)
def test_rejects_invalid_lag(field, value):
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(dict(VALID_REQUEST, **{field: value}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("day_of_week", -1),
        ("day_of_week", 7),
        ("day_of_week", 1.0),
        ("day_of_week", True),
        ("month", 0),
        ("month", 13),
        ("month", "1"),
        ("month", False),
    ],
)
def test_rejects_invalid_calendar(field, value):
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate_json(json.dumps(dict(VALID_REQUEST, **{field: value})))


@pytest.mark.parametrize("field", list(VALID_REQUEST))
def test_requires_every_feature(field):
    payload = VALID_REQUEST.copy()
    del payload[field]
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(payload)


def test_rejects_unexpected_input():
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(dict(VALID_REQUEST, model_path="untrusted.joblib"))


def test_response_serializes_as_json():
    response = PredictionResponse(predicted_demand=12.5, model_version="demand-v1")
    assert json.loads(response.model_dump_json()) == {
        "predicted_demand": 12.5,
        "model_version": "demand-v1",
    }


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "12"])
def test_response_rejects_invalid_prediction(value):
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_demand=value, model_version="demand-v1")


@pytest.mark.parametrize("version", ["", "   ", "\n", "x" * 129, 1, None])
def test_response_requires_usable_model_version(version):
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_demand=0, model_version=version)
