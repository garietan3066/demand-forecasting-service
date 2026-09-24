"""Verify fixed averaging and rejection of invalid prediction vectors."""

import numpy as np
import pytest

from demand_forecasting.ensemble import equal_weight_blend


def test_equal_weight_calculation():
    result = equal_weight_blend([10, 20], [14, 16])
    np.testing.assert_allclose(result, [12, 18])


def test_identical_predictions_remain_unchanged():
    result = equal_weight_blend([0, 5], [0, 5])
    np.testing.assert_allclose(result, [0, 5])


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="shapes"):
        equal_weight_blend([1, 2], [1])


def test_two_dimensional_inputs_are_rejected():
    with pytest.raises(ValueError, match="one-dimensional"):
        equal_weight_blend([[1, 2]], [[3, 4]])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_are_rejected(invalid):
    with pytest.raises(ValueError, match="finite"):
        equal_weight_blend([1], [invalid])


def test_negative_predictions_are_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        equal_weight_blend([-1], [2])
