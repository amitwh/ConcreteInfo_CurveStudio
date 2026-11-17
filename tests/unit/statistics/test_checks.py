"""Tests for the shared argument checks of the statistics layer."""

from typing import Any

import numpy as np
import pytest

from curvestudio.statistics._checks import check_level, check_x
from curvestudio.statistics.errors import StatisticsError


@pytest.mark.parametrize("level", [0.95, 0.5, 0.001, 0.999])
def test_valid_level_is_accepted(level: float) -> None:
    assert check_level(level) is None


@pytest.mark.parametrize(
    "level",
    [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), float("-inf"), True, False],
)
def test_level_outside_the_open_unit_interval_is_rejected(level: Any) -> None:
    with pytest.raises(StatisticsError) as info:
        check_level(level)
    assert info.value.code == "INVALID_LEVEL"
    assert "level must lie strictly between 0 and 1" in str(info.value)


def test_valid_x_is_returned_as_a_finite_float64_1d_array() -> None:
    from_list = check_x([1.0, 2.0, 3.0])
    assert from_list.ndim == 1
    assert from_list.dtype == np.float64
    np.testing.assert_allclose(from_list, [1.0, 2.0, 3.0])

    from_ints = check_x(np.array([1, 2, 3]))
    assert from_ints.dtype == np.float64
    np.testing.assert_allclose(from_ints, [1.0, 2.0, 3.0])

    single = check_x(np.array([7.5]))
    assert single.shape == (1,)


@pytest.mark.parametrize("x", [[], [[]], [[1.0, 2.0], [3.0, 4.0]], [[1.0]], 5.0])
def test_empty_or_non_1d_x_is_rejected(x: Any) -> None:
    with pytest.raises(StatisticsError) as info:
        check_x(x)
    assert info.value.code == "INVALID_X"
    assert "non-empty one-dimensional array" in str(info.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_x_is_rejected(bad: float) -> None:
    with pytest.raises(StatisticsError) as info:
        check_x([1.0, bad, 3.0])
    assert info.value.code == "INVALID_X"
    assert "only finite values" in str(info.value)
