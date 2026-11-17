"""Tests for Auto-fit screening."""

from collections.abc import Iterator

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.discovery import (
    AutoFitError,
    PolynomialRef,
    RegisteredRef,
    screen,
)
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry


def _xy(x: list[float], y: list[float], weights: list[float] | None = None) -> XY:
    return XY(
        x=np.array(x),
        y=np.array(y),
        weights=None if weights is None else np.array(weights),
        x_name="x",
        y_name="y",
        fingerprint="test",
    )


def _positive_x(x: np.ndarray, y: np.ndarray) -> list[str]:
    return ["x must be strictly positive for ln"] if np.any(x <= 0.0) else []


def _poly_spec(model_id: str, k: int) -> ModelSpec:
    names = [f"p{i}" for i in range(k)]
    return ModelSpec(
        id=model_id,
        name=model_id,
        category="growth",
        expression=" + ".join(f"{n}*x**{i}" for i, n in enumerate(names)),
        parameters=tuple(Parameter(n) for n in names),
        initial_guess=lambda x, y: dict.fromkeys(names, 1.0),
    )


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(
        ModelSpec(
            id="logarithmic",
            name="Logarithmic",
            category="logarithmic",
            expression="a + b*ln(x)",
            parameters=(Parameter("a"), Parameter("b")),
            initial_guess=lambda x, y: {"a": 0.0, "b": 1.0},
            domain_check=_positive_x,
        )
    )
    register(_poly_spec("line", 2))
    register(_poly_spec("cubic", 4))
    yield
    _clear_registry()


LOG = RegisteredRef(model_id="logarithmic")
LINE = RegisteredRef(model_id="line")
CUBIC = RegisteredRef(model_id="cubic")


def test_log_models_skipped_for_non_positive_x_with_reason() -> None:
    data = _xy([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8])
    result = screen(data, [LOG, LINE])
    assert result.runnable == [LINE]
    assert result.skipped == {"logarithmic": ("x must be strictly positive for ln",)}


def test_five_points_only_small_models_run() -> None:
    data = _xy([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
    candidates = [
        PolynomialRef(degree=1),
        PolynomialRef(degree=2),
        PolynomialRef(degree=3),
        PolynomialRef(degree=2, through_origin=True),
        PolynomialRef(degree=3, through_origin=True),
        LINE,
        CUBIC,
    ]
    result = screen(data, candidates)
    # k <= n - 3 = 2
    assert [c.id for c in result.runnable] == ["poly1", "poly2_origin", "line"]
    assert result.skipped["poly2"] == ("needs at least 6 active points, has 5",)
    assert result.skipped["cubic"] == ("needs at least 7 active points, has 5",)
    assert result.skipped["poly3_origin"] == ("needs at least 6 active points, has 5",)
    assert result.n_active == 5


def test_fewer_than_four_points_refuses_with_code() -> None:
    data = _xy([1, 2, 3], [1, 2, 3])
    with pytest.raises(AutoFitError) as excinfo:
        screen(data, [LINE])
    assert excinfo.value.code == "TOO_FEW_POINTS"
    assert isinstance(excinfo.value, ValueError)
    assert "found 3" in str(excinfo.value)


def test_four_points_are_screened_not_refused() -> None:
    result = screen(
        _xy([1, 2, 3, 4], [1, 2, 3, 4]),
        [PolynomialRef(degree=1, through_origin=True), PolynomialRef(degree=1), LINE],
    )
    assert [c.id for c in result.runnable] == ["poly1_origin"]  # k = 1 -> k + 3 = 4 points
    assert set(result.skipped) == {"poly1", "line"}


def test_zero_weight_points_do_not_count() -> None:
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = _xy(x, x, weights=[1, 1, 1, 0, 0, 0])
    with pytest.raises(AutoFitError, match="found 3"):
        screen(data, [LINE])
    # the zero-weight points are also invisible to applicability checks
    data = _xy([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0] * 8, [0, 0, 1, 1, 1, 1, 1, 1])
    result = screen(data, [LOG])
    assert result.runnable == [LOG]
    assert result.n_active == 6


def test_repeated_x_limits_models_by_distinct_values() -> None:
    data = _xy([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1, 2, 3, 4, 5, 6, 7, 8])
    result = screen(data, [PolynomialRef(degree=1), LINE])
    assert result.runnable == []
    assert result.skipped["line"] == ("needs at least 2 distinct x values, has 1",)


def test_polynomial_through_origin_needs_non_zero_x_values() -> None:
    data = _xy([0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0], [0, 0, 0, 0, 0, 1, 1, 1])
    result = screen(data, [PolynomialRef(degree=2, through_origin=True)])
    assert result.runnable == []
    (reason,) = result.skipped["poly2_origin"]
    assert "through the origin" in reason and "non-zero" in reason


def test_screening_reports_every_reason() -> None:
    data = _xy([-1.0, -1.0, -1.0, -1.0], [1, 2, 3, 4])
    result = screen(data, [LOG])
    reasons = result.skipped["logarithmic"]
    assert "x must be strictly positive for ln" in reasons
    assert any("distinct" in r for r in reasons)
