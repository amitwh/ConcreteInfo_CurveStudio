"""Tests for FitView / build_view: the shared, validated input of every statistic."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.fitting import FitStatus, fit, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, get_model, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import StatisticsError, build_view

EXP_GROWTH = ModelSpec(
    id="exp_growth",
    name="Exponential growth",
    category="test",
    expression="a*exp(b*x)",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": 1.0, "b": 0.5},
)
OTHER_MODEL = ModelSpec(
    id="saturating",
    name="Saturating",
    category="test",
    expression="v*x/(k + x)",
    parameters=(Parameter("v"), Parameter("k")),
    initial_guess=lambda x, y: {"v": 1.0, "k": 1.0},
)


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(EXP_GROWTH)
    register(OTHER_MODEL)
    yield
    _clear_registry()


def make_xy(x: Any, y: Any, weights: Any = None, fingerprint: str = "fp-view") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=fingerprint,
    )


def noisy_quadratic(n: int = 20, seed: int = 11) -> tuple[np.ndarray[Any, Any], ...]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 6.0, n)
    y = 1.0 + 0.5 * x - 0.2 * x**2 + rng.normal(0.0, 0.3, n)
    return x, y


def test_view_matches_fit_for_a_polynomial() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    result = fit_polynomial(data, 2)
    view = build_view(data, result)

    assert view.model is None
    assert view.result is result
    np.testing.assert_array_equal(view.x, x)
    np.testing.assert_array_equal(view.y, y)
    np.testing.assert_array_equal(view.w, np.ones_like(x))
    np.testing.assert_array_equal(view.fitted, result.fitted)
    np.testing.assert_array_equal(view.residuals, result.residuals)
    assert view.names == ("b0", "b1", "b2")
    np.testing.assert_array_equal(view.theta, [p.value for p in result.parameters])
    assert result.stats is not None and result.covariance is not None
    assert view.df == result.stats.df == 17
    assert view.s2 == pytest.approx(result.stats.sse / 17, rel=1e-14)
    assert view.s2 == pytest.approx(float(np.sum(view.residuals**2)) / 17, rel=1e-12)
    np.testing.assert_array_equal(view.cov, np.asarray(result.covariance))
    np.testing.assert_allclose(view.jacobian_at(x), np.vander(x, 3, increasing=True), rtol=1e-14)
    np.testing.assert_allclose(view.predict(x), result.fitted, rtol=1e-12)
    np.testing.assert_allclose(
        view.predict(np.array([0.0, 10.0])), np.polyval(view.theta[::-1], [0.0, 10.0])
    )


def test_polynomial_through_origin_uses_powers_from_one() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    result = fit_polynomial(data, 2, through_origin=True)
    view = build_view(data, result)

    assert view.names == ("b1", "b2")
    np.testing.assert_allclose(view.jacobian_at(x), np.column_stack([x, x**2]), rtol=1e-14)
    np.testing.assert_allclose(view.predict(x), result.fitted, rtol=1e-12)
    assert view.predict(np.array([0.0]))[0] == 0.0


def test_view_matches_fit_for_a_nonlinear_model() -> None:
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 4.0, 30)
    y = 2.5 * np.exp(0.6 * x) * (1.0 + rng.normal(0.0, 0.01, x.size))
    model = get_model("exp_growth")
    assert model is EXP_GROWTH
    data = make_xy(x, y)
    result = fit(model, data)
    assert result.status is FitStatus.CONVERGED

    for supplied in (model, "exp_growth", None):
        view = build_view(data, result, supplied)
        assert view.model is model
        assert view.names == tuple(p.name for p in model.parameters)
        theta = dict(zip(view.names, view.theta, strict=True))
        np.testing.assert_array_equal(view.jacobian_at(x), model.jacobian(x, **theta))
        np.testing.assert_array_equal(view.predict(x), model.evaluate(x, **theta))
    assert result.stats is not None
    assert view.df == result.stats.df
    assert view.s2 == pytest.approx(result.stats.sse / result.stats.df, rel=1e-14)
    assert view.cov is not None
    np.testing.assert_array_equal(view.cov, np.asarray(result.covariance))


def test_weighted_jacobian_scales_rows_by_root_weight() -> None:
    x, y = noisy_quadratic()
    w = 1.0 / x**2
    data = make_xy(x, y, w)
    view = build_view(data, fit_polynomial(data, 1))
    np.testing.assert_array_equal(view.w, w)
    np.testing.assert_allclose(
        view.weighted_jacobian, np.sqrt(w)[:, None] * np.vander(x, 2, increasing=True), rtol=1e-14
    )


def test_zero_weight_points_are_excluded_from_the_view() -> None:
    x, y = noisy_quadratic(12)
    w = np.ones(12)
    w[[2, 7]] = 0.0
    data = make_xy(x, y, w)
    result = fit_polynomial(data, 1)
    view = build_view(data, result)

    keep = w > 0
    assert view.x.size == view.y.size == view.w.size == view.fitted.size == 10
    np.testing.assert_array_equal(view.x, x[keep])
    np.testing.assert_array_equal(view.y, y[keep])
    np.testing.assert_array_equal(view.w, w[keep])
    np.testing.assert_array_equal(view.fitted, np.asarray(result.fitted)[keep])
    np.testing.assert_array_equal(view.residuals, np.asarray(result.residuals)[keep])
    assert view.df == 8


def test_view_arrays_are_read_only() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    view = build_view(data, fit_polynomial(data, 1))
    for array in (view.x, view.y, view.w, view.fitted, view.residuals, view.theta, view.cov):
        assert array is not None
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0.0


def test_unconverged_fit_is_refused() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    result = fit_polynomial(data, 1)
    for status in (FitStatus.NOT_CONVERGED, FitStatus.FAILED, FitStatus.INVALID_INPUT):
        bad = result.model_copy(update={"status": status})
        with pytest.raises(StatisticsError) as info:
            build_view(data, bad)
        assert info.value.code == "FIT_NOT_CONVERGED"


def test_wrong_data_is_refused() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    result = fit_polynomial(data, 1)

    other_fingerprint = make_xy(x, y, fingerprint="somewhere-else")
    with pytest.raises(StatisticsError) as info:
        build_view(other_fingerprint, result)
    assert info.value.code == "DATA_MISMATCH"

    fewer_points = make_xy(x[:-1], y[:-1])
    with pytest.raises(StatisticsError) as info:
        build_view(fewer_points, result)
    assert info.value.code == "DATA_MISMATCH"

    w = np.ones(x.size)
    w[0] = 0.0
    with pytest.raises(StatisticsError) as info:
        build_view(make_xy(x, y, w), result)
    assert info.value.code == "DATA_MISMATCH"


def test_wrong_model_is_refused() -> None:
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 3.0, 25)
    y = 2.0 * np.exp(0.5 * x) + rng.normal(0.0, 0.05, x.size)
    data = make_xy(x, y)
    result = fit("exp_growth", data)
    with pytest.raises(StatisticsError) as info:
        build_view(data, result, OTHER_MODEL)
    assert info.value.code == "MODEL_MISMATCH"


def test_missing_covariance_raises_only_when_required() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    result = fit_polynomial(data, 1)

    without_cov = result.model_copy(update={"covariance": None, "correlation": None})
    view = build_view(data, without_cov)  # must not raise
    assert view.cov is None
    with pytest.raises(StatisticsError) as info:
        view.require_covariance()
    assert info.value.code == "NO_COVARIANCE"

    assert result.stats is not None
    saturated = result.model_copy(update={"stats": result.stats.model_copy(update={"df": 0})})
    saturated_view = build_view(data, saturated)
    assert saturated_view.s2 is None
    with pytest.raises(StatisticsError) as info:
        saturated_view.require_covariance()
    assert info.value.code == "NO_DEGREES_OF_FREEDOM"

    assert build_view(data, result).require_covariance().shape == (2, 2)


def test_statistics_error_carries_code_and_message() -> None:
    err = StatisticsError("NO_COVARIANCE", "no covariance available")
    assert isinstance(err, ValueError)
    assert err.code == "NO_COVARIANCE"
    assert str(err) == "no covariance available"
