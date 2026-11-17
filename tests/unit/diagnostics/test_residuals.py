"""Tests for the residual set and influence measures (leverage, Cook's distance, PRESS)."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from curvestudio.data import XY
from curvestudio.diagnostics.residuals import (
    ResidualSet,
    influence_measures,
    leverage,
    press_rmse,
    residual_set,
)
from curvestudio.fitting import fit, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import StatisticsError, build_view

QUADRATIC = ModelSpec(
    id="quad_nl",
    name="Quadratic as a general model",
    category="test",
    expression="a + b*x + c*x**2",
    parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    initial_guess=lambda x, y: {"a": 0.0, "b": 0.0, "c": 0.0},
)


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(QUADRATIC)
    yield
    _clear_registry()


def make_xy(x: Any, y: Any, weights: Any = None) -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint="fp-residuals",
    )


def noisy_quadratic(n: int = 24, seed: int = 5) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 6.0, n)
    return x, 1.0 + 0.5 * x - 0.2 * x**2 + rng.normal(0.0, 0.3, n)


def poly_set(degree: int = 2, **kwargs: Any) -> ResidualSet:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    return residual_set(build_view(data, fit_polynomial(data, degree)), **kwargs)


def test_leverage_sums_to_the_parameter_count() -> None:
    result = poly_set(2)
    assert sum(result.leverage) == pytest.approx(3.0, abs=1e-12)


def test_leverage_is_in_unit_interval() -> None:
    result = poly_set(2)
    assert all(0.0 < h < 1.0 for h in result.leverage)


def test_point_with_extreme_x_is_flagged_high_leverage() -> None:
    x, y = noisy_quadratic()
    x = np.append(x, 40.0)
    y = np.append(y, 1.0 + 0.5 * 40.0 - 0.2 * 40.0**2)
    data = make_xy(x, y)
    result = residual_set(build_view(data, fit_polynomial(data, 1)))
    assert result.leverage_threshold == pytest.approx(2 * 2 / 25)
    assert len(x) - 1 in result.flagged_high_leverage
    assert result.leverage[-1] == max(result.leverage)


def test_zero_weight_points_do_not_enter_any_diagnostic() -> None:
    x, y = noisy_quadratic(n=20)
    weights = np.ones(20)
    weights[[3, 11]] = 0.0
    data = make_xy(x, y, weights)
    result = residual_set(build_view(data, fit_polynomial(data, 2)))
    n_active = 18
    for name in (
        "x",
        "fitted",
        "residuals",
        "weighted_residuals",
        "standardized",
        "studentized_external",
        "leverage",
        "cooks_distance",
        "scale_location",
        "qq_theoretical",
        "qq_ordered",
    ):
        assert len(getattr(result, name)) == n_active, name
    assert x[3] not in result.x and x[11] not in result.x
    assert sum(result.leverage) == pytest.approx(3.0, abs=1e-12)
    assert all(i < n_active for i in result.flagged_high_leverage + result.flagged_high_influence)


def test_leverage_one_points_give_none_not_nan() -> None:
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    h = leverage(jacobian)
    assert h[0] == pytest.approx(1.0, abs=1e-12)
    assert h[1:] == pytest.approx([0.5, 0.5])

    measures = influence_measures(
        weighted_residuals=np.array([0.0, 0.3, -0.3]), weighted_jacobian=jacobian, df=1, s2=0.18
    )
    assert measures.standardized[0] is None and measures.cooks_distance[0] is None
    assert measures.studentized_external[0] is None and measures.scale_location[0] is None
    assert measures.standardized[1] is not None and measures.cooks_distance[1] is not None
    assert measures.leverage_one == [0]
    for series in (
        measures.standardized,
        measures.studentized_external,
        measures.cooks_distance,
        measures.scale_location,
    ):
        assert not any(v is not None and np.isnan(v) for v in series)


def test_acf_lag_zero_is_one_and_follows_x_order() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    ordered = residual_set(build_view(data, fit_polynomial(data, 2)), nlags=6)
    assert ordered.acf[0] == 1.0 and len(ordered.acf) == 7
    assert ordered.acf_bound == pytest.approx(1.96 / np.sqrt(len(x)))

    perm = np.random.default_rng(0).permutation(len(x))
    shuffled_data = make_xy(x[perm], y[perm])
    shuffled = residual_set(build_view(shuffled_data, fit_polynomial(shuffled_data, 2)), nlags=6)
    np.testing.assert_allclose(shuffled.acf, ordered.acf, atol=1e-10)


def test_acf_is_the_demeaned_biased_autocorrelation() -> None:
    result = poly_set(2, nlags=3)
    r = np.asarray(result.weighted_residuals)[np.argsort(result.x, kind="stable")]
    r = r - r.mean()
    expected = [float(np.dot(r[: len(r) - k], r[k:]) / np.dot(r, r)) for k in range(4)]
    np.testing.assert_allclose(result.acf, expected, atol=1e-12)


def test_qq_theoretical_is_symmetric() -> None:
    result = poly_set(2)
    q = np.asarray(result.qq_theoretical)
    np.testing.assert_allclose(q, -q[::-1], atol=1e-12)
    assert list(result.qq_ordered) == sorted(result.qq_ordered)
    n = len(q)
    np.testing.assert_allclose(q, norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25)))


def test_nonlinear_leverage_matches_a_polynomial_of_the_same_model() -> None:
    x, y = noisy_quadratic()
    data = make_xy(x, y)
    poly = residual_set(build_view(data, fit_polynomial(data, 2)))
    nonlinear = residual_set(build_view(data, fit(QUADRATIC, data)))
    np.testing.assert_allclose(nonlinear.leverage, poly.leverage, rtol=1e-6)
    np.testing.assert_allclose(nonlinear.cooks_distance, poly.cooks_distance, rtol=1e-4)


def test_standardized_definitions_hold_for_a_weighted_fit() -> None:
    x, y = noisy_quadratic()
    weights = 1.0 / x**2
    data = make_xy(x, y, weights)
    view = build_view(data, fit_polynomial(data, 2))
    assert view.s2 is not None
    result = residual_set(view)
    r = np.sqrt(weights) * view.residuals
    np.testing.assert_allclose(result.weighted_residuals, r)
    h = np.asarray(result.leverage)
    std = r / (np.sqrt(view.s2) * np.sqrt(1 - h))
    np.testing.assert_allclose(result.standardized, std)
    df = view.df
    np.testing.assert_allclose(result.studentized_external, std * np.sqrt((df - 1) / (df - std**2)))
    np.testing.assert_allclose(result.cooks_distance, std**2 * h / (3 * (1 - h)))
    np.testing.assert_allclose(result.scale_location, np.sqrt(np.abs(std)))
    assert result.cooks_threshold == pytest.approx(4 / len(x))


def test_histogram_covers_every_residual_and_honours_explicit_bins() -> None:
    result = poly_set(2)
    assert len(result.hist_edges) == len(result.hist_counts) + 1
    assert sum(result.hist_counts) == len(result.x)
    assert len(poly_set(2, bins=5).hist_counts) == 5


def test_press_rmse_is_the_exact_leave_one_out_error_for_a_linear_model() -> None:
    x, y = noisy_quadratic(n=15)
    data = make_xy(x, y)
    view = build_view(data, fit_polynomial(data, 2))
    errors = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        coefficients = np.polynomial.polynomial.polyfit(x[keep], y[keep], 2)
        errors.append(y[i] - np.polynomial.polynomial.polyval(x[i], coefficients))
    assert press_rmse(view) == pytest.approx(float(np.sqrt(np.mean(np.square(errors)))), rel=1e-9)


def test_press_rmse_is_none_when_a_point_has_leverage_one() -> None:
    # the lone point at x = 0 alone determines the intercept, so its leverage is 1
    data = make_xy([0.0, 5.0, 5.0], [0.0, 1.0, 1.2])
    assert press_rmse(build_view(data, fit_polynomial(data, 1))) is None


def test_view_without_residual_degrees_of_freedom_is_refused() -> None:
    # build_view never yields df <= 0 for a converged fit; the guard protects hand-built views
    x, y = noisy_quadratic(n=6)
    data = make_xy(x, y)
    view = replace(build_view(data, fit_polynomial(data, 2)), df=0, s2=None)
    with pytest.raises(StatisticsError) as raised:
        residual_set(view)
    assert raised.value.code == "NO_DEGREES_OF_FREEDOM"


def test_noiseless_data_gives_no_nan_and_no_warning() -> None:
    x = np.linspace(0.0, 5.0, 12)
    data = make_xy(x, 2.0 + 3.0 * x)
    view = build_view(data, fit_polynomial(data, 1))
    with np.errstate(all="raise"):
        result = residual_set(view)
    for series in (result.standardized, result.cooks_distance, result.studentized_external):
        assert not any(v is not None and np.isnan(v) for v in series)
    assert all(np.isfinite(result.acf))


def test_invalid_nlags_and_bins_are_refused_and_large_nlags_clamped() -> None:
    x, y = noisy_quadratic(n=10)
    data = make_xy(x, y)
    view = build_view(data, fit_polynomial(data, 1))
    assert len(residual_set(view, nlags=500).acf) == 10
    with pytest.raises(StatisticsError) as raised:
        residual_set(view, nlags=-1)
    assert raised.value.code == "INVALID_ARGUMENT"
    with pytest.raises(StatisticsError):
        residual_set(view, bins=0)
