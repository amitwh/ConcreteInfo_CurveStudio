"""Tests for confidence and prediction bands (delta method)."""

import warnings
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import stats as sps

from curvestudio.data import XY
from curvestudio.fitting import fit, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import (
    Band,
    FitView,
    StatisticsError,
    build_view,
    confidence_band,
    prediction_band,
)

QUADRATIC = ModelSpec(
    id="quad_test",
    name="Quadratic",
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


def make_xy(x: Any, y: Any, weights: Any = None, fingerprint: str = "fp-bands") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=fingerprint,
    )


def noisy_line(n: int = 25, seed: int = 3) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    return make_xy(x, 2.0 + 1.5 * x + rng.normal(0.0, 0.8, n))


def line_view() -> FitView:
    data = noisy_line()
    return build_view(data, fit_polynomial(data, 1))


def width(band: Band) -> np.ndarray[Any, Any]:
    return np.asarray(band.upper) - np.asarray(band.lower)


def test_prediction_band_is_wider_than_confidence_band_everywhere() -> None:
    view = line_view()
    grid = np.linspace(-2.0, 12.0, 29)
    conf = confidence_band(view, grid)
    pred = prediction_band(view, grid)

    assert conf.kind == "confidence" and pred.kind == "prediction"
    assert np.all(width(pred) > width(conf))
    np.testing.assert_allclose(conf.fit, pred.fit)
    assert conf.df == pred.df == view.df


def test_bands_widen_toward_the_edges_of_a_line() -> None:
    view = line_view()
    mean_x = float(view.x.mean())
    grid = np.array([float(view.x.min()), mean_x, float(view.x.max())])
    for band in (confidence_band(view, grid), prediction_band(view, grid)):
        w = width(band)
        assert w[0] > w[1]
        assert w[2] > w[1]


def test_confidence_band_equals_the_textbook_line_formula() -> None:
    view = line_view()
    x0 = np.array([1.0, 4.0, 9.5])
    n = view.x.size
    sxx = float(np.sum((view.x - view.x.mean()) ** 2))
    assert view.s2 is not None
    se = np.sqrt(view.s2 * (1.0 / n + (x0 - view.x.mean()) ** 2 / sxx))
    q = sps.t.ppf(0.975, view.df)
    band = confidence_band(view, x0)
    np.testing.assert_allclose(band.upper, np.asarray(band.fit) + q * se, rtol=1e-10)
    np.testing.assert_allclose(band.lower, np.asarray(band.fit) - q * se, rtol=1e-10)


def test_level_controls_the_width() -> None:
    view = line_view()
    grid = np.array([3.0, 6.0])
    narrow = confidence_band(view, grid, level=0.80)
    wide = confidence_band(view, grid, level=0.99)
    assert narrow.level == 0.80 and wide.level == 0.99
    assert np.all(width(wide) > width(narrow))


def test_noiseless_data_gives_zero_width_bands_without_warnings() -> None:
    x = np.linspace(0.0, 5.0, 12)
    data = make_xy(x, 1.0 + 2.0 * x - 0.5 * x**2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view = build_view(data, fit_polynomial(data, 2))
        grid = np.array([0.0, 2.5, 5.0, 7.0])
        for band in (confidence_band(view, grid), prediction_band(view, grid)):
            assert np.all(np.isfinite(width(band)))
            np.testing.assert_allclose(width(band), 0.0, atol=1e-6)
            np.testing.assert_allclose(band.fit, 1.0 + 2.0 * grid - 0.5 * grid**2, atol=1e-8)


def test_extrapolated_points_are_flagged() -> None:
    view = line_view()  # active x range is [0, 10]
    grid = np.array([-0.5, 0.0, 5.0, 10.0, 10.5])
    band = confidence_band(view, grid)
    assert band.extrapolated == [True, False, False, False, True]
    assert prediction_band(view, grid).extrapolated == band.extrapolated


def test_zero_weight_points_do_not_extend_the_interpolation_range() -> None:
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 10.0, 21)
    y = 1.0 + 0.5 * x + rng.normal(0.0, 0.3, x.size)
    weights = np.ones_like(x)
    weights[-1] = 0.0  # x = 10 is inactive
    data = make_xy(x, y, weights)
    view = build_view(data, fit_polynomial(data, 1))
    band = confidence_band(view, np.array([9.0, 9.75, 10.0]))
    assert band.extrapolated == [False, True, True]


def test_new_weight_scales_prediction_variance() -> None:
    view = line_view()
    grid = np.array([1.0, 5.0, 9.0])
    conf = confidence_band(view, grid)
    unit = prediction_band(view, grid, new_weight=1.0)
    quad = prediction_band(view, grid, new_weight=4.0)
    q = sps.t.ppf(0.975, view.df)
    assert view.s2 is not None

    def variance(band: Band) -> np.ndarray[Any, Any]:
        return (width(band) / (2.0 * q)) ** 2

    added_unit = variance(unit) - variance(conf)
    added_quad = variance(quad) - variance(conf)
    np.testing.assert_allclose(added_unit, view.s2, rtol=1e-9)
    np.testing.assert_allclose(added_quad, added_unit / 4.0, rtol=1e-9)


def test_new_weight_may_vary_per_point() -> None:
    view = line_view()
    grid = np.array([1.0, 5.0, 9.0])
    weights = np.array([1.0, 2.0, 4.0])
    per_point = prediction_band(view, grid, new_weight=weights)
    for i, w in enumerate(weights):
        single = prediction_band(view, grid[i : i + 1], new_weight=float(w))
        assert per_point.lower[i] == pytest.approx(single.lower[0], rel=1e-12)
        assert per_point.upper[i] == pytest.approx(single.upper[0], rel=1e-12)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_level_is_rejected(level: float) -> None:
    view = line_view()
    for builder in (confidence_band, prediction_band):
        with pytest.raises(StatisticsError) as info:
            builder(view, np.array([1.0]), level=level)
        assert info.value.code == "INVALID_LEVEL"


@pytest.mark.parametrize(
    "x", [np.array([]), np.array([1.0, np.nan]), np.array([np.inf]), np.array([[1.0, 2.0]])]
)
def test_invalid_x_is_rejected(x: Any) -> None:
    view = line_view()
    for builder in (confidence_band, prediction_band):
        with pytest.raises(StatisticsError) as info:
            builder(view, x)
        assert info.value.code == "INVALID_X"


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_new_weight_is_rejected(weight: float) -> None:
    with pytest.raises(StatisticsError) as info:
        prediction_band(line_view(), np.array([1.0, 2.0]), new_weight=weight)
    assert info.value.code == "INVALID_WEIGHT"


def test_new_weight_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(StatisticsError) as info:
        prediction_band(line_view(), np.array([1.0, 2.0]), new_weight=np.array([1.0, 2.0, 3.0]))
    assert info.value.code == "INVALID_WEIGHT"


def test_view_without_degrees_of_freedom_or_covariance_is_refused() -> None:
    view = line_view()  # a real fit cannot have df == 0 (it is INVALID_INPUT), so adjust the view
    no_df = replace(view, df=0, s2=None)
    no_cov = replace(view, cov=None)
    for broken, code in ((no_df, "NO_DEGREES_OF_FREEDOM"), (no_cov, "NO_COVARIANCE")):
        for builder in (confidence_band, prediction_band):
            with pytest.raises(StatisticsError) as info:
                builder(broken, np.array([0.5]))
            assert info.value.code == code


def test_one_degree_of_freedom_still_gives_finite_bands() -> None:
    data = make_xy([0.0, 1.0, 2.0], [0.0, 1.1, 1.9])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view = build_view(data, fit_polynomial(data, 1))
        band = prediction_band(view, np.array([0.5, 1.5]))
    assert band.df == 1
    assert np.all(np.isfinite(band.lower)) and np.all(np.isfinite(band.upper))


def test_model_that_is_not_finite_at_x_is_rejected() -> None:
    log_model = ModelSpec(
        id="log_test",
        name="Log",
        category="test",
        expression="a + b*log(x)",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 0.0, "b": 1.0},
    )
    register(log_model)
    rng = np.random.default_rng(1)
    x = np.linspace(1.0, 5.0, 15)
    data = make_xy(x, 1.0 + 2.0 * np.log(x) + rng.normal(0.0, 0.05, x.size))
    view = build_view(data, fit(log_model, data))
    with pytest.raises(StatisticsError) as info:
        confidence_band(view, np.array([2.0, -1.0]))
    assert info.value.code == "INVALID_X"


def test_bands_of_a_nonlinear_fit_match_a_polynomial_fit_of_the_same_quadratic() -> None:
    rng = np.random.default_rng(8)
    x = np.linspace(-2.0, 4.0, 30)
    data = make_xy(x, 1.0 - 0.7 * x + 0.4 * x**2 + rng.normal(0.0, 0.3, x.size))
    poly_view = build_view(data, fit_polynomial(data, 2))
    nl_view = build_view(data, fit(QUADRATIC, data), QUADRATIC)
    grid = np.linspace(-3.0, 5.0, 17)

    for builder in (confidence_band, prediction_band):
        a, b = builder(poly_view, grid), builder(nl_view, grid)
        np.testing.assert_allclose(b.fit, a.fit, rtol=1e-6)
        np.testing.assert_allclose(b.lower, a.lower, rtol=1e-6)
        np.testing.assert_allclose(b.upper, a.upper, rtol=1e-6)
        assert b.extrapolated == a.extrapolated


def test_weighted_fit_bands_use_the_weighted_covariance() -> None:
    rng = np.random.default_rng(2)
    x = np.linspace(1.0, 10.0, 30)
    y = 3.0 + 2.0 * x + rng.normal(0.0, 0.3, x.size) * x
    data = make_xy(x, y, 1.0 / x**2)
    view = build_view(data, fit_polynomial(data, 1))
    band = confidence_band(view, np.array([5.0]))
    jac = np.array([[1.0, 5.0]])
    assert view.cov is not None
    se = float(np.sqrt((jac @ view.cov @ jac.T)[0, 0]))
    q = sps.t.ppf(0.975, view.df)
    assert band.upper[0] - band.fit[0] == pytest.approx(q * se, rel=1e-12)


def test_band_is_frozen_and_documents_itself() -> None:
    band = confidence_band(line_view(), np.array([1.0, 2.0]))
    assert band.pointwise is True
    assert "pointwise" in band.note.lower()
    with pytest.raises(ValueError):
        band.level = 0.5  # type: ignore[misc]
