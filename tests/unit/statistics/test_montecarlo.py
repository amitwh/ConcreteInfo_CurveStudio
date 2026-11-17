"""Tests for Monte-Carlo parameter sampling and uncertainty propagation."""

import math
from collections.abc import Iterator, Mapping
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
    FitView,
    Propagation,
    StatisticsError,
    build_view,
    confidence_band,
    monte_carlo_band,
    prediction_band,
    propagate,
    sample_parameters,
)

LINE = ModelSpec(
    id="mc_line",
    name="Straight line",
    category="test",
    expression="a + b*x",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": float(y[0]), "b": 1.0},
)


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(LINE)
    yield
    _clear_registry()


def _xy(x: Any, y: Any, tag: str = "mc") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None,
        x_name="x",
        y_name="y",
        fingerprint=f"mc-{tag}",
    )


def noisy_data(n: int = 25, seed: int = 3) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    return _xy(x, 2.0 + 1.5 * x + rng.normal(0.0, 0.8, n), tag=f"noisy-{n}-{seed}")


def line_view() -> FitView:
    """A view of a nonlinear fit of ``a + b*x``; the parameters are named ``a`` and ``b``."""
    data = noisy_data()
    return build_view(data, fit(LINE, data), LINE)


def polynomial_view() -> FitView:
    """A view of a least-squares polynomial fit (the ``b0``/``b1`` parameter names)."""
    data = noisy_data()
    return build_view(data, fit_polynomial(data, 1))


def half_width(lower: Any, fit: Any, upper: Any) -> np.ndarray[Any, Any]:
    """Half the width of a band, i.e. the distance from the fitted curve to a bound."""
    return (np.asarray(upper) - np.asarray(lower)) / 2.0


# --- sampling -----------------------------------------------------------------------


def test_sample_mean_and_covariance_recover_the_fit() -> None:
    view = line_view()
    n = 200_000
    draws = sample_parameters(view, n=n, seed=7)

    assert draws.shape == (n, view.theta.size)
    assert view.cov is not None
    mean = draws.mean(axis=0)
    standard_error = np.sqrt(np.diag(view.cov) / n)
    # three standard errors of the sample mean around the estimates
    assert np.all(np.abs(mean - view.theta) <= 3.0 * standard_error)

    sample_covariance = np.cov(draws, rowvar=False)
    np.testing.assert_allclose(sample_covariance, view.cov, rtol=0.02)


def test_same_seed_same_draws() -> None:
    view = line_view()
    first = sample_parameters(view, n=1000, seed=3)
    again = sample_parameters(view, n=1000, seed=3)
    other = sample_parameters(view, n=1000, seed=4)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_samples_are_centred_on_the_estimates() -> None:
    view = polynomial_view()
    assert view.cov is not None
    n = 5000
    draws = sample_parameters(view, n=n, seed=11)
    standard_error = np.sqrt(np.diag(view.cov) / n)
    assert np.all(np.abs(draws.mean(axis=0) - view.theta) <= 3.0 * standard_error)


def test_non_psd_covariance_is_refused() -> None:
    view = line_view()
    # eigenvalues 3 and -1: the covariance is indefinite and cannot be sampled from
    broken = replace(view, cov=np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(StatisticsError) as info:
        sample_parameters(broken, n=10)
    assert info.value.code == "COVARIANCE_NOT_PSD"


def test_a_negative_eigenvalue_within_rounding_is_clipped() -> None:
    view = line_view()
    # -1e-15 is rounding, not a failure: the eigenvalue is clipped at zero
    almost_psd = replace(view, cov=np.array([[1.0, 0.0], [0.0, -1e-15]]))
    draws = sample_parameters(almost_psd, n=1000, seed=2)
    assert np.all(np.isfinite(draws))
    np.testing.assert_allclose(draws[:, 1], view.theta[1], rtol=0.0, atol=1e-12)


def test_covariance_that_is_not_symmetric_is_symmetrised() -> None:
    view = line_view()
    assert view.cov is not None
    skew = replace(view, cov=view.cov + np.array([[0.0, 0.5], [-0.5, 0.0]]))
    draws = sample_parameters(skew, n=20000, seed=4)
    assert np.all(np.isfinite(draws))


def test_covariance_with_a_non_finite_value_is_refused() -> None:
    view = line_view()
    broken = replace(view, cov=np.array([[np.nan, 0.0], [0.0, 1.0]]))
    with pytest.raises(StatisticsError) as info:
        sample_parameters(broken, n=10)
    assert info.value.code == "INVALID_COVARIANCE"


def test_a_fit_without_covariance_is_refused() -> None:
    view = line_view()
    for broken, code in (
        (replace(view, cov=None), "NO_COVARIANCE"),
        (replace(view, df=0, s2=None), "NO_DEGREES_OF_FREEDOM"),
    ):
        with pytest.raises(StatisticsError) as info:
            sample_parameters(broken, n=10)
        assert info.value.code == code


@pytest.mark.parametrize("n", [0, -1])
def test_a_non_positive_number_of_draws_is_refused(n: int) -> None:
    with pytest.raises(StatisticsError) as info:
        sample_parameters(line_view(), n=n)
    assert info.value.code == "INVALID_N"


# --- propagation --------------------------------------------------------------------


def test_propagate_matches_the_delta_method_for_a_linear_function() -> None:
    view = line_view()
    assert view.cov is not None
    # g = a + 2b is linear in the parameters, so its standard deviation is exactly
    # sqrt(v' C v) with v = (1, 2).
    vector = np.array([1.0, 2.0])
    expected = float(np.sqrt(vector @ view.cov @ vector))

    result = propagate(view, lambda p: p["a"] + 2.0 * p["b"], n=100_000, seed=5)

    assert isinstance(result, Propagation)
    assert result.n == 100_000 and result.seed == 5
    assert result.n_nonfinite == 0
    assert result.level == 0.95
    assert result.std == pytest.approx(expected, rel=0.02)
    assert result.mean == pytest.approx(float(view.theta @ vector), rel=0.01)
    # the propagated distribution is normal, so the central interval matches mean +- z * std
    z = float(sps.norm.ppf(0.975))
    assert result.ci_low == pytest.approx(result.mean - z * result.std, rel=0.02)
    assert result.ci_high == pytest.approx(result.mean + z * result.std, rel=0.02)


def test_propagate_counts_nonfinite_results() -> None:
    view = line_view()

    def upper_tail_is_nan(p: Mapping[str, float]) -> float:
        value = p["a"] + p["b"]
        return value if value < 4.0 else math.nan

    result = propagate(view, upper_tail_is_nan, n=20_000, seed=2)

    assert 0 < result.n_nonfinite < result.n // 10
    # the non-finite draws are left out of the statistics, not counted as zero
    assert result.ci_high <= 4.0
    assert result.mean < 4.0


def test_propagate_reports_a_draw_that_raised() -> None:
    view = line_view()

    def always_raises(p: Mapping[str, float]) -> float:
        raise ValueError("no such thing")

    with pytest.raises(StatisticsError) as info:
        propagate(view, always_raises, n=100, seed=1)
    assert info.value.code == "PROPAGATION_UNSTABLE"
    assert "no such thing" in str(info.value)


def test_a_bug_in_the_function_is_not_counted_as_a_nonfinite_draw() -> None:
    view = line_view()

    def buggy(p: Mapping[str, float]) -> float:
        # fails only for some parameter sets, which a catch-all would hide in n_nonfinite
        return p["a"] + p["c_missing"] if p["a"] + p["b"] > 4.0 else p["a"]

    with pytest.raises(KeyError):
        propagate(view, buggy, n=2000, seed=2)


def test_arithmetic_errors_count_as_nonfinite_draws() -> None:
    view = line_view()
    assert view.cov is not None
    cutoff = float(view.theta[0] + 1.28 * np.sqrt(view.cov[0, 0]))  # ~10 % of draws lie above

    def divides_by_zero_in_the_upper_tail(p: Mapping[str, float]) -> float:
        if p["a"] > cutoff:
            return 1.0 / (p["a"] - p["a"])
        return p["a"]

    result = propagate(view, divides_by_zero_in_the_upper_tail, n=5000, seed=4)
    assert 0.05 * 5000 < result.n_nonfinite < 0.15 * 5000


def test_propagate_refuses_an_unstable_propagation() -> None:
    view = line_view()

    def mostly_finite(p: Mapping[str, float]) -> float:
        value = p["a"] + p["b"]
        return value if value < 1.0 else math.nan

    with pytest.raises(StatisticsError) as info:
        propagate(view, mostly_finite, n=20_000, seed=6)
    assert info.value.code == "PROPAGATION_UNSTABLE"


def test_propagate_is_reproducible_from_its_seed() -> None:
    view = line_view()
    first = propagate(view, lambda p: p["a"] * p["b"], n=5000, seed=9)
    again = propagate(view, lambda p: p["a"] * p["b"], n=5000, seed=9)
    assert first == again


def test_propagate_checks_the_level() -> None:
    with pytest.raises(StatisticsError) as info:
        propagate(line_view(), lambda p: p["a"], level=1.0)
    assert info.value.code == "INVALID_LEVEL"


def test_propagation_result_is_frozen() -> None:
    result = propagate(line_view(), lambda p: p["a"], n=100, seed=1)
    with pytest.raises(ValueError):
        result.mean = 0.0  # type: ignore[misc]


# --- Monte-Carlo band ----------------------------------------------------------------


def test_monte_carlo_band_agrees_with_confidence_band_for_a_line() -> None:
    view = line_view()
    assert view.cov is not None
    points = np.array([1.0, 4.0, 9.0])
    band = monte_carlo_band(view, points, n=20_000, seed=8)

    assert band.kind == "monte_carlo"
    assert band.level == 0.95
    assert band.x == points.tolist()
    assert band.df == view.df
    np.testing.assert_allclose(band.fit, view.predict(points), rtol=1e-12)

    # The delta-method half width uses the Student-t quantile; the Monte-Carlo band is a
    # percentile band of normal draws, so it carries no t correction and is compared with
    # the normal quantile instead. The empirical percentiles carry a sampling error of
    # about 1% of the half width at 20 000 draws, which is why the tolerance is 2%.
    jacobian = view.jacobian_at(points)
    standard_error = np.sqrt(np.einsum("ij,jk,ik->i", jacobian, view.cov, jacobian))
    expected = float(sps.norm.ppf(0.975)) * standard_error
    got = half_width(band.lower, band.fit, band.upper)
    np.testing.assert_allclose(got, expected, rtol=0.02)


def test_monte_carlo_band_is_narrower_than_the_t_based_confidence_band() -> None:
    view = line_view()
    points = np.array([2.0, 5.0, 8.0])
    monte_carlo = monte_carlo_band(view, points, n=20_000, seed=2)
    confidence = confidence_band(view, points)

    mc_width = half_width(monte_carlo.lower, monte_carlo.fit, monte_carlo.upper)
    delta_width = half_width(confidence.lower, confidence.fit, confidence.upper)
    assert np.all(mc_width > 0.0)
    assert np.all(mc_width < delta_width)


def test_monte_carlo_band_is_inside_the_prediction_band() -> None:
    view = line_view()
    points = np.array([2.0, 5.0, 8.0])
    monte_carlo = monte_carlo_band(view, points, n=20_000, seed=3)
    prediction = prediction_band(view, points)

    mc_width = half_width(monte_carlo.lower, monte_carlo.fit, monte_carlo.upper)
    prediction_width = half_width(prediction.lower, prediction.fit, prediction.upper)
    assert np.all(mc_width < prediction_width)


def test_monte_carlo_band_is_reproducible_from_its_seed() -> None:
    view = line_view()
    points = np.array([1.0, 6.0])
    first = monte_carlo_band(view, points, n=2000, seed=8)
    again = monte_carlo_band(view, points, n=2000, seed=8)
    other = monte_carlo_band(view, points, n=2000, seed=9)

    assert first == again
    assert first != other


def test_monte_carlo_band_of_a_polynomial_fit_widens_away_from_the_centre() -> None:
    view = polynomial_view()
    points = np.array([0.0, 5.0, 10.0])
    band = monte_carlo_band(view, points, n=20_000, seed=4)

    assert band.kind == "monte_carlo"
    np.testing.assert_allclose(band.fit, view.predict(points), rtol=1e-12)
    assert all(np.isfinite(band.lower)) and all(np.isfinite(band.upper))
    widths = np.asarray(band.upper) - np.asarray(band.lower)
    assert np.all(widths > 0.0)
    # the band widens away from the centre of the data
    assert widths[0] > widths[1] < widths[2]


def test_monte_carlo_band_of_noiseless_data_has_no_width() -> None:
    x = np.linspace(0.0, 5.0, 12)
    data = _xy(x, 1.0 + 2.0 * x - 0.5 * x**2, tag="noiseless")
    view = build_view(data, fit_polynomial(data, 2))
    points = np.array([0.0, 2.5, 5.0, 7.0])

    band = monte_carlo_band(view, points, n=1000, seed=1)

    assert np.all(np.isfinite(band.lower)) and np.all(np.isfinite(band.upper))
    np.testing.assert_allclose(np.asarray(band.upper) - np.asarray(band.lower), 0.0, atol=1e-9)
    np.testing.assert_allclose(band.fit, 1.0 + 2.0 * points - 0.5 * points**2, rtol=1e-9)


def test_monte_carlo_band_flags_extrapolated_points() -> None:
    view = line_view()  # the active x range is [0, 10]
    band = monte_carlo_band(view, np.array([-0.5, 0.0, 5.0, 10.0, 10.5]), n=1000, seed=1)
    assert band.extrapolated == [True, False, False, False, True]


def test_monte_carlo_band_checks_its_arguments() -> None:
    view = line_view()
    with pytest.raises(StatisticsError) as info:
        monte_carlo_band(view, np.array([1.0]), level=0.0)
    assert info.value.code == "INVALID_LEVEL"
    with pytest.raises(StatisticsError) as info:
        monte_carlo_band(view, np.array([]))
    assert info.value.code == "INVALID_X"
    with pytest.raises(StatisticsError) as info:
        monte_carlo_band(view, np.array([1.0]), n=0)
    assert info.value.code == "INVALID_N"


def test_monte_carlo_band_of_a_fit_without_covariance_is_refused() -> None:
    view = line_view()
    with pytest.raises(StatisticsError) as info:
        monte_carlo_band(replace(view, cov=None), np.array([1.0]))
    assert info.value.code == "NO_COVARIANCE"


def square_root_view() -> FitView:
    """A view of a fit whose model is only defined between two parameter values.

    ``sqrt((a-1)*(3-a))`` needs ``1 < a < 3``; the estimate sits inside that window, so
    the fitted curve is finite while a wide covariance pushes many draws outside it.
    """
    square_root = ModelSpec(
        id="mc_sqrt",
        name="Bounded square root",
        category="test",
        expression="sqrt((a-1)*(3-a)) + b*x",
        parameters=(Parameter("a"), Parameter("b")),
        initial_guess=lambda x, y: {"a": 1.5, "b": 0.0},
    )
    register(square_root)
    rng = np.random.default_rng(4)
    x = np.linspace(1.0, 5.0, 15)
    truth = np.sqrt((1.5 - 1.0) * (3.0 - 1.5)) + 0.5 * x
    data = _xy(x, truth + rng.normal(0.0, 0.02, x.size), tag="sqrt")
    return build_view(data, fit(square_root, data), square_root)


def test_monte_carlo_band_leaves_out_draws_that_are_not_finite() -> None:
    view = square_root_view()
    # a wider covariance pushes some draws of "a" outside the window the model needs
    wide = replace(view, cov=np.diag([0.25, 1e-6]))
    band = monte_carlo_band(wide, np.array([2.0]), n=2000, seed=1)

    assert all(np.isfinite(band.lower)) and all(np.isfinite(band.upper))
    assert "not finite" in band.note


def test_monte_carlo_band_refuses_a_point_where_most_draws_fail() -> None:
    view = square_root_view()
    # most draws of "a" now fall outside the window the model needs
    hopeless = replace(view, cov=np.diag([4.0, 1e-6]))
    with pytest.raises(StatisticsError) as info:
        monte_carlo_band(hopeless, np.array([2.0]), n=2000, seed=1)
    assert info.value.code == "PROPAGATION_UNSTABLE"
