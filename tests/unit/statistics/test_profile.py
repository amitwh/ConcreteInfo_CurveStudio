"""Tests for profile-likelihood confidence intervals."""

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.optimize import least_squares

from curvestudio.data import XY
from curvestudio.fitting import FitConfig, FitResult, FitStatus, fit, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import ProfileResult, StatisticsError, profile_likelihood

QUADRATIC = ModelSpec(
    id="quad_profile",
    name="Quadratic",
    category="test",
    expression="a + b*x + c*x**2",
    parameters=(Parameter("a"), Parameter("b"), Parameter("c")),
    initial_guess=lambda x, y: {"a": 0.0, "b": 1.0, "c": 0.1},
)
DECAY = ModelSpec(
    id="decay_profile",
    name="Exponential decay",
    category="test",
    expression="a*exp(-b*x)",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": float(y[0]), "b": 0.5},
)


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(QUADRATIC)
    register(DECAY)
    yield
    _clear_registry()


def _xy(x: NDArray[np.float64], y: NDArray[np.float64], tag: str) -> XY:
    return XY(x=x, y=y, weights=None, x_name="x", y_name="y", fingerprint=f"profile-{tag}")


def _quadratic_data() -> XY:
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 5.0, 25)
    y = 1.0 + 0.8 * x - 0.15 * x**2 + rng.normal(0.0, 0.3, x.size)
    return _xy(x, y, "quad")


def _decay_data(n: int, noise: float, seed: int, b: float = 0.6) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, n)
    y = 5.0 * np.exp(-b * x) + rng.normal(0.0, noise, n)
    return _xy(x, y, f"decay-{n}-{seed}")


def _fit(model: ModelSpec, data: XY, config: FitConfig = FitConfig()) -> FitResult:  # noqa: B008
    result = fit(model, data, config)
    assert result.status is FitStatus.CONVERGED, result.termination_reason
    return result


def _constrained_sse(
    model: ModelSpec, data: XY, result: FitResult, name: str, value: float
) -> float:
    """Independent re-computation of ``min SSE`` with parameter ``name`` held at ``value``."""
    free = [p for p in result.parameters if p.name != name]
    start = np.array([p.value for p in free])
    free_names = [p.name for p in free]

    def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        values = dict(zip(free_names, theta.tolist(), strict=True)) | {name: value}
        return np.asarray(data.y - model.evaluate(data.x, **values))

    solution = least_squares(residuals, start, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return float(np.sum(solution.fun**2))


def test_profile_interval_of_a_linear_model_equals_the_wald_interval() -> None:
    data = _quadratic_data()
    result = _fit(QUADRATIC, data)

    profile = profile_likelihood(data, QUADRATIC, result)

    assert isinstance(profile, ProfileResult)
    assert profile.level == 0.95
    assert [i.name for i in profile.intervals] == ["a", "b", "c"]
    for interval, estimate in zip(profile.intervals, result.parameters, strict=True):
        assert estimate.ci_low is not None and estimate.ci_high is not None
        width = estimate.ci_high - estimate.ci_low
        assert interval.low_status == interval.high_status == "found"
        assert interval.estimate == pytest.approx(estimate.value)
        assert interval.ci_low == pytest.approx(estimate.ci_low, rel=1e-6, abs=1e-8 * width)
        assert interval.ci_high == pytest.approx(estimate.ci_high, rel=1e-6, abs=1e-8 * width)
        assert interval.n_evals > 0


def test_threshold_follows_the_bates_watts_formula() -> None:
    data = _quadratic_data()
    result = _fit(QUADRATIC, data)
    assert result.stats is not None

    profile = profile_likelihood(data, QUADRATIC, result, level=0.9)

    df = result.stats.df
    expected = result.stats.sse * (1.0 + scipy_stats.f.ppf(0.9, 1, df) / df)
    assert profile.level == 0.9
    assert profile.threshold_sse == pytest.approx(expected, rel=1e-12)


def test_sse_at_the_interval_ends_equals_the_threshold() -> None:
    data = _decay_data(n=12, noise=0.25, seed=3)
    result = _fit(DECAY, data)

    profile = profile_likelihood(data, DECAY, result)

    for interval in profile.intervals:
        for limit in (interval.ci_low, interval.ci_high):
            assert limit is not None
            sse = _constrained_sse(DECAY, data, result, interval.name, limit)
            assert sse == pytest.approx(profile.threshold_sse, rel=1e-6)


def test_profile_interval_is_asymmetric_for_a_strongly_nonlinear_model() -> None:
    data = _decay_data(n=7, noise=0.5, seed=4, b=0.9)
    result = _fit(DECAY, data)

    profile = profile_likelihood(data, DECAY, result)

    interval = next(i for i in profile.intervals if i.name == "b")
    assert interval.ci_low is not None and interval.ci_high is not None
    width = interval.ci_high - interval.ci_low
    skew = abs((interval.ci_high - interval.estimate) - (interval.estimate - interval.ci_low))
    assert skew / width > 0.02
    # The Wald interval of the same fit is symmetric by construction, so they differ.
    wald = result.parameters[1]
    assert wald.ci_low is not None and wald.ci_high is not None
    assert abs((wald.ci_high - wald.value) - (wald.value - wald.ci_low)) / width < 1e-9


def test_parameter_at_a_bound_reports_at_bound() -> None:
    # A nearly flat series: the decay rate cannot be told apart from zero, which is its bound.
    data = _decay_data(n=10, noise=1.0, seed=5, b=0.02)
    config = FitConfig(bounds={"b": (0.0, 5.0)})
    result = _fit(DECAY, data, config)

    profile = profile_likelihood(data, DECAY, result, config)

    b = next(i for i in profile.intervals if i.name == "b")
    assert b.low_status == "at_bound"
    assert b.ci_low == 0.0
    assert b.high_status == "found"
    assert b.ci_high is not None and b.ci_high > b.estimate


def test_an_estimate_sitting_on_its_bound_reports_at_bound_without_searching() -> None:
    data = _decay_data(n=10, noise=0.2, seed=6, b=0.5)
    config = FitConfig(bounds={"b": (0.8, 5.0)})
    result = fit(DECAY, data, config)
    assert result.status is FitStatus.CONVERGED

    profile = profile_likelihood(data, DECAY, result, config)

    b = next(i for i in profile.intervals if i.name == "b")
    assert b.low_status == "at_bound"
    assert b.ci_low == 0.8


def test_a_limit_that_cannot_be_bracketed_is_not_found() -> None:
    # Three points for two parameters leave one residual degree of freedom, so at a very high
    # level the threshold is astronomically large and no finite limit lies within 50 SEs.
    data = _xy(np.array([0.0, 1.0, 2.0]), np.array([5.0, 3.0, 1.9]), "open")
    result = _fit(DECAY, data)

    profile = profile_likelihood(data, DECAY, result, level=1.0 - 1e-12)

    statuses = {status for i in profile.intervals for status in (i.low_status, i.high_status)}
    assert "not_found" in statuses
    for interval in profile.intervals:
        for status, limit in (
            (interval.low_status, interval.ci_low),
            (interval.high_status, interval.ci_high),
        ):
            assert (limit is None) == (status == "not_found")


def test_polynomial_and_robust_loss_are_not_supported_with_clear_codes() -> None:
    data = _quadratic_data()
    poly = fit_polynomial(data, 2)
    with pytest.raises(StatisticsError, match="Wald") as poly_error:
        profile_likelihood(data, "quad_profile", poly)
    assert poly_error.value.code == "PROFILE_NOT_SUPPORTED"

    result = _fit(QUADRATIC, data)
    with pytest.raises(StatisticsError) as loss_error:
        profile_likelihood(data, QUADRATIC, result, FitConfig(loss="soft_l1"))
    assert loss_error.value.code == "PROFILE_NOT_SUPPORTED"

    robust = fit(QUADRATIC, data, FitConfig(loss="huber"))
    with pytest.raises(StatisticsError) as robust_error:
        profile_likelihood(data, QUADRATIC, robust)
    assert robust_error.value.code == "PROFILE_NOT_SUPPORTED"

    decay_data = _decay_data(12, 0.2, 1)
    scalar = fit(DECAY, decay_data, FitConfig(method="nelder"))
    assert scalar.status is FitStatus.CONVERGED
    with pytest.raises(StatisticsError) as scalar_error:
        profile_likelihood(decay_data, DECAY, scalar)
    assert scalar_error.value.code == "PROFILE_NOT_SUPPORTED"


def test_a_failed_fit_is_refused_with_the_view_codes() -> None:
    data = _xy(np.array([0.0, 1.0]), np.array([1.0, 2.0]), "tiny")
    result = fit(DECAY, data)
    assert result.status is not FitStatus.CONVERGED
    with pytest.raises(StatisticsError) as error:
        profile_likelihood(data, DECAY, result)
    assert error.value.code == "FIT_NOT_CONVERGED"


def test_a_level_outside_zero_one_is_rejected() -> None:
    data = _quadratic_data()
    result = _fit(QUADRATIC, data)
    for level in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="level"):
            profile_likelihood(data, QUADRATIC, result, level=level)


def test_noiseless_data_gives_zero_width_intervals_without_warnings() -> None:
    x = np.linspace(0.0, 4.0, 9)
    data = _xy(x, 2.0 * np.exp(-0.5 * x), "noiseless")
    result = _fit(DECAY, data)
    assert result.stats is not None and result.stats.sse < 1e-20

    profile = profile_likelihood(data, DECAY, result)

    for interval in profile.intervals:
        assert interval.low_status == interval.high_status == "found"
        assert interval.ci_low == pytest.approx(interval.estimate, abs=1e-6)
        assert interval.ci_high == pytest.approx(interval.estimate, abs=1e-6)


def test_zero_weight_points_do_not_influence_the_profile() -> None:
    base = _decay_data(n=14, noise=0.2, seed=8)
    weights = np.ones(14)
    weights[[3, 9]] = 0.0
    weighted = XY(
        x=base.x,
        y=base.y,
        weights=weights,
        x_name="x",
        y_name="y",
        fingerprint="profile-zero-weight",
    )
    kept = weights > 0.0
    reduced = _xy(base.x[kept], base.y[kept], "profile-reduced")

    with_zero = profile_likelihood(weighted, DECAY, _fit(DECAY, weighted))
    without = profile_likelihood(reduced, DECAY, _fit(DECAY, reduced))

    for a, b in zip(with_zero.intervals, without.intervals, strict=True):
        assert a.ci_low == pytest.approx(b.ci_low, rel=1e-6)
        assert a.ci_high == pytest.approx(b.ci_high, rel=1e-6)


def test_same_result_every_run() -> None:
    data = _decay_data(n=12, noise=0.3, seed=9)
    result = _fit(DECAY, data)

    first = profile_likelihood(data, DECAY, result)
    second = profile_likelihood(data, DECAY, result)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
