"""Tests for the residual hypothesis tests (normality, heteroscedasticity, autocorrelation)."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from curvestudio.data import XY
from curvestudio.diagnostics.hypothesis_tests import (
    NOTE,
    DiagnosticsReport,
    DiagnosticTest,
    anderson_darling,
    breusch_pagan,
    durbin_watson,
    ljung_box,
    run_diagnostics,
    shapiro_wilk,
)
from curvestudio.fitting import FitStatus, fit, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import StatisticsError, build_view

TEST_NAMES = ("shapiro_wilk", "anderson_darling", "breusch_pagan", "ljung_box", "durbin_watson")

QUADRATIC = ModelSpec(
    id="quad_nl_tests",
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
        fingerprint="fp-hypothesis-tests",
    )


def noisy_line(n: int = 24, seed: int = 9) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 5.0, n)
    return x, 1.0 + 0.4 * x + rng.normal(0.0, 0.2, n)


def ar1(n: int = 200, rho: float = 0.8, seed: int = 11) -> Any:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, float(np.sqrt(1.0 - rho**2)), n)
    series = np.empty(n)
    series[0] = rng.normal()
    for t in range(1, n):
        series[t] = rho * series[t - 1] + innovations[t]
    return series


def all_results() -> list[DiagnosticTest]:
    """One result of every kind, computed and not applicable."""
    rng = np.random.default_rng(3)
    residuals = rng.normal(size=40)
    fitted = np.linspace(1.0, 8.0, 40)
    return [
        shapiro_wilk(residuals),
        anderson_darling(residuals),
        breusch_pagan(residuals, fitted),
        ljung_box(residuals),
        durbin_watson(residuals),
        shapiro_wilk(np.full(5, 1.0)),
        ljung_box(residuals, lags=0),
    ]


def test_shapiro_rejects_strongly_skewed_residuals_and_accepts_gaussian() -> None:
    gaussian = np.random.default_rng(2).normal(size=120)
    skewed = np.random.default_rng(2).exponential(size=120)

    accepted = shapiro_wilk(gaussian)
    rejected = shapiro_wilk(skewed)

    assert accepted.status == "ok"
    assert rejected.status == "ok"
    assert accepted.statistic is not None and accepted.p_value is not None
    assert rejected.statistic is not None and rejected.p_value is not None
    # the seeds are chosen so neither outcome is borderline
    assert accepted.p_value > 0.05
    assert rejected.p_value < 0.05


def test_each_test_states_assumptions_and_limitations_and_never_says_valid() -> None:
    for result in all_results():
        assert result.assumptions, result.name
        assert result.limitations, result.name
        assert result.null_hypothesis, result.name
        assert result.note == NOTE
        texts = (
            result.null_hypothesis,
            result.note,
            result.reason or "",
            *result.assumptions,
            *result.limitations,
        )
        for text in texts:
            lowered = text.lower()
            assert "valid model" not in lowered, (result.name, text)
            assert "model is valid" not in lowered, (result.name, text)


def test_constant_residuals_are_not_applicable() -> None:
    constant = np.full(12, 0.5)
    results = [
        shapiro_wilk(constant),
        anderson_darling(constant),
        breusch_pagan(constant, np.linspace(0.0, 1.0, 12)),
        ljung_box(constant),
        durbin_watson(constant),
    ]
    for result in results:
        assert result.status == "not_applicable", result.name
        assert result.statistic is None
        assert result.p_value is None
        assert result.reason, result.name
        assert "constant" in result.reason.lower()


def test_n_below_minimum_is_not_applicable() -> None:
    assert shapiro_wilk(np.linspace(-1.0, 1.0, 2)).status == "not_applicable"
    assert anderson_darling(np.linspace(-1.0, 1.0, 7)).status == "not_applicable"


def test_ljung_box_clamps_lags_and_says_so() -> None:
    residuals = np.random.default_rng(0).normal(size=12)

    too_many = ljung_box(residuals, lags=50)
    assert too_many.status == "ok"
    assert too_many.df == 11.0
    assert too_many.reason is not None
    assert "clamp" in too_many.reason.lower()

    too_few = ljung_box(residuals, lags=0)
    assert too_few.status == "ok"
    assert too_few.df == 1.0
    assert too_few.reason is not None
    assert "clamp" in too_few.reason.lower()

    default = ljung_box(residuals)
    assert default.df == 2.0  # min(10, n // 5)
    assert default.reason is None


def test_durbin_watson_is_near_two_for_white_noise_and_small_for_ar1() -> None:
    white = np.random.default_rng(7).normal(size=200)
    white_result = durbin_watson(white)
    assert white_result.status == "ok"
    assert white_result.statistic == pytest.approx(2.0, abs=0.2)
    assert white_result.p_value is None
    assert white_result.df is None

    correlated = durbin_watson(ar1())
    assert correlated.status == "ok"
    assert correlated.statistic is not None
    assert correlated.statistic < 1.0


def test_breusch_pagan_flags_funnel_shaped_residuals() -> None:
    fitted = np.linspace(1.0, 10.0, 40)
    funnel = fitted * np.random.default_rng(4).normal(size=40)
    result = breusch_pagan(funnel, fitted)
    assert result.status == "ok"
    assert result.df == 1.0
    assert result.statistic is not None and result.statistic > 5.0
    assert result.p_value is not None and result.p_value < 0.01

    plain = breusch_pagan(np.random.default_rng(5).normal(size=40), fitted)
    assert plain.status == "ok"
    assert plain.statistic is not None and plain.statistic < 1.0
    assert plain.p_value is not None and plain.p_value > 0.05


def test_run_diagnostics_is_order_independent() -> None:
    x, y = noisy_line()
    ordered = run_diagnostics(build_view(make_xy(x, y), fit_polynomial(make_xy(x, y), 2)))

    permutation = np.random.default_rng(13).permutation(len(x))
    shuffled_data = make_xy(x[permutation], y[permutation])
    shuffled = run_diagnostics(build_view(shuffled_data, fit_polynomial(shuffled_data, 2)))

    assert [test.name for test in ordered.tests] == list(TEST_NAMES)
    assert [test.name for test in shuffled.tests] == list(TEST_NAMES)
    for left, right in zip(ordered.tests, shuffled.tests, strict=True):
        assert left.status == right.status
        assert left.statistic == pytest.approx(right.statistic, rel=1e-9, abs=1e-12)
        if left.p_value is None or right.p_value is None:
            assert left.p_value is None and right.p_value is None
        else:
            assert left.p_value == pytest.approx(right.p_value, rel=1e-9, abs=1e-12)


def test_run_diagnostics_returns_a_report_of_the_five_tests() -> None:
    x, y = noisy_line()
    report = run_diagnostics(build_view(make_xy(x, y), fit_polynomial(make_xy(x, y), 2)))
    assert isinstance(report, DiagnosticsReport)
    assert [test.name for test in report.tests] == list(TEST_NAMES)
    assert all(test.status == "ok" for test in report.tests)


def test_run_diagnostics_refuses_a_fit_that_did_not_converge() -> None:
    x, y = noisy_line()
    data = make_xy(x, y)
    view = build_view(data, fit_polynomial(data, 2))
    failed = replace(view, result=view.result.model_copy(update={"status": FitStatus.FAILED}))
    with pytest.raises(StatisticsError) as raised:
        run_diagnostics(failed)
    assert raised.value.code == "FIT_NOT_CONVERGED"


def test_run_diagnostics_reports_not_applicable_for_an_exact_fit() -> None:
    x = np.linspace(0.0, 5.0, 12)
    data = make_xy(x, 2.0 + 3.0 * x)
    report = run_diagnostics(build_view(data, fit_polynomial(data, 1)))
    assert all(test.status == "not_applicable" for test in report.tests)
    assert all(test.reason for test in report.tests)
    assert all(test.statistic is None and test.p_value is None for test in report.tests)


def test_non_finite_residuals_are_not_applicable() -> None:
    broken = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for result in (
        shapiro_wilk(broken),
        anderson_darling(broken),
        ljung_box(broken),
        durbin_watson(broken),
        breusch_pagan(broken, np.linspace(0.0, 1.0, 10)),
    ):
        assert result.status == "not_applicable", result.name
        assert result.reason, result.name


def test_breusch_pagan_refuses_mismatched_lengths() -> None:
    with pytest.raises(StatisticsError) as raised:
        breusch_pagan(np.zeros(5), np.zeros(4))
    assert raised.value.code == "DATA_MISMATCH"


def test_durbin_watson_notes_tied_x_in_its_limitations() -> None:
    residuals = np.random.default_rng(6).normal(size=20)
    plain = durbin_watson(residuals)
    tied = durbin_watson(residuals, tied_x=True)
    assert plain.statistic == tied.statistic
    assert not any("duplicate" in text.lower() for text in plain.limitations)
    assert any("duplicate" in text.lower() for text in tied.limitations)


def test_weighted_residuals_are_used_for_a_weighted_fit() -> None:
    x, y = noisy_line(n=20)
    weights = 1.0 / (x + 1.0) ** 2
    data = make_xy(x, y, weights)
    view = build_view(data, fit_polynomial(data, 1))
    report = run_diagnostics(view)
    weighted = np.sqrt(view.w) * view.residuals
    expected = shapiro_wilk(weighted)
    actual = next(test for test in report.tests if test.name == "shapiro_wilk")
    assert actual.statistic == pytest.approx(expected.statistic, rel=1e-12)


def test_the_same_code_path_serves_a_nonlinear_model() -> None:
    x, y = noisy_line(n=24)
    data = make_xy(x, y)
    report = run_diagnostics(build_view(data, fit(QUADRATIC, data)))
    assert [test.name for test in report.tests] == list(TEST_NAMES)
    assert all(test.status == "ok" for test in report.tests)


def test_a_diagnostic_test_refuses_a_not_a_number_statistic() -> None:
    with pytest.raises(ValidationError):
        DiagnosticTest(
            name="shapiro_wilk",
            statistic=float("nan"),
            p_value=0.5,
            df=None,
            status="ok",
            reason=None,
            null_hypothesis="h",
            assumptions=("a",),
            limitations=("l",),
            note=NOTE,
        )
