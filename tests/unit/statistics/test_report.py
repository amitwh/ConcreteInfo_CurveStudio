"""Tests for the ``describe_fit`` bundle."""

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from curvestudio.data import XY
from curvestudio.diagnostics import FitReport, describe_fit
from curvestudio.fitting import FitResult, FitStatus, fit, fit_polynomial
from curvestudio.models import ModelSpec, Parameter, register
from curvestudio.models.registry import _clear_registry
from curvestudio.statistics import NonlinearFitter, PolynomialFitter, StatisticsError

SATURATION = ModelSpec(
    id="report_saturation",
    name="Saturation",
    category="test",
    expression="a*(1 - exp(-b*x))",
    parameters=(Parameter("a"), Parameter("b")),
    initial_guess=lambda x, y: {"a": float(y.max()), "b": 1.0},
)


@pytest.fixture(autouse=True)
def registry() -> Iterator[None]:
    _clear_registry()
    register(SATURATION)
    yield
    _clear_registry()


def make_xy(x: Any, y: Any, weights: Any = None, tag: str = "report") -> XY:
    return XY(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        x_name="x",
        y_name="y",
        fingerprint=f"fp-{tag}",
    )


def quadratic_data(n: int = 30, seed: int = 5) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, n)
    return make_xy(x, 1.0 + 0.5 * x - 0.03 * x**2 + rng.normal(0.0, 0.2, n), tag="quad")


def saturation_data(n: int = 28, seed: int = 8) -> XY:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.1, 6.0, n)
    return make_xy(x, 4.0 * (1.0 - np.exp(-0.9 * x)) + rng.normal(0.0, 0.05, n), tag="sat")


def test_describe_fit_round_trips_through_json() -> None:
    data = quadratic_data()
    result = fit_polynomial(data, 2)
    report = describe_fit(data, result, fitter=PolynomialFitter(2), cv_folds=4, seed=3)

    text = report.to_json()
    json.loads(text)  # standard JSON: no NaN / Infinity tokens
    assert "NaN" not in text and "Infinity" not in text
    again = FitReport.from_json(text)

    assert again == report
    assert again.to_json() == text
    assert report.cv is not None and again.cv is not None
    assert again.cv.seed == 3 and again.cv.folds == 4


def test_round_trip_keeps_undefined_entries_as_none() -> None:
    # One residual degree of freedom: externally studentized residuals are undefined.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    data = make_xy(x, [0.1, 1.2, 1.9, 3.2], tag="tiny")
    report = describe_fit(data, fit_polynomial(data, 2))

    assert all(value is None for value in report.residuals.studentized_external)
    again = FitReport.from_json(report.to_json())
    assert again.residuals.studentized_external == report.residuals.studentized_external
    assert again == report


def test_from_json_rejects_non_finite_constants() -> None:
    data = quadratic_data()
    text = describe_fit(data, fit_polynomial(data, 2)).to_json()
    corrupted = text.replace('"acf_bound": ', '"acf_bound": NaN, "_": ', 1)
    with pytest.raises(ValueError, match="non-finite"):
        FitReport.from_json(corrupted)


def test_to_json_rejects_a_mutated_non_finite_value() -> None:
    data = quadratic_data()
    report = describe_fit(data, fit_polynomial(data, 2))
    report.residuals.residuals[0] = float("nan")  # lists are not deep-frozen
    with pytest.raises(ValueError):
        report.to_json()


def test_describe_fit_refuses_an_unconverged_fit() -> None:
    data = quadratic_data()
    good = fit_polynomial(data, 2)
    bad = good.model_copy(update={"status": FitStatus.NOT_CONVERGED})
    with pytest.raises(StatisticsError) as raised:
        describe_fit(data, bad, fitter=PolynomialFitter(2))
    assert raised.value.code == "FIT_NOT_CONVERGED"


def test_describe_fit_refuses_data_the_fit_was_not_computed_from() -> None:
    # A different dataset than the fit was computed from is refused, not reported on.
    data = quadratic_data()
    other = make_xy(data.x, data.y + 1.0, tag="other")
    with pytest.raises(StatisticsError) as raised:
        describe_fit(other, fit_polynomial(data, 2))
    assert raised.value.code == "DATA_MISMATCH"


def test_describe_fit_without_fitter_has_no_cv() -> None:
    data = quadratic_data()
    report = describe_fit(data, fit_polynomial(data, 2))
    assert report.cv is None
    assert FitReport.from_json(report.to_json()).cv is None


def test_result_summary_lists_the_headline_numbers() -> None:
    data = quadratic_data()
    result = fit_polynomial(data, 2)
    report = describe_fit(data, result)
    stats = result.stats
    assert stats is not None

    summary = report.result_summary
    assert summary["model_id"] == "poly2"
    assert summary["status"] == "converged"
    assert summary["n"] == stats.n
    assert summary["k"] == stats.k
    assert summary["df"] == stats.df
    assert summary["r2"] == stats.r2
    assert summary["adj_r2"] == stats.adj_r2
    assert summary["rmse"] == stats.rmse


def test_undefined_summary_numbers_are_left_out_not_zero() -> None:
    # Constant y: R^2 is not defined, so the key is absent (never 0.0 or NaN).
    x = np.linspace(0.0, 1.0, 8)
    data = make_xy(x, np.full(8, 3.0), tag="flat")
    result = fit_polynomial(data, 1)
    assert result.stats is not None and result.stats.r2 is None
    report = describe_fit(data, result)
    assert "r2" not in report.result_summary
    assert report.result_summary["n"] == 8


def test_report_for_a_nonlinear_fit_is_complete() -> None:
    data = saturation_data()
    result = fit(SATURATION, data)
    assert result.status is FitStatus.CONVERGED
    fitter = NonlinearFitter(SATURATION, start={p.name: p.value for p in result.parameters})
    report = describe_fit(data, result, model=SATURATION, fitter=fitter, cv_folds=4, seed=1)

    n = int(data.x.size)
    assert report.result_summary["model_id"] == "report_saturation"
    assert report.criteria.n == n
    assert report.criteria.k_total == 3
    residuals = report.residuals
    for series in (
        residuals.x,
        residuals.fitted,
        residuals.residuals,
        residuals.weighted_residuals,
        residuals.standardized,
        residuals.studentized_external,
        residuals.leverage,
        residuals.cooks_distance,
        residuals.scale_location,
    ):
        assert len(series) == n
    assert len(residuals.qq_theoretical) == len(residuals.qq_ordered) == n
    assert sum(residuals.hist_counts) == n
    names = [test.name for test in report.diagnostics.tests]
    assert names == [
        "shapiro_wilk",
        "anderson_darling",
        "breusch_pagan",
        "ljung_box",
        "durbin_watson",
    ]
    assert all(test.assumptions and test.limitations for test in report.diagnostics.tests)
    assert report.cv is not None and report.cv.status == "ok"
    assert len(report.cv.fold_rmse) == 4
    assert FitReport.from_json(report.to_json()) == report


def test_zero_weight_points_are_excluded_from_every_section() -> None:
    base = quadratic_data()
    weights = np.ones(base.x.size)
    weights[[2, 11]] = 0.0
    data = make_xy(base.x, base.y, weights, tag="zero-weight")
    result = fit_polynomial(data, 2)
    report = describe_fit(data, result, fitter=PolynomialFitter(2), cv_folds=3)

    active = int(base.x.size) - 2
    assert report.criteria.n == active
    assert len(report.residuals.x) == active
    assert report.result_summary["n"] == active


def test_warnings_carry_the_fit_warnings_and_report_findings() -> None:
    data = quadratic_data()
    result = fit_polynomial(data, 2)
    noisy: FitResult = result.model_copy(update={"warnings": ("fit warning from the engine",)})
    report = describe_fit(data, noisy)
    assert "fit warning from the engine" in report.warnings


def test_incomplete_cross_validation_is_reported_as_a_warning() -> None:
    data = quadratic_data(n=12)
    result = fit_polynomial(data, 2)
    # Three folds of four points leave eight to fit a quadratic: fine.  Ten folds of ~one
    # point still fit.  A fitter whose model needs more parameters than the folds provide
    # fails every fold, which the report must surface rather than hide.
    report = describe_fit(data, result, fitter=PolynomialFitter(10), cv_folds=6)
    assert report.cv is not None and report.cv.status == "incomplete"
    assert any("cross-validation" in message for message in report.warnings)


def test_a_report_is_immutable() -> None:
    data = quadratic_data()
    report = describe_fit(data, fit_polynomial(data, 2))
    with pytest.raises(ValueError):
        report.warnings = ("x",)  # type: ignore[misc]
