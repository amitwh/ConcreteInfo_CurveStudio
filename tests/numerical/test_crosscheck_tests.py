"""Normality, heteroscedasticity and autocorrelation tests equal Statsmodels and SciPy.

Reference: ``statsmodels.stats.diagnostic.normal_ad`` (Anderson-Darling),
``statsmodels.stats.diagnostic.het_breuschpagan`` (Breusch-Pagan),
``statsmodels.stats.diagnostic.acorr_ljungbox`` (Ljung-Box),
``statsmodels.stats.stattools.durbin_watson`` and ``scipy.stats.shapiro`` on the stored
datasets (see ``reference_tests`` in ``scripts/make_crosscheck_fixtures.py``).  The
residuals are the weighted residuals ``sqrt(w) * (y - fitted)`` of each fit, in
ascending-x order, which is the vector ``run_diagnostics`` works with.
"""

from typing import Any

import pytest
from crosscheck import fixture_xy, load_fixtures

from curvestudio.diagnostics.hypothesis_tests import (
    DiagnosticsReport,
    DiagnosticTest,
    run_diagnostics,
)
from curvestudio.fitting import fit_polynomial
from curvestudio.statistics import build_view

FIXTURES = load_fixtures()
STAT_RTOL = 1e-8
"""Relative tolerance on the test statistics."""

P_ATOL = 1e-6
"""Absolute tolerance on the p-values."""


def report_for(name: str, entry: dict[str, Any], lags: int | None = None) -> DiagnosticsReport:
    data = fixture_xy(name, entry)
    return run_diagnostics(build_view(data, fit_polynomial(data, entry["degree"])), lags=lags)


def by_name(report: DiagnosticsReport, name: str) -> DiagnosticTest:
    return next(test for test in report.tests if test.name == name)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_normality_tests_equal_statsmodels(name: str) -> None:
    entry = FIXTURES[name]
    ref = entry["tests"]
    report = report_for(name, entry)

    shapiro = by_name(report, "shapiro_wilk")
    assert shapiro.statistic == pytest.approx(ref["shapiro_w"], rel=STAT_RTOL)
    assert shapiro.p_value == pytest.approx(ref["shapiro_p"], rel=0.0, abs=P_ATOL)

    anderson = by_name(report, "anderson_darling")
    n = len(entry["x"])
    assert anderson.statistic == pytest.approx(ref["ad_a2a"], rel=STAT_RTOL)
    assert anderson.statistic == pytest.approx(
        ref["ad_a2"] * (1.0 + 0.75 / n + 2.25 / n**2), rel=STAT_RTOL
    )
    assert anderson.p_value == pytest.approx(ref["ad_p"], rel=0.0, abs=P_ATOL)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_breusch_pagan_equal_statsmodels(name: str) -> None:
    entry = FIXTURES[name]
    ref = entry["tests"]
    report = report_for(name, entry)

    breusch = by_name(report, "breusch_pagan")
    assert breusch.df == 1.0
    assert breusch.statistic == pytest.approx(ref["bp_lm"], rel=STAT_RTOL)
    assert breusch.p_value == pytest.approx(ref["bp_p"], rel=0.0, abs=P_ATOL)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_ljung_box_equal_statsmodels(name: str) -> None:
    entry = FIXTURES[name]
    ref = entry["tests"]

    one = by_name(report_for(name, entry, lags=1), "ljung_box")
    assert one.df == 1.0
    assert one.statistic == pytest.approx(ref["ljung_box_q1"], rel=STAT_RTOL)
    assert one.p_value == pytest.approx(ref["ljung_box_p1"], rel=0.0, abs=P_ATOL)

    five = by_name(report_for(name, entry, lags=5), "ljung_box")
    assert five.df == 5.0
    assert five.statistic == pytest.approx(ref["ljung_box_q5"], rel=STAT_RTOL)
    assert five.p_value == pytest.approx(ref["ljung_box_p5"], rel=0.0, abs=P_ATOL)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_durbin_watson_equal_statsmodels(name: str) -> None:
    entry = FIXTURES[name]
    ref = entry["tests"]
    report = report_for(name, entry)

    durbin = by_name(report, "durbin_watson")
    assert durbin.statistic == pytest.approx(ref["durbin_watson"], rel=STAT_RTOL)
    assert durbin.p_value is None
