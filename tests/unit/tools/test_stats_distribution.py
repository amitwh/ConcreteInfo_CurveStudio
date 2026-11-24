"""Distribution analysis: one column's mean, spread, normality and shape.

The reference numbers come from SciPy and statsmodels, and the worked example from the
NIST/SEMATECH e-Handbook of Statistical Methods. The handbook's pages were fetched on
2026-11-24:

- ``1.4.2.4.1. Background and Data`` (itl.nist.gov/div898/handbook/eda/section4/eda4241.htm)
  prints the 700 Josephson-junction voltage counts embedded below.
- ``1.4.2.4.3. Quantitative Output and Interpretation``
  (itl.nist.gov/div898/handbook/eda/section4/eda4243.htm) prints, for those values, a mean of
  2898.562 and a standard deviation of 1.305, and Anderson-Darling output with the unadjusted
  statistic 16.76349, the adjusted statistic 16.85843 and the 5 % critical value 0.787. The
  data file is itl.nist.gov/div898/handbook/datasets/SOULEN.DAT.

The handbook's Dataplot adjusts the statistic with the constant ``1 + 4/n - 25/n**2``; this
core uses Stephens's (1974) normal constant ``1 + 0.75/n + 2.25/n**2``, so the two adjusted
statistics differ in the third significant figure while the unadjusted statistic they share
is reproduced. The handbook has no worked Shapiro-Wilk example with its data (its Shapiro-
Wilk page, 7.2.1.3, gives the definition only), so the Shapiro-Wilk figures are held to SciPy.
"""

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import normal_ad

from curvestudio.data import Column, GroupColumn
from curvestudio.diagnostics import DiagnosticTest, anderson_darling, shapiro_wilk
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    HistogramPlot,
    HypothesisTest,
    QQPlot,
    ToolResult,
    Verdict,
    assert_size,
    p_text,
    simulated_size,
)
from curvestudio.tools.stats._base import MAX_LISTED_EXCLUDED
from curvestudio.tools.stats.distribution import (
    MAX_BINS,
    MAX_SHAPIRO,
    DistributionRequest,
    run_distribution,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, TOOLS, catalogue

P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""
STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""


def _column(name: str, values: list[float]) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    return Column(
        name=name,
        values=array,
        rows=np.arange(array.size, dtype=np.intp),
    )


def _run(
    values: list[float], *, name: str = "y", level: float = 0.95, bins: int | None = None
) -> ToolResult:
    request = DistributionRequest(values=_column(name, values), level=level, bins=bins)
    return run_distribution(request)


def _test(result: ToolResult, name: str) -> HypothesisTest:
    return next(test for test in result.tests if test.name == name)


TestFunction = Callable[[NDArray[np.float64]], DiagnosticTest]
"""A normality test as the core exposes it: values in, a diagnostic out."""


def _no_group(name: str) -> GroupColumn:
    raise AssertionError(f"the tool takes no group column, asked for {name!r}")


# --------------------------------------------------------------------------- location


def test_the_mean_and_standard_deviation_are_the_values_own() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    result = _run(values)

    mean, spread = result.estimates
    array = np.asarray(values)
    n = array.size
    se = float(np.std(array, ddof=1)) / float(np.sqrt(n))
    t = float(scipy_stats.t.ppf(0.975, n - 1))

    assert (result.tool, result.level) == ("distribution", 0.95)
    assert mean.value == pytest.approx(float(np.mean(array)), rel=STAT_RTOL)
    assert mean.se == pytest.approx(se, rel=STAT_RTOL)
    assert mean.lower == pytest.approx(float(np.mean(array)) - t * se, rel=STAT_RTOL)
    assert mean.upper == pytest.approx(float(np.mean(array)) + t * se, rel=STAT_RTOL)
    assert spread.value == pytest.approx(float(np.std(array, ddof=1)), rel=STAT_RTOL)
    assert result.counts == Counts(n_offered=8, n_used=8, n_excluded=0)


def test_the_interval_widens_with_the_confidence_level() -> None:
    values = list(np.random.default_rng(3).standard_normal(40))

    narrow = _run(values, level=0.90).estimates[0]
    wide = _run(values, level=0.99).estimates[0]

    assert wide.lower < narrow.lower < narrow.upper < wide.upper


# --------------------------------------------------------------------------- normality


def test_shapiro_wilk_is_scipys_own_test() -> None:
    values = np.random.default_rng(7).standard_normal(100)
    result = _run(values.tolist())
    test = _test(result, "Shapiro-Wilk")

    reference = scipy_stats.shapiro(values)
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), rel=0.0, abs=P_ATOL)
    assert test.p_value_text == p_text(test.p_value)
    assert test.null_hypothesis == (
        "The values are drawn from a normal distribution, with its mean and variance estimated."
    )


def test_anderson_darling_matches_scipy_for_the_statistic_and_statsmodels_for_the_p_value() -> None:
    values = np.random.default_rng(0).standard_normal(100)
    result = _run(values.tolist())
    test = _test(result, "Anderson-Darling")

    n = values.size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        unadjusted = float(scipy_stats.anderson(values).statistic)
    factor = 1.0 + 0.75 / n + 2.25 / n**2
    assert test.statistic == pytest.approx(unadjusted * factor, rel=STAT_RTOL)

    _, p_value = normal_ad(values)
    assert test.p_value == pytest.approx(float(p_value), rel=0.0, abs=P_ATOL)
    assert "Stephens" in test.method


def test_the_tool_reports_the_cores_own_tests() -> None:
    values = np.random.default_rng(20).standard_normal(40)
    result = _run(values.tolist())

    assert _test(result, "Shapiro-Wilk").statistic == pytest.approx(
        float(shapiro_wilk(values).statistic), rel=STAT_RTOL
    )
    assert _test(result, "Anderson-Darling").statistic == pytest.approx(
        float(anderson_darling(values).statistic), rel=STAT_RTOL
    )


def test_between_three_and_seven_values_only_shapiro_wilk_runs() -> None:
    result = _run([1.0, 2.0, 4.0, 8.0, 16.0])

    shapiro = _test(result, "Shapiro-Wilk")
    anderson = _test(result, "Anderson-Darling")
    assert shapiro.p_value is not None
    assert anderson.statistic is None and anderson.p_value is None
    assert anderson.p_value_text == "not computed"
    assert anderson.reason is not None and "8" in anderson.reason
    assert "Anderson-Darling" not in result.interpretation


# --------------------------------------------------------------------------- the NIST example

_SOULEN = (
    2899,
    2898,
    2898,
    2900,
    2898,
    2901,
    2899,
    2901,
    2900,
    2898,
    2898,
    2898,
    2898,
    2900,
    2898,
    2897,
    2899,
    2897,
    2899,
    2899,
    2900,
    2897,
    2900,
    2900,
    2899,
    2898,
    2898,
    2899,
    2899,
    2899,
    2899,
    2899,
    2898,
    2899,
    2899,
    2899,
    2902,
    2899,
    2900,
    2898,
    2899,
    2899,
    2899,
    2899,
    2899,
    2899,
    2900,
    2899,
    2900,
    2898,
    2901,
    2900,
    2899,
    2899,
    2899,
    2899,
    2899,
    2900,
    2899,
    2898,
    2898,
    2898,
    2900,
    2896,
    2897,
    2899,
    2899,
    2900,
    2898,
    2900,
    2901,
    2898,
    2899,
    2901,
    2900,
    2898,
    2900,
    2899,
    2899,
    2897,
    2899,
    2898,
    2899,
    2899,
    2898,
    2899,
    2897,
    2899,
    2899,
    2897,
    2899,
    2897,
    2899,
    2897,
    2897,
    2899,
    2897,
    2898,
    2898,
    2899,
    2897,
    2898,
    2897,
    2899,
    2899,
    2898,
    2898,
    2897,
    2898,
    2895,
    2897,
    2898,
    2898,
    2896,
    2898,
    2898,
    2897,
    2896,
    2898,
    2898,
    2897,
    2897,
    2898,
    2898,
    2896,
    2898,
    2898,
    2896,
    2899,
    2898,
    2898,
    2898,
    2899,
    2899,
    2898,
    2898,
    2899,
    2899,
    2899,
    2900,
    2900,
    2901,
    2899,
    2898,
    2898,
    2900,
    2899,
    2898,
    2901,
    2897,
    2898,
    2898,
    2900,
    2899,
    2899,
    2898,
    2898,
    2899,
    2898,
    2901,
    2900,
    2897,
    2897,
    2898,
    2898,
    2900,
    2898,
    2899,
    2898,
    2898,
    2898,
    2896,
    2895,
    2898,
    2898,
    2898,
    2898,
    2897,
    2897,
    2895,
    2897,
    2897,
    2900,
    2898,
    2896,
    2897,
    2898,
    2898,
    2899,
    2898,
    2897,
    2898,
    2898,
    2896,
    2900,
    2899,
    2898,
    2896,
    2898,
    2896,
    2896,
    2896,
    2897,
    2897,
    2896,
    2897,
    2897,
    2896,
    2898,
    2896,
    2898,
    2896,
    2897,
    2896,
    2897,
    2897,
    2898,
    2897,
    2896,
    2895,
    2898,
    2896,
    2896,
    2898,
    2896,
    2898,
    2898,
    2897,
    2897,
    2898,
    2897,
    2899,
    2896,
    2897,
    2899,
    2900,
    2898,
    2898,
    2897,
    2898,
    2899,
    2899,
    2900,
    2900,
    2900,
    2900,
    2899,
    2899,
    2899,
    2898,
    2900,
    2901,
    2899,
    2898,
    2900,
    2901,
    2901,
    2900,
    2899,
    2898,
    2901,
    2899,
    2901,
    2900,
    2901,
    2898,
    2900,
    2900,
    2898,
    2900,
    2900,
    2898,
    2899,
    2901,
    2900,
    2899,
    2899,
    2900,
    2900,
    2899,
    2900,
    2901,
    2899,
    2898,
    2898,
    2899,
    2896,
    2898,
    2897,
    2898,
    2898,
    2897,
    2897,
    2897,
    2898,
    2897,
    2899,
    2900,
    2899,
    2897,
    2898,
    2900,
    2900,
    2898,
    2898,
    2899,
    2900,
    2898,
    2900,
    2900,
    2898,
    2900,
    2898,
    2898,
    2898,
    2898,
    2898,
    2899,
    2898,
    2900,
    2897,
    2899,
    2898,
    2899,
    2898,
    2897,
    2900,
    2901,
    2899,
    2898,
    2898,
    2901,
    2898,
    2899,
    2897,
    2899,
    2897,
    2896,
    2898,
    2898,
    2899,
    2900,
    2896,
    2897,
    2897,
    2898,
    2899,
    2899,
    2898,
    2898,
    2897,
    2897,
    2898,
    2897,
    2897,
    2898,
    2898,
    2898,
    2896,
    2895,
    2898,
    2898,
    2898,
    2896,
    2898,
    2898,
    2898,
    2897,
    2897,
    2899,
    2896,
    2900,
    2897,
    2897,
    2898,
    2896,
    2897,
    2898,
    2898,
    2898,
    2897,
    2897,
    2898,
    2899,
    2897,
    2898,
    2899,
    2897,
    2900,
    2896,
    2899,
    2897,
    2898,
    2897,
    2900,
    2899,
    2900,
    2897,
    2897,
    2898,
    2897,
    2899,
    2899,
    2898,
    2897,
    2901,
    2900,
    2898,
    2901,
    2899,
    2900,
    2899,
    2898,
    2900,
    2900,
    2899,
    2898,
    2897,
    2900,
    2898,
    2898,
    2897,
    2899,
    2898,
    2900,
    2899,
    2898,
    2899,
    2897,
    2900,
    2898,
    2902,
    2897,
    2898,
    2899,
    2899,
    2899,
    2898,
    2897,
    2898,
    2897,
    2898,
    2899,
    2900,
    2900,
    2899,
    2898,
    2899,
    2900,
    2899,
    2900,
    2899,
    2899,
    2899,
    2899,
    2899,
    2898,
    2899,
    2899,
    2900,
    2902,
    2899,
    2900,
    2900,
    2901,
    2899,
    2901,
    2899,
    2899,
    2902,
    2898,
    2898,
    2898,
    2898,
    2899,
    2899,
    2900,
    2900,
    2900,
    2898,
    2899,
    2899,
    2900,
    2899,
    2900,
    2899,
    2900,
    2898,
    2898,
    2898,
    2900,
    2898,
    2899,
    2900,
    2899,
    2899,
    2900,
    2898,
    2898,
    2899,
    2899,
    2899,
    2899,
    2898,
    2898,
    2897,
    2898,
    2899,
    2897,
    2897,
    2901,
    2898,
    2897,
    2898,
    2899,
    2898,
    2897,
    2899,
    2898,
    2897,
    2898,
    2898,
    2897,
    2898,
    2899,
    2899,
    2899,
    2899,
    2900,
    2899,
    2899,
    2897,
    2898,
    2899,
    2900,
    2898,
    2897,
    2901,
    2899,
    2901,
    2898,
    2899,
    2901,
    2900,
    2900,
    2899,
    2900,
    2900,
    2900,
    2900,
    2901,
    2900,
    2901,
    2899,
    2897,
    2900,
    2900,
    2901,
    2899,
    2898,
    2900,
    2899,
    2899,
    2900,
    2899,
    2900,
    2899,
    2900,
    2899,
    2901,
    2900,
    2900,
    2899,
    2899,
    2898,
    2899,
    2900,
    2898,
    2899,
    2899,
    2901,
    2898,
    2898,
    2900,
    2899,
    2899,
    2898,
    2897,
    2898,
    2897,
    2899,
    2899,
    2899,
    2898,
    2898,
    2897,
    2898,
    2899,
    2897,
    2897,
    2899,
    2898,
    2898,
    2899,
    2899,
    2901,
    2899,
    2899,
    2899,
    2897,
    2900,
    2896,
    2898,
    2898,
    2900,
    2897,
    2899,
    2897,
    2896,
    2898,
    2897,
    2898,
    2899,
    2896,
    2899,
    2901,
    2898,
    2898,
    2896,
    2897,
    2899,
    2897,
    2898,
    2899,
    2898,
    2898,
    2898,
    2898,
    2898,
    2898,
    2899,
    2900,
    2899,
    2901,
    2898,
    2899,
    2899,
    2898,
    2900,
    2898,
    2899,
    2899,
    2901,
    2900,
    2901,
    2899,
    2901,
    2899,
    2901,
    2899,
    2900,
    2902,
    2899,
    2898,
    2899,
    2900,
    2899,
    2900,
    2900,
    2901,
    2900,
    2899,
    2901,
    2901,
    2899,
    2898,
    2901,
    2897,
    2898,
    2901,
    2900,
    2902,
    2899,
    2900,
    2898,
    2900,
    2899,
    2900,
    2899,
    2899,
    2899,
    2898,
    2900,
    2898,
    2899,
    2899,
    2899,
    2899,
    2898,
    2900,
)
"""The 700 Josephson-junction voltage counts, as SOULEN.DAT holds them."""


def test_the_nist_josephson_example_reproduces_its_tabulated_statistics() -> None:
    values = np.asarray(_SOULEN, dtype=np.float64)
    result = _run(values.tolist(), name="voltage counts")

    mean, spread = result.estimates
    assert mean.value == pytest.approx(2898.562, rel=1e-6)  # the handbook prints 3 decimals
    assert spread.value == pytest.approx(1.305, rel=1e-3)

    anderson = _test(result, "Anderson-Darling")
    n = values.size
    unadjusted = anderson.statistic / (1.0 + 0.75 / n + 2.25 / n**2)
    assert unadjusted == pytest.approx(16.76349, rel=1e-4)

    # The two printed numbers imply Dataplot's own constant, 1 + 4/n - 25/n**2, and agree
    # with it at their printed precision.
    assert 16.76349 * (1.0 + 4.0 / n - 25.0 / n**2) == pytest.approx(16.85843, rel=1e-5)

    histogram = result.plots[0]
    assert isinstance(histogram, HistogramPlot)
    assert sum(histogram.counts) == pytest.approx(700.0, rel=0.0, abs=1e-9)
    assert sum(histogram.overlays[0].values) == pytest.approx(700.0, rel=1e-12)


# --------------------------------------------------------------------------- plots


def test_the_histogram_and_the_overlay_each_sum_to_n() -> None:
    values = np.random.default_rng(11).normal(5.0, 2.0, 137)
    result = _run(values.tolist())

    histogram = result.plots[0]
    assert isinstance(histogram, HistogramPlot)
    assert len(histogram.edges) == len(histogram.counts) + 1
    assert sum(histogram.counts) == pytest.approx(137.0, rel=0.0, abs=1e-9)
    assert len(histogram.overlays) == 1
    (overlay,) = histogram.overlays
    assert sum(overlay.values) == pytest.approx(137.0, rel=1e-12)
    assert all(value >= 0.0 for value in overlay.values)


def test_the_histogram_bins_follow_freedman_diaconis_when_none_is_asked() -> None:
    values = np.random.default_rng(5).normal(0.0, 1.0, 200)
    result = _run(values.tolist())

    histogram = result.plots[0]
    assert isinstance(histogram, HistogramPlot)
    expected = np.histogram_bin_edges(values, bins="fd")
    assert list(histogram.edges) == pytest.approx(expected.tolist(), rel=1e-15, abs=1e-15)


def test_a_requested_bin_count_is_used() -> None:
    result = _run(list(np.random.default_rng(6).standard_normal(50)), bins=7)

    histogram = result.plots[0]
    assert isinstance(histogram, HistogramPlot)
    assert len(histogram.counts) == 7 and len(histogram.edges) == 8
    assert "the requested number of bins" in result.methods_fragment


def test_a_column_with_no_interquartile_range_falls_back_to_sturges() -> None:
    # 70 of 80 values share the median, so both quartiles are 1.0 and the range cannot be
    # divided by an interquartile range.
    values = [1.0] * 70 + [5.0] * 5 + [9.0] * 5
    result = _run(values)

    histogram = result.plots[0]
    assert isinstance(histogram, HistogramPlot)
    expected = np.histogram_bin_edges(np.asarray(values), bins="sturges")
    assert list(histogram.edges) == pytest.approx(expected.tolist())


def test_the_qq_plot_uses_blom_positions_for_the_fitted_normal() -> None:
    values = np.random.default_rng(8).normal(10.0, 3.0, 50)
    result = _run(values.tolist())

    plot = result.plots[1]
    assert isinstance(plot, QQPlot)
    n = values.size
    blom = (np.arange(1.0, n + 1.0) - 0.375) / (n + 0.25)
    mean = float(np.mean(values))
    spread = float(np.std(values, ddof=1))
    expected = mean + spread * scipy_stats.norm.ppf(blom)
    assert list(plot.theoretical) == pytest.approx(expected.tolist(), rel=STAT_RTOL)
    assert list(plot.ordered) == pytest.approx(np.sort(values).tolist())
    assert plot.line.x0 == plot.line.y0 and plot.line.x1 == plot.line.y1


def test_a_plot_holds_at_most_two_thousand_points() -> None:
    values = np.random.default_rng(9).standard_normal(3_000)
    result = _run(values.tolist())

    histogram = result.plots[0]
    qq = result.plots[1]
    assert isinstance(histogram, HistogramPlot)
    assert isinstance(qq, QQPlot)
    assert len(histogram.counts) <= MAX_BINS and len(histogram.edges) <= 2_000
    assert sum(histogram.counts) == pytest.approx(3_000.0, rel=0.0, abs=1e-9)
    assert len(qq.theoretical) == 2_000 and len(qq.ordered) == 2_000
    assert qq.note is not None and "stride" in qq.note
    assert qq.theoretical[0] < qq.theoretical[-1] and qq.ordered[0] < qq.ordered[-1]


# --------------------------------------------------------------------------- many rows


def test_above_five_thousand_shapiro_wilk_uses_a_seeded_subsample_and_says_so() -> None:
    values = np.random.default_rng(12).standard_normal(6_000)
    request = DistributionRequest(values=_column("y", values.tolist()))

    first = run_distribution(request)
    again = run_distribution(request)
    assert first == again

    assert any(f"{MAX_SHAPIRO:,}" in warning and "6,000" in warning for warning in first.warnings)
    chosen = np.sort(np.random.default_rng(0).choice(values.size, size=MAX_SHAPIRO, replace=False))
    shapiro = _test(first, "Shapiro-Wilk")
    reference = scipy_stats.shapiro(values[chosen])
    assert shapiro.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)

    # Anderson-Darling is run on every value, not the subsample.
    anderson = _test(first, "Anderson-Darling")
    assert anderson.statistic == pytest.approx(
        float(anderson_darling(values).statistic), rel=STAT_RTOL
    )


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("size", [0, 1, 2])
def test_fewer_than_three_usable_values_is_too_few_rows(size: int) -> None:
    with pytest.raises(ToolError) as refused:
        _run([float(index) for index in range(size)])

    assert refused.value.code == "TOO_FEW_ROWS"
    assert "3" in refused.value.message


def test_a_constant_column_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run([4.0, 4.0, 4.0, 4.0])

    assert refused.value.code == "CONSTANT_COLUMN"
    assert "repeated" in refused.value.message


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        _run([1.0, 2.0, 3.0, 4.0], level=level)

    assert refused.value.code == "INVALID_OPTION"


@pytest.mark.parametrize("bins", [0, -3, MAX_BINS + 1, 10_000])
def test_a_bin_count_a_plot_cannot_hold_is_refused(bins: int) -> None:
    with pytest.raises(ToolError) as refused:
        _run([float(index) for index in range(20)], bins=bins)

    assert refused.value.code == "INVALID_OPTION"
    assert "bins" in refused.value.message


# --------------------------------------------------------------------------- excluded rows


def test_missing_and_non_finite_values_are_listed_not_dropped() -> None:
    values = [1.0, float("nan"), 3.0, float("inf"), 5.0, 7.0, float("-inf")]
    result = _run(values, name="depth")

    assert result.counts == Counts(n_offered=7, n_used=4, n_excluded=3)
    assert result.excluded == (
        ExcludedRow(index=1, reason="missing in depth"),
        ExcludedRow(index=3, reason="not a finite number in depth"),
        ExcludedRow(index=6, reason="not a finite number in depth"),
    )
    assert result.excluded_unlisted == 0


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    values = [1.0, 2.0, 3.0] + [float("nan")] * 255
    result = _run(values)

    assert result.counts.n_used == 3 and result.counts.n_excluded == 255
    assert len(result.excluded) == MAX_LISTED_EXCLUDED
    assert result.excluded_unlisted == 255 - MAX_LISTED_EXCLUDED


# --------------------------------------------------------------------------- the result's words


def test_the_assumption_says_independence_cannot_be_read_from_a_column() -> None:
    result = _run([float(index) for index in range(10)])

    (check,) = result.assumptions
    assert check.id == "independence"
    assert check.verdict is Verdict.NOT_APPLICABLE
    assert "column" in check.evidence
    assert check.explanation == "This assumption could not be read from the data."
    assert check.next_step is not None


def test_every_test_carries_the_interval_for_the_mean() -> None:
    result = _run(list(np.random.default_rng(13).standard_normal(25)))
    mean = result.estimates[0]

    for test in result.tests:
        assert test.effect == mean
        assert test.effect is not None
        assert test.effect.lower is not None and test.effect.upper is not None
        assert test.p_value_text == p_text(test.p_value)


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    result = _run(list(np.random.default_rng(14).standard_normal(30)))

    texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
    for text in texts:
        assert not claims_proof(text), text

    for method in ("Shapiro-Wilk", "Anderson-Darling", "Stephens", "Freedman-Diaconis", "Blom"):
        assert method in result.methods_fragment


# --------------------------------------------------------------------------- size and power


@pytest.mark.parametrize(
    ("name", "test"), [("Shapiro-Wilk", shapiro_wilk), ("Anderson-Darling", anderson_darling)]
)
def test_each_test_rejects_at_its_level_on_normal_data(name: str, test: TestFunction) -> None:
    def draw(rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(20)

    share = simulated_size(draw, lambda sample: test(sample).p_value, draws=4_000, alpha=0.05)
    assert_size(share)


@pytest.mark.parametrize(
    ("name", "test"), [("Shapiro-Wilk", shapiro_wilk), ("Anderson-Darling", anderson_darling)]
)
def test_each_test_rejects_ninety_percent_of_exponential_samples(
    name: str, test: TestFunction
) -> None:
    def draw(rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(1.0, 50)

    share = simulated_size(draw, lambda sample: test(sample).p_value, draws=500, alpha=0.05)
    assert share >= 0.90, f"{name} rejected {share:.1%} of exponential samples"


# --------------------------------------------------------------------------- the registry


def test_the_registry_lists_and_runs_the_tool() -> None:
    assert "curvestudio.tools.stats.distribution" in TOOL_MODULES

    info = TOOLS.info("distribution")
    assert (info.name, info.family, info.title, info.needs_data) == (
        "distribution",
        "summarise",
        "Distribution analysis",
        True,
    )
    assert [(field.name, field.kind, field.label, field.required) for field in info.fields] == [
        ("values", "column", "Values", True),
        ("level", "option", "Level", False),
        ("bins", "option", "Bins", False),
    ]
    assert all(field.description for field in info.fields)
    assert "distribution" in [entry.name for entry in catalogue()]

    values = _column("y", [1.0, 2.0, 3.0, 4.0, 10.0])
    request = TOOLS.build_request(
        "distribution",
        {"values": "y", "level": 0.9, "bins": 4},
        column=lambda name: values,
        group=_no_group,
    )
    assert isinstance(request, DistributionRequest)
    assert (request.level, request.bins) == (0.9, 4)
    assert TOOLS.run("distribution", request).tool == "distribution"
