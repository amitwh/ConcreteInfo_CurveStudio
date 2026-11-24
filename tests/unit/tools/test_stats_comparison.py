"""The Bland-Altman agreement tool: bias, limits of agreement and proportional bias.

Every number is checked against an independent implementation written here from the paper's
formulas: the bias is the mean of the differences and its interval is the bias plus or minus
t times ``s / sqrt(n)``; the limits are the bias plus or minus 1.96 s, and a limit's standard
error is ``s * sqrt(1/n + 1.96**2 / (2 (n - 1)))``, the form Bland and Altman (1999) give
because the sample mean and standard deviation of normal differences are independent. The
proportional-bias test is the ordinary least squares slope of the difference on the mean of
the two methods, with the t statistic on n - 2 degrees of freedom.

The worked example is Bland and Altman (1986), "Statistical methods for assessing agreement
between two methods of clinical measurement", Lancet 1:307-310, Table 1: peak expiratory flow
rate measured by a Wright and a mini Wright meter on 17 subjects. The PDF was downloaded from
Martin Bland's University of York archive (https://www-users.york.ac.uk/~mb55/meas/ba.pdf) and
read on 2026-11-24; only the first reading by each method is used, as the paper says. Its
printed summary is a bias of -2.1 l/min, a standard deviation of 38.8 l/min and a 95 %
interval for the bias of -22.0 to 17.8 l/min, all reproduced here. The paper prints its
limits with a multiplier of 2, not 1.96 (-79.7 and 75.5 l/min), and the standard error of a
limit with the approximation ``sqrt(3 s**2 / n)``, so the test checks those printed numbers
with the paper's own convention and the 1.96-based limits and their intervals with the
formula above.

Bland and Altman (1999), "Measuring agreement in method comparison studies", Statistical
Methods in Medical Research 8:135-160, could not be reached: the SAGE copy is closed and the
DOI has no open copy in OpenAlex or Semantic Scholar, and the York archive does not host it.
The 1986 paper above is therefore the published source, and the 1999 interval formula is
implemented from the independence of the sample mean and standard deviation, as written out
in the reference implementation and in the tool's ``_limit_se``.

The size of the slope's t test is measured on differences drawn independently of the means,
and its power on a proportional bias of 0.2 per unit at 50 pairs. The property every such
fit has is checked: adding a constant to both methods leaves the differences unchanged and
moves only the intercept of the regression of the difference on the mean.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats

from curvestudio.data import Column
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    ScatterPlot,
    ToolResult,
    Verdict,
    assert_size,
    build_request,
    p_text,
    run_tool,
    simulated_size,
)
from curvestudio.tools.stats.comparison import (
    BlandAltmanRequest,
    _difference_fit,
    run_blandaltman,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, catalogue

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""
P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""

LIMIT_MULTIPLIER = 1.96


class _Reference(NamedTuple):
    """The comparison as the paper's formulas give it, written out independently."""

    bias: float
    bias_lower: float
    bias_upper: float
    sd: float
    lower: float
    lower_lower: float
    lower_upper: float
    upper: float
    upper_lower: float
    upper_upper: float
    slope: float
    slope_se: float
    slope_statistic: float
    slope_p: float
    slope_lower: float
    slope_upper: float


def _reference(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> _Reference:
    """Bland-Altman agreement by the formulas, from the two methods' readings."""
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    means = (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0
    n = differences.size
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, n - 1))
    bias_half = critical * sd / math.sqrt(n)
    lower = bias - LIMIT_MULTIPLIER * sd
    upper = bias + LIMIT_MULTIPLIER * sd
    limit_se = sd * math.sqrt(1.0 / n + LIMIT_MULTIPLIER**2 / (2.0 * (n - 1)))
    limit_half = critical * limit_se
    centred = means - float(np.mean(means))
    sxx = float(centred @ centred)
    slope = float(centred @ (differences - float(np.mean(differences)))) / sxx
    intercept = float(np.mean(differences)) - slope * float(np.mean(means))
    residual = differences - (intercept + slope * means)
    df = n - 2
    slope_se = math.sqrt(float(residual @ residual) / df / sxx)
    statistic = slope / slope_se
    p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), df))
    slope_critical = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    return _Reference(
        bias=bias,
        bias_lower=bias - bias_half,
        bias_upper=bias + bias_half,
        sd=sd,
        lower=lower,
        lower_lower=lower - limit_half,
        lower_upper=lower + limit_half,
        upper=upper,
        upper_lower=upper - limit_half,
        upper_upper=upper + limit_half,
        slope=slope,
        slope_se=slope_se,
        slope_statistic=statistic,
        slope_p=p_value,
        slope_lower=slope - slope_critical * slope_se,
        slope_upper=slope + slope_critical * slope_se,
    )


def _column(name: str, values: object) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    return Column(name=name, values=array, rows=np.arange(array.size, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_blandaltman(BlandAltmanRequest(**options))  # type: ignore[arg-type]


def _sample(n: int = 40, seed: int = 0, noise: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Two methods that measure a common reading, the second with independent noise."""
    rng = np.random.default_rng(seed)
    first = rng.normal(50.0, 10.0, n)
    return first, first + rng.normal(0.0, noise, n)


# --------------------------------------------------------------------------- the references


def test_every_number_matches_the_independent_formulas() -> None:
    first, second = _sample(seed=0)
    expected = _reference(first, second)

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    bias, lower, upper, sd = result.estimates
    assert bias.label == "Bias"
    assert bias.value == pytest.approx(expected.bias, rel=STAT_RTOL)
    assert bias.lower == pytest.approx(expected.bias_lower, rel=STAT_RTOL)
    assert bias.upper == pytest.approx(expected.bias_upper, rel=STAT_RTOL)
    assert sd.label == "Standard deviation of the differences"
    assert sd.value == pytest.approx(expected.sd, rel=STAT_RTOL)

    assert lower.label == "Lower limit of agreement"
    assert lower.value == pytest.approx(expected.lower, rel=STAT_RTOL)
    assert lower.se == pytest.approx(
        expected.sd * math.sqrt(1.0 / 40 + LIMIT_MULTIPLIER**2 / 78.0), rel=STAT_RTOL
    )
    assert lower.lower == pytest.approx(expected.lower_lower, rel=STAT_RTOL)
    assert lower.upper == pytest.approx(expected.lower_upper, rel=STAT_RTOL)

    assert upper.label == "Upper limit of agreement"
    assert upper.value == pytest.approx(expected.upper, rel=STAT_RTOL)
    assert upper.lower == pytest.approx(expected.upper_lower, rel=STAT_RTOL)
    assert upper.upper == pytest.approx(expected.upper_upper, rel=STAT_RTOL)

    test = result.tests[0]
    assert test.statistic == pytest.approx(expected.slope_statistic, rel=STAT_RTOL)
    assert test.p_value == pytest.approx(expected.slope_p, abs=P_ATOL)
    assert test.df == 38.0
    assert test.effect is not None
    assert test.effect.label == "Slope"
    assert test.effect.value == pytest.approx(expected.slope, rel=STAT_RTOL)
    assert test.effect.se == pytest.approx(expected.slope_se, rel=STAT_RTOL)
    assert test.effect.lower == pytest.approx(expected.slope_lower, rel=STAT_RTOL)
    assert test.effect.upper == pytest.approx(expected.slope_upper, rel=STAT_RTOL)


def test_the_level_moves_the_intervals_but_not_the_limits() -> None:
    first, second = _sample(n=30, seed=1)
    ninety = _reference(first, second, level=0.90)

    result = _run(method_a=_column("a", first), method_b=_column("b", second), level=0.90)

    bias, lower, upper, _sd = result.estimates
    assert bias.lower == pytest.approx(ninety.bias_lower, rel=STAT_RTOL)
    assert bias.upper == pytest.approx(ninety.bias_upper, rel=STAT_RTOL)
    assert lower.value == pytest.approx(ninety.lower, rel=STAT_RTOL)
    assert lower.lower == pytest.approx(ninety.lower_lower, rel=STAT_RTOL)
    assert lower.upper == pytest.approx(ninety.lower_upper, rel=STAT_RTOL)
    assert upper.lower == pytest.approx(ninety.upper_lower, rel=STAT_RTOL)
    assert upper.upper == pytest.approx(ninety.upper_upper, rel=STAT_RTOL)
    assert result.tests[0].effect is not None
    assert result.tests[0].effect.lower == pytest.approx(ninety.slope_lower, rel=STAT_RTOL)
    assert result.tests[0].effect.upper == pytest.approx(ninety.slope_upper, rel=STAT_RTOL)

    at_default = _run(method_a=_column("a", first), method_b=_column("b", second))
    assert at_default.estimates[1].value == pytest.approx(lower.value, rel=STAT_RTOL)
    assert at_default.estimates[2].value == pytest.approx(upper.value, rel=STAT_RTOL)


def test_the_1986_pefr_example_reproduces_the_tabulated_summary() -> None:
    """Bland and Altman (1986), Table 1, first readings, read from the PDF named above.

    The paper rounds its summary, so each number is held to that rounding. Its limits use a
    multiplier of 2 and its limit standard errors the approximation ``sqrt(3 s**2 / n)``;
    this tool uses 1.96 and the 1999 formula, so the two are compared where they agree and
    the tool's own numbers are held to the formula written out above.
    """
    large = np.asarray(
        [494, 395, 516, 434, 476, 557, 413, 442, 650, 433, 417, 656, 267, 478, 178, 423, 427],
        dtype=np.float64,
    )
    mini = np.asarray(
        [512, 430, 520, 428, 500, 600, 364, 380, 658, 445, 432, 626, 260, 477, 259, 350, 451],
        dtype=np.float64,
    )
    expected = _reference(large, mini)

    result = _run(method_a=_column("large", large), method_b=_column("mini", mini))
    bias, lower, upper, sd = result.estimates

    # The paper's printed bias, standard deviation and interval for the bias.
    assert bias.value == pytest.approx(-2.1, abs=0.05)
    assert sd.value == pytest.approx(38.8, abs=0.05)
    assert bias.lower == pytest.approx(-22.0, abs=0.1)
    assert bias.upper == pytest.approx(17.8, abs=0.1)
    # The paper's printed limits are the bias plus or minus two standard deviations.
    assert bias.value - 2.0 * sd.value == pytest.approx(-79.7, abs=0.1)
    assert bias.value + 2.0 * sd.value == pytest.approx(75.5, abs=0.1)
    # This tool's 1.96 limits and their intervals are the formulas' own, within the 2 l/min
    # the paper's approximation moves the printed intervals by.
    assert lower.value == pytest.approx(expected.lower, rel=STAT_RTOL)
    assert upper.value == pytest.approx(expected.upper, rel=STAT_RTOL)
    assert lower.lower == pytest.approx(-114.3, abs=2.0)
    assert lower.upper == pytest.approx(-45.1, abs=2.0)
    assert upper.lower == pytest.approx(40.9, abs=2.0)
    assert upper.upper == pytest.approx(110.1, abs=2.0)
    # The paper finds no obvious relation between the difference and the mean.
    assert result.tests[0].p_value is not None
    assert result.tests[0].p_value > 0.05


# --------------------------------------------------------------------------- size and power


def _null_draw(n: int = 50) -> Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]:
    """A mean that varies, and a difference drawn independently of it: no proportional bias."""

    def draw(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        means = rng.normal(100.0, 20.0, n)
        return means, rng.normal(0.0, 1.0, n)

    return draw


def _slope_p(sample: tuple[np.ndarray, np.ndarray]) -> float | None:
    means, differences = sample
    return _difference_fit(means, differences, 0.95).p_value


def test_the_slope_test_rejects_at_its_level_when_the_difference_is_unrelated_to_the_mean() -> None:
    share = simulated_size(_null_draw(), _slope_p, draws=4_000, seed=0)

    assert_size(share)


def _proportional_draw(
    n: int = 50,
) -> Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]]:
    """A difference that grows by 0.2 per unit of the mean, with normal noise around it."""

    def draw(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        means = rng.normal(100.0, 20.0, n)
        return means, 0.2 * means + rng.normal(0.0, 5.0, n)

    return draw


def test_a_proportional_bias_of_point_two_per_unit_is_found_in_ninety_percent_of_draws() -> None:
    power = simulated_size(_proportional_draw(), _slope_p, draws=1_000, seed=1)

    assert power >= 0.90


# --------------------------------------------------------------------------- the properties


def test_adding_a_constant_to_both_methods_changes_only_the_intercept() -> None:
    first, second = _sample(seed=2)
    plain = _run(method_a=_column("a", first), method_b=_column("b", second))
    shifted = _run(method_a=_column("a", first + 7.0), method_b=_column("b", second + 7.0))

    for before, after in zip(plain.estimates, shifted.estimates, strict=True):
        assert after.label == before.label
        assert after.value == pytest.approx(before.value, rel=STAT_RTOL)
        if before.se is None:
            assert after.se is None
        else:
            assert after.se == pytest.approx(before.se, rel=STAT_RTOL)
        for bound in ("lower", "upper"):
            old, new = getattr(before, bound), getattr(after, bound)
            if old is None:
                assert new is None
            else:
                assert new == pytest.approx(old, rel=STAT_RTOL)

    differences = first - second
    means = (first + second) / 2.0
    plain_fit = _difference_fit(means, differences, 0.95)
    shifted_fit = _difference_fit(means + 7.0, differences, 0.95)
    assert shifted_fit.slope == pytest.approx(plain_fit.slope, rel=STAT_RTOL)
    assert shifted_fit.intercept == pytest.approx(
        plain_fit.intercept - 7.0 * plain_fit.slope, rel=STAT_RTOL
    )
    assert shifted.tests[0].statistic == pytest.approx(
        plain.tests[0].statistic or 0.0, rel=STAT_RTOL
    )
    assert shifted.tests[0].p_value == pytest.approx(plain.tests[0].p_value or 0.0, abs=P_ATOL)


# --------------------------------------------------------------------------- the edge


def test_identical_methods_give_zero_limits_no_slope_test_and_a_warning() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(50.0, 10.0, 25)

    result = _run(method_a=_column("a", values), method_b=_column("a", values))

    assert [estimate.value for estimate in result.estimates] == [0.0, 0.0, 0.0, 0.0]
    for estimate in result.estimates[:3]:
        assert estimate.lower == 0.0 and estimate.upper == 0.0
    test = result.tests[0]
    assert test.statistic is None and test.p_value is None and test.effect is None
    assert test.reason is not None and "difference" in test.reason
    assert result.warnings and "zero" in result.warnings[0]
    assert all(check.verdict is Verdict.NOT_APPLICABLE for check in result.assumptions)
    assert result.tables == ()
    assert len(result.plots) == 2


def test_a_mean_that_does_not_vary_leaves_the_slope_untested() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    first = values
    second = 6.0 - values

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    test = result.tests[0]
    assert test.statistic is None and test.p_value is None and test.effect is None
    assert test.reason is not None and "mean of the two methods" in test.reason
    assert result.assumptions[1].verdict is Verdict.NOT_APPLICABLE


# --------------------------------------------------------------------------- the blocks


def test_the_result_is_the_contract_with_the_estimates_and_the_slope_test() -> None:
    first, second = _sample(n=30, seed=4)

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    assert type(result) is ToolResult
    assert result.tool == "blandaltman"
    assert result.title == "Bland\u2013Altman agreement between a and b"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=30, n_used=30, n_excluded=0)
    assert result.excluded == ()
    assert result.excluded_unlisted == 0
    assert [estimate.label for estimate in result.estimates] == [
        "Bias",
        "Lower limit of agreement",
        "Upper limit of agreement",
        "Standard deviation of the differences",
    ]
    assert len(result.tests) == 1
    test = result.tests[0]
    assert test.name == "Ordinary least squares of the difference on the mean"
    assert test.method == (
        "Ordinary least squares regression of the difference on the mean of the two methods, "
        "t test of the slope"
    )
    assert test.effect is not None and test.effect.label == "Slope"
    assert test.p_value_text == p_text(test.p_value)


def test_the_plots_are_the_bland_altman_and_the_identity_scatter() -> None:
    first, second = _sample(n=25, seed=5)

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    assert [type(plot) for plot in result.plots] == [ScatterPlot, ScatterPlot]
    difference_plot = result.plots[0]
    assert isinstance(difference_plot, ScatterPlot)
    assert difference_plot.title == "Bland\u2013Altman"
    assert difference_plot.x_label == "Mean of a and b"
    assert difference_plot.y_label == "Difference (a minus b)"
    assert len(difference_plot.x) == len(first)
    assert [line.label for line in difference_plot.lines] == [
        "Bias",
        "Lower limit of agreement",
        "Upper limit of agreement",
        "Zero",
    ]
    bias = result.estimates[0].value
    lower = result.estimates[1].value
    upper = result.estimates[2].value
    assert [line.y0 for line in difference_plot.lines] == pytest.approx([bias, lower, upper, 0.0])
    assert all(line.y0 == line.y1 for line in difference_plot.lines)

    identity = result.plots[1]
    assert isinstance(identity, ScatterPlot)
    assert identity.title == "a against b"
    assert identity.x_label == "b" and identity.y_label == "a"
    assert np.allclose(np.asarray(identity.x), np.asarray(second))
    assert np.allclose(np.asarray(identity.y), np.asarray(first))
    assert len(identity.lines) == 1 and identity.lines[0].label == "Identity"
    assert identity.lines[0].y0 == identity.lines[0].x0
    assert identity.lines[0].y1 == identity.lines[0].x1


def test_the_outside_table_lists_the_points_beyond_the_limits() -> None:
    first = np.asarray([10.0, 12.0, 14.0, 16.0, 18.0, 30.0])
    second = np.asarray([11.0, 11.0, 15.0, 15.0, 19.0, 20.0])

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    assert [table.title for table in result.tables] == ["Points outside the limits of agreement"]
    outside = result.tables[0]
    assert [column.label for column in outside.columns] == ["Row", "Mean", "Difference"]
    assert outside.rows == ((5, 25.0, 10.0),)
    assert outside.note is not None and "position" in outside.note


def test_the_assumptions_check_normality_and_a_constant_variance() -> None:
    first, second = _sample(n=200, seed=6)

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    assert [check.id for check in result.assumptions] == ["normality", "constant_variance"]
    normality, variance = result.assumptions
    assert normality.verdict is not Verdict.NOT_APPLICABLE
    assert normality.statistic is not None and normality.statistic.startswith("W = ")
    assert variance.verdict is not Verdict.NOT_APPLICABLE
    assert variance.statistic is not None and variance.statistic.startswith("LM = ")
    assert normality.threshold is not None and variance.threshold is not None


def test_a_spread_that_grows_with_the_mean_fails_the_constant_variance_check() -> None:
    rng = np.random.default_rng(7)
    means = np.linspace(1.0, 50.0, 300)
    differences = means * rng.standard_normal(300) * 0.4

    result = _run(
        method_a=_column("a", means + differences / 2.0),
        method_b=_column("b", means - differences / 2.0),
    )

    variance = next(check for check in result.assumptions if check.id == "constant_variance")
    assert variance.verdict is Verdict.FAIL


# --------------------------------------------------------------------------- missing data


def test_a_row_missing_in_one_method_is_left_out_of_both_and_listed() -> None:
    first = _column("a", [1.0, 2.0, float("nan"), 4.0, 5.0])
    second = _column("b", [1.0, 2.0, 3.0, 4.0, 5.0])

    result = _run(method_a=first, method_b=second)

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="missing in a"),)


def test_a_row_missing_in_the_second_method_is_listed_after_the_first() -> None:
    first = _column("a", [1.0, 2.0, 3.0, 4.0])
    second = _column("b", [1.0, float("nan"), 4.0, 4.0])

    result = _run(method_a=first, method_b=second)

    assert result.counts == Counts(n_offered=4, n_used=3, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=1, reason="missing in b"),)


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    first = _column("a", [float("nan")] * 205 + list(np.arange(1.0, 11.0)))
    second = _column("b", np.arange(1.0, 216.0))

    result = _run(method_a=first, method_b=second)

    assert result.counts.n_excluded == 205
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 5


# --------------------------------------------------------------------------- the refusals


def test_fewer_than_three_pairs_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(method_a=_column("a", [1.0, 2.0]), method_b=_column("b", [1.0, 3.0]))
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "at least 3 complete pairs" in refused.value.message


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    first, second = _sample(n=10, seed=8)
    with pytest.raises(ToolError) as refused:
        _run(method_a=_column("a", first), method_b=_column("b", second), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_blandaltman(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    first, second = _sample(n=10, seed=9)
    with pytest.raises(ValidationError):
        BlandAltmanRequest(  # type: ignore[call-arg]
            method_a=_column("a", first),
            method_b=_column("b", second),
            method="passing-bablok",
        )


# ---------------------------------------------------------------------- the words and the registry


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    first, second = _sample(seed=10)

    result = _run(method_a=_column("a", first), method_b=_column("b", second))

    texts = [result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
    for table in result.tables:
        texts.extend([table.title, table.note or ""])
    for plot in result.plots:
        texts.extend([plot.title, plot.note or ""])
    test = result.tests[0]
    texts.extend([test.name, test.method, test.null_hypothesis])
    for text in texts:
        assert not claims_proof(text), text


def test_every_request_field_is_described_and_the_tool_is_in_its_family() -> None:
    found = next(info for info in catalogue() if info.name == "blandaltman")

    assert found.family == "comparison"
    assert found.title == "Bland\u2013Altman agreement"
    assert found.needs_data
    assert [field.name for field in found.fields] == ["method_a", "method_b", "level"]
    assert [field.kind for field in found.fields] == ["column", "column", "option"]
    assert [field.label for field in found.fields] == ["Method A", "Method B", "Level"]
    assert all(field.description for field in found.fields)
    assert "curvestudio.tools.stats.comparison" in TOOL_MODULES


def test_the_registry_builds_and_runs_the_tool_from_column_names() -> None:
    first, second = _sample(n=15, seed=11)
    columns = {"a": _column("a", first), "b": _column("b", second)}

    def group(name: str) -> object:
        raise AssertionError("blandaltman takes no group column")

    request = build_request(
        "blandaltman",
        {"method_a": "a", "method_b": "b", "level": 0.9},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    result = run_tool("blandaltman", request)

    assert isinstance(request, BlandAltmanRequest)
    assert result.tool == "blandaltman"
    assert result.level == 0.9
    assert result.title == "Bland\u2013Altman agreement between a and b"
