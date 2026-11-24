"""The t tests: one sample, two independent samples and paired samples.

The reference numbers come from SciPy and statsmodels. Two worked examples come from the
NIST/SEMATECH e-Handbook of Statistical Methods, both pages fetched on 2026-11-24:

- ``1.3.5.3. Two-Sample t-Test for Equal Means``
  (itl.nist.gov/div898/handbook/eda/section3/eda353.htm) prints, for the AUTO83B.DAT data
  set, the two samples' sizes, means and standard deviations, a pooled standard deviation
  of 6.34260, a test statistic of -12.62059 and 326 degrees of freedom under the
  equal-variances assumption. The data file is
  itl.nist.gov/div898/handbook/datasets/AUTO83B.DAT; its Japan column marks the missing
  entries with -999, which the handbook's own Dataplot script deletes.
- ``7.3.1.1. Analysis of paired observations``
  (itl.nist.gov/div898/handbook/prc/section3/prc311.htm) gives the paired test's formulas
  and degrees of freedom but tabulates no example data, so no paired numbers are
  reproduced from it; the paired test is held to SciPy and statsmodels instead.

The size of every test is measured on data drawn from its null hypothesis with a fixed
seed, and the Student control shows why Welch's test is the default: on the same unequal-
variance data Student's test rejects far more often than its level allows. The size tests
call the statistic functions the tool itself calls, so they measure the same test without
building 4,000 results.
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import optimize
from scipy import stats as scipy_stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ExcludedRow,
    ToolResult,
    Verdict,
    assert_size,
    p_text,
    simulated_size,
)
from curvestudio.tools.stats.means import (
    TTestRequest,
    _independent_statistic,
    _one_sample_statistic,
    _paired_statistic,
    _sample,
    run_ttest,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, TOOLS, catalogue

P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""
STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _group(name: str, labels: object, rows: object | None = None) -> GroupColumn:
    """A group column whose labels are the data layer's object array of strings."""
    array = np.asarray(labels, dtype=object)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return GroupColumn(name=name, labels=array, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_ttest(TTestRequest(**options))  # type: ignore[arg-type]


def _one(values: object, *, null: float = 0.0, level: float = 0.95) -> ToolResult:
    return _run(kind="one_sample", a=_column("y", values), null=null, level=level)


def _independent_groups(
    a: object,
    b: object,
    *,
    null: float = 0.0,
    equal_variances: bool = False,
    level: float = 0.95,
) -> ToolResult:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    values = np.concatenate([first, second])
    labels = ["first"] * first.size + ["second"] * second.size
    return _run(
        kind="independent",
        a=_column("y", values),
        groups=_group("g", labels),
        null=null,
        equal_variances=equal_variances,
        level=level,
    )


def _independent_columns(
    a: object,
    b: object,
    *,
    null: float = 0.0,
    equal_variances: bool = False,
    level: float = 0.95,
) -> ToolResult:
    return _run(
        kind="independent",
        a=_column("first", a),
        b=_column("second", b),
        null=null,
        equal_variances=equal_variances,
        level=level,
    )


def _paired(
    a: object, b: object | None = None, *, null: float = 0.0, level: float = 0.95
) -> ToolResult:
    return _run(
        kind="paired",
        a=_column("before", a),
        b=None if b is None else _column("after", b),
        null=null,
        level=level,
    )


def _no_group(name: str) -> GroupColumn:
    raise AssertionError(f"the tool takes no group column, asked for {name!r}")


# --------------------------------------------------------------------------- one sample


def test_the_one_sample_test_is_scipys_own() -> None:
    values = np.random.default_rng(1).normal(4.5, 1.5, 25)
    result = _one(values, null=4.0)
    (test,) = result.tests

    reference = scipy_stats.ttest_1samp(values, 4.0)
    assert test.name == "One-sample t test"
    assert test.method == "One-sample t test, two-sided"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == 24.0
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.p_value_text == p_text(test.p_value)
    assert test.null_hypothesis == "The mean of y is 4."
    assert result.tool == "ttest" and result.level == 0.95


def test_the_one_sample_difference_has_the_t_interval_for_the_mean() -> None:
    values = np.random.default_rng(2).normal(2.0, 0.8, 30)
    result = _one(values, null=1.5)

    difference, effect = result.estimates
    lower, upper = DescrStatsW(values).tconfint_mean(alpha=0.05)
    assert difference.value == pytest.approx(float(np.mean(values)) - 1.5, rel=STAT_RTOL)
    assert difference.se == pytest.approx(
        float(np.std(values, ddof=1)) / math.sqrt(values.size), rel=STAT_RTOL
    )
    assert difference.lower == pytest.approx(float(lower) - 1.5, rel=STAT_RTOL)
    assert difference.upper == pytest.approx(float(upper) - 1.5, rel=STAT_RTOL)
    assert effect.label == "Hedges' g"
    assert effect.value > 0.0


def test_the_one_sample_hedges_g_follows_the_non_central_t_formula() -> None:
    values = np.random.default_rng(3).normal(1.0, 2.0, 40)
    result = _one(values, null=0.25)

    effect = result.estimates[1]
    expected, lower, upper = _expected_one_sample_g(values, 0.25, 0.95)
    assert effect.value == pytest.approx(expected, rel=STAT_RTOL)
    assert effect.lower == pytest.approx(lower, rel=STAT_RTOL)
    assert effect.upper == pytest.approx(upper, rel=STAT_RTOL)
    assert effect.lower < effect.value < effect.upper


# --------------------------------------------------------------------------- independent


def test_welch_matches_scipy_and_statsmodels() -> None:
    first = np.random.default_rng(4).normal(0.5, 1.0, 25)
    second = np.random.default_rng(5).normal(0.0, 2.5, 40)
    result = _independent_columns(first, second)
    (test,) = result.tests

    reference = scipy_stats.ttest_ind(first, second, equal_var=False)
    assert test.name == "Welch's t test"
    assert test.method == "Welch's t test, two-sided"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == pytest.approx(float(reference.df), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)

    difference = result.estimates[0]
    lower, upper = CompareMeans(DescrStatsW(first), DescrStatsW(second)).tconfint_diff(
        alpha=0.05, usevar="unequal"
    )
    assert difference.value == pytest.approx(float(np.mean(first) - np.mean(second)), rel=STAT_RTOL)
    assert difference.lower == pytest.approx(float(lower), rel=STAT_RTOL)
    assert difference.upper == pytest.approx(float(upper), rel=STAT_RTOL)


def test_students_test_matches_scipy_and_statsmodels() -> None:
    first = np.random.default_rng(6).normal(0.5, 1.0, 25)
    second = np.random.default_rng(7).normal(0.0, 1.5, 40)
    result = _independent_columns(first, second, equal_variances=True)
    (test,) = result.tests

    reference = scipy_stats.ttest_ind(first, second, equal_var=True)
    assert test.name == "Student's t test"
    assert test.method == "Student's t test, two-sided"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == float(first.size + second.size - 2)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)

    difference = result.estimates[0]
    lower, upper = CompareMeans(DescrStatsW(first), DescrStatsW(second)).tconfint_diff(
        alpha=0.05, usevar="pooled"
    )
    assert difference.lower == pytest.approx(float(lower), rel=STAT_RTOL)
    assert difference.upper == pytest.approx(float(upper), rel=STAT_RTOL)
    assert [check.id for check in result.assumptions] == [
        "normality_first",
        "normality_second",
        "equal_variances",
    ]


def test_the_two_sample_hedges_g_follows_the_non_central_t_formula() -> None:
    first = np.random.default_rng(8).normal(1.0, 2.0, 30)
    second = np.random.default_rng(9).normal(0.0, 1.0, 45)
    result = _independent_columns(first, second)

    effect = result.estimates[1]
    expected, lower, upper = _expected_independent_g(first, second, 0.95, equal_variances=False)
    assert effect.value == pytest.approx(expected, rel=STAT_RTOL)
    assert effect.lower == pytest.approx(lower, rel=STAT_RTOL)
    assert effect.upper == pytest.approx(upper, rel=STAT_RTOL)


def test_the_group_column_and_the_two_columns_agree() -> None:
    first = np.random.default_rng(10).normal(0.5, 1.0, 30)
    second = np.random.default_rng(11).normal(0.0, 2.0, 45)

    by_group = _independent_groups(first, second)
    by_columns = _independent_columns(first, second)

    assert by_group.tests[0].statistic == pytest.approx(
        by_columns.tests[0].statistic, rel=STAT_RTOL
    )
    assert by_group.tests[0].df == pytest.approx(by_columns.tests[0].df, rel=STAT_RTOL)
    assert by_group.tests[0].p_value == pytest.approx(by_columns.tests[0].p_value, abs=P_ATOL)


# --------------------------------------------------------------------------- paired


def test_the_paired_test_is_scipys_own() -> None:
    rng = np.random.default_rng(12)
    before = rng.normal(5.0, 1.0, 22)
    after = before + rng.normal(0.4, 0.6, 22)
    result = _paired(before, after)
    (test,) = result.tests

    reference = scipy_stats.ttest_rel(before, after)
    assert test.name == "Paired t test"
    assert test.method == "Paired t test, two-sided"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == 21.0
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)

    difference = result.estimates[0]
    lower, upper = DescrStatsW(before - after).tconfint_mean(alpha=0.05)
    assert difference.value == pytest.approx(float(np.mean(before - after)), rel=STAT_RTOL)
    assert difference.lower == pytest.approx(float(lower), rel=STAT_RTOL)
    assert difference.upper == pytest.approx(float(upper), rel=STAT_RTOL)


def test_the_paired_cohens_dz_follows_the_non_central_t_formula() -> None:
    rng = np.random.default_rng(13)
    before = rng.normal(5.0, 1.0, 25)
    after = before + rng.normal(0.4, 0.6, 25)
    result = _paired(before, after)

    effect = result.estimates[1]
    expected, lower, upper = _expected_paired_dz(before - after, 0.95)
    assert effect.label == "Cohen's dz"
    assert effect.value == pytest.approx(expected, rel=STAT_RTOL)
    assert effect.lower == pytest.approx(lower, rel=STAT_RTOL)
    assert effect.upper == pytest.approx(upper, rel=STAT_RTOL)


def test_the_differences_in_one_column_are_the_paired_test() -> None:
    rng = np.random.default_rng(14)
    before = rng.normal(3.0, 1.0, 18)
    after = before + rng.normal(0.5, 0.7, 18)
    differences = before - after

    paired = _paired(before, after)
    alone = _paired(differences)

    assert paired.tests[0].statistic == pytest.approx(alone.tests[0].statistic, rel=STAT_RTOL)
    assert paired.tests[0].p_value == pytest.approx(alone.tests[0].p_value, abs=P_ATOL)
    assert paired.tests[0].df == alone.tests[0].df
    assert paired.estimates[0].value == pytest.approx(alone.estimates[0].value, rel=STAT_RTOL)


# --------------------------------------------------------------------------- properties


def test_welch_equals_student_when_the_sample_variances_are_equal() -> None:
    rng = np.random.default_rng(15)
    first = rng.normal(0.0, 1.0, 30)
    second = np.roll(first, 7) + 1.5  # the same deviations, shifted: equal variances
    assert float(np.var(first, ddof=1)) == pytest.approx(float(np.var(second, ddof=1)), rel=1e-12)

    welch = _independent_columns(first, second)
    student = _independent_columns(first, second, equal_variances=True)

    assert welch.tests[0].statistic == pytest.approx(student.tests[0].statistic, rel=STAT_RTOL)
    assert welch.tests[0].df == pytest.approx(student.tests[0].df, rel=STAT_RTOL)
    assert welch.tests[0].p_value == pytest.approx(student.tests[0].p_value, abs=P_ATOL)
    assert welch.estimates[0].se == pytest.approx(student.estimates[0].se, rel=STAT_RTOL)
    assert welch.estimates[1].value == pytest.approx(student.estimates[1].value, rel=STAT_RTOL)


def test_adding_a_constant_leaves_the_two_sample_statistics_unchanged() -> None:
    rng = np.random.default_rng(16)
    first = rng.normal(0.0, 1.0, 24)
    second = rng.normal(0.5, 2.0, 31)
    plain = _independent_columns(first, second)
    shifted = _independent_columns(first + 7.5, second + 7.5)

    for before, after in zip(
        plain.tests + plain.estimates, shifted.tests + shifted.estimates, strict=False
    ):
        assert _numbers(before) == pytest.approx(_numbers(after), rel=STAT_RTOL)


def test_adding_a_constant_leaves_the_paired_statistics_unchanged() -> None:
    rng = np.random.default_rng(17)
    before = rng.normal(0.0, 1.0, 20)
    after = before + rng.normal(0.3, 0.5, 20)
    plain = _paired(before, after)
    shifted = _paired(before - 4.0, after - 4.0)

    for left, right in zip(
        plain.tests + plain.estimates, shifted.tests + shifted.estimates, strict=False
    ):
        assert _numbers(left) == pytest.approx(_numbers(right), rel=STAT_RTOL)


def test_scaling_by_a_positive_constant_leaves_the_one_sample_t_and_p_unchanged() -> None:
    values = np.random.default_rng(18).normal(1.0, 2.0, 26)
    plain = _one(values)
    scaled = _one(values * 3.7)

    assert scaled.tests[0].statistic == pytest.approx(plain.tests[0].statistic, rel=STAT_RTOL)
    assert scaled.tests[0].p_value == pytest.approx(plain.tests[0].p_value, abs=P_ATOL)
    assert scaled.estimates[1].value == pytest.approx(plain.estimates[1].value, rel=STAT_RTOL)
    assert scaled.estimates[0].value == pytest.approx(3.7 * plain.estimates[0].value, rel=STAT_RTOL)


def test_scaling_by_a_positive_constant_leaves_the_two_sample_t_and_p_unchanged() -> None:
    rng = np.random.default_rng(19)
    first = rng.normal(0.0, 1.0, 21)
    second = rng.normal(0.5, 1.5, 33)
    plain = _independent_columns(first, second)
    scaled = _independent_columns(first * 2.5, second * 2.5)

    assert scaled.tests[0].statistic == pytest.approx(plain.tests[0].statistic, rel=STAT_RTOL)
    assert scaled.tests[0].p_value == pytest.approx(plain.tests[0].p_value, abs=P_ATOL)
    assert scaled.estimates[1].value == pytest.approx(plain.estimates[1].value, rel=STAT_RTOL)


def test_the_hypothesised_difference_moves_the_test_but_not_the_two_sample_effect() -> None:
    rng = np.random.default_rng(31)
    first = rng.normal(1.0, 1.0, 28)
    second = rng.normal(0.0, 1.5, 34)
    zero = _independent_columns(first, second)
    shifted = _independent_columns(first, second, null=0.75)
    assert shifted.tests[0].statistic != pytest.approx(zero.tests[0].statistic)
    assert shifted.estimates[0].value == pytest.approx(zero.estimates[0].value)
    assert shifted.estimates[1].value == pytest.approx(zero.estimates[1].value, rel=STAT_RTOL)
    assert shifted.estimates[1].lower == pytest.approx(zero.estimates[1].lower, rel=STAT_RTOL)

    before = rng.normal(4.0, 1.0, 22)
    after = before + rng.normal(0.6, 0.7, 22)
    paired_zero = _paired(before, after)
    paired_shifted = _paired(before, after, null=0.75)
    assert paired_shifted.tests[0].statistic != pytest.approx(paired_zero.tests[0].statistic)
    assert paired_shifted.estimates[1].value == pytest.approx(
        paired_zero.estimates[1].value, rel=STAT_RTOL
    )


def _numbers(model: object) -> list[float]:
    """The finite numbers a test or an estimate carries, for a wholesale comparison."""
    values = []
    for name in ("statistic", "df", "p_value", "value", "se", "lower", "upper"):
        found = getattr(model, name, None)
        if isinstance(found, float):
            values.append(found)
    return values


# --------------------------------------------------------------------------- size


def _size_one_sample(sample: NDArray[np.float64]) -> float:
    return _one_sample_statistic(_sample("y", sample), 0.0).p_value


def _size_welch(pair: tuple[NDArray[np.float64], NDArray[np.float64]]) -> float:
    first, second = pair
    return _independent_statistic(
        _sample("first", first), _sample("second", second), 0.0, equal_variances=False
    ).p_value


def _size_student(pair: tuple[NDArray[np.float64], NDArray[np.float64]]) -> float:
    first, second = pair
    return _independent_statistic(
        _sample("first", first), _sample("second", second), 0.0, equal_variances=True
    ).p_value


def _size_paired(pair: tuple[NDArray[np.float64], NDArray[np.float64]]) -> float:
    first, second = pair
    return _paired_statistic(_sample("d", first - second), 0.0).p_value


def test_the_one_sample_test_rejects_at_its_level() -> None:
    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.standard_normal(20)

    share = simulated_size(draw, _size_one_sample, draws=4_000, alpha=0.05)
    assert_size(share)


def test_welch_rejects_at_its_level_with_unequal_variances_and_sizes() -> None:
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return rng.normal(0.0, 1.0, 40), rng.normal(0.0, 3.0, 10)

    share = simulated_size(draw, _size_welch, draws=4_000, alpha=0.05)
    assert_size(share)


def test_students_test_leaves_the_band_on_the_data_where_welch_holds_its_level() -> None:
    # A documented control: pooled variances weight the large-variance group's small sample
    # far too heavily here, so Student's test rejects far more often than its 5 % level.
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return rng.normal(0.0, 1.0, 40), rng.normal(0.0, 3.0, 10)

    share = simulated_size(draw, _size_student, draws=4_000, alpha=0.05)
    assert share > 0.062, f"Student's test measured {share:.1%}, inside the 3.8-6.2 % band"
    with pytest.raises(AssertionError):
        assert_size(share)


def test_the_paired_test_rejects_at_its_level() -> None:
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return rng.standard_normal(20), rng.standard_normal(20)

    share = simulated_size(draw, _size_paired, draws=4_000, alpha=0.05)
    assert_size(share)


# --------------------------------------------------------------------------- the NIST example

US_MPG = (
    18,
    15,
    18,
    16,
    17,
    15,
    14,
    14,
    14,
    15,
    15,
    14,
    15,
    14,
    22,
    18,
    21,
    21,
    10,
    10,
    11,
    9,
    28,
    25,
    19,
    16,
    17,
    19,
    18,
    14,
    14,
    14,
    14,
    12,
    13,
    13,
    18,
    22,
    19,
    18,
    23,
    26,
    25,
    20,
    21,
    13,
    14,
    15,
    14,
    17,
    11,
    13,
    12,
    13,
    15,
    13,
    13,
    14,
    22,
    28,
    13,
    14,
    13,
    14,
    15,
    12,
    13,
    13,
    14,
    13,
    12,
    13,
    18,
    16,
    18,
    18,
    23,
    11,
    12,
    13,
    12,
    18,
    21,
    19,
    21,
    15,
    16,
    15,
    11,
    20,
    21,
    19,
    15,
    26,
    25,
    16,
    16,
    18,
    16,
    13,
    14,
    14,
    14,
    28,
    19,
    18,
    15,
    15,
    16,
    15,
    16,
    14,
    17,
    16,
    15,
    18,
    21,
    20,
    13,
    23,
    20,
    23,
    18,
    19,
    25,
    26,
    18,
    16,
    16,
    15,
    22,
    22,
    24,
    23,
    29,
    25,
    20,
    18,
    19,
    18,
    27,
    13,
    17,
    13,
    13,
    13,
    30,
    26,
    18,
    17,
    16,
    15,
    18,
    21,
    19,
    19,
    16,
    16,
    16,
    16,
    25,
    26,
    31,
    34,
    36,
    20,
    19,
    20,
    19,
    21,
    20,
    25,
    21,
    19,
    21,
    21,
    19,
    18,
    19,
    18,
    18,
    18,
    30,
    31,
    23,
    24,
    22,
    20,
    22,
    20,
    21,
    17,
    18,
    17,
    18,
    17,
    16,
    19,
    19,
    36,
    27,
    23,
    24,
    34,
    35,
    28,
    29,
    27,
    34,
    32,
    28,
    26,
    24,
    19,
    28,
    24,
    27,
    27,
    26,
    24,
    30,
    39,
    35,
    34,
    30,
    22,
    27,
    20,
    18,
    28,
    27,
    34,
    31,
    29,
    27,
    24,
    23,
    38,
    36,
    25,
    38,
    26,
    22,
    36,
    27,
    27,
    32,
    28,
    31,
)
"""The U.S. cars' miles per gallon, as AUTO83B.DAT holds them."""

JAPAN_MPG = (
    24,
    27,
    27,
    25,
    31,
    35,
    24,
    19,
    28,
    23,
    27,
    20,
    22,
    18,
    20,
    31,
    32,
    31,
    32,
    24,
    26,
    29,
    24,
    24,
    33,
    33,
    32,
    28,
    19,
    32,
    34,
    26,
    30,
    22,
    22,
    33,
    39,
    36,
    28,
    27,
    21,
    24,
    30,
    34,
    32,
    38,
    37,
    30,
    31,
    37,
    32,
    47,
    41,
    45,
    34,
    33,
    24,
    32,
    39,
    35,
    32,
    37,
    38,
    34,
    34,
    32,
    33,
    32,
    25,
    24,
    37,
    31,
    36,
    36,
    34,
    38,
    32,
    38,
    32,
)
"""The Japan cars' miles per gallon, the -999 entries of the file deleted."""


def test_the_nist_auto83b_two_sample_example_reproduces_its_tabulated_statistics() -> None:
    result = _independent_groups(US_MPG, JAPAN_MPG, equal_variances=True)

    table = result.tables[0]
    assert table.title == "Summary by group"
    rows = {(row[0], row[1]): row for row in table.rows}
    assert len(rows) == 2
    us = rows[("first", 249)]
    japan = rows[("second", 79)]
    assert us[2] == pytest.approx(20.14458, abs=5e-6)
    assert us[3] == pytest.approx(6.41470, abs=5e-6)
    assert japan[2] == pytest.approx(30.48101, abs=5e-6)
    assert japan[3] == pytest.approx(6.10771, abs=5e-6)

    (test,) = result.tests
    assert test.statistic == pytest.approx(-12.62059, abs=1e-4)
    assert test.df == 326.0

    us_values = np.asarray(US_MPG, dtype=np.float64)
    japan_values = np.asarray(JAPAN_MPG, dtype=np.float64)
    pooled = math.sqrt(
        (
            (us_values.size - 1) * float(np.var(us_values, ddof=1))
            + (japan_values.size - 1) * float(np.var(japan_values, ddof=1))
        )
        / (us_values.size + japan_values.size - 2)
    )
    assert pooled == pytest.approx(6.34260, abs=5e-6)
    assert test.p_value == pytest.approx(
        float(scipy_stats.ttest_ind(us_values, japan_values, equal_var=True).pvalue),
        abs=P_ATOL,
    )


# --------------------------------------------------------------------------- refusals


def test_a_field_that_does_not_fit_the_kind_is_refused() -> None:
    values = _column("y", np.arange(10.0))
    groups = _group("g", ["a"] * 5 + ["b"] * 5)
    cases = [
        {"kind": "one_sample", "a": values, "b": _column("z", np.arange(10.0))},
        {"kind": "one_sample", "a": values, "groups": groups},
        {"kind": "one_sample", "a": values, "equal_variances": True},
        {"kind": "independent", "a": values},
        {"kind": "independent", "a": values, "b": _column("z", np.arange(10.0)), "groups": groups},
        {"kind": "paired", "a": values, "groups": groups},
        {"kind": "paired", "a": values, "equal_variances": True},
    ]
    for options in cases:
        with pytest.raises(ToolError) as refused:
            _run(**options)
        assert refused.value.code == "INVALID_OPTION", options


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        _one(np.arange(10.0), level=level)
    assert refused.value.code == "INVALID_OPTION"


@pytest.mark.parametrize("null", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_hypothesised_value_is_refused(null: float) -> None:
    with pytest.raises(ToolError) as refused:
        _one(np.arange(10.0), null=null)
    assert refused.value.code == "INVALID_OPTION"


def test_a_one_sample_test_needs_three_values() -> None:
    with pytest.raises(ToolError) as refused:
        _one([1.0, 2.0])
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "3" in refused.value.message


def test_a_constant_one_sample_column_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _one([4.0, 4.0, 4.0, 4.0])
    assert refused.value.code == "CONSTANT_COLUMN"


def test_a_group_column_needs_exactly_two_levels() -> None:
    values = np.arange(12.0)
    for labels in (["a"] * 12, ["a"] * 4 + ["b"] * 4 + ["c"] * 4):
        with pytest.raises(ToolError) as refused:
            _run(
                kind="independent",
                a=_column("y", values),
                groups=_group("g", labels),
            )
        assert refused.value.code == "LEVELS_REQUIRED"


def test_a_group_of_one_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _independent_groups([1.0, 2.0, 3.0], [4.0])
    assert refused.value.code == "GROUP_OF_ONE"
    assert "second" in refused.value.message


def test_constant_differences_are_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _paired([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert refused.value.code == "CONSTANT_COLUMN"


def test_a_paired_test_needs_two_complete_pairs() -> None:
    with pytest.raises(ToolError) as refused:
        _paired([1.0], [2.0])
    assert refused.value.code == "TOO_FEW_ROWS"


# --------------------------------------------------------------------------- excluded rows


def test_missing_and_non_finite_values_are_listed_not_dropped() -> None:
    result = _one([1.0, float("nan"), 3.0, float("inf"), 5.0, 7.0, float("-inf")])

    assert result.counts == Counts(n_offered=7, n_used=4, n_excluded=3)
    assert result.excluded == (
        ExcludedRow(index=1, reason="missing in y"),
        ExcludedRow(index=3, reason="not a finite number in y"),
        ExcludedRow(index=6, reason="not a finite number in y"),
    )
    assert result.excluded_unlisted == 0


def test_a_row_missing_in_one_column_of_a_pair_is_left_out_of_both() -> None:
    before = _column("before", [1.0, 2.0, 3.0, float("nan"), 5.0])
    after = _column("after", [2.0, 2.0, float("nan"), 4.0, 6.0])
    result = run_ttest(TTestRequest(kind="paired", a=before, b=after))

    assert result.counts == Counts(n_offered=5, n_used=3, n_excluded=2)
    assert result.excluded == (
        ExcludedRow(index=2, reason="missing in after"),
        ExcludedRow(index=3, reason="missing in before"),
    )
    assert result.tests[0].df == 2.0


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    result = _one([1.0, 2.0, 3.0] + [float("nan")] * 255)

    assert result.counts.n_used == 3 and result.counts.n_excluded == 255
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 55


# --------------------------------------------------------------------------- the result's words


def test_every_test_carries_the_difference_with_its_interval() -> None:
    result = _one(np.random.default_rng(20).normal(1.0, 1.0, 25))
    difference = result.estimates[0]
    (test,) = result.tests

    assert test.effect == difference
    assert test.effect is not None
    assert test.effect.lower is not None and test.effect.upper is not None
    assert test.p_value_text == p_text(test.p_value)


def test_assumption_verdicts_follow_the_p_thresholds() -> None:
    normal = _one(np.random.default_rng(21).normal(0.0, 1.0, 60))
    (normality,) = normal.assumptions
    assert normality.verdict is Verdict.PASS
    assert normality.statistic is not None and normality.statistic.startswith("W = ")
    assert normality.threshold == "p >= 0.05 pass; 0.001 <= p < 0.05 caution; p < 0.001 fail"

    skewed = _one(np.random.default_rng(22).exponential(1.0, 80))
    (normality,) = skewed.assumptions
    assert normality.verdict is Verdict.FAIL


def test_students_test_checks_equal_variances_and_welchs_does_not() -> None:
    rng = np.random.default_rng(23)
    narrow = rng.normal(0.0, 0.1, 30)
    wide = rng.normal(0.0, 5.0, 30)

    welch = _independent_columns(narrow, wide)
    assert [check.id for check in welch.assumptions] == ["normality_first", "normality_second"]

    student = _independent_columns(narrow, wide, equal_variances=True)
    (levene,) = [check for check in student.assumptions if check.id == "equal_variances"]
    assert levene.verdict is Verdict.FAIL
    assert "Levene's test, centred on the median" in levene.evidence


def test_the_paired_test_checks_that_the_pairs_are_complete() -> None:
    complete = _paired([1.0, 2.0, 4.0, 5.0], [2.0, 4.0, 5.0, 8.0])
    (completeness,) = [check for check in complete.assumptions if check.id == "complete_pairs"]
    assert completeness.verdict is Verdict.PASS

    incomplete = run_ttest(
        TTestRequest(
            kind="paired",
            a=_column("before", [1.0, 2.0, 3.0, float("nan"), 5.0]),
            b=_column("after", [2.0, 4.0, 5.0, 7.0, 9.0]),
        )
    )
    (completeness,) = [check for check in incomplete.assumptions if check.id == "complete_pairs"]
    assert completeness.verdict is Verdict.CAUTION
    assert "1" in completeness.title

    differences = _paired([1.0, 2.0, 3.0])
    (completeness,) = [check for check in differences.assumptions if check.id == "complete_pairs"]
    assert completeness.verdict is Verdict.NOT_APPLICABLE


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    results = [
        _one(np.random.default_rng(24).normal(0.0, 1.0, 30)),
        _independent_groups(
            np.random.default_rng(25).normal(0.0, 1.0, 30),
            np.random.default_rng(26).normal(1.0, 2.0, 40),
        ),
        _paired(
            np.random.default_rng(27).normal(0.0, 1.0, 25),
            np.random.default_rng(28).normal(0.5, 1.5, 25),
        ),
    ]
    for result in results:
        texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
        for check in result.assumptions:
            texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
        for test in result.tests:
            texts += [test.name, test.method, test.null_hypothesis]
        for text in texts:
            assert not claims_proof(text), text
    assert "Welch" in results[1].methods_fragment
    assert "Hedges" in results[1].methods_fragment
    assert "Cohen's dz" in results[2].methods_fragment


def test_the_result_carries_a_summary_by_group_and_one_box_plot() -> None:
    result = _independent_groups(
        np.random.default_rng(29).normal(0.0, 1.0, 20),
        np.random.default_rng(30).normal(1.0, 2.0, 25),
    )

    (table,) = result.tables
    assert table.title == "Summary by group"
    assert [(row[0], row[1]) for row in table.rows] == [("first", 20), ("second", 25)]
    (plot,) = result.plots
    assert plot.kind == "box"
    assert [box.label for box in plot.boxes] == ["first", "second"]


# --------------------------------------------------------------------------- the registry


def test_the_registry_lists_and_runs_the_tool() -> None:
    assert "curvestudio.tools.stats.means" in TOOL_MODULES

    info = TOOLS.info("ttest")
    assert (info.name, info.family, info.title, info.needs_data) == (
        "ttest",
        "means",
        "t tests",
        True,
    )
    assert [(field.name, field.kind, field.label, field.required) for field in info.fields] == [
        ("kind", "option", "Kind of test", True),
        ("a", "column", "First column", True),
        ("b", "column", "Second column", False),
        ("groups", "group", "Groups", False),
        ("null", "option", "Hypothesised value", False),
        ("equal_variances", "option", "Equal variances", False),
        ("level", "option", "Level", False),
    ]
    assert all(field.description for field in info.fields)
    assert "ttest" in [entry.name for entry in catalogue()]

    values = _column("y", np.arange(20.0))
    groups = _group("g", ["a"] * 10 + ["b"] * 10)
    request = TOOLS.build_request(
        "ttest",
        {"kind": "independent", "a": "y", "groups": "g", "level": 0.9},
        column=lambda name: values,
        group=lambda name: groups,
    )
    assert isinstance(request, TTestRequest)
    assert request.kind == "independent" and request.level == 0.9
    assert TOOLS.run("ttest", request).tool == "ttest"


# --------------------------------------------------------------------------- the formulas

_ALPHA = 0.05


def _ncp_interval(statistic: float, df: float, level: float) -> tuple[float, float]:
    """The interval for the non-centrality parameter, computed here from its definition."""

    def boundary(target: float) -> float:
        def excess(noncentrality: float) -> float:
            return float(scipy_stats.nct.cdf(statistic, df, noncentrality)) - target

        low = statistic - 1.0
        step = 1.0
        while excess(low) <= 0.0:
            low -= step
            step *= 2.0
        high = statistic + 1.0
        step = 1.0
        while excess(high) >= 0.0:
            high += step
            step *= 2.0
        return float(optimize.brentq(excess, low, high))

    return boundary(1.0 - (1.0 - level) / 2.0), boundary((1.0 - level) / 2.0)


def _hedges(statistic: float, df: float, scale: float, level: float) -> tuple[float, float, float]:
    """Hedges' g and its interval from the non-central t inversion."""
    lower, upper = _ncp_interval(statistic, df, level)
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return (
        correction * statistic * scale,
        correction * lower * scale,
        correction * upper * scale,
    )


def _expected_one_sample_g(
    values: NDArray[np.float64], null: float, level: float
) -> tuple[float, float, float]:
    n = values.size
    sd = float(np.std(values, ddof=1))
    statistic = (float(np.mean(values)) - null) / (sd / math.sqrt(n))
    return _hedges(statistic, float(n - 1), 1.0 / math.sqrt(n), level)


def _expected_independent_g(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    level: float,
    *,
    equal_variances: bool,
) -> tuple[float, float, float]:
    n1, n2 = first.size, second.size
    first_variance = float(np.var(first, ddof=1)) / n1
    second_variance = float(np.var(second, ddof=1)) / n2
    pooled = math.sqrt(
        ((n1 - 1) * float(np.var(first, ddof=1)) + (n2 - 1) * float(np.var(second, ddof=1)))
        / (n1 + n2 - 2)
    )
    if equal_variances:
        degrees = float(n1 + n2 - 2)
        se = pooled * math.sqrt(1.0 / n1 + 1.0 / n2)
    else:
        degrees = (first_variance + second_variance) ** 2 / (
            first_variance**2 / (n1 - 1) + second_variance**2 / (n2 - 1)
        )
        se = math.sqrt(first_variance + second_variance)
    statistic = (float(np.mean(first) - np.mean(second))) / se
    return _hedges(statistic, degrees, se / pooled, level)


def _expected_paired_dz(
    differences: NDArray[np.float64], level: float
) -> tuple[float, float, float]:
    n = differences.size
    sd = float(np.std(differences, ddof=1))
    statistic = float(np.mean(differences)) / (sd / math.sqrt(n))
    lower, upper = _ncp_interval(statistic, float(n - 1), level)
    scale = 1.0 / math.sqrt(n)
    return statistic * scale, lower * scale, upper * scale
