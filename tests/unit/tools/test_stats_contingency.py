"""The contingency table and McNemar tools: their tests, their references and their size.

The reference numbers come from SciPy and statsmodels, computed here rather than typed:
``chi2_contingency`` for the Pearson statistic (with Yates' correction for a 2 x 2 table),
``fisher_exact`` for the exact test, ``scipy.stats.contingency.association`` for Cramér's V,
``statsmodels.stats.contingency_tables.Table2x2`` for the odds ratio and the risk ratio with
their intervals, and ``statsmodels.stats.contingency_tables.mcnemar`` for both forms of
McNemar's test.

Two worked examples were fetched and read on 2026-11-24:

- NIST/SEMATECH e-Handbook of Statistical Methods, section 7.4.5, "How can we compare the
  results of classifying according to several categories?", at
  ``https://itl.nist.gov/div898/handbook/prc/section4/prc45.htm``. The page's wafer-defect
  table gives 15, 21, 45 and 13 in shift 1, 26, 31, 34 and 5 in shift 2, and 33, 17, 49 and
  20 in shift 3, with 94, 96 and 119 as the row totals; it tabulates the chi-square statistic
  as 19.18 on 6 degrees of freedom with p = 0.00387.
- R's ``fisher.test`` documentation, ``https://stat.ethz.ch/R-manual/R-devel/library/stats/
  html/fisher.test.html``, fetched on 2026-11-24. Its Fisher Tea Drinker example gives the
  2 x 2 table ``matrix(c(3, 1, 1, 3), nrow = 2)`` and the one-sided p = 0.2429. The two-sided
  p that Fisher's rule gives for that table is the sum of the probabilities of the tables no
  more likely than the observed one, 34/70 = 0.485714..., which is the number the test below
  reproduces.

Newcombe's method 10 for the difference of paired proportions is described in the PubMed
abstract of Newcombe R.G., Statistics in Medicine 17:2635-2650 (1998), PMID 9839354, read on
2026-11-24, as the method based on the score interval for a single proportion; the closed-form
combination used here is the MOVER interval of Zou G.Y. and Donner A., Statistics in Medicine
27:1693-1702 (2008), PMID 17960596, whose abstract was read on 2026-11-24. No tabulated
numerical example of method 10 was found in a fetched source, so the interval is checked
against the formula written out independently in this file and against its defining
properties.

The sizes are measured, not assumed. The chi-square test is drawn from two independent binary
columns of 400 rows; Fisher's exact test from two independent random permutations of columns
whose margins are fixed (the exact test conditions on those margins), and McNemar's exact test
from 17 discordant pairs whose direction is a fair coin (the exact test conditions on the
discordant count). Each is measured at seed 0 over 4000 draws.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats
from scipy.stats.contingency import association
from statsmodels.stats.contingency_tables import Table2x2, mcnemar

from curvestudio.data import GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    Estimate,
    ExcludedRow,
    ToolResult,
    Verdict,
    assert_size,
    build_request,
    p_text,
    run_tool,
    simulated_size,
)
from curvestudio.tools.stats.contingency import (
    ContingencyRequest,
    McNemarRequest,
    _cramers_v,
    _independence_test,
    _mcnemar_test,
    _newcombe_interval,
    _two_by_two_estimates,
    run_contingency,
    run_mcnemar,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, catalogue

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""
P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""


def _group(name: str, labels: Sequence[str], rows: Sequence[int] | None = None) -> GroupColumn:
    positions = list(range(len(labels))) if rows is None else list(rows)
    return GroupColumn(
        name=name,
        labels=np.asarray(labels, dtype=object),
        rows=np.asarray(positions, dtype=np.intp),
    )


def _pairs(
    grid: np.ndarray,
    row_levels: Sequence[str],
    column_levels: Sequence[str],
    *,
    row_name: str = "rows",
    column_name: str = "columns",
) -> tuple[GroupColumn, GroupColumn]:
    """Two label columns whose counts are ``grid``, in row-major order."""
    rows: list[str] = []
    columns: list[str] = []
    for index, row in enumerate(row_levels):
        for other, column in enumerate(column_levels):
            count = int(grid[index][other])
            rows.extend([row] * count)
            columns.extend([column] * count)
    return _group(row_name, rows), _group(column_name, columns)


def _contingency(
    grid: np.ndarray,
    row_levels: Sequence[str] | None = None,
    column_levels: Sequence[str] | None = None,
    *,
    level: float = 0.95,
) -> ToolResult:
    rows = row_levels if row_levels is not None else [f"r{index}" for index in range(grid.shape[0])]
    columns = (
        column_levels
        if column_levels is not None
        else [f"c{index}" for index in range(grid.shape[1])]
    )
    row_column, column_column = _pairs(grid, rows, columns)
    return run_contingency(ContingencyRequest(rows=row_column, columns=column_column, level=level))


def _mcnemar(
    grid: np.ndarray,
    before_levels: Sequence[str] = ("before0", "before1"),
    after_levels: Sequence[str] = ("after0", "after1"),
    *,
    level: float = 0.95,
) -> ToolResult:
    before, after = _pairs(
        grid, before_levels, after_levels, row_name="before", column_name="after"
    )
    return run_mcnemar(McNemarRequest(before=before, after=after, level=level))


def _estimate(result: ToolResult, prefix: str) -> Estimate:
    for found in result.estimates:
        if found.label.startswith(prefix):
            return found
    raise AssertionError(f"no estimate labelled {prefix!r}: {[e.label for e in result.estimates]}")


def _table(result: ToolResult, title: str):
    return next(found for found in result.tables if found.title == title)


def _row(table, key: str, value: object) -> dict[str, object]:
    keys = [column.key for column in table.columns]
    for row in table.rows:
        cells = dict(zip(keys, row, strict=True))
        if cells[key] == value:
            return cells
    raise AssertionError(f"no row with {key} = {value!r}: {table.rows}")


def _wilson_reference(count: int, total: int, quantile: float) -> tuple[float, float]:
    proportion = count / total
    squared = quantile * quantile
    centre = (count + squared / 2.0) / (total + squared)
    half = quantile * math.sqrt(total * proportion * (1.0 - proportion) + squared / 4.0)
    half /= total + squared
    return centre - half, centre + half


def _newcombe_reference(
    first: int, second: int, third: int, fourth: int, level: float
) -> tuple[float, float, float]:
    """Newcombe's method 10 written out from the Wilson limits and the MOVER combination."""
    total = first + second + third + fourth
    quantile = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0))
    after = (first + third) / total
    before = (first + second) / total
    after_low, after_high = _wilson_reference(first + third, total, quantile)
    before_low, before_high = _wilson_reference(first + second, total, quantile)
    denominator = math.sqrt(
        (first + second) * (third + fourth) * (first + third) * (second + fourth)
    )
    correlation = 0.0 if denominator == 0.0 else (first * fourth - second * third) / denominator
    difference = after - before
    lower_square = (
        (after - after_low) ** 2
        + (before_high - before) ** 2
        - 2.0 * correlation * (after - after_low) * (before_high - before)
    )
    upper_square = (
        (after_high - after) ** 2
        + (before - before_low) ** 2
        - 2.0 * correlation * (after_high - after) * (before - before_low)
    )
    return (
        difference,
        difference - math.sqrt(max(lower_square, 0.0)),
        difference + math.sqrt(max(upper_square, 0.0)),
    )


# --------------------------------------------------------------------------- the references


def test_the_chi_square_test_is_scipys_with_yates_correction_for_a_two_by_two() -> None:
    grid = np.array([[20, 10], [5, 15]])

    result = _contingency(grid)
    test = result.tests[0]
    reference = scipy_stats.chi2_contingency(grid, correction=True)

    assert test.name == "Pearson chi-square"
    assert (
        test.method == "Pearson chi-square test of independence with Yates' continuity correction"
    )
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.df == pytest.approx(float(reference.dof))
    assert test.p_value_text == p_text(test.p_value)


def test_a_larger_table_uses_the_uncorrected_chi_square() -> None:
    grid = np.array([[12, 7, 3], [5, 15, 8]])

    result = _contingency(grid)
    test = result.tests[0]
    reference = scipy_stats.chi2_contingency(grid, correction=False)

    assert test.method == "Pearson chi-square test of independence"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=1e-4)
    assert test.df == pytest.approx(2.0)


def test_the_cramers_v_is_scipys_association() -> None:
    grid = np.array([[12, 7, 3], [5, 15, 8], [4, 6, 20]])

    result = _contingency(grid)
    cramers = _estimate(result, "Cramér's V")
    reference = float(association(grid, method="cramer"))

    assert cramers.value == pytest.approx(reference, rel=STAT_RTOL)
    assert result.tests[0].effect is not None
    assert result.tests[0].effect.value == pytest.approx(cramers.value, rel=STAT_RTOL)


def test_the_odds_ratio_and_risk_ratio_are_statsmodels_table_two_by_two() -> None:
    grid = np.array([[20, 10], [5, 15]])
    table = Table2x2(grid, shift_zeros=False)

    result = _contingency(grid, row_levels=("yes", "no"), column_levels=("yes", "no"))
    odds = _estimate(result, "Odds ratio")
    risk = _estimate(result, "Risk ratio")
    odds_lower, odds_upper = (float(value) for value in table.oddsratio_confint(alpha=0.05))
    risk_lower, risk_upper = (float(value) for value in table.riskratio_confint(alpha=0.05))

    assert odds.value == pytest.approx(float(table.oddsratio), rel=STAT_RTOL)
    assert odds.lower == pytest.approx(odds_lower, rel=STAT_RTOL)
    assert odds.upper == pytest.approx(odds_upper, rel=STAT_RTOL)
    assert risk.value == pytest.approx(float(table.riskratio), rel=STAT_RTOL)
    assert risk.lower == pytest.approx(risk_lower, rel=STAT_RTOL)
    assert risk.upper == pytest.approx(risk_upper, rel=STAT_RTOL)
    assert "reference cell: yes and yes" in odds.label
    assert "reference cell: yes and yes" in risk.label


def test_the_tea_tasting_table_reproduces_fishers_two_sided_p() -> None:
    """The table and the one-sided p are from R's fisher.test documentation, fetched 2026-11-24.

    The page gives ``matrix(c(3, 1, 1, 3), nrow = 2)`` and the one-sided p = 0.2429. Every
    table no more likely than the observed one holds 34 of the 70 equally likely arrangements,
    so the two-sided p is 34/70.
    """
    grid = np.array([[3, 1], [1, 3]])

    result = _contingency(grid, row_levels=("milk", "tea"), column_levels=("milk", "tea"))
    test = result.tests[0]
    reference = scipy_stats.fisher_exact(grid)

    assert test.name == "Fisher's exact test"
    assert test.method == "Fisher's exact test, two-sided"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=1e-12)
    assert test.p_value == pytest.approx(34 / 70, rel=1e-12)
    assert test.p_value == pytest.approx(0.4857, abs=1e-4)
    assert result.assumptions[1].verdict is Verdict.NOT_APPLICABLE
    assert any("Fisher's exact test is given" in warning for warning in result.warnings)


def test_a_large_table_with_a_small_expected_count_warns_and_keeps_the_chi_square() -> None:
    grid = np.array([[10, 0, 0], [0, 10, 1]])

    result = _contingency(grid)

    assert result.tests[0].name == "Pearson chi-square"
    assert any("in doubt" in warning for warning in result.warnings)
    assert result.assumptions[1].verdict is Verdict.CAUTION


def test_the_nist_wafer_defect_table_reproduces_the_tabulated_chi_square() -> None:
    """The counts, the statistic, the degrees of freedom and the p are the NIST page's own.

    NIST/SEMATECH e-Handbook, section 7.4.5, fetched 2026-11-24: the table holds 15, 21, 45
    and 13; 26, 31, 34 and 5; 33, 17, 49 and 20, with the statistic 19.18 on 6 degrees of
    freedom and p = 0.00387.
    """
    grid = np.array([[15, 21, 45, 13], [26, 31, 34, 5], [33, 17, 49, 20]])

    result = _contingency(grid, row_levels=("1", "2", "3"), column_levels=("A", "B", "C", "D"))
    test = result.tests[0]
    reference = scipy_stats.chi2_contingency(grid, correction=False)

    assert test.statistic == pytest.approx(19.18, abs=5e-3)
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == pytest.approx(6.0)
    assert test.p_value == pytest.approx(0.00387, abs=1e-5)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert _table(result, "Counts").rows[0] == ("1", 15, 21, 45, 13, 94)
    assert _table(result, "Counts").rows[3] == ("Total", 74, 69, 128, 38, 309)


def test_mcnemars_exact_test_is_statsmodels_own() -> None:
    grid = np.array([[10, 8], [3, 20]])

    result = _mcnemar(grid)
    test = result.tests[0]
    reference = mcnemar(grid, exact=True, correction=True)

    assert test.method == "McNemar's test, exact binomial"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_mcnemars_chi_square_form_is_statsmodels_own() -> None:
    grid = np.array([[10, 20], [8, 30]])

    result = _mcnemar(grid)
    test = result.tests[0]
    reference = mcnemar(grid, exact=False, correction=True)

    assert test.method == "McNemar's test, chi-square with continuity correction"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.df == pytest.approx(1.0)


def test_mcnemar_switches_to_chi_square_beyond_twenty_five_discordant_pairs() -> None:
    at_twenty_five = np.array([[10, 13], [12, 15]])
    at_twenty_six = np.array([[10, 14], [12, 15]])

    assert _mcnemar(at_twenty_five).tests[0].method == "McNemar's test, exact binomial"
    assert (
        _mcnemar(at_twenty_six).tests[0].method
        == "McNemar's test, chi-square with continuity correction"
    )


def test_the_difference_in_marginal_proportions_matches_the_formula_written_out() -> None:
    grid = np.array([[10, 8], [3, 20]])

    result = _mcnemar(grid)
    difference = result.estimates[0]
    first, second, third, fourth = (int(value) for value in grid.reshape(-1))
    value, lower, upper = _newcombe_reference(first, second, third, fourth, 0.95)

    assert difference.value == pytest.approx(value, rel=STAT_RTOL)
    assert difference.lower == pytest.approx(lower, rel=STAT_RTOL)
    assert difference.upper == pytest.approx(upper, rel=STAT_RTOL)
    assert "after after0 minus before before0" in difference.label
    assert result.tests[0].effect is difference


# --------------------------------------------------------------------------- size


def _binary_grid(rng: np.random.Generator, total: int, probability: float) -> np.ndarray:
    first = rng.binomial(1, probability, total)
    second = rng.binomial(1, probability, total)
    grid = np.zeros((2, 2), dtype=np.int64)
    np.add.at(grid, (first, second), 1)
    return grid


def _fixed_margin_grid(
    rng: np.random.Generator, total: int, first_ones: int, second_ones: int
) -> np.ndarray:
    first = np.zeros(total, dtype=np.intp)
    first[:first_ones] = 1
    rng.shuffle(first)
    second = np.zeros(total, dtype=np.intp)
    second[:second_ones] = 1
    rng.shuffle(second)
    grid = np.zeros((2, 2), dtype=np.int64)
    np.add.at(grid, (first, second), 1)
    return grid


_DUMMY = Estimate(label="difference", value=0.0)
"""The effect a McNemar test stores but does not need for its p-value."""


def _chi_square_p(grid: np.ndarray) -> float | None:
    test = _independence_test(grid)
    assert test.name == "Pearson chi-square", test.name
    return test.p_value


def _fisher_p(grid: np.ndarray) -> float | None:
    test = _independence_test(grid)
    assert test.name == "Fisher's exact test", test.name
    return test.p_value


def _mcnemar_exact_p(grid: np.ndarray) -> float | None:
    test = _mcnemar_test(grid, _DUMMY)
    assert test.method == "McNemar's test, exact binomial", test.method
    return test.p_value


def test_the_chi_square_with_yates_rejects_at_its_level() -> None:
    """Two independent binary columns of 400 rows; the expected counts are far above 5."""
    share = simulated_size(
        lambda rng: _binary_grid(rng, 400, 0.5), _chi_square_p, draws=4_000, seed=0
    )

    assert_size(share)


def test_fishers_exact_test_rejects_at_its_level() -> None:
    """Two independent permutations of fixed margins, which is the null the exact test takes.

    The first column holds 9 ones and the second 16 ones in 45 rows; each is shuffled on its
    own, so the rows stay independent and the margins stay fixed. The expected counts fall
    below 5, so the tool gives Fisher's exact test on every draw.
    """
    share = simulated_size(
        lambda rng: _fixed_margin_grid(rng, 45, 9, 16), _fisher_p, draws=4_000, seed=0
    )

    assert_size(share, exact=True)


def test_mcnemars_exact_test_rejects_at_its_level() -> None:
    """Seventeen discordant pairs whose direction is a fair coin, the null the exact test takes.

    Sixteen rows hold the first level in both columns and seventeen hold the second level in
    both, so both columns keep two levels on every draw; the remaining seventeen rows are
    discordant, and a fair coin decides which way.
    """

    def draw(rng: np.random.Generator) -> np.ndarray:
        before = np.zeros(50, dtype=np.intp)
        after = np.zeros(50, dtype=np.intp)
        before[16:33] = 1
        after[16:33] = 1
        direction = rng.binomial(1, 0.5, 17)
        before[33:] = direction
        after[33:] = 1 - direction
        grid = np.zeros((2, 2), dtype=np.int64)
        np.add.at(grid, (before, after), 1)
        return grid

    share = simulated_size(draw, _mcnemar_exact_p, draws=4_000, seed=0)

    assert_size(share, exact=True)


# --------------------------------------------------------------------------- the properties


def test_transposing_the_table_leaves_the_statistic_and_cramers_v() -> None:
    grid = np.array([[12, 7, 3], [5, 15, 8], [4, 6, 20]])

    forward = _independence_test(grid)
    backward = _independence_test(grid.T)

    assert backward.statistic == pytest.approx(forward.statistic, rel=STAT_RTOL)
    assert backward.effect is not None and forward.effect is not None
    assert backward.effect.value == pytest.approx(forward.effect.value, rel=STAT_RTOL)
    assert backward.p_value == pytest.approx(forward.p_value, rel=STAT_RTOL)


def test_doubling_every_count_doubles_chi_square_and_leaves_cramers_v() -> None:
    grid = np.array([[12, 7, 3], [5, 15, 8], [4, 6, 20]])

    plain = _independence_test(grid)
    doubled = _independence_test(grid * 2)

    assert plain.statistic is not None and doubled.statistic is not None
    assert doubled.statistic == pytest.approx(2.0 * plain.statistic, rel=STAT_RTOL)
    assert plain.effect is not None and doubled.effect is not None
    assert doubled.effect.value == pytest.approx(plain.effect.value, rel=STAT_RTOL)
    assert _cramers_v(grid * 2) == pytest.approx(_cramers_v(grid), rel=STAT_RTOL)


def test_doubling_every_count_leaves_the_odds_ratio() -> None:
    grid = np.array([[20, 10], [5, 15]])
    doubled = grid * 2

    plain, _ = _two_by_two_estimates(grid, ("yes", "no"), ("yes", "no"), 0.95)
    twice, _ = _two_by_two_estimates(doubled, ("yes", "no"), ("yes", "no"), 0.95)
    plain_odds = next(found for found in plain if found.label.startswith("Odds ratio"))
    twice_odds = next(found for found in twice if found.label.startswith("Odds ratio"))

    assert twice_odds.value == pytest.approx(plain_odds.value, rel=STAT_RTOL)


def test_the_newcombe_interval_is_symmetric_when_the_discordant_pairs_are_equal() -> None:
    lower, upper = _newcombe_interval(10, 7, 7, 15, 39, 0.95)

    assert lower == pytest.approx(-upper, rel=STAT_RTOL)
    assert lower <= 0.0 <= upper


def test_the_newcombe_interval_collapses_when_every_pair_agrees() -> None:
    lower, upper = _newcombe_interval(25, 0, 0, 25, 50, 0.95)

    assert lower == pytest.approx(0.0, abs=1e-12)
    assert upper == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- the contract


def test_the_contingency_result_is_the_contract_with_four_tables() -> None:
    result = _contingency(
        np.array([[15, 21, 45, 13], [26, 31, 34, 5], [33, 17, 49, 20]]),
        row_levels=("1", "2", "3"),
        column_levels=("A", "B", "C", "D"),
    )

    assert type(result) is ToolResult
    assert result.tool == "contingency"
    assert result.title == "Contingency table of rows by columns"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=309, n_used=309, n_excluded=0)
    assert result.excluded == () and result.excluded_unlisted == 0
    assert [table.title for table in result.tables] == [
        "Counts",
        "Row percentages",
        "Column percentages",
        "Of all",
    ]
    assert [column.label for column in result.tables[0].columns] == [
        "Row",
        "A",
        "B",
        "C",
        "D",
        "Total",
    ]
    assert len(result.tests) == 1
    assert result.tests[0].effect is not None
    assert result.estimates[0].label == "Cramér's V"


def test_the_mcnemar_result_is_the_contract_with_one_table() -> None:
    result = _mcnemar(np.array([[10, 8], [3, 20]]))

    assert result.tool == "mcnemar"
    assert result.title == "McNemar's test of before and after"
    assert result.counts == Counts(n_offered=41, n_used=41, n_excluded=0)
    assert [table.title for table in result.tables] == ["Before by after"]
    table = result.tables[0]
    assert [column.label for column in table.columns] == ["Row", "after0", "after1", "Total"]
    assert table.rows == (
        ("before0", 10, 8, 18),
        ("before1", 3, 20, 23),
        ("Total", 13, 28, 41),
    )


def test_the_percentages_are_per_cent_and_their_totals_read_consistently() -> None:
    grid = np.array([[20, 10], [5, 15]])

    result = _contingency(grid)
    row_percentages = _table(result, "Row percentages")
    column_percentages = _table(result, "Column percentages")
    of_all = _table(result, "Of all")
    yes = _row(row_percentages, "row", "r0")

    assert yes["c0"] == pytest.approx(100.0 * 20 / 30)
    assert yes["total"] == pytest.approx(100.0)
    assert _row(row_percentages, "row", "Total")["c0"] == pytest.approx(100.0 * 25 / 50)
    assert _row(column_percentages, "row", "Total")["c0"] == pytest.approx(100.0)
    assert _row(column_percentages, "row", "r0")["total"] == pytest.approx(100.0 * 30 / 50)
    assert _row(of_all, "row", "r0")["c0"] == pytest.approx(100.0 * 20 / 50)
    assert _row(of_all, "row", "Total")["total"] == pytest.approx(100.0)


def test_the_assumptions_name_the_expected_counts_and_independence() -> None:
    small = _contingency(np.array([[3, 1], [1, 3]]))
    large = _contingency(np.array([[20, 10], [5, 15]]))

    assert [check.id for check in small.assumptions] == ["independence", "expected_counts"]
    assert small.assumptions[0].verdict is Verdict.NOT_APPLICABLE
    assert small.assumptions[1].verdict is Verdict.NOT_APPLICABLE
    assert large.assumptions[1].verdict is Verdict.PASS
    assert "smallest expected count" in large.assumptions[1].evidence


def test_the_mcnemar_assumption_names_the_independence_of_the_pairs() -> None:
    result = _mcnemar(np.array([[10, 8], [3, 20]]))

    assert [check.id for check in result.assumptions] == ["independence"]
    assert result.assumptions[0].verdict is Verdict.NOT_APPLICABLE


def test_a_row_missing_a_label_is_left_out_and_listed() -> None:
    rows = _group("rows", ["a", "b", "c"], rows=(0, 1, 2))
    columns = _group("columns", ["x", "y"], rows=(0, 1))

    result = run_contingency(ContingencyRequest(rows=rows, columns=columns))

    assert result.counts == Counts(n_offered=3, n_used=2, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=2, reason="no group label in columns"),)
    assert result.excluded_unlisted == 0


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    labels = ["a" if index % 2 == 0 else "b" for index in range(208)]
    rows = _group("rows", labels, rows=range(208))
    columns = _group("columns", ["x", "y", "x"], rows=(0, 1, 2))

    result = run_contingency(ContingencyRequest(rows=rows, columns=columns))

    assert result.counts.n_excluded == 205
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 5


def test_an_empty_level_is_dropped_with_a_warning() -> None:
    rows = _group("rows", ["a", "b", "c"], rows=(0, 1, 2))
    columns = _group("columns", ["x", "y"], rows=(0, 1))

    result = run_contingency(ContingencyRequest(rows=rows, columns=columns))

    assert any(
        "The row level 'c' holds no paired observation" in warning for warning in result.warnings
    )
    assert result.tables[0].rows[0][0] == "a"
    assert result.tables[0].rows[1][0] == "b"


def test_an_odds_ratio_with_a_zero_cell_is_left_out_with_a_warning() -> None:
    grid = np.array([[10, 0], [5, 15]])

    result = _contingency(grid)

    assert all(not found.label.startswith("Odds ratio") for found in result.estimates)
    assert any("odds ratio is undefined" in warning for warning in result.warnings)
    assert any("holds 0" in warning for warning in result.warnings)


def test_a_risk_ratio_with_a_zero_reference_cell_is_left_out_with_a_warning() -> None:
    grid = np.array([[10, 5], [0, 15]])

    result = _contingency(grid)

    assert all(not found.label.startswith("Risk ratio") for found in result.estimates)
    assert any("risk ratio is undefined" in warning for warning in result.warnings)


def test_mcnemar_with_no_discordant_pairs_gives_no_p_value() -> None:
    result = _mcnemar(np.array([[10, 0], [0, 20]]))
    test = result.tests[0]

    assert test.statistic is None and test.p_value is None
    assert test.p_value_text == "not computed"
    assert test.reason is not None and "no discordant pairs" in test.reason
    assert result.estimates[0].value == pytest.approx(0.0)
    assert "could not be run" in result.interpretation


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    results = (
        _contingency(
            np.array([[20, 10], [5, 15]]), row_levels=("yes", "no"), column_levels=("yes", "no")
        ),
        _contingency(np.array([[3, 1], [1, 3]])),
        _mcnemar(np.array([[10, 8], [3, 20]])),
        _mcnemar(np.array([[10, 0], [0, 20]])),
    )

    for result in results:
        texts = [result.interpretation, result.methods_fragment, *result.warnings]
        for check in result.assumptions:
            texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
        for text in texts:
            assert not claims_proof(text), text


# --------------------------------------------------------------------------- the refusals


def test_more_than_twenty_levels_is_refused() -> None:
    rows = _group("rows", [f"r{index}" for index in range(21)])
    columns = _group("columns", ["x", "y"])

    with pytest.raises(ToolError) as refused:
        run_contingency(ContingencyRequest(rows=rows, columns=columns))
    assert refused.value.code == "TOO_MANY_ROWS"
    assert "21 labels" in refused.value.message


def test_a_contingency_column_with_one_level_is_refused() -> None:
    rows = _group("rows", ["a", "a", "a"])
    columns = _group("columns", ["x", "y", "x"])

    with pytest.raises(ToolError) as refused:
        run_contingency(ContingencyRequest(rows=rows, columns=columns))
    assert refused.value.code == "SINGLE_GROUP"


@pytest.mark.parametrize("labels", [("a", "a", "a"), ("a", "b", "c")])
def test_mcnemar_needs_exactly_two_levels(labels: tuple[str, ...]) -> None:
    before = _group("before", list(labels))
    after = _group("after", ["x", "y", "x"])

    with pytest.raises(ToolError) as refused:
        run_mcnemar(McNemarRequest(before=before, after=after))
    assert refused.value.code == "LEVELS_REQUIRED"


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        _contingency(np.array([[20, 10], [5, 15]]), level=level)
    assert refused.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as refused:
        _mcnemar(np.array([[10, 8], [3, 20]]), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_contingency(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as refused:
        run_mcnemar(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_option_is_refused_by_the_model() -> None:
    rows = _group("rows", ["a", "b"])
    columns = _group("columns", ["x", "y"])

    with pytest.raises(ValidationError):
        ContingencyRequest(rows=rows, columns=columns, alternative="greater")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- the registry


def test_the_registry_lists_and_runs_both_tools() -> None:
    groups = {"rows": _group("rows", ["a", "b"]), "columns": _group("columns", ["x", "y"])}

    def column(name: str) -> object:
        raise AssertionError(f"{name} is not a numeric column")

    request = build_request(
        "contingency",
        {"rows": "rows", "columns": "columns"},
        column=column,  # type: ignore[arg-type]
        group=groups.__getitem__,
    )
    result = run_tool("contingency", request)
    assert result.tool == "contingency"

    found = next(info for info in catalogue() if info.name == "contingency")
    assert found.family == "counts"
    assert [field.name for field in found.fields] == ["rows", "columns", "level"]
    assert [field.kind for field in found.fields] == ["group", "group", "option"]
    assert all(field.description for field in found.fields)

    before = _group("before", ["a", "b"])
    after = _group("after", ["x", "y"])
    pairs = {"before": before, "after": after}
    mcnemar_request = build_request(
        "mcnemar",
        {"before": "before", "after": "after"},
        column=column,  # type: ignore[arg-type]
        group=pairs.__getitem__,
    )
    assert run_tool("mcnemar", mcnemar_request).tool == "mcnemar"

    found = next(info for info in catalogue() if info.name == "mcnemar")
    assert found.family == "counts"
    assert [field.name for field in found.fields] == ["before", "after", "level"]
    assert [field.kind for field in found.fields] == ["group", "group", "option"]
    assert all(field.description for field in found.fields)

    assert "curvestudio.tools.stats.contingency" in TOOL_MODULES
