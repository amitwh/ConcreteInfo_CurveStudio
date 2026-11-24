"""The rank-based comparisons: the Mann-Whitney, the Wilcoxon signed-rank and the sign test.

The numbers are checked against what they must reproduce: the Mann-Whitney p-value and U
against SciPy in both its exact and its asymptotic mode, the Wilcoxon p-value and statistic
against SciPy in both modes, and the sign test against SciPy's exact binomial test. Two worked
examples are derived from the tests' own definitions and checked there.

The NIST/SEMATECH e-Handbook was searched for the worked examples these tests call for, on
2026-11-24: its full detailed table of contents (``itl.nist.gov/div898/handbook/dtoc.htm``,
read from the Internet Archive because the live host now redirects to ``nist.gov/itl``) lists
no Mann-Whitney or Wilcoxon signed-rank page; Kruskal-Wallis is the only rank test in it. The
worked examples below are therefore derived from the U distribution and the signed-rank
distribution themselves and cross-checked against SciPy, not copied from a table that does
not exist in that handbook.

The rest of the file holds the contract of the three tools, the rows they list as left out,
what they do when the data are too thin, and the properties every rank test must have: a
monotone transformation leaves the ranks and the test where they were, the two U statistics
add up to the product of the sample sizes, and the sign test counts the signs it says it does.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import BaseModel
from scipy import stats

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ToolResult,
    assert_size,
    build_request,
    p_text,
    run_tool,
    simulated_size,
)
from curvestudio.tools.stats.ranks import (
    MannWhitneyRequest,
    SignTestRequest,
    WilcoxonRequest,
    run_mannwhitney,
    run_signtest,
    run_wilcoxon,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, TOOLS, catalogue

P_ATOL = 1e-6


def _column(values: Sequence[float], name: str = "y", rows: Sequence[int] | None = None) -> Column:
    positions = list(range(len(values))) if rows is None else list(rows)
    return Column(
        name=name,
        values=np.asarray(values, dtype=np.float64),
        rows=np.asarray(positions, dtype=np.intp),
    )


def _group(
    labels: Sequence[str], name: str = "g", rows: Sequence[int] | None = None
) -> GroupColumn:
    positions = list(range(len(labels))) if rows is None else list(rows)
    return GroupColumn(
        name=name,
        labels=np.asarray(labels, dtype=object),
        rows=np.asarray(positions, dtype=np.intp),
    )


def _groups(x: Sequence[float], y: Sequence[float]) -> tuple[Column, GroupColumn]:
    """One value column holding both samples, and the labels that split it."""
    values = list(x) + list(y)
    labels = ["x"] * len(x) + ["y"] * len(y)
    return _column(values, name="value"), _group(labels, name="sample")


def _estimate(result: ToolResult, label: str):
    for found in result.estimates:
        if found.label == label:
            return found
    raise AssertionError(f"no estimate labelled {label!r}: {[e.label for e in result.estimates]}")


def _row_of(table, key: str, value: object) -> dict[str, object]:
    keys = [column.key for column in table.columns]
    for row in table.rows:
        cells = dict(zip(keys, row, strict=True))
        if cells[key] == value:
            return cells
    raise AssertionError(f"no row with {key} = {value!r}: {table.rows}")


# --------------------------------------------------------------------------- the contract


def test_the_mann_whitney_result_is_the_contract_with_a_table_and_a_box_plot() -> None:
    column, groups = _groups([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0])

    result = run_mannwhitney(MannWhitneyRequest(a=column, groups=groups))

    assert type(result) is ToolResult
    assert result.tool == "mannwhitney"
    assert result.title == "Mann-Whitney comparison of value by sample"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=8, n_used=8, n_excluded=0)
    assert [estimate.label for estimate in result.estimates] == [
        "Hodges-Lehmann shift",
        "Rank-biserial correlation",
    ]
    assert len(result.tests) == 1
    assert result.tests[0].name == "Mann-Whitney U"
    assert result.tests[0].effect == result.estimates[1]
    assert [found.title for found in result.tables] == ["Summary by group"]
    assert [column.label for column in result.tables[0].columns] == [
        "Group",
        "n",
        "Median",
        "Q1",
        "Q3",
    ]
    assert [column.kind for column in result.tables[0].columns] == [
        "text",
        "int",
        "number",
        "number",
        "number",
    ]
    assert len(result.plots) == 1 and result.plots[0].kind == "box"
    assert [box.label for box in result.plots[0].boxes] == ["x", "y"]
    assert [check.id for check in result.assumptions] == ["independence"]


def test_the_wilcoxon_result_is_the_contract_with_two_estimates_and_no_table() -> None:
    result = run_wilcoxon(WilcoxonRequest(a=_column([1.0, 2.0, 3.0, 5.0, 8.0])))

    assert result.tool == "wilcoxon"
    assert result.title == "Wilcoxon signed-rank test of y"
    assert [estimate.label for estimate in result.estimates] == [
        "Pseudo-median of the differences",
        "Matched-pairs rank-biserial correlation",
    ]
    assert result.tests[0].name == "Wilcoxon signed-rank"
    assert result.tables == () and result.plots == ()
    assert result.tests[0].effect == result.estimates[1]


def test_the_sign_result_is_the_contract_with_the_table_of_signs() -> None:
    result = run_signtest(SignTestRequest(a=_column([1.0, 2.0, 3.0, -1.0, -2.0, 0.0])))

    assert result.tool == "signtest"
    assert result.title == "Sign test of y"
    assert [estimate.label for estimate in result.estimates] == ["Median of the differences"]
    assert result.tests[0].name == "Sign test"
    assert result.tests[0].method == "Sign test, two-sided, exact binomial"
    assert result.tests[0].effect == result.estimates[0]
    assert [found.title for found in result.tables] == ["Signs of the differences"]
    assert result.tables[0].rows == (("positive", 3), ("negative", 2), ("zero", 1))


# --------------------------------------------------------------------------- references


def test_the_mann_whitney_asymptotic_p_and_u_match_scipy() -> None:
    x = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0]
    y = [2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    column, groups = _groups(x, y)

    result = run_mannwhitney(MannWhitneyRequest(a=column, groups=groups))

    reference = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    test = result.tests[0]
    assert test.method == (
        "Mann-Whitney U test, two-sided, normal approximation with ties and continuity correction"
    )
    assert test.statistic == pytest.approx(float(reference.statistic), rel=1e-8)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_the_mann_whitney_exact_p_and_u_match_scipy() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    y = [5.0, 6.0, 7.0]
    column, groups = _groups(x, y)

    result = run_mannwhitney(MannWhitneyRequest(a=column, groups=groups))

    reference = stats.mannwhitneyu(x, y, alternative="two-sided", method="exact")
    test = result.tests[0]
    assert test.method == "Mann-Whitney U test, two-sided, exact"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=1e-8)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_the_wilcoxon_exact_p_and_statistic_match_scipy() -> None:
    differences = [1.0, 2.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]

    result = run_wilcoxon(WilcoxonRequest(a=_column(differences)))

    reference = stats.wilcoxon(differences, alternative="two-sided", method="exact")
    test = result.tests[0]
    assert test.method.startswith("Wilcoxon signed-rank test, two-sided, exact distribution")
    assert "zero differences left out following Wilcoxon's original rule" in test.method
    assert test.statistic == pytest.approx(float(reference.statistic), rel=1e-8)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_the_wilcoxon_approximation_matches_scipy_with_ties() -> None:
    differences = [1.0, 1.0, 2.0, 3.0, 3.0, -1.0, -2.0, -3.0, 4.0, -4.0, 5.0]

    result = run_wilcoxon(WilcoxonRequest(a=_column(differences)))

    reference = stats.wilcoxon(
        differences, alternative="two-sided", method="approx", correction=True
    )
    test = result.tests[0]
    assert test.method.startswith("Wilcoxon signed-rank test, two-sided, normal approximation")
    assert test.statistic == pytest.approx(float(reference.statistic), rel=1e-8)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_the_sign_test_p_matches_scipy_s_exact_binomial() -> None:
    differences = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1.0, -2.0, -3.0, -4.0, -5.0, 0.0]

    result = run_signtest(SignTestRequest(a=_column(differences)))

    positive = sum(1 for value in differences if value > 0)
    nonzero = sum(1 for value in differences if value != 0)
    reference = stats.binomtest(positive, nonzero, 0.5, alternative="two-sided")
    test = result.tests[0]
    assert test.statistic == float(min(positive, nonzero - positive))
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.p_value_text == p_text(test.p_value)


# --------------------------------------------------------------------------- worked examples


def test_the_mann_whitney_worked_example_from_the_definitions() -> None:
    """The U distribution of 3 against 5 values, worked out from its own definition.

    With samples of 3 and 5 distinct values there are C(8, 3) = 56 equally likely rank
    assignments. A U of 0 happens when every first-sample value ranks below every second-sample
    value, in 1 assignment, so the two-sided exact p is 2 / 56 = 1 / 28. The Hodges-Lehmann
    shift is the median of the 3 * 5 = 15 pairwise differences, -7, -6, -6, -5, -5, -5, -4, -4,
    -4, -3, -3, -3, -2, -2, -1, which is -4, and its interval is the whole range because the
    left tail of the exact U distribution runs out before it holds 2.5 %.
    """
    column, groups = _groups([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0])

    result = run_mannwhitney(MannWhitneyRequest(a=column, groups=groups))
    test = result.tests[0]
    shift = _estimate(result, "Hodges-Lehmann shift")
    correlation = _estimate(result, "Rank-biserial correlation")

    assert test.statistic == 0.0
    assert test.p_value == pytest.approx(1 / 28, abs=1e-12)
    assert shift.value == pytest.approx(-4.0)
    assert (shift.lower, shift.upper) == (pytest.approx(-7.0), pytest.approx(-1.0))
    assert correlation.value == pytest.approx(-1.0)


def test_the_wilcoxon_worked_example_from_the_definitions() -> None:
    """Ten increasing differences, all positive, read from the signed-rank distribution.

    Under the null each of the 2^10 sign patterns is equally likely, and all ten differences
    being positive is one of the two patterns with an extreme rank sum, so the two-sided exact
    p is 2 / 2^10 = 2^-9. The positive ranks sum to 1 + ... + 10 = 55, so T- is 0 and the
    matched-pairs rank-biserial correlation is (55 - 0) / 55 = 1. The Walsh averages of 1 to 10
    are symmetric about their mean, 5.5, which is therefore the pseudo-median.
    """
    result = run_wilcoxon(WilcoxonRequest(a=_column([float(value) for value in range(1, 11)])))
    test = result.tests[0]
    median = _estimate(result, "Pseudo-median of the differences")
    correlation = _estimate(result, "Matched-pairs rank-biserial correlation")

    assert test.statistic == 0.0
    assert test.p_value == pytest.approx(2.0**-9, abs=1e-12)
    assert median.value == pytest.approx(5.5)
    assert correlation.value == pytest.approx(1.0)
    assert result.counts.n_used == 10


# --------------------------------------------------------------------------- size under the null


def test_the_mann_whitney_holds_its_size_with_ties_and_unequal_samples() -> None:
    """Unequal samples of 15 and 25 values, a 3 to 5 ratio, rounded to one decimal for ties.

    The ratio is the one the task asks for. Samples of 3 and 5 themselves cannot reach the
    3.8 to 6.2 % band: with only C(8, 3) = 56 rank assignments the two-sided exact test's
    largest achievable size below 5 % is 2 / 56 = 3.57 %, and the normal approximation with a
    continuity correction is smaller still. The unequal 3 to 5 shape is kept, at a size the
    normal approximation is calibrated for.
    """
    draws = 4_000

    def draw(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        first = np.round(rng.standard_normal(15), 1)
        second = np.round(rng.standard_normal(25), 1)
        return first, second

    def p_value(sample: tuple[np.ndarray, np.ndarray]) -> float | None:
        first, second = sample
        column, groups = _groups(first.tolist(), second.tolist())
        return run_mannwhitney(MannWhitneyRequest(a=column, groups=groups)).tests[0].p_value

    share = simulated_size(draw, p_value, draws=draws, alpha=0.05, seed=20261124)
    assert_size(share)


def test_the_wilcoxon_holds_its_size_on_differences_symmetric_about_zero() -> None:
    draws = 4_000

    def draw(rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(10)

    def p_value(differences: np.ndarray) -> float | None:
        request = WilcoxonRequest(a=_column(differences.tolist()))
        return run_wilcoxon(request).tests[0].p_value

    share = simulated_size(draw, p_value, draws=draws, alpha=0.05, seed=20261124)
    assert_size(share, exact=True)


def test_the_sign_test_holds_its_size_on_differences_symmetric_about_zero() -> None:
    draws = 4_000

    def draw(rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(17)

    def p_value(differences: np.ndarray) -> float | None:
        request = SignTestRequest(a=_column(differences.tolist()))
        return run_signtest(request).tests[0].p_value

    share = simulated_size(draw, p_value, draws=draws, alpha=0.05, seed=20261124)
    assert_size(share, exact=True)


# --------------------------------------------------------------------------- properties


def test_the_two_u_statistics_add_up_to_the_product_of_the_sample_sizes() -> None:
    first = _column([1.0, 3.0, 5.0, 7.0], name="a")
    second = _column([2.0, 4.0, 6.0, 8.0], name="b")

    forward = run_mannwhitney(MannWhitneyRequest(a=first, b=second)).tests[0].statistic
    backward = run_mannwhitney(MannWhitneyRequest(a=second, b=first)).tests[0].statistic

    assert forward is not None and backward is not None
    assert forward + backward == pytest.approx(4 * 4)


def test_a_monotone_transformation_leaves_the_mann_whitney_untouched() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    labels = ["a"] * 3 + ["b"] * 5
    base = run_mannwhitney(MannWhitneyRequest(a=_column(values), groups=_group(labels)))
    raised = run_mannwhitney(
        MannWhitneyRequest(
            a=_column([math.exp(value) for value in values]),
            groups=_group(labels),
        )
    )

    assert raised.tests[0].statistic == pytest.approx(base.tests[0].statistic)
    assert raised.tests[0].p_value == pytest.approx(base.tests[0].p_value)
    assert _estimate(raised, "Rank-biserial correlation").value == pytest.approx(
        _estimate(base, "Rank-biserial correlation").value
    )


def test_a_monotone_transformation_leaves_the_sign_test_untouched() -> None:
    differences = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0]
    base = run_signtest(SignTestRequest(a=_column(differences)))
    raised = run_signtest(SignTestRequest(a=_column([value**3 for value in differences])))

    assert raised.tables[0].rows == base.tables[0].rows
    assert raised.tests[0].statistic == base.tests[0].statistic
    assert raised.tests[0].p_value == pytest.approx(base.tests[0].p_value)


def test_a_positive_scaling_leaves_the_signed_rank_and_its_p_value_untouched() -> None:
    differences = [-8.0, -5.0, -3.0, -1.0, 2.0, 4.0, 6.0, 7.0]
    base = run_wilcoxon(WilcoxonRequest(a=_column(differences)))
    scaled = run_wilcoxon(WilcoxonRequest(a=_column([3.0 * value for value in differences])))

    assert scaled.tests[0].statistic == pytest.approx(base.tests[0].statistic)
    assert scaled.tests[0].p_value == pytest.approx(base.tests[0].p_value)
    assert _estimate(scaled, "Matched-pairs rank-biserial correlation").value == pytest.approx(
        _estimate(base, "Matched-pairs rank-biserial correlation").value
    )
    # A positive scaling is equivariant for the pseudo-median, not just rank-preserving.
    assert _estimate(scaled, "Pseudo-median of the differences").value == pytest.approx(
        3.0 * _estimate(base, "Pseudo-median of the differences").value
    )


def test_the_sign_counts_are_the_counts_of_the_differences_signs() -> None:
    differences = [1.0, 2.0, 3.0, -1.0, -2.0, 0.0, 0.0, 4.0, -3.0]

    result = run_signtest(SignTestRequest(a=_column(differences)))

    assert result.tables[0].rows == (("positive", 4), ("negative", 3), ("zero", 2))
    assert result.counts.n_used == 9


# --------------------------------------------------------------------------- edge cases


def test_mann_whitney_needs_exactly_one_of_the_second_column_and_the_group_column() -> None:
    column = _column([1.0, 2.0, 3.0])
    for request in (
        MannWhitneyRequest(a=column),
        MannWhitneyRequest(a=column, b=_column([4.0, 5.0]), groups=_group(["a", "b", "a"])),
    ):
        with pytest.raises(ToolError) as refused:
            run_mannwhitney(request)
        assert refused.value.code == "INVALID_OPTION"


def test_mann_whitney_refuses_a_group_column_with_more_than_two_levels() -> None:
    column = _column([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    groups = _group(["a", "b", "c", "a", "b", "c"])

    with pytest.raises(ToolError) as refused:
        run_mannwhitney(MannWhitneyRequest(a=column, groups=groups))
    assert refused.value.code == "LEVELS_REQUIRED"
    assert "3" in refused.value.message


def test_mann_whitney_refuses_a_group_column_with_one_level() -> None:
    column = _column([1.0, 2.0, 3.0])
    with pytest.raises(ToolError) as refused:
        run_mannwhitney(MannWhitneyRequest(a=column, groups=_group(["a", "a", "a"])))
    assert refused.value.code == "SINGLE_GROUP"


def test_mann_whitney_refuses_a_sample_left_empty_by_the_alignment() -> None:
    # The group column names two levels, but every row of "b" has no value to compare.
    column = _column([1.0, 2.0, float("nan"), float("nan")])

    with pytest.raises(ToolError) as refused:
        run_mannwhitney(MannWhitneyRequest(a=column, groups=_group(["a", "a", "b", "b"])))
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "'b'" in refused.value.message


def test_a_level_outside_zero_to_one_is_refused_before_the_data_are_read() -> None:
    for run in (
        lambda: run_mannwhitney(
            MannWhitneyRequest(a=_column([1.0, 2.0]), b=_column([3.0, 4.0]), level=1.0)
        ),
        lambda: run_wilcoxon(WilcoxonRequest(a=_column([1.0, 2.0]), level=0.0)),
        lambda: run_signtest(SignTestRequest(a=_column([1.0, 2.0]), level=2.0)),
    ):
        with pytest.raises(ToolError) as refused:
            run()
        assert refused.value.code == "INVALID_OPTION"


def test_wilcoxon_refuses_a_column_that_is_all_zero_against_the_null() -> None:
    with pytest.raises(ToolError) as refused:
        run_wilcoxon(WilcoxonRequest(a=_column([0.0, 0.0, 0.0])))
    assert refused.value.code == "NO_DATA"
    assert "nonzero" in refused.value.message


def test_the_sign_test_reports_no_p_value_when_every_difference_is_at_the_null() -> None:
    result = run_signtest(SignTestRequest(a=_column([0.0, 0.0, 0.0])))

    test = result.tests[0]
    assert test.p_value is None and test.statistic is None
    assert test.reason is not None and "no sign to count" in test.reason
    assert result.tables[0].rows == (("positive", 0), ("negative", 0), ("zero", 3))
    assert _estimate(result, "Median of the differences").value == pytest.approx(0.0)


def test_more_rows_than_the_tool_is_built_for_is_refused() -> None:
    values = list(np.random.default_rng(1).standard_normal(2_001))
    with pytest.raises(ToolError) as refused:
        run_wilcoxon(WilcoxonRequest(a=_column(values)))
    assert refused.value.code == "TOO_MANY_ROWS"


def test_each_tool_refuses_a_request_that_is_not_its_own() -> None:
    class Other(BaseModel):
        x: int = 1

    for run, name in (
        (run_mannwhitney, "MannWhitneyRequest"),
        (run_wilcoxon, "WilcoxonRequest"),
        (run_signtest, "SignTestRequest"),
    ):
        with pytest.raises(ToolError) as refused:
            run(Other())
        assert refused.value.code == "INVALID_OPTION"
        assert name in refused.value.message


# --------------------------------------------------------------------------- excluded rows


def test_each_independent_column_keeps_its_own_rows() -> None:
    """Two columns of an independent comparison are two samples, not a pairing: a value one
    column lacks does not cost the other column its row."""
    first = _column([1.0, float("nan"), 3.0, float("inf"), 5.0], name="a")
    second = _column([float("nan"), 2.0, 3.0, 4.0, 5.0], name="b")

    result = run_mannwhitney(MannWhitneyRequest(a=first, b=second))

    assert result.counts == Counts(n_offered=10, n_used=7, n_excluded=3)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (0, "missing in b"),
        (1, "missing in a"),
        (3, "not a finite number in a"),
    ]
    assert result.excluded_unlisted == 0


def test_wilcoxon_lists_a_zero_difference_it_leaves_out() -> None:
    differences = [1.0, 0.0, -2.0, 3.0, 0.0]

    result = run_wilcoxon(WilcoxonRequest(a=_column(differences)))

    assert result.counts == Counts(n_offered=5, n_used=3, n_excluded=2)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (1, "no difference from the null value (left out of the signed-rank test)"),
        (4, "no difference from the null value (left out of the signed-rank test)"),
    ]


def test_more_than_two_hundred_missing_rows_are_counted_not_listed() -> None:
    values = [float("nan")] * 205 + [1.0, 2.0, 3.0]

    result = run_signtest(SignTestRequest(a=_column(values)))

    assert len(result.excluded) == 200
    assert result.excluded[0].index == 0 and result.excluded[-1].index == 199
    assert result.excluded_unlisted == 5
    assert result.counts.n_excluded == 205 and result.counts.n_used == 3


# --------------------------------------------------------------------------- words


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    column, groups = _groups([1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0])
    results = (
        run_mannwhitney(MannWhitneyRequest(a=column, groups=groups)),
        run_wilcoxon(WilcoxonRequest(a=_column([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))),
        run_signtest(SignTestRequest(a=_column([1.0, 2.0, 3.0, -1.0, -2.0, -3.0, -4.0]))),
    )

    for result in results:
        texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
        for check in result.assumptions:
            texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
        for text in texts:
            assert not claims_proof(text), text

    for phrase in ("Mann-Whitney", "Hodges-Lehmann", "DeLong", "exact"):
        assert phrase in results[0].methods_fragment, phrase
    for phrase in ("Wilcoxon", "Walsh", "original rule", "jackknife"):
        assert phrase in results[1].methods_fragment, phrase
    for phrase in ("sign test", "binomial", "order statistics"):
        assert phrase in results[2].methods_fragment, phrase


# --------------------------------------------------------------------------- registration


def test_the_three_tools_are_registered_under_ranks_with_described_fields() -> None:
    assert "curvestudio.tools.stats.ranks" in TOOL_MODULES

    expected = {
        "mannwhitney": (
            "Mann-Whitney U test",
            ["a", "b", "groups", "level"],
            ["column", "column", "group", "option"],
            ["First column", "Second column", "Groups", "Level"],
        ),
        "wilcoxon": (
            "Wilcoxon signed-rank test",
            ["a", "b", "null", "level"],
            ["column", "column", "option", "option"],
            ["First column", "Second column", "Null value", "Level"],
        ),
        "signtest": (
            "Sign test",
            ["a", "b", "null", "level"],
            ["column", "column", "option", "option"],
            ["First column", "Second column", "Null value", "Level"],
        ),
    }
    for name, (title, fields, kinds, labels) in expected.items():
        assert name in TOOLS
        spec = TOOLS[name]
        assert spec.family == "ranks" and spec.title == title
        info = TOOLS.info(name)
        assert info.needs_data
        assert [field.name for field in info.fields] == fields
        assert [field.kind for field in info.fields] == kinds
        assert [field.label for field in info.fields] == labels
        assert all(field.description for field in info.fields)
        assert info.fields[-1].option_schema is not None
    assert {info.name for info in catalogue() if info.family == "ranks"} == set(expected)


def test_the_registry_builds_and_runs_the_mann_whitney_from_column_names() -> None:
    columns = {"v": _column([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])}
    groups = {"s": _group(["a", "a", "a", "b", "b", "b", "b", "b"])}

    request = build_request(
        "mannwhitney",
        {"a": "v", "groups": "s", "level": 0.9},
        column=lambda name: columns[name],
        group=lambda name: groups[name],
    )
    assert isinstance(request, MannWhitneyRequest)
    result = run_tool("mannwhitney", request)
    assert result.tool == "mannwhitney" and result.level == 0.9

    wilcoxon = build_request(
        "wilcoxon",
        {"a": "v", "null": 1.0},
        column=lambda name: columns[name],
        group=lambda name: groups[name],
    )
    assert isinstance(wilcoxon, WilcoxonRequest) and wilcoxon.b is None
    assert wilcoxon.null == 1.0
