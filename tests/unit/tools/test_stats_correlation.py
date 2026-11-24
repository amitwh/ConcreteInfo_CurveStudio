"""The correlation tool: every pair of columns, its coefficient, interval and test.

The reference numbers come from SciPy: ``pearsonr`` for the product-moment coefficient and its
t test, ``spearmanr`` for the rank coefficient and its t approximation, and ``kendalltau`` with
``variant="b"`` and ``method="asymptotic"`` for tau-b and its normal approximation. The Fisher
z interval is written out independently here, and statsmodels supplies the asymptotic variance
of the Pearson coefficient on the normal model (``statsmodels.stats.api.transform_corr_normal``
with ``method="pearson"`` and ``return_var=True``): statsmodels has no Fisher z interval helper,
so the interval itself is checked against the formula and the variance against statsmodels.

The NIST/SEMATECH e-Handbook was searched for a correlation page on 2026-11-24. Its
quantitative-techniques section index (``itl.nist.gov/div898/handbook/eda/section3/eda35.htm``)
was fetched and read; it lists Measures of Location through Yates's algorithm and no
correlation page, so the handbook no longer serves one. The worked example is therefore
Anscombe's quartet, a published dataset with a tabulated coefficient:

- Wikipedia, "Anscombe's quartet" (``en.wikipedia.org/wiki/Anscombe%27s_quartet``), read on
  2026-11-24. The table "For all four datasets", row "Correlation between x and y", gives
  ``0.816`` to 3 decimal places; the data table "Dataset I" through "Dataset IV" gives the
  paired values reproduced below. The printed data give 0.8164, 0.8162, 0.8163 and 0.8165 for
  the four sets, each within the 0.001 the table's 3 decimal places allow.

The size of each coefficient's test is measured on independent normal draws at a fixed seed,
and its power on draws with the coefficient's own population value at 0.3. The properties every
correlation has are checked here too: the matrix is symmetric with 1 on its diagonal, a
monotone increasing transformation leaves the rank coefficients unchanged, and a positive
linear rescaling leaves Pearson's coefficient unchanged.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats
from statsmodels.stats.api import transform_corr_normal

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
from curvestudio.tools.stats.correlation import (
    CorrelationRequest,
    _coefficient,
    _tau_b_se,
    run_correlation,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, catalogue

STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""
P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""


def _column(name: str, values: object, rows: object | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=np.asarray(positions, dtype=np.intp))


def _run(**options: object) -> ToolResult:
    return run_correlation(CorrelationRequest(**options))  # type: ignore[arg-type]


def _pair_row(result: ToolResult, left: str, right: str) -> dict[str, object]:
    """The Pairs row for ``left`` with ``right``, as a dictionary by column key."""
    table = next(found for found in result.tables if found.title == "Pairs")
    keys = [column.key for column in table.columns]
    for row in table.rows:
        cells = dict(zip(keys, row, strict=True))
        if cells["x"] == left and cells["y"] == right:
            return cells
    raise AssertionError(f"no pair {left!r} with {right!r}: {table.rows}")


def _matrix(result: ToolResult) -> list[tuple[object, ...]]:
    table = next(found for found in result.tables if found.title == "Correlation matrix")
    return list(table.rows)


def _fisher_interval(r: float, n: int, level: float) -> tuple[float, float]:
    """The Fisher z interval, written out from the transformation."""
    z = math.atanh(r)
    half = float(scipy_stats.norm.ppf(1.0 - (1.0 - level) / 2.0)) / math.sqrt(n - 3)
    return math.tanh(z - half), math.tanh(z + half)


# --------------------------------------------------------------------------- the references


def test_the_pearson_test_is_scipys_and_the_interval_is_fishers() -> None:
    values = np.random.default_rng(0).normal(size=40)
    other = 0.4 * values + np.random.default_rng(1).normal(size=40)
    x, y = _column("x", values), _column("y", other)

    result = _run(columns=(x, y), method="pearson", level=0.95)
    test = result.tests[0]
    reference = scipy_stats.pearsonr(values, other)
    lower, upper = _fisher_interval(float(reference.statistic), values.size, 0.95)

    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.df == pytest.approx(38.0)
    assert test.effect is not None
    assert test.effect.value == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.effect.lower == pytest.approx(lower, rel=STAT_RTOL)
    assert test.effect.upper == pytest.approx(upper, rel=STAT_RTOL)


def test_the_pearson_asymptotic_variance_is_statsmodels_own() -> None:
    """statsmodels has no Fisher z interval; its normal-model variance is checked instead."""
    values = np.random.default_rng(2).normal(size=30)
    other = -0.3 * values + np.random.default_rng(3).normal(size=30)
    x, y = _column("x", values), _column("y", other)

    result = _run(columns=(x, y), method="pearson")
    r = float(result.tests[0].effect.value)  # type: ignore[union-attr]
    variance = float(transform_corr_normal(r, "pearson", return_var=True).var)

    assert variance == pytest.approx((1.0 - r * r) ** 2, rel=STAT_RTOL)


def test_the_spearman_test_is_scipys_and_the_interval_is_fishers_approximation() -> None:
    values = np.random.default_rng(4).normal(size=35)
    other = np.exp(0.5 * values) + np.random.default_rng(5).normal(size=35) * 0.2
    x, y = _column("x", values), _column("y", other)

    result = _run(columns=(x, y), method="spearman")
    test = result.tests[0]
    reference = scipy_stats.spearmanr(values, other)
    lower, upper = _fisher_interval(float(reference.statistic), values.size, 0.95)

    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.df == pytest.approx(33.0)
    assert test.effect is not None
    assert test.effect.value == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.effect.lower == pytest.approx(lower, rel=STAT_RTOL)
    assert test.effect.upper == pytest.approx(upper, rel=STAT_RTOL)
    assert "approximation" in result.methods_fragment


def test_the_kendall_test_is_scipys_normal_approximation_with_a_tie_corrected_se() -> None:
    rng = np.random.default_rng(6)
    values = np.round(rng.normal(size=40), 1)
    other = np.round(0.5 * values + rng.normal(size=40), 1)
    x, y = _column("x", values), _column("y", other)

    result = _run(columns=(x, y), method="kendall", level=0.95)
    test = result.tests[0]
    reference = scipy_stats.kendalltau(values, other, variant="b", method="asymptotic")
    se = _tau_b_se(values, other)
    quantile = float(scipy_stats.norm.ppf(0.975))
    tau = float(reference.statistic)

    assert test.statistic == pytest.approx(tau, rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.df is None
    assert test.effect is not None
    assert test.effect.se == pytest.approx(se, rel=STAT_RTOL)
    assert test.effect.lower == pytest.approx(max(-1.0, tau - quantile * se), rel=STAT_RTOL)
    assert test.effect.upper == pytest.approx(min(1.0, tau + quantile * se), rel=STAT_RTOL)


def test_the_anscombe_quartet_reproduces_the_tabulated_correlation() -> None:
    """The data and the tabulated 0.816 are read from Wikipedia's Anscombe's quartet page.

    Dataset IV's printed values give 0.8165, which the page's "to 3 decimal places" states as
    0.816; every set therefore lies within the 0.001 that precision allows of the tabulated
    number.
    """
    quartet = {
        "I": (
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        ),
        "II": (
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        ),
        "III": (
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
        ),
        "IV": (
            [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
        ),
    }
    for name, (xs, ys) in quartet.items():
        result = _run(columns=(_column("x", xs), _column("y", ys)), method="pearson")
        coefficient = float(result.tests[0].effect.value)  # type: ignore[union-attr]

        assert abs(coefficient - 0.816) < 1e-3, name
        assert coefficient == pytest.approx(
            float(scipy_stats.pearsonr(xs, ys).statistic), rel=STAT_RTOL
        ), name


# --------------------------------------------------------------------------- size and power


def _independent(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.standard_normal((size, 2))


def _dependent(rng: np.random.Generator, size: int, r: float) -> np.ndarray:
    x = rng.standard_normal(size)
    y = r * x + math.sqrt(1.0 - r * r) * rng.standard_normal(size)
    return np.column_stack((x, y))


def _p_value(method: str, level: float = 0.95) -> Callable[[np.ndarray], float]:
    """The tool's own p-value for one sample of pairs, without building a whole result."""

    def run(sample: np.ndarray) -> float:
        return _coefficient(sample[:, 0], sample[:, 1], method, level).p_value  # type: ignore[arg-type]

    return run


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_each_coefficient_rejects_at_its_level_under_independence(method: str) -> None:
    share = simulated_size(lambda rng: _independent(rng, 30), _p_value(method), draws=4_000, seed=0)

    assert_size(share)


@pytest.mark.parametrize(
    ("method", "population"),
    [
        ("pearson", 0.3),
        ("spearman", 2.0 * math.sin(math.pi * 0.3 / 6.0)),
        ("kendall", math.sin(math.pi * 0.3 / 2.0)),
    ],
)
def test_each_coefficient_detects_its_own_rho_of_point_three(
    method: str, population: float
) -> None:
    """The draws carry the method's own population coefficient at 0.3.

    A bivariate normal with Pearson coefficient ``r`` has Spearman's
    ``6/pi arcsin(r/2)`` and Kendall's ``2/pi arcsin(r)``; the Pearson coefficient is chosen
    from each of those so that every method faces an association of 0.3 on its own scale.
    """
    share = simulated_size(
        lambda rng: _dependent(rng, 100, population), _p_value(method), draws=3_000, seed=7
    )

    assert share >= 0.85, f"{method} detects rho = 0.3 in only {share * 100:.1f} % of draws"


# --------------------------------------------------------------------------- the properties


def test_the_matrix_is_symmetric_with_one_on_its_diagonal() -> None:
    rng = np.random.default_rng(8)
    first = rng.normal(size=25)
    second = 0.6 * first + rng.normal(size=25)
    third = -0.4 * first + rng.normal(size=25)
    columns = (_column("a", first), _column("b", second), _column("c", third))

    result = _run(columns=columns, method="pearson")
    rows = _matrix(result)

    assert [row[0] for row in rows] == ["a", "b", "c"]
    for index in range(3):
        assert rows[index][index + 1] == 1.0
    for row in range(3):
        for column in range(3):
            assert rows[row][column + 1] == rows[column][row + 1]
    assert rows[0][2] == pytest.approx(float(result.tests[0].statistic), rel=STAT_RTOL)


def test_a_monotone_increasing_transformation_leaves_the_rank_coefficients() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=30)
    other = 0.4 * values + rng.normal(size=30)

    for method in ("spearman", "kendall"):
        plain = _coefficient(values, other, method, 0.95)  # type: ignore[arg-type]
        shaped = _coefficient(np.exp(values), other**3, method, 0.95)  # type: ignore[arg-type]
        assert shaped.coefficient == pytest.approx(plain.coefficient, abs=1e-12), method


def test_a_positive_linear_rescaling_of_either_column_leaves_pearson() -> None:
    rng = np.random.default_rng(10)
    values = rng.normal(size=30)
    other = 0.4 * values + rng.normal(size=30)

    plain = _coefficient(values, other, "pearson", 0.95)
    rescaled = _coefficient(3.0 * values + 7.0, 0.5 * other - 2.0, "pearson", 0.95)

    assert rescaled.coefficient == pytest.approx(plain.coefficient, abs=1e-12)
    assert rescaled.p_value == pytest.approx(plain.p_value, abs=1e-12)


# --------------------------------------------------------------------------- the contract


def test_the_result_is_the_contract_with_a_matrix_a_pairs_table_and_a_plot() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=20)
    other = 0.5 * values + rng.normal(size=20)

    result = _run(columns=(_column("x", values), _column("y", other)), method="pearson")

    assert type(result) is ToolResult
    assert result.tool == "correlation"
    assert result.title == "Correlation of x and y"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=20, n_used=20, n_excluded=0)
    assert result.excluded == ()
    assert result.excluded_unlisted == 0
    assert len(result.tests) == 1
    assert result.tests[0].name == "x with y"
    assert result.tests[0].method == "Pearson correlation, t test, two-sided"
    assert result.tests[0].effect is not None
    assert [table.title for table in result.tables] == ["Correlation matrix", "Pairs"]
    assert [column.label for column in result.tables[1].columns] == [
        "X",
        "Y",
        "n",
        "Coefficient",
        "Lower",
        "Upper",
        "p",
    ]
    assert len(result.plots) == 1
    assert isinstance(result.plots[0], ScatterPlot)
    assert result.plots[0].x_label == "x" and result.plots[0].y_label == "y"


def test_three_columns_give_three_tests_and_three_scatter_plots() -> None:
    rng = np.random.default_rng(12)
    first = rng.normal(size=15)
    columns = (
        _column("a", first),
        _column("b", 0.5 * first + rng.normal(size=15)),
        _column("c", -0.3 * first + rng.normal(size=15)),
    )

    result = _run(columns=columns)

    assert len(result.tests) == 3
    assert [test.name for test in result.tests] == ["a with b", "a with c", "b with c"]
    assert len(result.plots) == 3


def test_four_columns_give_six_pairs_and_no_plots() -> None:
    rng = np.random.default_rng(13)
    first = rng.normal(size=15)
    columns = (
        _column("a", first),
        _column("b", 0.5 * first + rng.normal(size=15)),
        _column("c", -0.3 * first + rng.normal(size=15)),
        _column("d", 0.2 * first + rng.normal(size=15)),
    )

    result = _run(columns=columns)

    assert len(result.tests) == 6
    assert result.plots == ()


@pytest.mark.parametrize(
    ("method", "expected", "degrees"),
    [
        ("pearson", "Pearson correlation, t test, two-sided", 23.0),
        ("spearman", "Spearman rank correlation, t test on the ranks", 23.0),
        ("kendall", "Kendall's tau-b, normal approximation", None),
    ],
)
def test_each_method_names_its_test(method: str, expected: str, degrees: float | None) -> None:
    rng = np.random.default_rng(14)
    values = rng.normal(size=25)
    other = 0.4 * values + rng.normal(size=25)

    result = _run(columns=(_column("x", values), _column("y", other)), method=method)
    test = result.tests[0]

    assert test.method == expected
    assert test.df == degrees


def test_every_test_carries_its_coefficient_and_interval() -> None:
    rng = np.random.default_rng(15)
    values = rng.normal(size=30)
    other = 0.4 * values + rng.normal(size=30)

    result = _run(columns=(_column("x", values), _column("y", other)), method="kendall")
    test = result.tests[0]

    assert test.effect is not None
    assert test.effect.value == test.statistic
    assert test.effect.lower is not None and test.effect.upper is not None
    assert test.effect.lower <= test.effect.value <= test.effect.upper
    assert test.p_value_text == p_text(test.p_value)


def test_the_pairs_table_lists_each_pair_with_its_numbers() -> None:
    rng = np.random.default_rng(16)
    values = rng.normal(size=18)
    other = 0.5 * values + rng.normal(size=18)

    result = _run(columns=(_column("x", values), _column("y", other)), method="spearman")
    cells = _pair_row(result, "x", "y")
    test = result.tests[0]

    assert cells["n"] == 18
    assert cells["coefficient"] == test.statistic
    assert cells["lower"] == test.effect.lower  # type: ignore[union-attr]
    assert cells["upper"] == test.effect.upper  # type: ignore[union-attr]
    assert cells["p"] == test.p_value


def test_the_assumptions_name_what_a_coefficient_cannot_check() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=12)
    other = rng.normal(size=12)

    result = _run(columns=(_column("x", values), _column("y", other)), method="spearman")

    assert [check.id for check in result.assumptions] == ["independence", "association_shape"]
    assert all(check.verdict is Verdict.NOT_APPLICABLE for check in result.assumptions)
    assert "monotone" in result.assumptions[1].title


# --------------------------------------------------------------------------- thin and missing data


def test_a_pair_with_fewer_than_three_complete_rows_has_no_coefficient() -> None:
    x = _column("x", [1.0, 2.0])
    y = _column("y", [3.0, 4.0, 5.0, 6.0])

    result = _run(columns=(x, y), method="pearson")
    test = result.tests[0]

    assert test.name == "x with y"
    assert test.statistic is None and test.p_value is None and test.effect is None
    assert test.p_value_text == "not computed"
    assert test.reason is not None and "at least 3" in test.reason
    assert result.warnings and "x with y" in result.warnings[0]
    assert _pair_row(result, "x", "y")["coefficient"] is None
    assert _matrix(result)[0][2] is None
    assert _matrix(result)[0][1] == 1.0


def test_a_constant_column_leaves_its_pairs_without_a_coefficient() -> None:
    x = _column("x", [2.0] * 8)
    y = _column("y", np.arange(8, dtype=np.float64))

    result = _run(columns=(x, y))
    test = result.tests[0]

    assert test.statistic is None and test.effect is None
    assert test.reason is not None and "one value repeated" in test.reason
    assert result.warnings and "no coefficient" in result.warnings[0]
    assert _pair_row(result, "x", "y")["coefficient"] is None


def test_a_row_missing_in_one_column_of_a_pair_is_left_out_of_both() -> None:
    x = _column("x", [1.0, 2.0, 3.0, float("nan"), 5.0])
    y = _column("y", [2.0, 2.0, 4.0, 4.0, 6.0])

    result = _run(columns=(x, y), method="pearson")

    assert result.counts == Counts(n_offered=5, n_used=4, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=3, reason="missing in x"),)
    assert _pair_row(result, "x", "y")["n"] == 4


def test_each_pair_uses_its_own_complete_rows_while_the_counts_use_all_columns() -> None:
    """A row missing only outside a pair still counts as excluded, but that pair keeps it."""
    x = _column("x", [1.0, 2.0, 3.0, float("nan"), 5.0, 6.0, 7.0])
    y = _column("y", [2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0])
    z = _column("z", [3.0, 1.0, 5.0, 2.0, 7.0, 3.0, 9.0])

    result = _run(columns=(x, y, z), method="pearson")

    assert result.counts == Counts(n_offered=7, n_used=6, n_excluded=1)
    assert result.excluded == (ExcludedRow(index=3, reason="missing in x"),)
    assert _pair_row(result, "x", "y")["n"] == 6
    assert _pair_row(result, "y", "z")["n"] == 7


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    x = _column("x", [float("nan")] * 205 + [1.0, 2.0, 3.0])
    y = _column("y", np.arange(1.0, 209.0))

    result = _run(columns=(x, y))

    assert result.counts.n_excluded == 205
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 5


def test_a_pair_of_three_rows_has_a_coefficient_but_no_fisher_interval() -> None:
    x = _column("x", [1.0, 2.0, 4.0])
    y = _column("y", [2.0, 1.0, 5.0])

    pearson = _run(columns=(x, y), method="pearson").tests[0]
    kendall = _run(columns=(x, y), method="kendall").tests[0]

    assert pearson.statistic is not None
    assert pearson.effect is not None
    assert pearson.effect.lower is None and pearson.effect.upper is None
    assert kendall.statistic is not None
    assert kendall.effect is not None
    assert kendall.effect.lower is not None and kendall.effect.upper is not None


# --------------------------------------------------------------------------- the refusals


def test_fewer_than_two_or_more_than_twenty_columns_is_refused() -> None:
    one = _column("a", [1.0, 2.0, 3.0])
    with pytest.raises(ToolError) as refused:
        _run(columns=(one,))
    assert refused.value.code == "INVALID_OPTION"
    assert "between 2 and 20" in refused.value.message

    many = tuple(_column(f"c{index}", [1.0, 2.0, 3.0]) for index in range(21))
    with pytest.raises(ToolError) as refused:
        _run(columns=many)
    assert refused.value.code == "INVALID_OPTION"


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    x = _column("x", [1.0, 2.0, 3.0, 4.0])
    y = _column("y", [2.0, 1.0, 4.0, 3.0])

    with pytest.raises(ToolError) as refused:
        _run(columns=(x, y), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_an_unknown_method_is_refused_by_the_model() -> None:
    x = _column("x", [1.0, 2.0, 3.0])
    y = _column("y", [2.0, 1.0, 4.0])

    with pytest.raises(ValidationError):
        CorrelationRequest(columns=(x, y), method="kendall_tau")  # type: ignore[arg-type]


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_correlation(object())  # type: ignore[arg-type]
    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- the words and registry


def test_the_words_state_evidence_and_claim_no_proof() -> None:
    rng = np.random.default_rng(18)
    values = rng.normal(size=20)
    other = 0.5 * values + rng.normal(size=20)

    result = _run(columns=(_column("x", values), _column("y", other)), method="pearson")

    texts = [result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts.extend([check.title, check.evidence, check.explanation, check.next_step or ""])
    for text in texts:
        assert not claims_proof(text), text


def test_the_registry_lists_and_runs_the_tool() -> None:
    x = _column("x", [1.0, 2.0, 3.0, 4.0, 5.0])
    y = _column("y", [2.0, 1.0, 4.0, 3.0, 6.0])
    columns = {"x": x, "y": y}

    def group(name: str) -> object:
        raise AssertionError("correlation takes no group column")

    request = build_request(
        "correlation",
        {"columns": ["x", "y"], "method": "spearman"},
        column=columns.__getitem__,
        group=group,  # type: ignore[arg-type]
    )
    result = run_tool("correlation", request)

    assert result.tool == "correlation"
    assert result.tests[0].method == "Spearman rank correlation, t test on the ranks"
    assert "curvestudio.tools.stats.correlation" in TOOL_MODULES
    found = next(info for info in catalogue() if info.name == "correlation")
    assert found.family == "relationships"
    assert [field.name for field in found.fields] == ["columns", "method", "level"]
    assert found.fields[0].kind == "columns"
