"""Two-way analysis of variance, repeated measures and the Friedman test.

Every number is traced to a public source fetched and read on 2026-11-24:

- Two-way analysis of variance: the NIST/SEMATECH e-Handbook of Statistical Methods,
  ``7.4.3.8. Models and calculations for the two-way ANOVA``
  (itl.nist.gov/div898/handbook/prc/section4/prc438.htm) tabulates, for a coating evaluated
  on 3 materials at 2 laboratories with 3 samples per cell, the sums of squares 5.0139
  (laboratory), 2.1811 (material), 0.1344 (interaction) and 0.6000 (error), their mean
  squares, F statistics of 100.28, 21.81 and 1.34, and p-values 0, .0001 and .298.
- Repeated-measures example: the ``rm_anova_wide`` data set of the pingouin project, and the
  values its test suite tabulates against JAMOVI in ``tests/test_parametric.py``
  (github.com/raphaelvallat/pingouin, fetched 2026-11-24): F = 5.20065, p = 0.00656,
  generalised eta squared = 0.34639, Greenhouse-Geisser epsilon = 0.69433, Mauchly's
  W = 0.30678 and the corrected p = 0.01670, over the 9 complete subjects.
- Epsilons: the A/B/C example of ``pingouin.distribution.epsilon``'s docstring prints
  ``0.50 0.56 0.62`` for the lower bound, Greenhouse-Geisser and Huynh-Feldt epsilons; the
  formulas themselves are car's ``summary.Anova.mlm`` and pingouin's ``_gg_epsilon`` /
  ``_mauchly`` (both fetched 2026-11-24).
- Friedman and Durbin-Conover: the data ``df_bn`` and the raw Durbin-Conover p-value matrix
  of the scikit-posthocs test suite, ``tests/test_posthocs.py``, test
  ``test_posthoc_conover_friedman_non_melted`` (github.com/maximtrp/scikit-posthocs, fetched
  2026-11-24); the statistic itself is SciPy's ``friedmanchisquare``.

statsmodels 0.15 is the reference for the Type II table (``ols(...).fit().anova_lm(typ=2)``)
and the repeated-measures F (``AnovaRM``). Every test is shown to run at its nominal size on
data drawn from its null hypothesis, with a fixed seed.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM, anova_lm

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof, num
from curvestudio.tools.stats import Verdict, assert_size, p_text, simulated_size
from curvestudio.tools.stats.anova_more import (
    FriedmanRequest,
    RepeatedMeasuresRequest,
    TwoWayRequest,
    _contrast_covariance,
    _friedman_sums,
    _gg_epsilon,
    _hf_epsilon,
    _mauchly,
    _rm_sums,
    _twoway_sums,
    run_friedman,
    run_repeated,
    run_twoway,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, TOOLS, catalogue

P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""
STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""

NIST_VALUES = (
    4.1,
    3.9,
    4.3,
    3.1,
    2.8,
    3.3,
    3.5,
    3.2,
    3.6,
    2.7,
    3.1,
    2.6,
    1.9,
    2.2,
    2.3,
    2.7,
    2.3,
    2.5,
)
"""The NIST two-way example's 18 measurements, laboratory by laboratory."""

NIST_LABS = ("1",) * 9 + ("2",) * 9
"""The NIST example's laboratory labels."""

NIST_MATERIALS = ("1", "1", "1", "2", "2", "2", "3", "3", "3") * 2
"""The NIST example's material labels."""

RM_WIDE = (
    (4.3, 5.3, 4.8, 6.3),
    (3.9, 2.3, 5.6, 4.3),
    (4.5, 2.6, 4.1, None),
    (5.1, 4.2, 6.0, 6.3),
    (3.8, 3.6, 4.8, 6.8),
    (3.8, 5.4, 5.1, 6.3),
    (5.1, 3.7, None, None),
    (4.5, 3.0, 7.7, 5.8),
    (3.7, 7.1, 5.5, 4.8),
    (4.4, 4.7, 4.2, 5.5),
    (None, 5.1, 6.5, 5.7),
    (3.7, 3.0, 4.0, 7.1),
)
"""The ``rm_anova_wide`` data set: 12 subjects, 4 conditions, and three incomplete rows."""

RM_NAMES = ("Before", "1 week", "2 week", "3 week")
"""The conditions of the repeated-measures example."""

RM_COMPLETE = np.array(
    [
        [4.3, 5.3, 4.8, 6.3],
        [3.9, 2.3, 5.6, 4.3],
        [5.1, 4.2, 6.0, 6.3],
        [3.8, 3.6, 4.8, 6.8],
        [3.8, 5.4, 5.1, 6.3],
        [4.5, 3.0, 7.7, 5.8],
        [3.7, 7.1, 5.5, 4.8],
        [4.4, 4.7, 4.2, 5.5],
        [3.7, 3.0, 4.0, 7.1],
    ],
    dtype=np.float64,
)
"""The 9 complete subjects of the repeated-measures example, listwise."""

ABC = np.array(
    [
        [2.2, 1.1, 8.2],
        [3.1, 2.5, 4.5],
        [4.3, 4.1, 3.4],
        [4.1, 5.2, 6.2],
        [7.2, 6.4, 7.2],
    ],
    dtype=np.float64,
)
"""The A/B/C example of the ``epsilon`` docstring: 5 subjects, 3 conditions."""

DC_DATA = np.array(
    [[4, 3, 4, 4, 5, 6, 3], [1, 2, 3, 5, 6, 7, 7], [1, 2, 6, 4, 1, 5, 1]], dtype=np.float64
)
"""The 3-block, 7-condition matrix of the scikit-posthocs Durbin-Conover test."""

DC_PUBLISHED = np.array(
    [
        [1.0, 0.9147508, 0.1518030, 0.2140927, 0.2140927, 0.0181602, 0.5242303],
        [0.9147508, 1.0, 0.18071036, 0.25232845, 0.25232845, 0.02222747, 0.59465124],
        [0.1518030, 0.18071036, 1.0, 0.8305955, 0.8305955, 0.2523284, 0.3989535],
        [0.2140927, 0.25232845, 0.8305955, 1.0, 1.0, 0.1807104, 0.5242303],
        [0.2140927, 0.25232845, 0.8305955, 1.0, 1.0, 0.1807104, 0.5242303],
        [0.0181602, 0.02222747, 0.2523284, 0.1807104, 0.1807104, 1.0, 0.05991984],
        [0.5242303, 0.59465124, 0.3989535, 0.5242303, 0.5242303, 0.05991984, 1.0],
    ]
)
"""The raw Durbin-Conover p-values scikit-posthocs' test suite tabulates."""


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


def _wide_columns(
    rows: Sequence[Sequence[float | None]], names: Sequence[str]
) -> tuple[Column, ...]:
    """One column per condition from a wide table that may hold ``None`` for a missing cell."""
    columns: list[Column] = []
    for j, name in enumerate(names):
        values = [row[j] for row in rows if row[j] is not None]
        positions = [index for index, row in enumerate(rows) if row[j] is not None]
        columns.append(_column(name, values, positions))
    return tuple(columns)


def _matrix_columns(
    matrix: NDArray[np.float64], names: Sequence[str] | None = None
) -> tuple[Column, ...]:
    """One column per column of a complete matrix, named after it or as ``condition j``."""
    labels = (
        list(names) if names is not None else [f"condition{j + 1}" for j in range(matrix.shape[1])]
    )
    return tuple(_column(label, matrix[:, j]) for j, label in enumerate(labels))


def _nist() -> TwoWayRequest:
    """The NIST two-way example as a request."""
    return TwoWayRequest(
        values=_column("strength", NIST_VALUES),
        factor_a=_group("lab", NIST_LABS),
        factor_b=_group("material", NIST_MATERIALS),
    )


def _rm(
    values: NDArray[np.float64] | None = None, names: Sequence[str] | None = None
) -> RepeatedMeasuresRequest:
    """A repeated-measures request, over ``values`` or the wide example."""
    if values is None:
        return RepeatedMeasuresRequest(measures=_wide_columns(RM_WIDE, RM_NAMES))
    return RepeatedMeasuresRequest(
        measures=_matrix_columns(values, names),
        names=tuple(names) if names is not None else None,
    )


def _friedman(
    values: NDArray[np.float64] | None = None, names: Sequence[str] | None = None
) -> FriedmanRequest:
    """A Friedman request, over ``values`` or the scikit-posthocs matrix."""
    matrix = DC_DATA if values is None else values
    return FriedmanRequest(
        measures=_matrix_columns(matrix, names),
        names=tuple(names) if names is not None else None,
    )


def _design_codes(na: int, nb: int, replicates: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The A and B codes of a balanced ``na x nb`` design with ``replicates`` per cell."""
    a_codes: list[int] = []
    b_codes: list[int] = []
    for i in range(na):
        for j in range(nb):
            for _ in range(replicates):
                a_codes.append(i)
                b_codes.append(j)
    return (
        np.array(a_codes, dtype=np.intp),
        np.array(b_codes, dtype=np.intp),
    )


# --------------------------------------------------------------------------- the two-way table


def test_the_nist_two_way_example_reproduces_its_table() -> None:
    result = run_twoway(_nist())
    rows = {row[0]: row for row in result.tables[0].rows}

    published = {
        "lab": (5.0139, 1, 5.0139, 100.28),
        "material": (2.1811, 2, 1.0906, 21.81),
        "lab × material": (0.1344, 2, 0.0672, 1.34),  # noqa: RUF001
        "Residual": (0.6000, 12, 0.0500, None),
    }
    for source, (ss, df, ms, f) in published.items():
        assert rows[source][1] == pytest.approx(ss, rel=1e-3), source
        assert rows[source][2] == pytest.approx(df, rel=1e-9), source
        assert rows[source][3] == pytest.approx(ms, rel=1e-3), source
        if f is not None:
            # The page prints F to two decimals, so the last digit is not to be trusted.
            assert rows[source][4] == pytest.approx(f, rel=5e-3), source

    assert rows["lab"][5] < 1e-6
    assert rows["material"][5] == pytest.approx(0.0001, abs=1e-5)
    assert rows["lab × material"][5] == pytest.approx(0.298, abs=1e-3)  # noqa: RUF001
    assert result.tables[0].title == "Two-way analysis of variance (Type II)"


def test_the_type_two_table_matches_statsmodels_on_an_unbalanced_design() -> None:
    rng = np.random.default_rng(11)
    counts = ((4, 3, 2), (3, 5, 2), (2, 3, 4))
    values: list[float] = []
    a_labels: list[str] = []
    b_labels: list[str] = []
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            values.extend(rng.normal(1.0 * i + 0.5 * j + 0.1 * i * j, 1.0, count))
            a_labels.extend([f"a{i}"] * count)
            b_labels.extend([f"b{j}"] * count)
    request = TwoWayRequest(
        values=_column("y", values),
        factor_a=_group("a", a_labels),
        factor_b=_group("b", b_labels),
    )
    result = run_twoway(request)
    rows = {row[0]: row for row in result.tables[0].rows}

    frame = pd.DataFrame({"y": values, "a": a_labels, "b": b_labels})
    reference = anova_lm(ols("y ~ C(a)*C(b)", data=frame).fit(), typ=2)
    residual = float(reference.loc["Residual", "sum_sq"])
    for source, term in (("a", "C(a)"), ("b", "C(b)"), ("a × b", "C(a):C(b)")):  # noqa: RUF001
        assert rows[source][1] == pytest.approx(
            float(reference.loc[term, "sum_sq"]), rel=STAT_RTOL
        ), source
        assert rows[source][2] == pytest.approx(float(reference.loc[term, "df"]), rel=STAT_RTOL)
        assert rows[source][3] == pytest.approx(
            float(reference.loc[term, "sum_sq"]) / float(reference.loc[term, "df"]),
            rel=STAT_RTOL,
        )
        assert rows[source][4] == pytest.approx(float(reference.loc[term, "F"]), rel=STAT_RTOL)
        assert rows[source][5] == pytest.approx(float(reference.loc[term, "PR(>F)"]), abs=P_ATOL)
        assert rows[source][6] == pytest.approx(
            float(reference.loc[term, "sum_sq"])
            / (float(reference.loc[term, "sum_sq"]) + residual),
            rel=STAT_RTOL,
        )
    assert rows["Residual"][1] == pytest.approx(residual, rel=STAT_RTOL)


def test_a_balanced_design_gives_the_same_table_for_type_one_two_and_three() -> None:
    rng = np.random.default_rng(12)
    a_codes, b_codes = _design_codes(2, 3, 4)
    a_effect = np.array([0.0, 1.5])
    b_effect = np.array([0.0, 0.5, 1.0])
    values = a_effect[a_codes] + b_effect[b_codes] + rng.normal(0.0, 1.0, a_codes.size)
    request = TwoWayRequest(
        values=_column("y", values),
        factor_a=_group("a", [f"a{code}" for code in a_codes]),
        factor_b=_group("b", [f"b{code}" for code in b_codes]),
    )
    result = run_twoway(request)
    rows = {row[0]: row for row in result.tables[0].rows}

    frame = pd.DataFrame(
        {
            "y": values,
            "a": [f"a{code}" for code in a_codes],
            "b": [f"b{code}" for code in b_codes],
        }
    )
    # Sum-to-zero contrasts, as R's car uses, so Type III tests the main effects.
    fitted = ols("y ~ C(a, Sum)*C(b, Sum)", data=frame).fit()
    type_one = anova_lm(fitted, typ=1)
    type_three = anova_lm(fitted, typ=3)
    terms = {
        "a": "C(a, Sum)",
        "b": "C(b, Sum)",
        "a × b": "C(a, Sum):C(b, Sum)",  # noqa: RUF001
    }
    for source, term in terms.items():
        assert rows[source][1] == pytest.approx(float(type_one.loc[term, "sum_sq"]), rel=STAT_RTOL)
        assert rows[source][1] == pytest.approx(
            float(type_three.loc[term, "sum_sq"]), rel=STAT_RTOL
        )


def test_an_empty_cell_is_a_warning_and_leaves_the_interaction_untested() -> None:
    values = (1.0, 1.2, 2.0, 2.2, 3.0, 3.4)
    labs = ("1",) * 2 + ("2",) * 2 + ("2",) * 2
    materials = ("1",) * 2 + ("1",) * 2 + ("2",) * 2
    result = run_twoway(
        TwoWayRequest(
            values=_column("y", values),
            factor_a=_group("lab", labs),
            factor_b=_group("material", materials),
        )
    )
    assert result.warnings
    assert "holds no value" in result.warnings[0]
    interaction = result.tests[2]
    assert interaction.df == 0.0
    assert interaction.p_value is None
    assert interaction.reason is not None


def test_the_cell_means_table_holds_every_cell() -> None:
    result = run_twoway(_nist())
    cell_table = result.tables[1]
    assert cell_table.title == "Cell means"
    assert len(cell_table.rows) == 6
    assert all(row[2] == 3 for row in cell_table.rows)
    expected = {
        ("1", "1"): 4.1,
        ("1", "2"): (3.1 + 2.8 + 3.3) / 3,
        ("1", "3"): (3.5 + 3.2 + 3.6) / 3,
        ("2", "1"): (2.7 + 3.1 + 2.6) / 3,
        ("2", "2"): (1.9 + 2.2 + 2.3) / 3,
        ("2", "3"): (2.7 + 2.3 + 2.5) / 3,
    }
    for row in cell_table.rows:
        assert row[3] == pytest.approx(expected[(str(row[0]), str(row[1]))], rel=1e-12)


def test_the_interaction_plot_is_one_line_per_material() -> None:
    result = run_twoway(_nist())
    plot = result.plots[0]
    assert plot.kind == "lines"
    assert plot.x_labels == ("1", "2")
    assert [series.label for series in plot.series] == ["1", "2", "3"]
    assert all(series.x == (0.0, 1.0) for series in plot.series)


def test_the_two_way_result_carries_its_assumptions_and_counts() -> None:
    result = run_twoway(_nist())
    assert [check.id for check in result.assumptions] == [
        "normality_of_residuals",
        "equal_variances",
    ]
    assert result.counts.n_offered == 18
    assert result.counts.n_used == 18
    assert result.counts.n_excluded == 0
    assert "Type II" in result.methods_fragment


# --------------------------------------------------------------------------- repeated measures


def test_the_repeated_measures_f_matches_statsmodels_and_the_published_value() -> None:
    result = run_repeated(_rm())
    test = result.tests[0]
    subjects, conditions = RM_COMPLETE.shape
    frame = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(subjects), conditions),
            "condition": np.tile(np.arange(conditions), subjects),
            "value": RM_COMPLETE.reshape(-1),
        }
    )
    reference = AnovaRM(frame, "value", "subject", within=["condition"]).fit().anova_table
    row = reference.loc["condition"]
    assert test.statistic == pytest.approx(float(row["F Value"]), rel=1e-6)
    assert test.df == float(row["Num DF"])
    assert test.df_denominator == float(row["Den DF"])
    assert test.p_value == pytest.approx(float(row["Pr > F"]), abs=P_ATOL)
    assert test.statistic == pytest.approx(5.20065, rel=1e-5)
    assert test.p_value == pytest.approx(0.00656, abs=5e-5)
    assert p_text(test.p_value) == test.p_value_text


def test_generalised_eta_squared_matches_the_published_value() -> None:
    result = run_repeated(_rm())
    effect = result.estimates[0]
    assert effect.label == "Generalised eta squared"
    assert effect.value == pytest.approx(0.34639, rel=1e-4)
    assert all(test.effect == effect for test in result.tests)


def test_the_greenhouse_geisser_epsilon_and_its_p_value_match_the_published_values() -> None:
    result = run_repeated(_rm())
    sums = _rm_sums(RM_COMPLETE)
    assert sums.epsilon_gg == pytest.approx(0.69433, rel=1e-4)
    assert sums.p_gg == pytest.approx(0.01670, abs=5e-5)
    corrected = result.tests[1]
    assert corrected.df == pytest.approx(sums.epsilon_gg * sums.df_conditions, rel=STAT_RTOL)
    assert corrected.p_value == pytest.approx(sums.p_gg, abs=0.0)
    assert f"epsilon = {num(sums.epsilon_gg)}" in corrected.method


def test_the_huynh_feldt_epsilon_matches_the_published_example() -> None:
    sums = _rm_sums(ABC)
    assert sums.epsilon_gg == pytest.approx(0.56, abs=5e-3)
    assert sums.epsilon_hf == pytest.approx(0.62, abs=5e-3)
    # Written out independently from car's summary.Anova.mlm.
    subjects, conditions = ABC.shape
    contrast = _contrast_covariance(ABC)
    dimension = conditions - 1
    expected_gg = min(
        float(np.trace(contrast) ** 2 / (dimension * np.trace(contrast @ contrast))), 1.0
    )
    expected_hf = min(
        (subjects * dimension * expected_gg - 2.0)
        / (dimension * (subjects - 1 - dimension * expected_gg)),
        1.0,
    )
    assert _gg_epsilon(contrast) == pytest.approx(expected_gg, rel=STAT_RTOL)
    assert _hf_epsilon(expected_gg, subjects, dimension) == pytest.approx(
        expected_hf, rel=STAT_RTOL
    )


def test_mauchly_matches_the_published_example_and_its_formula() -> None:
    sums = _rm_sums(RM_COMPLETE)
    assert sums.epsilon_gg == pytest.approx(0.69433, rel=1e-4)
    subjects, _conditions = RM_COMPLETE.shape
    contrast = _contrast_covariance(RM_COMPLETE)
    found = _mauchly(contrast, subjects - 1, 4)
    assert found.w == pytest.approx(0.30678, rel=1e-4)

    # Written out independently from R's mauchly.test, which pingouin follows.
    dimension = contrast.shape[0]
    sign, log_determinant = np.linalg.slogdet(contrast)
    log_w = log_determinant - dimension * math.log(float(np.trace(contrast)) / dimension)
    assert sign > 0
    correction = 1.0 - (2.0 * dimension**2 + dimension + 2.0) / (6.0 * dimension * (subjects - 1))
    w2 = (
        (dimension + 2.0)
        * (dimension - 1.0)
        * (dimension - 2.0)
        * (2.0 * dimension**3 + 6.0 * dimension**2 + 3.0 * 4 + 2.0)
        / (288.0 * ((subjects - 1) * dimension * correction) ** 2)
    )
    chi_square = -(subjects - 1) * correction * log_w
    degrees = dimension * (dimension + 1) / 2.0 - 1.0
    p_one = float(scipy_stats.chi2.sf(chi_square, degrees))
    p_two = float(scipy_stats.chi2.sf(chi_square, degrees + 4.0))
    expected_p = p_one + w2 * (p_two - p_one)
    assert found.chi_square == pytest.approx(chi_square, rel=STAT_RTOL)
    assert found.df == pytest.approx(degrees, rel=STAT_RTOL)
    assert found.p_value == pytest.approx(expected_p, rel=1e-6)

    result = run_repeated(_rm())
    check = result.assumptions[0]
    assert check.id == "sphericity"
    assert check.statistic == f"W = {num(found.w)}"
    assert check.verdict is Verdict.PASS


def test_two_conditions_equal_the_paired_t_test() -> None:
    rng = np.random.default_rng(21)
    values = rng.normal(0.0, 1.0, (12, 2)) + np.array([0.0, 0.7])
    result = run_repeated(_rm(values, ("before", "after")))
    paired = scipy_stats.ttest_rel(values[:, 0], values[:, 1])
    test = result.tests[0]
    assert test.df == 1.0
    assert test.df_denominator == 11.0
    assert test.statistic == pytest.approx(float(paired.statistic) ** 2, rel=STAT_RTOL)
    assert test.p_value == pytest.approx(float(paired.pvalue), abs=P_ATOL)
    for corrected in result.tests[1:]:
        assert corrected.df == pytest.approx(1.0, rel=STAT_RTOL)
        assert corrected.df_denominator == pytest.approx(11.0, rel=STAT_RTOL)
        assert corrected.p_value == pytest.approx(test.p_value, abs=0.0)
    assert result.assumptions[0].verdict is Verdict.NOT_APPLICABLE


def test_the_repeated_result_carries_the_tables_and_a_box_plot() -> None:
    result = run_repeated(_rm())
    assert result.tables[0].title == "Repeated-measures analysis of variance"
    assert result.tables[0].columns[0].key == "source"
    summary = result.tables[1]
    assert summary.title == "Summary by condition"
    assert [row[0] for row in summary.rows] == list(RM_NAMES)
    assert all(row[1] == 9 for row in summary.rows)
    plot = result.plots[0]
    assert plot.kind == "box"
    assert [box.label for box in plot.boxes] == list(RM_NAMES)


def test_incomplete_subjects_are_counted_and_never_dropped_silently() -> None:
    result = run_repeated(_rm())
    assert result.counts.n_offered == 12
    assert result.counts.n_used == 9
    assert result.counts.n_excluded == 3
    assert [row.index for row in result.excluded] == [2, 6, 10]
    assert all("missing in" in row.reason for row in result.excluded)


def test_a_repeated_measures_request_needs_between_two_and_twenty_conditions() -> None:
    one = _matrix_columns(np.arange(6.0).reshape(3, 2))
    with pytest.raises(ToolError) as refused:
        run_repeated(RepeatedMeasuresRequest(measures=one[:1]))
    assert refused.value.code == "INVALID_OPTION"

    many = _matrix_columns(np.arange(63.0).reshape(3, 21))
    with pytest.raises(ToolError) as too_many:
        run_repeated(RepeatedMeasuresRequest(measures=many))
    assert too_many.value.code == "INVALID_OPTION"


def test_condition_names_default_to_the_columns_and_a_mismatch_is_refused() -> None:
    result = run_repeated(_rm())
    assert result.title.endswith("Before, 1 week, 2 week and 3 week")
    columns = _matrix_columns(np.arange(8.0).reshape(4, 2))
    with pytest.raises(ToolError) as refused:
        run_repeated(RepeatedMeasuresRequest(measures=columns, names=("only one",)))
    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- Friedman


def test_the_friedman_statistic_matches_scipy() -> None:
    result = run_friedman(_friedman())
    test = result.tests[0]
    reference = scipy_stats.friedmanchisquare(*(DC_DATA[:, j] for j in range(DC_DATA.shape[1])))
    assert test.name == "Friedman test"
    assert test.method == "Friedman test, chi-square approximation with ties"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == 6.0
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)


def test_kendalls_w_follows_its_formula() -> None:
    result = run_friedman(_friedman())
    effect = result.estimates[0]
    test = result.tests[0]
    subjects, conditions = DC_DATA.shape
    expected = float(test.statistic) / (subjects * (conditions - 1))
    assert effect.label == "Kendall's W"
    assert effect.value == pytest.approx(expected, rel=STAT_RTOL)
    assert test.effect == effect


def test_durbin_conover_matches_the_published_p_values_and_holm() -> None:
    result = run_friedman(_friedman())
    comparisons = result.tests[1:]
    conditions = DC_DATA.shape[1]
    pairs = [(i, j) for i in range(conditions) for j in range(i + 1, conditions)]
    assert len(comparisons) == len(pairs)
    for test, (first, second) in zip(comparisons, pairs, strict=True):
        assert test.p_value == pytest.approx(DC_PUBLISHED[first, second], abs=P_ATOL)
        assert test.adjustment == "Holm"

    raw = [float(test.p_value) for test in comparisons]
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    running = 0.0
    expected = [0.0] * len(raw)
    for step, index in enumerate(order, start=1):
        running = max(running, (len(raw) - step + 1) * raw[index])
        expected[index] = min(1.0, running)
    for test, adjusted in zip(comparisons, expected, strict=True):
        assert test.adjusted_p == pytest.approx(adjusted, abs=1e-12)


def test_the_mean_ranks_by_condition_table_holds_each_mean_rank() -> None:
    result = run_friedman(_friedman())
    sums = _friedman_sums(DC_DATA)
    table = result.tables[0]
    assert table.title == "Mean ranks by condition"
    subjects = DC_DATA.shape[0]
    for row, rank_sum in zip(table.rows, sums.rank_sums, strict=True):
        assert row[2] == pytest.approx(float(rank_sum) / subjects, rel=STAT_RTOL)


def test_the_friedman_result_carries_a_box_plot_and_counts() -> None:
    result = run_friedman(_friedman())
    assert result.plots[0].kind == "box"
    assert result.counts.n_offered == 3
    assert result.counts.n_used == 3
    assert result.counts.n_excluded == 0


def test_the_friedman_test_needs_three_conditions() -> None:
    columns = _matrix_columns(np.arange(10.0).reshape(5, 2))
    with pytest.raises(ToolError) as refused:
        run_friedman(FriedmanRequest(measures=columns))
    assert refused.value.code == "INVALID_OPTION"


def test_an_incomplete_friedman_subject_is_counted_and_listed() -> None:
    rows = ((1.0, 2.0, 3.0), (2.0, 1.0, None), (1.0, 3.0, 2.0))
    columns = _wide_columns(rows, ("a", "b", "c"))
    result = run_friedman(FriedmanRequest(measures=columns))
    assert result.counts.n_offered == 3
    assert result.counts.n_used == 2
    assert result.counts.n_excluded == 1
    assert result.excluded[0].index == 1


# --------------------------------------------------------------------------- the result's words


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    results = (
        run_twoway(_nist()),
        run_repeated(_rm()),
        run_friedman(_friedman()),
    )
    for result in results:
        texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
        for check in result.assumptions:
            texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
        for test in result.tests:
            texts += [test.name, test.method, test.null_hypothesis]
        for text in texts:
            assert not claims_proof(text), text

    assert "Type II" in results[0].methods_fragment
    assert "Mauchly" in results[1].methods_fragment
    assert "Kendall" in results[2].methods_fragment


def test_the_interpretations_name_the_statistics_and_effect_sizes() -> None:
    two_way = run_twoway(_nist())
    assert p_text(two_way.tests[0].p_value) in two_way.interpretation
    assert "Partial eta squared" in two_way.interpretation
    assert len(two_way.interpretation.split(". ")) <= 3

    repeated = run_repeated(_rm())
    assert "generalised eta squared" in repeated.interpretation.lower()
    assert "Greenhouse-Geisser" in repeated.interpretation
    assert len(repeated.interpretation.split(". ")) <= 3


# --------------------------------------------------------------------------- refusals


def test_a_request_of_another_tool_is_refused() -> None:
    with pytest.raises(ToolError) as two_way:
        run_twoway(RepeatedMeasuresRequest(measures=_rm().measures))
    assert two_way.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as repeated:
        run_repeated(FriedmanRequest(measures=_friedman().measures))
    assert repeated.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as friedman:
        run_friedman(RepeatedMeasuresRequest(measures=_rm().measures))
    assert friedman.value.code == "INVALID_OPTION"


def test_a_factor_of_one_level_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_twoway(
            TwoWayRequest(
                values=_column("y", NIST_VALUES),
                factor_a=_group("lab", ("1",) * 18),
                factor_b=_group("material", NIST_MATERIALS),
            )
        )
    assert refused.value.code == "SINGLE_GROUP"


def test_a_confidence_level_outside_zero_and_one_is_refused() -> None:
    with pytest.raises(ToolError) as two_way:
        run_twoway(
            TwoWayRequest(
                values=_nist().values,
                factor_a=_nist().factor_a,
                factor_b=_nist().factor_b,
                level=1.0,
            )
        )
    assert two_way.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as repeated:
        run_repeated(RepeatedMeasuresRequest(measures=_rm().measures, level=0.0))
    assert repeated.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as friedman:
        run_friedman(FriedmanRequest(measures=_friedman().measures, level=0.0))
    assert friedman.value.code == "INVALID_OPTION"


def test_a_two_way_value_column_with_no_value_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_twoway(
            TwoWayRequest(
                values=_column("y", [np.nan] * len(NIST_VALUES)),
                factor_a=_group("lab", NIST_LABS),
                factor_b=_group("material", NIST_MATERIALS),
            )
        )
    assert refused.value.code == "SINGLE_GROUP"


def test_a_repeated_measure_with_a_single_complete_subject_is_refused() -> None:
    rows = ((1.0, 2.0), (2.0, None), (None, 3.0))
    columns = _wide_columns(rows, ("a", "b"))
    with pytest.raises(ToolError) as refused:
        run_repeated(RepeatedMeasuresRequest(measures=columns))
    assert refused.value.code == "TOO_FEW_ROWS"


def test_a_two_way_row_without_a_value_is_listed_not_dropped() -> None:
    values = np.append(np.array(NIST_VALUES), np.nan)
    result = run_twoway(
        TwoWayRequest(
            values=_column("strength", values),
            factor_a=_group("lab", (*NIST_LABS, "1")),
            factor_b=_group("material", (*NIST_MATERIALS, "1")),
        )
    )
    assert result.counts.n_offered == 19
    assert result.counts.n_used == 18
    assert result.counts.n_excluded == 1
    assert result.excluded[0].index == 18
    assert "missing in strength" in result.excluded[0].reason


# --------------------------------------------------------------------------- size


def test_the_two_way_interaction_rejects_at_its_level_under_no_interaction() -> None:
    a_codes, b_codes = _design_codes(2, 3, 8)
    a_effect = np.array([0.0, 1.0])
    b_effect = np.array([0.0, 0.5, 1.0])
    mean = a_effect[a_codes] + b_effect[b_codes]

    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        return mean + rng.normal(0.0, 1.0, mean.size)

    def run(values: NDArray[np.float64]) -> float | None:
        return _twoway_sums(values, a_codes, b_codes, 2, 3).ab.p_value

    share = simulated_size(draw, run, draws=4_000, alpha=0.05)
    assert_size(share)


def test_the_repeated_measures_f_rejects_at_its_level_under_the_null() -> None:
    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.normal(0.0, 1.0, (10, 3))

    def run(values: NDArray[np.float64]) -> float | None:
        return _rm_sums(values).p_value

    share = simulated_size(draw, run, draws=4_000, alpha=0.05)
    assert_size(share)


def test_friedman_rejects_at_its_level_with_three_conditions() -> None:
    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        return rng.normal(0.0, 1.0, (10, 3))

    def run(values: NDArray[np.float64]) -> float | None:
        return _friedman_sums(values).p_value

    share = simulated_size(draw, run, draws=4_000, alpha=0.05)
    assert_size(share)


# --------------------------------------------------------------------------- the registry


def test_the_registry_lists_and_runs_the_three_tools() -> None:
    assert "curvestudio.tools.stats.anova_more" in TOOL_MODULES

    two_way = TOOLS.info("twoway")
    assert (two_way.name, two_way.family, two_way.title, two_way.needs_data) == (
        "twoway",
        "means",
        "Two-way analysis of variance",
        True,
    )
    assert [(field.name, field.kind, field.required) for field in two_way.fields] == [
        ("values", "column", True),
        ("factor_a", "group", True),
        ("factor_b", "group", True),
        ("level", "option", False),
    ]

    repeated = TOOLS.info("repeated")
    assert (repeated.name, repeated.family, repeated.title) == (
        "repeated",
        "means",
        "Repeated-measures analysis of variance",
    )
    assert [(field.name, field.kind, field.required) for field in repeated.fields] == [
        ("measures", "columns", True),
        ("names", "option", False),
        ("level", "option", False),
    ]

    friedman = TOOLS.info("friedman")
    assert (friedman.name, friedman.family, friedman.title) == (
        "friedman",
        "ranks",
        "Friedman test",
    )
    for info in (two_way, repeated, friedman):
        assert all(field.description for field in info.fields)
        assert info.name in [entry.name for entry in catalogue()]

    lab = _group("lab", NIST_LABS)
    material = _group("material", NIST_MATERIALS)
    values = _column("strength", NIST_VALUES)
    request = TOOLS.build_request(
        "twoway",
        {"values": "strength", "factor_a": "lab", "factor_b": "material", "level": 0.9},
        column=lambda name: values,
        group=lambda name: lab if name == "lab" else material,
    )
    assert isinstance(request, TwoWayRequest)
    assert TOOLS.run("twoway", request).tool == "twoway"

    columns = _wide_columns(RM_WIDE, RM_NAMES)
    repeated_request = TOOLS.build_request(
        "repeated",
        {"measures": list(RM_NAMES), "names": list(RM_NAMES), "level": 0.9},
        column=lambda name: columns[RM_NAMES.index(name)],
        group=lambda name: lab,
    )
    assert isinstance(repeated_request, RepeatedMeasuresRequest)
    assert repeated_request.names == RM_NAMES
    assert TOOLS.run("repeated", repeated_request).tool == "repeated"

    friedman_columns = _matrix_columns(DC_DATA)
    friedman_request = TOOLS.build_request(
        "friedman",
        {"measures": [f"condition{j + 1}" for j in range(DC_DATA.shape[1])]},
        column=lambda name: friedman_columns[int(name.removeprefix("condition")) - 1],
        group=lambda name: lab,
    )
    assert isinstance(friedman_request, FriedmanRequest)
    assert TOOLS.run("friedman", friedman_request).tool == "friedman"
