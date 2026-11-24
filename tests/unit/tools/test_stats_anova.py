"""One-way analysis of variance, its Welch counterpart, and the Kruskal-Wallis test.

The reference numbers come from SciPy and statsmodels. Two worked examples come from the
NIST/SEMATECH e-Handbook of Statistical Methods, both pages fetched on 2026-11-24:

- ``1.3.5.4. One-Factor ANOVA`` (itl.nist.gov/div898/handbook/eda/section3/eda354.htm)
  tabulates, for the GEAR.DAT data set of 100 measurements in ten batches, the sums of
  squares 0.000729 (batch), 0.003174 (residual) and 0.003903 (total), their mean squares
  0.000081 and 0.000035, an F statistic of 2.2969 on 9 and 90 degrees of freedom, and the
  ten batch means. The data file is itl.nist.gov/div898/handbook/datasets/GEAR.DAT.
- ``1.3.5.10. Levene Test for Equality of Variances``
  (itl.nist.gov/div898/handbook/eda/section3/eda35a.htm) tabulates, for the same data set,
  a median-centred Levene statistic of W = 1.705910 on 9 and 90 degrees of freedom.

SciPy's ``alexandergovern`` is the Alexander-Govern chi-square approximation, not the Welch F
statistic; it tests the same null under variance heterogeneity but returns a different
statistic, so only its evidence is compared with the Welch test's. The size of every test is
measured on data drawn from its null hypothesis with a fixed seed, and the classical F is
shown to leave its band on the unequal-variance data where Welch's test holds its level.
"""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.oneway import anova_oneway

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    ExcludedRow,
    ToolResult,
    Verdict,
    assert_size,
    p_text,
    simulated_size,
)
from curvestudio.tools.stats.anova import (
    DUNNETT_SEED,
    AnovaRequest,
    KruskalRequest,
    _classical_f,
    _kruskal_h,
    _levene_test,
    _summarise,
    _tukey_p_values,
    _welch_f,
    run_anova,
    run_kruskal,
)
from curvestudio.tools.stats.registry import TOOL_MODULES, TOOLS, catalogue

P_ATOL = 1e-6
"""The tolerance on a p-value the reference implementation also computes."""
STAT_RTOL = 1e-8
"""The tolerance on an estimate or a statistic the reference implementation also computes."""

GEAR_VALUES = (
    1.006,
    0.996,
    0.998,
    1.000,
    0.992,
    0.993,
    1.002,
    0.999,
    0.994,
    1.000,
    0.998,
    1.006,
    1.000,
    1.002,
    0.997,
    0.998,
    0.996,
    1.000,
    1.006,
    0.988,
    0.991,
    0.987,
    0.997,
    0.999,
    0.995,
    0.994,
    1.000,
    0.999,
    0.996,
    0.996,
    1.005,
    1.002,
    0.994,
    1.000,
    0.995,
    0.994,
    0.998,
    0.996,
    1.002,
    0.996,
    0.998,
    0.998,
    0.982,
    0.990,
    1.002,
    0.984,
    0.996,
    0.993,
    0.980,
    0.996,
    1.009,
    1.013,
    1.009,
    0.997,
    0.988,
    1.002,
    0.995,
    0.998,
    0.981,
    0.996,
    0.990,
    1.004,
    0.996,
    1.001,
    0.998,
    1.000,
    1.018,
    1.010,
    0.996,
    1.002,
    0.998,
    1.000,
    1.006,
    1.000,
    1.002,
    0.996,
    0.998,
    0.996,
    1.002,
    1.006,
    1.002,
    0.998,
    0.996,
    0.995,
    0.996,
    1.004,
    1.004,
    0.998,
    0.999,
    0.991,
    0.991,
    0.995,
    0.984,
    0.994,
    0.997,
    0.997,
    0.991,
    0.998,
    1.004,
    0.997,
)
"""The GEAR.DAT gear diameters, in the file's order."""

GEAR_BATCHES = tuple(
    str(batch)
    for batch in (
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    )
)
"""The GEAR.DAT batches, in the file's order, as the group column's labels."""

GEAR_MEANS = (
    0.99800,
    0.99910,
    0.99540,
    0.99820,
    0.99190,
    0.99880,
    1.00150,
    1.00040,
    0.99830,
    0.99480,
)
"""The ten batch means the NIST page tabulates."""


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


def _data(samples: object, labels: object) -> tuple[Column, GroupColumn]:
    """A value column and a group column, each group's labels over its own values."""
    arrays = [np.asarray(sample, dtype=np.float64) for sample in samples]
    values = np.concatenate(arrays)
    names = [str(label) for label in labels]
    group_labels = [
        name for name, array in zip(names, arrays, strict=True) for _ in range(array.size)
    ]
    return _column("y", values), _group("g", group_labels)


def _default_labels(samples: object) -> list[str]:
    return [f"group{index + 1}" for index in range(len(samples))]  # type: ignore[arg-type]


def _anova(samples: object, labels: object | None = None, **options: object) -> ToolResult:
    names = _default_labels(samples) if labels is None else labels
    values, groups = _data(samples, names)
    return run_anova(AnovaRequest(values=values, groups=groups, **options))  # type: ignore[arg-type]


def _kruskal(samples: object, labels: object | None = None, **options: object) -> ToolResult:
    names = _default_labels(samples) if labels is None else labels
    values, groups = _data(samples, names)
    return run_kruskal(KruskalRequest(values=values, groups=groups, **options))  # type: ignore[arg-type]


def _three(seed: int = 0) -> tuple[NDArray[np.float64], ...]:
    """Three normal groups of unequal sizes, drawn from one seed."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, 10), rng.normal(1.0, 2.0, 20), rng.normal(2.0, 1.5, 15)


# --------------------------------------------------------------------------- analysis of variance


def test_the_one_way_f_matches_scipy_and_statsmodels() -> None:
    samples = _three(1)
    result = _anova(samples)
    (test, welch, *_comparisons) = result.tests

    reference = scipy_stats.f_oneway(*samples)
    assert test.name == "One-way analysis of variance"
    assert test.method == "One-way analysis of variance, F test"
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == 2.0
    assert test.df_denominator == 42.0
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.p_value_text == p_text(test.p_value)
    assert test.null_hypothesis == "The means of every group are equal."

    values = np.concatenate(samples)
    labels = [f"group{index + 1}" for index in range(3) for _ in range(samples[index].size)]
    frame = pd.DataFrame({"y": values, "g": labels})
    table = anova_lm(ols("y ~ C(g)", data=frame).fit())
    between, within = table.loc["C(g)"], table.loc["Residual"]
    anova_table = result.tables[0]
    rows = {(row[0]): row for row in anova_table.rows}
    assert rows["Between groups"][1] == pytest.approx(float(between["sum_sq"]), rel=STAT_RTOL)
    assert rows["Between groups"][2] == pytest.approx(float(between["df"]), rel=STAT_RTOL)
    assert rows["Between groups"][3] == pytest.approx(float(between["mean_sq"]), rel=STAT_RTOL)
    assert rows["Between groups"][4] == pytest.approx(float(between["F"]), rel=STAT_RTOL)
    assert rows["Between groups"][5] == pytest.approx(float(between["PR(>F)"]), abs=P_ATOL)
    assert rows["Within groups"][1] == pytest.approx(float(within["sum_sq"]), rel=STAT_RTOL)
    assert rows["Within groups"][2] == pytest.approx(float(within["df"]), rel=STAT_RTOL)
    assert rows["Within groups"][3] == pytest.approx(float(within["mean_sq"]), rel=STAT_RTOL)
    assert rows["Total"][1] == pytest.approx(
        float(between["sum_sq"]) + float(within["sum_sq"]), rel=STAT_RTOL
    )
    assert welch.name == "Welch's analysis of variance"


def test_the_welch_f_matches_statsmodels_welch_anova() -> None:
    samples = _three(2)
    result = _anova(samples)
    welch = result.tests[1]

    reference = anova_oneway(samples, use_var="unequal")
    assert welch.method == "Welch's analysis of variance, F test"
    assert welch.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert welch.df == pytest.approx(float(reference.df_num), rel=STAT_RTOL)
    assert welch.df_denominator == pytest.approx(float(reference.df_denom), rel=STAT_RTOL)
    assert welch.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert welch.p_value_text == p_text(welch.p_value)
    assert welch.effect is not None


def test_the_alexander_govern_test_reaches_the_same_evidence_as_welch() -> None:
    # scipy's alexandergovern is a chi-square approximation, not the Welch F, so its
    # statistic and p-value are not numerically equal to Welch's; on data that separate the
    # means, the two heteroscedastic tests point the same way.
    rng = np.random.default_rng(3)
    samples = (
        rng.normal(0.0, 1.0, 25),
        rng.normal(1.0, 2.0, 25),
        rng.normal(2.0, 2.0, 25),
    )
    welch = _anova(samples).tests[1]
    govern = scipy_stats.alexandergovern(*samples)
    assert (float(govern.pvalue) < 0.05) == (welch.p_value < 0.05)
    assert float(govern.pvalue) < 0.05
    assert float(govern.statistic) != pytest.approx(welch.statistic, rel=1e-3)


def test_eta_and_omega_squared_follow_their_formulas() -> None:
    samples = _three(4)
    result = _anova(samples)
    eta, omega = result.estimates

    values = np.concatenate(samples)
    grand = float(np.mean(values))
    between = sum(sample.size * (float(np.mean(sample)) - grand) ** 2 for sample in samples)
    within = sum(float(np.sum((sample - np.mean(sample)) ** 2)) for sample in samples)
    total = between + within
    group_count = len(samples)
    count = values.size
    mean_square_within = within / (count - group_count)
    expected_eta = between / total
    expected_omega = (between - (group_count - 1) * mean_square_within) / (
        total + mean_square_within
    )
    assert eta.label == "Eta squared"
    assert eta.value == pytest.approx(expected_eta, rel=STAT_RTOL)
    assert eta.lower is None and eta.upper is None
    assert omega.label == "Omega squared"
    assert omega.value == pytest.approx(expected_omega, rel=STAT_RTOL)
    assert omega.lower is None and omega.upper is None


def test_tukey_matches_tukey_hsd_and_its_intervals() -> None:
    samples = _three(5)
    result = _anova(samples)
    comparisons = result.tests[2:]
    reference = scipy_stats.tukey_hsd(*samples)
    lower, upper = reference.confidence_interval(0.95)

    assert len(comparisons) == 3
    means = [float(np.mean(sample)) for sample in samples]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for test, (first, second) in zip(comparisons, pairs, strict=True):
        assert test.name == f"Tukey HSD, group{first + 1} less group{second + 1}"
        assert test.p_value == pytest.approx(float(reference.pvalue[first, second]), abs=P_ATOL)
        assert test.adjusted_p == pytest.approx(test.p_value, abs=0.0)
        assert test.adjustment == "Tukey HSD"
        assert test.effect is not None
        assert test.effect.value == pytest.approx(means[first] - means[second], rel=STAT_RTOL)
        assert test.effect.lower == pytest.approx(float(lower[first, second]), rel=STAT_RTOL)
        assert test.effect.upper == pytest.approx(float(upper[first, second]), rel=STAT_RTOL)


def test_dunnett_matches_scipy_dunnett_against_the_control() -> None:
    samples = _three(6)
    result = _anova(samples, post_hoc="dunnett", control="group1")
    comparisons = result.tests[2:]

    reference = scipy_stats.dunnett(
        samples[1], samples[2], control=samples[0], rng=np.random.default_rng(DUNNETT_SEED)
    )
    lower, upper = reference.confidence_interval(0.95)
    assert [test.name for test in comparisons] == [
        "Dunnett's test, group2 against group1",
        "Dunnett's test, group3 against group1",
    ]
    for index, test in enumerate(comparisons):
        assert test.statistic == pytest.approx(float(reference.statistic[index]), rel=STAT_RTOL)
        assert test.p_value == pytest.approx(float(reference.pvalue[index]), abs=P_ATOL)
        assert test.adjusted_p == pytest.approx(test.p_value, abs=0.0)
        assert test.adjustment == "Dunnett"
        assert test.df == 42.0
        assert test.effect is not None
        assert test.effect.value == pytest.approx(
            float(np.mean(samples[index + 1]) - np.mean(samples[0])), rel=STAT_RTOL
        )
        assert test.effect.lower == pytest.approx(float(lower[index]), rel=STAT_RTOL)
        assert test.effect.upper == pytest.approx(float(upper[index]), rel=STAT_RTOL)


def test_two_groups_have_no_post_hoc_comparisons() -> None:
    rng = np.random.default_rng(7)
    result = _anova((rng.normal(0.0, 1.0, 12), rng.normal(1.0, 1.0, 14)))

    assert [test.name for test in result.tests] == [
        "One-way analysis of variance",
        "Welch's analysis of variance",
    ]
    assert all(test.adjusted_p is None for test in result.tests)


# --------------------------------------------------------------------------- assumptions


def test_each_group_gets_a_normality_check_and_one_levene_check() -> None:
    samples = _three(8)
    result = _anova(samples)

    assert [check.id for check in result.assumptions] == [
        "normality_1",
        "normality_2",
        "normality_3",
        "equal_variances",
    ]
    normality = result.assumptions[0]
    assert normality.statistic is not None and normality.statistic.startswith("W = ")
    assert normality.threshold == "p >= 0.05 pass; 0.001 <= p < 0.05 caution; p < 0.001 fail"
    levene = result.assumptions[-1]
    assert "Levene's test, centred on the median" in levene.evidence

    skewed = _anova(
        (
            np.random.default_rng(9).exponential(1.0, 40),
            np.random.default_rng(10).exponential(1.0, 40),
            np.random.default_rng(11).exponential(1.0, 40),
        )
    )
    assert skewed.assumptions[0].verdict is Verdict.FAIL


def test_levene_follows_scipy_and_the_p_thresholds() -> None:
    rng = np.random.default_rng(12)
    narrow = rng.normal(0.0, 0.1, 20)
    wide = rng.normal(0.0, 5.0, 20)
    middle = rng.normal(0.0, 1.0, 20)
    check = _anova((narrow, middle, wide)).assumptions[-1]

    reference = scipy_stats.levene(narrow, middle, wide, center="median")
    assert check.statistic == f"W = {float(reference.statistic):.3g}"
    assert check.verdict is Verdict.FAIL
    assert check.next_step is not None and "Welch" in check.next_step


# --------------------------------------------------------------------------- the NIST examples


def test_the_nist_gear_example_reproduces_its_anova_table() -> None:
    values = _column("diameter", GEAR_VALUES)
    groups = _group("batch", GEAR_BATCHES)
    result = run_anova(AnovaRequest(values=values, groups=groups))

    rows = {(row[0]): row for row in result.tables[0].rows}
    assert rows["Between groups"][1] == pytest.approx(0.000729, abs=5e-7)
    assert rows["Between groups"][2] == pytest.approx(9.0, abs=0.0)
    assert rows["Between groups"][3] == pytest.approx(0.000081, abs=5e-7)
    assert rows["Between groups"][4] == pytest.approx(2.2969, abs=1e-4)
    assert rows["Within groups"][1] == pytest.approx(0.003174, abs=5e-7)
    assert rows["Within groups"][2] == pytest.approx(90.0, abs=0.0)
    assert rows["Within groups"][3] == pytest.approx(0.000035, abs=5e-7)
    assert rows["Total"][1] == pytest.approx(0.003903, abs=5e-7)
    assert rows["Total"][2] == pytest.approx(99.0, abs=0.0)

    test = result.tests[0]
    assert test.statistic == pytest.approx(2.2969, abs=1e-4)
    assert test.df == 9.0 and test.df_denominator == 90.0
    # The page concludes significance at the 0.05 level against the critical value 1.9856.
    assert test.p_value is not None and 0.0 < test.p_value < 0.05

    summary = result.tables[1]
    means = {row[0]: row[2] for row in summary.rows}
    for batch, expected in zip(range(1, 11), GEAR_MEANS, strict=True):
        assert means[str(batch)] == pytest.approx(expected, abs=5e-6)


def test_the_nist_gear_example_reproduces_levenes_statistic() -> None:
    values = _column("diameter", GEAR_VALUES)
    groups = _group("batch", GEAR_BATCHES)
    result = run_anova(AnovaRequest(values=values, groups=groups))
    check = result.assumptions[-1]

    # The check's statistic text is rounded to three significant digits, so the test reads
    # the full-precision W the tool computed, and the check's words as the result shows them.
    grouped = [
        _summarise(label, np.asarray(GEAR_VALUES)[np.asarray(GEAR_BATCHES) == label])
        for label in sorted(set(GEAR_BATCHES), key=int)
    ]
    statistic, p_value = _levene_test(grouped)
    assert statistic == pytest.approx(1.705910, abs=1e-5)
    assert p_value == pytest.approx(
        float(
            scipy_stats.levene(
                *[
                    np.asarray(GEAR_VALUES)[np.asarray(GEAR_BATCHES) == label]
                    for label in sorted(set(GEAR_BATCHES), key=int)
                ],
                center="median",
            ).pvalue
        ),
        abs=P_ATOL,
    )
    assert check.statistic == f"W = {statistic:.3g}"
    assert check.verdict is Verdict.PASS
    assert "Levene's test, centred on the median" in check.evidence


# --------------------------------------------------------------------------- Kruskal-Wallis


def test_kruskal_wallis_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(13)
    samples = (
        np.round(rng.normal(0.0, 1.0, 12), 1),
        np.round(rng.normal(0.5, 1.0, 16), 1),
        np.round(rng.normal(1.0, 1.0, 14), 1),
    )
    result = _kruskal(samples)
    test = result.tests[0]
    reference = scipy_stats.kruskal(*samples)

    assert test.name == "Kruskal–Wallis test"  # noqa: RUF001
    assert test.method == "Kruskal–Wallis test, chi-square approximation with ties"  # noqa: RUF001
    assert test.statistic == pytest.approx(float(reference.statistic), rel=STAT_RTOL)
    assert test.df == 2.0
    assert test.p_value == pytest.approx(float(reference.pvalue), abs=P_ATOL)
    assert test.p_value_text == p_text(test.p_value)
    assert test.null_hypothesis == "The groups' mean ranks are equal."
    assert test.effect is not None and test.effect.label == "Epsilon squared"


def test_epsilon_squared_is_h_over_n_minus_one() -> None:
    rng = np.random.default_rng(14)
    samples = (rng.normal(0.0, 1.0, 9), rng.normal(1.0, 1.0, 12), rng.normal(2.0, 1.0, 15))
    result = _kruskal(samples)
    test, *_rest = result.tests
    (epsilon,) = result.estimates

    count = sum(sample.size for sample in samples)
    assert epsilon.value == pytest.approx(test.statistic / (count - 1), rel=STAT_RTOL)
    assert test.effect == epsilon


def test_dunns_test_follows_its_formula_and_holm() -> None:
    rng = np.random.default_rng(15)
    samples = (
        np.round(rng.normal(0.0, 1.0, 10), 1),
        np.round(rng.normal(0.5, 1.0, 12), 1),
        np.round(rng.normal(1.5, 1.0, 14), 1),
    )
    result = _kruskal(samples)
    comparisons = result.tests[1:]

    combined = np.concatenate(samples)
    ranks = scipy_stats.rankdata(combined)
    count = combined.size
    splits = np.split(ranks, np.cumsum([sample.size for sample in samples])[:-1])
    mean_ranks = [float(chunk.mean()) for chunk in splits]
    _, tied = np.unique(combined, return_counts=True)
    tie_sum = float(np.sum(tied.astype(float) ** 3 - tied))
    sigma = count * (count + 1) / 12.0 - tie_sum / (12.0 * (count - 1))

    raw = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for test, (first, second) in zip(comparisons, pairs, strict=True):
        scale = math.sqrt(sigma * (1.0 / samples[first].size + 1.0 / samples[second].size))
        expected = (mean_ranks[first] - mean_ranks[second]) / scale
        assert test.statistic == pytest.approx(expected, rel=STAT_RTOL)
        expected_p = float(2.0 * scipy_stats.norm.sf(abs(expected)))
        raw.append(expected_p)
        assert test.p_value == pytest.approx(expected_p, abs=P_ATOL)
        assert test.adjustment == "Holm"
        assert test.effect is not None
        assert test.effect.value == pytest.approx(
            mean_ranks[first] - mean_ranks[second], rel=STAT_RTOL
        )
        assert test.effect.lower is not None and test.effect.upper is not None

    order = np.argsort(raw)
    family = len(raw)
    running = 0.0
    adjusted = [0.0] * family
    for step, index in enumerate(order, start=1):
        running = max(running, (family - step + 1) * raw[index])
        adjusted[index] = min(1.0, running)
    for test, expected in zip(comparisons, adjusted, strict=True):
        assert test.adjusted_p == pytest.approx(expected, abs=P_ATOL)


def test_the_shapes_assumption_is_not_applicable_and_points_at_the_box_plot() -> None:
    result = _kruskal(_three(16))
    (shapes,) = result.assumptions

    assert shapes.id == "equal_shapes"
    assert shapes.verdict is Verdict.NOT_APPLICABLE
    assert "look at the box plot" in shapes.evidence.lower()
    assert "look at the box plot" in (shapes.next_step or "").lower()


# --------------------------------------------------------------------------- the tables and plots


def test_the_result_carries_the_anova_table_the_summary_and_one_box_plot() -> None:
    samples = _three(17)
    result = _anova(samples)
    anova_table, summary = result.tables

    assert anova_table.title == "Analysis of variance"
    assert [column.label for column in anova_table.columns] == [
        "Source",
        "SS",
        "df",
        "MS",
        "F",
        "p",
    ]
    assert [row[0] for row in anova_table.rows] == [
        "Between groups",
        "Within groups",
        "Total",
    ]
    assert summary.title == "Summary by group"
    assert [row[0] for row in summary.rows] == ["group1", "group2", "group3"]
    (plot,) = result.plots
    assert plot.kind == "box"
    assert [box.label for box in plot.boxes] == ["group1", "group2", "group3"]


def test_the_ranks_by_group_table_holds_each_groups_mean_rank() -> None:
    rng = np.random.default_rng(18)
    samples = (
        np.round(rng.normal(0.0, 1.0, 10), 1),
        np.round(rng.normal(0.5, 1.0, 12), 1),
        np.round(rng.normal(1.0, 1.0, 14), 1),
    )
    result = _kruskal(samples)
    (ranks_table,) = result.tables

    assert ranks_table.title == "Ranks by group"
    assert [column.label for column in ranks_table.columns] == [
        "Group",
        "n",
        "Mean rank",
        "Median",
        "Q1",
        "Q3",
    ]
    ranks = scipy_stats.rankdata(np.concatenate(samples))
    splits = np.split(ranks, np.cumsum([sample.size for sample in samples])[:-1])
    for row, sample, chunk in zip(ranks_table.rows, samples, splits, strict=True):
        assert row[1] == sample.size
        assert row[2] == pytest.approx(float(chunk.mean()), rel=STAT_RTOL)
        assert row[3] == pytest.approx(float(np.quantile(sample, 0.5)), rel=STAT_RTOL)
    (plot,) = result.plots
    assert plot.kind == "box"


# --------------------------------------------------------------------------- excluded rows


def test_a_row_without_a_value_or_a_label_is_listed_not_dropped() -> None:
    # The data layer leaves a blank label out of the group column's rows, so position 3 has
    # no label; position 1 holds a missing value.
    values = _column("y", [1.0, float("nan"), 3.0, 4.0, 5.0, 6.0])
    groups = _group("g", ["a", "a", "a", "b", "b"], rows=[0, 1, 2, 4, 5])
    result = run_anova(AnovaRequest(values=values, groups=groups))

    assert result.counts.n_offered == 6
    assert result.counts.n_used == 4
    assert result.counts.n_excluded == 2
    assert result.excluded == (
        ExcludedRow(index=1, reason="missing in y"),
        ExcludedRow(index=3, reason="no group label in g"),
    )
    assert result.excluded_unlisted == 0


def test_more_than_two_hundred_excluded_rows_are_counted_not_listed() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [float("nan")] * 255
    groups = _group("g", ["a"] * 3 + ["b"] * 3, rows=range(6))
    result = run_kruskal(KruskalRequest(values=_column("y", values), groups=groups))

    assert result.counts.n_used == 6 and result.counts.n_excluded == 255
    assert len(result.excluded) == 200
    assert result.excluded_unlisted == 55


# --------------------------------------------------------------------------- refusals


def test_a_single_group_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        run_anova(
            AnovaRequest(
                values=_column("y", np.arange(12.0)),
                groups=_group("g", ["a"] * 12),
            )
        )
    assert refused.value.code == "SINGLE_GROUP"

    with pytest.raises(ToolError) as refused_kruskal:
        run_kruskal(
            KruskalRequest(
                values=_column("y", np.arange(12.0)),
                groups=_group("g", ["a"] * 12),
            )
        )
    assert refused_kruskal.value.code == "SINGLE_GROUP"


def test_a_group_of_one_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _anova(([1.0, 2.0], [3.0, 4.0], [5.0]))
    assert refused.value.code == "GROUP_OF_ONE"
    assert "group3" in refused.value.message


def test_a_group_with_no_usable_value_is_refused() -> None:
    values = _column("y", [1.0, 2.0, 3.0, 4.0, 5.0, float("nan"), float("nan")])
    labels = ["a", "a", "b", "b", "b", "c", "c"]
    with pytest.raises(ToolError) as refused:
        run_anova(AnovaRequest(values=values, groups=_group("g", labels)))
    assert refused.value.code == "TOO_FEW_ROWS"
    assert "c" in refused.value.message


def test_a_constant_group_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _anova(([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]))
    assert refused.value.code == "CONSTANT_COLUMN"


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_a_confidence_level_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        _anova(_three(19), level=level)
    assert refused.value.code == "INVALID_OPTION"


def test_dunnett_needs_a_control_that_is_one_of_the_levels() -> None:
    samples = _three(20)
    for control in (None, "missing"):
        with pytest.raises(ToolError) as refused:
            _anova(samples, post_hoc="dunnett", control=control)
        assert refused.value.code == "LEVELS_REQUIRED"


def test_tukeys_test_takes_no_control_group() -> None:
    with pytest.raises(ToolError) as refused:
        _anova(_three(21), post_hoc="tukey", control="group1")
    assert refused.value.code == "INVALID_OPTION"


def test_a_request_of_another_tool_is_refused() -> None:
    values, groups = _data(_three(22), ["group1", "group2", "group3"])
    with pytest.raises(ToolError) as refused:
        run_anova(KruskalRequest(values=values, groups=groups))
    assert refused.value.code == "INVALID_OPTION"

    with pytest.raises(ToolError) as refused_kruskal:
        run_kruskal(AnovaRequest(values=values, groups=groups))
    assert refused_kruskal.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- the result's words


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    anova_result = _anova(_three(23))
    kruskal_result = _kruskal(_three(24))
    for result in (anova_result, kruskal_result):
        texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
        for check in result.assumptions:
            texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
        for test in result.tests:
            texts += [test.name, test.method, test.null_hypothesis]
        for text in texts:
            assert not claims_proof(text), text

    assert "Welch" in anova_result.methods_fragment
    assert "Eta squared" in anova_result.methods_fragment
    assert "Tukey" in anova_result.interpretation
    assert "Holm" in kruskal_result.methods_fragment
    assert "box plot" in kruskal_result.methods_fragment


def test_the_interpretation_names_the_statistics_and_the_effect_sizes() -> None:
    result = _anova(_three(25))
    interpretation = result.interpretation

    assert p_text(result.tests[0].p_value) in interpretation
    assert "eta squared" in interpretation.lower()
    assert "omega squared" in interpretation.lower()
    assert "Welch's analysis of variance" in interpretation
    assert len(interpretation.split(". ")) <= 3


# --------------------------------------------------------------------------- properties


def test_adding_a_constant_to_every_value_changes_nothing_in_f() -> None:
    # A shift of the whole sample moves the grand mean with it, so every deviation from a
    # group mean is unchanged and the F statistic is unchanged. A shift of ONE group would
    # move that group's mean against the others and would change F, so the invariance the
    # plan's property can mean is the one tested here.
    samples = _three(26)
    shifted = tuple(sample + 7.5 for sample in samples)
    reference = _anova(samples)
    moved = _anova(shifted)

    assert moved.tests[0].statistic == pytest.approx(reference.tests[0].statistic, rel=STAT_RTOL)
    assert moved.tests[0].p_value == pytest.approx(reference.tests[0].p_value, rel=STAT_RTOL)
    assert moved.tests[0].df == reference.tests[0].df
    assert moved.tests[0].df_denominator == reference.tests[0].df_denominator


def test_kruskal_wallis_is_invariant_under_a_shift_of_every_value() -> None:
    samples = _three(27)
    shifted = tuple(sample + 3.0 for sample in samples)
    reference = _kruskal(samples)
    moved = _kruskal(shifted)

    assert moved.tests[0].statistic == pytest.approx(reference.tests[0].statistic, rel=STAT_RTOL)
    assert moved.tests[0].p_value == pytest.approx(reference.tests[0].p_value, rel=STAT_RTOL)


# --------------------------------------------------------------------------- size


def _groups_of(samples: tuple[NDArray[np.float64], ...]) -> list:
    return [_summarise(f"group{index + 1}", sample) for index, sample in enumerate(samples)]


def _size_classical(samples: tuple[NDArray[np.float64], ...]) -> float:
    return _classical_f(_groups_of(samples)).p_value


def _size_welch(samples: tuple[NDArray[np.float64], ...]) -> float:
    return _welch_f(_groups_of(samples)).p_value


def _size_kruskal(samples: tuple[NDArray[np.float64], ...]) -> float:
    return _kruskal_h(_groups_of(samples)).p_value


def _size_tukey(samples: tuple[NDArray[np.float64], ...]) -> float:
    groups = _groups_of(samples)
    count = sum(group.n for group in groups)
    df_within = float(count - len(groups))
    ms_within = sum((group.n - 1) * group.sd**2 for group in groups) / df_within
    return min(_tukey_p_values(groups, ms_within, df_within))


def test_the_one_way_f_rejects_at_its_level_with_unequal_group_sizes() -> None:
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], ...]:
        return rng.normal(0.0, 1.0, 10), rng.normal(0.0, 1.0, 15), rng.normal(0.0, 1.0, 20)

    share = simulated_size(draw, _size_classical, draws=4_000, alpha=0.05)
    assert_size(share)


def test_welch_rejects_at_its_level_with_variances_in_ratio_one_four_nine() -> None:
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], ...]:
        return rng.normal(0.0, 1.0, 10), rng.normal(0.0, 2.0, 20), rng.normal(0.0, 3.0, 30)

    share = simulated_size(draw, _size_welch, draws=4_000, alpha=0.05)
    assert_size(share)


def test_the_classical_f_leaves_the_band_on_the_unequal_variance_data() -> None:
    # A documented control: pooling the variances weights the large-variance large group far
    # too heavily, so the classical F rejects far less often than its 5 % level allows.
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], ...]:
        return rng.normal(0.0, 1.0, 10), rng.normal(0.0, 2.0, 20), rng.normal(0.0, 3.0, 30)

    share = simulated_size(draw, _size_classical, draws=4_000, alpha=0.05)
    assert share < 0.038, f"the classical F measured {share:.1%}, inside the 3.8-6.2 % band"
    with pytest.raises(AssertionError):
        assert_size(share)


def test_kruskal_wallis_rejects_at_its_level_with_ties_and_unequal_sizes() -> None:
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], ...]:
        return (
            np.round(rng.normal(0.0, 1.0, 12), 1),
            np.round(rng.normal(0.0, 1.0, 16), 1),
            np.round(rng.normal(0.0, 1.0, 20), 1),
        )

    share = simulated_size(draw, _size_kruskal, draws=4_000, alpha=0.05)
    assert_size(share)


def test_tukeys_family_wise_error_is_about_five_percent() -> None:
    def draw(rng: np.random.Generator) -> tuple[NDArray[np.float64], ...]:
        return rng.normal(0.0, 1.0, 10), rng.normal(0.0, 1.0, 15), rng.normal(0.0, 1.0, 20)

    share = simulated_size(draw, _size_tukey, draws=4_000, alpha=0.05)
    assert 0.035 <= share <= 0.065, f"Tukey's family-wise error measured {share:.1%}"
    assert_size(share)


# --------------------------------------------------------------------------- the registry


def test_the_registry_lists_and_runs_both_tools() -> None:
    assert "curvestudio.tools.stats.anova" in TOOL_MODULES

    anova_info = TOOLS.info("anova")
    assert (anova_info.name, anova_info.family, anova_info.title, anova_info.needs_data) == (
        "anova",
        "means",
        "Analysis of variance",
        True,
    )
    assert [(field.name, field.kind, field.required) for field in anova_info.fields] == [
        ("values", "column", True),
        ("groups", "group", True),
        ("level", "option", False),
        ("post_hoc", "option", False),
        ("control", "option", False),
    ]
    assert all(field.description for field in anova_info.fields)

    kruskal_info = TOOLS.info("kruskal")
    assert (kruskal_info.name, kruskal_info.family, kruskal_info.title) == (
        "kruskal",
        "ranks",
        "Kruskal–Wallis test",  # noqa: RUF001
    )
    assert [field.name for field in kruskal_info.fields] == ["values", "groups", "level"]
    assert all(field.description for field in kruskal_info.fields)
    assert "anova" in [entry.name for entry in catalogue()]
    assert "kruskal" in [entry.name for entry in catalogue()]

    values, groups = _data(_three(28), ["group1", "group2", "group3"])
    request = TOOLS.build_request(
        "anova",
        {"values": "y", "groups": "g", "level": 0.9, "control": "group1", "post_hoc": "dunnett"},
        column=lambda name: values,
        group=lambda name: groups,
    )
    assert isinstance(request, AnovaRequest)
    assert request.post_hoc == "dunnett" and request.control == "group1"
    assert TOOLS.run("anova", request).tool == "anova"

    kruskal_request = TOOLS.build_request(
        "kruskal",
        {"values": "y", "groups": "g"},
        column=lambda name: values,
        group=lambda name: groups,
    )
    assert isinstance(kruskal_request, KruskalRequest)
    assert TOOLS.run("kruskal", kruskal_request).tool == "kruskal"
