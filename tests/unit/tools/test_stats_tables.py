"""The numbers behind the tables and plots every statistical tool shares.

Each helper is checked against what it must reproduce: Holm's adjustment against
statsmodels, the confidence interval against SciPy's t quantile, the quartiles against
values worked out by hand. These helpers are what the tools' tables and box plots are made
of, so a fault here would show in every family.
"""

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from curvestudio.tools import ToolError

# The model is called Test by its contract. Under that name pytest would take it for a class
# of tests, so it is imported here under another.
from curvestudio.tools.stats import Test as StatsTest
from curvestudio.tools.stats import (
    box_plot,
    confidence_interval,
    group_summary,
    group_table,
    holm,
    mean_ci,
    p_text,
    quantiles,
    stride,
    table,
)
from curvestudio.tools.stats._display import MAX_PLOT_POINTS
from curvestudio.tools.stats._tables import GroupSummary


def _test(p: float | None) -> StatsTest:
    return StatsTest(
        name=f"p = {p}",
        method="A toy test, two-sided",
        null_hypothesis="no difference",
        statistic=None if p is None else 1.0,
        p_value=p,
        p_value_text=p_text(p),
        reason=None if p is not None else "the toy had too few rows",
    )


# --------------------------------------------------------------------------- Holm


def test_holm_steps_down_and_names_itself() -> None:
    adjusted = holm([_test(p) for p in (0.01, 0.04, 0.03)])
    assert [test.adjusted_p for test in adjusted] == pytest.approx([0.03, 0.06, 0.06])
    assert [test.adjustment for test in adjusted] == ["Holm"] * 3
    # The adjusted p is added; what the test said about itself stays as it was.
    assert [test.p_value for test in adjusted] == [0.01, 0.04, 0.03]
    assert [test.name for test in adjusted] == ["p = 0.01", "p = 0.04", "p = 0.03"]


@pytest.mark.parametrize(
    "ps",
    [
        [0.01, 0.04, 0.03],
        [0.9, 0.95],
        [0.02, 0.02, 0.5],
        [0.5],
        [0.001, 0.2, 0.19, 0.9, 0.03],
        [0.3, 0.0],
        [0.0, 0.0],
    ],
)
def test_holm_agrees_with_statsmodels(ps: list[float]) -> None:
    expected = multipletests(ps, method="holm")[1]
    found = [test.adjusted_p for test in holm([_test(p) for p in ps])]
    assert found == pytest.approx(list(expected))


def test_holm_leaves_a_test_without_a_p_value_as_it_is() -> None:
    missing, other = _test(None), _test(0.02)

    adjusted = holm([missing, other])

    assert adjusted[0] == missing
    assert adjusted[0].adjusted_p is None and adjusted[0].adjustment is None
    assert adjusted[1].adjusted_p == pytest.approx(0.02) and adjusted[1].adjustment == "Holm"


def test_holm_returns_new_tests_and_leaves_the_ones_given_alone() -> None:
    tests = [_test(0.01), _test(0.04), _test(None)]

    adjusted = holm(tests)

    # A test that was adjusted comes back as a new test; one that could not be is the same.
    assert adjusted[0] is not tests[0] and adjusted[1] is not tests[1]
    assert adjusted[2] is tests[2]
    assert [test.adjusted_p for test in tests] == [None, None, None]
    assert [test.adjustment for test in tests] == [None, None, None]
    assert [found.p_value for found in adjusted] == [0.01, 0.04, None]
    assert adjusted[2].reason == "the toy had too few rows"
    assert holm(()) == ()


# --------------------------------------------------------------------------- intervals


@pytest.mark.parametrize(("df", "level"), [(12.0, 0.95), (17.3, 0.99), (1.0, 0.9), (2.5, 0.5)])
def test_confidence_interval_is_the_t_quantile(df: float, level: float) -> None:
    t = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    assert confidence_interval(10.0, 2.0, df, level) == pytest.approx(
        (10.0 - 2.0 * t, 10.0 + 2.0 * t)
    )


def test_confidence_interval_refuses_a_level_outside_zero_to_one() -> None:
    with pytest.raises(ToolError) as refused:
        confidence_interval(10.0, 2.0, 12.0, 1.0)
    assert refused.value.code == "INVALID_OPTION"


def test_mean_ci_gives_the_mean_with_its_interval_by_the_t_distribution() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])

    estimate = mean_ci(values, "Mean of y", 0.95)

    sd = float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1))
    assert estimate.label == "Mean of y"
    assert estimate.value == pytest.approx(3.0)
    assert estimate.se == pytest.approx(sd / np.sqrt(5.0))
    assert (estimate.lower, estimate.upper) == pytest.approx(
        confidence_interval(3.0, sd / np.sqrt(5.0), 4.0, 0.95)
    )
    assert estimate.lower == pytest.approx(3.0 - float(stats.t.ppf(0.975, 4.0)) * sd / np.sqrt(5.0))


def test_mean_ci_of_one_value_has_no_spread() -> None:
    estimate = mean_ci(np.array([7.0]), "x", 0.95)
    assert (estimate.value, estimate.se, estimate.lower, estimate.upper) == (7.0, None, None, None)


def test_mean_ci_refuses_nothing_to_average_and_a_level_outside_zero_to_one() -> None:
    with pytest.raises(ValueError, match="finite"):
        mean_ci(np.array([np.nan, np.inf]), "x", 0.95)
    with pytest.raises(ToolError) as refused:
        mean_ci(np.array([1.0]), "x", 1.0)
    assert refused.value.code == "INVALID_OPTION"


def test_quantiles_are_numpy_s_default() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 100.0])
    assert quantiles(values) == tuple(np.quantile([1.0, 2.0, 3.0, 4.0, 100.0], [0.25, 0.75]))
    assert quantiles(np.array([7.0])) == (7.0, 7.0)
    with pytest.raises(ValueError, match="finite"):
        quantiles(np.array([]))


# --------------------------------------------------------------------------- groups


def test_group_summary_reads_the_finite_values_only() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan, np.inf, -np.inf])

    summary = group_summary("a", values)

    assert isinstance(summary, GroupSummary)
    assert (summary.label, summary.count) == ("a", 5)
    assert summary.mean == pytest.approx(22.0)
    assert summary.sd == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0, 100.0], ddof=1))
    assert summary.se == pytest.approx(summary.sd / np.sqrt(5.0))
    assert (summary.median, summary.q1, summary.q3) == (3.0, 2.0, 4.0)
    assert (summary.minimum, summary.maximum) == (1.0, 100.0)


def test_group_summary_leaves_what_is_undefined_none() -> None:
    one = group_summary("one", np.array([7.0]))
    assert (one.count, one.mean, one.sd, one.se) == (1, 7.0, None, None)
    assert (one.median, one.q1, one.q3, one.minimum, one.maximum) == (7.0, 7.0, 7.0, 7.0, 7.0)

    none = group_summary("none", np.array([np.nan, np.inf]))
    assert none.count == 0
    numbers = (
        none.mean,
        none.sd,
        none.se,
        none.median,
        none.q1,
        none.q3,
        none.minimum,
        none.maximum,
    )
    assert all(value is None for value in numbers)


def test_group_table_has_a_cell_per_column_and_none_where_a_number_is_undefined() -> None:
    summaries = [
        group_summary("a", np.array([1.0, 2.0, 3.0, 4.0])),
        group_summary("b", np.array([5.0])),
        group_summary("c", np.array([np.nan])),
    ]

    found = group_table(summaries, level=0.95)

    assert found.title == "Summary by group"
    assert [column.label for column in found.columns] == [
        "Group",
        "n",
        "Mean",
        "SD",
        "SE of the mean",
        "Median",
        "Q1",
        "Q3",
        "Minimum",
        "Maximum",
    ]
    assert [column.kind for column in found.columns] == ["text", "int"] + ["number"] * 8
    assert all(len(row) == len(found.columns) for row in found.rows)

    a_row, b_row, c_row = found.rows
    assert a_row[:3] == ("a", 4, 2.5)
    assert a_row[3:5] == pytest.approx(
        (np.std([1.0, 2.0, 3.0, 4.0], ddof=1), np.std([1.0, 2.0, 3.0, 4.0], ddof=1) / 2.0)
    )
    assert a_row[5:] == (2.5, 1.75, 3.25, 1.0, 4.0)
    assert b_row == ("b", 1, 5.0, None, None, 5.0, 5.0, 5.0, 5.0, 5.0)
    assert c_row == ("c", 0) + (None,) * 8

    assert found.note is not None
    for phrase in ("n - 1", "NumPy", "t distribution", "95 %"):
        assert phrase in found.note


def test_group_table_names_the_level_it_was_asked_for() -> None:
    found = group_table([group_summary("a", np.array([1.0, 2.0]))], level=0.99)
    assert found.note is not None and "99 %" in found.note
    with pytest.raises(ToolError) as refused:
        group_table([], level=1.0)
    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- tables and plots


def test_table_refuses_a_row_that_does_not_match_its_columns() -> None:
    columns = [("group", "Group", "text"), ("n", "n", "int")]

    built = table("T", columns, [("a", 3), ("b", 4)], note="Counted by hand.")

    assert built.title == "T" and built.note == "Counted by hand."
    assert [(column.key, column.label, column.kind) for column in built.columns] == [
        ("group", "Group", "text"),
        ("n", "n", "int"),
    ]
    assert built.rows == (("a", 3), ("b", 4))
    with pytest.raises(ValueError, match="row 1"):
        table("T", columns, [("a", 3), ("b",)])
    with pytest.raises(ValueError, match="row 0"):
        table("T", columns, [("a", 3, 1.0)])


def test_box_plot_reads_the_quartiles_and_flags_a_far_point() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]

    plot = box_plot(
        "Strength by mix",
        "Strength (MPa)",
        [
            ("A", np.array(values)),
            ("empty", np.array([])),
            ("B", np.array([np.nan, 2.0, 2.0, 2.0])),
        ],
    )

    assert plot.kind == "box"
    assert plot.title == "Strength by mix" and plot.y_label == "Strength (MPa)"
    assert plot.note is None
    assert [box.label for box in plot.boxes] == ["A", "B"]
    first, second = plot.boxes
    # Quartiles of the nine values by NumPy's default sit at positions 2, 4 and 6.
    assert (first.q1, first.median, first.q3) == (3.0, 5.0, 7.0)
    # The fences are at 3 - 6 and 7 + 6; the whiskers stop at the last points inside them.
    assert (first.lower_whisker, first.upper_whisker) == (1.0, 8.0)
    assert first.outliers == (100.0,)
    assert (second.q1, second.median, second.q3) == (2.0, 2.0, 2.0)
    assert (second.lower_whisker, second.upper_whisker, second.outliers) == (2.0, 2.0, ())


def test_box_plot_stops_a_whisker_at_the_data_not_at_the_fence() -> None:
    # 1 to 12: quartiles at 3.75, 6.5 and 9.25 put the fences at -4.5 and 17.5, beyond every
    # point, so each whisker stops at the last value rather than at its fence.
    plot = box_plot("B", "y", [("even", np.arange(1.0, 13.0))])
    box = plot.boxes[0]
    assert (box.q1, box.median, box.q3) == (3.75, 6.5, 9.25)
    assert (box.lower_whisker, box.upper_whisker, box.outliers) == (1.0, 12.0, ())


def test_box_plot_sets_its_fences_at_one_and_a_half_spreads() -> None:
    # Of 1 to 8 the quartiles are 3 and 7, a spread of 4, so the fences are at -3 and 13. A
    # 12 and a 13 lie inside the upper one and a 14 does not; a fence at one spread (11) or at
    # two (15) would read them otherwise. With a -3 added below, the quartiles are 2 and 6 and
    # the lower fence is at -4: the -3 lies inside it, and outside a fence at one spread (-2).
    eight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    plot = box_plot(
        "B",
        "y",
        [
            ("12", [*eight, 12.0]),
            ("13", [*eight, 13.0]),
            ("14", [*eight, 14.0]),
            ("-3", [-3.0, *eight]),
        ],
    )

    first, second, third, fourth = plot.boxes
    assert (first.lower_whisker, first.upper_whisker, first.outliers) == (1.0, 12.0, ())
    assert (second.lower_whisker, second.upper_whisker, second.outliers) == (1.0, 13.0, ())
    assert (third.lower_whisker, third.upper_whisker, third.outliers) == (1.0, 8.0, (14.0,))
    assert (fourth.lower_whisker, fourth.upper_whisker, fourth.outliers) == (-3.0, 8.0, ())
    assert [(box.q1, box.median, box.q3) for box in plot.boxes[:3]] == [(3.0, 5.0, 7.0)] * 3
    assert (fourth.q1, fourth.median, fourth.q3) == (2.0, 4.0, 6.0)


def test_box_plot_shows_at_most_two_thousand_outliers_and_says_so() -> None:
    # A quarter of each group sits far above the rest: every one of them is beyond the upper
    # fence, so each group has 2,000 outliers.
    far = np.concatenate([np.zeros(6_000), np.full(2_000, 100.0)])

    plot = box_plot("B", "y", [("a", far), ("b", far)])

    assert sum(len(box.outliers) for box in plot.boxes) == MAX_PLOT_POINTS
    assert len(plot.boxes[0].outliers) == MAX_PLOT_POINTS and plot.boxes[1].outliers == ()
    assert plot.note == "2,000 of 4,000 outliers shown"
    just_under = box_plot("B", "y", [("a", far)])
    assert len(just_under.boxes[0].outliers) == MAX_PLOT_POINTS and just_under.note is None


def test_box_plot_hands_back_numbers_a_result_can_hold() -> None:
    rng = np.random.default_rng(20261124)
    groups = [(name, rng.normal(size=n)) for name, n in (("a", 1), ("b", 2), ("c", 500), ("d", 3))]
    plot = box_plot("B", "y", groups)
    assert [box.label for box in plot.boxes] == ["a", "b", "c", "d"]
    assert all(type(box.median) is float for box in plot.boxes)


def test_stride_returns_the_whole_array_under_the_limit() -> None:
    for size in (0, 1, 1_999, 2_000):
        values = np.arange(size, dtype=float)
        taken, note = stride(values)
        assert taken.size == size and np.array_equal(taken, values)
        assert note is None


def test_stride_takes_an_even_stride_above_the_limit_and_says_so() -> None:
    values = np.arange(53_210, dtype=float)

    taken, note = stride(values)

    assert note == "2,000 of 53,210 points shown, taken at an even stride"
    assert taken.size == MAX_PLOT_POINTS
    assert taken[0] == 0.0 and taken[-1] == 53_209.0
    gaps = np.diff(taken)
    assert gaps.max() - gaps.min() <= 1
    assert taken.dtype == np.float64 and taken.ndim == 1


def test_stride_honours_a_limit_it_is_given() -> None:
    # Positions are whole numbers: of ten points the middle one is the fifth, its value 4.
    taken, note = stride(np.arange(10.0), limit=3)
    assert taken.tolist() == [0.0, 4.0, 9.0]
    assert note == "3 of 10 points shown, taken at an even stride"
    exactly, quiet = stride(np.arange(3.0), limit=3)
    assert exactly.tolist() == [0.0, 1.0, 2.0] and quiet is None
