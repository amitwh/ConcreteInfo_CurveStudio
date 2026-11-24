"""Descriptive statistics: one column, as a whole or by group.

The numbers are checked against what they must reproduce: the sample standard deviation,
skewness, kurtosis and quartiles against SciPy, the interval for a mean against statsmodels,
and a worked example against the NIST/SEMATECH e-Handbook's ceramic-strength data. The rest
of the file holds the contract of the tool: the table and box plot it returns, the rows it
lists as left out, what it does when the data are too thin, and what survives a shift or a
rescaling of the values.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import Counts, ToolResult, build_request, run_tool
from curvestudio.tools.stats.descriptives import DescriptivesRequest, run_descriptives
from curvestudio.tools.stats.registry import TOOLS

ALL = "All rows"


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


def _run(values: Column, groups: GroupColumn | None = None, *, level: float = 0.95) -> ToolResult:
    return run_descriptives(DescriptivesRequest(values=values, groups=groups, level=level))


def _row(result: ToolResult, group: str) -> dict[str, object]:
    """One row of the descriptive table, as a mapping from column key to cell."""
    table = result.tables[0]
    keys = [column.key for column in table.columns]
    for row in table.rows:
        cells = dict(zip(keys, row, strict=True))
        if cells["group"] == group:
            return cells
    raise AssertionError(f"no row for group {group!r}: {table.rows}")


def _flag_rows(result: ToolResult) -> tuple[int, ...]:
    """The positions in the table of points beyond Tukey's fences, or none."""
    for table in result.tables:
        if table.title == "Points outside Tukey's fences":
            keys = [column.key for column in table.columns]
            return tuple(int(row[keys.index("row")]) for row in table.rows)
    return ()


def _type7(values: Sequence[float], p: float) -> float:
    """A type-7 quantile worked out from its definition, without NumPy."""
    ordered = sorted(values)
    h = (len(ordered) - 1) * p
    lower = math.floor(h)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (h - lower) * (ordered[upper] - ordered[lower])


# 1.4.2.10.1, "Background and Data", the 480 observations whose direction is longitudinal.
# The Y column only, in the order the page prints it. Tabulated data, kept six to a line.
# fmt: off
NIST_CERAMIC_Y: tuple[float, ...] = (
    608.781, 569.67, 689.556, 747.541, 618.134, 612.182,
    680.203, 607.766, 726.232, 605.38, 518.655, 589.226,
    740.447, 588.375, 666.83, 531.384, 710.272, 633.417,
    751.669, 619.06, 697.979, 632.447, 708.583, 624.256,
    624.972, 575.143, 695.07, 549.278, 769.391, 624.972,
    720.186, 587.695, 723.657, 569.207, 703.7, 613.257,
    697.626, 565.737, 714.98, 662.131, 657.712, 543.177,
    609.989, 512.394, 650.771, 611.19, 707.977, 659.982,
    712.199, 569.245, 709.631, 725.792, 703.16, 608.96,
    744.822, 586.06, 719.217, 617.441, 619.137, 592.845,
    753.333, 631.754, 677.933, 588.113, 735.919, 555.724,
    695.274, 702.411, 504.167, 631.754, 693.333, 698.254,
    625.0, 616.791, 596.667, 551.953, 640.898, 636.738,
    720.506, 571.551, 700.748, 521.667, 691.604, 587.451,
    636.738, 700.422, 731.667, 595.819, 635.079, 534.236,
    716.926, 606.188, 759.581, 575.303, 673.903, 590.628,
    736.648, 729.314, 675.957, 619.313, 729.23, 624.234,
    697.239, 651.304, 728.499, 724.175, 797.662, 583.034,
    668.53, 620.227, 815.754, 584.861, 777.392, 565.391,
    712.14, 622.506, 663.622, 628.336, 684.181, 587.145,
    629.012, 584.319, 640.193, 538.239, 644.156, 538.097,
    642.469, 595.686, 639.09, 648.935, 439.418, 583.827,
    614.664, 534.905, 537.161, 569.858, 656.773, 617.246,
    659.534, 610.337, 695.278, 584.192, 734.04, 598.853,
    687.665, 554.774, 710.858, 605.694, 701.716, 627.516,
    382.133, 574.522, 719.744, 582.682, 756.82, 563.872,
    690.978, 715.962, 670.864, 616.43, 670.308, 778.011,
    660.062, 604.255, 790.382, 571.906, 714.75, 625.925,
    716.959, 682.426, 603.363, 707.604, 713.796, 617.4,
    444.963, 689.576, 723.276, 676.678, 745.527, 563.29,
    778.333, 581.879, 723.349, 447.701, 708.229, 557.772,
    681.667, 593.537, 566.085, 632.585, 687.448, 671.35,
    597.5, 569.53, 637.41, 581.667, 755.864, 643.449,
    692.945, 581.593, 766.532, 494.122, 725.663, 620.948,
    698.818, 615.903, 760.0, 606.667, 775.272, 579.167,
    708.885, 662.51, 727.201, 436.237, 642.56, 644.223,
    690.773, 586.035, 688.333, 620.833, 743.973, 652.535,
    682.461, 593.516, 761.43, 587.451, 691.542, 570.964,
    643.392, 645.192, 697.075, 540.079, 708.229, 707.117,
    746.467, 621.779, 744.819, 585.777, 655.029, 703.98,
    715.224, 698.237, 614.417, 757.12, 761.363, 621.751,
    716.106, 472.125, 659.502, 612.7, 730.781, 583.17,
    546.928, 599.771, 734.203, 549.227, 682.051, 605.453,
    701.341, 569.599, 759.729, 637.233, 689.942, 621.774,
    769.424, 558.041, 715.286, 583.17, 776.197, 345.294,
    547.099, 570.999, 619.942, 603.232, 696.046, 595.335,
    573.109, 581.047, 638.794, 455.878, 708.193, 627.88,
    502.825, 464.085, 632.633, 596.129, 683.382, 640.371,
    684.812, 621.471, 738.161, 612.727, 671.492, 606.46,
    709.771, 571.76, 685.199, 599.304, 624.973, 579.459,
    757.363, 761.511, 633.417, 566.969, 658.754, 654.397,
    664.666, 611.719, 663.009, 577.409, 773.226, 576.731,
    708.261, 617.441, 739.086, 577.409, 667.786, 548.957,
    674.481, 623.315, 695.688, 621.761, 588.288, 553.978,
    545.61, 657.157, 752.305, 610.882, 684.523, 552.304,
    717.159, 545.303, 721.343, 651.934, 750.623, 635.24,
    776.488, 641.083, 750.623, 645.321, 600.84, 566.127,
    686.196, 647.844, 687.87, 554.815, 725.527, 620.087,
    658.796, 711.301, 690.38, 644.355, 737.144, 713.812,
    663.851, 696.707, 766.63, 589.453, 625.922, 634.468,
    694.43, 599.751, 730.217, 624.542, 700.77, 723.505,
    722.242, 674.717, 763.828, 608.539, 695.668, 612.135,
    688.887, 591.935, 531.021, 676.656, 698.915, 647.323,
    735.905, 811.97, 732.039, 603.883, 751.832, 608.643,
    618.663, 630.778, 744.845, 623.063, 690.826, 472.463,
    666.893, 645.932, 759.86, 577.176, 683.752, 567.53,
    729.591, 821.654, 730.706, 684.49, 763.124, 600.427,
    724.193, 686.023, 630.352, 628.109, 750.338, 605.214,
    752.417, 640.26, 707.899, 700.767, 715.582, 665.924,
    728.746, 555.926, 591.193, 543.299, 592.252, 511.03,
    740.833, 583.994, 786.367, 611.048, 712.386, 623.338,
    738.333, 679.585, 741.48, 665.004, 729.167, 655.86,
    795.833, 715.711, 723.502, 611.999, 718.333, 577.722,
    768.08, 615.129, 747.5, 540.316, 775.0, 711.667,
    760.599, 639.167, 758.333, 549.491, 682.5, 684.167,
    658.116, 672.153, 738.213, 594.534, 681.236, 627.65,
    704.904, 551.87, 693.623, 594.534, 624.993, 602.66,
    700.228, 585.45, 611.874, 555.724, 579.167, 574.934,
    720.872, 584.625, 690.32, 555.724, 677.933, 611.874,
    674.6, 698.254, 611.999, 748.13, 530.68, 689.942,
)
# fmt: on


# --------------------------------------------------------------------------- the contract


def test_the_result_is_the_contract_with_a_table_and_a_box_plot() -> None:
    result = _run(_column([1.0, 2.0, 3.0, 4.0]))

    assert type(result) is ToolResult
    assert result.tool == "descriptives"
    assert result.title == "Descriptive statistics of y"
    assert result.level == 0.95
    assert result.counts == Counts(n_offered=4, n_used=4, n_excluded=0)
    assert [found.title for found in result.tables] == ["Descriptive statistics"]
    assert [column.label for column in result.tables[0].columns] == [
        "Group",
        "n",
        "Missing",
        "Mean",
        "SD",
        "SE of the mean",
        "Median",
        "Q1",
        "Q3",
        "Minimum",
        "Maximum",
        "CV",
        "Skewness",
        "Excess kurtosis",
        "Mean lower",
        "Mean upper",
    ]
    assert [column.kind for column in result.tables[0].columns] == [
        "text",
        "int",
        "int",
    ] + ["number"] * 13
    assert len(result.plots) == 1 and result.plots[0].kind == "box"
    assert result.plots[0].title == "Box plot of y"
    assert [box.label for box in result.plots[0].boxes] == [ALL]


# --------------------------------------------------------------------------- references


def test_sd_skewness_kurtosis_and_quartiles_match_scipy() -> None:
    rng = np.random.default_rng(20261124)
    values = rng.normal(10.0, 3.0, size=37)

    row = _row(_run(_column(values)), ALL)

    sd = stats.tstd(values)
    assert row["mean"] == pytest.approx(float(np.mean(values)), rel=1e-12)
    assert row["sd"] == pytest.approx(sd, rel=1e-12)
    assert row["se"] == pytest.approx(sd / math.sqrt(values.size), rel=1e-12)
    assert row["skewness"] == pytest.approx(stats.skew(values, bias=False), rel=1e-12)
    assert row["kurtosis"] == pytest.approx(stats.kurtosis(values, bias=False), rel=1e-12)
    for key, percent in (("q1", 25), ("median", 50), ("q3", 75)):
        assert row[key] == pytest.approx(stats.scoreatpercentile(values, percent), rel=1e-12)


def test_the_interval_for_the_mean_matches_statsmodels() -> None:
    rng = np.random.default_rng(20261124)
    values = rng.normal(size=23)
    statsmodels_summary = DescrStatsW(values, ddof=1)

    row = _row(_run(_column(values), level=0.9), ALL)

    lower, upper = statsmodels_summary.tconfint_mean(alpha=0.1)
    assert row["mean"] == pytest.approx(float(statsmodels_summary.mean), rel=1e-12)
    assert row["sd"] == pytest.approx(float(statsmodels_summary.std), rel=1e-12)
    assert row["se"] == pytest.approx(float(statsmodels_summary.std_mean), rel=1e-12)
    assert (row["lower"], row["upper"]) == pytest.approx((lower, upper), rel=1e-8)


def test_the_nist_ceramic_strength_summary_is_reproduced() -> None:
    """The worked example from the NIST/SEMATECH e-Handbook of Statistical Methods.

    Source: https://www.itl.nist.gov/div898/handbook/eda/section4/eda42a1.htm (1.4.2.10.1,
    "Background and Data", the 480 longitudinal-direction strength values) and
    https://www.itl.nist.gov/div898/handbook/eda/section4/eda42a2.htm (1.4.2.10.2,
    "Analysis of the Response Variable", "Numerical Summary"). Looked up on 2026-11-24.

    The page tabulates the sample size, mean, median, minimum, maximum and standard
    deviation; it does not print the quartiles. They are therefore read from the same
    published data with the type-7 definition, worked out here by :func:`_type7` rather than
    by NumPy, so the quartile check does not merely echo the tool's own estimator.
    """
    row = _row(_run(_column(NIST_CERAMIC_Y, name="strength")), ALL)

    assert row["n"] == 480
    # Tabulated to four decimal places, so the half of the last digit bounds each check.
    assert row["mean"] == pytest.approx(650.0773, abs=5e-5)
    assert row["sd"] == pytest.approx(74.6383, abs=5e-5)
    assert row["median"] == pytest.approx(646.6275, abs=5e-5)
    assert row["minimum"] == pytest.approx(345.2940, abs=5e-5)
    assert row["maximum"] == pytest.approx(821.6540, abs=5e-5)
    assert row["q1"] == pytest.approx(_type7(NIST_CERAMIC_Y, 0.25), rel=1e-12)
    assert row["q3"] == pytest.approx(_type7(NIST_CERAMIC_Y, 0.75), rel=1e-12)


# --------------------------------------------------------------------------- properties


def _spread_example() -> list[float]:
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]


def test_adding_a_constant_leaves_spread_shape_and_flagged_rows_unchanged() -> None:
    base = _run(_column(_spread_example()))
    shifted = _run(_column([value + 10.0 for value in _spread_example()]))

    first, second = _row(base, ALL), _row(shifted, ALL)
    for key in ("sd", "se", "skewness", "kurtosis"):
        assert second[key] == pytest.approx(first[key], rel=1e-12)
    assert _flag_rows(shifted) == _flag_rows(base) == (8,)
    # The same points are flagged with the fence shifted by the same constant.
    fence = shifted.tables[1]
    keys = [column.key for column in fence.columns]
    assert fence.rows[0][keys.index("fence")] == pytest.approx(23.0)


def test_multiplying_by_a_positive_constant_scales_the_sd() -> None:
    base = _run(_column(_spread_example()))
    scaled = _run(_column([3.0 * value for value in _spread_example()]))

    first, second = _row(base, ALL), _row(scaled, ALL)
    assert second["sd"] == pytest.approx(3.0 * float(first["sd"]), rel=1e-12)
    assert second["mean"] == pytest.approx(3.0 * float(first["mean"]), rel=1e-12)
    assert second["cv"] == pytest.approx(first["cv"], rel=1e-12)
    assert second["skewness"] == pytest.approx(first["skewness"], rel=1e-12)
    assert second["kurtosis"] == pytest.approx(first["kurtosis"], rel=1e-12)
    assert _flag_rows(scaled) == _flag_rows(base)


# --------------------------------------------------------------------------- edge cases


@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), float("nan")],
        [float("inf"), float("-inf")],
        [float("nan"), float("inf"), float("-inf")],
    ],
)
def test_a_column_with_every_value_missing_raises_no_data(values: list[float]) -> None:
    with pytest.raises(ToolError) as refused:
        _run(_column(values))
    assert refused.value.code == "NO_DATA"


def test_two_values_give_the_statistics_that_exist_and_a_warning() -> None:
    result = _run(_column([1.0, 3.0]))

    row = _row(result, ALL)
    assert row["n"] == 2
    assert row["missing"] == 0
    assert row["mean"] == pytest.approx(2.0)
    assert row["sd"] == pytest.approx(math.sqrt(2.0))
    assert row["se"] == pytest.approx(1.0)
    assert row["skewness"] is None and row["kurtosis"] is None
    assert row["lower"] is not None and row["upper"] is not None
    assert any("Only 2 values" in warning for warning in result.warnings)


def test_one_value_gives_count_mean_median_minimum_and_maximum_alone() -> None:
    result = _run(_column([7.0]))

    row = _row(result, ALL)
    assert (row["n"], row["mean"], row["median"], row["minimum"], row["maximum"]) == (
        1,
        7.0,
        7.0,
        7.0,
        7.0,
    )
    for key in ("sd", "se", "cv", "skewness", "kurtosis", "lower", "upper"):
        assert row[key] is None, key
    assert any("Only 1 value" in warning for warning in result.warnings)


def test_a_constant_column_has_a_zero_sd_and_no_shape() -> None:
    row = _row(_run(_column([5.0, 5.0, 5.0, 5.0])), ALL)
    assert row["sd"] == 0.0
    assert row["cv"] == 0.0
    assert row["skewness"] is None and row["kurtosis"] is None


def test_the_coefficient_of_variation_is_undefined_when_the_mean_is_zero() -> None:
    row = _row(_run(_column([-2.0, -1.0, 0.0, 1.0, 2.0])), ALL)
    assert row["mean"] == pytest.approx(0.0)
    assert row["cv"] is None


def test_a_group_with_one_value_reports_only_what_exists_and_warns() -> None:
    result = _run(_column([1.0, 2.0, 3.0, 4.0]), _group(["a", "a", "a", "b"]))

    group = _row(result, "b")
    assert (group["n"], group["mean"], group["median"], group["minimum"], group["maximum"]) == (
        1,
        4.0,
        4.0,
        4.0,
        4.0,
    )
    # A group of one carries count, mean, median, minimum and maximum, and `None` for the
    # rest: one value has no quartiles to describe a spread with.
    for key in ("sd", "se", "q1", "q3", "cv", "skewness", "kurtosis", "lower", "upper"):
        assert group[key] is None, key
    assert any("'b'" in warning and "single value" in warning for warning in result.warnings)


def test_a_group_whose_values_are_all_missing_is_shown_with_its_missing_count() -> None:
    result = _run(_column([1.0, 2.0, 3.0, float("nan")]), _group(["a", "a", "a", "b"]))

    group = _row(result, "b")
    assert group["n"] == 0 and group["missing"] == 1
    for key in ("mean", "sd", "se", "median", "q1", "q3", "minimum", "maximum"):
        assert group[key] is None, key
    # The group is still on the box plot only when it has a value; here it has none.
    assert [box.label for box in result.plots[0].boxes] == ["a"]


def test_the_level_is_checked_before_the_data_are_read() -> None:
    with pytest.raises(ToolError) as refused:
        _run(_column([1.0, 2.0, 3.0]), level=1.0)
    assert refused.value.code == "INVALID_OPTION"


# --------------------------------------------------------------------------- excluded rows


def test_excluded_lists_each_missing_row_by_position() -> None:
    column = _column([1.0, float("nan"), 3.0, float("inf")], rows=[0, 1, 2, 3])

    result = _run(column)

    assert result.counts == Counts(n_offered=4, n_used=2, n_excluded=2)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (1, "missing in y"),
        (3, "not a finite number in y"),
    ]
    assert result.excluded_unlisted == 0


def test_more_than_two_hundred_missing_rows_are_counted_not_listed() -> None:
    values = [float("nan")] * 205 + [1.0, 2.0]

    result = _run(_column(values))

    assert len(result.excluded) == 200
    assert result.excluded[0].index == 0 and result.excluded[-1].index == 199
    assert result.excluded_unlisted == 5
    assert result.counts.n_excluded == 205 and result.counts.n_used == 2


def test_grouped_rows_without_a_value_or_a_label_are_listed() -> None:
    values = _column([1.0, 2.0, float("nan"), 4.0, 5.0])
    groups = _group(["a", "b", "a"], rows=[0, 1, 3])

    result = _run(values, groups)

    assert {row.index: row.reason for row in result.excluded} == {
        2: "missing in y",
        4: "no group label in g",
    }
    assert result.counts.n_offered == 5 and result.counts.n_used == 3


# --------------------------------------------------------------------------- fences and plot


def test_points_beyond_tukey_fences_are_flagged_and_kept_in_the_statistics() -> None:
    result = _run(_column([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]))

    fence = result.tables[1]
    assert fence.title == "Points outside Tukey's fences"
    assert [column.label for column in fence.columns] == ["Row", "Group", "Value", "Fence"]
    # Q1 and Q3 are 3 and 7, so the upper fence is 7 + 1.5 * 4 = 13.
    assert fence.rows == ((8, ALL, 100.0, 13.0),)
    assert result.plots[0].boxes[0].outliers == (100.0,)
    # Never removed: the value is still in the count and the maximum.
    row = _row(result, ALL)
    assert row["n"] == 9 and row["maximum"] == 100.0


def test_quartiles_follow_the_type_7_definition() -> None:
    row = _row(_run(_column([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])), ALL)
    assert row["q1"] == pytest.approx(2.75)
    assert row["median"] == pytest.approx(4.5)
    assert row["q3"] == pytest.approx(6.25)


def test_a_grouped_column_is_plotted_one_box_per_group() -> None:
    result = _run(
        _column([1.0, 2.0, 3.0, 10.0, 11.0]),
        _group(["a", "a", "a", "b", "b"]),
    )
    assert result.title == "Descriptive statistics of y by g"
    assert result.plots[0].title == "Box plot of y by g"
    assert [box.label for box in result.plots[0].boxes] == ["a", "b"]
    assert result.interpretation.startswith("The mean of y is highest in b")


# --------------------------------------------------------------------------- words


def test_the_interpretation_names_the_group_with_the_larger_mean_and_spread() -> None:
    result = _run(
        _column([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
        _group(["a"] * 6 + ["b"] * 5),
    )

    assert "The mean of y is highest in b, at 22" in result.interpretation
    assert "lowest in a" in result.interpretation
    assert "standard deviation is largest in a" in result.interpretation
    assert "smallest in b" in result.interpretation


def test_the_ungrouped_interpretation_gives_the_mean_and_the_spread() -> None:
    result = _run(_column([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert result.interpretation == (
        "The mean of y is 3, with a sample standard deviation of 1.58."
    )


def test_the_methods_fragment_names_each_definition_once_and_claims_no_proof() -> None:
    result = _run(_column([1.0, 2.0, 3.0, 4.0, 5.0]))
    text = result.methods_fragment

    for phrase in (
        "sample standard deviation uses n - 1",
        "linear interpolation, type 7",
        "coefficient of variation",
        "bias-corrected moment estimators",
        "t distribution on n - 1 degrees of freedom",
        "Q1 - 1.5 IQR and Q3 + 1.5 IQR",
        "flagged, never removed",
        "95 %",
    ):
        assert text.count(phrase) == 1, phrase

    note = result.tables[0].note
    assert note is not None
    for phrase in ("n - 1", "type 7", "coefficient of variation", "Tukey's fences"):
        assert phrase in note, phrase
    for words in (text, note, result.interpretation):
        assert not claims_proof(words)


# --------------------------------------------------------------------------- registration


def test_the_tool_is_registered_under_summarise_with_described_fields() -> None:
    assert "descriptives" in TOOLS
    spec = TOOLS["descriptives"]
    assert spec.family == "summarise"
    assert spec.title == "Descriptive statistics"
    assert spec.request is DescriptivesRequest

    info = TOOLS.info("descriptives")
    assert info.needs_data
    assert [field.name for field in info.fields] == ["values", "groups", "level"]
    assert [field.label for field in info.fields] == ["Values", "Groups", "Level"]
    assert [field.kind for field in info.fields] == ["column", "group", "option"]
    assert all(field.description for field in info.fields)


def test_the_registry_builds_the_request_from_column_names_and_runs_it() -> None:
    columns = {"y": _column([1.0, 2.0, 3.0, 4.0])}
    groups = {"g": _group(["a", "b", "a", "b"])}

    def column(name: str) -> Column:
        return columns[name]

    def group(name: str) -> GroupColumn:
        return groups[name]

    request = build_request(
        "descriptives",
        {"values": "y", "groups": "g", "level": 0.9},
        column=column,
        group=group,
    )
    assert isinstance(request, DescriptivesRequest)
    result = run_tool("descriptives", request)
    assert result.tool == "descriptives" and result.level == 0.9
    assert [row[0] for row in result.tables[0].rows] == ["a", "b"]


def test_run_descriptives_refuses_a_request_that_is_not_its_own() -> None:
    class Other(BaseModel):
        x: int = 1

    with pytest.raises(ToolError) as refused:
        run_descriptives(Other())
    assert refused.value.code == "INVALID_OPTION"
    assert "DescriptivesRequest" in refused.value.message
