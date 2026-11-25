"""Control charts and process capability: the limits, the rules and the indices.

Every number is checked against its definition written out in this file: the mean and the
moving ranges for the individuals chart, the subgroup means and ranges for the X-bar and R
chart, the published constants, and the capability formulas. The constant table is the one
Montgomery's *Introduction to Statistical Quality Control* prints in Appendix VI (Table VI),
and its D3 and D4 columns are held to their own definitions from ``d3``; the same columns are
cross-checked against the NIST/SEMATECH e-Handbook's table at
``itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm``, read on 2026-11-25, which prints
A2, D3 and D4 for n = 2 to 10 (its D4 for n = 3 and n = 5 differ from Montgomery's in the last
decimal, 2.575 against 2.574 and 2.115 against 2.114, and both are accepted within 0.0015).

The published worked example is the individuals chart of the NIST/SEMATECH e-Handbook's page
``6.3.2.2. Individuals Control Charts``, ``itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm``,
read on 2026-11-25. Its table prints ten flow-rate batches (49.6, 47.6, 49.9, 51.3, 47.8,
51.2, 52.6, 52.4, 53.6, 52.1), the mean 50.81, the mean moving range 1.8778, the upper limit
55.8041 and the lower limit 45.8159. The tool refuses fewer than twenty individuals, so the
example is checked through ``_individuals_chart``, the arithmetic the tool itself calls, and
the refusal of those ten points is pinned separately. No X-bar and R worked example with a
complete table could be fetched, so that chart's arithmetic is held to the published constants
and to its definitions, and its capability indices to their formulas.

The Western Electric rules are pinned by datasets built to fire exactly one rule, and the
capability indices by their shift and scale properties.
"""

import math

import numpy as np
import pytest
from pydantic import BaseModel
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.tools import ToolError
from curvestudio.tools._wording import claims_proof
from curvestudio.tools.stats import (
    Counts,
    ToolResult,
    Verdict,
    build_request,
    catalogue,
    run_tool,
)
from curvestudio.tools.stats.quality import (
    CONSTANTS,
    MIN_POINTS,
    MIN_SUBGROUPS,
    ControlChartRequest,
    _individuals_chart,
    run_controlchart,
)
from curvestudio.tools.stats.registry import FAMILIES, TOOL_MODULES, TOOLS

STAT_RTOL = 1e-12
"""The tolerance on a number this test computes from the same values by its formula."""

PUBLISHED_ATOL = 5e-5
"""The tolerance on a limit the published table prints to four decimals."""

_MONTGOMERY = {
    2: (1.880, 0.000, 3.267, 1.128),
    3: (1.023, 0.000, 2.574, 1.693),
    4: (0.729, 0.000, 2.282, 2.059),
    5: (0.577, 0.000, 2.114, 2.326),
    6: (0.483, 0.000, 2.004, 2.534),
    7: (0.419, 0.076, 1.924, 2.704),
    8: (0.373, 0.136, 1.864, 2.847),
    9: (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}
"""Montgomery's Appendix VI, Table VI: A2, D3, D4 and d2 for subgroup sizes 2 to 10."""

_D3_OF_N = {
    2: 0.853,
    3: 0.888,
    4: 0.880,
    5: 0.864,
    6: 0.848,
    7: 0.833,
    8: 0.820,
    9: 0.808,
    10: 0.797,
}
"""``d3``, the standard deviation of the range in units of sigma, from the same table."""

_NIST = {
    2: (1.880, 0.000, 3.267),
    3: (1.023, 0.000, 2.575),
    4: (0.729, 0.000, 2.282),
    5: (0.577, 0.000, 2.115),
    6: (0.483, 0.000, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777),
}
"""The NIST e-Handbook's A2, D3 and D4 for n = 2 to 10 (pmc321.htm, read 2026-11-25)."""

_NIST_FLOW = [49.6, 47.6, 49.9, 51.3, 47.8, 51.2, 52.6, 52.4, 53.6, 52.1]
"""The ten flow-rate batches of the NIST individuals example (pmc322.htm)."""

_STEADY = [
    1.0,
    -1.0,
    0.0,
    2.0,
    -2.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    0.0,
    -2.0,
    2.0,
    -1.0,
    1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    1.0,
    -1.0,
    0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.0,
    1.0,
    -1.0,
]
"""Thirty values with no Western Electric signal: no rule fires on this chart."""

_SPIKE = [*_STEADY]
_SPIKE[14] = 30.0
"""One value far above the rest: rule 1 fires at the fifteenth point, and no other rule."""

_PAIR = [*_STEADY]
_PAIR[14] = 4.5
_PAIR[15] = 4.5
"""Two neighbouring values beyond two sigma on the upper side: rule 2, and no other rule."""

_RUN = [*_STEADY]
for _index in range(10, 18):
    _RUN[_index] = 1.5
"""Eight consecutive values above the centre line: rule 4, and no other rule."""

_DIAMETERS = (
    (10.2, 10.0, 10.1, 10.3, 9.9),
    (10.1, 10.4, 9.8, 10.2, 10.0),
    (10.0, 10.1, 10.3, 10.2, 9.9),
    (10.3, 9.9, 10.1, 10.2, 10.0),
    (10.1, 10.2, 10.0, 10.3, 9.9),
    (9.9, 10.3, 10.1, 10.0, 10.2),
    (10.2, 10.1, 9.9, 10.3, 10.0),
    (10.1, 10.0, 10.2, 10.3, 9.9),
    (10.0, 10.3, 10.1, 9.9, 10.2),
    (10.3, 10.1, 10.0, 10.2, 9.9),
    (10.1, 10.2, 10.0, 10.3, 9.9, 10.1, 10.2),
    (10.2, 10.0, 10.1),
)
"""Ten complete runs of five, one run of seven and a last run of three: 60 rows."""

_DIAMETER_SIZE = 5


def _column(values: list[float], name: str = "x", rows: list[int] | None = None) -> Column:
    """A column over positions 0..n-1, as the data layer builds it."""
    array = np.asarray(values, dtype=np.float64)
    positions = np.arange(array.size, dtype=np.intp) if rows is None else np.asarray(rows)
    return Column(name=name, values=array, rows=positions)


def _group(labels: list[str], name: str = "batch", rows: list[int] | None = None) -> GroupColumn:
    """A group column over positions 0..n-1, as the data layer builds it."""
    positions = np.arange(len(labels), dtype=np.intp) if rows is None else np.asarray(rows)
    return GroupColumn(name=name, labels=np.asarray(labels, dtype=object), rows=positions)


def _run(values: list[float], **options: object) -> ToolResult:
    """The individuals chart, unless the options say otherwise."""
    request = ControlChartRequest(values=_column(values), **options)
    return run_controlchart(request)


def _estimate(result: ToolResult, label: str):
    """The estimate with ``label``, or a failure naming the labels present."""
    for found in result.estimates:
        if found.label == label:
            return found
    raise AssertionError(f"no estimate labelled {label!r}: {[e.label for e in result.estimates]}")


def _check(result: ToolResult, check_id: str):
    """The assumption check with ``check_id``."""
    for found in result.assumptions:
        if found.id == check_id:
            return found
    raise AssertionError(f"no check {check_id!r}: {[a.id for a in result.assumptions]}")


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _moving_ranges(values: list[float]) -> np.ndarray:
    return np.abs(np.diff(np.asarray(values, dtype=np.float64)))


def _frames(
    chunks: tuple[tuple[float, ...], ...], name: str = "diameter"
) -> tuple[Column, GroupColumn]:
    """The rows of ``chunks`` as one column and one label column, a label per chunk."""
    values: list[float] = []
    labels: list[str] = []
    for index, chunk in enumerate(chunks):
        values.extend(chunk)
        labels.extend([chr(ord("A") + index)] * len(chunk))
    return _column(values, name), _group(labels)


def _subgroup_means(chunks: tuple[tuple[float, ...], ...]) -> np.ndarray:
    return np.asarray([float(np.mean(np.asarray(chunk))) for chunk in chunks])


def _subgroup_ranges(chunks: tuple[tuple[float, ...], ...]) -> np.ndarray:
    return np.asarray([float(np.ptp(np.asarray(chunk))) for chunk in chunks])


# --------------------------------------------------------------------------- constants


def test_the_constants_are_montgomerys_published_table() -> None:
    for size, (a2, d3, d4, d2) in _MONTGOMERY.items():
        constants = CONSTANTS[size]
        assert (constants.a2, constants.d3, constants.d4, constants.d2) == (a2, d3, d4, d2)


def test_the_constants_follow_their_definitions_from_d3() -> None:
    for size, (a2, d3, d4, d2) in _MONTGOMERY.items():
        # A2 = 3 / (d2 * sqrt(n)), D3 = max(0, 1 - 3 d3 / d2), D4 = 1 + 3 d3 / d2.
        assert a2 == pytest.approx(3.0 / (d2 * math.sqrt(size)), abs=1e-3)
        assert d3 == pytest.approx(max(0.0, 1.0 - 3.0 * _D3_OF_N[size] / d2), abs=1.5e-3)
        assert d4 == pytest.approx(1.0 + 3.0 * _D3_OF_N[size] / d2, abs=3e-3)


def test_the_constants_agree_with_the_nist_table_within_its_last_decimal() -> None:
    for size, (a2, d3, d4) in _NIST.items():
        constants = CONSTANTS[size]
        assert constants.a2 == pytest.approx(a2, abs=1e-9)
        assert constants.d3 == pytest.approx(d3, abs=1e-9)
        assert constants.d4 == pytest.approx(d4, abs=1.5e-3)


def test_subgroup_sizes_outside_two_to_ten_have_no_constants() -> None:
    assert sorted(CONSTANTS) == list(range(2, 11))


# --------------------------------------------------------------------------- individuals


def test_the_individuals_chart_is_the_mean_and_the_moving_range() -> None:
    result = _run(_STEADY)

    values = np.asarray(_STEADY, dtype=np.float64)
    moving = _moving_ranges(_STEADY)
    mean = float(np.mean(values))
    mean_range = float(np.mean(moving))
    sigma = mean_range / 1.128

    assert type(result) is ToolResult
    assert (result.tool, result.title, result.level) == (
        "controlchart",
        "Control chart of x",
        0.95,
    )
    assert [estimate.label for estimate in result.estimates] == [
        "Centre line",
        "Upper control limit",
        "Lower control limit",
        "Mean moving range",
    ]
    assert _estimate(result, "Centre line").value == pytest.approx(mean, rel=STAT_RTOL)
    assert _estimate(result, "Mean moving range").value == pytest.approx(mean_range, rel=STAT_RTOL)
    assert _estimate(result, "Upper control limit").value == pytest.approx(
        mean + 3.0 * sigma, rel=STAT_RTOL
    )
    assert _estimate(result, "Lower control limit").value == pytest.approx(
        mean - 3.0 * sigma, rel=STAT_RTOL
    )
    # The limit the brief's words carry: 2.66 times the mean moving range.
    assert _estimate(result, "Upper control limit").value == pytest.approx(
        mean + 2.66 * mean_range, rel=1e-3
    )
    assert result.tests == ()
    assert result.counts == Counts(n_offered=30, n_used=30, n_excluded=0)


def test_the_nist_flow_rate_example_reproduces_its_published_limits() -> None:
    # The tool refuses ten individuals, so the published arithmetic is read directly.
    chart = _individuals_chart(np.asarray(_NIST_FLOW, dtype=np.float64), "flow rate")

    assert chart.centre == pytest.approx(50.81, abs=1e-12)
    assert chart.mean_range == pytest.approx(1.8778, abs=5e-5)
    assert chart.upper == pytest.approx(55.8041, abs=PUBLISHED_ATOL)
    assert chart.lower == pytest.approx(45.8159, abs=PUBLISHED_ATOL)
    # The handbook's own formula, 50.81 + 3 * 1.8778 / 1.128, to its printed precision.
    assert chart.upper == pytest.approx(50.81 + 3.0 * 1.8778 / 1.128, abs=1e-4)


def test_a_chart_of_fewer_than_twenty_individuals_is_refused() -> None:
    with pytest.raises(ToolError) as refused:
        _run(_STEADY[:19])

    assert refused.value.code == "TOO_FEW_ROWS"
    assert str(MIN_POINTS) in refused.value.message
    assert "19" in refused.value.message


# --------------------------------------------------------------------------- x-bar and R


def test_the_xbar_r_chart_uses_the_published_constants_and_the_subgroup_means() -> None:
    values, groups = _frames(_DIAMETERS)
    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    complete = (*_DIAMETERS[:10], _DIAMETERS[10][:5])
    means = _subgroup_means(complete)
    ranges = _subgroup_ranges(complete)
    grand = float(np.mean(means))
    mean_range = float(np.mean(ranges))
    size = _DIAMETER_SIZE
    sigma = mean_range / _MONTGOMERY[size][3]

    assert result.title == "X-bar and R chart of diameter by batch"
    assert [estimate.label for estimate in result.estimates] == [
        "Centre line",
        "Upper control limit",
        "Lower control limit",
        "Mean range",
    ]
    assert _estimate(result, "Centre line").value == pytest.approx(grand, rel=STAT_RTOL)
    assert _estimate(result, "Mean range").value == pytest.approx(mean_range, rel=STAT_RTOL)
    assert _estimate(result, "Upper control limit").value == pytest.approx(
        grand + _MONTGOMERY[size][0] * mean_range, rel=STAT_RTOL
    )
    assert _estimate(result, "Lower control limit").value == pytest.approx(
        grand - _MONTGOMERY[size][0] * mean_range, rel=STAT_RTOL
    )
    # The constants' own definition, A2 = 3 / (d2 * sqrt(n)), gives the same line to the
    # table's third decimal.
    assert _estimate(result, "Upper control limit").value == pytest.approx(
        grand + 3.0 * sigma / math.sqrt(size), rel=1e-4
    )

    statistics = result.tables[0]
    assert statistics.title == "Chart statistics"
    rows = dict(statistics.rows)
    assert rows["Number of subgroups"] == 11
    assert rows["Subgroup size"] == 5
    assert rows["Grand mean"] == pytest.approx(grand, rel=STAT_RTOL)
    assert rows["Within-subgroup standard deviation"] == pytest.approx(sigma, rel=STAT_RTOL)
    assert rows["R chart upper control limit"] == pytest.approx(
        _MONTGOMERY[size][2] * mean_range, rel=STAT_RTOL
    )
    assert rows["R chart lower control limit"] == pytest.approx(0.0, abs=0.0)


def test_the_short_subgroups_are_left_out_counted_and_listed() -> None:
    values, groups = _frames(_DIAMETERS)
    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    assert result.counts == Counts(n_offered=60, n_used=55, n_excluded=5)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (55, "in a short subgroup of 2 rows for a size of 5"),
        (56, "in a short subgroup of 2 rows for a size of 5"),
        (57, "in a short subgroup of 3 rows for a size of 5"),
        (58, "in a short subgroup of 3 rows for a size of 5"),
        (59, "in a short subgroup of 3 rows for a size of 5"),
    ]
    assert any("shorter than the size of 5" in warning for warning in result.warnings)


def test_a_lone_row_is_a_subgroup_of_one_and_is_left_out() -> None:
    chunks = tuple(
        tuple(10.0 + 0.1 * run + 0.01 * column for column in range(5)) for run in range(12)
    )
    values, groups = _frames((*chunks, (9.0,)))

    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    assert result.counts == Counts(n_offered=61, n_used=60, n_excluded=1)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (60, "in a short subgroup of 1 row for a size of 5"),
    ]


def test_one_label_cut_into_consecutive_subgroups_of_the_size() -> None:
    measured = [10.0 + 0.1 * run + 0.01 * column for run in range(10) for column in range(5)]
    values = _column(measured, "length")
    labels = _group(["run"] * 50)
    result = run_controlchart(ControlChartRequest(values=values, subgroups=labels, chart="xbar_r"))

    rows = dict(result.tables[0].rows)
    assert rows["Number of subgroups"] == MIN_SUBGROUPS
    assert rows["Subgroup size"] == 5
    assert result.counts == Counts(n_offered=50, n_used=50, n_excluded=0)


def test_a_subgroup_with_a_missing_value_is_left_out_as_a_short_subgroup() -> None:
    chunks = tuple(
        tuple(10.0 + 0.1 * run + 0.01 * column for column in range(5)) for run in range(11)
    )
    values, groups = _frames(chunks)
    raw = values.values.copy()
    raw[7] = np.nan
    values = Column(name="diameter", values=raw, rows=values.rows)

    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    assert result.counts == Counts(n_offered=55, n_used=50, n_excluded=5)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (5, "in a short subgroup of 4 rows for a size of 5"),
        (6, "in a short subgroup of 4 rows for a size of 5"),
        (7, "missing in diameter"),
        (8, "in a short subgroup of 4 rows for a size of 5"),
        (9, "in a short subgroup of 4 rows for a size of 5"),
    ]


def test_a_row_without_a_subgroup_label_is_left_out_with_its_own_reason() -> None:
    chunks = tuple(
        tuple(10.0 + 0.1 * run + 0.01 * column for column in range(5)) for run in range(11)
    )
    values, groups = _frames(chunks)
    kept = [position for position in range(55) if position != 3]
    groups = GroupColumn(
        name="batch",
        labels=np.asarray([groups.labels[position] for position in kept], dtype=object),
        rows=np.asarray(kept, dtype=np.intp),
    )

    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    assert result.counts == Counts(n_offered=55, n_used=50, n_excluded=5)
    assert (3, "no group label in batch") in [(row.index, row.reason) for row in result.excluded]


def test_fewer_than_ten_subgroups_are_refused() -> None:
    chunks = tuple(
        tuple(10.0 + 0.1 * run + 0.01 * column for column in range(5)) for run in range(9)
    )
    values, groups = _frames(chunks)

    with pytest.raises(ToolError) as refused:
        run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    assert refused.value.code == "TOO_FEW_ROWS"
    assert str(MIN_SUBGROUPS) in refused.value.message
    assert "9" in refused.value.message


def test_the_xbar_r_chart_needs_a_subgroups_column() -> None:
    with pytest.raises(ToolError) as refused:
        run_controlchart(ControlChartRequest(values=_column(_STEADY), chart="xbar_r"))

    assert refused.value.code == "INVALID_OPTION"
    assert "subgroups" in refused.value.message


@pytest.mark.parametrize("size", [1, 11, 0, -5])
def test_a_subgroup_size_outside_two_to_ten_is_refused(size: int) -> None:
    values, groups = _frames(_DIAMETERS)

    with pytest.raises(ToolError) as refused:
        run_controlchart(
            ControlChartRequest(values=values, subgroups=groups, subgroup_size=size, chart="xbar_r")
        )

    assert refused.value.code == "INVALID_OPTION"
    assert "subgroup size" in refused.value.message


def test_a_constant_series_is_refused_on_either_chart() -> None:
    with pytest.raises(ToolError) as flat:
        _run([4.0] * 25)
    assert flat.value.code == "CONSTANT_COLUMN"

    chunks = tuple((4.0,) * 5 for _ in range(10))
    values, groups = _frames(chunks)
    with pytest.raises(ToolError) as flat_groups:
        run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))
    assert flat_groups.value.code == "CONSTANT_COLUMN"
    assert "mean range is zero" in flat_groups.value.message


def test_subgroups_are_ignored_by_the_individuals_chart_with_a_warning() -> None:
    values = _column(_STEADY)
    groups = _group(["run"] * len(_STEADY))

    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, subgroup_size=5))

    assert result.counts.n_used == len(_STEADY)
    assert any("read only by the X-bar and R chart" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- capability


def test_the_capability_indices_follow_their_formulas() -> None:
    values, groups = _frames(_DIAMETERS)
    request = ControlChartRequest(
        values=values,
        subgroups=groups,
        chart="xbar_r",
        lower_spec=9.6,
        upper_spec=10.5,
    )
    result = run_controlchart(request)

    complete = (*_DIAMETERS[:10], _DIAMETERS[10][:5])
    used = [value for chunk in complete for value in chunk]
    means = _subgroup_means(complete)
    ranges = _subgroup_ranges(complete)
    mean = float(np.mean(means))
    mean_range = float(np.mean(ranges))
    sigma_within = mean_range / _MONTGOMERY[5][3]
    sigma_overall = float(np.std(np.asarray(used), ddof=1))
    lower, upper = 9.6, 10.5

    cp = (upper - lower) / (6.0 * sigma_within)
    cpu = (upper - mean) / (3.0 * sigma_within)
    cpl = (mean - lower) / (3.0 * sigma_within)
    pp = (upper - lower) / (6.0 * sigma_overall)
    ppu = (upper - mean) / (3.0 * sigma_overall)
    ppl = (mean - lower) / (3.0 * sigma_overall)
    outside = (
        float(scipy_stats.norm.cdf(lower, loc=mean, scale=sigma_within))
        + float(scipy_stats.norm.sf(upper, loc=mean, scale=sigma_within))
    ) * 100.0

    assert [estimate.label for estimate in result.estimates][-5:] == [
        "Cp",
        "Cpk",
        "Pp",
        "Ppk",
        "Estimated values outside the limits %",
    ]
    assert _estimate(result, "Cp").value == pytest.approx(cp, rel=STAT_RTOL)
    assert _estimate(result, "Cpk").value == pytest.approx(min(cpu, cpl), rel=STAT_RTOL)
    assert _estimate(result, "Pp").value == pytest.approx(pp, rel=STAT_RTOL)
    assert _estimate(result, "Ppk").value == pytest.approx(min(ppu, ppl), rel=STAT_RTOL)
    assert _estimate(result, "Estimated values outside the limits %").value == pytest.approx(
        outside, rel=1e-10
    )

    rows = dict(result.tables[0].rows)
    assert rows["Cpu"] == pytest.approx(cpu, rel=STAT_RTOL)
    assert rows["Cpl"] == pytest.approx(cpl, rel=STAT_RTOL)
    assert rows["Overall standard deviation"] == pytest.approx(sigma_overall, rel=STAT_RTOL)
    assert "1.33" in result.interpretation
    assert "convention" in result.interpretation
    assert _check(result, "normality").verdict is not Verdict.NOT_APPLICABLE


def test_capability_is_not_computed_without_both_limits() -> None:
    result = _run(_STEADY, lower_spec=0.0)

    assert [estimate.label for estimate in result.estimates] == [
        "Centre line",
        "Upper control limit",
        "Lower control limit",
        "Mean moving range",
    ]
    assert [check.id for check in result.assumptions] == ["stability"]
    assert any(
        "both limits" in warning and "not computed" in warning for warning in result.warnings
    )


def test_a_series_far_from_normal_with_capability_is_warned_about() -> None:
    skewed = [
        0.1,
        0.2,
        0.15,
        0.12,
        0.18,
        0.11,
        0.16,
        0.14,
        0.13,
        0.17,
        0.19,
        0.12,
        0.15,
        0.11,
        0.16,
        0.14,
        0.13,
        0.18,
        0.12,
        0.15,
        5.0,
        5.1,
        5.2,
        5.3,
        5.4,
    ]
    result = _run(skewed, lower_spec=0.0, upper_spec=6.0)

    check = _check(result, "normality")
    assert check.verdict in (Verdict.CAUTION, Verdict.FAIL)
    assert any(
        "Shapiro-Wilk" in warning and "normal distribution" in warning
        for warning in result.warnings
    )


def test_the_capability_indices_are_unchanged_by_a_shift_and_by_a_scale() -> None:
    values, groups = _frames(_DIAMETERS)
    lower, upper = 9.6, 10.5
    base = run_controlchart(
        ControlChartRequest(
            values=values,
            subgroups=groups,
            chart="xbar_r",
            lower_spec=lower,
            upper_spec=upper,
        )
    )

    shifted_values = _column([float(value) + 3.0 for value in values.values])
    shifted = run_controlchart(
        ControlChartRequest(
            values=shifted_values,
            subgroups=groups,
            chart="xbar_r",
            lower_spec=lower + 3.0,
            upper_spec=upper + 3.0,
        )
    )
    for line in ("Centre line", "Upper control limit", "Lower control limit"):
        assert _estimate(shifted, line).value == pytest.approx(
            _estimate(base, line).value + 3.0, rel=STAT_RTOL
        )
    for index in ("Cp", "Cpk", "Pp", "Ppk"):
        assert _estimate(shifted, index).value == pytest.approx(
            _estimate(base, index).value, rel=STAT_RTOL
        )

    scaled_values = _column([float(value) * 2.0 for value in values.values])
    scaled = run_controlchart(
        ControlChartRequest(
            values=scaled_values,
            subgroups=groups,
            chart="xbar_r",
            lower_spec=lower * 2.0,
            upper_spec=upper * 2.0,
        )
    )
    for line in ("Centre line", "Upper control limit", "Lower control limit"):
        assert _estimate(scaled, line).value == pytest.approx(
            _estimate(base, line).value * 2.0, rel=STAT_RTOL
        )
    for index in ("Cp", "Cpk", "Pp", "Ppk"):
        assert _estimate(scaled, index).value == pytest.approx(
            _estimate(base, index).value, rel=STAT_RTOL
        )
    assert _estimate(scaled, "Estimated values outside the limits %").value == pytest.approx(
        _estimate(base, "Estimated values outside the limits %").value, rel=STAT_RTOL
    )


def test_a_shift_and_a_scale_leave_the_individuals_indices_alone() -> None:
    base = _run(_STEADY, lower_spec=-4.0, upper_spec=4.0)
    shifted = _run([value + 10.0 for value in _STEADY], lower_spec=6.0, upper_spec=14.0)
    scaled = _run([value * 3.0 for value in _STEADY], lower_spec=-12.0, upper_spec=12.0)

    for line in ("Centre line", "Upper control limit", "Lower control limit"):
        assert _estimate(shifted, line).value == pytest.approx(
            _estimate(base, line).value + 10.0, rel=STAT_RTOL
        )
        assert _estimate(scaled, line).value == pytest.approx(
            _estimate(base, line).value * 3.0, rel=STAT_RTOL
        )
    for index in ("Cp", "Cpk", "Pp", "Ppk"):
        assert _estimate(shifted, index).value == pytest.approx(
            _estimate(base, index).value, rel=1e-10
        )
        assert _estimate(scaled, index).value == pytest.approx(
            _estimate(base, index).value, rel=1e-10
        )


# --------------------------------------------------------------------------- the rules


def test_a_point_beyond_a_limit_fires_rule_one_and_fails_stability() -> None:
    result = _run(_SPIKE)

    stability = _check(result, "stability")
    assert stability.verdict is Verdict.FAIL
    assert [row[0] for row in result.tables[1].rows] == ["Rule 1"]
    assert result.tables[1].rows[0][1] == "15"
    assert "beyond a control limit" in result.tables[1].rows[0][2]
    assert "does not look stable" in result.interpretation
    assert "rule 1" in result.interpretation


def test_two_of_three_beyond_two_sigma_fires_rule_two_and_cautions() -> None:
    result = _run(_PAIR)

    stability = _check(result, "stability")
    assert stability.verdict is Verdict.CAUTION
    assert [row[0] for row in result.tables[1].rows] == ["Rule 2"]
    assert result.tables[1].rows[0][1] == "16, 17"
    assert "two standard deviations on the same side" in result.tables[1].rows[0][2]
    assert "rule 2" in result.interpretation


def test_eight_points_on_one_side_fires_rule_four_and_cautions() -> None:
    result = _run(_RUN)

    stability = _check(result, "stability")
    assert stability.verdict is Verdict.CAUTION
    assert [row[0] for row in result.tables[1].rows] == ["Rule 4"]
    assert result.tables[1].rows[0][1].startswith("11, 12, 13, 14, 15, 16, 17, 18")
    assert "consecutive points lie above the centre line" in result.tables[1].rows[0][2]


def test_a_steady_chart_fires_no_rule_and_passes_stability() -> None:
    result = _run(_STEADY)

    stability = _check(result, "stability")
    assert stability.verdict is Verdict.PASS
    assert "no special cause" in result.interpretation
    assert [table.title for table in result.tables] == ["Chart statistics"]


def test_the_rules_are_applied_to_the_subgroup_means() -> None:
    # Twelve subgroups of five, each with a range of one and a mean near zero; the seventh
    # subgroup's mean is far above the rest, so rule 1 fires at subgroup 7 and at no other.
    offsets = (-0.5, -0.25, 0.0, 0.25, 0.5)
    chunks = tuple(
        tuple(mean + offset for offset in offsets)
        for mean in (0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 2.0, 0.1, -0.1, 0.1, -0.1, 0.1)
    )
    values, groups = _frames(chunks)
    result = run_controlchart(ControlChartRequest(values=values, subgroups=groups, chart="xbar_r"))

    rules = [row[0] for row in result.tables[1].rows]
    assert rules[0] == "Rule 1" and result.tables[1].rows[0][1] == "7"
    assert _check(result, "stability").verdict is Verdict.FAIL


# --------------------------------------------------------------------------- plots and words


def test_the_plot_carries_the_points_and_the_three_lines() -> None:
    result = _run(_STEADY)
    plot = result.plots[0]

    assert plot.kind == "lines"
    assert plot.title == "Control chart"
    assert (plot.x_label, plot.y_label) == ("Observation", "x")
    assert [series.label for series in plot.series] == [
        "Observations",
        "Centre line",
        "Upper limit",
        "Lower limit",
    ]
    observations, centre, upper, lower = plot.series
    assert observations.x == tuple(float(index) for index in range(1, 31))
    assert observations.y == pytest.approx(_STEADY, rel=STAT_RTOL)
    assert plot.x_labels == tuple(str(index) for index in range(1, 31))
    lines = {
        "Centre line": centre,
        "Upper control limit": upper,
        "Lower control limit": lower,
    }
    for label, series in lines.items():
        value = _estimate(result, label).value
        assert series.x == (1.0, 30.0)
        assert series.y == pytest.approx((value, value), rel=STAT_RTOL)


def test_a_plot_is_strided_to_two_thousand_points_with_a_note() -> None:
    values = list(np.random.default_rng(4).standard_normal(2_500))
    result = _run(values)

    plot = result.plots[0]
    assert len(plot.series[0].x) == 2_000
    assert len(plot.series[0].y) == 2_000
    assert plot.note is not None and "2,000 of 2,500 points shown" in plot.note


def test_the_words_name_the_methods_and_claim_no_proof() -> None:
    values, groups = _frames(_DIAMETERS)
    result = run_controlchart(
        ControlChartRequest(
            values=values,
            subgroups=groups,
            chart="xbar_r",
            lower_spec=9.6,
            upper_spec=10.5,
        )
    )

    texts = [result.title, result.interpretation, result.methods_fragment, *result.warnings]
    for check in result.assumptions:
        texts += [check.title, check.evidence, check.explanation, check.next_step or ""]
    for table in result.tables:
        texts.append(table.note or "")
        texts += [str(cell) for row in table.rows for cell in row]
    for text in texts:
        assert not claims_proof(text), text

    for phrase in ("Montgomery", "Appendix VI", "Western Electric", "A2", "D3", "D4", "d2"):
        assert phrase in result.methods_fragment, phrase
    assert "Cp" in result.methods_fragment and "Pp" in result.methods_fragment
    assert "not a population" in result.interpretation

    individuals = _run(_STEADY, lower_spec=-4.0, upper_spec=4.0)
    for phrase in ("individuals chart", "moving range", "d2 = 1.128", "Montgomery"):
        assert phrase in individuals.methods_fragment, phrase


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_a_level_outside_zero_and_one_is_refused(level: float) -> None:
    with pytest.raises(ToolError) as refused:
        _run(_STEADY, level=level)

    assert refused.value.code == "INVALID_OPTION"


@pytest.mark.parametrize(("lower", "upper"), [(10.0, 5.0), (10.0, 10.0)])
def test_specification_limits_in_the_wrong_order_are_refused(lower: float, upper: float) -> None:
    with pytest.raises(ToolError) as refused:
        _run(_STEADY, lower_spec=lower, upper_spec=upper)

    assert refused.value.code == "INVALID_OPTION"
    assert "below" in refused.value.message


def test_a_request_that_is_not_its_own_is_refused() -> None:
    class Other(BaseModel):
        x: int = 1

    with pytest.raises(ToolError) as refused:
        run_controlchart(Other())

    assert refused.value.code == "INVALID_OPTION"
    assert "ControlChartRequest" in refused.value.message


def test_fewer_than_twenty_five_points_are_warned_about() -> None:
    result = _run(_STEADY[:22])

    assert any("fewer than 25" in warning for warning in result.warnings)


# --------------------------------------------------------------------------- excluded rows


def test_missing_and_non_finite_values_are_listed_not_dropped() -> None:
    values = [*_STEADY[:27], float("nan"), float("inf"), float("-inf")]
    result = _run(values)

    assert result.counts == Counts(n_offered=30, n_used=27, n_excluded=3)
    assert [(row.index, row.reason) for row in result.excluded] == [
        (27, "missing in x"),
        (28, "not a finite number in x"),
        (29, "not a finite number in x"),
    ]
    assert result.excluded_unlisted == 0


# --------------------------------------------------------------------------- the result


def test_the_result_survives_a_json_round_trip_without_a_nan() -> None:
    result = _run(_STEADY, lower_spec=-4.0, upper_spec=4.0)
    payload = result.model_dump_json()

    assert ToolResult.model_validate_json(payload) == result
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_running_twice_gives_the_same_json() -> None:
    values, groups = _frames(_DIAMETERS)
    request = ControlChartRequest(
        values=values, subgroups=groups, chart="xbar_r", lower_spec=9.6, upper_spec=10.5
    )

    assert (
        run_controlchart(request).model_dump_json() == run_controlchart(request).model_dump_json()
    )


# --------------------------------------------------------------------------- registration


def test_the_tool_is_registered_under_measurement_with_described_fields() -> None:
    assert "curvestudio.tools.stats.quality" in TOOL_MODULES
    assert ("measurement", "Measurement and classification") in FAMILIES
    assert "controlchart" in TOOLS
    spec = TOOLS["controlchart"]
    assert spec.family == "measurement"
    assert spec.title == "Control chart and process capability"

    info = TOOLS.info("controlchart")
    assert info.needs_data
    assert [field.name for field in info.fields] == [
        "values",
        "subgroups",
        "subgroup_size",
        "chart",
        "lower_spec",
        "upper_spec",
        "level",
    ]
    assert [field.kind for field in info.fields] == [
        "column",
        "group",
        "option",
        "option",
        "option",
        "option",
        "option",
    ]
    assert [field.label for field in info.fields] == [
        "Values",
        "Subgroups",
        "Subgroup size",
        "Chart",
        "Lower specification limit",
        "Upper specification limit",
        "Level",
    ]
    assert all(field.description for field in info.fields)
    assert info.fields[3].option_schema is not None
    assert "controlchart" in [entry.name for entry in catalogue()]


def test_the_registry_builds_and_runs_the_chart_from_column_names() -> None:
    values, groups = _frames(_DIAMETERS)
    request = build_request(
        "controlchart",
        {
            "values": "diameter",
            "subgroups": "batch",
            "subgroup_size": 5,
            "chart": "xbar_r",
            "lower_spec": 9.6,
            "upper_spec": 10.5,
        },
        column=lambda name: values,
        group=lambda name: groups,
    )

    assert isinstance(request, ControlChartRequest)
    assert request.chart == "xbar_r" and request.subgroup_size == 5
    result = run_tool("controlchart", request)
    assert result.tool == "controlchart"
