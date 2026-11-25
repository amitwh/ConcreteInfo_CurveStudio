"""Control charts and process capability: the variation a process is read against.

The tool draws one control chart from one numeric column. In its default mode the values are
read in row order as individuals, and the moving range of two consecutive values estimates the
short-term standard deviation. In the X-bar and R mode the rows are read in their order and
split into subgroups: each maximal run of one label in the subgroups column is cut into
consecutive groups of the chosen size, and a piece shorter than that size is left out, counted
and listed by position. The plotted points are the values themselves, or the subgroup means.

The limits are the customary three-sigma limits of a Shewhart chart. For individuals the
centre line is the mean, and the limits are the mean plus or minus three times the mean moving
range divided by ``d2 = 1.128`` (2.66 times the mean moving range). For the X-bar and R chart
the centre line is the grand mean, the limits are the grand mean plus or minus ``A2`` times the
mean range, and the range chart's limits are ``D3`` and ``D4`` times the mean range. The
constants for subgroup sizes 2 to 10 are written below from Montgomery's *Introduction to
Statistical Quality Control*, Appendix VI (Table VI). The mean range divided by ``d2``
estimates the within-subgroup standard deviation, and that estimate, divided by the square
root of the subgroup size, is the standard deviation of a subgroup mean.

Three Western Electric rules flag special causes on the plotted points: rule 1, a point beyond
a control limit; rule 2, two of three consecutive points beyond two standard deviations on the
same side of the centre line; and rule 4, eight consecutive points on one side. Each finding
gives the 1-based positions of the points it names.

Process capability is computed only when both specification limits are given. Cp and Cpk use
the within-subgroup standard deviation; Pp and Ppk use the sample standard deviation with
n - 1 in its denominator, the overall spread. The fraction outside the limits is read from a
normal distribution with the mean and the within-subgroup standard deviation. A capability
index describes a process, not a population, and it means something only for a stable process
whose values are close enough to normal: stability is read from the rules, and normality is
checked with Shapiro-Wilk's W test and reported as an assumption. A process that does not look
stable is not repaired here; the special causes are named so a person can look for them.
"""

import math
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.diagnostics import DiagnosticTest, shapiro_wilk
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import joined, num, p_threshold
from curvestudio.tools.stats._base import (
    BoolArray,
    Counts,
    Estimate,
    ToolResult,
    aligned,
    assumption,
    check_level,
    list_excluded,
    p_text,
    pairs,
)
from curvestudio.tools.stats._display import (
    Cell,
    ColumnKind,
    FloatArray,
    MultiLinePlot,
    Table,
    XYSeries,
    stride,
)
from curvestudio.tools.stats._tables import table
from curvestudio.tools.stats.registry import TOOLS, ToolSpec

MIN_POINTS = 20
"""The fewest values an individuals chart plots."""

MIN_SUBGROUPS = 10
"""The fewest complete subgroups an X-bar and R chart takes."""

WARN_POINTS = 25
"""Fewer plotted points than this makes the limits imprecise, and the result says so."""

DEFAULT_SUBGROUP_SIZE = 5
"""The subgroup size the X-bar and R chart uses when none is given."""

MIN_SUBGROUP_SIZE = 2
"""The smallest subgroup: a range needs two values."""

MAX_SUBGROUP_SIZE = 10
"""The largest subgroup: above it the range loses its efficiency and the table ends."""

LIMIT_WIDTH = 3.0
"""The west-east sigma width of a Shewhart control limit."""

TWO_SIGMA = 2.0
"""The zone rule 2 reads."""

RULE_FOUR_RUN = 8
"""The run length rule 4 flags."""

CAUTION_P = 0.05
"""At or below this p-value the normal assumption is in doubt."""

FAIL_P = 0.001
"""Below this p-value the normal assumption reads as contradicted."""

MIN_SHAPIRO = 3
"""Shapiro-Wilk's own floor; every chart here already holds more values than this."""

_IGNORED_GROUP_OPTIONS = (
    "The subgroups column and the subgroup size are read only by the X-bar and R chart; the "
    "individuals chart read the values in row order and did not use them."
)

_STATISTIC_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("statistic", "Statistic", "text"),
    ("value", "Value", "number"),
)

_CAUSE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("rule", "Rule", "text"),
    ("points", "Points", "text"),
    ("found", "What was found", "text"),
)


class _Constants(NamedTuple):
    """The control-chart constants for one subgroup size, from Montgomery's Table VI."""

    a2: float
    """The X-bar chart's limit width, in units of the mean range."""
    d3: float
    """The lower range-chart limit, in units of the mean range."""
    d4: float
    """The upper range-chart limit, in units of the mean range."""
    d2: float
    """The mean range, in units of the within-subgroup standard deviation."""


CONSTANTS: dict[int, _Constants] = {
    2: _Constants(1.880, 0.000, 3.267, 1.128),
    3: _Constants(1.023, 0.000, 2.574, 1.693),
    4: _Constants(0.729, 0.000, 2.282, 2.059),
    5: _Constants(0.577, 0.000, 2.114, 2.326),
    6: _Constants(0.483, 0.000, 2.004, 2.534),
    7: _Constants(0.419, 0.076, 1.924, 2.704),
    8: _Constants(0.373, 0.136, 1.864, 2.847),
    9: _Constants(0.337, 0.184, 1.816, 2.970),
    10: _Constants(0.308, 0.223, 1.777, 3.078),
}
"""A2, D3, D4 and d2 for subgroup sizes 2 to 10 (Montgomery, Appendix VI, Table VI).

The estimators are ``sigma = R-bar / d2``, ``A2 = 3 / (d2 * sqrt(n))``,
``D3 = 1 - 3 * d3 / d2`` and ``D4 = 1 + 3 * d3 / d2``, with ``d3`` the standard deviation of
the range in units of sigma. The table is round at the third decimal, so a limit computed from
the rounded constant carries that rounding.
"""


class ControlChartRequest(BaseModel):
    """One column to chart, how to subgroup it, and the specification limits if any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column = Field(
        description="The numeric column whose values, or subgroup means, are charted."
    )
    subgroups: GroupColumn | None = Field(
        default=None,
        description=(
            "The label column that forms the subgroups of the X-bar and R chart; each maximal "
            "run of one label is cut into consecutive subgroups. Not used by the individuals "
            "chart."
        ),
    )
    subgroup_size: int | None = Field(
        default=None,
        description=(
            "The number of rows in a subgroup of the X-bar and R chart, from 2 to 10; five "
            "when empty. Not used by the individuals chart."
        ),
    )
    chart: Literal["individuals", "xbar_r"] = Field(
        default="individuals",
        description=(
            "The chart to draw: individuals for one value per row read in row order, or xbar_r "
            "for the X-bar and R chart of subgroups."
        ),
    )
    lower_spec: float | None = Field(
        default=None,
        description=(
            "The lower specification limit. Capability indices need both limits; the value "
            "must be below the upper limit."
        ),
        json_schema_extra={"label": "Lower specification limit"},
    )
    upper_spec: float | None = Field(
        default=None,
        description=(
            "The upper specification limit. Capability indices need both limits; the value "
            "must be above the lower limit."
        ),
        json_schema_extra={"label": "Upper specification limit"},
    )
    level: float = Field(
        default=0.95,
        description=(
            "The confidence level, between 0 and 1. This tool reports no interval, so the "
            "level is recorded on the result and used for nothing else."
        ),
    )


class _Finding(NamedTuple):
    """One special-cause pattern: the rule, the 1-based points it names, and what was read."""

    rule: str
    points: tuple[int, ...]
    found: str


class _Capability(NamedTuple):
    """The capability indices and the estimated percentage outside the specification limits."""

    cp: float
    cpu: float
    cpl: float
    cpk: float
    pp: float
    ppu: float
    ppl: float
    ppk: float
    outside: float


class _Chart(NamedTuple):
    """One chart's plotted points, its lines, and the numbers its table shows."""

    values: FloatArray
    """Every value used, in row order: the plotted points themselves, or the subgroups'."""
    points: FloatArray
    """The plotted points: the values, or the subgroup means."""
    centre: float
    sigma: float
    """The within-subgroup standard deviation estimate."""
    sigma_points: float
    """The standard deviation of one plotted point."""
    upper: float
    lower: float
    mean_range: float
    range_label: str
    range_upper: float | None
    range_lower: float | None
    x_label: str
    y_label: str
    unit: str
    """What one plotted point is called: ``point`` or ``subgroup``."""


def _individuals_chart(values: FloatArray, name: str = "Value") -> _Chart:
    """The individuals chart's lines from the values, in the order given.

    The centre line is the mean; the moving ranges are the absolute differences of consecutive
    values, and their mean over ``d2 = 1.128`` estimates the short-term standard deviation.
    """
    centre = float(np.mean(values))
    moving = np.abs(np.diff(values))
    mean_range = float(np.mean(moving))
    constants = CONSTANTS[2]
    sigma = mean_range / constants.d2
    return _Chart(
        values=values,
        points=values,
        centre=centre,
        sigma=sigma,
        sigma_points=sigma,
        upper=centre + LIMIT_WIDTH * sigma,
        lower=centre - LIMIT_WIDTH * sigma,
        mean_range=mean_range,
        range_label="Mean moving range",
        range_upper=None,
        range_lower=None,
        x_label="Observation",
        y_label=name,
        unit="point",
    )


def _xbar_r_chart(groups: FloatArray, name: str = "Value") -> _Chart:
    """The X-bar and R chart's lines from the complete subgroups, one subgroup per row.

    The centre line is the grand mean of the subgroup means; the mean range over ``d2``
    estimates the within-subgroup standard deviation, and the X-bar limits are ``A2`` times
    the mean range about the grand mean. The range chart's limits are ``D3`` and ``D4`` times
    the mean range.
    """
    size = int(groups.shape[1])
    constants = CONSTANTS[size]
    means = np.asarray(np.mean(groups, axis=1), dtype=np.float64)
    ranges = np.asarray(np.ptp(groups, axis=1), dtype=np.float64)
    centre = float(np.mean(means))
    mean_range = float(np.mean(ranges))
    sigma = mean_range / constants.d2
    return _Chart(
        values=groups.reshape(-1),
        points=means,
        centre=centre,
        sigma=sigma,
        sigma_points=sigma / math.sqrt(size),
        upper=centre + constants.a2 * mean_range,
        lower=centre - constants.a2 * mean_range,
        mean_range=mean_range,
        range_label="Mean range",
        range_upper=constants.d4 * mean_range,
        range_lower=constants.d3 * mean_range,
        x_label="Subgroup",
        y_label=f"Mean of {name}",
        unit="subgroup",
    )


def _runs(mask: BoolArray) -> list[tuple[int, int]]:
    """The maximal runs of ``True`` in ``mask``, each as ``(start, stop)`` with stop exclusive."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, inside in enumerate(mask):
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _counted(count: int, unit: str) -> str:
    """``1 point`` or ``3 points``, ``1 subgroup`` or ``3 subgroups``."""
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def _rule_one(points: FloatArray, upper: float, lower: float, unit: str) -> _Finding | None:
    """Rule 1: every point beyond a control limit, with how many lie on each side."""
    beyond = np.flatnonzero((points > upper) | (points < lower))
    if beyond.size == 0:
        return None
    above = int(np.count_nonzero(points[beyond] > upper))
    below = int(beyond.size) - above
    sides: list[str] = []
    if above:
        sides.append(f"{above} above the upper limit")
    if below:
        sides.append(f"{below} below the lower limit")
    found = f"{_counted(int(beyond.size), unit)} beyond a control limit: {', '.join(sides)}"
    return _Finding("Rule 1", tuple(int(index) + 1 for index in beyond), found)


def _rule_two(points: FloatArray, centre: float, sigma: float, unit: str) -> _Finding | None:
    """Rule 2: the later point of every window of three holding two beyond two sigma."""
    high = centre + TWO_SIGMA * sigma
    low = centre - TWO_SIGMA * sigma
    flagged = [
        index + 1
        for index in range(2, int(points.size))
        if np.count_nonzero(points[index - 2 : index + 1] > high) >= 2
        or np.count_nonzero(points[index - 2 : index + 1] < low) >= 2
    ]
    if not flagged:
        return None
    found = (
        f"Each listed {unit} ends a window of three consecutive {unit}s in which two lie "
        "beyond two standard deviations on the same side of the centre line."
    )
    return _Finding("Rule 2", tuple(flagged), found)


def _rule_four(points: FloatArray, centre: float, unit: str) -> tuple[_Finding, ...]:
    """Rule 4: every run of eight or more consecutive points on one side of the centre line."""
    findings: list[_Finding] = []
    for inside, side in ((points > centre, "above"), (points < centre, "below")):
        for start, stop in _runs(inside):
            length = stop - start
            if length >= RULE_FOUR_RUN:
                findings.append(
                    _Finding(
                        "Rule 4",
                        tuple(range(start + 1, stop + 1)),
                        f"{length} consecutive {unit}s lie {side} the centre line.",
                    )
                )
    return tuple(findings)


def _special_causes(
    points: FloatArray, centre: float, sigma: float, upper: float, lower: float, unit: str
) -> tuple[_Finding, ...]:
    """The Western Electric rules 1, 2 and 4 on the plotted points, in that order."""
    findings: list[_Finding] = []
    one = _rule_one(points, upper, lower, unit)
    if one is not None:
        findings.append(one)
    two = _rule_two(points, centre, sigma, unit)
    if two is not None:
        findings.append(two)
    findings.extend(_rule_four(points, centre, unit))
    return tuple(findings)


def _specs(lower: float | None, upper: float | None) -> tuple[float | None, float | None]:
    """The specification limits, once their order and finiteness are checked.

    Raises:
        ToolError: ``INVALID_OPTION`` for a limit that is not finite, or both given with the
            lower one not below the upper.
    """
    for name, value in (("lower", lower), ("upper", upper)):
        if value is not None and not math.isfinite(value):
            raise ToolError(
                "INVALID_OPTION", f"the {name} specification limit must be a finite number"
            )
    if lower is not None and upper is not None and lower >= upper:
        raise ToolError(
            "INVALID_OPTION",
            f"the lower specification limit must be below the upper one, and {num(lower)} is "
            f"not below {num(upper)}",
        )
    return lower, upper


def _capability(
    lower: float, upper: float, mean: float, sigma_within: float, sigma_overall: float
) -> _Capability:
    """Cp, Cpk, Pp and Ppk and the normal's estimated percentage outside the limits."""
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
    return _Capability(
        cp=cp,
        cpu=cpu,
        cpl=cpl,
        cpk=min(cpu, cpl),
        pp=pp,
        ppu=ppu,
        ppl=ppl,
        ppk=min(ppu, ppl),
        outside=outside,
    )


def _verdict(p_value: float) -> Verdict:
    """The normal assumption's verdict from Shapiro-Wilk's p-value."""
    if p_value < FAIL_P:
        return Verdict.FAIL
    if p_value < CAUTION_P:
        return Verdict.CAUTION
    return Verdict.PASS


def _normality(diagnostic: DiagnosticTest) -> AuditCheck:
    """Shapiro-Wilk's W test of the values, as an assumption check of normality."""
    statistic_value = diagnostic.statistic
    p_value = diagnostic.p_value
    if statistic_value is None or p_value is None:
        return assumption(
            "normality",
            "The normality of the values could not be checked",
            Verdict.NOT_APPLICABLE,
            diagnostic.reason or "Shapiro-Wilk's W test could not be run on these values.",
        )
    verdict = _verdict(float(p_value))
    titles = {
        Verdict.PASS: "The values are consistent with a normal distribution",
        Verdict.CAUTION: "The values leave the normal assumption in doubt",
        Verdict.FAIL: "The values show evidence against a normal distribution",
    }
    statistic = f"W = {num(float(statistic_value))}"
    next_step = (
        None
        if verdict is Verdict.PASS
        else "A capability index read from a distribution far from normal can mislead; a "
        "transformation, or a distribution-free index, is worth considering."
    )
    return assumption(
        "normality",
        titles[verdict],
        verdict,
        f"Shapiro-Wilk's W test gives {statistic}, {p_text(float(p_value))}.",
        statistic=statistic,
        threshold=p_threshold(CAUTION_P, FAIL_P),
        next_step=next_step,
    )


def _positions(points: tuple[int, ...]) -> str:
    """The point positions for a sentence, the first eight and a count of the rest."""
    shown = ", ".join(str(point) for point in points[:8])
    if len(points) > 8:
        return f"{shown} and {len(points) - 8} more"
    return shown


def _stability(findings: tuple[_Finding, ...], plotted: int, unit: str) -> AuditCheck:
    """The chart's stability, read from the Western Electric rules that fired."""
    plural = f"{unit}s"
    if not findings:
        return assumption(
            "stability",
            "No special cause is visible on the control chart",
            Verdict.PASS,
            f"None of the {plotted} plotted {plural} lies beyond a control limit, no two of "
            f"three consecutive {plural} lie beyond two standard deviations on one side, and "
            f"no eight consecutive {plural} lie on one side.",
        )
    beyond = any(finding.rule == "Rule 1" for finding in findings)
    summary = "; ".join(f"{finding.rule} at {_positions(finding.points)}" for finding in findings)
    counted = (
        "1 special-cause pattern was found"
        if len(findings) == 1
        else f"{len(findings)} special-cause patterns were found"
    )
    return assumption(
        "stability",
        (
            "The chart shows a point beyond a control limit"
            if beyond
            else "The chart shows a run the rules flag"
        ),
        Verdict.FAIL if beyond else Verdict.CAUTION,
        f"{counted}: {summary}.",
        statistic=f"{len(findings)} patterns",
        threshold="no point beyond a limit; no rule 2 or rule 4 run",
        next_step=(
            "Look at the process at the flagged points: a change of material, machine, method "
            "or measurement often explains a run, and the limits are read as in control only "
            "when the causes are understood."
        ),
    )


def _stability_sentence(findings: tuple[_Finding, ...], plotted: int, unit: str) -> str:
    """What the stability check reads, in one sentence for the interpretation."""
    if not findings:
        return (
            f"The {_counted(plotted, unit)} plotted show no special cause, so the process looks "
            "stable."
        )
    patterns = [f"{finding.rule.lower()} at {_positions(finding.points)}" for finding in findings]
    return f"The process does not look stable: {joined(patterns)}."


def _few_points_warning(plotted: int, unit: str) -> str:
    """The warning for a chart built on fewer points than :data:`WARN_POINTS`."""
    verb = "is" if plotted == 1 else "are"
    return (
        f"Only {_counted(plotted, unit)} {verb} plotted, fewer than {WARN_POINTS}; the "
        "control limits rest on few values and are imprecise."
    )


def _constant_message(chart: str, name: str) -> str:
    """Why a chart with no within-subgroup variation cannot be drawn."""
    if chart == "individuals":
        return (
            f"the column {name!r} holds one value repeated, so its moving range is zero and no "
            "control limit can be drawn"
        )
    return (
        f"every subgroup of the column {name!r} holds one value repeated, so the mean range is "
        "zero and no control limit can be drawn"
    )


def _subgroups(
    values: Column, groups: GroupColumn, size: int
) -> tuple[FloatArray, tuple[ExcludedRow, ...], int]:
    """The complete subgroups, the rows a short piece left out, and how many those are.

    The rows are read in their order. Each maximal run of one label is cut into consecutive
    pieces of ``size`` rows; a piece shorter than ``size`` is left out, each of its rows listed
    with its position and the reason. A row missing a value or a label is left out first, by
    :func:`~curvestudio.tools.stats._base.aligned`, and listed with that reason.
    """
    prepared = aligned(values, groups)
    labels = prepared.groups[0].labels
    positions = prepared.values.rows
    numbers = prepared.values.values
    excluded: list[ExcludedRow] = list(prepared.excluded)
    pieces: list[np.ndarray] = []
    start = 0
    for stop in range(1, int(labels.size) + 1):
        if stop == labels.size or labels[stop] != labels[start]:
            for begin in range(start, stop, size):
                end = min(begin + size, stop)
                if end - begin == size:
                    pieces.append(np.arange(begin, end))
                else:
                    short = (
                        f"in a short subgroup of {_counted(end - begin, 'row')} for a size of "
                        f"{size}"
                    )
                    excluded.extend(
                        ExcludedRow(index=int(positions[at]), reason=short)
                        for at in range(begin, end)
                    )
            start = stop
    if pieces:
        matrix = np.asarray([[numbers[at] for at in piece] for piece in pieces], dtype=np.float64)
    else:
        matrix = np.empty((0, size), dtype=np.float64)
    excluded.sort(key=lambda row: row.index)
    return matrix, tuple(excluded), int(labels.size) - int(matrix.shape[0]) * size


def _xbar_r_rows(
    request: ControlChartRequest, warnings: list[str]
) -> tuple[_Chart, tuple[ExcludedRow, ...], int, str]:
    """The X-bar and R chart, its excluded rows, the rows offered and the result's title."""
    if request.subgroups is None:
        raise ToolError(
            "INVALID_OPTION",
            "the X-bar and R chart needs the subgroups column, and none was given",
        )
    size = DEFAULT_SUBGROUP_SIZE if request.subgroup_size is None else request.subgroup_size
    if not MIN_SUBGROUP_SIZE <= size <= MAX_SUBGROUP_SIZE:
        raise ToolError(
            "INVALID_OPTION",
            f"the subgroup size must be between {MIN_SUBGROUP_SIZE} and {MAX_SUBGROUP_SIZE}, "
            f"not {size}",
        )
    matrix, excluded, short = _subgroups(request.values, request.subgroups, size)
    count = int(matrix.shape[0])
    if count < MIN_SUBGROUPS:
        raise ToolError(
            "TOO_FEW_ROWS",
            f"the X-bar and R chart needs at least {MIN_SUBGROUPS} complete subgroups of "
            f"{size} values, and the rows give {count}",
        )
    if short:
        warnings.append(
            f"{_counted(short, 'row')} fell in a subgroup shorter than the size of {size} and "
            "were left out; they are listed by position."
        )
    chart = _xbar_r_chart(matrix, request.values.name)
    return (
        chart,
        excluded,
        count * size + len(excluded),
        (f"X-bar and R chart of {request.values.name} by {request.subgroups.name}"),
    )


def _statistics(
    chart: _Chart,
    overall: float,
    capability: _Capability | None,
    lower_spec: float | None,
    upper_spec: float | None,
) -> Table:
    """Every number the chart rests on, and the capability indices when they were computed."""
    rows: list[tuple[Cell, ...]] = []
    is_range_chart = chart.range_upper is not None
    if is_range_chart:
        rows.append(("Number of subgroups", int(chart.points.size)))
        rows.append(("Subgroup size", chart.values.size // chart.points.size))
        rows.append(("Grand mean", chart.centre))
    else:
        rows.append(("Number of points", int(chart.points.size)))
        rows.append(("Mean", chart.centre))
    rows.append((chart.range_label, chart.mean_range))
    rows.append(("Within-subgroup standard deviation", chart.sigma))
    if is_range_chart:
        rows.append(("Standard deviation of a subgroup mean", chart.sigma_points))
    rows.append(("Upper control limit", chart.upper))
    rows.append(("Lower control limit", chart.lower))
    if chart.range_upper is not None and chart.range_lower is not None:
        rows.append(("R chart upper control limit", chart.range_upper))
        rows.append(("R chart lower control limit", chart.range_lower))
    rows.append(("Overall standard deviation", overall))
    if capability is not None:
        assert lower_spec is not None and upper_spec is not None
        rows.extend(
            (
                ("Lower specification limit", lower_spec),
                ("Upper specification limit", upper_spec),
                ("Cp", capability.cp),
                ("Cpu", capability.cpu),
                ("Cpl", capability.cpl),
                ("Cpk", capability.cpk),
                ("Pp", capability.pp),
                ("Ppk", capability.ppk),
                ("Estimated values outside the limits %", capability.outside),
            )
        )
    if is_range_chart:
        note = (
            "The centre line is the grand mean of the subgroup means. A subgroup is a "
            "consecutive run of one label in the subgroups column, cut into pieces of the "
            "subgroup size; a piece shorter than that size is left out. The mean range divided "
            "by d2 estimates the within-subgroup standard deviation, and the X-bar limits are "
            "the grand mean plus or minus A2 times the mean range; the range chart's limits are "
            "D3 and D4 times the mean range. Cp and Cpk use the within-subgroup standard "
            "deviation; Pp and Ppk use the sample standard deviation with n - 1 in its "
            "denominator. The percentage outside is read from a normal distribution with the "
            "grand mean and the within-subgroup standard deviation."
        )
    else:
        note = (
            "The centre line is the mean of the values. The moving ranges are the absolute "
            "differences of consecutive values, and the mean moving range divided by d2 = 1.128 "
            "estimates the within-subgroup standard deviation, which puts the limits at the "
            "mean plus or minus 2.66 times the mean moving range. Cp and Cpk use the "
            "within-subgroup standard deviation; Pp and Ppk use the sample standard deviation "
            "with n - 1 in its denominator. The percentage outside is read from a normal "
            "distribution with the mean and the within-subgroup standard deviation."
        )
    return table("Chart statistics", _STATISTIC_COLUMNS, rows, note=note)


def _causes(findings: tuple[_Finding, ...]) -> Table:
    """The special-cause patterns the rules found, one row per pattern."""
    rows = [(finding.rule, _positions(finding.points), finding.found) for finding in findings]
    return table(
        "Special causes",
        _CAUSE_COLUMNS,
        rows,
        note=(
            "The Western Electric rules were applied to the plotted points: rule 1 is a point "
            "beyond a control limit, rule 2 is two of three consecutive points beyond two "
            "standard deviations on the same side of the centre line, and rule 4 is eight "
            "consecutive points on one side. Points are numbered from one in the order plotted."
        ),
    )


def _plot(chart: _Chart) -> MultiLinePlot:
    """The control chart: the points, the centre line and the two limits."""
    points = chart.points
    positions = np.arange(1.0, float(points.size) + 1.0)
    shown_points, note_points = stride(points)
    shown_positions, note_positions = stride(positions)
    last = float(points.size)
    return MultiLinePlot(
        title="Control chart",
        x_label=chart.x_label,
        y_label=chart.y_label,
        x_labels=tuple(str(int(position)) for position in shown_positions),
        series=(
            XYSeries(
                label="Observations",
                x=tuple(float(position) for position in shown_positions),
                y=tuple(float(point) for point in shown_points),
            ),
            XYSeries(label="Centre line", x=(1.0, last), y=(chart.centre, chart.centre)),
            XYSeries(label="Upper limit", x=(1.0, last), y=(chart.upper, chart.upper)),
            XYSeries(label="Lower limit", x=(1.0, last), y=(chart.lower, chart.lower)),
        ),
        note=note_points or note_positions,
    )


def _capability_sentence(capability: _Capability, lower: float, upper: float) -> str:
    """The capability paragraph, with the 1.33 guide named as a convention."""
    return (
        f"With specification limits {num(lower)} and {num(upper)}, about "
        f"{num(capability.outside)} % of values are estimated to fall outside them, and Cpk is "
        f"{num(capability.cpk)}; 1.33 is the figure often used as a guide for an established "
        "process, and it is a convention rather than a requirement."
    )


def _methods(chart: _Chart, capability: bool) -> str:
    """One fragment a methods paragraph can carry as it stands."""
    if chart.range_upper is None:
        chart_words = (
            "The individuals chart plots each value in row order. The centre line is the mean "
            "of the values; the moving ranges are the absolute differences of consecutive "
            "values, and the mean moving range divided by d2 = 1.128 (a moving range of two) "
            "estimates the within-subgroup standard deviation. The limits are the mean plus or "
            "minus three times that estimate, 2.66 times the mean moving range."
        )
    else:
        chart_words = (
            "The X-bar and R chart plots the mean of each subgroup. Subgroups are the maximal "
            "runs of one label in the subgroups column, read in their order and cut into "
            "consecutive pieces of the subgroup size; a piece shorter than that size is left "
            "out and listed. The centre line is the grand mean; the mean range divided by d2 "
            "estimates the within-subgroup standard deviation, and the X-bar limits are the "
            "grand mean plus or minus A2 times the mean range, with the R chart's limits at D3 "
            "and D4 times the mean range. A2, D3 and D4 are the constants for the subgroup size."
        )
    rules = (
        "The Western Electric rules applied to the plotted points are rule 1 (a point beyond a "
        "control limit), rule 2 (two of three consecutive points beyond two standard deviations "
        "on the same side of the centre line) and rule 4 (eight consecutive points on one side "
        "of the centre line)."
    )
    constants = (
        "The control-chart constants A2, D3, D4 and d2 for subgroup sizes 2 to 10 are taken "
        "from Montgomery, Introduction to Statistical Quality Control, Appendix VI (Table VI)."
    )
    if capability:
        capability_words = (
            "Cp and Cpk use the within-subgroup standard deviation; Pp and Ppk use the overall "
            "sample standard deviation, with n - 1 in its denominator. The fraction outside the "
            "specification limits is read from a normal distribution with the mean and the "
            "within-subgroup standard deviation. A capability index describes a process, not a "
            "population, and it is read only when the chart looks stable and the values are "
            "close enough to normal."
        )
    else:
        capability_words = (
            "No capability index was computed: both specification limits are needed. A "
            "capability index describes a process, not a population."
        )
    return " ".join((chart_words, rules, constants, capability_words))


def run_controlchart(request: BaseModel) -> ToolResult:
    """Run the control chart and process capability tool on its own request."""
    if not isinstance(request, ControlChartRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"controlchart takes a ControlChartRequest, not a request of {type(request).__name__}",
        )
    return _analyse(request)


def _analyse(request: ControlChartRequest) -> ToolResult:
    """The chart itself, once the request is known to be this tool's own."""
    check_level(request.level)
    lower_spec, upper_spec = _specs(request.lower_spec, request.upper_spec)
    warnings: list[str] = []
    if request.chart == "individuals":
        if request.subgroups is not None or request.subgroup_size is not None:
            warnings.append(_IGNORED_GROUP_OPTIONS)
        prepared = pairs(request.values)
        values = prepared.columns[0].values
        if values.size < MIN_POINTS:
            raise ToolError(
                "TOO_FEW_ROWS",
                f"the individuals chart needs at least {MIN_POINTS} values, and the column "
                f"{request.values.name!r} holds {values.size} that can be plotted",
            )
        chart = _individuals_chart(values, request.values.name)
        excluded = prepared.excluded
        offered = int(values.size) + len(excluded)
        title = f"Control chart of {request.values.name}"
    else:
        chart, excluded, offered, title = _xbar_r_rows(request, warnings)
    if chart.mean_range == 0.0:
        raise ToolError("CONSTANT_COLUMN", _constant_message(request.chart, request.values.name))

    plotted = int(chart.points.size)
    if plotted < WARN_POINTS:
        warnings.append(_few_points_warning(plotted, chart.unit))
    findings = _special_causes(
        chart.points, chart.centre, chart.sigma_points, chart.upper, chart.lower, chart.unit
    )
    overall = float(np.std(chart.values, ddof=1))
    checks: list[AuditCheck] = [_stability(findings, plotted, chart.unit)]
    capability: _Capability | None = None
    if lower_spec is not None and upper_spec is not None:
        capability = _capability(lower_spec, upper_spec, chart.centre, chart.sigma, overall)
        diagnostic = shapiro_wilk(chart.values)
        checks.append(_normality(diagnostic))
        if diagnostic.p_value is not None and diagnostic.p_value < CAUTION_P:
            warnings.append(
                f"Shapiro-Wilk's W test gives {p_text(diagnostic.p_value)}, so the values are "
                "not consistent with a normal distribution; the capability indices assume one "
                "and may mislead."
            )
    elif (lower_spec is None) != (upper_spec is None):
        missing = "upper" if upper_spec is None else "lower"
        warnings.append(
            f"Only the {missing} specification limit was given; the capability indices need "
            "both limits and were not computed."
        )

    estimates: list[Estimate] = [
        Estimate(label="Centre line", value=chart.centre),
        Estimate(label="Upper control limit", value=chart.upper),
        Estimate(label="Lower control limit", value=chart.lower),
        Estimate(label=chart.range_label, value=chart.mean_range),
    ]
    if capability is not None:
        assert lower_spec is not None and upper_spec is not None
        estimates.extend(
            (
                Estimate(label="Cp", value=capability.cp),
                Estimate(label="Cpk", value=capability.cpk),
                Estimate(label="Pp", value=capability.pp),
                Estimate(label="Ppk", value=capability.ppk),
                Estimate(label="Estimated values outside the limits %", value=capability.outside),
            )
        )

    interpretation = _stability_sentence(findings, plotted, chart.unit)
    if capability is not None:
        assert lower_spec is not None and upper_spec is not None
        interpretation = " ".join(
            (
                interpretation,
                _capability_sentence(capability, lower_spec, upper_spec),
                "A capability index describes a process, not a population, and it is read only "
                "when the process is stable and the values are close enough to normal.",
            )
        )
    else:
        interpretation = " ".join(
            (
                interpretation,
                "A capability index describes a process, not a population, and it needs both "
                "specification limits, a stable process and values close enough to normal.",
            )
        )

    listed, unlisted = list_excluded(excluded)
    tables: list[Table] = [
        _statistics(chart, overall, capability, lower_spec, upper_spec),
    ]
    if findings:
        tables.append(_causes(findings))
    return ToolResult(
        tool="controlchart",
        title=title,
        level=request.level,
        estimates=tuple(estimates),
        assumptions=tuple(checks),
        warnings=tuple(warnings),
        interpretation=interpretation,
        methods_fragment=_methods(chart, capability is not None),
        counts=Counts(n_offered=offered, n_used=int(chart.values.size), n_excluded=len(excluded)),
        excluded=listed,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=(_plot(chart),),
    )


TOOLS.register(
    ToolSpec(
        name="controlchart",
        family="measurement",
        title="Control chart and process capability",
        request=ControlChartRequest,
        run=run_controlchart,
    )
)
