"""Descriptive statistics of one numeric column, as a whole or by group.

The tool reads one column, drops the rows without a finite value (listing them, never
silently), and reports where the values sit and how far they spread: the mean, the sample
standard deviation, the quartiles, the coefficient of variation, the bias-corrected skewness
and excess kurtosis, and the t interval for each mean. A group column, when given, splits the
rows into the groups it labels and summarises each of them; the rows are aligned with the
value column first, so a row needs both a finite value and a label to take part.

Points beyond Tukey's fences are flagged in their own table and kept in every statistic: a
descriptive summary describes the data it was given, and does not decide which points to
believe. A number the tool cannot compute is ``None`` in the table, never a NaN, and each
definition behind a column is written once in the methods fragment.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as scipy_stats

from curvestudio.data import Column, GroupColumn
from curvestudio.data.dataset import IntArray
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import ExcludedRow
from curvestudio.tools._wording import num
from curvestudio.tools.stats._base import (
    Counts,
    ToolResult,
    aligned,
    check_level,
    levels,
    list_excluded,
    pairs,
)
from curvestudio.tools.stats._display import Cell, ColumnKind, FloatArray, box_plot
from curvestudio.tools.stats._tables import (
    GroupSummary,
    confidence_interval,
    group_summary,
    table,
)
from curvestudio.tools.stats.registry import TOOLS, ToolSpec


class DescriptivesRequest(BaseModel):
    """What the descriptives tool takes: one numeric column, optionally by group."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column = Field(description="The numeric column to summarise.")
    groups: GroupColumn | None = Field(
        default=None,
        description="A grouping column; each of its levels is summarised as its own group.",
    )
    level: float = Field(
        default=0.95,
        description="The confidence level for the interval around each mean.",
    )


DESCRIPTIVE_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("group", "Group", "text"),
    ("n", "n", "int"),
    ("missing", "Missing", "int"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
    ("se", "SE of the mean", "number"),
    ("median", "Median", "number"),
    ("q1", "Q1", "number"),
    ("q3", "Q3", "number"),
    ("minimum", "Minimum", "number"),
    ("maximum", "Maximum", "number"),
    ("cv", "CV", "number"),
    ("skewness", "Skewness", "number"),
    ("kurtosis", "Excess kurtosis", "number"),
    ("lower", "Mean lower", "number"),
    ("upper", "Mean upper", "number"),
)
"""The columns of the descriptive table, in the order a person reads them."""

FLAG_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("row", "Row", "int"),
    ("group", "Group", "text"),
    ("value", "Value", "number"),
    ("fence", "Fence", "number"),
)
"""The columns of the table of points beyond Tukey's fences."""

ALL_ROWS = "All rows"
"""The label the one group of an ungrouped column carries."""

Labels = NDArray[np.object_]


def _percent(level: float) -> str:
    return f"{num(level * 100)} %"


def _definitions(level: float) -> tuple[str, ...]:
    """What each column means, as the sentences the table note and the methods text share.

    Each definition is written here once: the note and the methods fragment carry the same
    sentences, so the two can never drift apart.
    """
    return (
        "The sample standard deviation uses n - 1 in its denominator, and the standard error "
        "of the mean is that standard deviation divided by the square root of n.",
        "The median and the quartiles follow NumPy's default definition (linear "
        "interpolation, type 7).",
        "The coefficient of variation is the standard deviation divided by the absolute mean, "
        "and is left undefined when the mean is 0.",
        "Skewness and excess kurtosis are the bias-corrected moment estimators that SciPy "
        "computes, so a normal distribution reads 0 for each.",
        f"At the {_percent(level)} level the interval for each mean is the mean plus or minus "
        "t times the standard error, with t from the t distribution on n - 1 degrees of "
        "freedom.",
        "Tukey's fences sit at Q1 - 1.5 IQR and Q3 + 1.5 IQR, and a point beyond either fence "
        "is flagged, never removed.",
    )


def _note(level: float) -> str:
    return " ".join(_definitions(level))


def _finite_or_none(value: float) -> float | None:
    """``value`` as a table holds it: the number, or ``None`` when it is not finite."""
    return float(value) if math.isfinite(value) else None


class _Group(NamedTuple):
    """One group's summary, its finite values and positions, and the statistics built on them."""

    summary: GroupSummary
    missing: int
    cv: float | None
    skewness: float | None
    kurtosis: float | None
    lower: float | None
    upper: float | None
    values: FloatArray
    rows: IntArray


def _group_statistics(
    label: str, values: FloatArray, rows: IntArray, missing: int, level: float
) -> _Group:
    """The summary of one group's finite ``values`` and the statistics the summary lacks.

    ``values`` holds finite values only. The coefficient of variation needs a nonzero mean,
    skewness needs three values and excess kurtosis four, and the interval for the mean needs
    two: a statistic whose condition fails is ``None``.
    """
    summary = group_summary(label, values)
    if summary.count == 1:
        # A group of one gives count, mean, median, minimum and maximum, and `None` for the
        # rest: a single value has no quartiles to describe a spread with.
        summary = summary.model_copy(update={"q1": None, "q3": None})
    count, mean, sd, se = summary.count, summary.mean, summary.sd, summary.se
    cv = None if sd is None or mean is None or mean == 0 else _finite_or_none(sd / abs(mean))
    skewness = None
    kurtosis = None
    if sd is not None and sd > 0:
        if count >= 3:
            skewness = _finite_or_none(float(scipy_stats.skew(values, bias=False)))
        if count >= 4:
            kurtosis = _finite_or_none(float(scipy_stats.kurtosis(values, bias=False)))
    lower = upper = None
    if count >= 2 and mean is not None and se is not None:
        lower, upper = confidence_interval(mean, se, float(count - 1), level)
    return _Group(
        summary=summary,
        missing=missing,
        cv=cv,
        skewness=skewness,
        kurtosis=kurtosis,
        lower=lower,
        upper=upper,
        values=values,
        rows=rows,
    )


def _missing_by_group(values: Column, groups: GroupColumn) -> dict[str, int]:
    """How many rows in each group carry no finite value, by the group's label.

    A row without a label belongs to no group and is not counted here; it is listed as
    excluded instead.
    """
    label_at = {int(row): str(label) for row, label in zip(groups.rows, groups.labels, strict=True)}
    counts: dict[str, int] = {}
    for row, value in zip(values.rows, values.values, strict=True):
        label = label_at.get(int(row))
        if label is None or math.isfinite(float(value)):
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


class _Kept(NamedTuple):
    """The rows a tool carries forward: the finite values, their positions and their labels."""

    values: FloatArray
    rows: IntArray
    labels: Labels | None
    group_names: tuple[str, ...]
    missing: dict[str, int]
    offered: int
    excluded_rows: tuple[ExcludedRow, ...]


def _read(request: DescriptivesRequest) -> _Kept:
    """Align the request's columns and hand back the rows that take part.

    Raises:
        ToolError: ``NO_DATA`` when no row holds both a finite value and (with groups) a
            group label.
    """
    offered = int(np.unique(request.values.rows).size)
    if request.groups is None:
        pairing = pairs(request.values)
        kept = pairing.columns[0]
        if kept.values.size == 0:
            raise ToolError(
                "NO_DATA", f"{request.values.name!r} holds no finite value to summarise"
            )
        return _Kept(
            values=kept.values,
            rows=kept.rows,
            labels=None,
            group_names=(ALL_ROWS,),
            missing={ALL_ROWS: len(pairing.excluded)},
            offered=offered,
            excluded_rows=pairing.excluded,
        )
    paired = aligned(request.values, request.groups)
    if paired.values.values.size == 0:
        raise ToolError(
            "NO_DATA",
            f"no row of {request.values.name!r} holds both a finite value and a label in "
            f"{request.groups.name!r}",
        )
    return _Kept(
        values=paired.values.values,
        rows=paired.values.rows,
        labels=paired.groups[0].labels,
        group_names=levels(request.groups, minimum=1),
        missing=_missing_by_group(request.values, request.groups),
        offered=offered,
        excluded_rows=paired.excluded,
    )


def _groups(kept: _Kept, level: float) -> list[_Group]:
    """One summary per group, in the order the labels first appear."""
    found: list[_Group] = []
    for label in kept.group_names:
        if kept.labels is None:
            values, rows = kept.values, kept.rows
        else:
            mask = kept.labels == label
            values, rows = kept.values[mask], kept.rows[mask]
        found.append(_group_statistics(label, values, rows, kept.missing.get(label, 0), level))
    return found


def _flagged(groups: Sequence[_Group]) -> list[tuple[int, str, float, float]]:
    """Every point beyond Tukey's fences: its position, group, value and the fence it crossed.

    The rows come in position order.
    """
    flagged: list[tuple[int, str, float, float]] = []
    for group in groups:
        q1, q3 = group.summary.q1, group.summary.q3
        if q1 is None or q3 is None:
            continue
        low = q1 - 1.5 * (q3 - q1)
        high = q3 + 1.5 * (q3 - q1)
        label = group.summary.label
        for value, row in zip(group.values, group.rows, strict=True):
            if float(value) < low:
                flagged.append((int(row), label, float(value), float(low)))
            elif float(value) > high:
                flagged.append((int(row), label, float(value), float(high)))
    flagged.sort(key=lambda entry: entry[0])
    return flagged


def _row(group: _Group) -> tuple[Cell, ...]:
    """One row of the descriptive table, a cell per column."""
    summary = group.summary
    return (
        summary.label,
        summary.count,
        group.missing,
        summary.mean,
        summary.sd,
        summary.se,
        summary.median,
        summary.q1,
        summary.q3,
        summary.minimum,
        summary.maximum,
        group.cv,
        group.skewness,
        group.kurtosis,
        group.lower,
        group.upper,
    )


def _interpretation(name: str, groups: Sequence[_Group], grouped: bool) -> str:
    """One to three sentences: where the mean and the spread are largest."""
    if not grouped:
        summary = groups[0].summary
        if summary.mean is None:
            return f"{name} holds no value to summarise."
        if summary.sd is None:
            return f"The mean of {name} is {num(summary.mean)}; a single value gives no spread."
        return (
            f"The mean of {name} is {num(summary.mean)}, with a sample standard deviation of "
            f"{num(summary.sd)}."
        )

    parts: list[str] = []
    means: list[tuple[float, str]] = []
    spreads: list[tuple[float, str]] = []
    for group in groups:
        mean, sd = group.summary.mean, group.summary.sd
        if mean is not None:
            means.append((mean, group.summary.label))
        if sd is not None:
            spreads.append((sd, group.summary.label))

    if means:
        high_mean, high_label = max(means)
        low_mean, low_label = min(means)
        if high_mean == low_mean:
            parts.append(f"The mean of {name} is {num(high_mean)} in every group with a mean.")
        elif len(means) == 1:
            parts.append(f"The mean of {name} is {num(high_mean)} in {high_label}.")
        else:
            parts.append(
                f"The mean of {name} is highest in {high_label}, at {num(high_mean)}, and "
                f"lowest in {low_label}, at {num(low_mean)}."
            )

    if spreads:
        high_sd, high_label = max(spreads)
        low_sd, low_label = min(spreads)
        if high_sd == low_sd:
            parts.append(f"The sample standard deviation is {num(high_sd)} in every group.")
        elif len(spreads) == 1:
            parts.append(f"The sample standard deviation is {num(high_sd)} in {high_label}.")
        else:
            parts.append(
                f"The sample standard deviation is largest in {high_label}, at {num(high_sd)}, "
                f"and smallest in {low_label}, at {num(low_sd)}."
            )
    else:
        parts.append("No group holds more than one value, so no spread can be estimated.")
    return " ".join(parts)


def _warnings(groups: Sequence[_Group], grouped: bool) -> list[str]:
    """What the result says about a summary the data were too thin for."""
    total = sum(group.summary.count for group in groups)
    said: list[str] = []
    if total < 3:
        noun = "value" if total == 1 else "values"
        said.append(f"Only {total} {noun} available, so some statistics are left empty.")
    if grouped:
        for group in groups:
            if group.summary.count == 1:
                said.append(
                    f"Group {group.summary.label!r} holds a single value, so its spread cannot "
                    "be estimated."
                )
    return said


def _run(request: DescriptivesRequest) -> ToolResult:
    """Run the descriptives tool on its typed request."""
    check_level(request.level)
    kept = _read(request)
    grouped = request.groups is not None
    groups = _groups(kept, request.level)
    total = sum(group.summary.count for group in groups)

    tables = [
        table(
            "Descriptive statistics",
            DESCRIPTIVE_COLUMNS,
            map(_row, groups),
            note=_note(request.level),
        )
    ]
    flagged = _flagged(groups)
    if flagged:
        tables.append(
            table(
                "Points outside Tukey's fences",
                FLAG_COLUMNS,
                flagged,
                note=(
                    "A point beyond Q1 - 1.5 IQR or Q3 + 1.5 IQR is flagged here and kept in "
                    "every statistic."
                ),
            )
        )

    title = f"Box plot of {request.values.name}"
    if request.groups is not None:
        title += f" by {request.groups.name}"
    plot = box_plot(
        title,
        request.values.name,
        [(group.summary.label, group.values) for group in groups],
    )

    what = f"the column {request.values.name!r}"
    if request.groups is not None:
        what += f", by {request.groups.name!r}"
    methods = f"Descriptive statistics were computed for {what}. " + " ".join(
        _definitions(request.level)
    )

    excluded, unlisted = list_excluded(kept.excluded_rows)
    return ToolResult(
        tool="descriptives",
        title=(
            f"Descriptive statistics of {request.values.name}"
            if request.groups is None
            else f"Descriptive statistics of {request.values.name} by {request.groups.name}"
        ),
        level=request.level,
        warnings=tuple(_warnings(groups, grouped)),
        interpretation=_interpretation(request.values.name, groups, grouped),
        methods_fragment=methods,
        counts=Counts(
            n_offered=kept.offered,
            n_used=total,
            n_excluded=len(kept.excluded_rows),
        ),
        excluded=excluded,
        excluded_unlisted=unlisted,
        tables=tuple(tables),
        plots=(plot,),
    )


def run_descriptives(request: BaseModel) -> ToolResult:
    """The registry's entry point: the tool's own request, or ``INVALID_OPTION``.

    Raises:
        ToolError: ``INVALID_OPTION`` when ``request`` is not a :class:`DescriptivesRequest`.
    """
    if not isinstance(request, DescriptivesRequest):
        raise ToolError(
            "INVALID_OPTION",
            f"descriptives takes a DescriptivesRequest, not a request of {type(request).__name__}",
        )
    return _run(request)


SPEC = ToolSpec(
    name="descriptives",
    family="summarise",
    title="Descriptive statistics",
    request=DescriptivesRequest,
    run=run_descriptives,
)
TOOLS.register(SPEC)
