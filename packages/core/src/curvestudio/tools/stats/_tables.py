"""The tables a result shows, and the numbers in them that every statistical tool shares."""

import math
from collections.abc import Iterable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import stats as scipy_stats

from curvestudio.tools.stats._base import Estimate, HypothesisTest, check_level
from curvestudio.tools.stats._display import Cell, ColumnKind, FloatArray, Table, TableColumn

GROUP_COLUMNS: tuple[tuple[str, str, ColumnKind], ...] = (
    ("group", "Group", "text"),
    ("n", "n", "int"),
    ("mean", "Mean", "number"),
    ("sd", "SD", "number"),
    ("se", "SE of the mean", "number"),
    ("median", "Median", "number"),
    ("q1", "Q1", "number"),
    ("q3", "Q3", "number"),
    ("minimum", "Minimum", "number"),
    ("maximum", "Maximum", "number"),
)
"""The columns of the summary by group, as :func:`table` takes them."""


def _finite(values: FloatArray) -> FloatArray:
    """The finite values of ``values``, in their order: a tool reads no other."""
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _percent(level: float) -> str:
    return f"{level * 100:g} %"


def table(
    title: str,
    columns: Sequence[tuple[str, str, ColumnKind]],
    rows: Iterable[Sequence[Cell]],
    *,
    note: str | None = None,
) -> Table:
    """A table from its columns, each given as ``(key, label, kind)``, and its rows.

    Raises:
        ValueError: a row whose length differs from the columns' (a :class:`Table` refuses it,
            and its refusal is a ``ValueError``); a cell that does not fit its column's kind.
    """
    return Table(
        title=title,
        note=note,
        columns=tuple(TableColumn(key=key, label=label, kind=kind) for key, label, kind in columns),
        rows=tuple(tuple(row) for row in rows),
    )


class GroupSummary(BaseModel):
    """The numbers of one group; ``None`` where a number is undefined.

    A group of one has no SD and so no SE of the mean; an empty group has nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    count: int
    mean: float | None
    sd: float | None
    se: float | None
    median: float | None
    q1: float | None
    q3: float | None
    minimum: float | None
    maximum: float | None


def group_summary(label: str, values: FloatArray) -> GroupSummary:
    """The numbers of one group, over the finite values of ``values``."""
    finite = _finite(values)
    count = int(finite.size)
    if count == 0:
        return GroupSummary(
            label=str(label),
            count=0,
            mean=None,
            sd=None,
            se=None,
            median=None,
            q1=None,
            q3=None,
            minimum=None,
            maximum=None,
        )
    q1, median, q3 = (float(value) for value in np.quantile(finite, [0.25, 0.5, 0.75]))
    sd = None if count < 2 else float(np.std(finite, ddof=1))
    return GroupSummary(
        label=str(label),
        count=count,
        mean=float(np.mean(finite)),
        sd=sd,
        se=None if sd is None else sd / math.sqrt(count),
        median=median,
        q1=q1,
        q3=q3,
        minimum=float(finite.min()),
        maximum=float(finite.max()),
    )


def group_table(summaries: Sequence[GroupSummary], *, level: float) -> Table:
    """The summary by group: one row per group, and a note naming the definitions.

    Raises:
        ToolError: ``INVALID_OPTION`` for a confidence level outside (0, 1).
    """
    check_level(level)
    return table(
        "Summary by group",
        GROUP_COLUMNS,
        (
            (
                summary.label,
                summary.count,
                summary.mean,
                summary.sd,
                summary.se,
                summary.median,
                summary.q1,
                summary.q3,
                summary.minimum,
                summary.maximum,
            )
            for summary in summaries
        ),
        note=(
            "SD is the sample standard deviation, with n - 1 in its denominator; the SE of the "
            "mean is SD divided by the square root of n. The median and the quartiles follow "
            f"NumPy's default definition (type 7). At the {_percent(level)} level the interval "
            "for a mean is the mean plus or minus t times the SE, with t from the t "
            "distribution on n - 1 degrees of freedom."
        ),
    )


def holm(tests: Sequence[HypothesisTest]) -> tuple[HypothesisTest, ...]:
    """``tests`` with Holm's step-down adjusted p-values, in the order given.

    Each test that carries a p-value comes back as a new :class:`HypothesisTest` whose
    ``adjusted_p`` is set and whose ``adjustment`` reads ``Holm``. A test with no p-value is
    left as it is, and counts for nothing in the family.
    """
    carrying: list[tuple[float, int]] = []
    for index, test in enumerate(tests):
        if test.p_value is not None:
            carrying.append((test.p_value, index))
    carrying.sort()
    family = len(carrying)
    running = 0.0
    adjusted: dict[int, float] = {}
    for step, (p_value, index) in enumerate(carrying, start=1):
        running = max(running, (family - step + 1) * p_value)
        adjusted[index] = min(1.0, running)
    return tuple(
        HypothesisTest(**{**test.model_dump(), "adjusted_p": adjusted[index], "adjustment": "Holm"})
        if index in adjusted
        else test
        for index, test in enumerate(tests)
    )


def confidence_interval(estimate: float, se: float, df: float, level: float) -> tuple[float, float]:
    """The interval ``estimate - t * SE`` to ``estimate + t * SE`` at ``level``, with ``t``
    from the t distribution on ``df`` degrees of freedom.

    Raises:
        ToolError: ``INVALID_OPTION`` for a confidence level outside (0, 1).
    """
    check_level(level)
    t = float(scipy_stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    return (estimate - t * se, estimate + t * se)


def mean_ci(values: FloatArray, label: str, level: float) -> Estimate:
    """The mean of the finite values of ``values`` with its SE and interval by the t
    distribution. One value gives the mean alone: a spread needs two.

    Raises:
        ValueError: no finite value to average.
        ToolError: ``INVALID_OPTION`` for a confidence level outside (0, 1).
    """
    check_level(level)
    finite = _finite(values)
    if finite.size == 0:
        raise ValueError(f"no finite value to average for {label!r}")
    mean = float(np.mean(finite))
    if finite.size < 2:
        return Estimate(label=label, value=mean)
    se = float(np.std(finite, ddof=1)) / math.sqrt(finite.size)
    lower, upper = confidence_interval(mean, se, float(finite.size - 1), level)
    return Estimate(label=label, value=mean, se=se, lower=lower, upper=upper)


def quantiles(values: FloatArray) -> tuple[float, float]:
    """The first and third quartile of the finite values of ``values``, by NumPy's default
    method (type 7).

    Raises:
        ValueError: no finite value to read.
    """
    finite = _finite(values)
    if finite.size == 0:
        raise ValueError("no finite value to take the quartiles of")
    lower, upper = (float(value) for value in np.quantile(finite, [0.25, 0.75]))
    return lower, upper
