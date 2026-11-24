"""What every statistical tool returns, and the helpers the tools share.

A tool is a pure function: it takes a typed request and returns ONE :class:`ToolResult`, built
from blocks (estimates, tests, assumptions, tables, plots). One renderer shows that result on
the command line, in the desktop app and on the web, so nothing a tool computes can lie
outside what they show. A number a tool cannot give is ``None`` with a reason, never NaN: a
NaN or infinite number is a fault of the tool, and the result refuses it where the tool runs.

Rows a tool leaves out are listed, never dropped without a word: :func:`pairs` and
:func:`aligned` do the leaving out and say why, and :func:`list_excluded` bounds the list.

The :class:`~curvestudio.tools.ToolError` codes the statistical tools raise:

- ``NO_DATA``: a column the tool was given holds no value at all.
- ``TOO_FEW_ROWS``: fewer usable rows than the tool needs.
- ``SINGLE_GROUP``: a group column holds fewer levels than the comparison needs.
- ``GROUP_OF_ONE``: a group holds a single row, so its spread cannot be estimated.
- ``NON_NUMERIC``: a column that should hold numbers holds something else.
- ``INVALID_OPTION``: an option outside what the tool takes; also a request that is not the
  tool's own, or a key that is not one of its fields.
- ``TOO_MANY_ROWS``: more rows than the tool is built for (``ToolSpec.max_rows``).
- ``CONSTANT_COLUMN``: a column that does not vary, so a statistic is undefined.
- ``LEVELS_REQUIRED``: a grouping the tool cannot run without was not given.
- ``UNKNOWN_TOOL``: a name the registry does not hold.
- ``UNKNOWN_COLUMN``: a column name that is not in the dataset (the data layer's code, raised
  again).
- ``COLLINEAR``: columns that are linearly dependent, so their separate effects cannot be
  estimated.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from curvestudio.data import Column, GroupColumn
from curvestudio.data.dataset import IntArray
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.tools._errors import ToolError
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import num
from curvestudio.tools.stats._display import FloatArray, Plot, Table

BoolArray = NDArray[np.bool_]

MAX_LISTED_EXCLUDED = 200
"""The most excluded rows a result lists by position; the rest are counted."""


class Estimate(BaseModel):
    """One estimated number, with its standard error and interval where the tool has them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    value: FiniteFloat
    """The estimate itself: a tool that has none gives no :class:`Estimate`."""
    se: float | None = None
    lower: float | None = None
    upper: float | None = None


class Test(BaseModel):
    """One hypothesis test as every surface shows it.

    ``name`` is short, for a table; ``method`` is exact, for the methods text. A test that
    could not run gives ``statistic`` and ``p_value`` as ``None`` and says why in ``reason``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    method: str
    null_hypothesis: str
    statistic: float | None
    df: float | None = None
    df_denominator: float | None = None
    """An F test's denominator degrees of freedom; ``df`` is then the numerator's."""
    p_value: float | None = None
    p_value_text: str
    """What ``p_value`` reads as, from :func:`p_text`: the same on every surface."""
    effect: Estimate | None = None
    reason: str | None = None
    adjusted_p: float | None = None
    adjustment: str | None = None
    """The multiple-comparison adjustment ``adjusted_p`` went through, by name."""


class Counts(BaseModel):
    """How many rows the tool was offered, used and left out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_offered: int
    n_used: int
    n_excluded: int


def _non_finite(value: object, where: str) -> str | None:
    """The path to the first NaN or infinite float in ``value``, or ``None`` when there is none.

    Walks every field of a nested model and every element of a container; a NumPy array is
    read as a whole when it holds floating-point numbers.
    """
    if isinstance(value, float):
        return None if math.isfinite(value) else where
    if isinstance(value, BaseModel):
        return _first(
            _non_finite(getattr(value, name), f"{where}.{name}")
            for name in type(value).model_fields
        )
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "fc" or np.isfinite(value).all():
            return None
        return f"{where}[{int(np.flatnonzero(~np.isfinite(value).reshape(-1))[0])}]"
    if isinstance(value, Mapping):
        return _first(_non_finite(item, f"{where}[{key!r}]") for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _first(_non_finite(item, f"{where}[{index}]") for index, item in enumerate(value))
    return None


def _first(found: Iterable[str | None]) -> str | None:
    return next((path for path in found if path is not None), None)


def _refuse_non_finite[ModelT: BaseModel](model: ModelT) -> ModelT:
    """A validator for a result: a NaN or infinite number anywhere in it is a fault of the tool."""
    path = _first(_non_finite(getattr(model, name), name) for name in type(model).model_fields)
    if path is not None:
        raise ValueError(
            f"{path} is NaN or infinite: a tool gives None for a number it cannot compute, "
            "and says why"
        )
    return model


class ToolResult(BaseModel):
    """What one statistical tool returns: the only thing a renderer is shown.

    ``interpretation`` is one to three plain sentences for a person; ``methods_fragment`` is
    one a methods paragraph can carry as it stands. ``level`` is the confidence level the
    intervals were built at. ``excluded`` lists the rows the tool left out and why: a tool
    gives it what :func:`list_excluded` returns, at most :data:`MAX_LISTED_EXCLUDED` rows,
    and ``excluded_unlisted`` counts the rest. No tool subclasses this: a result is built from
    its blocks and nothing else.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str
    title: str
    level: float
    estimates: tuple[Estimate, ...] = ()
    tests: tuple[Test, ...] = ()
    assumptions: tuple[AuditCheck, ...] = ()
    warnings: tuple[str, ...] = ()
    interpretation: str
    methods_fragment: str
    counts: Counts
    excluded: tuple[ExcludedRow, ...] = ()
    excluded_unlisted: int = 0
    tables: tuple[Table, ...] = ()
    plots: tuple[Plot, ...] = ()

    _finite_everywhere = model_validator(mode="after")(_refuse_non_finite)


class Pairs(BaseModel):
    """Columns aligned on the positions they all hold, and the positions they do not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: tuple[Column, ...]
    """The columns as given, each over the positions they all hold, in the order given."""
    excluded: tuple[ExcludedRow, ...]


class Aligned(BaseModel):
    """A value column with group columns aligned on the rows they all cover."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Column
    groups: tuple[GroupColumn, ...]
    excluded: tuple[ExcludedRow, ...]


class _Kept:
    """Which positions are still kept as the columns are read one after another, and why
    the others went."""

    def __init__(self, positions: IntArray) -> None:
        self.positions = positions
        self.kept = np.ones(positions.size, dtype=bool)
        self.reasons: dict[int, str] = {}

    def drop(self, at_fault: BoolArray, reason: str) -> None:
        """Drop the positions ``at_fault`` marks that are still kept, each for ``reason``.

        A position an earlier column already dropped keeps that column's reason.
        """
        at_fault = at_fault & self.kept
        for at in np.flatnonzero(at_fault):
            self.reasons[int(at)] = reason
        self.kept[at_fault] = False

    def excluded(self) -> tuple[ExcludedRow, ...]:
        return tuple(
            ExcludedRow(index=int(self.positions[at]), reason=self.reasons[at])
            for at in sorted(self.reasons)
        )


def _union(*arrays: IntArray) -> IntArray:
    """Every position any of ``arrays`` names, each once, in position order."""
    if not arrays:
        return np.empty(0, dtype=np.intp)
    return np.unique(np.concatenate(arrays))


def _spread[ScalarT: np.generic](
    positions: IntArray, rows: IntArray, values: NDArray[ScalarT], blank: Any
) -> tuple[BoolArray, NDArray[ScalarT]]:
    """``values`` laid out over ``positions``: which ones ``rows`` covers, and the value at each.

    A position ``rows`` does not cover reads ``blank``.
    """
    covered = np.zeros(positions.size, dtype=bool)
    spread = np.full(positions.size, blank, dtype=values.dtype)
    _common, here, there = np.intersect1d(positions, rows, return_indices=True)
    covered[here] = True
    spread[here] = values[there]
    return covered, spread


def pairs(*columns: Column) -> Pairs:
    """Align ``columns`` on the positions they all hold a finite value for.

    A position that is missing in ANY column (not among its ``rows``, or NaN in its
    ``values``), or that holds a value that is not finite, is left out of all of them and
    listed with the reason, named after the first column (in the order given) at fault:
    ``missing in <name>`` or ``not a finite number in <name>``. The excluded rows come in
    position order.
    """
    positions = _union(*(column.rows for column in columns))
    rows = _Kept(positions)
    spread: list[tuple[BoolArray, FloatArray]] = []
    for column in columns:
        covered, values = _spread(positions, column.rows, column.values, np.nan)
        spread.append((covered, values))
        rows.drop(~covered | np.isnan(values), f"missing in {column.name}")
        rows.drop(np.isinf(values), f"not a finite number in {column.name}")
    kept = rows.kept
    return Pairs(
        columns=tuple(
            Column(name=column.name, values=values[kept], rows=positions[kept])
            for column, (_covered, values) in zip(columns, spread, strict=True)
        ),
        excluded=rows.excluded(),
    )


def aligned(values: Column, *groups: GroupColumn) -> Aligned:
    """Align ``values`` with the ``groups``, on the rows that have both.

    A row is used only when it has a finite value and a label in EVERY group column. A row
    without a value is listed as ``missing in <name>`` or ``not a finite number in <name>``,
    a row without a label as ``no group label in <name>``, the first group (in the order
    given) without one. The rows are the value column's ``rows``; the excluded ones come in
    position order.
    """
    positions = np.unique(values.rows)
    rows = _Kept(positions)
    covered, spread = _spread(positions, values.rows, values.values, np.nan)
    rows.drop(~covered | np.isnan(spread), f"missing in {values.name}")
    rows.drop(np.isinf(spread), f"not a finite number in {values.name}")
    labels: list[tuple[BoolArray, NDArray[np.object_]]] = []
    for group in groups:
        has, text = _spread(positions, group.rows, group.labels, "")
        labels.append((has, text))
        rows.drop(~has, f"no group label in {group.name}")
    kept = rows.kept
    return Aligned(
        values=Column(name=values.name, values=spread[kept], rows=positions[kept]),
        groups=tuple(
            GroupColumn(name=group.name, labels=text[kept], rows=positions[kept])
            for group, (_has, text) in zip(groups, labels, strict=True)
        ),
        excluded=rows.excluded(),
    )


def levels(groups: GroupColumn, *, minimum: int = 2) -> tuple[str, ...]:
    """The distinct labels of ``groups``, in order of first appearance.

    Raises:
        ToolError: ``SINGLE_GROUP`` when fewer than ``minimum`` labels are found, naming how
            many there are.
    """
    found, first = np.unique(groups.labels, return_index=True)
    ordered = found[np.argsort(first)]
    if ordered.size < minimum:
        raise ToolError(
            "SINGLE_GROUP",
            f"the group column {groups.name!r} holds {ordered.size} "
            f"{'label' if ordered.size == 1 else 'labels'}, and this tool needs at least "
            f"{minimum}",
        )
    return tuple(str(label) for label in ordered)


def check_level(level: float) -> None:
    """Refuse a confidence level outside (0, 1) with ``ToolError("INVALID_OPTION", ...)``."""
    if not 0.0 < level < 1.0:
        raise ToolError(
            "INVALID_OPTION", f"the confidence level must lie between 0 and 1, not {level}"
        )


def list_excluded(rows: Sequence[ExcludedRow]) -> tuple[tuple[ExcludedRow, ...], int]:
    """The first :data:`MAX_LISTED_EXCLUDED` of ``rows``, and how many follow them."""
    listed = tuple(rows[:MAX_LISTED_EXCLUDED])
    return listed, len(rows) - len(listed)


_MEANING: dict[Verdict, str] = {
    Verdict.PASS: "The data are consistent with this assumption.",
    Verdict.CAUTION: "The data leave this assumption in doubt.",
    Verdict.FAIL: "The data contradict this assumption.",
    Verdict.NOT_APPLICABLE: "This assumption could not be read from the data.",
}
"""What a verdict means for any assumption of a test; the check's own words go elsewhere."""


def assumption(
    check_id: str,
    title: str,
    verdict: Verdict,
    evidence: str,
    *,
    statistic: str | None = None,
    threshold: str | None = None,
    next_step: str | None = None,
) -> AuditCheck:
    """One assumption of a test, as a check.

    ``title`` is a statement of what was found (the fit audit's convention), never the name
    of a test: the test that was run belongs in ``evidence``. The explanation is what the
    verdict means for an assumption, the same few sentences for every tool.
    """
    return AuditCheck(
        id=check_id,
        title=title,
        verdict=verdict,
        statistic=statistic,
        threshold=threshold,
        evidence=evidence,
        explanation=_MEANING[verdict],
        next_step=next_step,
    )


def p_text(p: float | None) -> str:
    """How a p-value reads, the same on every surface (the web app keeps a copy of this rule).

    ``None`` is ``not computed``. Below 0.0001 only a bound is shown; from there to 0.1 three
    significant digits are kept, and above that three decimals.
    """
    if p is None:
        return "not computed"
    if p == 0.0:
        return "p < 1e-300"
    if p < 1e-4:
        return "p < 0.0001"
    if p < 0.1:
        return f"p = {num(p)}"
    return f"p = {p:.3f}"
