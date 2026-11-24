"""The blocks a result shows: a table, and the plots.

A block holds plain JSON types and no NumPy array. The web app's types for a result are
generated from the API's OpenAPI document, which is produced from these models, and a NumPy
array has no JSON schema. So every array of points is a tuple of finite floats, at most
``MAX_PLOT_POINTS`` long: a plot takes any sequence of numbers and stores the tuple.

This module depends on nothing else of the statistical tools; :mod:`._base` builds the
result on it.
"""

import math
from collections.abc import Sequence
from typing import Annotated, Literal, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from curvestudio.fitting.metrics import FiniteFloat

FloatArray = NDArray[np.float64]

MAX_PLOT_POINTS = 2_000
"""The most points one array of a plot holds; :func:`stride` brings a longer array down to it."""

ColumnKind = Literal["text", "int", "number", "p_value"]
"""How a table cell reads: ``int`` takes a whole number and ``number`` any finite one, and
``p_value`` is a number the renderer writes as a p-value."""

Cell = str | int | float | None
"""A table cell. ``None`` is an empty cell: a number the tool could not give."""

Points = Annotated[tuple[FiniteFloat, ...], Field(max_length=MAX_PLOT_POINTS)]
"""An array of plot points: finite floats, ``MAX_PLOT_POINTS`` at most, stored as a tuple."""


def _count(size: int) -> str:
    return "1 point" if size == 1 else f"{size} points"


def _same_length(model: BaseModel, reference: str, *others: str) -> None:
    """Refuse arrays of different lengths, naming the first that differs from ``reference``."""
    size = len(getattr(model, reference))
    for name in others:
        found = len(getattr(model, name))
        if found != size:
            raise ValueError(f"{name} has {_count(found)} for {size} in {reference}")


_KINDS: dict[str, ColumnKind] = {
    "text": "text",
    "int": "int",
    "number": "number",
    "p_value": "p_value",
}
"""Each kind by the name a column is given with."""


def _kind_of(column: object) -> ColumnKind | None:
    """A column's kind as given: a :class:`TableColumn`, or the dict of one still to be checked."""
    if isinstance(column, TableColumn):
        return column.kind
    if isinstance(column, dict):
        return _KINDS.get(column.get("kind", "number"))
    return None


def _cell(cell: object, kind: ColumnKind | None, row: int, column: int) -> object:
    """The cell as the table stores it, once it is found to be of its column's kind."""
    if cell is None or kind is None:
        return cell
    where = f"row {row}, column {column}"
    if kind == "text":
        if not isinstance(cell, str):
            raise ValueError(f"{where}: {cell!r} is not text")
        return str(cell)
    # A bool is an int to Python and to pydantic, but a table never shows one as a number.
    if isinstance(cell, (bool, np.bool_)):
        raise ValueError(f"{where}: a bool is not a number: {cell!r}")
    number: int | float
    if isinstance(cell, (int, np.integer)):
        number = int(cell)
    elif isinstance(cell, (float, np.floating)):
        number = float(cell)
    else:
        raise ValueError(f"{where}: {cell!r} is not a number")
    if kind == "int" and not isinstance(number, int):
        raise ValueError(f"{where}: {number!r} is not a whole number")
    if not math.isfinite(number):
        raise ValueError(
            f"{where}: {number!r} is NaN or infinite; a table shows an empty cell instead"
        )
    return number


class _Block(BaseModel):
    """What every block shares: frozen, closed to unknown fields, a title and an optional note."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    note: str | None = None


class TableColumn(BaseModel):
    """One column of a table: ``key`` names the cell in a row, ``kind`` says how it reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    label: str
    kind: ColumnKind = "number"


class Table(_Block):
    """A table: its columns, and rows of one cell per column.

    A cell fits its column's kind (a bool is not a number here) and is never NaN or infinite:
    what a tool could not give is an empty cell.
    """

    columns: tuple[TableColumn, ...]
    rows: tuple[tuple[Cell, ...], ...]

    @model_validator(mode="before")
    @classmethod
    def _cells_fit_their_columns(cls, data: object) -> object:
        """Check the cells as they were given, before pydantic could turn a bool or a NumPy
        number into something else."""
        if not isinstance(data, dict):
            return data
        columns = data.get("columns", ())
        given = data.get("rows", ())
        # Anything that is not a list or a tuple is malformed, and pydantic's own refusal
        # of it is the clearer one.
        if not isinstance(columns, (list, tuple)) or not isinstance(given, (list, tuple)):
            return data
        kinds = [_kind_of(column) for column in columns]
        checked = []
        for index, row in enumerate(given):
            if not isinstance(row, (list, tuple)):
                return data
            if len(row) != len(kinds):
                raise ValueError(f"row {index} has {len(row)} cells for {len(kinds)} columns")
            checked.append(
                tuple(
                    _cell(cell, kind, index, column)
                    for column, (cell, kind) in enumerate(zip(row, kinds, strict=True))
                )
            )
        return {**data, "rows": tuple(checked)}


class Line(BaseModel):
    """A straight reference line from ``(x0, y0)`` to ``(x1, y1)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: FiniteFloat
    y0: FiniteFloat
    x1: FiniteFloat
    y1: FiniteFloat
    label: str | None = None


class Band(BaseModel):
    """An interval along a curve: ``lower`` and ``upper`` at every point of ``x``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: Points
    lower: Points
    upper: Points

    @model_validator(mode="after")
    def _one_bound_per_point(self) -> Self:
        _same_length(self, "x", "lower", "upper")
        return self


class ScatterPlot(_Block):
    """Points against each other, with optional reference lines and an optional band."""

    kind: Literal["scatter"] = "scatter"
    x_label: str
    y_label: str
    x: Points
    y: Points
    lines: tuple[Line, ...] = ()
    band: Band | None = None

    @model_validator(mode="after")
    def _y_for_every_x(self) -> Self:
        _same_length(self, "x", "y")
        return self


class Series(BaseModel):
    """A line drawn over a histogram: one value per bin, read at the bin's centre."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    values: Points


class HistogramPlot(_Block):
    """A histogram: counts per bin, with one more edge than bins."""

    kind: Literal["histogram"] = "histogram"
    x_label: str
    y_label: str = "Count"
    edges: Points
    counts: Points
    overlays: tuple[Series, ...] = ()

    @model_validator(mode="after")
    def _one_more_edge_than_bins(self) -> Self:
        bins = len(self.counts)
        if len(self.edges) != bins + 1:
            raise ValueError(
                f"edges has {_count(len(self.edges))} for {bins} bins: a histogram needs one "
                "more edge than bins"
            )
        for index, overlay in enumerate(self.overlays):
            if len(overlay.values) != bins:
                raise ValueError(
                    f"overlays[{index}] ({overlay.label}) has {_count(len(overlay.values))} for "
                    f"{bins} bins: an overlay carries one value per bin"
                )
        return self


class QQPlot(_Block):
    """A quantile-quantile plot: ordered values against theoretical quantiles."""

    kind: Literal["qq"] = "qq"
    theoretical: Points
    ordered: Points
    line: Line

    @model_validator(mode="after")
    def _one_ordered_value_per_quantile(self) -> Self:
        _same_length(self, "theoretical", "ordered")
        return self


class BoxGroup(BaseModel):
    """One box of a box plot: a group's quartiles and median, Tukey's whiskers and outliers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    q1: FiniteFloat
    median: FiniteFloat
    q3: FiniteFloat
    lower_whisker: FiniteFloat
    upper_whisker: FiniteFloat
    outliers: Points = ()


class BoxPlot(_Block):
    """A box per group, all on one axis."""

    kind: Literal["box"] = "box"
    y_label: str
    boxes: tuple[BoxGroup, ...]


class XYSeries(BaseModel):
    """One line of several on a plot: y against x."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    x: Points
    y: Points

    @model_validator(mode="after")
    def _y_for_every_x(self) -> Self:
        _same_length(self, "x", "y")
        return self


class MultiLinePlot(_Block):
    """Several lines on one plot. ``x_labels`` names the x values when x is a category."""

    kind: Literal["lines"] = "lines"
    x_label: str
    y_label: str
    series: tuple[XYSeries, ...]
    x_labels: tuple[str, ...] | None = None


Plot = Annotated[
    ScatterPlot | HistogramPlot | QQPlot | BoxPlot | MultiLinePlot, Field(discriminator="kind")
]
"""Any plot, told apart by its ``kind``: the one field every plot model has."""


def stride(values: FloatArray, *, limit: int = MAX_PLOT_POINTS) -> tuple[FloatArray, str | None]:
    """``values`` as a plot can hold them, and the note that says when some were left out.

    The whole array when it fits ``limit``. Otherwise exactly ``limit`` points, evenly spaced
    from the first to the last, and a note such as ``2,000 of 53,210 points shown, taken at an
    even stride``.
    """
    values = np.asarray(values, dtype=np.float64)
    size = values.size
    if size <= limit:
        return values, None
    taken = np.round(np.linspace(0.0, size - 1.0, limit)).astype(np.intp)
    shown = values[taken]
    return shown, f"{shown.size:,} of {size:,} points shown, taken at an even stride"


class _Read(NamedTuple):
    """One group's box as its values give it, before its outliers are cut to the limit."""

    label: str
    q1: float
    median: float
    q3: float
    lower_whisker: float
    upper_whisker: float
    outliers: FloatArray


def box_plot(title: str, y_label: str, groups: Sequence[tuple[str, FloatArray]]) -> BoxPlot:
    """A box per group, over the group's finite values.

    A box shows the first and third quartile and the median. Tukey's whiskers stop at the
    last values inside ``Q1 - 1.5 IQR`` and ``Q3 + 1.5 IQR``, and the values beyond them are
    the outliers. A group with no finite value is left out. At most ``MAX_PLOT_POINTS``
    outliers are shown, the first groups first, and the note says how many there are.
    """
    read: list[_Read] = []
    for label, values in groups:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        q1, median, q3 = (float(value) for value in np.quantile(finite, [0.25, 0.5, 0.75]))
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        inside = finite[(finite >= low) & (finite <= high)]
        read.append(
            _Read(
                label=str(label),
                q1=q1,
                median=median,
                q3=q3,
                lower_whisker=float(inside.min()),
                upper_whisker=float(inside.max()),
                outliers=finite[(finite < low) | (finite > high)],
            )
        )

    total = sum(box.outliers.size for box in read)
    budget = MAX_PLOT_POINTS
    boxes: list[BoxGroup] = []
    for box in read:
        shown = tuple(float(value) for value in box.outliers[:budget])
        budget -= len(shown)
        boxes.append(
            BoxGroup(
                label=box.label,
                q1=box.q1,
                median=box.median,
                q3=box.q3,
                lower_whisker=box.lower_whisker,
                upper_whisker=box.upper_whisker,
                outliers=shown,
            )
        )
    note = None if total <= MAX_PLOT_POINTS else f"{MAX_PLOT_POINTS:,} of {total:,} outliers shown"
    return BoxPlot(title=title, y_label=y_label, boxes=tuple(boxes), note=note)
