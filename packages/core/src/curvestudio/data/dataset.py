"""Immutable dataset with a reversible log of operations."""

import datetime
import hashlib
import numbers
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from curvestudio.data.errors import DataError
from curvestudio.data.operations import ExcludeRows, Operation, RenameColumn, TransformColumn

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]
StrArray = NDArray[np.str_]


@dataclass(frozen=True, eq=False)
class XY:
    """Validated, read-only numeric arrays ready for fitting.

    Constructing one validates and normalises the arrays: they become read-only
    float64 copies, must be one-dimensional and equal-length, hold only finite values,
    and weights must be ``>= 0`` (zero weights are supported: such points appear in
    ``fitted``/``residuals`` but influence nothing). Row numbers in the raised
    :class:`DataError` are 0-based positions in these arrays; :meth:`Dataset.xy`
    reports original-table positions instead, because it checks the table first.
    """

    x: FloatArray
    y: FloatArray
    weights: FloatArray | None
    x_name: str
    y_name: str
    fingerprint: str

    def __post_init__(self) -> None:
        arrays: dict[str, Any] = {"x": self.x, "y": self.y}
        if self.weights is not None:
            arrays["weights"] = self.weights
        columns: dict[str, FloatArray] = {}
        for name, values in arrays.items():
            column = np.asarray(values, dtype=np.float64)
            if column.ndim != 1:
                raise DataError(
                    "INVALID_SHAPE", f"{name} must be one-dimensional, got shape {column.shape}"
                )
            columns[name] = column
        sizes = {name: column.size for name, column in columns.items()}
        if len(set(sizes.values())) > 1:
            detail = ", ".join(f"{name}={size}" for name, size in sizes.items())
            raise DataError(
                "LENGTH_MISMATCH", f"x, y and weights must have equal lengths ({detail})"
            )
        for name, column in columns.items():
            _require_finite(name, column)
        if "weights" in columns:
            negative = np.flatnonzero(columns["weights"] < 0.0)
            if negative.size:
                raise DataError(
                    "INVALID_WEIGHT",
                    "weights must be >= 0; use weight 0 to leave an observation out of the fit",
                    tuple(int(i) for i in negative),
                )
        for name, column in columns.items():
            object.__setattr__(self, name, _frozen(column))


class Column(BaseModel):
    """One numeric column of a dataset over the rows that take part, as plain arrays.

    :meth:`Dataset.column` builds it for the statistical tools, which take NumPy
    arrays and never pandas. ``values`` holds one float per entry of ``rows`` (0-based
    positions in :meth:`Dataset.table`); a blank cell or one that is not a number is
    NaN, never an error, while an infinite cell stays infinite. Every array is a
    detached read-only copy, also when the model is built directly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    values: FloatArray
    rows: IntArray

    @field_validator("values", mode="before")
    @classmethod
    def _detached_values(cls, value: object) -> FloatArray:
        return _one_dimensional(np.asarray(value, dtype=np.float64), "values")

    @field_validator("rows", mode="before")
    @classmethod
    def _detached_rows(cls, value: object) -> IntArray:
        return _one_dimensional(np.asarray(value, dtype=np.intp), "rows")

    @model_validator(mode="after")
    def _one_value_per_row(self) -> Self:
        if self.values.shape != self.rows.shape:
            raise ValueError(
                f"values and rows must have equal lengths ({self.values.size} != {self.rows.size})"
            )
        return self

    @property
    def present(self) -> int:
        """How many values are not NaN."""
        return int(np.count_nonzero(~np.isnan(self.values)))

    @property
    def missing(self) -> BoolArray:
        """``True`` where ``values`` is NaN: a blank cell, or one that is not a number."""
        return _read_only(np.isnan(self.values))


class GroupColumn(BaseModel):
    """One grouping column of a dataset over the rows that take part, as plain arrays.

    :meth:`Dataset.group` builds it for the statistical tools, which take NumPy
    arrays and never pandas. ``labels`` holds one string per entry of ``rows`` (0-based
    positions in :meth:`Dataset.table`); blank cells are left out of both. Every array
    is a detached read-only copy, also when the model is built directly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    labels: StrArray
    rows: IntArray

    @field_validator("labels", mode="before")
    @classmethod
    def _detached_labels(cls, value: object) -> StrArray:
        return _one_dimensional(np.asarray(value, dtype=np.str_), "labels")

    @field_validator("rows", mode="before")
    @classmethod
    def _detached_rows(cls, value: object) -> IntArray:
        return _one_dimensional(np.asarray(value, dtype=np.intp), "rows")

    @model_validator(mode="after")
    def _one_label_per_row(self) -> Self:
        if self.labels.shape != self.rows.shape:
            raise ValueError(
                f"labels and rows must have equal lengths ({self.labels.size} != {self.rows.size})"
            )
        return self

    @property
    def present(self) -> int:
        """How many rows carry a label."""
        return int(self.labels.size)


@dataclass(frozen=True, eq=False, init=False)
class Dataset:
    """A table of observations plus the ordered operations applied to it.

    The original table is held privately and never modified; :attr:`original`
    returns a fresh copy, so nothing done to it can affect this dataset. Excluded
    rows stay in :meth:`table`; they are only masked out by :meth:`active_mask`
    and therefore by :meth:`xy`, :meth:`column` and :meth:`group`.
    """

    name: str
    ops: tuple[Operation, ...]
    import_notes: tuple[str, ...]
    _original: pd.DataFrame = field(repr=False)

    def __init__(
        self,
        name: str,
        original: pd.DataFrame,
        ops: tuple[Operation, ...] = (),
        import_notes: tuple[str, ...] = (),
    ) -> None:
        duplicated = sorted({str(c) for c in original.columns[original.columns.duplicated()]})
        if duplicated:
            raise DataError("DUPLICATE_COLUMN", f"Duplicate column names: {', '.join(duplicated)}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "ops", tuple(ops))
        object.__setattr__(self, "import_notes", tuple(import_notes))
        object.__setattr__(self, "_original", original.copy(deep=True))
        self._materialise()  # reject an invalid operation list at construction

    @property
    def original(self) -> pd.DataFrame:
        """A copy of the table as imported; mutating it does not affect the dataset."""
        return self._original.copy(deep=True)

    def _with_ops(self, ops: tuple[Operation, ...]) -> Dataset:
        derived = object.__new__(Dataset)
        object.__setattr__(derived, "name", self.name)
        object.__setattr__(derived, "ops", ops)
        object.__setattr__(derived, "import_notes", self.import_notes)
        object.__setattr__(derived, "_original", self._original)  # never mutated, so shared
        derived._materialise()
        return derived

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dataset):
            return NotImplemented
        return (
            self.name == other.name
            and self.ops == other.ops
            and self.import_notes == other.import_notes
            and self._original.equals(other._original)
        )

    __hash__ = None  # type: ignore[assignment]

    def table(self) -> pd.DataFrame:
        """The original table with all operations applied (excluded rows kept)."""
        return self._materialise()[0].copy(deep=True)

    def active_mask(self) -> BoolArray:
        """Boolean mask over the original rows; ``False`` where a row is excluded."""
        return self._materialise()[1]

    def with_op(self, op: Operation) -> Dataset:
        """A new dataset with ``op`` appended. Raises :class:`DataError` if invalid."""
        return self._with_ops((*self.ops, op))

    def undo(self) -> Dataset:
        """A new dataset without the most recent operation."""
        if not self.ops:
            raise DataError("NOTHING_TO_UNDO", "There is no operation to undo")
        return self._with_ops(self.ops[:-1])

    def fingerprint(self) -> str:
        """SHA-256 hex digest of the original data (names, dtypes, exact values) and ops."""
        digest = hashlib.sha256()
        frame = self._original
        _feed(digest, f"{frame.shape[0]}x{frame.shape[1]}")
        for column in frame.columns:
            series = frame[column]
            _feed(digest, str(column))
            _feed(digest, str(series.dtype))
            if series.dtype.kind == "f":
                values = np.array(series.to_numpy(), copy=True)
                values[np.isnan(values)] = np.nan  # one canonical NaN bit pattern
                digest.update(np.ascontiguousarray(values).tobytes())
            elif series.dtype.kind in "biu":
                digest.update(np.ascontiguousarray(series.to_numpy()).tobytes())
            else:
                for value in series.to_numpy(dtype=object):
                    _feed(digest, "<null>" if pd.isna(value) else repr(value))
        for op in self.ops:
            _feed(digest, repr(op))
        return digest.hexdigest()

    def xy(self, x: str, y: str, weights: str | None = None) -> XY:
        """Extract validated float64 arrays from the active rows.

        Raises :class:`DataError` with ``MISSING_VALUE`` (blank/NaN cells),
        ``NON_FINITE`` (infinite cells) or ``INVALID_WEIGHT`` (negative weights),
        listing original-table row positions. Missing values are never dropped
        silently; zero weights are allowed and are surfaced by the fitting engines
        as a ``ZERO_WEIGHT_POINTS`` warning.
        """
        table, active = self._materialise()
        names = [x, y] if weights is None else [x, y, weights]
        for name in names:
            if name not in table.columns:
                raise DataError("UNKNOWN_COLUMN", f"Column {name!r} does not exist")
        columns = {name: _numeric_column(table, name, active) for name in names}

        missing = np.zeros(len(table), dtype=bool)
        infinite = np.zeros(len(table), dtype=bool)
        for values in columns.values():
            missing |= np.isnan(values)
            infinite |= np.isinf(values)
        bad_missing = np.flatnonzero(missing & active)
        if bad_missing.size:
            raise DataError(
                "MISSING_VALUE",
                "Active rows contain blank or NaN values; fill or exclude them first",
                tuple(int(i) for i in bad_missing),
            )
        bad_infinite = np.flatnonzero(infinite & active)
        if bad_infinite.size:
            raise DataError(
                "NON_FINITE",
                "Active rows contain infinite values; fix or exclude them first",
                tuple(int(i) for i in bad_infinite),
            )
        if weights is not None:
            negative = np.flatnonzero((columns[weights] < 0.0) & active)
            if negative.size:
                raise DataError(
                    "INVALID_WEIGHT",
                    "Active rows contain negative weights; use weight 0 to leave an "
                    "observation out of the fit",
                    tuple(int(i) for i in negative),
                )

        x_arr, y_arr = (_frozen(columns[n][active]) for n in (x, y))
        w_arr = None if weights is None else _frozen(columns[weights][active])
        digest = hashlib.sha256()
        _feed(digest, x)
        _feed(digest, y)
        digest.update(x_arr.tobytes())
        digest.update(y_arr.tobytes())
        if w_arr is None:
            _feed(digest, "<no weights>")
        else:
            _feed(digest, "<weights>")
            digest.update(w_arr.tobytes())
        return XY(x_arr, y_arr, w_arr, x, y, digest.hexdigest())

    def column(self, name: str) -> Column:
        """Column ``name`` over the rows that take part, as plain arrays.

        Unlike :meth:`xy` this never raises for a missing value: a blank cell, or
        one that is not a number, is NaN in ``values``. Only an unknown column
        (``UNKNOWN_COLUMN``) or one holding dates or durations (``NON_NUMERIC``)
        raises; the tools refuse a column with too few values themselves.
        """
        table, active = self._materialise()
        if name not in table.columns:
            raise DataError("UNKNOWN_COLUMN", f"Column {name!r} does not exist")
        series = table[name]
        if series.dtype.kind in "Mm":
            raise DataError("NON_NUMERIC", f"Column {name!r} holds dates or durations, not numbers")
        values, _unparseable = coerce_numeric(series)
        rows = np.flatnonzero(active)
        return Column(name=name, values=values[rows], rows=rows)

    def group(self, name: str) -> GroupColumn:
        """Column ``name`` as group labels over the rows that take part.

        Blank cells take no label and are left out of ``rows``. Only an unknown
        column raises (``UNKNOWN_COLUMN``); any other column may group the rows,
        numbers and dates included.
        """
        table, active = self._materialise()
        if name not in table.columns:
            raise DataError("UNKNOWN_COLUMN", f"Column {name!r} does not exist")
        cells = table[name].to_numpy(dtype=object)
        labels: list[str] = []
        rows: list[int] = []
        for position in np.flatnonzero(active):
            label = _label(cells[position])
            if label is not None:
                rows.append(int(position))
                labels.append(label)
        return GroupColumn(
            name=name,
            labels=np.array(labels, dtype=np.str_),
            rows=np.array(rows, dtype=np.intp),
        )

    def excluded_rows(self) -> IntArray:
        """0-based positions in :meth:`table` of the rows that take no part."""
        return _read_only(np.flatnonzero(~self._materialise()[1]))

    def _materialise(self) -> tuple[pd.DataFrame, BoolArray]:
        frame = self._original  # operations never mutate in place
        active = np.ones(len(frame), dtype=bool)
        for op in self.ops:
            frame, active = _apply(op, frame, active)
        return frame, active


def _feed(digest: Any, text: str) -> None:
    """Hash a length-prefixed string so adjacent fields cannot run together."""
    raw = text.encode("utf-8")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _frozen(values: FloatArray) -> FloatArray:
    out = np.array(values, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _read_only[ScalarT: np.generic](array: NDArray[ScalarT]) -> NDArray[ScalarT]:
    """A detached copy of ``array`` that cannot be written to."""
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _one_dimensional[ScalarT: np.generic](array: NDArray[ScalarT], name: str) -> NDArray[ScalarT]:
    """``array`` as a read-only copy; anything but a flat array raises ``ValueError``."""
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return _read_only(array)


def _require_finite(name: str, column: FloatArray) -> None:
    """Reject NaN (``MISSING_VALUE``) and infinite (``NON_FINITE``) cells with rows."""
    nan = np.flatnonzero(np.isnan(column))
    if nan.size:
        raise DataError("MISSING_VALUE", f"{name} contains NaN values", tuple(int(i) for i in nan))
    infinite = np.flatnonzero(np.isinf(column))
    if infinite.size:
        raise DataError(
            "NON_FINITE", f"{name} contains infinite values", tuple(int(i) for i in infinite)
        )


def coerce_numeric(series: pd.Series) -> tuple[FloatArray, BoolArray]:
    """``series`` as float64 plus a mask of non-null cells that are not numbers.

    Unparseable cells become NaN in the values and ``True`` in the mask. Never raises.
    """
    if series.dtype.kind in "iuf":
        values = np.asarray(series.to_numpy(dtype=np.float64), dtype=np.float64)
        return values, np.zeros(len(series), dtype=bool)
    coerced = pd.to_numeric(series, errors="coerce")
    unparseable = coerced.isna().to_numpy() & ~series.isna().to_numpy()
    values = np.asarray(coerced.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
    return values, unparseable


def _numeric_column(table: pd.DataFrame, name: str, active: BoolArray) -> FloatArray:
    """Column ``name`` as float64; non-numeric active cells raise ``NON_NUMERIC``."""
    series = table[name]
    if series.dtype.kind in "Mm":
        raise DataError("NON_NUMERIC", f"Column {name!r} holds dates or durations, not numbers")
    values, unparseable = coerce_numeric(series)
    bad = np.flatnonzero(unparseable & active)
    if bad.size:
        raise DataError(
            "NON_NUMERIC",
            f"Column {name!r} contains non-numeric values",
            tuple(int(i) for i in bad),
        )
    return values


def _label(cell: object) -> str | None:
    """A group label for ``cell``, or ``None`` when the cell is blank.

    A number reads as a person sees it: a whole number without ``.0``, and no
    rounding noise (15 significant digits). Any other cell is ``str(cell)``
    without surrounding whitespace.
    """
    if cell is None or pd.isna(cell):
        return None
    if isinstance(cell, numbers.Integral):
        return str(cell)
    if isinstance(cell, numbers.Real):
        return f"{float(cell):.15g}"
    if isinstance(cell, datetime.datetime):
        # A column of dates holds every cell at midnight; the day alone is the label.
        if cell.time() == datetime.time():
            return cell.strftime("%Y-%m-%d")
        return str(cell)
    if isinstance(cell, datetime.date):
        return cell.isoformat()
    text = str(cell).strip()
    return text or None


def _apply(op: Operation, frame: pd.DataFrame, active: BoolArray) -> tuple[pd.DataFrame, BoolArray]:
    if isinstance(op, ExcludeRows):
        out_of_range = tuple(r for r in op.rows if not 0 <= r < len(frame))
        if out_of_range:
            raise DataError("ROW_OUT_OF_RANGE", "Row positions outside the table", out_of_range)
        active = active.copy()
        active[list(op.rows)] = False
        return frame, active
    if isinstance(op, RenameColumn):
        if op.old not in frame.columns:
            raise DataError("UNKNOWN_COLUMN", f"Column {op.old!r} does not exist")
        if op.new in frame.columns:
            raise DataError("DUPLICATE_COLUMN", f"Column {op.new!r} already exists")
        return frame.rename(columns={op.old: op.new}), active
    return _transform(op, frame, active), active


def _transform(op: TransformColumn, frame: pd.DataFrame, active: BoolArray) -> pd.DataFrame:
    if op.source not in frame.columns:
        raise DataError("UNKNOWN_COLUMN", f"Column {op.source!r} does not exist")
    if op.target in frame.columns:
        raise DataError("DUPLICATE_COLUMN", f"Column {op.target!r} already exists")
    values = _numeric_column(frame, op.source, active)
    with np.errstate(all="ignore"):
        match op.func:
            case "log":
                violation, result = values <= 0, np.log(values)
            case "log10":
                violation, result = values <= 0, np.log10(values)
            case "sqrt":
                violation, result = values < 0, np.sqrt(values)
            case "reciprocal":
                violation, result = values == 0, 1.0 / values
            case "square":
                violation, result = np.zeros(len(values), dtype=bool), values * values
            case "affine":
                violation, result = np.zeros(len(values), dtype=bool), op.scale * values + op.offset
            case _:
                raise DataError("UNKNOWN_FUNCTION", f"Unknown transform function {op.func!r}")
    bad = np.flatnonzero(violation & active)
    if bad.size:
        raise DataError(
            "DOMAIN_ERROR",
            f"{op.func} is undefined for some values of column {op.source!r}",
            tuple(int(i) for i in bad),
        )
    out = frame.copy()
    out[op.target] = result
    return out
