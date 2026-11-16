"""Immutable dataset with a reversible log of operations."""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from curvestudio.data.errors import DataError
from curvestudio.data.operations import ExcludeRows, Operation, RenameColumn, TransformColumn

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class XY:
    """Validated, read-only numeric arrays ready for fitting."""

    x: FloatArray
    y: FloatArray
    weights: FloatArray | None
    x_name: str
    y_name: str
    fingerprint: str


@dataclass(frozen=True, eq=False, init=False)
class Dataset:
    """A table of observations plus the ordered operations applied to it.

    The original table is held privately and never modified; :attr:`original`
    returns a fresh copy, so nothing done to it can affect this dataset. Excluded
    rows stay in :meth:`table`; they are only masked out by :meth:`active_mask`
    and therefore by :meth:`xy`.
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

        Raises :class:`DataError` with ``MISSING_VALUE`` (blank/NaN cells) or
        ``NON_FINITE`` (infinite cells), listing original-table row positions.
        Missing values are never dropped silently.
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


def _numeric_column(table: pd.DataFrame, name: str, active: BoolArray) -> FloatArray:
    """Column ``name`` as float64; non-numeric active cells raise ``NON_NUMERIC``."""
    series = table[name]
    if series.dtype.kind in "Mm":
        raise DataError("NON_NUMERIC", f"Column {name!r} holds dates or durations, not numbers")
    if series.dtype.kind in "iuf":
        return np.asarray(series.to_numpy(dtype=np.float64), dtype=np.float64)
    coerced = pd.to_numeric(series, errors="coerce")
    unparseable = coerced.isna().to_numpy() & ~series.isna().to_numpy() & active
    if unparseable.any():
        raise DataError(
            "NON_NUMERIC",
            f"Column {name!r} contains non-numeric values",
            tuple(int(i) for i in np.flatnonzero(unparseable)),
        )
    return np.asarray(coerced.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)


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
