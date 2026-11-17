"""Table ⇄ CSV conversion for project archives.

Floats are written with :func:`repr` (shortest round-trip form), so parsing the text
back yields exactly the same bits. Text columns are written as their UTF-8 characters
(``None``/NaN become an empty cell); numeric, boolean and datetime columns keep their
dtype through the stored column list. Every value either round-trips exactly or the
save is refused: an arm of the format that silently loses data is worse than no arm.
"""

import csv
import io
import math
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from curvestudio.project.errors import ProjectError, ProjectErrorCode

ColumnSpec = tuple[str, str]
"""A column's name and its stored dtype string (``"float64"``, ``"str"``, ...)."""

_FLOAT_DTYPES = frozenset({"float16", "float32", "float64"})
_INT_DTYPES = frozenset(
    {*(f"int{bits}" for bits in (8, 16, 32, 64)), *(f"uint{bits}" for bits in (8, 16, 32, 64))}
)
_TEXT_DTYPES = frozenset({"object", "str"})


def _kind(dtype: str) -> str | None:
    """The storage kind for a dtype string, or ``None`` when the format cannot hold it."""
    if dtype in _FLOAT_DTYPES:
        return "float"
    if dtype in _INT_DTYPES:
        return "int"
    if dtype == "bool":
        return "bool"
    if dtype in _TEXT_DTYPES:
        return "text"
    if dtype.startswith("datetime64[") and dtype.endswith("]") and "," not in dtype:
        return "datetime"
    return None


def column_specs(frame: pd.DataFrame) -> tuple[ColumnSpec, ...]:
    """``frame``'s columns as ``(name, dtype)`` pairs, or ``UNSUPPORTED_DATA``.

    The project format stores the dtype so loading rebuilds the exact table; a column
    whose dtype it cannot represent (nullable integers, timezone-aware datetimes,
    interval, categorical, ...) is refused loudly at save time.
    """
    if frame.shape[1] == 0:
        raise ProjectError(ProjectErrorCode.UNSUPPORTED_DATA, "a dataset needs at least one column")
    specs: list[ColumnSpec] = []
    for column in frame.columns:
        if not isinstance(column, str):
            raise ProjectError(
                ProjectErrorCode.UNSUPPORTED_DATA,
                f"column name {column!r} is not a string",
            )
        dtype = str(frame[column].dtype)
        if _kind(dtype) is None:
            raise ProjectError(
                ProjectErrorCode.UNSUPPORTED_DATA,
                f"column {column!r} has dtype {dtype!r}, which the project format cannot store",
            )
        specs.append((column, dtype))
    return tuple(specs)


def encode_frame(frame: pd.DataFrame) -> bytes:
    """Serialise ``frame`` to deterministic UTF-8 CSV bytes."""
    specs = column_specs(frame)
    kinds = [_kind(dtype) for _, dtype in specs]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([name for name, _ in specs])
    for row in frame.itertuples(index=False, name=None):
        cells: list[str] = []
        for value, kind in zip(row, kinds, strict=True):
            assert kind is not None  # column_specs validated every dtype
            cells.append(_encode_cell(value, kind))
        writer.writerow(cells)
    return buffer.getvalue().encode("utf-8")


def decode_frame(data: bytes, specs: Sequence[ColumnSpec]) -> pd.DataFrame:
    """Rebuild a frame from :func:`encode_frame` output; malformed input is ``CORRUPT``."""
    if not specs:
        raise ProjectError(ProjectErrorCode.CORRUPT, "the dataset declares no columns")
    names = [name for name, _ in specs]
    if len(set(names)) != len(names):
        raise ProjectError(ProjectErrorCode.CORRUPT, "the dataset declares duplicate column names")
    kinds: list[str] = []
    for name, dtype in specs:
        kind = _kind(dtype)
        if kind is None:
            raise ProjectError(
                ProjectErrorCode.CORRUPT, f"column {name!r} has unsupported dtype {dtype!r}"
            )
        kinds.append(kind)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectError(ProjectErrorCode.CORRUPT, f"dataset CSV is not UTF-8: {exc}") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header != names:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, "dataset CSV header does not match its column list"
        )
    values: dict[str, list[Any]] = {name: [] for name in names}
    for line, row in enumerate(reader, start=2):
        if len(row) != len(names):
            raise ProjectError(
                ProjectErrorCode.CORRUPT,
                f"dataset CSV row {line} has {len(row)} cells, expected {len(names)}",
            )
        for (name, _), kind, cell in zip(specs, kinds, row, strict=True):
            values[name].append(_decode_cell(cell, kind, line, name))
    try:
        return pd.DataFrame({name: pd.Series(values[name], dtype=dtype) for name, dtype in specs})
    except (TypeError, ValueError) as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT, f"dataset CSV cannot be rebuilt: {exc}"
        ) from exc


def _encode_cell(value: Any, kind: str) -> str:
    if kind == "float":
        return repr(float(value))
    if kind == "int":
        return str(int(value))
    if kind == "bool":
        return "True" if value else "False"
    if kind == "datetime":
        return "" if pd.isna(value) else pd.Timestamp(value).isoformat()
    return _encode_text(value)


def _encode_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if not isinstance(value, str):
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_DATA,
            f"text column holds a {type(value).__name__} value; only text and blanks can be stored",
        )
    if value == "":
        raise ProjectError(
            ProjectErrorCode.UNSUPPORTED_DATA,
            "text column holds an empty string, which the format cannot tell apart from a blank",
        )
    return value


def _decode_cell(cell: str, kind: str, line: int, name: str) -> Any:
    if kind == "text":
        return None if cell == "" else cell
    if kind == "int" and _integer_digits_exceeded(cell):
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"dataset CSV cell at row {line}, column {name!r}: number too large",
        )
    try:
        if kind == "float":
            return float(cell)
        if kind == "int":
            return int(cell)
        if kind == "bool":
            if cell == "True":
                return True
            if cell == "False":
                return False
            raise ValueError("expected True or False")
        if cell == "":  # datetime
            return pd.NaT
        stamp = pd.Timestamp(cell)
        if stamp.tzinfo is not None:
            raise ValueError("timezone-aware timestamps are not supported")
        return stamp
    except (TypeError, ValueError) as exc:
        raise ProjectError(
            ProjectErrorCode.CORRUPT,
            f"dataset CSV cell at row {line}, column {name!r} is invalid: {cell!r}",
        ) from exc


def _integer_digits_exceeded(cell: str) -> bool:
    """Whether ``int(cell)`` would trip Python's integer-string conversion limit."""
    limit = sys.get_int_max_str_digits()
    digits = cell.strip().lstrip("+-")
    return bool(limit) and digits.isdecimal() and len(digits) > limit
