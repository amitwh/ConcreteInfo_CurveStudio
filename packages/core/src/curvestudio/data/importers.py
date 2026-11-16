"""Importers for CSV, TSV, TXT, JSON, XLSX files and pasted (clipboard) tables.

Every importer returns a :class:`~curvestudio.data.dataset.Dataset` and records each
change it made to the raw data in ``Dataset.import_notes``; nothing is dropped silently.

What is changed, and noted:

* a UTF-8 byte order mark is stripped;
* fully blank rows and fully blank columns (blank header too) are dropped;
* blank or duplicate header names become ``col_N`` / ``name_2``;
* a first row with no text cell is data, so columns are named ``col_1..col_N``
  (numbers, dates, booleans and blanks alone do not make a header).

Numbers: a column is numeric when every non-blank cell parses as a number. Thousands
separators (``1,234.5``) are honoured only in a column whose every non-blank cell is a
plain number or fits that grouping. If at least half of a column's non-blank cells parse,
the column is numeric, the others become NaN and each one is recorded as
``coerced:<column>:<row>``, where ``<row>`` is the 0-based position in the final table (the
indexing of ``Dataset.original``, after blank rows were dropped). If fewer than half
parse, the column stays text with no notes. Blank cells are NaN with no note.

Not supported: decimal commas (``2,5`` stays text) and encodings other than UTF-8 (a
different encoding is a ``PARSE_ERROR``, never guessed). Nothing is ever evaluated or
unpickled; XLSX is opened read-only and without evaluating formulas.

Limits (:class:`ImportLimits`) are enforced before or while parsing, never after the
whole table has been built. ``max_rows`` bounds data rows (a header row does not count).
"""

import csv
import io
import json
import math
import os
import re
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from curvestudio.data.dataset import Dataset
from curvestudio.data.errors import DataError

ZIP_EXPANSION_LIMIT = 10  # an xlsx may expand to at most this many times max_bytes
_SNIFF_LINES = 20
_WHITESPACE = "ws"
_BOM = "\ufeff"

_PLAIN = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_SPECIAL = re.compile(r"[+-]?(?:inf|infinity|nan)", re.IGNORECASE)
_GROUPED = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")


@dataclass(frozen=True)
class _Raw:
    """Non-blank rows plus what was dropped to get them."""

    rows: list[list[Any]]
    blank_rows: int
    blank_cols: int  # trailing columns that were blank in every non-blank row


Kind = Literal["blank", "number", "grouped", "date", "text", "other"]


@dataclass(frozen=True)
class ImportLimits:
    """Upper bounds checked while importing: file size, data rows and columns."""

    max_bytes: int = 50_000_000
    max_rows: int = 1_000_000
    max_cols: int = 1_000


DEFAULT_LIMITS = ImportLimits()


def read_dataset(
    source: str | os.PathLike[str],
    *,
    fmt: str | None = None,
    sheet: str | int = 0,
    limits: ImportLimits = DEFAULT_LIMITS,
) -> Dataset:
    """Import a csv, tsv, txt, json or xlsx file as a :class:`Dataset` named after its stem.

    ``fmt`` overrides the file extension. ``sheet`` (xlsx only) is a sheet name or a
    0-based index. Raises :class:`DataError` for a missing or oversized file, an
    unsupported format, content that cannot be parsed, or a breached limit.
    """
    path = Path(source)
    kind = (fmt if fmt is not None else path.suffix).lower().lstrip(".")
    if kind == "xls":
        raise DataError(
            "UNSUPPORTED_FORMAT", "Legacy .xls is not supported; save the workbook as .xlsx"
        )
    if kind not in {"csv", "tsv", "txt", "json", "xlsx"}:
        raise DataError(
            "UNSUPPORTED_FORMAT",
            f"Unsupported format {kind!r}; use csv, tsv, txt, json or xlsx",
        )
    if not path.is_file():
        raise DataError("FILE_NOT_FOUND", f"File not found: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _unreadable(path, exc) from exc
    if size > limits.max_bytes:
        raise _too_large(size, limits)
    if kind == "xlsx":
        return _read_xlsx(path, sheet, limits)

    try:
        with path.open("rb") as handle:
            raw = handle.read(limits.max_bytes + 1)
    except OSError as exc:
        raise _unreadable(path, exc) from exc
    if len(raw) > limits.max_bytes:
        raise _too_large(len(raw), limits)
    notes: list[str] = []
    text = _decode(raw, notes)
    if kind == "json":
        return _from_json(path.stem, text, limits, notes)
    delimiter = {"tsv": "\t"}.get(kind) or _sniff_delimiter(text, kind)
    return _from_delimited(path.stem, text, delimiter, limits, notes)


def read_text(text: str, *, name: str = "pasted", limits: ImportLimits = DEFAULT_LIMITS) -> Dataset:
    """Import a pasted table; the delimiter is sniffed among tab, ``;``, ``,`` and spaces."""
    size = len(text.encode("utf-8", errors="replace"))
    if size > limits.max_bytes:
        raise _too_large(size, limits)
    notes: list[str] = []
    if text.startswith(_BOM):
        text = text[1:]
        notes.append("stripped UTF-8 byte order mark")
    return _from_delimited(name, text, _sniff_delimiter(text, "txt"), limits, notes)


# --- reading ---------------------------------------------------------------------------


def _too_large(size: int, limits: ImportLimits) -> DataError:
    return DataError(
        "FILE_TOO_LARGE", f"Input is {size:,} bytes; the limit is {limits.max_bytes:,} bytes"
    )


def _unreadable(path: Path, exc: OSError) -> DataError:
    return DataError("FILE_UNREADABLE", f"Cannot read {path}: {exc}")


def _decode(raw: bytes, notes: list[str]) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataError("PARSE_ERROR", f"Not valid UTF-8 text ({exc.reason})") from exc
    if text.startswith(_BOM):
        text = text[1:]
        notes.append("stripped UTF-8 byte order mark")
    return text


def _sniff_delimiter(text: str, kind: str) -> str:
    """Pick the delimiter whose field count is most consistent over the first lines.

    ``.csv`` uses a comma unless a semicolon is strictly more consistent. Other text
    uses tab, ``;``, ``,`` or runs of whitespace, ties going in that order.
    """
    candidates: tuple[str, ...] = (",", ";") if kind == "csv" else ("\t", ";", ",", _WHITESPACE)
    default = candidates[0]
    lines = [line for line in text.splitlines() if line.strip()][:_SNIFF_LINES]
    best, best_hits = default, 0
    for delimiter in candidates:
        try:
            counts = Counter(len(row) for row in _delimited_rows(lines, delimiter))
        except csv.Error:
            continue  # the real parse reports it
        if not counts:
            continue
        fields, hits = max(counts.items(), key=lambda item: (item[1], item[0]))
        if fields > 1 and hits > best_hits:
            best, best_hits = delimiter, hits
    return best


def _delimited_rows(lines: Iterable[str] | str, delimiter: str) -> Iterable[Sequence[str]]:
    if delimiter == _WHITESPACE:
        source = lines.splitlines() if isinstance(lines, str) else lines
        return (line.split() for line in source)
    if isinstance(lines, str):
        return csv.reader(io.StringIO(lines, newline=""), delimiter=delimiter)
    return csv.reader(lines, delimiter=delimiter)


def _from_delimited(
    name: str, text: str, delimiter: str, limits: ImportLimits, notes: list[str]
) -> Dataset:
    try:
        raw = _collect(_delimited_rows(text, delimiter), limits)
    except csv.Error as exc:
        raise DataError("PARSE_ERROR", f"Cannot parse delimited text: {exc}") from exc
    return _from_rows(name, raw, limits, notes)


def _read_xlsx(path: Path, sheet: str | int, limits: ImportLimits) -> Dataset:
    notes: list[str] = []
    try:
        with path.open("rb") as handle:
            with zipfile.ZipFile(handle) as archive:
                expanded = sum(info.file_size for info in archive.infolist())
            if expanded > ZIP_EXPANSION_LIMIT * limits.max_bytes:
                raise DataError(
                    "FILE_TOO_LARGE",
                    f"Workbook expands to {expanded:,} bytes; the limit is "
                    f"{ZIP_EXPANSION_LIMIT * limits.max_bytes:,} bytes",
                )
            handle.seek(0)
            workbook = load_workbook(handle, read_only=True, data_only=True)
            try:
                names = workbook.sheetnames
                if isinstance(sheet, str):
                    if sheet not in names:
                        raise DataError("UNKNOWN_SHEET", f"No sheet named {sheet!r}")
                    worksheet = workbook[sheet]
                else:
                    if not 0 <= sheet < len(names):
                        raise DataError("UNKNOWN_SHEET", f"No sheet at index {sheet}")
                    worksheet = workbook[names[sheet]]
                # Ignore the sheet's <dimension> tag: read-only openpyxl pads every row
                # to that declared width and materialises every gap row, so a lying or
                # corrupt tag turns a 5 KB workbook into ~20 minutes of CPU (the cells
                # that exist are yielded either way).
                worksheet.reset_dimensions()
                raw = _collect(worksheet.iter_rows(values_only=True), limits)
            finally:
                workbook.close()
    except DataError:
        raise  # UNKNOWN_SHEET / FILE_TOO_LARGE reach the caller unchanged
    except OSError as exc:
        raise DataError("FILE_UNREADABLE", f"Cannot read {path}: {exc}") from exc
    except Exception as exc:  # any malformed part: bad deflate, broken XML, stray types
        raise DataError(
            "PARSE_ERROR", f"Cannot read workbook: {type(exc).__name__}: {exc}"
        ) from exc
    return _from_rows(path.stem, raw, limits, notes)


def _from_json(name: str, text: str, limits: ImportLimits, notes: list[str]) -> Dataset:
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise DataError("PARSE_ERROR", f"Invalid JSON: {exc}") from exc
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
        if len(data) > limits.max_rows:
            raise _too_many_rows(limits)
        keys = list(dict.fromkeys(key for item in data for key in item))
        _check_cols(len(keys), limits)
        rows = [[item.get(key) for key in keys] for item in data]
    elif isinstance(data, dict) and data and all(isinstance(v, list) for v in data.values()):
        _check_cols(len(data), limits)
        keys = list(data)
        lengths = {len(column) for column in data.values()}
        if len(lengths) != 1:
            raise DataError("UNSUPPORTED_FORMAT", "JSON columns must all have the same length")
        if lengths.pop() > limits.max_rows:
            raise _too_many_rows(limits)
        rows = [list(row) for row in zip(*data.values(), strict=True)]
    elif data in ([], {}):
        raise DataError("EMPTY_DATA", "The JSON document holds no data")
    else:
        raise DataError(
            "UNSUPPORTED_FORMAT",
            "JSON must be a list of records (objects) or an object of equal-length columns",
        )
    if any(isinstance(cell, dict | list) for row in rows for cell in row):
        raise DataError("UNSUPPORTED_FORMAT", "JSON cells must be scalars, not nested values")
    raw = _collect(rows, limits)
    return _assemble(name, [_clean(str(key)) for key in keys], raw.rows, raw, limits, notes)


# --- building the table ----------------------------------------------------------------


def _too_many_rows(limits: ImportLimits) -> DataError:
    return DataError("TOO_MANY_ROWS", f"More than {limits.max_rows:,} data rows")


def _check_cols(width: int, limits: ImportLimits) -> None:
    if width > limits.max_cols:
        raise DataError("TOO_MANY_COLS", f"More than {limits.max_cols:,} columns")


def _clean(cell: Any) -> Any:
    if isinstance(cell, str):
        stripped = cell.strip()
        return stripped or None
    return cell


def _collect(raw_rows: Iterable[Sequence[Any]], limits: ImportLimits) -> _Raw:
    """Non-blank rows (cells cleaned, trailing blanks trimmed) and what was dropped.

    Stops with ``TOO_MANY_COLS`` / ``TOO_MANY_ROWS`` as soon as a limit is crossed. One
    row more than ``max_rows`` is tolerated here because the first row may be a header.

    Two budgets bound the scan on pathological inputs (a crafted or corrupt workbook
    can stream enormous blank regions): blank rows are capped at ``max_rows`` and raw
    cells at ``max_rows * max_cols``; past either, the import stops with
    ``TOO_MANY_ROWS`` instead of grinding through the padding.
    """
    rows: list[list[Any]] = []
    blank = 0
    cells = 0
    widest_raw = widest = 0
    cell_budget = limits.max_rows * limits.max_cols
    for raw in raw_rows:
        row = [_clean(cell) for cell in raw]
        raw_width = len(row)
        cells += raw_width
        if cells > cell_budget:
            raise DataError(
                "TOO_MANY_ROWS",
                f"More than {cell_budget:,} cells (max_rows x max_cols) before the table "
                "ended; the input is pathologically large or padded",
            )
        while row and row[-1] is None:
            row.pop()
        if not row:
            blank += 1
            if blank > limits.max_rows:
                raise DataError(
                    "TOO_MANY_ROWS", f"More than {limits.max_rows:,} blank rows in the input"
                )
            continue
        _check_cols(len(row), limits)
        widest_raw, widest = max(widest_raw, raw_width), max(widest, len(row))
        rows.append(row)
        if len(rows) > limits.max_rows + 1:
            raise _too_many_rows(limits)
    return _Raw(rows, blank, widest_raw - widest)


def _classify(cell: Any) -> tuple[Kind, float]:
    """Kind of a cleaned cell and, for numbers, its float value."""
    if cell is None:
        return "blank", math.nan
    if isinstance(cell, bool):
        return "other", math.nan
    if isinstance(cell, int | float):
        try:
            return "number", float(cell)
        except OverflowError:
            return "other", math.nan
    if isinstance(cell, date):  # datetime is a date
        return "date", math.nan
    if isinstance(cell, str):
        if _PLAIN.fullmatch(cell) or _SPECIAL.fullmatch(cell):
            return "number", float(cell)
        if _GROUPED.fullmatch(cell):
            return "grouped", float(cell.replace(",", ""))
        return "text", math.nan
    return "other", math.nan


def _from_rows(name: str, raw: _Raw, limits: ImportLimits, notes: list[str]) -> Dataset:
    rows = raw.rows
    if not rows:
        raise DataError("EMPTY_DATA", "The input holds no data")
    if any(_classify(cell)[0] == "text" for cell in rows[0]):
        return _assemble(name, rows[0], rows[1:], raw, limits, notes)
    return _assemble(name, None, rows, raw, limits, notes)


def _assemble(
    name: str,
    header: list[Any] | None,
    body: list[list[Any]],
    raw: _Raw,
    limits: ImportLimits,
    notes: list[str],
) -> Dataset:
    if len(body) > limits.max_rows:
        raise _too_many_rows(limits)
    if raw.blank_rows:
        notes.append(f"dropped {raw.blank_rows} blank row(s)")
    head = list(header) if header is not None else []
    width = max([len(row) for row in body] + [len(head)])
    head += [None] * (width - len(head))
    grid = [row + [None] * (width - len(row)) for row in body]

    keep = [
        j for j in range(width) if head[j] is not None or any(row[j] is not None for row in grid)
    ]
    blank_cols = width - len(keep) + raw.blank_cols
    if blank_cols:
        notes.append(f"dropped {blank_cols} blank column(s)")
    if not keep or not grid:
        raise DataError("EMPTY_DATA", "The input holds no data rows")

    if header is None:
        names = [f"col_{position}" for position in range(1, len(keep) + 1)]
        notes.append(f"no header row found; columns named col_1..col_{len(keep)}")
    else:
        names = _header_names([head[j] for j in keep], notes)
    columns: dict[str, Any] = {}
    for position, j in enumerate(keep):
        columns[names[position]] = _column(names[position], [row[j] for row in grid], notes)
    return Dataset(name=name, original=pd.DataFrame(columns), import_notes=tuple(notes))


def _header_names(header: Sequence[Any], notes: list[str]) -> list[str]:
    taken: set[str] = set()
    names: list[str] = []
    for position, cell in enumerate(header, start=1):
        label = "" if cell is None else str(cell).strip()
        unique = _unique(label or f"col_{position}", taken)
        if not label:
            notes.append(f"blank header in column {position} named {unique!r}")
        elif unique != label:
            notes.append(f"duplicate header {label!r} in column {position} renamed to {unique!r}")
        names.append(unique)
    return names


def _unique(base: str, taken: set[str]) -> str:
    candidate, suffix = base, 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _column(name: str, cells: list[Any], notes: list[str]) -> Any:
    """One column as float64 (numeric), datetime64, or an object column of text."""
    kinds = [_classify(cell) for cell in cells]
    present = [kind for kind, _ in kinds if kind != "blank"]
    if not present:
        return np.full(len(cells), np.nan)
    if all(kind == "date" for kind in present):
        return pd.to_datetime(pd.Series(cells, dtype=object))
    grouped_ok = "grouped" in present and all(kind in ("number", "grouped") for kind in present)

    def parses(kind: Kind) -> bool:
        return kind == "number" or (kind == "grouped" and grouped_ok)

    if 2 * sum(parses(kind) for kind in present) < len(present):
        return pd.Series(cells, dtype=object)
    values = np.full(len(cells), np.nan)
    for row, (kind, value) in enumerate(kinds):
        if parses(kind):
            values[row] = value
        elif kind != "blank":
            notes.append(f"coerced:{name}:{row}")
    return values
