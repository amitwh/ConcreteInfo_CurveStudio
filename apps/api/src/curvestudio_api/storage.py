"""The stored form of a dataset: plain CSV bytes, with no pickle.

Two payloads are kept, and nothing else from the upload:

* The **fit payload** (:func:`encode_payload`) is the cleaned x, y (and weight) columns under
  the header ``x,y`` or ``x,y,w``. Floats are written with their shortest round-trip form, so
  reading it back gives exactly the numbers that were fitted.
* The **table payload** (:func:`encode_table`) is every other column of the table under its
  own name, over the rows that take part: the group labels, second measurements and batch
  codes that the statistical tools read. A table with no other column stores none, and a
  dataset stored before this payload existed has none at all.

A table payload is read back with :func:`decode_table` (the columns as arrays) or described
with :func:`table_columns` (each column's kind and blank cells). One reader serves both, so
they cannot disagree about a column's kind.
"""

import csv
import io
import sys
import threading
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from curvestudio.data import XY, Dataset

NumberArray = NDArray[np.float64]
TextArray = NDArray[np.str_]
ColumnKind = Literal["number", "text"]
TableColumn = dict[str, Any]
"""``{"name": str, "kind": "number" | "text", "n_missing": int}``: as stored on a dataset."""

_HEADER = ("x", "y", "w")
_FIELD_LIMIT = min(sys.maxsize, 2**31 - 1)
_reading = threading.Lock()
"""Held while a stored table is read: the csv module's field-size limit is a setting of the
whole process, lifted for the read and put back afterwards."""


def encode_payload(xy: XY) -> bytes:
    """CSV bytes (header ``x,y`` or ``x,y,w``) for the data of ``xy``."""
    columns = {"x": xy.x, "y": xy.y}
    if xy.weights is not None:
        columns["w"] = xy.weights
    buffer = io.StringIO()
    pd.DataFrame(columns).to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("ascii")


def decode_payload(payload: bytes, *, x_name: str, y_name: str, fingerprint: str) -> XY:
    """The :class:`XY` stored by :func:`encode_payload`, with the original fingerprint."""
    header, _, body = payload.partition(b"\n")
    columns = tuple(header.decode("ascii").strip().split(","))
    if columns not in (_HEADER[:2], _HEADER):
        raise ValueError(f"stored dataset has unexpected columns {list(columns)}")
    values = np.loadtxt(
        io.BytesIO(body), delimiter=",", dtype=np.float64, ndmin=2, encoding="ascii"
    ).reshape(-1, len(columns))
    return XY(
        x=values[:, 0],
        y=values[:, 1],
        weights=values[:, 2] if len(columns) == 3 else None,
        x_name=x_name,
        y_name=y_name,
        fingerprint=fingerprint,
    )


def encode_table(
    dataset: Dataset,
    *,
    x_column: str,
    y_column: str,
    weight_column: str | None,
    n_rows: int,
) -> bytes | None:
    """CSV bytes of the table's other columns over the rows that take part, or ``None``.

    The columns are those of ``dataset.table()`` that are not the x, y or weight column, in
    the table's order and under their own names; a table with no other column has no payload.
    The rows are those that take part, so the payload lines up row for row with the fit
    payload. A number is as pandas writes it and a missing one is an empty cell (pandas
    writes ``""`` for an empty cell when the table has one column; both read back as empty);
    any other cell is its string.

    Raises:
        ValueError: when a fit column is not a column of the table, or when the rows that
            take part do not number ``n_rows``. The fit's input is built from the same rows,
            so a disagreement is a fault of the caller and is never stored.
    """
    table = dataset.table()
    fit = {name for name in (x_column, y_column, weight_column) if name is not None}
    unknown = fit.difference(table.columns)
    if unknown:
        raise ValueError(
            f"not a column of the table: {', '.join(repr(n) for n in sorted(unknown))}"
        )
    active = dataset.active_mask()
    taking_part = int(np.count_nonzero(active))
    if taking_part != n_rows:
        raise ValueError(
            f"{taking_part:,} rows take part but the fit's input has {n_rows:,}: the stored "
            "table would not line up with it"
        )
    others = [name for name in table.columns if name not in fit]
    if not others:
        return None
    buffer = io.BytesIO()
    # Encoded as it is written: the whole table never exists once more as text, which for a
    # large upload would double what the import process holds.
    with io.TextIOWrapper(buffer, encoding="utf-8", newline="") as text:
        table.loc[active, others].to_csv(text, index=False, lineterminator="\n")
        text.flush()
        return buffer.getvalue()


def decode_table(payload: bytes) -> dict[str, NumberArray | TextArray]:
    """The columns of a stored table by name, in the payload's order; ``{}`` for no payload.

    Every cell is read as text: no NA-like word becomes a missing value and nothing is
    stripped. A blank cell is empty or spaces only. A column is a **number** column when
    every cell of it that is not blank parses as a number: ``float64``, with NaN for a blank
    cell. Any other column is a **text** column: an array of ``numpy.str_`` holding the cells
    as written.

    Raises:
        ValueError: when the payload is not a well-formed table (not UTF-8, a row that does
            not match its header, a column named twice).
    """
    decoded: dict[str, NumberArray | TextArray] = {}
    for name, cells in _read(payload):
        decoded[name] = _classify(cells)[2]
    return decoded


def table_columns(payload: bytes) -> list[TableColumn]:
    """Every column of a stored table as a :data:`TableColumn`, in the payload's order.

    ``n_missing`` counts the column's blank cells, whichever its kind. ``[]`` when there is
    no table payload. Raises :class:`ValueError` as :func:`decode_table` does.
    """
    described: list[TableColumn] = []
    for name, cells in _read(payload):
        kind, missing, _values = _classify(cells)
        described.append({"name": name, "kind": kind, "n_missing": missing})
    return described


def table_column_list(
    columns: list[str],
    *,
    x_column: str,
    y_column: str,
    weight_column: str | None,
    table_payload: bytes,
) -> list[TableColumn]:
    """The columns of the whole table, in the order of ``columns``, as :data:`TableColumn`.

    The x, y and weight column are ``"number"`` with nothing missing: the fit input cannot
    hold a missing value. Every other column is as :func:`table_columns` describes it in
    ``table_payload``.

    Raises:
        ValueError: when ``columns`` names a column that is neither, or leaves one out.
    """
    stored = {entry["name"]: entry for entry in table_columns(table_payload)}
    fit = {name for name in (x_column, y_column, weight_column) if name is not None}
    listed: list[TableColumn] = []
    for name in columns:
        if name in fit:
            listed.append({"name": name, "kind": "number", "n_missing": 0})
        elif name in stored:
            listed.append(stored[name])
        else:
            raise ValueError(f"column {name!r} is neither a fit column nor a stored one")
    left_out = (set(stored) | fit).difference(columns)
    if left_out:
        raise ValueError(
            "not among the table's columns: " + ", ".join(repr(n) for n in sorted(left_out))
        )
    return listed


def _read(payload: bytes) -> list[tuple[str, list[str]]]:
    """Each column of a stored table: its name and its cells, exactly as written."""
    if not payload:
        return []
    # The csv module refuses a field of more than 128 KiB. A text cell may be longer (a JSON
    # upload can hold one), so the limit is lifted for this read only.
    limit = csv.field_size_limit()
    columns: list[tuple[str, list[str]]] = []
    with _reading:
        csv.field_size_limit(_FIELD_LIMIT)
        try:
            rows = csv.reader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
            columns = [(name, []) for name in next(rows, [])]
            for position, row in enumerate(rows, start=1):
                if len(row) != len(columns):
                    raise ValueError(
                        f"row {position} of the stored table has {len(row)} cell(s); its "
                        f"header names {len(columns)} column(s)"
                    )
                for cell, (_name, cells) in zip(row, columns, strict=True):
                    cells.append(cell)
        except csv.Error as exc:
            raise ValueError(f"the stored table is not well-formed CSV: {exc}") from exc
        finally:
            csv.field_size_limit(limit)
    if len({name for name, _cells in columns}) != len(columns):
        raise ValueError("the stored table names a column twice")
    return columns


def _classify(cells: list[str]) -> tuple[ColumnKind, int, NumberArray | TextArray]:
    """One column's cells as an array, with its kind and how many cells are blank."""
    blank = [not cell.strip() for cell in cells]
    missing = sum(blank)
    written = [(cell, is_blank) for cell, is_blank in zip(cells, blank, strict=True)]
    if all(_is_number(cell) for cell, is_blank in written if not is_blank):
        values = np.array(
            [np.nan if is_blank else float(cell) for cell, is_blank in written], dtype=np.float64
        )
        return "number", missing, values
    return "text", missing, np.array(cells, dtype=np.str_)


def _is_number(cell: str) -> bool:
    """Whether a cell that is not blank is a number, as the core's reader judges one at
    import. Python's ``float`` would also take ``1_000``, which the reader keeps as text."""
    if "_" in cell:
        return False
    try:
        float(cell)
    except ValueError:
        return False
    return True
