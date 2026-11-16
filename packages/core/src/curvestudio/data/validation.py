"""Fitting-readiness validation: report every problem, raise only for unknown columns."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from curvestudio.data.dataset import BoolArray, Dataset, FloatArray, coerce_numeric
from curvestudio.data.errors import DataError

Severity = Literal["error", "warning"]

MIN_ACTIVE_ROWS = 3
WIDE_RANGE_RATIO = 1e12
_COERCED_PREFIX = "coerced:"


@dataclass(frozen=True)
class Issue:
    """One validation finding; ``rows`` are 0-based positions in the original table."""

    code: str
    severity: Severity
    message: str
    rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Column:
    values: FloatArray
    unparseable: BoolArray  # cell holds something that is not a number


def validate(
    ds: Dataset,
    x: str,
    y: str,
    weights: str | None = None,
    positive_required: tuple[str, ...] = (),
) -> list[Issue]:
    """Check the active rows of ``ds`` for problems that would stop or distort a fit.

    Never raises for bad data; only an unknown column name raises
    ``DataError("UNKNOWN_COLUMN")``. Each code is reported at most once, with sorted
    unique ``rows`` and the affected column names in ``message``. Excluded rows are
    ignored. A blank (NaN) cell is ``MISSING_VALUE``; a cell that is text, or a date, is
    ``NON_NUMERIC`` and not also ``MISSING_VALUE``. A cell the importer recorded as
    coerced (``import_notes``) is blank in the table, so it is reported as both.
    Results are ordered errors first, then warnings, then by code.
    """
    table = ds.table()
    active = ds.active_mask()
    fit_names = [x, y] if weights is None else [x, y, weights]
    for name in (*fit_names, *positive_required):
        if name not in table.columns:
            raise DataError("UNKNOWN_COLUMN", f"Column {name!r} does not exist")
    columns = {name: _read(table[name]) for name in dict.fromkeys((*fit_names, *positive_required))}

    non_numeric = _non_numeric_rows(ds, fit_names, columns, active)
    missing: dict[str, BoolArray] = {}
    non_finite: dict[str, BoolArray] = {}
    for name in fit_names:
        values = columns[name].values
        missing[name] = np.isnan(values) & ~columns[name].unparseable & active
        non_finite[name] = np.isinf(values) & active

    found: list[Issue] = []
    _add(found, "NON_NUMERIC", "error", "Non-numeric values in", non_numeric)
    _add(found, "MISSING_VALUE", "error", "Blank or NaN values in", missing)
    _add(found, "NON_FINITE", "error", "Infinite values in", non_finite)

    finite_rows = active.copy()
    for name in fit_names:
        finite_rows &= np.isfinite(columns[name].values)
    duplicates = _duplicate_rows([columns[n].values for n in fit_names], finite_rows)
    if duplicates:
        found.append(
            Issue(
                "DUPLICATE_ROW",
                "warning",
                f"Repeated ({', '.join(fit_names)}) observations",
                duplicates,
            )
        )

    n_active = int(active.sum())
    if n_active < MIN_ACTIVE_ROWS:
        found.append(
            Issue(
                "INSUFFICIENT_DATA",
                "error",
                f"{n_active} active row(s); at least {MIN_ACTIVE_ROWS} are required",
            )
        )

    x_values = columns[x].values
    finite_x = x_values[active & np.isfinite(x_values)]
    if finite_x.size >= 2 and finite_x.min() == finite_x.max():
        found.append(Issue("CONSTANT_X", "error", f"Every active value of {x!r} is identical"))

    nonpositive = {
        name: active & np.isfinite(columns[name].values) & (columns[name].values <= 0)
        for name in positive_required
    }
    _add(
        found,
        "NONPOSITIVE_FOR_LOG",
        "error",
        "Values <= 0 where positive values are required in",
        nonpositive,
    )

    if weights is not None:
        w = columns[weights].values
        bad_weight = active & ~np.isnan(w) & (~np.isfinite(w) | (w <= 0))
        _add(
            found,
            "INVALID_WEIGHT",
            "error",
            "Weights must be positive and finite in",
            {weights: bad_weight},
        )

    magnitudes = np.abs(finite_x[finite_x != 0])
    if magnitudes.size:
        with np.errstate(over="ignore"):
            ratio = magnitudes.max() / magnitudes.min()
        if ratio > WIDE_RANGE_RATIO:
            found.append(
                Issue(
                    "WIDE_DYNAMIC_RANGE",
                    "warning",
                    f"{x!r} spans more than 12 orders of magnitude; expect ill-conditioning",
                )
            )

    return sorted(found, key=lambda issue: (issue.severity != "error", issue.code))


def _read(series: pd.Series) -> _Column:
    if series.dtype.kind in "Mm":  # dates and durations are never numbers
        return _Column(np.full(len(series), np.nan), np.ones(len(series), dtype=bool))
    values, unparseable = coerce_numeric(series)
    return _Column(values, unparseable)


def _non_numeric_rows(
    ds: Dataset, names: Sequence[str], columns: dict[str, _Column], active: BoolArray
) -> dict[str, BoolArray]:
    """Unparseable active cells plus rows the importer recorded as ``coerced:<column>:<row>``."""
    rows = {name: columns[name].unparseable & active for name in names}
    for note in ds.import_notes:
        if not note.startswith(_COERCED_PREFIX):
            continue
        name, _, position = note[len(_COERCED_PREFIX) :].rpartition(":")
        if name in rows and position.isdecimal() and int(position) < len(active):
            index = int(position)
            if active[index]:
                rows[name][index] = True
    return rows


def _add(
    found: list[Issue], code: str, severity: Severity, label: str, per_column: dict[str, BoolArray]
) -> None:
    """Append one issue for ``code`` covering every column that has flagged rows."""
    flagged = {name: mask for name, mask in per_column.items() if mask.any()}
    if not flagged:
        return
    union = np.logical_or.reduce(list(flagged.values()))
    names = ", ".join(repr(name) for name in flagged)
    found.append(
        Issue(code, severity, f"{label} {names}", tuple(int(i) for i in np.flatnonzero(union)))
    )


def _duplicate_rows(arrays: list[FloatArray], eligible: BoolArray) -> tuple[int, ...]:
    seen: set[tuple[float, ...]] = set()
    repeated: list[int] = []
    for position in np.flatnonzero(eligible):
        key = tuple(float(array[position]) for array in arrays)
        if key in seen:
            repeated.append(int(position))
        else:
            seen.add(key)
    return tuple(repeated)
