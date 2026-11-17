"""CSV export of a fit: a parameters table, then one row per observation with fitted values.

Undefined values are empty cells; the ``note`` column says which cells are empty and why.
"""

import csv
import io
from collections.abc import Iterator

from curvestudio.data import XY
from curvestudio.workflow import FitBundle

CHUNK_ROWS = 5000
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _text(value: str) -> str:
    """A text cell that a spreadsheet will not run as a formula."""
    return "'" + value if value.startswith(_FORMULA_PREFIXES) else value


def _number(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _lines(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def csv_chunks(bundle: FitBundle, xy: XY) -> Iterator[str]:
    """The export as text chunks, in order."""
    result = bundle.result
    yield _lines(
        [
            ["# CurveStudio fit export"],
            ["# model", _text(result.equation_fitted or result.equation)],
            ["# status", result.status.value],
            ["# termination", _text(result.termination_reason)],
            ["# method", _text(result.method), "loss", _text(result.loss)],
            ["# rows", str(xy.x.size)],
            ["# empty cell = not defined (see the note column)"],
            [],
            ["# parameters"],
            ["name", "estimate", "std_err", "ci_low", "ci_high", "note"],
        ]
    )
    parameter_rows: list[list[str]] = []
    for p in result.parameters:
        missing = [
            label
            for label, value in (
                ("std_err", p.std_err),
                ("ci_low", p.ci_low),
                ("ci_high", p.ci_high),
            )
            if value is None
        ]
        note = f"not defined: {', '.join(missing)}" if missing else ""
        parameter_rows.append(
            [
                _text(p.name),
                _number(p.value),
                _number(p.std_err),
                _number(p.ci_low),
                _number(p.ci_high),
                note,
            ]
        )
    if not parameter_rows:
        parameter_rows.append(["", "", "", "", "", "no parameters: the fit did not converge"])
    yield _lines([*parameter_rows, [], ["# observations"]])
    header = ["x", "y", "weight", "fitted", "residual", "note"]
    yield _lines([header])
    fitted: list[float] | None = result.fitted if len(result.fitted) == xy.x.size else None
    residuals: list[float] | None = result.residuals if len(result.residuals) == xy.x.size else None
    n = int(xy.x.size)
    for start in range(0, n, CHUNK_ROWS):
        rows: list[list[str]] = []
        for i in range(start, min(start + CHUNK_ROWS, n)):
            weight = None if xy.weights is None else float(xy.weights[i])
            f = None if fitted is None else fitted[i]
            r = None if residuals is None else residuals[i]
            note = "" if f is not None and r is not None else "fitted/residual not defined"
            rows.append(
                [
                    _number(float(xy.x[i])),
                    _number(float(xy.y[i])),
                    _number(weight),
                    _number(f),
                    _number(r),
                    note,
                ]
            )
        yield _lines(rows)
