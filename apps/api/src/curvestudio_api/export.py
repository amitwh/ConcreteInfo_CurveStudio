"""Export of a fit and of an Auto-fit run: the CSV writers here, and the XLSX workbook
(a summary with the fit's methods paragraph and audit, a parameters table, one row per
observation with fitted values, statistics and a plot sheet) through ``curvestudio_reporting``.

Undefined values are empty cells; in a fit export the ``note`` column says which cells are
empty and why.
"""

import csv
import io
from collections.abc import Iterator

from curvestudio.data import XY
from curvestudio.discovery import AutoFitResult, RankBy
from curvestudio.tools import FitAudit, MethodsText
from curvestudio.workflow import FitBundle
from curvestudio_reporting import fit_to_xlsx

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

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


def xlsx_bytes(
    bundle: FitBundle,
    *,
    audit: FitAudit | None = None,
    methods: MethodsText | None = None,
) -> bytes:
    """The fit as an Excel workbook (summary, parameters, observations, statistics, plot),
    written into memory and ready to stream.

    ``audit`` and ``methods`` are the fit's audit and methods text; given, the summary
    sheet carries them (see ``fit_to_xlsx``).
    """
    buffer = io.BytesIO()
    fit_to_xlsx(bundle, buffer, audit=audit, methods=methods)
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


AUTOFIT_COLUMNS = (
    "rank",
    "id",
    "name",
    "category",
    "equation",
    "k",
    "r2",
    "adj_r2",
    "rmse",
    "aicc",
    "bic",
    "cv_rmse",
    "status",
    "notes",
)


def autofit_csv(result: AutoFitResult, rank_by: RankBy) -> str:
    """One table: the ranked candidates in ``rank_by`` order, then every other candidate
    (not converged, failed, timed out, skipped, or an exact fit that ``rank_by`` cannot place)
    without a rank. ``status`` is the fit status or the outcome; ``notes`` holds warnings,
    same-curve hints and the reason a candidate has no rank. Empty cell = not defined."""
    rows: list[list[str]] = [list(AUTOFIT_COLUMNS)]
    ranked = result.ranked_by(rank_by)
    for row in ranked:
        m = row.metrics
        notes = [*row.warnings]
        if row.same_curve_as:
            notes.append("same curve as " + ", ".join(row.same_curve_as))
        if row.overfitting_note:
            notes.append(row.overfitting_note)
        rows.append(
            [
                str(row.rank),
                _text(row.model_id),
                _text(row.name),
                _text(row.category),
                _text(row.equation_fitted or ""),
                str(m.k),
                _number(m.r2),
                _number(m.adj_r2),
                _number(m.rmse),
                _number(m.aicc),
                _number(m.bic),
                _number(m.cv_rmse),
                _text(row.status),
                _text("; ".join(notes)),
            ]
        )
    placed = {row.model_id for row in ranked}
    for o in result.outcomes:
        if o.candidate_id in placed:
            continue
        unplaced = o.metrics
        equation = (o.fit.equation_fitted if o.fit is not None else None) or ""
        rows.append(
            [
                "",
                _text(o.candidate_id),
                _text(o.name),
                _text(o.category),
                _text(equation),
                "" if unplaced is None else str(unplaced.k),
                *(
                    _number(None if unplaced is None else getattr(unplaced, name))
                    for name in ("r2", "adj_r2", "rmse", "aicc", "bic", "cv_rmse")
                ),
                o.outcome,
                _text(o.reason),
            ]
        )
    return _lines(rows)
