"""Off-thread export bodies: result reports, the ranking table, publication plots.

File writing and figure rendering never belong on the GUI thread, so each export is a
plain function the window submits through its JobRunner. The ranking CSV writes through
the :mod:`csv` module (quoting and embedded quotes handled for real) and applies the
reporting layer's formula neutralisation to the *text* columns only — a negative AICc is
a number, never something a spreadsheet would execute.
"""

import csv
import io
from pathlib import Path
from typing import cast

from curvestudio.discovery import AutoFitResult, RankBy
from curvestudio.workflow import FitBundle, format_equation
from curvestudio_plotting import (
    FigureFormat,
    add_byline,
    fit_figure,
    residual_figure,
    save_figure,
)
from curvestudio_reporting import (
    fit_to_json,
    fit_to_pdf,
    fit_to_xlsx,
    observations_csv,
    parameters_csv,
    safe_text,
    statistics_csv,
)

RANKING_HEADER = (
    "rank",
    "model",
    "category",
    "equation_fitted",
    "r2",
    "adj_r2",
    "rmse",
    "aicc",
    "bic",
    "cv_rmse",
    "k",
    "status",
    "notes",
)
_TEXT_COLUMNS = frozenset({1, 2, 3, 11, 12})
"""model, category, equation, status, notes: the columns that carry free text."""


def _metric(value: float | None) -> str:
    return "" if value is None else repr(value)


def ranking_csv(result: AutoFitResult, rank_by: RankBy) -> str:
    """The ranking table as CSV: text columns neutralised, numbers left as numbers."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(RANKING_HEADER)
    for row in result.ranked_by(rank_by):
        notes = []
        if row.same_curve_as:
            notes.append("same curve as " + "; ".join(row.same_curve_as))
        if row.degenerate:
            notes.append("degenerate")
        if row.redundant_with is not None:
            notes.append(f"redundant with {row.redundant_with}")
        if row.overfitting_note is not None:
            notes.append(row.overfitting_note)
        metrics = row.metrics
        cells = (
            str(row.rank),
            row.name,
            row.category,
            format_equation(row.equation_fitted),
            _metric(metrics.r2),
            _metric(metrics.adj_r2),
            _metric(metrics.rmse),
            _metric(metrics.aicc),
            _metric(metrics.bic),
            _metric(metrics.cv_rmse),
            str(metrics.k),
            row.status,
            "; ".join(notes),
        )
        writer.writerow(
            safe_text(cell) if column in _TEXT_COLUMNS else cell
            for column, cell in enumerate(cells)
        )
    return out.getvalue()


def export_ranking_csv(target: Path, result: AutoFitResult, rank_by: RankBy) -> None:
    target.write_text(ranking_csv(result, rank_by), encoding="utf-8", newline="")


def export_fit(target: Path, bundle: FitBundle, *, byline: str | None = None) -> None:
    """Write one fit's report; the suffix picks the format (csv writes the three tables).

    A ``.csv`` export writes ``<stem>.csv`` (parameters) plus ``<stem>_statistics.csv`` and
    ``<stem>_observations.csv``: the reporting layer deliberately produces three plain
    tables rather than one multi-section file, and the desktop export keeps that shape.
    A set ``byline`` is the identity text (a name, an organisation, or both); the PDF
    report and the workbook print it as a "Prepared by …" line. JSON and CSV carry none.
    """
    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text(fit_to_json(bundle), encoding="utf-8")
    elif suffix == ".xlsx":
        fit_to_xlsx(bundle, target, byline=byline)
    elif suffix == ".pdf":
        fit_to_pdf(bundle, target, byline=byline)
    elif suffix == ".csv":
        target.write_text(parameters_csv(bundle), encoding="utf-8", newline="")
        target.with_name(f"{target.stem}_statistics.csv").write_text(
            statistics_csv(bundle), encoding="utf-8", newline=""
        )
        target.with_name(f"{target.stem}_observations.csv").write_text(
            observations_csv(bundle), encoding="utf-8", newline=""
        )
    else:
        raise ValueError(f"unsupported export format {suffix!r}; use .json .xlsx .pdf or .csv")


def export_publication_plot(
    target: Path, bundle: FitBundle, residuals: bool = False, *, byline: str | None = None
) -> None:
    """The Matplotlib publication figure of the fit (or its residual panels) as PNG/SVG/PDF.

    A set ``byline`` is the identity text (a name, an organisation, or both); the figure
    prints it as a "Prepared by …" line along its top edge.
    """
    fmt = target.suffix.lower().lstrip(".")
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError(f"unsupported plot format {target.suffix!r}; use .png .svg or .pdf")
    figure = residual_figure(bundle) if residuals else fit_figure(bundle)
    if byline and byline.strip():  # "set": neither missing nor blank
        add_byline(figure, byline)
    save_figure(figure, target, cast("FigureFormat", fmt))


__all__ = [
    "export_fit",
    "export_publication_plot",
    "export_ranking_csv",
    "ranking_csv",
]
