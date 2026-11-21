"""``fit_to_xlsx``: the fit as an Excel workbook.

Sheets: Summary (identification, equation, status, warnings, conventions),
Parameters, Observations, Statistics and Plot (the embedded fit figure). Numbers
are real numeric cells; undefined values stay empty with a note. Text cells are
neutralised against spreadsheet formula injection, and the workbook properties
are pinned to a fixed epoch so two exports differ only by their content.
"""

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Font
from openpyxl.worksheet.worksheet import Worksheet

from curvestudio.fitting import FitStatus
from curvestudio.workflow import FitBundle, format_equation
from curvestudio_plotting import fit_figure, save_figure
from curvestudio_reporting.csv_report import (
    StatValue,
    aligned,
    safe_text,
    statistics_rows,
    subsampled_note,
)
from curvestudio_reporting.meta import (
    TOOL_NAME,
    assumptions_text,
    convention_of,
    generated_at,
    tool_versions,
)

FIXED_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
"""Pinned workbook creation time: the file bytes say nothing about when it was written."""

_STATUS_RED = Font(color="FFB2182B", bold=True)


def fit_to_xlsx(bundle: FitBundle, target: Path | BinaryIO) -> None:
    """Write the workbook for ``bundle`` to ``target`` (a path or binary stream)."""
    workbook = Workbook()
    _fix_properties(workbook)
    summary = workbook.active
    assert isinstance(summary, Worksheet)
    summary.title = "Summary"
    _write_summary(summary, bundle)
    _write_parameters(workbook.create_sheet("Parameters"), bundle)
    _write_observations(workbook.create_sheet("Observations"), bundle)
    _write_statistics(workbook.create_sheet("Statistics"), bundle)
    _write_plot(workbook.create_sheet("Plot"), bundle)
    workbook.save(target)


def _fix_properties(workbook: Workbook) -> None:
    workbook.properties.creator = TOOL_NAME
    workbook.properties.lastModifiedBy = TOOL_NAME
    workbook.properties.created = FIXED_EPOCH
    workbook.properties.modified = FIXED_EPOCH
    workbook.properties.title = "CurveStudio fit report"


def _label(sheet: Worksheet, label: str, value: StatValue, *, bold: bool = False) -> None:
    row = sheet.max_row + 1 if sheet.max_row and sheet["A1"].value is not None else 1
    sheet.cell(row=row, column=1, value=safe_text(label))
    if isinstance(value, str):
        value = safe_text(value)
    sheet.cell(row=row, column=2, value=value)
    if bold:
        sheet.cell(row=row, column=2).font = _STATUS_RED


def _write_summary(sheet: Worksheet, bundle: FitBundle) -> None:
    result = bundle.result
    provenance = result.provenance
    _label(sheet, "tool", TOOL_NAME)
    _label(
        sheet,
        "versions",
        "; ".join(f"{name} {version}" for name, version in tool_versions().items()),
    )
    _label(sheet, "generated_at", generated_at())
    _label(sheet, "x column", bundle.data.x_name)
    _label(sheet, "y column", bundle.data.y_name)
    _label(sheet, "observations", bundle.data.n_total)
    if (note := subsampled_note(bundle.data)) is not None:
        _label(sheet, "observation export", note)
    _label(sheet, "dataset fingerprint", provenance.dataset_fingerprint)
    _label(sheet, "seed", provenance.seed)
    _label(sheet, "model id", result.model_id)
    _label(sheet, "method", result.method)
    _label(sheet, "loss", result.loss)
    _label(sheet, "equation (model)", format_equation(result.equation))
    _label(sheet, "equation (fitted)", format_equation(bundle.equation_text))
    converged = result.status is FitStatus.CONVERGED
    _label(sheet, "status", result.status.value, bold=not converged)
    _label(sheet, "termination reason", result.termination_reason)
    for warning in bundle.warnings or ("no warnings",):
        _label(sheet, "warning", warning)
    convention = convention_of(bundle)
    _label(
        sheet,
        "information criteria convention",
        convention
        if convention is not None
        else f"not defined: the fit is {result.status.value}, so no criterion was computed",
    )
    _label(sheet, "assumptions and limitations", assumptions_text())
    sheet.column_dimensions["A"].width = 34.0
    sheet.column_dimensions["B"].width = 100.0


def _write_parameters(sheet: Worksheet, bundle: FitBundle) -> None:
    sheet.append(["parameter", "value", "std_err", "ci_low", "ci_high"])
    for estimate in bundle.result.parameters:
        sheet.append(
            [
                safe_text(estimate.name),
                estimate.value,
                estimate.std_err,
                estimate.ci_low,
                estimate.ci_high,
            ]
        )
    for column, width in zip("ABCDE", (18.0, 16.0, 12.0, 12.0, 12.0), strict=True):
        sheet.column_dimensions[column].width = width


def _write_observations(sheet: Worksheet, bundle: FitBundle) -> None:
    data = bundle.data
    fitted = aligned(bundle.result.fitted, data.x)
    residuals = aligned(bundle.result.residuals, data.x)
    weights = [None] * len(data.x) if data.weights is None else list(data.weights)
    sheet.append([safe_text(data.x_name), safe_text(data.y_name), "weight", "fitted", "residual"])
    for x, y, weight, fit_at_x, residual in zip(
        data.x, data.y, weights, fitted, residuals, strict=True
    ):
        sheet.append([x, y, weight, fit_at_x, residual])
    if (note := subsampled_note(data)) is not None:
        sheet.append([safe_text(note), None, None, None, None])


def _write_statistics(sheet: Worksheet, bundle: FitBundle) -> None:
    sheet.append(["statistic", "value", "note"])
    for name, value, note in statistics_rows(bundle):
        value = safe_text(value) if isinstance(value, str) else value
        sheet.append([safe_text(name), value, safe_text(note)])
    sheet.column_dimensions["A"].width = 18.0
    sheet.column_dimensions["C"].width = 90.0


def _write_plot(sheet: Worksheet, bundle: FitBundle) -> None:
    """Embed the fit figure PNG (a failed fit shows data + banner, never a curve)."""
    buffer = io.BytesIO()
    save_figure(fit_figure(bundle), buffer, "png")
    buffer.seek(0)
    sheet["A1"] = "fit figure"
    sheet.add_image(ExcelImage(buffer), "A3")
