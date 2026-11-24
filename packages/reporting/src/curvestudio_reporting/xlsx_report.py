"""``fit_to_xlsx``: the fit as an Excel workbook.

Sheets: Summary (identification, equation, status, warnings, conventions),
Parameters, Observations, Statistics and Plot (the embedded fit figure). Numbers
are real numeric cells; undefined values stay empty with a note. Text cells are
neutralised against spreadsheet formula injection, and the workbook properties
are pinned to a fixed epoch so two exports differ only by their content. An
optional "Prepared by" byline prints in every sheet's page header. A caller that
gives the fit's methods text and audit (``curvestudio.tools``) gets two more
blocks on the Summary sheet under the line that names the convention: the
methods paragraph (with the audit's methods fragment) and a "Fit audit" block
that lists, one row each, the checks that read caution or fail.
"""

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.worksheet import Worksheet

from curvestudio.fitting import FitStatus
from curvestudio.tools import FitAudit, MethodsText
from curvestudio.workflow import FitBundle, format_equation
from curvestudio_plotting import fit_figure, save_figure
from curvestudio_reporting.csv_report import (
    PREPARED_BY,
    StatValue,
    aligned,
    byline_text,
    safe_text,
    statistics_rows,
    subsampled_note,
)
from curvestudio_reporting.meta import (
    TOOL_NAME,
    assumptions_text,
    convention_of,
    flagged_checks,
    generated_at,
    methods_paragraph,
    tool_versions,
)

FIXED_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
"""Pinned workbook creation time: the file bytes say nothing about when it was written."""

_STATUS_RED = Font(color="FFB2182B", bold=True)
_NEXT_STEP_WIDTH = 60.0
"""Column width of a flagged check's next step on the Summary sheet (evidence sits in B)."""

#: Same literal as the PDF report's watermark and disclaimer. They are repeated here so
#: the spreadsheet writer never imports the PDF renderer (and reportlab with it).
WATERMARK_TEXT = "ConcreteInfo CurveStudio"
DISCLAIMER_TEXT = (
    "Results describe agreement with the sample, not compliance or engineering advice"
    " — verify before use."
)
_FOOTER_FONT_SIZE = 7
_FOOTER_COLOR = "808080"
_DISCLAIMER_FONT = Font(color="FFA6A6A6", italic=True, size=8)


def fit_to_xlsx(
    bundle: FitBundle,
    target: Path | BinaryIO,
    *,
    byline: str | None = None,
    audit: FitAudit | None = None,
    methods: MethodsText | None = None,
) -> None:
    """Write the workbook for ``bundle`` to ``target`` (a path or binary stream).

    ``byline`` is the identity text alone (a name, an organisation, or both). When it is
    set, "Prepared by <byline>" prints in the centre of every sheet's page header, in the
    footer's size and colour. Without one the sheets carry no header, as before; nothing
    else in the workbook depends on it.

    ``methods`` and ``audit`` are the tools layer's methods text and fit audit of the same
    fit (:func:`curvestudio.tools.methods_text` and :func:`curvestudio.tools.audit_fit`).
    Given, the Summary sheet carries two more blocks under the line that names the
    convention: the methods paragraph, followed by the audit's methods fragment; and a
    "Fit audit" block — the audit's summary, then one row (title, evidence, next step) for
    each check that reads caution or fail, or that summary alone when none does, which
    says that every check that could run passed. Given neither, the workbook is the one it
    has always been. A fit that did not converge gets neither block, whatever is given:
    its status row already says it, in red.
    """
    workbook = Workbook()
    _fix_properties(workbook)
    summary = workbook.active
    assert isinstance(summary, Worksheet)
    summary.title = "Summary"
    _write_summary(summary, bundle, audit=audit, methods=methods)
    _write_parameters(workbook.create_sheet("Parameters"), bundle)
    _write_observations(workbook.create_sheet("Observations"), bundle)
    _write_statistics(workbook.create_sheet("Statistics"), bundle)
    _write_plot(workbook.create_sheet("Plot"), bundle)
    _decorate_sheets(workbook, byline_text(byline))
    workbook.save(target)


def _decorate_sheets(workbook: Workbook, byline: str | None = None) -> None:
    """Print the brand line and disclaimer in every sheet's page footer; a byline in its header."""
    footer = f"{WATERMARK_TEXT} — {DISCLAIMER_TEXT}"
    # '&' introduces a header command in Excel (&B is bold, && a literal ampersand)
    header = None if byline is None else f"{PREPARED_BY}{byline}".replace("&", "&&")
    for sheet in workbook.worksheets:
        sheet.oddFooter.center.text = footer
        sheet.oddFooter.center.size = _FOOTER_FONT_SIZE
        sheet.oddFooter.center.color = _FOOTER_COLOR
        if header is not None:  # untouched without a byline: the file then stays as it was
            sheet.oddHeader.center.text = header
            sheet.oddHeader.center.size = _FOOTER_FONT_SIZE
            sheet.oddHeader.center.color = _FOOTER_COLOR


def _fix_properties(workbook: Workbook) -> None:
    workbook.properties.creator = TOOL_NAME
    workbook.properties.lastModifiedBy = TOOL_NAME
    workbook.properties.created = FIXED_EPOCH
    workbook.properties.modified = FIXED_EPOCH
    workbook.properties.title = "CurveStudio fit report"
    workbook.properties.description = DISCLAIMER_TEXT


def _label(sheet: Worksheet, label: str, value: StatValue, *, bold: bool = False) -> None:
    row = sheet.max_row + 1 if sheet.max_row and sheet["A1"].value is not None else 1
    sheet.cell(row=row, column=1, value=safe_text(label))
    if isinstance(value, str):
        value = safe_text(value)
    sheet.cell(row=row, column=2, value=value)
    if bold:
        sheet.cell(row=row, column=2).font = _STATUS_RED


def _write_summary(
    sheet: Worksheet,
    bundle: FitBundle,
    *,
    audit: FitAudit | None = None,
    methods: MethodsText | None = None,
) -> None:
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
    if converged:  # a fit that did not converge has no methods to state and nothing to audit
        _write_methods(sheet, methods, audit)
        _write_fit_audit(sheet, audit)
    _label(sheet, "assumptions and limitations", assumptions_text())
    _write_disclaimer(sheet)
    sheet.column_dimensions["A"].width = 34.0
    sheet.column_dimensions["B"].width = 100.0


def _write_methods(sheet: Worksheet, methods: MethodsText | None, audit: FitAudit | None) -> None:
    """The methods paragraph with the audit's methods fragment; no row when neither is given."""
    paragraph = methods_paragraph(methods, audit)
    if paragraph:
        _label(sheet, "methods paragraph", paragraph)


def _write_fit_audit(sheet: Worksheet, audit: FitAudit | None) -> None:
    """The "Fit audit" block: the audit's summary, then a row per check that reads caution or fail.

    A check's row is its title, its evidence and, in a third column, its next step. With no
    flagged check the summary row is the block: it says that every check that could run
    passed.
    """
    if audit is None:
        return
    _label(sheet, "Fit audit", audit.summary)
    flagged = flagged_checks(audit)
    for check in flagged:
        _label(sheet, check.title, check.evidence)
        if check.next_step is not None:  # the tool leaves it unset where it has no advice
            sheet.cell(row=sheet.max_row, column=3, value=safe_text(check.next_step))
    if flagged:
        sheet.column_dimensions["C"].width = _NEXT_STEP_WIDTH


def _write_disclaimer(sheet: Worksheet) -> None:
    """One grey italic disclaimer row below the summary's identification lines."""
    row = sheet.max_row + 2
    cell = sheet.cell(row=row, column=1, value=safe_text(DISCLAIMER_TEXT))
    cell.font = _DISCLAIMER_FONT
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)


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
