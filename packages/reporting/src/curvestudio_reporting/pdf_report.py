"""``fit_to_pdf``: the human-readable report, built with reportlab platypus.

Layout: title, the optional "Prepared by" byline, identification block (fingerprint,
seed, versions, IST date), equation, parameter table with SE and CI, statistics table,
the fit and residual figures, and an "Assumptions and limitations" section that
includes the exact information-criteria convention. A caller that gives the fit's methods
text and audit (``curvestudio.tools``) gets two more sections after the statistics:
**Methods** (the paragraph, the audit's methods fragment, the equation with its standard
errors) and **Fit audit** (the audit's summary, then one paragraph per check that reads
caution or fail: its title, its evidence and, after an arrow, its next step). A fit that
did not converge gets a red first-page warning and **no** figures of the fit — nothing
that could pass for a result — and prints no methods or audit either: the warning says
it. User-controlled strings that reach reportlab's *paragraph parser* (which
reads XML-ish markup) are XML-escaped there — names and equations can contain
``<`` and ``&``. Table cells are escaped ``Paragraph`` objects too: long cell
text wraps at the column width instead of overflowing, and display strings are
truncated past a sane maximum. The body font is the DejaVu Sans TTF that ships
with matplotlib (registered for reportlab and embedded as a subset), so Greek
and other non-Latin names render as real glyphs rather than black boxes; if no
TTF is present the report falls back to Helvetica and non-WinAnsi text degrades.

Known limitation: DejaVu covers Latin/Greek/Cyrillic and common symbols, but not
CJK, and reportlab does no complex-script shaping, so RTL scripts are not
covered either.
"""

import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import BinaryIO
from xml.sax.saxutils import escape

import matplotlib
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from curvestudio.fitting import FitStatus
from curvestudio.tools import FitAudit, MethodsText
from curvestudio.workflow import FitBundle, format_equation
from curvestudio_plotting import (
    FIT_FIGURE_SIZE,
    RESIDUAL_FIGURE_SIZE,
    Figure,
    fit_figure,
    residual_figure,
    save_figure,
)
from curvestudio_reporting.csv_report import (
    PREPARED_BY,
    byline_text,
    sanitise,
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

_MARGIN = 0.7 * inch
_WIDTH = A4[0] - 2 * _MARGIN
_MAX_CELL_CHARS = 200
"""Display maximum for one table cell; longer text is cut and ends in an ellipsis."""

#: The faint diagonal brand watermark on every page. ``curvestudio_plotting`` carries the
#: same literal separately: the import contracts forbid plotting importing reporting, so
#: the two packages duplicate the text by design rather than sharing it.
WATERMARK_TEXT = "ConcreteInfo CurveStudio"
DISCLAIMER_TEXT = (
    "Results describe agreement with the sample, not compliance or engineering advice"
    " — verify before use."
)
WATERMARK_ALPHA = 0.06
_BRAND_GREY = (0.275, 0.275, 0.275)  # #464646
_WATERMARK_ANGLE = 35
_WATERMARK_FONT_SIZE = 34
_FOOTER_FONT_SIZE = 6.5
_FOOTER_ALPHA = 0.75
_FOOTER_GREY = colors.HexColor("#666666")
_BYLINE_FONT_SIZE = 8.5
_BYLINE_LEADING = 10.5
#: The footer baseline, 18 pt (~6.4 mm) above the page bottom: below the 0.7 in content
#: margin, so the disclaimer can never collide with body text or a figure.
_FOOTER_Y = 18.0


def _register_unicode_fonts() -> tuple[str, str]:
    """Register DejaVu Sans (shipped by matplotlib) as the PDF body font.

    Returns ``(normal, bold)`` font names — the DejaVu pair when the TTFs are
    present, Helvetica otherwise (a degraded fallback for stripped installs).
    """
    fonts_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular, bold = fonts_dir / "DejaVuSans.ttf", fonts_dir / "DejaVuSans-Bold.ttf"
    if not regular.is_file() or not bold.is_file():
        return "Helvetica", "Helvetica-Bold"
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold)))
    registerFontFamily(
        "DejaVuSans",
        normal="DejaVuSans",
        bold="DejaVuSans-Bold",
        italic="DejaVuSans",
        boldItalic="DejaVuSans-Bold",
    )
    return "DejaVuSans", "DejaVuSans-Bold"


_BODY_FONT, _BOLD_FONT = _register_unicode_fonts()


def _decorate_page(canvas: Canvas, _doc: SimpleDocTemplate) -> None:
    """Paint one page's faint diagonal watermark and footer disclaimer.

    ReportLab calls this at page start (``onFirstPage``/``onLaterPages``), so the
    watermark is painted underneath the flowables rather than over the text.
    """
    canvas.saveState()
    # the big diagonal brand mark, centred on the page, behind the content
    canvas.setFillAlpha(WATERMARK_ALPHA)
    canvas.setFillColorRGB(*_BRAND_GREY)
    canvas.translate(A4[0] / 2, A4[1] / 2)
    canvas.rotate(_WATERMARK_ANGLE)
    canvas.setFont(_BOLD_FONT, _WATERMARK_FONT_SIZE)
    canvas.drawCentredString(0, 0, WATERMARK_TEXT)
    canvas.restoreState()
    # the disclaimer sits in the bottom margin, under every page's content
    canvas.saveState()
    canvas.setFillColor(_FOOTER_GREY)
    canvas.setFillAlpha(_FOOTER_ALPHA)
    canvas.setFont(_BODY_FONT, _FOOTER_FONT_SIZE)
    canvas.drawCentredString(A4[0] / 2, _FOOTER_Y, DISCLAIMER_TEXT)
    canvas.restoreState()


def fit_to_pdf(
    bundle: FitBundle,
    target: Path | BinaryIO,
    *,
    byline: str | None = None,
    audit: FitAudit | None = None,
    methods: MethodsText | None = None,
) -> None:
    """Write the PDF report for ``bundle`` to ``target`` (a path or binary stream).

    ``byline`` is the identity text alone (a name, an organisation, or both). When it is
    set, the report prints "Prepared by <byline>" in a small muted style directly under
    the title and the identity alone becomes the document's author. Without one the title
    is followed by the report as it has always been, with the tool as the author.

    ``methods`` and ``audit`` are the tools layer's methods text and fit audit of the same
    fit (:func:`curvestudio.tools.methods_text` in the ``"text"`` style, and
    :func:`curvestudio.tools.audit_fit`). Given, a converged fit's report carries two more
    sections after the statistics table: **Methods**, the paragraph, then the audit's
    methods fragment, then the equation with its estimates and standard errors; and **Fit
    audit**, the audit's summary, then one paragraph for each check that reads caution or
    fail (its title, its evidence and, after an arrow, its next step). A check that passed
    or could not run prints nothing there. Given neither, the report is the one it has
    always been. A fit that did not converge prints neither section, whatever is given:
    its red banner already says it.
    """
    shown = byline_text(byline)
    filename: str | BinaryIO = str(target) if isinstance(target, Path) else target
    document = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title="CurveStudio fit report",
        author=TOOL_NAME if shown is None else shown,
        # read at call time so a caller can flip rl_config.pageCompression
        pageCompression=rl_config.pageCompression,
    )
    # reportlab 5 builds one page template per callback; passing them to build() is the
    # supported hook (the constructor ignores unknown keyword arguments such as these)
    document.build(
        _flowables(bundle, shown, audit=audit, methods=methods),
        onFirstPage=_decorate_page,
        onLaterPages=_decorate_page,
    )


def _flowables(
    bundle: FitBundle,
    byline: str | None = None,
    *,
    audit: FitAudit | None = None,
    methods: MethodsText | None = None,
) -> list[Flowable]:
    styles = getSampleStyleSheet()
    for style in styles.byName.values():  # DejaVu everywhere, not just the tables
        style.fontName = _BODY_FONT
    warning_style = ParagraphStyle(
        "NotConverged",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#B2182B"),
    )
    byline_style = ParagraphStyle(  # after the loop: it copies the body style as it now is
        "Byline",
        parent=styles["BodyText"],
        fontSize=_BYLINE_FONT_SIZE,
        leading=_BYLINE_LEADING,
        textColor=_FOOTER_GREY,
    )
    result = bundle.result
    converged = result.status is FitStatus.CONVERGED
    story: list[Flowable] = [Paragraph(escape("CurveStudio fit report"), styles["Title"])]
    if byline is not None:
        story.append(Paragraph(escape(f"{PREPARED_BY}{byline}"), byline_style))
    story.append(Spacer(1, 10))

    if not converged:
        banner = escape(
            sanitise(
                f"FIT DID NOT CONVERGE: {result.termination_reason} "
                f"(status: {result.status.value}). No fitted curve, band or figure of "
                "this fit is shown; the numbers below are the recorded outcome, not a fit."
            )
        )
        story += [Paragraph(banner, warning_style), Spacer(1, 12)]

    provenance = result.provenance
    versions = "; ".join(f"{name} {version}" for name, version in tool_versions().items())
    observations = str(bundle.data.n_total)
    if (note := subsampled_note(bundle.data)) is not None:
        observations = f"{observations} — {note}"
    identification = [
        ["generated_at", generated_at()],
        ["tool", f"{TOOL_NAME} ({versions})"],
        ["x column", bundle.data.x_name],
        ["y column", bundle.data.y_name],
        ["observations", observations],
        ["dataset fingerprint", provenance.dataset_fingerprint],
        ["seed", str(provenance.seed)],
        ["model id", result.model_id],
        ["method", result.method],
        ["loss", result.loss],
        ["status", result.status.value],
        ["termination reason", result.termination_reason],
    ]
    story += [
        _kv_table(identification),
        Spacer(1, 14),
        Paragraph("Model equation", styles["Heading2"]),
        Spacer(1, 4),
        Paragraph(escape(sanitise(format_equation(bundle.equation_text))), styles["BodyText"]),
        Spacer(1, 14),
    ]
    for warning in bundle.warnings:
        story.append(Paragraph(escape(sanitise(f"Warning: {warning}")), styles["BodyText"]))
    if bundle.warnings:
        story.append(Spacer(1, 14))

    story += [
        Paragraph("Parameters", styles["Heading2"]),
        Spacer(1, 4),
        _parameter_table(bundle),
        Spacer(1, 14),
        Paragraph("Statistics", styles["Heading2"]),
        Spacer(1, 4),
        _statistics_table(bundle),
    ]
    if converged:
        # a fit that did not converge has no methods to state and nothing to audit, and its
        # banner stands in for its figures
        story += _methods_section(styles, methods, audit)
        story += _audit_section(styles, audit)
        story += [
            PageBreak(),
            Paragraph("Figures", styles["Heading2"]),
            Spacer(1, 4),
            _figure_paragraph(bundle, fit_figure),
            Spacer(1, 10),
            _figure_paragraph(bundle, residual_figure),
        ]

    convention = convention_of(bundle)
    if convention is not None:
        convention_text = convention
    else:
        convention_text = (
            f"not defined: the fit is {result.status.value}, "
            "so no information criterion was computed"
        )
    diagnostics_notes = _diagnostics_notes(bundle)
    story += [
        Spacer(1, 14),
        Paragraph("Assumptions and limitations", styles["Heading2"]),
        Spacer(1, 4),
        Paragraph(escape(assumptions_text()), styles["BodyText"]),
        Spacer(1, 8),
        Paragraph(
            escape(f"Information-criteria convention: {convention_text}"), styles["BodyText"]
        ),
    ]
    if diagnostics_notes:
        story.append(Spacer(1, 8))
        story.append(Paragraph(escape(diagnostics_notes), styles["BodyText"]))
    return story


def _methods_section(
    styles: StyleSheet1, methods: MethodsText | None, audit: FitAudit | None
) -> list[Flowable]:
    """The Methods section: the methods paragraph with the audit's fragment, then the equation.

    Empty when neither is given: the report then has no such section, as it never had.
    """
    paragraph = methods_paragraph(methods, audit)
    if not paragraph:
        return []
    story: list[Flowable] = [
        Spacer(1, 14),
        Paragraph("Methods", styles["Heading2"]),
        Spacer(1, 4),
        Paragraph(escape(sanitise(paragraph)), styles["BodyText"]),
    ]
    if methods is not None:
        # the equation keeps its lines: one paragraph, broken where the text breaks it
        equation = "<br/>".join(escape(sanitise(line)) for line in methods.equation.split("\n"))
        story += [Spacer(1, 8), Paragraph(equation, styles["BodyText"])]
    return story


def _audit_section(styles: StyleSheet1, audit: FitAudit | None) -> list[Flowable]:
    """The Fit audit section: the audit's summary, then every check that reads caution or fail.

    Each is one paragraph — its title, its evidence and, after an arrow, its next step —
    with a gap before it, so two checks never read as one. When no check is flagged the
    summary is the whole section: it says that every check that could run passed.
    """
    if audit is None:
        return []
    story: list[Flowable] = [
        Spacer(1, 14),
        Paragraph("Fit audit", styles["Heading2"]),
        Spacer(1, 4),
        Paragraph(escape(sanitise(audit.summary)), styles["BodyText"]),
    ]
    for check in flagged_checks(audit):
        line = f"{check.title}. {check.evidence}"
        if check.next_step is not None:  # the tool leaves it unset where it has no advice
            line = f"{line} → {check.next_step}"
        story += [Spacer(1, 6), Paragraph(escape(sanitise(line)), styles["BodyText"])]
    return story


def _diagnostics_notes(bundle: FitBundle) -> str:
    """Per-test assumptions and limitations, joined into one honest paragraph."""
    if bundle.summary is None:
        return ""
    parts = []
    for test in bundle.summary.diagnostics.tests:
        assumptions = " ".join(test.assumptions)
        limitations = " ".join(test.limitations).rstrip(".")  # each entry ends in one already
        parts.append(
            f"{test.name} (null hypothesis: {test.null_hypothesis}) assumes {assumptions}; "
            f"it cannot show that: {limitations}."
        )
    return " ".join(parts)


def _kv_table(rows: list[list[str]]) -> Table:
    """Identification table: key column bold, values wrapped and truncated."""
    styles = _table_styles()
    cells = [
        [
            _cell(text, styles["cell_key"] if column == 0 else styles["cell"])
            for column, text in enumerate(row)
        ]
        for row in rows
    ]
    table = Table(cells, colWidths=[1.6 * inch, _WIDTH - 1.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _parameter_table(bundle: FitBundle) -> Table:
    header = ["parameter", "value", "std_err", "95% CI low", "95% CI high"]
    rows = [header]
    for estimate in bundle.result.parameters:
        rows.append(
            [
                estimate.name,
                repr(estimate.value),
                "" if estimate.std_err is None else repr(estimate.std_err),
                "" if estimate.ci_low is None else repr(estimate.ci_low),
                "" if estimate.ci_high is None else repr(estimate.ci_high),
            ]
        )
    if len(bundle.result.parameters) == 0:
        rows.append(["(no parameters were estimated)", "", "", "", ""])
    # one explicit width per column, summing to the frame width (never wider)
    widths = [fraction * _WIDTH for fraction in (0.30, 0.20, 0.16, 0.17, 0.17)]
    return _data_table(rows, widths)


def _statistics_table(bundle: FitBundle) -> Table:
    rows = [["statistic", "value", "note"]]
    for name, value, note in statistics_rows(bundle):
        cell = "" if value is None else str(value)
        rows.append([name, cell, note])
    widths = [fraction * _WIDTH for fraction in (0.16, 0.20, 0.64)]
    return _data_table(rows, widths)


def _data_table(rows: list[list[str]], widths: Sequence[float]) -> Table:
    assert len(widths) == len(rows[0]), "one width per column"
    assert sum(widths) <= _WIDTH + 1e-6, "a table must never exceed the frame width"
    styles = _table_styles()
    cells = [
        [_cell(text, styles["cell_header"] if row == 0 else styles["cell"]) for text in row_cells]
        for row, row_cells in enumerate(rows)
    ]
    table = Table(cells, colWidths=list(widths), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _table_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cell": ParagraphStyle(
            "TableCell", parent=base["BodyText"], fontName=_BODY_FONT, fontSize=8, leading=9.6
        ),
        "cell_header": ParagraphStyle(
            "TableHeaderCell", parent=base["BodyText"], fontName=_BOLD_FONT, fontSize=8, leading=9.6
        ),
        "cell_key": ParagraphStyle(
            "TableKeyCell", parent=base["BodyText"], fontName=_BOLD_FONT, fontSize=8, leading=9.6
        ),
    }


def _cell(text: str, style: ParagraphStyle) -> Paragraph:
    """One table cell: control characters stripped, display-truncated, XML-escaped.

    A ``Paragraph`` cell wraps at the column width (a 3000-character name cannot
    overflow the page) and escapes markup, so ``<b>`` stays literal text.
    """
    shown = sanitise(text)
    if len(shown) > _MAX_CELL_CHARS:
        shown = shown[:_MAX_CELL_CHARS] + "…"
    return Paragraph(escape(shown), style)


def _figure_paragraph(bundle: FitBundle, builder: Callable[[FitBundle], Figure]) -> Image:
    """Render one figure to PNG in memory and place it at document width."""
    figure = builder(bundle)
    height_inches = RESIDUAL_FIGURE_SIZE[1] if builder is residual_figure else FIT_FIGURE_SIZE[1]
    scale = (_WIDTH / inch) / FIT_FIGURE_SIZE[0]
    buffer = io.BytesIO()
    save_figure(figure, buffer, "png")
    buffer.seek(0)
    return Image(
        buffer,
        width=_WIDTH,
        height=height_inches * inch * scale,
        lazy=0,
    )
