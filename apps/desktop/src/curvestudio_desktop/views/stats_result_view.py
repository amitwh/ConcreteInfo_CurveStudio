"""StatsResultView: the desktop's one view for what a statistical tool returned.

A tool returns ONE :class:`~curvestudio.tools.stats.ToolResult`, built from blocks, and
this view shows every block in the order a person reads them: the title and the confidence
level the intervals were built at, the estimates, the tests (a test that could not run
keeps its reason right under its row), one card per assumption with its verdict badge, the
warnings, every result table with its title and note, the interpretation, the methods
fragment with a Copy methods text control, and the counts with the rows that took no part
— folded until opened, so a long list never pushes the reading out of sight.

The tools' words are never reworded here. A number the tool could not give is an em dash,
never 0; a p-value is read through :func:`p_text` wherever it appears, so the same p reads
the same on every surface. A result that has not been run shows one line telling the
person what to do. :meth:`StatsResultView.texts` returns the words of every block, which
is how the offscreen tests read the view without relying on how it paints.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from curvestudio.tools import AuditCheck, Verdict
from curvestudio.tools.stats import Counts, Table, ToolResult, p_text
from curvestudio.tools.stats._display import Cell, ColumnKind
from curvestudio_desktop.theme import mono_font
from curvestudio_desktop.views.fit_audit_view import AuditBlockView, en_in_digits
from curvestudio_desktop.views.fit_summary import apply_verdict_badge
from curvestudio_desktop.views.ranking_table import format_metric

EMPTY_TEXT = "Run a tool to see its result here."
"""What the view says before any tool has been run."""
MISSING = "—"
"""What a table cell or a number the tool could not give reads as (never 0)."""
DETAILS_LABEL = "Counts and excluded rows"
"""The control that unfolds the counts and the rows that took no part."""


def _line(name: str, text: str = "") -> QLabel:
    """A wrapping, selectable line of the result."""
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _heading(text: str) -> QLabel:
    label = _line("auditTitle", text)
    return label


def _number(value: float | None) -> str:
    """A number as the app writes it, or an em dash when the tool could not give it."""
    return MISSING if value is None else format_metric(value)


def _cell_text(cell: Cell, kind: ColumnKind) -> str:
    """One table cell as the view writes it; a missing number is an em dash."""
    if cell is None:
        return MISSING
    if kind == "p_value":
        return p_text(float(cell))
    if kind == "int":
        return str(int(cell))
    if kind == "number":
        return format_metric(float(cell))
    return str(cell)


def _counts_text(counts: Counts) -> str:
    """``250 offered, 50 used, 250 excluded``, in the app's digit grouping."""
    offered = en_in_digits(counts.n_offered)
    used = en_in_digits(counts.n_used)
    excluded = en_in_digits(counts.n_excluded)
    return f"{offered} offered, {used} used, {excluded} excluded"


@dataclass(frozen=True)
class _Card:
    """One assumption's card: the widget, its badge's words, and its block as the audit reads it."""

    widget: QWidget
    badge: str
    view: AuditBlockView


def _build_card(check: AuditCheck) -> _Card:
    """One assumption as a card, in the fit audit's pattern and the tool's own words."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(3)
    head = QHBoxLayout()
    badge = QLabel("")
    badge.setObjectName("statusBadge")
    title = _line("auditTitle", check.title)
    head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
    head.addWidget(title, 1)
    layout.addLayout(head)
    layout.addWidget(_line("auditEvidence", check.evidence))
    quiet = check.verdict is Verdict.NOT_APPLICABLE
    if check.statistic is not None:
        layout.addWidget(_line("auditStatistic", f"Statistic: {check.statistic}"))
    if check.threshold is not None:
        layout.addWidget(_line("auditThreshold", f"Threshold: {check.threshold}"))
    if not quiet:
        layout.addWidget(_line("auditExplanation", check.explanation))
    if check.next_step is not None:
        next_step = _line("auditNextStep", check.next_step)
        next_step.setContentsMargins(12, 0, 0, 0)
        layout.addWidget(next_step)
    apply_verdict_badge(badge, check.verdict)
    return _Card(
        widget=widget,
        badge=badge.text(),
        view=AuditBlockView(
            check_id=check.id,
            state=str(badge.property("state")),
            title=check.title,
            evidence=check.evidence,
            statistic="" if check.statistic is None else check.statistic,
            threshold="" if check.threshold is None else check.threshold,
            explanation="" if quiet else check.explanation,
            next_step="" if check.next_step is None else check.next_step,
        ),
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], name: str) -> QTableWidget:
    """A read-only table whose columns keep their content's width."""
    table = QTableWidget(len(rows), len(headers))
    table.setObjectName(name)
    table.setHorizontalHeaderLabels(list(headers))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setWordWrap(False)
    table.setFont(mono_font(9))
    table.verticalHeader().setVisible(False)
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    for row_index, row in enumerate(rows):
        for column, text in enumerate(row):
            item = QTableWidgetItem(text)
            if column:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, column, item)
    return table


class StatsResultView(QWidget):
    """One statistical tool's result, every block of it, with its plots shown elsewhere."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statsResultView")
        self._result: ToolResult | None = None
        self._texts: list[str] = [EMPTY_TEXT]
        self._cards: list[_Card] = []
        self._details_open = False
        self._details_body: QWidget | None = None
        self._details_toggle: QToolButton | None = None
        self._empty = QLabel(EMPTY_TEXT)
        self._empty.setObjectName("statsEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._stack = QStackedLayout()
        self._stack.addWidget(self._empty)
        self._stack.addWidget(self._scroll)
        self._stack.setCurrentWidget(self._empty)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._stack)

    # -- filling -----------------------------------------------------------------------------

    def show_result(self, result: ToolResult) -> None:
        """Show every block of ``result``, folded where the list is long."""
        self._result = result
        self._texts = []
        self._cards = []
        self._details_open = False
        self._details_body = None
        self._details_toggle = None
        self._set_body(self._build_body(result))
        self._stack.setCurrentWidget(self._scroll)

    def clear(self) -> None:
        """No result is shown: back to the one line that says what to do."""
        self._result = None
        self._texts = [EMPTY_TEXT]
        self._cards = []
        self._details_open = False
        self._details_body = None
        self._details_toggle = None
        self._set_body(None)
        self._stack.setCurrentWidget(self._empty)

    def _set_body(self, body: QWidget | None) -> None:
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        if body is not None:
            self._scroll.setWidget(body)

    def _build_body(self, result: ToolResult) -> QWidget:
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        self._texts.append(result.title)
        column.addWidget(_line("statsTitle", result.title))
        level = f"{result.level * 100:g}% confidence"
        self._texts.append(level)
        column.addWidget(_line("statsLevel", level))
        if result.estimates:
            self._add_estimates(column, result)
        if result.tests:
            self._add_tests(column, result)
        for check in result.assumptions:
            card = _build_card(check)
            self._cards.append(card)
            self._texts.extend(self._card_texts(card))
            column.addWidget(card.widget)
        self._add_warnings(column, result)
        for block in result.tables:
            self._add_table(column, block)
        self._add_interpretation(column, result)
        self._add_methods(column, result)
        self._add_details(column, result)
        column.addStretch(1)
        return body

    def _add_estimates(self, column: QVBoxLayout, result: ToolResult) -> None:
        self._texts.append("Estimates")
        column.addWidget(_heading("Estimates"))
        rows = [
            (
                estimate.label,
                _number(estimate.value),
                _number(estimate.se),
                _number(estimate.lower),
                _number(estimate.upper),
            )
            for estimate in result.estimates
        ]
        headers = ("Label", "Value", "SE", "Lower", "Upper")
        for row in rows:
            self._texts.extend(row)
        column.addWidget(_table(headers, rows, "statsEstimatesTable"))

    def _add_tests(self, column: QVBoxLayout, result: ToolResult) -> None:
        self._texts.append("Tests")
        column.addWidget(_heading("Tests"))
        headers = ("Test", "Statistic", "df", "p value", "Adjusted p")
        height = len(result.tests) + sum(1 for test in result.tests if test.reason)
        table = QTableWidget(height, len(headers))
        table.setObjectName("statsTestsTable")
        table.setHorizontalHeaderLabels(list(headers))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFont(mono_font(9))
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        row = 0
        for test in result.tests:
            if test.df is None:
                degrees = MISSING
            elif test.df_denominator is None:
                degrees = format_metric(test.df)
            else:
                degrees = f"{format_metric(test.df)}, {format_metric(test.df_denominator)}"
            adjusted = MISSING if test.adjusted_p is None else p_text(test.adjusted_p)
            cells = (
                test.name,
                _number(test.statistic),
                degrees,
                test.p_value_text,
                adjusted,
            )
            self._texts.extend(cells)
            for column_index, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column_index:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, column_index, item)
            row += 1
            if test.reason is not None:
                # the reason sits directly under the test's own row, across every column
                table.setSpan(row, 0, 1, len(headers))
                item = QTableWidgetItem(test.reason)
                item.setToolTip("why the test could not run")
                table.setItem(row, 0, item)
                self._texts.append(test.reason)
                row += 1
        column.addWidget(table)

    @staticmethod
    def _card_texts(card: _Card) -> list[str]:
        view = card.view
        texts = [view.title, view.evidence]
        for value in (view.statistic, view.threshold, view.explanation, view.next_step):
            if value:
                texts.append(value)
        texts.append(card.badge)
        return texts

    def _add_warnings(self, column: QVBoxLayout, result: ToolResult) -> None:
        self._texts.append("Warnings")
        column.addWidget(_heading("Warnings"))
        if not result.warnings:
            self._texts.append("None.")
            column.addWidget(_line("statsWarning", "None."))
            return
        for warning in result.warnings:
            self._texts.append(warning)
            column.addWidget(_line("statsWarning", warning))

    def _add_table(self, column: QVBoxLayout, block: Table) -> None:
        self._texts.append(block.title)
        column.addWidget(_heading(block.title))
        if block.note:
            self._texts.append(block.note)
            column.addWidget(_line("statsTableNote", block.note))
        headers = [part.label for part in block.columns]
        self._texts.extend(headers)
        rows = [
            [_cell_text(cell, part.kind) for cell, part in zip(row, block.columns, strict=True)]
            for row in block.rows
        ]
        for row in rows:
            self._texts.extend(row)
        column.addWidget(_table(headers, rows, "statsBlockTable"))

    def _add_interpretation(self, column: QVBoxLayout, result: ToolResult) -> None:
        self._texts.append("Interpretation")
        self._texts.append(result.interpretation)
        column.addWidget(_heading("Interpretation"))
        column.addWidget(_line("statsInterpretation", result.interpretation))

    def _add_methods(self, column: QVBoxLayout, result: ToolResult) -> None:
        self._texts.append("Methods")
        self._texts.append(result.methods_fragment)
        self._texts.append("Copy methods text")
        column.addWidget(_heading("Methods"))
        text = QPlainTextEdit(result.methods_fragment)
        text.setObjectName("statsMethodsText")
        text.setReadOnly(True)
        text.setFont(mono_font(9))
        text.setMaximumHeight(110)
        column.addWidget(text)
        row = QHBoxLayout()
        button = QPushButton("Copy methods text")
        button.setObjectName("statsCopyMethodsButton")
        button.clicked.connect(self.copy_methods_text)
        row.addStretch(1)
        row.addWidget(button)
        column.addLayout(row)

    def _add_details(self, column: QVBoxLayout, result: ToolResult) -> None:
        toggle = QToolButton()
        toggle.setObjectName("statsDetailsToggle")
        toggle.setText(DETAILS_LABEL)
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._texts.append(DETAILS_LABEL)
        body = QWidget()
        body.setObjectName("statsDetailsBody")
        details = QVBoxLayout(body)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(2)
        counts = _counts_text(result.counts)
        self._texts.append(counts)
        details.addWidget(_line("statsCounts", counts))
        for excluded in result.excluded:
            text = f"row {en_in_digits(excluded.index + 1)}: {excluded.reason}"
            self._texts.append(text)
            details.addWidget(_line("statsExcluded", text))
        if result.excluded_unlisted:
            more = f"and {en_in_digits(result.excluded_unlisted)} more"
            self._texts.append(more)
            details.addWidget(_line("statsExcludedMore", more))
        body.setVisible(False)
        self._details_body = body
        self._details_toggle = toggle
        toggle.toggled.connect(self._set_details_open)
        column.addWidget(toggle)
        column.addWidget(body)

    def _set_details_open(self, opened: bool) -> None:
        """The counts and the excluded rows follow the control that unfolds them."""
        self._details_open = opened
        if self._details_body is not None:
            self._details_body.setVisible(opened)
        if self._details_toggle is not None:
            self._details_toggle.setArrowType(
                Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow
            )

    # -- reading -----------------------------------------------------------------------------

    def result(self) -> ToolResult | None:
        """The result shown (``None`` before any tool has run)."""
        return self._result

    def texts(self) -> list[str]:
        """The words of every block, in reading order (the empty state's line when empty)."""
        return list(self._texts)

    def assumption_blocks(self) -> list[AuditBlockView]:
        """One block per assumption, in the tool's order."""
        return [card.view for card in self._cards]

    def details_open(self) -> bool:
        """Whether the counts and the excluded rows are unfolded."""
        return self._details_open

    def copy_methods_text(self) -> bool:
        """Put the shown result's methods fragment on the clipboard; ``False`` when empty."""
        if self._result is None:
            return False
        QGuiApplication.clipboard().setText(self._result.methods_fragment)
        return True


__all__ = ["EMPTY_TEXT", "MISSING", "StatsResultView"]
