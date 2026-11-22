"""A paste-aware spreadsheet grid — the clipboard-import dialog's editable table.

Copying a block of cells from Excel, LibreOffice Calc or Google Sheets and pressing
Ctrl+V here is the point: the whole block lands in the grid at the focused cell, rows
and columns growing to fit (a paste into an empty grid replaces it outright). Cells are
edited in place with the usual typing, Tab and Enter, cells and rows can be added or
removed, and the grid always serialises back to tab-separated text with proper quoting,
so the import pipeline parses exactly what is on screen.

Quoting matters: a spreadsheet cell may itself contain a comma, a tab or a newline, and
Excel escapes those by wrapping the field in double quotes. Both directions therefore go
through :mod:`csv` rather than ``split`` — a quoted value survives the round trip.
"""

import csv
from collections.abc import Iterable, Sequence
from io import StringIO

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

_DEFAULT_COLUMNS = 2
_DEFAULT_ROWS = 6


def _clipboard_delimiter(text: str) -> str:
    """Tab when the text carries tabs (every spreadsheet app's copy format), else comma.

    Semicolon-separated copies (continental Excel) are detected by the import sniffer
    downstream, but the grid has to split them itself: no tabs and no commas leaves the
    semicolon, which is also a legal one-column CSV delimiter.
    """
    if "\t" in text:
        return "\t"
    if ";" in text and "," not in text:
        return ";"
    return ","


def parse_table(text: str) -> list[list[str]]:
    """The pasted block as rows of cell text (quoted fields honoured, ragged rows padded)."""
    if not text.strip():
        return []
    delimiter = _clipboard_delimiter(text)
    reader = csv.reader(StringIO(text), delimiter=delimiter)
    rows = [list(row) for row in reader]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (width - len(row)))
    return rows


def write_table(rows: Sequence[Sequence[str]]) -> str:
    """Tab-separated text with minimal quoting — what the import pipeline reads back."""
    buffer = StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        writer.writerow(list(row))
    return buffer.getvalue()


class SpreadsheetGrid(QTableWidget):
    """An editable grid that accepts a whole pasted spreadsheet block."""

    changed = Signal()
    """The user edited a cell or pasted (programmatic builds stay silent)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(_DEFAULT_ROWS, _DEFAULT_COLUMNS, parent)
        self.setObjectName("spreadsheetGrid")
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.horizontalHeader().setDefaultSectionSize(110)
        self.verticalHeader().setDefaultSectionSize(24)
        self._set_headers()
        self.itemChanged.connect(self._on_item_changed)
        paste = QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self)
        paste.setContext(Qt.ShortcutContext.WidgetShortcut)
        paste.activated.connect(self.paste_from_clipboard)

    # -- geometry --------------------------------------------------------------------------

    def _set_headers(self) -> None:
        self.setHorizontalHeaderLabels(
            [_column_name(column) for column in range(self.columnCount())]
        )
        self.setVerticalHeaderLabels([str(row + 1) for row in range(self.rowCount())])

    def _ensure_size(self, rows: int, columns: int) -> None:
        grew = False
        if rows > self.rowCount():
            self.setRowCount(rows)
            grew = True
        if columns > self.columnCount():
            self.setColumnCount(columns)
            grew = True
        if grew:
            self._set_headers()

    def add_row(self) -> None:
        self._ensure_size(self.rowCount() + 1, self.columnCount())
        self.setCurrentCell(self.rowCount() - 1, max(0, self.currentColumn()))
        self.changed.emit()

    def add_column(self) -> None:
        self._ensure_size(self.rowCount(), self.columnCount() + 1)
        self.setCurrentCell(max(0, self.currentRow()), self.columnCount() - 1)
        self.changed.emit()

    def clear_grid(self) -> None:
        """Empty every cell, keeping the current shape (a blank canvas to type into)."""
        self.blockSignals(True)
        try:
            for row in range(self.rowCount()):
                for column in range(self.columnCount()):
                    self.setItem(row, column, QTableWidgetItem(""))
        finally:
            self.blockSignals(False)
        self.setCurrentCell(0, 0)
        self.changed.emit()

    # -- values ----------------------------------------------------------------------------

    def cell(self, row: int, column: int) -> str:
        item = self.item(row, column)
        return "" if item is None else item.text()

    def set_cell(self, row: int, column: int, text: str) -> None:
        self._ensure_size(row + 1, column + 1)
        item = self.item(row, column)
        if item is None:
            self.setItem(row, column, QTableWidgetItem(text))
        else:
            item.setText(text)

    def rows(self) -> list[list[str]]:
        """Every cell as text, trailing blank rows and columns trimmed."""
        rows = [
            [self.cell(row, column) for column in range(self.columnCount())]
            for row in range(self.rowCount())
        ]
        while len(rows) > 1 and not any(cell.strip() for cell in rows[-1]):
            rows.pop()
        if not rows:
            return [[""]]
        width = max((_last_filled(row) for row in rows), default=0)
        # pad to a rectangle: the delimited reader expects a consistent field count
        return [row[:width] + [""] * (width - len(row[:width])) for row in rows]

    def to_tsv(self) -> str:
        return write_table(self.rows())

    def is_blank(self) -> bool:
        return not any(cell.strip() for row in self.rows() for cell in row)

    # -- pasting ---------------------------------------------------------------------------

    def paste_from_clipboard(self) -> int:
        """Paste the clipboard's block at the current cell; returns the rows pasted."""
        clipboard = QApplication.clipboard()
        text = clipboard.text() if clipboard is not None else ""
        return self.paste_text(text)

    def paste_text(self, text: str, *, replace: bool = False) -> int:
        """Drop a pasted block into the grid at the current cell (or replace it all)."""
        pasted = parse_table(text)
        if not pasted:
            return 0
        if replace or self.is_blank():
            start_row = start_column = 0
            self.blockSignals(True)
            try:
                self.setRowCount(0)
                self.setColumnCount(0)
            finally:
                self.blockSignals(False)
        else:
            start_row = max(0, self.currentRow())
            start_column = max(0, self.currentColumn())
        width = max(len(row) for row in pasted)
        self._ensure_size(start_row + len(pasted), start_column + width)
        self.blockSignals(True)
        try:
            for offset, row in enumerate(pasted):
                for column, text in enumerate(row):
                    self.set_cell(start_row + offset, start_column + column, text)
        finally:
            self.blockSignals(False)
        self._set_headers()
        self.setCurrentCell(start_row, start_column)
        self.changed.emit()
        return len(pasted)

    # -- events ----------------------------------------------------------------------------

    def _on_item_changed(self, _item: QTableWidgetItem) -> None:
        self.changed.emit()


def _column_name(column: int) -> str:
    """Spreadsheet-style letters: A … Z, AA … — the header the user pastes against."""
    name = ""
    index = column
    while True:
        name = chr(ord("A") + index % 26) + name
        index = index // 26 - 1
        if index < 0:
            return name


def _last_filled(row: Iterable[str]) -> int:
    """The width of a row after trimming its trailing blank cells."""
    cells = list(row)
    while cells and not cells[-1].strip():
        cells.pop()
    return len(cells)


__all__ = ["SpreadsheetGrid", "parse_table", "write_table"]
