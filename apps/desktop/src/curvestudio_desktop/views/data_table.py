"""DataTableView: a virtualised, read-only table over the active dataset.

A QAbstractTableModel feeds a QTableView — no per-cell widgets, so a million rows
cost one materialised frame and nothing per row. Cells in rows flagged by import
issues are highlighted (a translucent red overlay, readable on light and dark
themes) with the issue text as tooltip. Editing and cleaning arrive in a later
release; this release shows the table read-only.
"""

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QDockWidget, QHeaderView, QTableView, QVBoxLayout, QWidget

from curvestudio_desktop.theme import mono_font

if TYPE_CHECKING:
    from curvestudio.data import Dataset
    from curvestudio.workflow import ImportIssue

_INCLUDED = "Included"
_WEIGHT_HEADER = "Weight"
_NO_WEIGHT = "—"
_HIGHLIGHT = QColor(255, 90, 90, 45)
_ROLE = Qt.ItemDataRole
_NO_PARENT = QModelIndex()  # the immutable empty index Qt passes for top-level queries


def _format_number(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return str(value)
    if math.isnan(number):
        return ""
    if math.isinf(number):
        return "∞" if number > 0 else "-∞"
    return f"{number:.10g}"


def _row_issue_map(issues: tuple[ImportIssue, ...]) -> dict[int, str]:
    """Map 0-based table rows to the combined issue text shown as tooltip."""
    combined: dict[int, list[str]] = {}
    for issue in issues:
        label = f"{issue.code}: {issue.message}"
        for row in issue.rows:
            combined.setdefault(row, []).append(label)
    return {row: "\n".join(labels) for row, labels in combined.items()}


class DatasetTableModel(QAbstractTableModel):
    """Columns: x, y, weight (a dash when none), included flag."""

    def __init__(
        self,
        dataset: Dataset,
        x_column: str,
        y_column: str,
        weight_column: str | None,
        issues: tuple[ImportIssue, ...] = (),
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        import numpy as np

        self._frame = dataset.table()
        self._active = np.asarray(dataset.active_mask())
        self._x_column = x_column
        self._y_column = y_column
        self._weight_column = weight_column
        self._row_issues = _row_issue_map(issues)

    def rowCount(  # Qt name
        self, parent: QModelIndex | QPersistentModelIndex = _NO_PARENT
    ) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._frame)

    def columnCount(  # Qt name
        self, parent: QModelIndex | QPersistentModelIndex = _NO_PARENT
    ) -> int:
        return 0 if parent is not None and parent.isValid() else 4

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = _ROLE.DisplayRole
    ) -> object:
        if role != _ROLE.DisplayRole:
            return None
        if orientation is Qt.Orientation.Horizontal:
            names = (
                self._x_column,
                self._y_column,
                self._weight_column or _WEIGHT_HEADER,
                _INCLUDED,
            )
            return names[section]
        return str(section + 1)  # people count table rows from one

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = _ROLE.DisplayRole,
    ) -> object:
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if column == 3:  # the included flag
            if role == _ROLE.DisplayRole:
                return "✓" if self._active[row] else "—"
            if role == _ROLE.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignCenter)
            if role == _ROLE.BackgroundRole and not self._active[row]:
                return QBrush(_HIGHLIGHT)
            return None
        name = (self._x_column, self._y_column, self._weight_column)[column]
        if name is None:  # no weight column chosen
            return _NO_WEIGHT if role == _ROLE.DisplayRole else None
        value = self._frame.iat[row, self._frame.columns.get_loc(name)]
        if role == _ROLE.DisplayRole:
            return _format_number(value)
        if role == _ROLE.ToolTipRole and row in self._row_issues:
            return self._row_issues[row]
        if role == _ROLE.BackgroundRole and row in self._row_issues:
            return QBrush(_HIGHLIGHT)
        return None

    def tooltip_for_row(self, row: int) -> str:
        return self._row_issues.get(row, "")


class DataTableDock(QDockWidget):
    """The bottom dock: the data table of the active dataset."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Data table", parent)
        self.setObjectName("tableDock")
        holder = QWidget(self)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(4, 4, 4, 4)
        self.view = QTableView()
        self.view.setObjectName("dataTable")
        self.view.setFont(mono_font(9))
        self.view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setDefaultSectionSize(24)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.view.horizontalHeader().setDefaultSectionSize(120)
        layout.addWidget(self.view)
        self.setWidget(holder)

    def set_model(self, model: DatasetTableModel | None) -> None:
        self.view.setModel(model)
