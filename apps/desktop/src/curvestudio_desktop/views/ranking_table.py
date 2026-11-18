"""The ranked Auto-fit table: a QAbstractTableModel over ``RankedFit`` rows.

Display rules that matter for honesty (Review Focus 3): undefined metrics render as
``n/a``, never 0 or a blank that reads as zero; markers in the Model cell point at
degenerate and redundant fits whose full explanation lives in the row tooltip; the
equation text is the row's fitted equation, with a copy action (context menu and Ctrl+C).
The table never re-sorts on its own — ordering is the ``RankBy`` selector's business, so
click-sorting cannot fight the ranked order.
"""

from collections.abc import Sequence

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    Qt,
    Signal,
)
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView, QWidget

from curvestudio.discovery import RankedFit

_NA = "n/a"
COLUMNS = (
    "Rank",
    "Model",
    "Category",
    "Equation (fitted)",
    "R²",
    "adj. R²",
    "RMSE",
    "AICc",
    "BIC",
    "CV-RMSE",
    "k",
    "Status",
)
_ROLE = Qt.ItemDataRole
_NO_PARENT = QModelIndex()


def format_metric(value: float | None, digits: int = 6) -> str:
    """``n/a`` for an undefined metric, else a compact significant-digits form."""
    return _NA if value is None else f"{value:.{digits}g}"


def _markers(row: RankedFit) -> str:
    marks = ""
    if row.degenerate:
        marks += " ⚠"
    if row.redundant_with is not None:
        marks += " ≡"
    if row.status == "exact_fit":
        marks += " ✓"
    return marks


def row_tooltip(row: RankedFit) -> str:
    """Everything the cell cannot show: ties, degeneracy, redundancy, overfitting, warnings."""
    lines = [f"{row.name} ({row.model_id}) — {row.status}"]
    if row.same_curve_as:
        lines.append("shares its curve with: " + ", ".join(row.same_curve_as))
    if row.degenerate:
        lines.append(
            "degenerate: the optimum collapsed onto a simpler model "
            "(a parameter at its bound with negligible effect, or a singular Jacobian)"
        )
    if row.redundant_with is not None:
        lines.append(f"redundant with {row.redundant_with} (same curve, more parameters)")
    if row.status == "exact_fit":
        lines.append("exact fit: information criteria are not defined for it")
    if row.overfitting_note is not None:
        lines.append(row.overfitting_note)
    lines.extend(row.warnings)
    return "\n".join(lines)


class RankingTableModel(QAbstractTableModel):
    """Immutable view of ranked rows; replace wholesale via :meth:`set_rows`."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: tuple[RankedFit, ...] = ()

    def set_rows(self, rows: Sequence[RankedFit]) -> None:
        self.beginResetModel()
        self._rows = tuple(rows)
        self.endResetModel()

    def rows(self) -> tuple[RankedFit, ...]:
        return self._rows

    def row_at(self, row: int) -> RankedFit | None:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def rowCount(  # Qt name
        self, parent: QModelIndex | QPersistentModelIndex = _NO_PARENT
    ) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(  # Qt name
        self, parent: QModelIndex | QPersistentModelIndex = _NO_PARENT
    ) -> int:
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = _ROLE.DisplayRole
    ) -> object:
        if role != _ROLE.DisplayRole:
            return None
        if orientation is Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return str(section + 1)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = _ROLE.DisplayRole,
    ) -> object:
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        column = index.column()
        if role == _ROLE.ToolTipRole:
            return row_tooltip(row)
        if role == _ROLE.TextAlignmentRole and column != 3:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role != _ROLE.DisplayRole:
            return None
        return self._text(row, column)

    def _text(self, row: RankedFit, column: int) -> str:
        metrics = row.metrics
        if column == 0:
            return str(row.rank)
        if column == 1:
            return row.name + _markers(row)
        if column == 2:
            return row.category
        if column == 3:
            return row.equation_fitted or _NA
        if column == 4:
            return format_metric(metrics.r2, 4)
        if column == 5:
            return format_metric(metrics.adj_r2, 4)
        if column == 6:
            return format_metric(metrics.rmse)
        if column == 7:
            return format_metric(metrics.aicc)
        if column == 8:
            return format_metric(metrics.bic)
        if column == 9:
            return format_metric(metrics.cv_rmse)
        if column == 10:
            return str(metrics.k)
        return row.status


class RankingTableView(QTableView):
    """Keyboard-operable, read-only ranking table with a copy-equation action."""

    openRequested = Signal()
    """The user asked for the full result of the current row (double-click or Enter)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("rankingTable")
        self.setModel(RankingTableModel(self))
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.doubleClicked.connect(lambda _index: self.openRequested.emit())

    def selected_row(self) -> RankedFit | None:
        indexes = self.selectedIndexes()
        if not indexes:
            return None
        model = self.model()
        assert isinstance(model, RankingTableModel)
        return model.row_at(indexes[0].row())

    def set_rows(self, rows: Sequence[RankedFit]) -> None:
        model = self.model()
        assert isinstance(model, RankingTableModel)
        model.set_rows(rows)

    def rows(self) -> tuple[RankedFit, ...]:
        model = self.model()
        assert isinstance(model, RankingTableModel)
        return model.rows()

    def select_row(self, row: int) -> RankedFit | None:
        """Select row ``row`` (0-based) and report what became selected.

        The current index is set explicitly: ``selectRow`` alone moves only the selection,
        so the currentRowChanged the view listens to would never fire.
        """
        model = self.model()
        assert isinstance(model, RankingTableModel)
        if not 0 <= row < len(model.rows()):
            return None
        self.setCurrentIndex(model.index(row, 0))
        self.selectRow(row)
        return model.row_at(row)

    def copy_equation(self) -> bool:
        """Copy the current row's fitted equation to the clipboard; False when there is none."""
        row = self.selected_row()
        if row is None or row.equation_fitted is None:
            return False
        QGuiApplication.clipboard().setText(row.equation_fitted)
        return True

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        wants_copy = key == Qt.Key.Key_C and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        if wants_copy and self.copy_equation():
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.openRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _show_menu(self, position: QPoint) -> None:
        # popup (not exec): a context menu is non-modal, and popup-at-position keeps it
        # exactly under the cursor
        row = self.selected_row()
        if row is None:
            return
        menu = QMenu(self)
        menu.addAction("Copy equation", self.copy_equation)
        menu.addAction("Copy row summary", self._copy_row_summary)
        menu.popup(self.viewport().mapToGlobal(position))

    def _copy_row_summary(self) -> None:
        model = self.model()
        assert isinstance(model, RankingTableModel)
        assert self.currentIndex().isValid()
        cells = [
            str(model.data(model.index(self.currentIndex().row(), column)) or "")
            for column in range(len(COLUMNS))
        ]
        QGuiApplication.clipboard().setText("\t".join(cells))
