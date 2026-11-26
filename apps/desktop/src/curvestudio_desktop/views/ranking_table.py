"""The ranked Auto-fit table: a QAbstractTableModel over ``RankedFit`` rows.

The dock shows the decisive columns only — rank, model, R², adj. R², RMSE and k. The
fitted equation, the information criteria, category and status live in the details
window (ranked rows are converged by definition, and exact/degenerate/redundant state is
already marked in the Model cell). Every numeric column is sized from its widest actual
value, so a value is never elided; only the model column stretches and elides. Showing
R² and adj. R² at full precision makes those columns wide, so a narrow dock scrolls
horizontally rather than rounding a value. Display rules that
matter for honesty (Review Focus 3): undefined metrics render as ``n/a``, never 0 or a
blank that reads as zero; markers in the Model cell point at degenerate and redundant
fits whose full explanation lives in the row tooltip; the row tooltip and the context
menu still carry the fitted equation. The ranking itself is the ``RankBy`` selector's
business and the model's rows never change order — but a header click is a *display*
ordering: a sort proxy reorders what the view shows (ascending → descending → back to
the ranked order), the rank column stays visible, and an undefined metric reads as
missing and sorts last in both directions.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QGuiApplication, QKeyEvent
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView, QWidget

if TYPE_CHECKING:
    from curvestudio.discovery import RankedFit


_NA = "n/a"
COLUMNS = (
    "Rank",
    "Model",
    "R²",
    "adj. R²",
    "RMSE",
    "k",
)
_MODEL_COLUMN = 1
_CENTERED = frozenset({0})
_RIGHT_ALIGNED = frozenset({2, 3, 4, 5})
#: Horizontal padding the stylesheet consumes inside a ranking cell, from
#: ``views/styles.py``: ``QTableView#rankingTable::item { padding: 3px 4px; }`` (4 px
#: left + 4 px right). It is part of the column width, so it must be added explicitly.
ITEM_HORIZONTAL_PADDING = 8
#: Horizontal padding the stylesheet consumes inside a ranking header section, from
#: ``views/styles.py``: ``QHeaderView#rankingTableHeader::section { padding: 5px 4px; }``
#: (4 px left + 4 px right). The same section width carries both the header title and
#: the cells, so this padding is added as well.
HEADER_HORIZONTAL_PADDING = 8
#: Slack beyond both paddings: the eliding delegate reserves its own focus-frame margin
#: (about 2 px a side) inside the cell before laying out the text, so the text rect is
#: narrower than ``section - padding``. This covers that reservation and width rounding.
DELEGATE_HORIZONTAL_SLACK = 8
#: Total margin added to a numeric column's widest measured text. Composed from the QSS
#: padding rules in ``views/styles.py`` and the delegate's own reservation above, so the
#: three cannot drift apart behind a magic number.
COLUMN_PADDING = ITEM_HORIZONTAL_PADDING + HEADER_HORIZONTAL_PADDING + DELEGATE_HORIZONTAL_SLACK
_ROLE = Qt.ItemDataRole
_NO_PARENT = QModelIndex()


def format_metric(value: float | None, digits: int = 6) -> str:
    """``n/a`` for an undefined metric, else a compact significant-digits form."""
    return _NA if value is None else f"{value:.{digits}g}"


def format_precise(value: float | None) -> str:
    """``n/a`` for an undefined metric, else the shortest string that round-trips.

    ``repr`` is Python's shortest exact representation, so a near-perfect fit can
    never display as exactly 1: 1.0 -> "1.0", 0.9999999999999999 -> that value.
    """
    return _NA if value is None else repr(value)


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


def _sort_value(row: RankedFit, column: int) -> float | str | None:
    """What a column is compared by; ``None`` is the ``n/a`` case."""
    if column == 0:
        return row.rank
    if column == 1:
        return row.name.casefold()
    metrics = row.metrics
    if column == 2:
        return metrics.r2
    if column == 3:
        return metrics.adj_r2
    if column == 4:
        return metrics.rmse
    return metrics.k


class _DisplaySortProxyModel(QSortFilterProxyModel):
    """Reorders the *view* only: the source model (the stored ranking) never changes."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._descending = False

    def set_descending(self, descending: bool) -> None:
        self._descending = descending

    def lessThan(  # Qt name
        self, left: QModelIndex, right: QModelIndex
    ) -> bool:
        source = self.sourceModel()
        assert isinstance(source, RankingTableModel)
        left_row = source.row_at(left.row())
        right_row = source.row_at(right.row())
        if left_row is None or right_row is None:
            return False
        a = _sort_value(left_row, left.column())
        b = _sort_value(right_row, right.column())
        # an undefined metric is missing information, not a number: it sorts last
        # whichever way the column is ordered
        if a is None and b is None:
            return left.row() < right.row()
        if a is None:
            return False
        if b is None:
            return True
        if self._descending:
            return b < a
        return a < b


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
        if role == _ROLE.DecorationRole and column == 0:
            # The row's own colour swatch: the same colour its curve is drawn in, matched
            # by the 1-based display position (row index), not the tied rank value, so
            # tied rows are still distinguishable. A decoration adds no column.
            from curvestudio_plotting.palette import model_color

            return QColor(model_color(index.row() + 1))
        if role == _ROLE.TextAlignmentRole:
            if column in _CENTERED:
                return int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
            if column in _RIGHT_ALIGNED:
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
            # the full stored value: the shortest round-tripping string, so a near-perfect
            # fit (0.9999999999999999) can never read as the same "1.0" as an exact one
            return format_precise(metrics.r2)
        if column == 3:
            return format_precise(metrics.adj_r2)
        if column == 4:
            # four significant digits keep the comparison readable; the details
            # window carries the full precision
            return format_metric(metrics.rmse, 4)
        return str(metrics.k)


class RankingTableView(QTableView):
    """Keyboard-operable, read-only ranking table with a copy-equation action.

    A header click cycles a display-only ordering (ascending → descending → ranked);
    ``set_rows``/``rows``/``select_row`` keep speaking in *stored* row numbers, and the
    mapping to and from the display order happens here.
    """

    openRequested = Signal()
    """The user asked for the full result of the current row (double-click or Enter)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("rankingTable")
        self._source_model = RankingTableModel(self)
        self._sort_proxy = _DisplaySortProxyModel(self)
        self._sort_proxy.setSourceModel(self._source_model)
        self.setModel(self._sort_proxy)
        self._sort_column: int | None = None
        self._sort_descending = False
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        # the Rank column already identifies rows: the vertical header only steals width
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setObjectName("rankingTableHeader")
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(24)
        # sections are clickable, but the sorting is ours (below): the view's built-in
        # sorting stays off so it cannot fight the ranked order either
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)  # no column sorted yet
        header.sectionClicked.connect(self._cycle_column_sort)
        for column in range(len(COLUMNS)):
            if column != _MODEL_COLUMN:
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        # the model name takes the leftover width and elides; nothing else moves
        header.setSectionResizeMode(_MODEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        self._fit_numeric_columns()
        self._source_model.modelReset.connect(self._fit_numeric_columns)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.doubleClicked.connect(lambda _index: self.openRequested.emit())

    # -- display ordering --------------------------------------------------------------------

    def _cycle_column_sort(self, column: int) -> None:
        """The three-state header cycle: ascending → descending → the ranked order."""
        header = self.horizontalHeader()
        if self._sort_column != column:
            self._apply_column_sort(column, descending=False)
        elif not self._sort_descending:
            self._apply_column_sort(column, descending=True)
        else:
            self._sort_column = None
            self._sort_descending = False
            self._sort_proxy.sort(-1)  # -1: the proxy passes the source order through
            header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

    def _apply_column_sort(self, column: int, *, descending: bool) -> None:
        self._sort_column = column
        self._sort_descending = descending
        self._sort(column)
        self.horizontalHeader().setSortIndicator(
            column,
            Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder,
        )

    def _reapply_sort(self) -> None:
        if self._sort_column is not None:
            self._sort(self._sort_column)

    def _sort(self, column: int) -> None:
        """(Re)apply the display sort. The proxy always "ascends" — the comparator itself
        turns around, so an undefined metric can stay last in the descending order too —
        and the sort is cleared first because Qt skips a sort that asks for the very
        column and order it already holds."""
        self._sort_proxy.set_descending(self._sort_descending)
        self._sort_proxy.sort(-1)
        self._sort_proxy.sort(column, Qt.SortOrder.AscendingOrder)

    def _source_row_of(self, index: QModelIndex) -> int:
        """The stored row a (possibly proxied) view index points at."""
        if index.model() is self._sort_proxy:
            return self._sort_proxy.mapToSource(index).row()
        return index.row()

    def _proxy_index_of(self, source_row: int) -> QModelIndex:
        return self._sort_proxy.mapFromSource(self._source_model.index(source_row, 0))

    def _fit_numeric_columns(self) -> None:
        """Size every non-model column to its widest rendered value plus padding.

        A value like ``0.9999`` or ``1234`` must never elide, so the width comes from
        the actual font metrics rather than a guessed table of fixed widths. The model
        column keeps stretching into whatever is left.
        """
        model = self._source_model
        header = self.horizontalHeader()
        values = self.fontMetrics()
        headers = header.fontMetrics()
        for column in range(model.columnCount()):
            if column == _MODEL_COLUMN:
                continue
            title = str(model.headerData(column, Qt.Orientation.Horizontal))
            widest = headers.horizontalAdvance(title)
            for row in range(model.rowCount()):
                text = model.data(model.index(row, column))
                if text is not None:
                    widest = max(widest, values.horizontalAdvance(str(text)))
            header.resizeSection(column, widest + COLUMN_PADDING)

    def selected_row(self) -> RankedFit | None:
        indexes = self.selectedIndexes()
        if not indexes:
            return None
        return self._source_model.row_at(self._source_row_of(indexes[0]))

    def set_rows(self, rows: Sequence[RankedFit]) -> None:
        self._source_model.set_rows(rows)
        self._reapply_sort()

    def rows(self) -> tuple[RankedFit, ...]:
        return self._source_model.rows()

    def select_row(self, row: int) -> RankedFit | None:
        """Select stored row ``row`` (0-based) and report what became selected.

        The current index is set explicitly: ``selectRow`` alone moves only the selection,
        so the currentRowChanged the view listens to would never fire.
        """
        if not 0 <= row < len(self._source_model.rows()):
            return None
        proxy_index = self._proxy_index_of(row)
        self.setCurrentIndex(proxy_index)
        self.selectRow(proxy_index.row())
        return self._source_model.row_at(row)

    def copy_equation(self) -> bool:
        """Copy the current row's fitted equation as shown (``^`` powers); False if none."""
        from curvestudio.workflow import format_equation

        row = self.selected_row()
        text = "" if row is None else format_equation(row.equation_fitted)
        if not text:
            return False
        QGuiApplication.clipboard().setText(text)
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
        assert self.currentIndex().isValid()
        source_row = self._source_row_of(self.currentIndex())
        cells = [
            str(self._source_model.data(self._source_model.index(source_row, column)) or "")
            for column in range(len(COLUMNS))
        ]
        QGuiApplication.clipboard().setText("\t".join(cells))
