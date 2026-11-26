"""The result tables' caches (D5): measuring and formatting happen once, not per paint.

Splitter drags beside the live diagnostics plots used to re-measure every parameters cell
(``horizontalAdvance`` per cell per resize), and the data table model formatted every cell
(``float`` + ``%.10g``) and built a fresh highlight brush per repaint. The caches must not
change what is shown: the same ``%.10g`` strings, the same highlight colours, the same
distributed widths.
"""

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFontMetrics
from PySide6.QtWidgets import QTableWidgetItem

from curvestudio.data import Dataset
from curvestudio.data.operations import ExcludeRows
from curvestudio.workflow import ImportIssue
from curvestudio_desktop.views.data_table import (
    DatasetTableModel,
    DataTableDock,
    _format_number,
)
from curvestudio_desktop.views.results_tabs import (
    _COLUMN_PADDING,
    _measure_column_widths,
    _ProportionalTable,
)

_HIGHLIGHT_RGB = QColor(255, 90, 90, 45)  # written out here, not imported: the test pins
# the exact colour the data table is supposed to highlight with
_ROLE = Qt.ItemDataRole


def _count_horizontal_advances(monkeypatch) -> dict[str, int]:
    """Route ``QFontMetrics.horizontalAdvance`` through a counter.

    Only Python-level calls hit the patch — Qt's C++ internals never do — so the count is
    exactly the measuring work the desktop code performs.
    """
    calls = {"measured": 0}
    original = QFontMetrics.horizontalAdvance

    def counting(self, text: str) -> int:
        calls["measured"] += 1
        return original(self, text)

    monkeypatch.setattr(QFontMetrics, "horizontalAdvance", counting)
    return calls


def _filled_proportional_table() -> _ProportionalTable:
    """A parameters-table-shaped grid: four columns, headings and three rows of cells."""
    table = _ProportionalTable(3, 4)
    table.setObjectName("parametersTable")
    table.setHorizontalHeaderLabels(["parameter", "value", "95% CI low", "95% CI high"])
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            table.setItem(row, column, QTableWidgetItem(f"cell {row}/{column}"))
    return table


def _directly_measured_minimums(table: _ProportionalTable) -> list[int]:
    """What the pre-cache implementation computed: walk every cell, no cache involved."""
    cell = table.fontMetrics()
    title = table.horizontalHeader().fontMetrics()
    minimums = []
    for column in range(table.columnCount()):
        widest = title.horizontalAdvance(table.horizontalHeaderItem(column).text())
        for row in range(table.rowCount()):
            item = table.item(row, column)
            if item is not None:
                widest = max(widest, cell.horizontalAdvance(item.text()))
        minimums.append(widest + _COLUMN_PADDING)
    return minimums


# -- the measuring cache ----------------------------------------------------------------------


def test_a_resize_after_layout_does_not_remeasure(qtbot, monkeypatch) -> None:
    """After one layout pass a pure viewport resize measures nothing; content does."""
    table = _filled_proportional_table()
    qtbot.addWidget(table)
    table.resize(600, 200)
    table.show()
    table.resize(650, 220)  # a real pass with a real viewport: the widths are measured
    assert table._content_widths._minimums is not None  # precondition: cache is populated

    calls = _count_horizontal_advances(monkeypatch)
    table.resize(950, 320)  # pure viewport resize, content unchanged
    assert calls["measured"] == 0
    header = table.horizontalHeader()
    assert sum(header.sectionSize(c) for c in range(header.count())) == table.viewport().width()

    table.setItem(1, 2, QTableWidgetItem("a much wider value"))  # the content changed…
    table.resize(980, 320)  # …so the next pass measures again
    assert calls["measured"] > 0


def test_the_cached_widths_equal_a_direct_measurement(qtbot) -> None:
    """The cache holds exactly what the measure-everything implementation computed."""
    table = _filled_proportional_table()
    qtbot.addWidget(table)
    table.resize(700, 220)
    table.show()
    table.resize(760, 240)
    expected = _directly_measured_minimums(table)
    assert table._content_widths.minimums() == expected

    viewport = table.viewport().width()
    content = sum(expected)
    if content < viewport:  # the same distribution the pre-cache implementation ran
        slack = viewport - content
        widths = [width + slack * width // content for width in expected]
        widths[-1] += viewport - sum(widths)
    else:
        widths = expected
    assert [
        table.horizontalHeader().sectionSize(c) for c in range(table.columnCount())
    ] == widths

    table.resize(200, 240)  # narrower than the content: every column keeps its minimum
    assert [
        table.horizontalHeader().sectionSize(c) for c in range(table.columnCount())
    ] == expected


def test_the_measure_helper_measures_every_column(qtbot) -> None:
    """The refactored helper measures what it always did: headings and every cell."""
    table = _filled_proportional_table()
    qtbot.addWidget(table)
    table.setItem(1, 1, QTableWidgetItem("0.000123456789"))
    assert _measure_column_widths(table) == _directly_measured_minimums(table)


def test_an_empty_table_measures_headings_only(qtbot) -> None:
    """Zero rows is a content state too: the headings alone set the minimums."""
    table = _ProportionalTable(0, 3)
    qtbot.addWidget(table)
    table.setHorizontalHeaderLabels(["a", "b", "c"])
    table.show()
    table.resize(500, 200)
    assert table._content_widths.minimums() == _directly_measured_minimums(table)
    assert table._content_widths._minimums is not None


def test_row_count_changes_invalidate_the_cache(qtbot) -> None:
    """Growing and clearing the table go through the row signals, and must re-measure."""
    table = _filled_proportional_table()
    qtbot.addWidget(table)
    table.resize(600, 200)
    table.show()
    table.resize(650, 220)
    assert table._content_widths._minimums is not None

    table.setRowCount(0)  # rowsRemoved: the cache can no longer describe the content
    assert table._content_widths._minimums is None

    table.setRowCount(1)
    table.setItem(0, 0, QTableWidgetItem("k1"))
    table.resize(660, 220)
    assert table._content_widths._minimums is not None  # re-measured for the new content


# -- the data table's format cache ------------------------------------------------------------


def _dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset("cache-test", frame)


def test_the_cached_formatting_matches_the_direct_implementation() -> None:
    """Every rendered string equals ``_format_number`` of the raw value, per cell."""
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.5, -2.5, 1e-13, 123456789.25, 3.0],
            "y": [10.0, float("nan"), -0.1, 12345678901234.0, 0.5, 2.0],
            "w": [1.0, 2.0, "abc", float("inf"), -float("inf"), 0.25],
        }
    )
    issues = (
        ImportIssue(
            code="MISSING_VALUE",
            severity="error",
            message="Blank or NaN values in 'y'",
            rows=(1, 3),
        ),
    )
    model = DatasetTableModel(_dataset(frame), "x", "y", "w", issues)
    assert (model.rowCount(), model.columnCount()) == (6, 4)
    highlighted = {1, 3}
    labels = [f"{issue.code}: {issue.message}" for issue in issues]
    for row in range(model.rowCount()):
        for column in range(3):  # x, y, weight
            index = model.index(row, column)
            expected_text = _format_number(frame.iat[row, column])
            assert model.data(index) == expected_text
            expected_brush = QBrush(_HIGHLIGHT_RGB) if row in highlighted else None
            assert model.data(index, _ROLE.BackgroundRole) == expected_brush
            expected_tooltip = "\n".join(labels) if row in highlighted else None
            assert model.data(index, _ROLE.ToolTipRole) == expected_tooltip
        assert model.data(model.index(row, 3)) == "✓"  # no rows excluded: all included


def test_the_cached_formatting_matches_when_the_weight_column_is_missing() -> None:
    """A dash and — as before — no highlight in the weight column when none is chosen."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    issues = (
        ImportIssue(code="MISSING_VALUE", severity="error", message="Blanks", rows=(1,)),
    )
    model = DatasetTableModel(_dataset(frame), "x", "y", None, issues)
    assert model.data(model.index(1, 2)) == "—"
    assert model.data(model.index(1, 2), _ROLE.BackgroundRole) is None  # the dash column
    assert model.data(model.index(1, 0), _ROLE.BackgroundRole) == QBrush(_HIGHLIGHT_RGB)
    assert model.data(model.index(1, 2), _ROLE.ToolTipRole) is None  # as it always was


def test_the_included_column_flags_excluded_rows_with_the_shared_brush() -> None:
    dataset = _dataset(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})).with_op(
        ExcludeRows((0,))
    )
    model = DatasetTableModel(dataset, "x", "y", None)
    assert model.data(model.index(0, 3)) == "—"
    assert model.data(model.index(0, 3), _ROLE.TextAlignmentRole) == int(
        Qt.AlignmentFlag.AlignCenter
    )
    brush = model.data(model.index(0, 3), _ROLE.BackgroundRole)
    assert isinstance(brush, QBrush)
    assert brush.color() == _HIGHLIGHT_RGB
    assert model.data(model.index(1, 3)) == "✓"
    assert model.data(model.index(1, 3), _ROLE.BackgroundRole) is None


def test_replacing_the_dataset_renders_the_new_values(qtbot) -> None:
    """The app replaces a dataset by setting a fresh model: its format cache is empty."""
    dock = DataTableDock()
    qtbot.addWidget(dock)
    first = DatasetTableModel(
        _dataset(pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})), "x", "y", None
    )
    dock.set_model(first)
    assert dock.view.model() is first
    assert first.data(first.index(0, 0)) == "1"

    second = DatasetTableModel(
        _dataset(pd.DataFrame({"x": [5.5, 6.0], "y": [1.0, 2.0]})), "x", "y", None
    )
    dock.set_model(second)
    assert dock.view.model() is second
    assert second.data(second.index(0, 0)) == "5.5"  # the first dataset's "1" is not served


def test_a_data_reset_invalidates_the_format_cache() -> None:
    """The model's data-reset path drops the cache: a changed value renders changed."""
    model = DatasetTableModel(
        _dataset(pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})), "x", "y", None
    )
    index = model.index(0, 0)
    assert model.data(index) == "1"
    assert model._display_cache  # precondition: the cell is cached after one read

    model._frame = pd.DataFrame({"x": [7.25, 8.0], "y": [1.0, 2.0]})  # the data changed
    model.beginResetModel()
    model.endResetModel()  # what a real data reset calls: resetInternalData clears
    assert not model._display_cache
    assert model.data(index) == "7.25"  # re-formatted, not the stale "1"
