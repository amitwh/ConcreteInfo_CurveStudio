"""The analysis dock's layout: the ranking table is the primary result.

The table dominates the dock (a vertical splitter gives the workflow page — the table —
at least 60% by default, with a minimum height of about five rows), a compact fit summary
sits below it (the five full result views live in the details window), the disclosure
sections start collapsed, banners wrap, the mode switch is a compact segmented control,
and the splitter split is remembered in QSettings.
"""

import pytest
from desktop_data import GOOD_CSV
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication
from test_desktop_autofit_run import NOISY_QUADRATIC, TINY, wait_stored

from curvestudio.discovery import CandidateMetrics, RankedFit
from curvestudio.workflow import format_equation
from curvestudio_desktop.views.analysis_panel import AnalysisPanel
from curvestudio_desktop.views.fit_summary import FitSummaryPanel
from curvestudio_desktop.views.ranking_table import (
    COLUMN_PADDING,
    COLUMNS,
    RankingTableView,
)
from curvestudio_desktop.views.results_tabs import ResultsTabs


@pytest.fixture()
def autofit_window(make_window, qtbot, write_file):
    """A window with a finished tiny Auto-fit run, shown and sized like a real screen."""
    window = make_window()
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    dataset_id = window.controller.project.datasets[0].id
    wait_stored(qtbot, window, dataset_id)
    window.resize(1400, 850)
    QApplication.processEvents()
    yield window


def test_ranking_table_gets_at_least_60_percent_of_the_dock(autofit_window) -> None:
    panel = autofit_window.analysis_panel
    sizes = panel.splitter.sizes()
    assert len(sizes) == 2 and sum(sizes) > 0
    assert sizes[0] >= 0.6 * sum(sizes), f"table page got {sizes} of {sum(sizes)}"
    # and the table itself can show a meaningful ranking without scrolling
    view = panel.autofit_view
    row = view.table.verticalHeader().defaultSectionSize()
    assert view.table.height() >= 8 * row, (
        f"table height {view.table.height()} under eight rows ({row}px each)"
    )


def test_summary_panel_sits_below_the_table_page(autofit_window) -> None:
    panel: AnalysisPanel = autofit_window.analysis_panel
    assert isinstance(panel.summary, FitSummaryPanel)
    # the five result views moved to the details window: the dock hosts the summary only
    assert panel.findChild(ResultsTabs) is None
    table_bottom = panel.autofit_view.mapTo(panel, panel.autofit_view.rect().bottomLeft()).y()
    summary_top = panel.summary.mapTo(panel, panel.summary.rect().topLeft()).y()
    assert summary_top >= table_bottom  # the summary never rides above the ranking


def test_ranking_drops_the_equation_and_never_needs_a_horizontal_scroll(
    autofit_window,
) -> None:
    view = autofit_window.analysis_panel.autofit_view
    assert COLUMNS == (
        "Rank",
        "Model",
        "R²",
        "adj. R²",
        "RMSE",
        "k",
    )
    assert "Equation" not in COLUMNS
    assert "Status" not in COLUMNS and "Category" not in COLUMNS
    model = view.table.model()
    assert model is not None
    assert [
        model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())
    ] == list(COLUMNS)
    QApplication.processEvents()
    # at the default dock width the fixed numerics leave the model column the rest:
    # no cell is pushed out of view, so no horizontal scrollbar is needed
    assert view.table.horizontalScrollBar().maximum() == 0
    assert view.table.verticalHeader().isHidden()  # rank is a column, not a row number


def test_summary_mirrors_the_selected_row(autofit_window) -> None:
    view = autofit_window.analysis_panel.autofit_view
    summary = autofit_window.analysis_panel.summary
    assert view.table.rows()
    assert view.table.select_row(1) is not None
    QApplication.processEvents()
    selected = view.table.selected_row()
    assert selected is not None
    assert summary.model_text() == selected.name
    assert summary.status_text() == selected.status
    assert summary.metric_text("k") == str(selected.metrics.k)
    assert not summary.hint_visible()


@pytest.mark.parametrize("size", [(1600, 920), (1280, 800)])
def test_summary_keeps_its_minimum_height_and_shows_its_content(
    make_window, qtbot, write_file, size
) -> None:
    """The summary is never a sliver: its own minimum wins over the workflow page."""
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    dataset_id = window.controller.project.datasets[0].id
    wait_stored(qtbot, window, dataset_id)
    window.resize(*size)
    QApplication.processEvents()

    view = window.analysis_panel.autofit_view
    assert view.table.rows()
    assert view.table.select_row(0) is not None
    QApplication.processEvents()

    summary = window.analysis_panel.summary
    assert summary.minimumHeight() > 0  # its own floor, not the splitter's default
    assert summary.height() >= summary.minimumHeight(), (
        summary.height(),
        summary.minimumHeight(),
    )
    assert summary.isVisible()
    assert summary.model_label.isVisible()
    for title in ("R²", "adj. R²", "RMSE", "k"):
        assert summary.metric_text(title)
        assert summary._metric_values[title].isVisible(), title
    assert summary.open_button.isVisible()
    assert summary.export_button.isVisible()
    assert "details" in summary.open_button.text().lower()
    # the ranking keeps its own space at both target sizes
    assert view.table.isVisible()
    assert view.table.height() >= view.table.minimumHeight()


@pytest.mark.parametrize("size", [(1600, 920), (1280, 800)])
def test_analysis_dock_never_overflows_into_a_neighbouring_dock(
    make_window, qtbot, write_file, size
) -> None:
    """The Auto-fit page scrolls its disclosures instead of pushing the splitter down.

    A tall disclosure (open Options, expanded sections) must never raise the page's
    minimum past its own rect: the splitter would overflow the dock, paint over the
    Data-table dock's title bar and bury the fit summary's own buttons.
    """
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    dataset_id = window.controller.project.datasets[0].id
    result = wait_stored(qtbot, window, dataset_id)
    window.resize(*size)
    QApplication.processEvents()

    panel = window.analysis_panel
    view = panel.autofit_view
    summary = panel.summary
    assert result is not None and result.ranked
    assert view.table.select_row(0) is not None
    QApplication.processEvents()

    # the rules for the two summary actions: open is always available; export waits
    # for a full result (the details window's bundle)
    assert summary.open_button.isEnabled()
    assert not summary.export_button.isEnabled()
    from curvestudio.workflow import bundle_for_candidate

    xy = window.selection.xy()
    assert xy is not None
    top = result.ranked[0]
    summary.show_bundle(bundle_for_candidate(xy, result, top.model_id), dataset_name="quad")
    QApplication.processEvents()
    assert summary.export_button.isEnabled()

    # stress the disclosure exactly as a user can: every section and the Options form open
    view.options_button.setChecked(True)
    for section in (view.exact_section, view.unconverged_section, view.skipped_section):
        section._button.setChecked(True)
    QApplication.processEvents()

    # the page's content fits its own rect and the splitter's minimum fits the dock
    assert view.minimumSizeHint().height() <= view.height(), (
        "the Auto-fit page's content overflows its rect and paints over the summary"
    )
    assert panel.splitter.minimumSizeHint().height() <= panel.height(), (
        f"splitter minimum {panel.splitter.minimumSizeHint().height()} exceeds dock "
        f"{panel.height()}"
    )
    assert panel.splitter.height() <= panel.height()
    # the summary itself stays inside the dock (mapped into the dock's coordinates)
    summary_bottom = panel.mapTo(panel, summary.rect().bottomLeft()).y()
    assert summary.geometry().bottom() <= panel.height()
    assert summary_bottom <= panel.height()
    # and neither action is buried under the page's content
    for button in (summary.open_button, summary.export_button):
        assert button.isEnabled()
        centre = panel.mapFromGlobal(button.mapToGlobal(button.rect().center()))
        assert panel.rect().contains(centre), f"{button.objectName()} is outside the dock"
        assert panel.childAt(centre) is button, (
            f"{button.objectName()} is covered by the Auto-fit page"
        )
    # the ranking keeps its own space and stays usable while the disclosure is open
    assert view.table.height() >= view.table.minimumHeight()
    assert view.table.select_row(1) is not None

    # collapsing the disclosure again leaves the table just as usable
    view.options_button.setChecked(False)
    for section in (view.exact_section, view.unconverged_section, view.skipped_section):
        section._button.setChecked(False)
    QApplication.processEvents()
    assert view.minimumSizeHint().height() <= view.height()
    assert view.table.height() >= view.table.minimumHeight()
    assert view.table.select_row(0) is not None


@pytest.mark.parametrize("width", (380, 420, 570, 700))
def test_ranking_numeric_columns_never_clip_their_values(qtbot, width) -> None:
    """Worst-case numbers fit their columns at every dock width from 380 px to 700 px.

    The width of a fixed column is derived from the widest rendered text plus the QSS
    padding of both the item and the header section (``views/styles.py``:
    ``QTableView#rankingTable::item { padding: 3px 4px; }`` and
    ``QHeaderView#rankingTableHeader::section { padding: 5px 4px; }`` — 8 px each).
    """
    rows = tuple(
        RankedFit(
            rank=rank,
            model_id=f"model_{rank}",
            name="A very long generated model name that must elide in the dock",
            category="polynomial",
            metrics=CandidateMetrics(
                k=12,
                n=20,
                r2=r2,
                adj_r2=adj_r2,
                # the second value formats as "0.04692": the long string whose RMSE
                # cell used to elide in the real 245-row ranking
                rmse=rmse,
            ),
            status="converged",
        )
        for rank, (r2, adj_r2, rmse) in enumerate(
            ((0.9999, 0.9999, 1234.567), (1.0, 1.0, 0.04692)), start=1
        )
    )
    view = RankingTableView()
    qtbot.addWidget(view)
    view.resize(width, 300)
    view.show()
    view.set_rows(rows)
    QApplication.processEvents()

    header = view.horizontalHeader()
    model = view.model()
    assert model is not None
    metrics = view.fontMetrics()
    assert view.horizontalScrollBar().maximum() == 0
    item_padding = 8  # QTableView#rankingTable::item horizontal padding
    header_padding = 8  # QHeaderView#rankingTableHeader::section horizontal padding
    for column in range(model.columnCount()):
        if column == 1:  # the model stretches and elides by design
            continue
        section = header.sectionSize(column)
        widest = metrics.horizontalAdvance(str(model.headerData(column, Qt.Orientation.Horizontal)))
        for row in range(model.rowCount()):
            text = str(model.data(model.index(row, column)))
            widest = max(widest, metrics.horizontalAdvance(text))
            # Qt's own elide logic against the text rect: the value is shown in full
            assert (
                metrics.elidedText(text, Qt.TextElideMode.ElideRight, section - item_padding)
                == text
            ), f"column {column} elides {text!r}: section {section}"
        # and the section pays for the item AND header padding, not a magic slack
        assert section >= widest + item_padding + header_padding, (
            f"column {column} section {section} does not cover {widest}px of text plus "
            f"{item_padding}+{header_padding}px of QSS padding"
        )
    # rank stays centred; the numeric columns stay right-aligned
    centred = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    right = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    assert model.data(model.index(0, 0), Qt.ItemDataRole.TextAlignmentRole) == centred
    for column in (2, 3, 4, 5):
        assert model.data(model.index(0, column), Qt.ItemDataRole.TextAlignmentRole) == right


def test_summary_strip_uses_fixed_decimals_and_na(qtbot) -> None:
    summary = FitSummaryPanel()
    qtbot.addWidget(summary)
    perfect = RankedFit(
        rank=1,
        model_id="poly1",
        name="Linear",
        category="polynomial",
        metrics=CandidateMetrics(k=2, n=5, r2=1.0, adj_r2=1.0, rmse=0.0),
    )
    summary.show_row(perfect, "exact data")
    assert summary.metric_text("R²") == "1.0000"
    assert summary.metric_text("adj. R²") == "1.0000"

    undefined = RankedFit(
        rank=2,
        model_id="poly2",
        name="Quadratic",
        category="polynomial",
        metrics=CandidateMetrics(k=3, n=5, r2=None, adj_r2=None, rmse=None),
    )
    summary.show_row(undefined, "exact data")
    assert summary.metric_text("R²") == "n/a"
    assert summary.metric_text("adj. R²") == "n/a"
    assert summary.metric_text("RMSE") == "n/a"


def test_summary_is_a_quiet_hint_when_nothing_is_selected(autofit_window) -> None:
    panel = autofit_window.analysis_panel
    panel.summary.clear()
    assert panel.summary.hint_visible()
    assert panel.summary.content.isHidden()
    assert "Select" in panel.summary.hint.text()


def test_missing_elapsed_time_is_a_dash_not_zero(autofit_window) -> None:
    view = autofit_window.analysis_panel.autofit_view
    view._started = 0.0  # no run in flight: no elapsed value exists
    view._tick_elapsed()
    assert view.elapsed_label.text() == "—"


def test_sections_and_options_start_collapsed(autofit_window) -> None:
    view = autofit_window.analysis_panel.autofit_view
    for section in (view.exact_section, view.unconverged_section, view.skipped_section):
        assert section._list.isHidden()  # counts are visible, the lists are not
    assert not view.options_button.isChecked()
    assert view.options_body.isHidden()


def test_banners_wrap_never_clip(autofit_window) -> None:
    view = autofit_window.analysis_panel.autofit_view
    # the fix for the overlapping banner text: every notice wraps inside its layout
    assert view.notice.wordWrap()
    assert view.overfit_banner.wordWrap()
    assert view.warnings_label.wordWrap()
    assert view.warnings_scroll.widget() is view.warnings_label


def test_mode_switch_is_a_horizontal_segmented_control(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")  # tests default to Manual; the default mode is Auto-fit
    panel = window.analysis_panel
    QApplication.processEvents()
    from PySide6.QtCore import QPoint

    autofit_y = panel.autofit_button.mapTo(panel, QPoint(0, 0)).y()
    manual_y = panel.manual_button.mapTo(panel, QPoint(0, 0)).y()
    assert autofit_y == manual_y  # side by side, not stacked full-width
    assert panel.autofit_button.isChecked()
    assert not panel.manual_button.isChecked()
    window.analysis.set_mode("manual")
    assert not panel.autofit_button.isChecked()
    assert panel.manual_button.isChecked()
    assert panel.mode() == "manual"


def test_splitter_sizes_are_remembered_in_settings(autofit_window, qtbot, settings) -> None:
    from PySide6.QtCore import QSettings

    panel = autofit_window.analysis_panel
    total = sum(panel.splitter.sizes())
    chosen = [int(total * 0.7), int(total * 0.3)]
    panel.splitter.setSizes(chosen)
    panel.save_splitter(settings)
    settings.sync()

    fresh = QSettings(settings.fileName(), QSettings.Format.IniFormat)
    panel.splitter.setSizes([int(total * 0.5), int(total * 0.5)])
    assert panel.restore_splitter(fresh) is True
    sizes = panel.splitter.sizes()
    assert abs(sizes[0] - chosen[0]) <= 6 and abs(sizes[1] - chosen[1]) <= 6

    # junk in the store is refused, never applied
    fresh.setValue("analysis/splitterSizes", "not, numbers")
    assert panel.restore_splitter(fresh) is False


def test_window_saves_and_restores_the_splitter(autofit_window, qtbot, settings) -> None:
    panel = autofit_window.analysis_panel
    total = sum(panel.splitter.sizes())
    chosen = [int(total * 0.7), int(total * 0.3)]
    panel.splitter.setSizes(chosen)
    autofit_window._save_layout()

    panel.splitter.setSizes([int(total * 0.5), int(total * 0.5)])
    assert panel.restore_splitter(settings) is True
    sizes = panel.splitter.sizes()
    assert abs(sizes[0] - chosen[0]) <= 6


def test_manual_page_still_fits_and_imports_data(make_window, qtbot, write_file) -> None:
    """The splitter must not break the manual workflow's basics."""
    window = make_window()
    window.show()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    panel = window.analysis_panel
    QApplication.processEvents()
    assert panel.manual_view.isVisible()
    assert panel.splitter.sizes()[0] > 0


@pytest.mark.parametrize("size", [(1600, 920), (1280, 800), (1280, 720)])
def test_analysis_dock_fits_above_the_data_table(make_window, qtbot, write_file, size) -> None:
    """The whole Analysis dock never reaches into the Data-table dock below it.

    The ranking table's old eight-row floor made the dock's minimum taller than the space
    the main window can give it at 1280x800 (the summary's Export button was painted over
    the other dock's title bar). The table's *minimum* is now a small usable window —
    three rows — because that floor is what decides how much height the user can give the
    chart and the data table; the default split still shows eight-plus rows.
    """
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    dataset_id = window.controller.project.datasets[0].id
    wait_stored(qtbot, window, dataset_id)
    window.resize(*size)
    QApplication.processEvents()

    view = window.analysis_panel.autofit_view
    assert view.table.rows()
    assert view.table.select_row(0) is not None
    QApplication.processEvents()

    panel = window.analysis_panel
    dock = window.table_dock
    available = dock.geometry().top() - panel.geometry().top()
    assert available > 0
    assert panel.minimumSizeHint().height() <= available, (
        f"analysis dock minimum {panel.minimumSizeHint().height()} exceeds "
        f"available {available} at {size}"
    )
    assert panel.geometry().bottom() <= dock.geometry().top(), (
        f"analysis dock bottom {panel.geometry().bottom()} reaches into the data dock "
        f"at {dock.geometry().top()} at {size}"
    )
    # three rows is the floor, not eight (the old floor is what overflowed the dock)
    row_height = view.table.verticalHeader().defaultSectionSize()
    assert view.table.minimumHeight() >= 3 * row_height
    assert view.table.minimumHeight() < 5 * row_height


@pytest.mark.parametrize(("size", "minimum_rows"), [((1600, 920), 7), ((1280, 800), 5)])
def test_fresh_window_shows_a_usable_number_of_ranking_rows(
    make_window, qtbot, write_file, size, minimum_rows
) -> None:
    """A fresh window spends its dock height on the ranking, not on the summary.

    The summary is still fully shown (its own minimum height) and the disclosure region
    stays a scrollable strip, but the table's viewport shows at least seven rows at
    1600x920 and five at 1280x800.
    """
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    dataset_id = window.controller.project.datasets[0].id
    wait_stored(qtbot, window, dataset_id)
    window.resize(*size)
    QApplication.processEvents()

    panel = window.analysis_panel
    view = panel.autofit_view
    assert view.table.rows()
    row_height = view.table.rowHeight(0)
    visible_rows = view.table.viewport().height() // row_height
    assert visible_rows >= minimum_rows, (
        f"{visible_rows} rows of {row_height}px at {size}; expected {minimum_rows}"
    )

    assert panel.summary.isVisible()
    assert panel.summary.height() >= panel.summary.minimumHeight()
    assert view.disclosure_scroll.isVisible()
    assert view.disclosure_scroll.widget() is view.disclosure_body


def test_ranking_table_scrollbar_is_visible_when_rows_overflow(make_window, qtbot) -> None:
    """245 rows scroll with a real vertical scrollbar, not a silently clipped last row."""
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.resize(1280, 800)
    view = window.analysis_panel.autofit_view
    rows = tuple(
        RankedFit(
            rank=index + 1,
            model_id=f"model_{index}",
            name=f"Model {index}",
            category="polynomial",
            metrics=CandidateMetrics(k=2, n=20, r2=0.99, adj_r2=0.98, rmse=0.1),
            status="converged",
        )
        for index in range(245)
    )
    view.table.set_rows(rows)
    assert view.table.select_row(0) is not None
    QApplication.processEvents()
    assert len(view.table.rows()) == 245
    bar = view.table.verticalScrollBar()
    assert bar.isVisible()
    assert bar.width() > 0
    assert bar.maximum() > 0


def test_order_combo_elides_the_closed_field_and_keeps_the_full_label(
    autofit_window,
) -> None:
    """The closed "Order by" combo never shows a word cut without an ellipsis."""
    combo = autofit_window.analysis_panel.autofit_view.rank_combo
    combo.setCurrentIndex(0)  # Best overall (recommended): the long label
    QApplication.processEvents()
    full = combo.currentText()
    assert full == "Best overall (recommended)"
    assert full in combo.toolTip()

    combo.setFixedWidth(90)  # narrower than the label: the field must elide
    QApplication.processEvents()
    shown = combo.display_text()
    assert shown.endswith("…"), f"closed field clips instead of eliding: {shown!r}"
    assert full.startswith(shown.removesuffix("…"))
    # the dropdown keeps every full label; only the closed field is elided
    assert combo.itemText(0) == full


def test_r2_columns_use_fixed_four_decimals(qtbot) -> None:
    """Fixed four decimals: "1.0000" and "0.9999" share one precision regime."""
    rows = tuple(
        RankedFit(
            rank=index + 1,
            model_id=f"poly{index}",
            name="Polynomial",
            category="polynomial",
            metrics=CandidateMetrics(k=2, n=20, r2=value, adj_r2=value, rmse=0.1),
            status="converged",
        )
        for index, value in enumerate((1.0, 0.9999))
    )
    view = RankingTableView()
    qtbot.addWidget(view)
    view.resize(440, 200)
    view.show()
    view.set_rows(rows)
    QApplication.processEvents()

    model = view.model()
    assert model is not None
    assert model.data(model.index(0, 2)) == "1.0000"
    assert model.data(model.index(0, 3)) == "1.0000"
    assert model.data(model.index(1, 2)) == "0.9999"
    assert model.data(model.index(1, 3)) == "0.9999"
    # the width comes from the strings the column actually paints
    metrics = view.fontMetrics()
    assert view.horizontalHeader().sectionSize(2) >= (
        metrics.horizontalAdvance("1.0000") + COLUMN_PADDING
    )


def test_ranking_copy_equation_uses_the_readable_caret(autofit_window) -> None:
    """The ranking's copy action copies ``^`` powers, exactly what the reader sees."""
    view = autofit_window.analysis_panel.autofit_view
    rows = view.table.rows()
    index = next(
        i for i, row in enumerate(rows) if row.equation_fitted and "**" in row.equation_fitted
    )
    row = view.table.select_row(index)
    assert row is not None and row.equation_fitted is not None
    QGuiApplication.clipboard().clear()
    assert view.table.copy_equation() is True
    assert QGuiApplication.clipboard().text() == format_equation(row.equation_fitted)
    assert "**" not in QGuiApplication.clipboard().text()


@pytest.mark.parametrize("mode", ["autofit", "manual"])
def test_the_data_table_can_grow_in_both_modes(make_window, qtbot, mode) -> None:
    """Whatever page the Analysis dock shows, the chart/data-table divider has travel.

    The dock's minimum height used to equal its content's (546 px), so the bottom dock
    could not gain a pixel. Both pages are now compact enough to leave real room.
    """
    window = make_window()
    qtbot.addWidget(window)
    window.resize(1280, 820)
    window.show()
    window.analysis.set_mode(mode)
    QApplication.processEvents()

    table = window.table_dock
    before = table.height()
    window.resizeDocks([table], [380], Qt.Orientation.Vertical)
    QApplication.processEvents()
    # the Auto-fit page carries the taller floor (two heading rows plus three ranking
    # rows), so it gives the table slightly less than the scrollable Manual fit page
    assert table.height() >= before + 70, f"{mode}: {before} -> {table.height()}"
    assert table.height() >= 290
    assert window.analysis_panel.geometry().bottom() < table.geometry().top()
