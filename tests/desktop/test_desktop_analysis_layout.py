"""The analysis dock's layout: the ranking table is the primary result.

The table dominates the dock (a vertical splitter gives the workflow page — the table —
at least 60% by default, with a minimum height of about eight rows), the detail tabs sit
below it, the disclosure sections start collapsed, banners are single-line, the mode
switch is a compact segmented control, and the splitter split is remembered in QSettings.
"""

import pytest
from desktop_data import GOOD_CSV
from PySide6.QtWidgets import QApplication
from test_desktop_autofit_run import NOISY_QUADRATIC, TINY, wait_stored


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


def test_detail_tabs_sit_below_the_table_page(autofit_window) -> None:
    panel = autofit_window.analysis_panel
    table_bottom = panel.autofit_view.mapTo(panel, panel.autofit_view.rect().bottomLeft()).y()
    tabs_top = panel.results.mapTo(panel, panel.results.rect().topLeft()).y()
    assert tabs_top >= table_bottom  # the tabs never ride above the ranking


def test_sections_and_options_start_collapsed(autofit_window) -> None:
    view = autofit_window.analysis_panel.autofit_view
    for section in (view.exact_section, view.unconverged_section, view.skipped_section):
        assert section._list.isHidden()  # counts are visible, the lists are not
    assert not view.options_button.isChecked()
    assert view.options_body.isHidden()


def test_banners_are_single_line(autofit_window) -> None:
    view = autofit_window.analysis_panel.autofit_view
    assert not view.notice.wordWrap()
    assert not view.overfit_banner.wordWrap()
    assert not view.warnings_label.wordWrap()


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
