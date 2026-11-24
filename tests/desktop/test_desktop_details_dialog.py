"""The details window: non-modal, reusable, geometry-persistent and wired to the ranking.

The dock keeps the ranking and a headline summary; the full six-view result opens in a
single details window per main window. These tests cover the window's own contract and
the two paths that feed it: a ranked row (refit then open) and a manual fit.
"""

import pytest
from desktop_data import GOOD_CSV
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication
from test_desktop_autofit_run import EXACT_LINEAR, NOISY_QUADRATIC, TINY, wait_stored

from curvestudio.tools import all_methods_texts
from curvestudio.workflow import format_equation
from curvestudio_desktop.views.details_dialog import GEOMETRY_KEY, MINIMUM_SIZE, TITLE
from curvestudio_desktop.views.main_window import MainWindow

TAB_TITLES = ["Parameters", "Statistics", "Diagnostics", "Fit audit", "Equation", "Predict"]
USER_EQUATION = (
    "y = 0.680918 + 2.37953*x + 0.0900383*x**2 + 0.0127839*x**3 "
    "+ (-0.000548964)*x**4 + 1.07529e-05*x**5"
)


@pytest.fixture()
def autofit_window(make_window, qtbot, write_file):
    """A window with a finished tiny Auto-fit run, shown and sized like a real screen."""
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("autofit")
    window.analysis_panel.autofit_view.set_config(TINY)
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    wait_stored(qtbot, window, window.selection.active_id)
    window.resize(1400, 850)
    QApplication.processEvents()
    yield window
    window.details_dialog.close()


def test_details_window_is_non_modal_resizable_and_has_the_six_views(
    autofit_window,
) -> None:
    dialog = autofit_window.details_dialog
    assert dialog.windowTitle() == TITLE
    assert dialog.isModal() is False  # non-modal: the ranking stays usable behind it
    assert dialog.minimumWidth() >= MINIMUM_SIZE[0]
    assert dialog.minimumHeight() >= MINIMUM_SIZE[1]
    assert dialog.windowFlags() & Qt.WindowType.Window
    assert dialog.results.tab_titles() == TAB_TITLES
    # no fit yet: the honest empty state, not invented numbers
    assert dialog.results.tab_text_at(0).strip() == "No fit yet"


def test_ranked_row_opens_the_bundle_with_equation_and_copy(autofit_window, qtbot) -> None:
    window = autofit_window
    view = window.analysis_panel.autofit_view
    row = view.table.rows()[0]
    view.table.select_row(0)
    view.table.openRequested.emit()  # the same path double-click and Enter take

    dialog = window.details_dialog
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=60000)
    bundle = dialog.bundle()
    assert bundle is not None and bundle.result.model_id == row.model_id
    shown = format_equation(bundle.equation_text)
    assert dialog.equation_text() == shown
    assert dialog.results.equation_text() == shown
    assert "**" not in shown
    assert dialog.model_text() == row.name
    assert dialog.status_text() == bundle.result.status.value

    QGuiApplication.clipboard().clear()
    dialog.copy_button.click()
    assert QGuiApplication.clipboard().text() == shown


def test_footer_copies_the_methods_text_beside_the_equation(autofit_window, qtbot) -> None:
    """Copy methods text sits beside Copy equation and offers the three styles."""
    window = autofit_window
    view = window.analysis_panel.autofit_view
    assert view.table.select_row(0) is not None
    window.action_details.trigger()
    dialog = window.details_dialog
    qtbot.waitUntil(lambda: dialog.bundle() is not None, timeout=60000)
    QApplication.processEvents()
    bundle, spec = dialog.bundle(), dialog.spec()
    assert bundle is not None and spec is not None

    footer = dialog.methods_button
    assert footer.text() == "Copy methods text"
    assert [action.text() for action in footer.menu().actions()] == ["Text", "Markdown", "LaTeX"]
    assert footer.isEnabled()  # the row's fit converged
    # beside Copy equation: the same row of the footer, to its right
    assert footer.y() == dialog.copy_button.y()
    assert dialog.copy_button.x() < footer.x()

    text = all_methods_texts(bundle, spec=spec)["latex"]
    QGuiApplication.clipboard().clear()
    footer.menu().actions()[2].trigger()
    assert (
        QGuiApplication.clipboard().text() == f"{text.paragraph}\n\n{text.equation}\n\n{text.table}"
    )


def test_equation_displays_and_copies_caret_powers_and_empty_state(autofit_window, qtbot) -> None:
    """The user's exact polynomial: ``**`` is shown as ``^`` and copied identically."""
    window = autofit_window
    view = window.analysis_panel.autofit_view
    assert view.table.select_row(0) is not None
    window.action_details.trigger()
    qtbot.waitUntil(lambda: window.details_dialog.bundle() is not None, timeout=60000)
    dialog = window.details_dialog
    bundle = dialog.bundle()
    assert bundle is not None

    example = bundle.model_copy(update={"equation_text": USER_EQUATION})
    dialog.show_bundle(example, dialog.spec(), dataset_name="quad")
    shown = dialog.equation_text()
    assert shown == format_equation(USER_EQUATION)
    for power in ("x^2", "x^3", "x^4", "x^5"):
        assert power in shown
    assert "1.07529e-05" in shown  # scientific notation is untouched
    assert "*x" in shown and "**" not in shown  # single products survive too
    assert dialog.results.equation_text() == shown
    assert dialog.results._equation_label.text() == shown
    # copy shows the reader's text, not the engine's storage syntax
    QGuiApplication.clipboard().clear()
    assert dialog.copy_button.isEnabled()
    assert dialog.copy_equation() is True
    assert QGuiApplication.clipboard().text() == shown
    assert dialog.results.copy_equation() is True
    assert QGuiApplication.clipboard().text() == shown

    # None/empty renders nothing and offers nothing to copy
    empty = bundle.model_copy(update={"equation_text": ""})
    dialog.show_bundle(empty, dialog.spec(), dataset_name="quad")
    assert dialog.equation_text() == ""
    assert dialog.results.equation_text() == ""
    assert dialog.results._equation_label.text() == ""
    assert dialog.copy_button.isEnabled() is False
    assert dialog.copy_equation() is False
    assert dialog.results.copy_equation() is False


def test_ctrl_d_opens_the_selected_rows_details(autofit_window, qtbot) -> None:
    window = autofit_window
    assert window.action_details.shortcut().toString() == "Ctrl+D"
    selected = window.analysis_panel.autofit_view.table.selected_row()
    assert selected is not None
    window.action_details.trigger()
    qtbot.waitUntil(lambda: window.details_dialog.isVisible(), timeout=60000)
    bundle = window.details_dialog.bundle()
    assert bundle is not None and bundle.result.model_id == selected.model_id


def test_enter_on_a_ranked_row_refits_and_opens_details(autofit_window, qtbot) -> None:
    window = autofit_window
    view = window.analysis_panel.autofit_view
    assert len(view.table.rows()) >= 2
    row = view.table.rows()[1]
    assert view.table.select_row(1) is not None
    fits_before = window.analysis.fit_count
    view.table.setFocus()
    qtbot.keyClick(view.table, Qt.Key.Key_Return)
    qtbot.waitUntil(lambda: window.details_dialog.isVisible(), timeout=60000)
    assert window.analysis.fit_count == fits_before + 1  # the row was refitted exactly once
    bundle = window.details_dialog.bundle()
    assert bundle is not None and bundle.result.model_id == row.model_id


def test_reopening_updates_the_same_window(autofit_window, qtbot) -> None:
    window = autofit_window
    view = window.analysis_panel.autofit_view
    dialog = window.details_dialog
    assert len(view.table.rows()) >= 2

    view.table.select_row(0)
    window.action_details.trigger()
    qtbot.waitUntil(lambda: dialog.isVisible(), timeout=60000)
    first = dialog.bundle()
    assert first is not None

    second_row = view.table.rows()[1]
    view.table.select_row(1)
    window.action_details.trigger()
    qtbot.waitUntil(
        lambda: (
            dialog.bundle() is not None and dialog.bundle().result.model_id == second_row.model_id
        ),
        timeout=60000,
    )
    # one window, not a stack: the same instance now shows the other fit
    assert window.details_dialog is dialog
    assert dialog.isVisible()
    assert first.result.model_id != dialog.bundle().result.model_id


def test_geometry_persists_across_windows(autofit_window, qtbot, settings, controller) -> None:
    dialog = autofit_window.details_dialog
    dialog.show()
    # the offscreen screen is 800x800, so only the height is meaningful there; the
    # width always comes back at the dialog's 900 px floor
    dialog.resize(900, 740)
    QApplication.processEvents()
    dialog.close()
    settings.sync()
    assert settings.contains(GEOMETRY_KEY)

    fresh = MainWindow(controller, settings=settings)
    qtbot.addWidget(fresh)
    try:
        assert fresh.details_dialog is not dialog  # a new window gets its own details window
        size = fresh.details_dialog.size()
        assert size.width() >= MINIMUM_SIZE[0]
        assert abs(size.height() - 740) <= 6, f"restored height {size.height()}"
        # and the dialog can still be resized after the restore (not frozen at the size)
        fresh.details_dialog.resize(900, 700)
        QApplication.processEvents()
        assert abs(fresh.details_dialog.height() - 700) <= 6
    finally:
        fresh.details_dialog.close()


def test_manual_fit_opens_details_and_fills_the_summary(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    view = window.analysis_panel.manual_view
    shown: list = []
    window.analysis.fit_finished.connect(lambda bundle, spec, meta: shown.append(bundle))
    view.run_fit()
    qtbot.waitUntil(lambda: bool(shown), timeout=30000)

    dialog = window.details_dialog
    assert dialog.isVisible()  # the manual path opens its result
    assert dialog.bundle() is shown[-1]
    summary = window.analysis_panel.summary
    assert not summary.hint_visible()
    assert summary.model_text()
    assert summary.status_text() == shown[-1].result.status.value
    assert summary.metric_text("RMSE") != "n/a"
    dialog.close()


def test_summary_export_is_only_available_for_a_complete_fit(autofit_window, qtbot) -> None:
    window = autofit_window
    view = window.analysis_panel.autofit_view
    summary = window.analysis_panel.summary
    # a ranked row is a summary, not a bundle: export must wait for the refit
    view.table.select_row(0)
    QApplication.processEvents()
    assert summary.bundle() is None
    assert summary.export_button.isEnabled() is False

    window.action_details.trigger()  # refits the row; the bundle then reaches the summary
    qtbot.waitUntil(lambda: summary.bundle() is not None, timeout=60000)
    assert summary.export_button.isEnabled() is True


def test_metrics_strips_show_full_precision_and_na(make_window, qtbot, write_file) -> None:
    """R² renders its exact stored value: a perfect fit reads "1.0", not a rounded form."""
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("exact.csv", EXACT_LINEAR))
    view = window.analysis_panel.manual_view
    view.degree_spin.setValue(1)  # y = 2 + 3x exactly: R² and adj. R² are both 1
    view.run_fit()
    dialog = window.details_dialog
    qtbot.waitUntil(lambda: dialog.bundle() is not None, timeout=30000)

    header = dialog.metrics_text()
    assert "R² 1.0" in header
    assert "adj. R² 1.0" in header
    summary = window.analysis_panel.summary
    assert summary.metric_text("R²") == "1.0"
    assert summary.metric_text("adj. R²") == "1.0"

    # an undefined metric is still n/a, never 0
    bundle = dialog.bundle()
    assert bundle is not None
    stats = bundle.result.stats
    assert stats is not None
    undefined_stats = stats.model_copy(update={"r2": None, "adj_r2": None})
    undefined = bundle.model_copy(
        update={"result": bundle.result.model_copy(update={"stats": undefined_stats})}
    )
    dialog.show_bundle(undefined, dialog.spec(), dataset_name="exact")
    assert "R² n/a" in dialog.metrics_text()
    assert "adj. R² n/a" in dialog.metrics_text()
    dialog.close()


def test_result_tables_fill_the_details_width(make_window, qtbot, write_file) -> None:
    """Every result table uses the whole window width, not just its content's share."""
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    window.analysis.set_mode("manual")
    assert window.import_file(write_file("quad.csv", NOISY_QUADRATIC))
    view = window.analysis_panel.manual_view
    view.degree_spin.setValue(4)  # five parameters: the full parameters table
    view.run_fit()
    dialog = window.details_dialog
    qtbot.waitUntil(lambda: dialog.bundle() is not None, timeout=30000)
    assert len(dialog.bundle().result.parameters) == 5

    dialog.resize(1150, 800)
    QApplication.processEvents()
    tabs = dialog.results
    predict = tabs.TAB_TITLES.index("Predict")
    tables = ((0, tabs._parameters), (1, tabs._statistics), (predict, tabs._predict_table))
    for tab_index, table in tables:
        tabs._tabs.setCurrentIndex(tab_index)
        QApplication.processEvents()
        header = table.horizontalHeader()
        assert table.isVisible()
        total = sum(header.sectionSize(column) for column in range(header.count()))
        viewport = table.viewport().width()
        assert abs(total - viewport) <= 2, f"tab {tab_index}: {total} != {viewport}"
        # a stretched last column keeps the table inside the viewport
        assert table.horizontalScrollBar().maximum() == 0
        last = header.count() - 1
        metrics = table.fontMetrics()
        content = header.fontMetrics().horizontalAdvance(table.horizontalHeaderItem(last).text())
        for row in range(table.rowCount()):
            item = table.item(row, last)
            if item is not None:
                content = max(content, metrics.horizontalAdvance(item.text()))
        if table is tabs._statistics:
            # the diagnostic prose in the note column is unbounded (it is the cell
            # tooltip and elides by design), so "wider than its full content" cannot
            # hold; the exact viewport fill above is what proves the stretch there
            assert header.sectionSize(last) < content
        else:
            assert header.sectionSize(last) > content, (
                f"tab {tab_index}: last column {header.sectionSize(last)} not wider than {content}"
            )
    dialog.close()
