"""The Statistics mode in the analysis dock: the third button, page, summary rule and wiring.

The dock gains a third mode; its page is a :class:`StatsView`. Statistics mode hides the fit
summary (the page fills the dock), starts no Auto-fit and no manual fit when the dataset
changes, and its chosen plot reaches the main chart — and leaves it when the mode does.
"""

from PySide6.QtWidgets import QApplication

from curvestudio_desktop.controllers import MODE_STATS
from curvestudio_desktop.views.stats_view import StatsView

STATS_CSV = "x,y,group\n1,2.0,a\n2,4.1,a\n3,6.2,b\n4,7.9,b\n5,10.3,b\n"
MESSAGE = "Statistics mode: pick a tool and its columns; nothing runs until you do"


def test_three_mode_buttons_with_the_third_checkable(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    panel = window.analysis_panel
    assert panel.stats_button.objectName() == "statsModeButton"
    assert panel.stats_button.text() == "Statistics"
    for button in (panel.autofit_button, panel.manual_button, panel.stats_button):
        assert button.isCheckable()


def test_mode_stats_is_accepted_and_announces_itself(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    messages: list[str] = []
    window.analysis.status_message.connect(messages.append)
    window.analysis.set_mode(MODE_STATS)
    assert window.analysis.mode == MODE_STATS
    assert MESSAGE in messages
    assert window.analysis_panel.stats_button.isChecked()


def test_statistics_mode_shows_the_stats_page_and_hides_the_summary(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    panel = window.analysis_panel
    window.analysis.set_mode(MODE_STATS)
    QApplication.processEvents()
    assert isinstance(panel.stats_view, StatsView)
    assert panel._pages.currentWidget() is panel.stats_view
    assert panel.stats_view.isVisible()
    assert panel.summary.isHidden()

    # switching back restores the summary and hides the statistics page
    window.analysis.set_mode("manual")
    QApplication.processEvents()
    assert panel._pages.currentWidget() is panel.manual_view
    assert panel.manual_view.isVisible()
    assert not panel.summary.isHidden()
    assert panel.stats_view.isHidden()


def test_selecting_a_different_dataset_in_statistics_mode_starts_no_fit_job(
    make_window, qtbot, write_file
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    assert window.import_file(write_file("one.csv", STATS_CSV))
    assert window.import_file(write_file("two.csv", STATS_CSV))
    first, second = (entry.id for entry in window.controller.project.datasets)
    window.analysis.set_mode(MODE_STATS)
    QApplication.processEvents()
    runs, fits = window.analysis.run_count, window.analysis.fit_count

    window.selection.set_active(first)
    window.selection.set_active(second)
    QApplication.processEvents()
    assert window.analysis.run_count == runs
    assert window.analysis.fit_count == fits
    assert not window.analysis.is_autofit_running()
    assert not window.analysis.is_fit_running()

    # importing another dataset in Statistics mode starts nothing either
    assert window.import_file(write_file("three.csv", STATS_CSV))
    QApplication.processEvents()
    assert window.analysis.run_count == runs
    assert window.analysis.fit_count == fits


def test_a_statistics_plot_reaches_the_main_plot_and_leaves_with_the_mode(
    make_window, qtbot, write_file
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    assert window.import_file(write_file("stats.csv", STATS_CSV))
    window.analysis.set_mode(MODE_STATS)
    QApplication.processEvents()
    stats_view = window.analysis_panel.stats_view
    assert stats_view.select_tool("descriptives")
    stats_view.run_button.click()
    # the tool runs off the GUI thread on the controller's runner
    qtbot.waitUntil(lambda: window.plot_view.has_stats_plot(), timeout=10_000)
    assert window.plot_view.has_stats_plot()

    window.analysis.set_mode("manual")
    QApplication.processEvents()
    assert not window.plot_view.has_stats_plot()
