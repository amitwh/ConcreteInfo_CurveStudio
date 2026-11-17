"""Widget smoke tests: the shell builds, docks exist, data reaches table and plot."""

import time

from desktop_data import CLIPBOARD_COMMA_TEXT, GOOD_CSV, MANY_CSV, NONPOSITIVE_CSV
from PySide6.QtCore import Qt, qVersion
from PySide6.QtWidgets import QDockWidget, QLabel

from curvestudio_desktop import APP_NAME, NO_NETWORK_NOTE, __version__

HORIZONTAL = Qt.Orientation.Horizontal


def find_dock(window, name: str) -> QDockWidget:
    dock = window.findChild(QDockWidget, name)
    assert dock is not None, f"dock {name!r} not found"
    return dock


def test_main_window_builds_with_all_docks(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    for name in ("projectDock", "analysisDock", "tableDock"):
        find_dock(window, name)
    assert APP_NAME in window.windowTitle()
    assert window.minimumSize().width() >= 900


def test_analysis_panel_has_four_result_tabs(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    panel = window.analysis_panel
    assert panel.tab_titles() == ["Parameters", "Statistics", "Diagnostics", "Equation"]
    for index in range(panel.count()):
        assert panel.tab_text_at(index).strip() == "No fit yet"


def test_status_bar_shows_no_network_indicator(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    label = window.statusBar().findChild(QLabel, "noNetworkLabel")
    assert label is not None
    assert NO_NETWORK_NOTE in label.text()
    observations = window.statusBar().findChild(QLabel, "observationsLabel")
    assert observations is not None
    assert "0" in observations.text()  # no dataset yet


def test_import_file_populates_table_and_plot(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    path = write_file("points.csv", GOOD_CSV)
    assert window.import_file(path) is True
    model = window.table_view.model()
    assert model is not None
    assert model.rowCount() == 5
    assert model.columnCount() == 4  # x, y, weight, included
    x, y = window.plot_view.scatter_data()
    assert len(x) == 5 and len(y) == 5
    observations = window.statusBar().findChild(QLabel, "observationsLabel")
    assert "5" in observations.text()
    headers = [model.headerData(col, HORIZONTAL) for col in range(model.columnCount())]
    assert headers[:2] == ["x", "y"]


def test_table_headers_use_column_names(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    text = "age_days,strength_mpa\n1,10\n2,20\n3,30\n4,40\n5,50\n"
    window.import_file(write_file("cubes.csv", text))
    model = window.table_view.model()
    names = [model.headerData(col, HORIZONTAL) for col in range(2)]
    assert names == ["age_days", "strength_mpa"]
    assert window.plot_view.axis_labels() == ("age_days", "strength_mpa")


def test_plot_axes_labels_follow_columns(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("p.csv", GOOD_CSV))
    assert window.plot_view.axis_labels() == ("x", "y")


def test_issue_rows_are_flagged_in_table(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    text = "x,y\n1,2.0\n2,abc\n3,6.2\n4,7.9\n5,10.3\n"
    assert window.import_file(write_file("bad.csv", text)) is False  # errors block import
    # import clean data, then feed the model the issues directly (the dialog path)
    assert window.import_file(write_file("good.csv", GOOD_CSV))
    from curvestudio.workflow import ImportIssue

    active_id = window.selection.active_id
    issue = ImportIssue(
        code="MISSING_VALUE", severity="error", message="Blank or NaN values in 'y'", rows=(1,)
    )
    window.selection.set_issues(active_id, [issue])
    window.refresh_table()
    model = window.table_view.model()
    assert model is not None
    assert model.tooltip_for_row(1)
    assert not model.tooltip_for_row(0)


def test_table_virtualised_for_100k_rows(make_window, qtbot, tmp_path) -> None:
    window = make_window()
    qtbot.addWidget(window)
    rows = 100_000
    path = tmp_path / "big.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("x,y\n")
        for i in range(rows):
            handle.write(f"{i / 1000:.3f},{(i % 97) / 50:.3f}\n")
    assert window.import_file(path) is True  # untimed: the parse cost is not the claim
    model = window.table_view.model()
    assert model is not None
    assert model.rowCount() == rows
    # only the model build is timed — building it again from the stored dataset
    from curvestudio_desktop.views.data_table import DatasetTableModel

    dataset = window.controller.project.datasets[0].dataset
    start = time.monotonic()
    rebuilt = DatasetTableModel(dataset, "x", "y", None)
    elapsed = time.monotonic() - start
    assert rebuilt.rowCount() == rows
    assert elapsed < 2.0, f"100k-row model build took {elapsed:.2f}s"
    value = model.data(model.index(99_999, 0))
    assert value is not None and value  # data access at the far end stays cheap


def test_log_toggles_disabled_for_nonpositive_data(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    assert window.action_log_x.isEnabled() is False  # nothing plotted yet
    window.import_file(write_file("good.csv", GOOD_CSV))
    assert window.action_log_x.isEnabled() is True
    assert window.action_log_y.isEnabled() is True
    window.import_file(write_file("neg.csv", NONPOSITIVE_CSV))  # becomes the active dataset
    assert window.action_log_y.isEnabled() is False
    tip = window.action_log_y.toolTip().lower()
    assert "log" in tip or "positive" in tip
    assert window.action_log_x.isEnabled() is True  # x itself stays positive


def test_clipboard_import_populates_project(make_window, qtbot, controller) -> None:
    window = make_window()
    qtbot.addWidget(window)
    dialog = window.make_clipboard_dialog(initial_text=CLIPBOARD_COMMA_TEXT)
    # the window's runner is async: the parse lands shortly
    qtbot.waitUntil(
        lambda: dialog.outcome() is not None and dialog.outcome().xy is not None, timeout=5000
    )
    outcome = dialog.outcome()
    assert outcome is not None and outcome.xy is not None
    assert controller.add_imported(outcome) is not None
    assert len(controller.project.datasets) == 1
    assert controller.project.datasets[0].dataset.name == "clipboard"


def test_project_tree_lists_datasets_and_analyses(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("points.csv", GOOD_CSV))
    window.import_file(write_file("more.csv", MANY_CSV))
    tree = window.project_tree
    labels = tree.dataset_labels()
    assert labels == ["points", "more"] or sorted(labels) == ["more", "points"]


def test_tree_selection_follows_latest_import(make_window, qtbot, write_file) -> None:
    """Importing a second dataset must leave the tree showing it selected (I1)."""
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("a.csv", GOOD_CSV))
    window.import_file(write_file("b.csv", MANY_CSV))
    second = window.controller.project.datasets[1]
    assert window.selection.active_id == second.id
    assert window.project_tree.selected_dataset_id() == second.id


def _count_rebuilds(window, monkeypatch) -> dict[str, int]:
    """Count real table-model and plot rebuilds triggered through the wiring."""
    from curvestudio_desktop.views import main_window as mw
    from curvestudio_desktop.views.plot_view import PlotView

    counts = {"table": 0, "plot": 0}
    origin_model = mw.DatasetTableModel
    origin_set_data = PlotView.set_data

    def counting_model(*args, **kwargs):
        counts["table"] += 1
        return origin_model(*args, **kwargs)

    def counting_set_data(*args, **kwargs):
        counts["plot"] += 1
        return origin_set_data(*args, **kwargs)

    monkeypatch.setattr(mw, "DatasetTableModel", counting_model)
    monkeypatch.setattr(PlotView, "set_data", counting_set_data)
    return counts


def test_import_refreshes_table_and_plot_exactly_once(
    make_window, qtbot, write_file, monkeypatch
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    counts = _count_rebuilds(window, monkeypatch)
    window.import_file(write_file("a.csv", GOOD_CSV))
    assert (counts["table"], counts["plot"]) == (1, 1)
    window.import_file(write_file("b.csv", MANY_CSV))
    assert (counts["table"], counts["plot"]) == (2, 2)


def test_rename_refreshes_neither_table_nor_plot(
    make_window, qtbot, write_file, monkeypatch
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("a.csv", GOOD_CSV))
    counts = _count_rebuilds(window, monkeypatch)
    entry_id = window.controller.project.datasets[0].id
    assert window.controller.rename_dataset(entry_id, "renamed") is True
    assert (counts["table"], counts["plot"]) == (0, 0)
    assert "renamed" in window.project_tree.dataset_labels()


def test_export_plot_uses_export_dialog_not_project_dialog(
    make_window, qtbot, write_file, tmp_path
) -> None:
    """Plot export must use choose_export_path with image filters (never *.cstudio)."""
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("p.csv", GOOD_CSV))
    target = tmp_path / "plot.png"
    window.controller.paths.export_paths.append(target)
    window.export_plot()
    assert window.controller.paths.export_calls, "the export dialog was never used"
    title, filters = window.controller.paths.export_calls[0]
    assert "export" in title.lower()
    assert "*.cstudio" not in filters and "*.png" in filters
    assert target.is_file() and target.stat().st_size > 0


def test_export_plot_cancelled_touches_nothing(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("p.csv", GOOD_CSV))
    window.export_plot()  # no scripted path: user cancelled
    assert window.controller.paths.export_calls  # the dialog was asked, then cancelled
    assert window.controller.messenger.infos == []


def test_weighted_dataset_keeps_weight_column_across_save_and_open(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    """A weighted dataset reopened from .cstudio must keep (and fit with) its weights."""
    from curvestudio.workflow import import_dataset

    window = make_window()
    qtbot.addWidget(window)
    text = "x,y,w\n1,2.0,1.0\n2,4.1,1.5\n3,6.2,1.0\n4,7.9,2.0\n5,10.3,1.0\n"
    outcome = import_dataset(
        write_file("w.csv", text), "w.csv", x_column="x", y_column="y", weight_column="w"
    )
    dataset_id = controller.add_imported(outcome)
    assert dataset_id is not None
    assert window.selection.columns(dataset_id) == ("x", "y", "w")

    target = tmp_path / "weighted.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True

    # the archive itself must carry the choice — a fresh selection (another session)
    from curvestudio.project import load_project
    from curvestudio_desktop.controllers.selection import SelectionController

    saved = load_project(target)
    stored = saved.plot_settings.get("desktop", {}).get("datasets", {})
    assert stored[dataset_id]["columns"] == ["x", "y", "w"]
    fresh = SelectionController()
    fresh.set_project(saved)
    assert fresh.columns(dataset_id) == ("x", "y", "w")
    fresh_xy = fresh.xy()
    assert fresh_xy is not None and fresh_xy.weights is not None

    assert controller.open_project(target) is True  # not dirty after save: no prompt
    reopened_id = controller.project.datasets[0].id
    assert window.selection.columns(reopened_id) == ("x", "y", "w")
    xy = window.selection.xy()
    assert xy is not None
    assert xy.weights is not None and len(xy.weights) == 5


def test_import_issues_survive_save_and_open(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    from curvestudio.workflow import import_dataset

    window = make_window()
    qtbot.addWidget(window)
    text = "x,y\n1,2.0\n2,4.1\n2,4.1\n3,6.2\n4,7.9\n"  # duplicate row: a warning
    outcome = import_dataset(write_file("dup.csv", text), "dup.csv")
    dataset_id = controller.add_imported(outcome)
    assert dataset_id is not None
    assert window.selection.issues(dataset_id), "the warning issues should be stored"

    target = tmp_path / "issues.cstudio"
    controller.paths.save_paths.append(target)
    assert controller.save() is True

    from curvestudio.project import load_project
    from curvestudio_desktop.controllers.selection import SelectionController

    saved = load_project(target)
    stored = saved.plot_settings.get("desktop", {}).get("datasets", {})
    assert any(issue["code"] == "DUPLICATE_ROW" for issue in stored[dataset_id]["issues"])
    fresh = SelectionController()
    fresh.set_project(saved)
    assert any(issue.code == "DUPLICATE_ROW" for issue in fresh.issues(dataset_id))

    assert controller.open_project(target) is True
    reopened_id = controller.project.datasets[0].id
    assert any(issue.code == "DUPLICATE_ROW" for issue in window.selection.issues(reopened_id))


def test_async_save_and_open_round_trip(
    make_window, qtbot, controller, write_file, tmp_path
) -> None:
    """Save and open run through the JobRunner without freezing the GUI thread."""
    from curvestudio_desktop.controllers.providers import UnsavedAnswer

    window = make_window()  # the production wiring: an async JobRunner
    qtbot.addWidget(window)
    assert window.import_file(write_file("p.csv", GOOD_CSV))
    target = tmp_path / "async.cstudio"
    controller.paths.save_paths.append(target)
    window._save_as_async()
    qtbot.waitUntil(lambda: controller.path == target, timeout=5000)
    assert controller.is_dirty is False
    assert target.is_file()

    assert window.import_file(write_file("q.csv", MANY_CSV))  # dirty again
    controller.paths.open_paths.append(target)
    controller.confirmation.answers.append(UnsavedAnswer.DISCARD)
    window._open_project_async()
    qtbot.waitUntil(lambda: len(controller.project.datasets) == 1, timeout=5000)
    assert controller.project.datasets[0].name == "p"
    assert window.selection.active_id == controller.project.datasets[0].id


def test_busy_indicator_and_cancel_button(make_window, qtbot) -> None:
    import threading

    from PySide6.QtWidgets import QApplication, QPushButton

    window = make_window()
    qtbot.addWidget(window)
    release = threading.Event()

    def slow() -> int:
        release.wait(timeout=5)
        return 42

    seen: list[object] = []
    window._run_job("Working", slow, on_result=seen.append, cancellable=True)
    bar = window.statusBar()
    assert "Working" in bar.currentMessage()
    cancel = bar.findChild(QPushButton, "cancelJobButton")
    assert cancel is not None and not cancel.isHidden()
    assert QApplication.overrideCursor() is not None  # busy cursor while running

    cancel.click()
    release.set()
    qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=5000)
    assert seen == []  # cancelled: the result was dropped
    assert cancel.isHidden()
    assert QApplication.overrideCursor() is None  # cursor restored


def test_uncancellable_job_shows_no_cancel_button(make_window, qtbot) -> None:
    from PySide6.QtWidgets import QPushButton

    window = make_window()
    qtbot.addWidget(window)
    window._run_job("Saving", lambda: "written", on_result=lambda _r: None)
    qtbot.waitUntil(lambda: not window.jobs.is_busy(), timeout=5000)
    cancel = window.statusBar().findChild(QPushButton, "cancelJobButton")
    assert cancel is not None and cancel.isHidden()


def test_project_tree_context_menu_renames_and_removes(make_window, qtbot, write_file) -> None:
    """Right-click on a dataset offers Rename/Remove, wired to the controller."""
    window = make_window()
    qtbot.addWidget(window)
    window.import_file(write_file("a.csv", GOOD_CSV))
    entry_id = window.controller.project.datasets[0].id

    menu = window.dataset_context_menu(entry_id)
    labels = [action.text() for action in menu.actions()]
    assert "Rename…" in labels and "Remove" in labels

    assert window.rename_dataset(entry_id, "Beam A") is True
    assert window.project_tree.dataset_labels() == ["Beam A"]

    assert window.remove_dataset(entry_id) is True
    assert window.project_tree.dataset_labels() == []
    assert window.controller.undo() is True  # undo brings it back
    assert window.project_tree.dataset_labels() == ["Beam A"]


def test_dock_layout_is_saved_and_restored(controller, qtbot, settings) -> None:
    from curvestudio_desktop.views.main_window import MainWindow

    first = MainWindow(controller, settings=settings)
    qtbot.addWidget(first)
    assert first.layout_restored is False  # nothing stored yet: honest default
    first.resize(1100, 700)
    first.analysis_panel.resize(520, 400)
    first.close()  # closeEvent saves the dock layout into the isolated settings
    assert settings.contains("main_window/state")

    second = MainWindow(controller, settings=settings)
    qtbot.addWidget(second)
    assert second.layout_restored is True


def test_analysis_dock_starts_wide(make_window, qtbot) -> None:
    from curvestudio_desktop.views.main_window import ANALYSIS_DOCK_WIDTH

    window = make_window()
    qtbot.addWidget(window)
    window.show()
    assert window.analysis_panel.width() >= ANALYSIS_DOCK_WIDTH - 2


def test_about_dialog_shows_licence_and_version(qtbot) -> None:
    from curvestudio_desktop.views.dialogs import AboutDialog

    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    text = dialog.full_text()
    assert __version__ in text
    assert "AGPL" in text
    assert "PySide6" in text and qVersion()[:3] in text
