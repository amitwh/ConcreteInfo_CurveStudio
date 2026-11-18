"""MainWindow: menu bar, toolbar, status bar and the four dock areas (PRD §16.1).

The window is a thin shell: every action delegates to the ProjectController, and
every display refresh is driven by controller signals. The plot controls (log
axes, reset, export) delegate to PlotView. Nothing here computes anything.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import QByteArray, QPoint, QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QTableView,
    QToolBar,
    QWidget,
)

from curvestudio.discovery import AutoFitResult
from curvestudio.workflow import FitBundle, FitSpec, ImportOutcome
from curvestudio_desktop import APP_NAME, NO_NETWORK_NOTE
from curvestudio_desktop.controllers import exports
from curvestudio_desktop.controllers.analysis import AnalysisController
from curvestudio_desktop.controllers.analysis_jobs import AutoFitLive
from curvestudio_desktop.controllers.project import ProjectController
from curvestudio_desktop.controllers.selection import SelectionController
from curvestudio_desktop.jobs import JobHandle, JobRunner
from curvestudio_desktop.views.analysis_panel import AnalysisPanel
from curvestudio_desktop.views.data_table import DatasetTableModel, DataTableDock
from curvestudio_desktop.views.import_dialog import ClipboardImportDialog, ImportDialog
from curvestudio_desktop.views.plot_view import PlotView
from curvestudio_desktop.views.project_tree import ProjectDock, ProjectTree

MINIMUM_SIZE = (980, 640)
ANALYSIS_DOCK_WIDTH = 420
"""The right dock starts wide: fit parameters and statistics need the room."""
_RECENT_LIMIT = 10
_LAYOUT_STATE_KEY = "main_window/state"
_LAYOUT_GEOMETRY_KEY = "main_window/geometry"


class MainWindow(QMainWindow):
    """The shell: left project tree, centre plot, right analysis, bottom data table."""

    def __init__(
        self,
        controller: ProjectController,
        parent: QWidget | None = None,
        jobs: JobRunner | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selection = SelectionController(self)
        self.jobs = jobs if jobs is not None else JobRunner(self)
        self._settings = settings if settings is not None else QSettings()
        self._running_job: JobHandle | None = None
        self._busy_cursor = False
        self._suppress_autostart = False
        self.layout_restored = False
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setObjectName("mainWindow")

        self.analysis = AnalysisController(
            controller, self.selection, settings=self._settings, parent=self
        )

        self.plot_view = PlotView(self)
        self.setCentralWidget(self.plot_view)

        self.project_dock = ProjectDock(self)
        self.analysis_panel = AnalysisPanel(self.analysis, self.selection, self)
        self.table_dock = DataTableDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.project_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.table_dock)
        self.resizeDocks((self.analysis_panel,), (ANALYSIS_DOCK_WIDTH,), Qt.Orientation.Horizontal)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._wire()
        self._restore_layout()
        self.refresh_all()

    # -- convenient accessors (docks own the widgets) -----------------------------------------

    @property
    def table_view(self) -> QTableView:
        return self.table_dock.view

    @property
    def project_tree(self) -> ProjectTree:
        return self.project_dock.tree

    # -- construction ------------------------------------------------------------------------

    def _build_actions(self) -> None:
        self.action_new = self._action(
            "&New project", QKeySequence(QKeySequence.StandardKey.New), "Start a clean project"
        )
        self.action_new.triggered.connect(lambda: self.controller.new_project())

        self.action_open = self._action(
            "&Open project…", QKeySequence(QKeySequence.StandardKey.Open), "Open a .cstudio file"
        )
        self.action_open.triggered.connect(lambda: self._open_project_async())

        self.action_save = self._action(
            "&Save", QKeySequence(QKeySequence.StandardKey.Save), "Save the project"
        )
        self.action_save.triggered.connect(lambda: self._save_async())

        self.action_save_as = self._action(
            "Save &as…", QKeySequence(QKeySequence.StandardKey.SaveAs), "Save to a new file"
        )
        self.action_save_as.triggered.connect(lambda: self.controller.save_as())

        self.action_import = self._action(
            "&Import data…", QKeySequence("Ctrl+I"), "Import a data file"
        )
        self.action_import.triggered.connect(self.open_import_dialog)

        self.action_clipboard = self._action(
            "Import from &clipboard",
            QKeySequence("Ctrl+Shift+V"),
            "Paste tab or comma separated data",
        )
        self.action_clipboard.triggered.connect(self.open_clipboard_dialog)

        self.action_quit = self._action("&Quit", QKeySequence("Ctrl+Q"), "Close CurveStudio")
        self.action_quit.triggered.connect(self.close)

        self.action_undo = self._action(
            "&Undo", QKeySequence(QKeySequence.StandardKey.Undo), "Undo the last change"
        )
        self.action_undo.triggered.connect(lambda: self.controller.undo())

        self.action_log_x = self._action("&Log x-axis", None, "Logarithmic x-axis", checkable=True)
        self.action_log_x.toggled.connect(self.plot_view.set_log_x)
        self.action_log_y = self._action("Log &y-axis", None, "Logarithmic y-axis", checkable=True)
        self.action_log_y.toggled.connect(self.plot_view.set_log_y)

        self.action_reset_view = self._action(
            "&Reset view", QKeySequence("Ctrl+R"), "Fit the plot to the data"
        )
        self.action_reset_view.triggered.connect(self.plot_view.reset_view)

        self.action_export_plot = self._action(
            "&Export plot…", QKeySequence("Ctrl+E"), "Export the plot as PNG or SVG"
        )
        self.action_export_plot.triggered.connect(self.export_plot)

        self.action_export_results = self._action(
            "Export &results…", None, "Export the current fit (JSON, XLSX, PDF or CSV)"
        )
        self.action_export_results.triggered.connect(self.export_results)

        self.action_export_ranking = self._action(
            "Export &ranking table…", None, "Export the Auto-fit ranking as CSV"
        )
        self.action_export_ranking.triggered.connect(self.export_ranking)

        self.action_export_publication = self._action(
            "Export plot (&publication)…", None, "Matplotlib figure of the fit (PNG/SVG/PDF)"
        )
        self.action_export_publication.triggered.connect(self.export_publication_plot)

        self.action_about = self._action("&About", None, "Licence and version")
        self.action_about.triggered.connect(self.show_about)

    def _action(
        self,
        text: str,
        shortcut: QKeySequence | None,
        status: str,
        *,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.setStatusTip(status)
        action.setCheckable(checkable)
        return action

    def _build_menus(self) -> None:
        bar: QMenuBar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.action_new)
        file_menu.addAction(self.action_open)
        self._recent_menu = file_menu.addMenu("Open &recent")
        file_menu.addAction(self.action_save)
        file_menu.addAction(self.action_save_as)
        file_menu.addSeparator()
        file_menu.addAction(self.action_import)
        file_menu.addAction(self.action_clipboard)
        file_menu.addSeparator()
        export_menu = file_menu.addMenu("&Export")
        export_menu.addAction(self.action_export_results)
        export_menu.addAction(self.action_export_ranking)
        export_menu.addAction(self.action_export_plot)
        export_menu.addAction(self.action_export_publication)
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        edit_menu = bar.addMenu("&Edit")
        edit_menu.addAction(self.action_undo)

        view_menu = bar.addMenu("&View")
        self._view_menu = view_menu
        for dock, label in (
            (self.project_dock, "&Project tree"),
            (self.analysis_panel, "&Analysis panel"),
            (self.table_dock, "&Data table"),
        ):
            view_menu.addAction(self._dock_toggle(dock, label))
        view_menu.addSeparator()
        view_menu.addAction(self.action_log_x)
        view_menu.addAction(self.action_log_y)
        view_menu.addAction(self.action_reset_view)
        view_menu.addAction(self.action_export_plot)
        view_menu.addAction(self.action_export_publication)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction(self.action_about)

    def _dock_toggle(self, dock: QDockWidget, label: str) -> QAction:
        dock.setWindowTitle(label.strip("&"))
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(dock.isVisible())
        action.toggled.connect(dock.setVisible)
        dock.visibilityChanged.connect(action.setChecked)
        return action

    def _build_toolbar(self) -> None:
        toolbar: QToolBar = self.addToolBar("Main")
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        for action in (
            self.action_new,
            self.action_open,
            self.action_save,
            self.action_import,
            self.action_clipboard,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.action_log_x)
        toolbar.addAction(self.action_log_y)
        toolbar.addSeparator()
        toolbar.addAction(self.action_reset_view)
        toolbar.addAction(self.action_export_plot)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("cancelJobButton")
        self._cancel_button.setToolTip("Stop waiting for the running task")
        self._cancel_button.hide()
        self._cancel_button.clicked.connect(self._cancel_running_job)
        bar.addWidget(self._cancel_button)
        self._observations = QLabel("Observations: 0")
        self._observations.setObjectName("observationsLabel")
        self._no_network = QLabel(NO_NETWORK_NOTE)
        self._no_network.setObjectName("noNetworkLabel")
        bar.addPermanentWidget(self._observations)
        bar.addPermanentWidget(self._no_network)
        bar.showMessage("Ready")

    # -- background jobs -----------------------------------------------------------------------

    def _run_job(
        self,
        label: str,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None],
        on_error: Callable[[BaseException], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        cancellable: bool = False,
    ) -> JobHandle:
        """Run ``fn`` off the GUI thread with a busy cursor and a status message.

        A cancellable job (open, import: side-effect-free reads) also gets the
        status-bar Cancel button; save does not — cancel after the write started
        would not un-write the file.
        """
        suffix = " (Cancel to stop)" if cancellable else ""
        self.statusBar().showMessage(f"{label}…{suffix}")
        if cancellable:
            self._cancel_button.setEnabled(True)
            self._cancel_button.show()
        handle = self.jobs.submit(
            fn,
            *args,
            on_result=on_result,
            on_error=on_error or self._job_failed,
            on_progress=on_progress,
        )
        self._running_job = handle if cancellable else None
        return handle

    def _job_failed(self, error: BaseException) -> None:
        self.controller.messenger.show_error("Task failed", str(error))

    def _cancel_running_job(self) -> None:
        if self._running_job is not None and not self._running_job.done:
            self._running_job.cancel()
            self._cancel_button.setEnabled(False)
            self.statusBar().showMessage("Cancelling…")

    def _on_busy_changed(self, busy: bool) -> None:
        if busy and not self._busy_cursor:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._busy_cursor = True
        elif not busy and self._busy_cursor:
            QApplication.restoreOverrideCursor()
            self._busy_cursor = False
            self._cancel_button.hide()
            self._running_job = None

    def _wire(self) -> None:
        controller = self.controller
        controller.set_state_provider(self.selection.state_for_project)
        self.jobs.busy_changed.connect(self._on_busy_changed)
        controller.project_replaced.connect(self.refresh_all)
        controller.datasets_changed.connect(self._on_datasets_changed)
        controller.dataset_renamed.connect(self._on_dataset_renamed)
        controller.dataset_imported.connect(self._on_dataset_imported)
        controller.analyses_changed.connect(self.refresh_tree)
        controller.dirty_changed.connect(lambda _dirty: self._refresh_title())
        controller.status_message.connect(self.statusBar().showMessage)
        controller.recent_files_changed.connect(self._refresh_recent_menu)
        controller.undo_available_changed.connect(self.action_undo.setEnabled)
        self.project_dock.tree.on_dataset_selected(self._on_tree_selection)
        self.project_dock.tree.on_dataset_menu(self._show_dataset_menu)
        self.project_dock.tree.on_analysis_selected(self._on_tree_analysis)
        self.project_dock.tree.set_analysis_labeller(self.analysis.analysis_label)
        self.selection.active_changed.connect(self.refresh_table)
        self.selection.active_changed.connect(self.refresh_plot)
        self.selection.active_changed.connect(self._sync_tree_selection)
        self.selection.active_changed.connect(self._on_active_dataset_changed)
        analysis = self.analysis
        analysis.status_message.connect(self.statusBar().showMessage)
        analysis.autofit_started.connect(self._on_autofit_started)
        analysis.autofit_live.connect(self._on_autofit_live)
        analysis.autofit_finished.connect(self._on_autofit_result)
        analysis.autofit_failed.connect(self._on_autofit_failed)
        analysis.fit_finished.connect(self._on_fit_shown)
        analysis.fit_failed.connect(self._on_fit_failed)
        panel = self.analysis_panel
        panel.autofit_view.rowSelected.connect(self._plot_candidate)
        panel.autofit_view.openRequested.connect(self._open_candidate)
        self._refresh_recent_menu()

    # -- imports -----------------------------------------------------------------------------

    def import_file(self, path: Path) -> bool:
        """Programmatic import (selftest, drag-drop later, tests): file into project."""
        return self.controller.import_dataset_file(path)

    def open_import_dialog(self) -> None:
        dialog = ImportDialog(parent=self, jobs=self.jobs)
        if dialog.exec():
            outcome = dialog.outcome()  # read after the dialog closes: the live choice
            if outcome is not None:
                self.controller.add_imported(outcome)

    def open_clipboard_dialog(self) -> None:
        dialog = self.make_clipboard_dialog()
        if dialog.exec():
            outcome = dialog.outcome()
            if outcome is not None:
                self.controller.add_imported(outcome)

    # -- async project actions ------------------------------------------------------------------

    def _open_project_async(self) -> None:
        """Open via the file dialog, then load in the background (GUI never freezes)."""
        path = self.controller.paths.choose_open_path()
        if path is None:
            return

        def adopt(project: object) -> None:
            self.controller.adopt_opened(project, path)  # type: ignore[arg-type]

        self._run_job(
            f"Opening {path.name}",
            self.controller.load_only,
            path,
            on_result=adopt,
            on_error=lambda exc: self.controller.messenger.show_error(
                "Cannot open project", str(exc)
            ),
            cancellable=True,  # dropping a loaded project before adopting is safe
        )

    def _save_async(self) -> None:
        if self.controller.path is None:
            self._save_as_async()
            return
        self._submit_save(self.controller.path)

    def _save_as_async(self) -> None:
        suggested = (
            self.controller.path.name
            if self.controller.path
            else f"{self.controller.project.name}.cstudio"
        )
        target = self.controller.paths.choose_save_path(suggested)
        if target is None:
            return
        if target.suffix.lower() != ".cstudio":
            target = target.with_suffix(".cstudio")
        self._submit_save(target)

    def _submit_save(self, target: Path) -> None:
        self._run_job(
            f"Saving {target.name}",
            self.controller.write_only,
            target,
            on_result=lambda project: self.controller.commit_save(project, target),
            on_error=lambda exc: self.controller.messenger.show_error(
                "Cannot save project", str(exc)
            ),
            # not cancellable: a cancel after the write started would not un-write
        )

    def make_clipboard_dialog(self, initial_text: str | None = None) -> ClipboardImportDialog:
        """The clipboard dialog, prefilled from the system clipboard unless given text."""
        if initial_text is None:
            return ClipboardImportDialog.from_clipboard(parent=self, jobs=self.jobs)
        return ClipboardImportDialog(initial_text=initial_text, parent=self, jobs=self.jobs)

    def _on_dataset_imported(self, dataset_id: str, outcome: ImportOutcome) -> None:
        # order matters: store the per-dataset state first (no signals — the new id
        # is not active yet), then make the project known, then activate it. Exactly
        # one active_changed comes out of this, whichever path emitted it.
        if outcome.x_column is not None and outcome.y_column is not None:
            self.selection.set_columns(
                dataset_id, outcome.x_column, outcome.y_column, outcome.weight_column
            )
        self.selection.set_issues(dataset_id, outcome.issues)
        self.selection.set_project(self.controller.project)
        self.selection.set_active(dataset_id)  # show what was just imported

    # -- analysis (D2): routing between the analysis controller and the views ----------------

    def _on_active_dataset_changed(self) -> None:
        """The active dataset (or its columns) moved: drop stale fit overlays, maybe auto-run."""
        self._refresh_analysis_views()

    def _refresh_analysis_views(self) -> None:
        """Point the analysis dock at the active dataset: stored run, empty state, auto-start."""
        panel = self.analysis_panel
        self.analysis.clear_current_fit()
        panel.results.clear_all()
        self.plot_view.clear_fit()
        panel.manual_view.refresh_data_options()
        dataset_id = self.selection.active_id
        if dataset_id is None:
            panel.autofit_view.clear()
            return
        result = self.analysis.stored_autofit(dataset_id)
        if result is not None and self.analysis.has_current_autofit(dataset_id):
            self.analysis.showing_stored_autofit(dataset_id, result)
            panel.autofit_view.show_result(result)
        else:
            self.analysis.showing_stored_autofit(dataset_id, None)
            panel.autofit_view.clear()
        if not self._suppress_autostart:
            self.analysis.maybe_autostart(dataset_id)

    def _on_autofit_started(self, dataset_id: str) -> None:
        if dataset_id == self.selection.active_id:
            self.analysis_panel.autofit_view.run_started(dataset_id)

    def _on_autofit_live(self, live: object) -> None:
        if self.analysis.running_dataset == self.selection.active_id:
            self.analysis_panel.autofit_view.show_live(cast("AutoFitLive", live))

    def _on_autofit_result(self, dataset_id: str, result: object) -> None:
        if dataset_id == self.selection.active_id:
            self.analysis_panel.autofit_view.show_result(cast("AutoFitResult", result))

    def _on_autofit_failed(self, message: str) -> None:
        # inline notice, not a modal box: an auto-started run the user never asked for
        # must not interrupt them (the status bar carries the same text)
        self.analysis_panel.autofit_view.show_failure(message)

    def _on_fit_shown(self, bundle: object, spec: object, meta: object) -> None:
        """A fit (fresh or restored) fills the tabs and draws its curve — if it has one."""
        fit_bundle = cast("FitBundle", bundle)
        panel = self.analysis_panel
        panel.results.show_bundle(fit_bundle, cast("FitSpec", spec), meta)
        panel.manual_view.clear_banner()
        curve = fit_bundle.curve
        if curve is None:
            self.plot_view.clear_fit()  # never a stand-in curve for a fit without one
            return
        confidence = fit_bundle.confidence_band
        prediction = fit_bundle.prediction_band
        self.plot_view.set_fit(
            curve.x,
            curve.y,
            (confidence.lower, confidence.upper) if confidence is not None else None,
            (prediction.lower, prediction.upper) if prediction is not None else None,
        )

    def _on_fit_failed(self, message: str) -> None:
        self.analysis_panel.manual_view.show_failure(message)

    def _plot_candidate(self, candidate_id: str) -> None:
        """A ranking row was selected: draw its stored curve (or none, honestly)."""
        dataset_id = self.selection.active_id
        result = None if dataset_id is None else self.analysis.autofit_for_display(dataset_id)
        if result is None:
            return
        curve = result.curves.get(candidate_id)
        if curve is None:
            self.plot_view.clear_fit()
            self.plot_view.show_hint("curve not sampled — Open full results")
            self.statusBar().showMessage(
                "no sampled curve for this row: use Open full results to refit it"
            )
            return
        confidence = (
            (curve.confidence_lower, curve.confidence_upper)
            if curve.confidence_lower is not None and curve.confidence_upper is not None
            else None
        )
        prediction = (
            (curve.prediction_lower, curve.prediction_upper)
            if curve.prediction_lower is not None and curve.prediction_upper is not None
            else None
        )
        self.plot_view.set_fit(curve.x, curve.y, confidence, prediction)

    def _open_candidate(self, candidate_id: str) -> None:
        dataset_id = self.selection.active_id
        result = None if dataset_id is None else self.analysis.autofit_for_display(dataset_id)
        if dataset_id is None or result is None:
            return
        self.analysis.open_candidate(dataset_id, result, candidate_id)

    def _on_tree_analysis(self, analysis_id: str) -> None:
        """An analysis row was clicked: activate its dataset, then restore without refitting."""
        dataset_id = self.analysis.dataset_of_analysis(analysis_id)
        self._suppress_autostart = True
        try:
            if dataset_id is not None:
                self.selection.set_active(dataset_id)
        finally:
            self._suppress_autostart = False
        self.analysis.restore_analysis(analysis_id)

    # -- refresh -----------------------------------------------------------------------------

    def refresh_all(self) -> None:
        """Full rebuild: a different project is current (new, open, replace)."""
        self.analysis.reset_for_project()
        if not self.selection.set_project(self.controller.project):
            # reconciliation did not move the active dataset (for example an empty
            # project) — the table and plot still must show the new reality
            self.refresh_table()
            self.refresh_plot()
            self._refresh_analysis_views()
        self.refresh_tree()
        self._refresh_title()
        self._refresh_recent_menu()

    def _on_datasets_changed(self) -> None:
        """Datasets were added/removed: reconcile the selection, rebuild the tree.

        ``set_project`` only emits when the active dataset actually moved (for
        example the active dataset was removed), so the table and plot refresh at
        most once — an import's refresh comes from ``set_active`` instead.
        """
        self.selection.set_project(self.controller.project)
        self.refresh_tree()

    def _on_dataset_renamed(self, dataset_id: str) -> None:
        entry = next((e for e in self.controller.project.datasets if e.id == dataset_id), None)
        if entry is not None:
            self.project_dock.tree.update_label(dataset_id, entry.name)

    def refresh_tree(self) -> None:
        self.project_dock.tree.refresh(self.controller.project, self.selection.active_id)

    def refresh_table(self) -> None:
        dataset = self.selection.active_dataset()
        active_id = self.selection.active_id
        if dataset is None or active_id is None:
            self.table_dock.set_model(None)
            self._observations.setText("Observations: 0")
            return
        columns = self.selection.columns(active_id)
        if columns is None:
            frame = dataset.table()
            names = [str(name) for name in frame.columns]
            x_name = names[0] if names else "x"
            y_name = names[1] if len(names) > 1 else x_name
            columns = (x_name, y_name, None)
        model = DatasetTableModel(
            dataset, columns[0], columns[1], columns[2], self.selection.issues(active_id)
        )
        self.table_dock.set_model(model)
        self._observations.setText(f"Observations: {model.rowCount():,}")

    def refresh_plot(self) -> None:
        xy = self.selection.xy()
        if xy is None:
            self.plot_view.clear_data()
            if self.selection.last_error:
                self.statusBar().showMessage(self.selection.last_error)
        else:
            self.plot_view.set_data(xy.x, xy.y, xy.x_name, xy.y_name)
        self._refresh_axis_actions()

    def _refresh_title(self) -> None:
        name = self.controller.project.name
        dirty = " *" if self.controller.is_dirty else ""
        self.setWindowTitle(f"{APP_NAME} — {name}{dirty}")

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        recents = self.controller.recent_files()
        self._recent_menu.setEnabled(bool(recents))
        for path in recents[:_RECENT_LIMIT]:
            action = self._recent_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda *_, target=path: self.controller.open_project(target))

    def _refresh_axis_actions(self) -> None:
        self.action_log_x.setEnabled(self.plot_view.can_log_x)
        self.action_log_y.setEnabled(self.plot_view.can_log_y)
        if not self.plot_view.can_log_x:
            self.action_log_x.setChecked(False)
            self.action_log_x.setToolTip("Log axis needs strictly positive x values")
        else:
            self.action_log_x.setToolTip("")
        if not self.plot_view.can_log_y:
            self.action_log_y.setChecked(False)
            self.action_log_y.setToolTip("Log axis needs strictly positive y values")
        else:
            self.action_log_y.setToolTip("")

    def _on_tree_selection(self, dataset_id: str | None) -> None:
        if dataset_id is not None:
            self.selection.set_active(dataset_id)

    def _sync_tree_selection(self) -> None:
        """The tree always highlights the selection controller's active dataset."""
        self.project_dock.tree.select_active(self.selection.active_id)

    # -- project-tree context menu -------------------------------------------------------------

    def _show_dataset_menu(self, dataset_id: str, global_position: QPoint) -> None:
        # popup (not exec): a context menu is non-modal, and popup-at-position keeps
        # it exactly under the cursor
        self.dataset_context_menu(dataset_id).popup(global_position)

    def dataset_context_menu(self, dataset_id: str) -> QMenu:
        """Right-click menu for a dataset row: rename and remove (undo exists)."""
        menu = QMenu(self)
        menu.addAction("Rename…", lambda: self.rename_dataset_interactive(dataset_id))
        menu.addAction("Remove", lambda: self.remove_dataset(dataset_id))
        return menu

    def rename_dataset_interactive(self, dataset_id: str) -> None:
        current = ""
        entry = next((e for e in self.controller.project.datasets if e.id == dataset_id), None)
        if entry is not None:
            current = entry.name
        name, confirmed = QInputDialog.getText(
            self, "Rename dataset", "New name:", QLineEdit.EchoMode.Normal, current
        )
        if confirmed:
            self.rename_dataset(dataset_id, name)

    def rename_dataset(self, dataset_id: str, new_name: str) -> bool:
        return self.controller.rename_dataset(dataset_id, new_name)

    def remove_dataset(self, dataset_id: str) -> bool:
        return self.controller.remove_dataset(dataset_id)

    # -- layout persistence ----------------------------------------------------------------------

    def _restore_layout(self) -> None:
        """Dock layout and geometry from QSettings, when a previous session left them."""
        state = self._settings.value(_LAYOUT_STATE_KEY, "")
        if isinstance(state, str) and state:
            self.layout_restored = bool(
                self.restoreState(QByteArray.fromBase64(bytes(state, encoding="ascii")))
            )
        geometry = self._settings.value(_LAYOUT_GEOMETRY_KEY, "")
        if isinstance(geometry, str) and geometry:
            self.restoreGeometry(QByteArray.fromBase64(bytes(geometry, encoding="ascii")))
        self.analysis_panel.restore_splitter(self._settings)  # the ranking/tabs split

    def _save_layout(self) -> None:
        self._settings.setValue(
            _LAYOUT_STATE_KEY, bytes(self.saveState().toBase64().data()).decode("ascii")
        )
        self._settings.setValue(
            _LAYOUT_GEOMETRY_KEY, bytes(self.saveGeometry().toBase64().data()).decode("ascii")
        )
        self.analysis_panel.save_splitter(self._settings)
        self._settings.sync()

    # -- actions without a controller home ---------------------------------------------------

    def export_results(self) -> None:
        """Export the current fit as a report (JSON / XLSX / PDF / CSV by suffix)."""
        bundle = self.analysis.current_bundle
        if bundle is None:
            self.controller.messenger.show_information(
                "Export results", "Run or open a fit first; there is nothing to export yet."
            )
            return
        target = self.controller.paths.choose_export_path(
            "Export results", "CurveStudio report (*.json *.xlsx *.pdf *.csv)"
        )
        if target is None:
            return
        self._run_job(
            f"Exporting {target.name}",
            exports.export_fit,
            target,
            bundle,
            on_result=lambda _done: self.statusBar().showMessage(f"Exported {target.name}"),
            on_error=lambda exc: self.controller.messenger.show_error("Export failed", str(exc)),
        )

    def export_ranking(self) -> None:
        """Export the active dataset's Auto-fit ranking table as formula-safe CSV."""
        dataset_id = self.selection.active_id
        result = None if dataset_id is None else self.analysis.stored_autofit(dataset_id)
        if result is None:
            self.controller.messenger.show_information(
                "Export ranking table", "Run Auto-fit first; there is no ranking to export."
            )
            return
        target = self.controller.paths.choose_export_path(
            "Export ranking table", "CSV table (*.csv)"
        )
        if target is None:
            return
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        self._run_job(
            f"Exporting {target.name}",
            exports.export_ranking_csv,
            target,
            result,
            self.analysis.rank_by,
            on_result=lambda _done: self.statusBar().showMessage(f"Exported {target.name}"),
            on_error=lambda exc: self.controller.messenger.show_error("Export failed", str(exc)),
        )

    def export_publication_plot(self) -> None:
        """Export the current fit as the Matplotlib publication figure (PNG/SVG/PDF)."""
        bundle = self.analysis.current_bundle
        if bundle is None:
            self.controller.messenger.show_information(
                "Export plot", "Run or open a fit first; there is nothing to export yet."
            )
            return
        target = self.controller.paths.choose_export_path(
            "Export plot (publication)", "PNG image (*.png);;SVG image (*.svg);;PDF report (*.pdf)"
        )
        if target is None:
            return
        self._run_job(
            f"Exporting {target.name}",
            exports.export_publication_plot,
            target,
            bundle,
            on_result=lambda _done: self.statusBar().showMessage(f"Exported {target.name}"),
            on_error=lambda exc: self.controller.messenger.show_error("Export failed", str(exc)),
        )

    def export_plot(self) -> None:
        """Export the current plot via the export dialog; PNG or SVG by suffix."""
        target = self.controller.paths.choose_export_path(
            "Export plot", "PNG image (*.png);;SVG image (*.svg)"
        )
        if target is None:
            return
        suffix = target.suffix.lower()
        if suffix == ".svg":
            done = self.plot_view.export_svg(target)
        else:
            done = self.plot_view.export_png(target)
        if not done:
            self.controller.messenger.show_information(
                "Export plot", "There is nothing plotted to export yet."
            )

    def show_about(self) -> None:
        from curvestudio_desktop.views.dialogs import AboutDialog

        AboutDialog(self).exec()

    # -- Qt events ---------------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller.maybe_quit():
            self._save_layout()
            event.accept()
        else:
            event.ignore()
