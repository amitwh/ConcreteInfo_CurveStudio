"""MainWindow: menu bar, toolbar, status bar and the four dock areas (PRD §16.1).

The window is a thin shell: every action delegates to the ProjectController, and
every display refresh is driven by controller signals. The plot controls (log
axes, reset, export) delegate to PlotView. Nothing here computes anything.
"""

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from PySide6.QtCore import QByteArray, QPoint, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QTableView,
    QToolBar,
    QWidget,
)

from curvestudio.discovery import AutoFitResult, RankedFit, RowCurve
from curvestudio.tools import CrossValidation
from curvestudio.workflow import FitBundle, FitSpec, ImportOutcome
from curvestudio_desktop import APP_NAME, NO_NETWORK_NOTE
from curvestudio_desktop.controllers import analysis_jobs, exports
from curvestudio_desktop.controllers.analysis import MODE_AUTOFIT, AnalysisController
from curvestudio_desktop.controllers.analysis_jobs import AutoFitLive
from curvestudio_desktop.controllers.project import ProjectController
from curvestudio_desktop.controllers.selection import SelectionController
from curvestudio_desktop.jobs import JobHandle, JobRunner
from curvestudio_desktop.profile import report_identity
from curvestudio_desktop.theme import (
    THEME_CHOICES,
    Preference,
    Scheme,
    apply_theme,
    set_theme_preference,
    theme_preference,
)
from curvestudio_desktop.views.analysis_panel import AnalysisPanel
from curvestudio_desktop.views.data_table import DatasetTableModel, DataTableDock
from curvestudio_desktop.views.details_dialog import DetailsDialog
from curvestudio_desktop.views.dock_shell import DockTitleBar
from curvestudio_desktop.views.fit_summary import model_identity
from curvestudio_desktop.views.icons import app_icon, icon
from curvestudio_desktop.views.import_dialog import ClipboardImportDialog, ImportDialog
from curvestudio_desktop.views.plot_view import OverlayCurve, PlotView
from curvestudio_desktop.views.preferences_dialog import PreferencesDialog
from curvestudio_desktop.views.project_tree import ProjectDock, ProjectTree

if TYPE_CHECKING:
    from curvestudio_desktop.views.help_dialog import HelpDialog
    from curvestudio_desktop.windows import WindowRegistry

MINIMUM_SIZE = (980, 640)
DEFAULT_WINDOW_SIZE = (1280, 820)
ANALYSIS_DOCK_WIDTH = 420
"""The right dock starts wide: fit parameters and statistics need the room."""
_RECENT_LIMIT = 10
_LAYOUT_STATE_KEY = "main_window/state"
_LAYOUT_GEOMETRY_KEY = "main_window/geometry"
_COLLAPSED_PREFIX = "main_window/collapsed/"
_EXPANDED_PREFIX = "main_window/expanded/"
_SHOW_TOP3_KEY = "view/showTop3"
_COLLAPSED_EXTENT = 40
_UNBOUNDED = 16777215  # QWIDGETSIZE_MAX
#: Where each panel belongs. View → Panels re-homes a panel here, so a panel that was
#: floated, closed or docked into a neighbour's slot always comes back to its own place.
_DOCK_HOME: dict[str, Qt.DockWidgetArea] = {
    "project": Qt.DockWidgetArea.LeftDockWidgetArea,
    "analysis": Qt.DockWidgetArea.RightDockWidgetArea,
    "table": Qt.DockWidgetArea.BottomDockWidgetArea,
}
#: The axis and default extent a re-shown panel is given.
_DOCK_EXTENT: dict[str, tuple[Qt.Orientation, int]] = {
    "project": (Qt.Orientation.Horizontal, 256),
    "analysis": (Qt.Orientation.Horizontal, ANALYSIS_DOCK_WIDTH),
    "table": (Qt.Orientation.Vertical, 220),
}


class MainWindow(QMainWindow):
    """The shell: left project tree, centre plot, right analysis, bottom data table."""

    closed = Signal()
    """The window has closed for good (the registry drops it)."""

    def __init__(
        self,
        controller: ProjectController,
        parent: QWidget | None = None,
        jobs: JobRunner | None = None,
        settings: QSettings | None = None,
        registry: WindowRegistry | None = None,
        persist_layout: bool = True,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        # File → New/Open project opens another window when a registry is wired (the
        # running application); a bare window keeps the single-project behaviour.
        self.registry = registry
        self.persist_layout = persist_layout
        self.selection = SelectionController(self)
        self.jobs = jobs if jobs is not None else JobRunner(self)
        self._settings = settings if settings is not None else QSettings()
        self._running_job: JobHandle | None = None
        self._busy_cursor = False
        self._suppress_autostart = False
        self.layout_restored = False
        self._theme_preference = theme_preference(self._settings)
        self._collapsed: dict[str, bool] = {}
        self._expanded_extent: dict[str, int] = {}
        self._title_bars: dict[str, DockTitleBar] = {}
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setObjectName("mainWindow")

        self.analysis = AnalysisController(
            controller, self.selection, settings=self._settings, parent=self
        )

        self.plot_view = PlotView(self)
        self.setCentralWidget(self.plot_view)
        # one reusable, non-modal details window per main window (never a stack of them)
        self.details_dialog = DetailsDialog(
            self,
            settings=self._settings,
            predict_fn=self.analysis.predict,
            cross_validate_fn=self._cross_validate,
        )
        # one reusable, non-modal help window per main window
        self.help_dialog: HelpDialog | None = None

        self.project_dock = ProjectDock(self)
        self.analysis_panel = AnalysisPanel(self.analysis, self.selection, self)
        self.table_dock = DataTableDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.project_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.table_dock)
        self.resizeDocks((self.analysis_panel,), (ANALYSIS_DOCK_WIDTH,), Qt.Orientation.Horizontal)
        self.resizeDocks((self.table_dock,), (220,), Qt.Orientation.Vertical)
        self._setup_docks()

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._apply_icons()
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

    @property
    def docks(self) -> dict[str, QDockWidget]:
        """The three panels by their persisted name."""
        return {
            "project": self.project_dock,
            "analysis": self.analysis_panel,
            "table": self.table_dock,
        }

    # -- construction ------------------------------------------------------------------------

    def _setup_docks(self) -> None:
        """Give every dock its custom title bar and the standard float/close features."""
        titles = {"project": "Project", "analysis": "Analysis", "table": "Data table"}
        for name, dock in self.docks.items():
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            bar = DockTitleBar(titles[name], dock, vertical=name != "table")
            dock.setTitleBarWidget(bar)
            bar.collapseToggled.connect(
                lambda collapsed, dock_name=name: self.set_dock_collapsed(dock_name, collapsed)
            )
            self._title_bars[name] = bar
            self._collapsed[name] = False

    def _build_actions(self) -> None:
        self.action_new = self._action(
            "&New project", QKeySequence(QKeySequence.StandardKey.New), "Start a clean project"
        )
        self.action_new.triggered.connect(self._new_project)

        self.action_open = self._action(
            "&Open project…", QKeySequence(QKeySequence.StandardKey.Open), "Open a .cstudio file"
        )
        self.action_open.triggered.connect(self._open_project)

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

        self.action_preferences = self._action(
            "Preferences…", QKeySequence("Ctrl+,"), "Report byline and theme"
        )
        self.action_preferences.triggered.connect(self.open_preferences)

        self.action_log_x = self._action("&Log x-axis", None, "Logarithmic x-axis", checkable=True)
        self.action_log_x.toggled.connect(self.plot_view.set_log_x)
        self.action_log_y = self._action("Log &y-axis", None, "Logarithmic y-axis", checkable=True)
        self.action_log_y.toggled.connect(self.plot_view.set_log_y)

        self.action_reset_view = self._action(
            "&Reset view", QKeySequence("Ctrl+R"), "Fit the plot to the data"
        )
        self.action_reset_view.triggered.connect(self.plot_view.reset_view)

        self.action_details = self._action(
            "&Details…",
            QKeySequence("Ctrl+D"),
            "Open the selected row's or the active fit's full results",
        )
        self.action_details.triggered.connect(self.open_details)

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

        self.action_help_contents = self._action(
            "&Help contents", QKeySequence("F1"), "Open the help browser"
        )
        self.action_help_contents.triggered.connect(lambda: self.open_help())
        self.action_shortcuts = self._action(
            "&Keyboard shortcuts", None, "Open the help browser at the shortcut list"
        )
        self.action_shortcuts.triggered.connect(lambda: self.open_help("Keyboard shortcuts"))

        self.action_about = self._action("&About", None, "Licence and version")
        self.action_about.triggered.connect(self.show_about)

        self.action_about_qt = self._action("About &Qt", None, "Qt version and licence")
        self.action_about_qt.triggered.connect(self.show_about_qt)

        self.action_reset_layout = self._action(
            "&Reset layout", None, "Restore the default panel arrangement"
        )
        self.action_reset_layout.triggered.connect(self._reset_layout)

        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for preference, label in THEME_CHOICES:
            theme_action = self._action(label, None, f"Use the {preference} theme")
            theme_action.setCheckable(True)
            theme_action.setChecked(preference == self._theme_preference)
            theme_action.triggered.connect(
                lambda _checked=False, choice=preference: self.apply_theme_preference(choice)
            )
            self._theme_group.addAction(theme_action)
            self._theme_actions[preference] = theme_action

        self._icon_actions: list[tuple[QAction, str]] = [
            (self.action_new, "new"),
            (self.action_open, "open"),
            (self.action_save, "save"),
            (self.action_import, "import"),
            (self.action_clipboard, "clipboard"),
            (self.action_undo, "undo"),
            (self.action_log_x, "log-x"),
            (self.action_log_y, "log-y"),
            (self.action_reset_view, "reset"),
            (self.action_details, "details"),
            (self.action_export_plot, "export"),
            (self.action_export_results, "export"),
            (self.action_export_ranking, "export"),
            (self.action_export_publication, "export"),
            (self.action_help_contents, "help"),
            (self.action_about, "about"),
            (self.action_about_qt, "about"),
            (self.action_reset_layout, "reset"),
        ]

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
        edit_menu.addSeparator()
        edit_menu.addAction(self.action_preferences)

        view_menu = bar.addMenu("&View")
        self._view_menu = view_menu
        panels_menu = view_menu.addMenu("&Panels")
        self._panel_actions: dict[str, QAction] = {}
        self._panel_icon_names = {
            "project": "panel-project",
            "analysis": "panel-analysis",
            "table": "panel-table",
        }
        for name, label in (
            ("project", "&Project tree"),
            ("analysis", "&Analysis panel"),
            ("table", "&Data table"),
        ):
            action = self._dock_toggle(name, label)
            panels_menu.addAction(action)
            self._panel_actions[name] = action
        view_menu.addSeparator()
        view_menu.addAction(self.action_log_x)
        view_menu.addAction(self.action_log_y)
        view_menu.addAction(self.action_reset_view)
        view_menu.addAction(self.action_details)
        view_menu.addAction(self.action_export_plot)
        view_menu.addAction(self.action_export_publication)
        view_menu.addSeparator()
        view_menu.addAction(self.action_reset_layout)
        theme_menu = view_menu.addMenu("&Theme")
        for preference, _label in THEME_CHOICES:
            theme_menu.addAction(self._theme_actions[preference])

        self.help_menu = bar.addMenu("&Help")
        self.help_menu.addAction(self.action_help_contents)
        self.help_menu.addAction(self.action_shortcuts)
        self.help_menu.addSeparator()
        self.help_menu.addAction(self.action_about)
        self.help_menu.addAction(self.action_about_qt)

    def _dock_toggle(self, name: str, label: str) -> QAction:
        """A View → Panels entry: show the panel in its home area, or hide it."""
        dock = self.docks[name]
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(dock.isVisible())
        action.toggled.connect(
            lambda visible, dock_name=name: self.set_dock_visible(dock_name, visible)
        )
        # The menu mirrors the dock, but that mirror must not loop back into
        # set_dock_visible: floating a panel briefly hides it, and an unguarded
        # setChecked -> toggled -> set_dock_visible would dock it again instantly.
        dock.visibilityChanged.connect(
            lambda visible, dock_name=name: self._sync_panel_action(dock_name, visible)
        )
        return action

    def _sync_panel_action(self, name: str, visible: bool) -> None:
        action = self._panel_actions.get(name)
        if action is None or action.isChecked() == visible:
            return
        blocked = action.blockSignals(True)
        try:
            action.setChecked(visible)
        finally:
            action.blockSignals(blocked)

    def set_dock_visible(self, name: str, visible: bool) -> None:
        """Show a panel where it belongs, or hide it.

        Re-showing never leaves a panel floating at a stale spot (an old monitor, a
        corner behind another window) and never drops it into a neighbour's slot: a
        hidden or floated panel is re-docked to its home area and given its remembered
        extent. That is what makes View → Panels a reliable way back to a sane layout.
        """
        dock = self.docks[name]
        if not visible:
            dock.hide()
            return
        if dock.isFloating():
            dock.setFloating(False)
        home = _DOCK_HOME[name]
        if self.dockWidgetArea(dock) != home:
            self.addDockWidget(home, dock)
        dock.show()
        dock.raise_()
        if self._collapsed.get(name, False):
            return  # a collapsed panel keeps its strip; Expand is on its title bar
        orientation, default = _DOCK_EXTENT[name]
        if orientation is Qt.Orientation.Horizontal:
            dock.setMaximumWidth(_UNBOUNDED)
        else:
            dock.setMaximumHeight(_UNBOUNDED)
        self.resizeDocks([dock], [self._expanded_extent.get(name, default)], orientation)

    def _build_toolbar(self) -> None:
        toolbar: QToolBar = self.addToolBar("Main")
        self._toolbar = toolbar
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(True)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setIconSize(QSize(18, 18))
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

    def _apply_icons(self) -> None:
        """(Re-)render every icon and shortcut tooltip for the current theme."""
        for action, icon_name in self._icon_actions:
            action.setIcon(icon(icon_name))
            tooltip = action.statusTip() or action.text().replace("&", "")
            shortcut = action.shortcut().toString()
            action.setToolTip(f"{tooltip} ({shortcut})" if shortcut else tooltip)
        for name, action in self._panel_actions.items():
            action.setIcon(icon(self._panel_icon_names[name]))
        for bar in self._title_bars.values():
            bar.refresh_icons()

    def apply_theme_preference(self, preference: Preference, *, persist: bool = True) -> None:
        """Switch the light/dark/system theme immediately and remember the choice."""
        app = QApplication.instance()
        resolved: Scheme = "light"
        if app is not None:
            resolved = apply_theme(cast("QApplication", app), preference=preference)
        if persist:
            set_theme_preference(preference, self._settings)
        self._theme_preference = preference
        for name, action in self._theme_actions.items():
            action.setChecked(name == preference)
        self._apply_icons()
        self.plot_view.apply_scheme(resolved)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("cancelJobButton")
        self._cancel_button.setToolTip("Stop waiting for the running task")
        self._cancel_button.hide()
        self._cancel_button.clicked.connect(self._cancel_running_job)
        bar.addWidget(self._cancel_button)
        self._job_progress = QProgressBar()
        self._job_progress.setObjectName("jobProgress")
        self._job_progress.setFixedWidth(120)
        self._job_progress.setTextVisible(False)
        self._job_progress.setRange(0, 0)  # indeterminate until a job reports counts
        self._job_progress.hide()
        self._observations = QLabel("Observations: 0")
        self._observations.setObjectName("observationsLabel")
        self._no_network = QLabel(NO_NETWORK_NOTE)
        self._no_network.setObjectName("noNetworkLabel")
        bar.addPermanentWidget(self._job_progress)
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
        on_settled: Callable[[], None] | None = None,
    ) -> JobHandle:
        """Run ``fn`` off the GUI thread with a busy cursor and a status message.

        A cancellable job (open, import: side-effect-free reads) also gets the
        status-bar Cancel button; save does not — cancel after the write started
        would not un-write the file. ``on_settled`` is delivered when the job has
        ended, a cancelled one included, whose result and error are dropped.
        """
        suffix = " (Cancel to stop)" if cancellable else ""
        self.statusBar().showMessage(f"{label}…{suffix}")
        if cancellable:
            self._cancel_button.setEnabled(True)
            self._cancel_button.show()

        def progress(done: int, total: int) -> None:
            self._show_job_progress(done, total)
            if on_progress is not None:
                on_progress(done, total)

        handle = self.jobs.submit(
            fn,
            *args,
            on_result=on_result,
            on_error=on_error or self._job_failed,
            on_progress=progress if on_progress is not None else None,
            on_settled=on_settled,
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
            self._job_progress.setRange(0, 0)
            self._job_progress.show()
        elif not busy and self._busy_cursor:
            QApplication.restoreOverrideCursor()
            self._busy_cursor = False
            self._cancel_button.hide()
            self._running_job = None
            self._job_progress.hide()
            self._job_progress.setRange(0, 0)
            self._job_progress.setValue(0)

    def _show_job_progress(self, done: int, total: int) -> None:
        """Feed the status-bar bar when a job reports determinate progress."""
        if total > 0:
            self._job_progress.setRange(0, total)
            self._job_progress.setValue(min(done, total))
        self._job_progress.show()

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
        panel.autofit_view.top3Changed.connect(self._on_show_top3_changed)
        panel.summary.openDetailsRequested.connect(self.open_details)
        panel.summary.exportRequested.connect(self.export_results)
        self.details_dialog.exportRequested.connect(self.export_results)
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

    def _new_project(self) -> None:
        """File → New project: a new window, so the project on screen is never replaced."""
        if self.registry is not None:
            self.registry.spawn()
        else:
            self.controller.new_project()

    def _open_project(self) -> None:
        """File → Open project…: another window when the registry is wired."""
        if self.registry is not None:
            self.registry.open_from(self)
        else:
            self._open_project_async()

    def _open_recent(self, path: Path) -> None:
        """A recent project opens in its own window when the registry is wired."""
        if self.registry is not None:
            self.registry.spawn(path)
        else:
            self.controller.open_project(path)

    def open_project_path(self, path: Path) -> None:
        """Load ``path`` into this window (the registry opens it here, not in the caller)."""
        self._open_project_async(path)

    def _open_project_async(self, path: Path | None = None) -> None:
        """Open via the file dialog (unless a path is given), then load in the background."""
        if path is None:
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
        panel.summary.clear()
        self.details_dialog.clear()
        self.plot_view.clear_fit()
        panel.manual_view.refresh_data_options()
        dataset_id = self.selection.active_id
        if dataset_id is None:
            panel.autofit_view.clear()
            self._refresh_export_actions()
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
        self._refresh_export_actions()

    def _on_autofit_started(self, dataset_id: str) -> None:
        if dataset_id == self.selection.active_id:
            self.analysis_panel.autofit_view.run_started(dataset_id)

    def _on_autofit_live(self, live: object) -> None:
        if self.analysis.running_dataset == self.selection.active_id:
            self.analysis_panel.autofit_view.show_live(cast("AutoFitLive", live))

    def _on_autofit_result(self, dataset_id: str, result: object) -> None:
        if dataset_id == self.selection.active_id:
            auto_result = cast("AutoFitResult", result)
            self.analysis_panel.autofit_view.show_result(auto_result)
            if not auto_result.ranked:
                # nothing to summarise: the hint, not a stale fit from another run
                self.analysis_panel.summary.clear()
        self._refresh_export_actions()

    def _on_autofit_failed(self, message: str) -> None:
        # inline notice, not a modal box: an auto-started run the user never asked for
        # must not interrupt them (the status bar carries the same text)
        self.analysis_panel.autofit_view.show_failure(message)
        self._refresh_export_actions()

    def _on_fit_shown(self, bundle: object, spec: object, meta: object) -> None:
        """A fit (fresh or restored) updates the summary/details window and draws its curve."""
        fit_bundle = cast("FitBundle", bundle)
        fit_spec = cast("FitSpec", spec)
        meta_dict = meta if isinstance(meta, dict) else {}
        dataset_name = self._dataset_name(meta_dict.get("dataset_id"))
        row = self._row_for_model(fit_bundle.result.model_id)
        if row is not None:
            name, category = row.name, row.category
        else:
            name, category = model_identity(fit_bundle.result.model_id)
        panel = self.analysis_panel
        panel.summary.show_bundle(
            fit_bundle, dataset_name=dataset_name, name=name, category=category
        )
        self.details_dialog.show_bundle(
            fit_bundle,
            fit_spec,
            dataset_name=dataset_name,
            name=name,
            category=category,
            meta=meta_dict,
            ranking=self._ranking_of(fit_meta=meta_dict),
        )
        panel.manual_view.clear_banner()
        self._refresh_export_actions()
        # a fit the user asked for opens the details window; restoring one does not
        if not meta_dict.get("restored", False):
            self.details_dialog.reveal()
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
        fitted = fit_bundle.result.equation_fitted or fit_bundle.result.equation
        if fitted:
            self.plot_view.set_equation(name, fitted)

    def _on_fit_failed(self, message: str) -> None:
        self.analysis_panel.manual_view.show_failure(message)
        self._refresh_export_actions()

    def _ranking_of(self, fit_meta: dict[str, Any]) -> AutoFitResult | None:
        """The Auto-fit run a fit opened from one of its rows came from; ``None`` for any other fit.

        It is the run the ranking table holds for that dataset — the one the row was clicked
        in — and it is what the fit audit's simpler-model check compares against.
        """
        source = fit_meta.get("source")
        dataset_id = fit_meta.get("dataset_id")
        if not isinstance(source, dict) or source.get("kind") != "autofit":
            return None
        if not isinstance(dataset_id, str):
            return None
        return self.analysis.autofit_for_display(dataset_id)

    def _cross_validate(
        self,
        bundle: FitBundle,
        spec: FitSpec,
        folds: int | Literal["loo"],
        *,
        on_result: Callable[[CrossValidation], None],
        on_error: Callable[[BaseException], None],
        on_settled: Callable[[], None],
    ) -> None:
        """The fit audit's cross-validation, run as a job the status bar can cancel.

        The audit tab cannot reach the window, so the window hands it this callable when it
        builds the details dialog; it is how the tab reaches the job machinery, as the
        opening of an Auto-fit row reaches it. A refusal is the tab's to show, inline.
        """
        status = self.statusBar()

        def finished(result: CrossValidation) -> None:
            on_result(result)
            status.showMessage("Cross-validation finished")

        def refused(error: BaseException) -> None:
            on_error(error)
            status.showMessage(f"Cross-validation could not run: {error}")

        self._run_job(
            "Cross-validating",
            analysis_jobs.cross_validate_job,
            bundle,
            spec,
            folds,
            on_result=finished,
            on_error=refused,
            on_settled=on_settled,
            cancellable=True,
        )

    def _plot_candidate(self, candidate_id: str) -> None:
        """A ranking row was selected: draw its stored curve and mirror it in the summary.

        With "Show top 3" on, the three best-ranked stored curves are overlaid in their
        rank colours (the selected row's own curve stays visible even when it ranks below
        them); the selected row's bands alone are drawn, and neutrally. Selecting another
        row recolours and relabels the overlay without refitting anything.
        """
        dataset_id = self.selection.active_id
        result = None if dataset_id is None else self.analysis.autofit_for_display(dataset_id)
        if result is None:
            return
        self._show_row_summary(dataset_id)
        rows = self.analysis_panel.autofit_view.table.rows()
        row = next((candidate for candidate in rows if candidate.model_id == candidate_id), None)
        # Colour by the 1-based display position, not the tied rank value.
        position = next(
            (
                index + 1
                for index, candidate in enumerate(rows)
                if candidate.model_id == candidate_id
            ),
            1,
        )
        curve = result.curves.get(candidate_id)
        show_top3 = self.analysis_panel.autofit_view.show_top3()
        overlay = self._overlay_curves(result, rows, candidate_id) if show_top3 else []
        if curve is None and not overlay:
            self.plot_view.clear_fit()
            self.plot_view.show_hint("curve not sampled — Open full results")
            self.statusBar().showMessage(
                "no sampled curve for this row: use Open full results to refit it"
            )
            return
        confidence, prediction = self._bands(curve)
        if overlay:
            self.plot_view.set_fit_overlay(
                overlay,
                band_x=curve.x if curve is not None else None,
                confidence=confidence,
                prediction=prediction,
            )
            self._show_equation(row, position)
            return
        assert curve is not None  # no overlay means the selected curve exists (checked above)
        self.plot_view.set_fit(
            curve.x,
            curve.y,
            confidence,
            prediction,
            position=position,
            name=row.name if row is not None else "fitted curve",
        )
        self._show_equation(row, position)

    def _show_equation(self, row: RankedFit | None, position: int) -> None:
        """Caption the plot with the selected row's fitted equation, in its curve colour."""
        if row is None or not row.equation_fitted:
            self.plot_view.clear_equation()
            return
        self.plot_view.set_equation(row.name, row.equation_fitted, position=position)

    @staticmethod
    def _bands(
        curve: RowCurve | None,
    ) -> tuple[
        tuple[Sequence[float], Sequence[float]] | None,
        tuple[Sequence[float], Sequence[float]] | None,
    ]:
        """The (confidence, prediction) band pairs of a ``RowCurve``, ``None`` when absent."""
        if curve is None:
            return None, None
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
        return confidence, prediction

    @staticmethod
    def _overlay_curves(
        result: AutoFitResult, rows: Sequence[RankedFit], selected_id: str
    ) -> list[OverlayCurve]:
        """The top-three stored curves plus the selected row's, ordered by display position.

        Colours come from each row's 1-based ordinal in ``rows``, never ``RankedFit.rank``:
        ties are common (rows drawing the same curve share a rank) and must stay visibly
        distinct.
        """
        chosen: list[tuple[int, OverlayCurve]] = []
        seen: set[str] = set()
        for position, row in enumerate(rows[:3], start=1):
            curve = result.curves.get(row.model_id)
            if curve is None or row.model_id in seen:
                continue
            seen.add(row.model_id)
            chosen.append((position, OverlayCurve(row.name, curve.x, curve.y, position)))
        # A selected row outside the top three still shows its own curve, matching the
        # summary and the bands (its swatch and this curve use the same display position).
        if selected_id not in seen:
            selected_index = next(
                (
                    index
                    for index, candidate in enumerate(rows)
                    if candidate.model_id == selected_id
                ),
                None,
            )
            curve = result.curves.get(selected_id)
            if selected_index is not None and curve is not None:
                selected_row = rows[selected_index]
                position = selected_index + 1
                chosen.append(
                    (position, OverlayCurve(selected_row.name, curve.x, curve.y, position))
                )
        chosen.sort(key=lambda entry: entry[0])
        return [entry[1] for entry in chosen]

    def _open_candidate(self, candidate_id: str) -> None:
        dataset_id = self.selection.active_id
        result = None if dataset_id is None else self.analysis.autofit_for_display(dataset_id)
        if dataset_id is None or result is None:
            return
        self.analysis.open_candidate(dataset_id, result, candidate_id)

    # -- details window ----------------------------------------------------------------------

    def open_details(self) -> None:
        """Open the details of the selected ranked row, or of the active fit.

        The ranking's selected row wins in Auto-fit mode: it is what the user is looking
        at. A row already refitted into the current bundle reuses that bundle instead of
        fitting twice; any other row refits first and the window opens when it arrives.
        """
        row = self._selected_row()
        bundle = self.analysis.current_bundle
        if row is not None and self.analysis.mode == MODE_AUTOFIT:
            if bundle is not None and bundle.result.model_id == row.model_id:
                self.details_dialog.reveal()
            else:
                self._open_candidate(row.model_id)
            return
        if bundle is not None:
            self.details_dialog.reveal()
            return
        self.statusBar().showMessage(
            "Select a ranked row, or run or open a fit, to see its details"
        )

    def _selected_row(self) -> RankedFit | None:
        return self.analysis_panel.autofit_view.table.selected_row()

    def _row_for_model(self, model_id: str) -> RankedFit | None:
        return next(
            (
                row
                for row in self.analysis_panel.autofit_view.table.rows()
                if row.model_id == model_id
            ),
            None,
        )

    def _dataset_name(self, dataset_id: object) -> str:
        if not isinstance(dataset_id, str):
            return "—"
        entry = next((e for e in self.controller.project.datasets if e.id == dataset_id), None)
        return entry.name if entry is not None else "—"

    def _show_row_summary(self, dataset_id: str | None) -> None:
        row = self._selected_row()
        if row is not None:
            self.analysis_panel.summary.show_row(row, self._dataset_name(dataset_id))

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
        self._refresh_export_actions()

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
        self._refresh_export_actions()

    def _on_datasets_changed(self) -> None:
        """Datasets were added/removed: reconcile the selection, rebuild the tree.

        ``set_project`` only emits when the active dataset actually moved (for
        example the active dataset was removed), so the table and plot refresh at
        most once — an import's refresh comes from ``set_active`` instead.
        """
        self.selection.set_project(self.controller.project)
        self.refresh_tree()
        self._refresh_export_actions()

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
            action.triggered.connect(lambda *_, target=path: self._open_recent(target))

    def _refresh_axis_actions(self) -> None:
        self.action_log_x.setEnabled(self.plot_view.can_log_x)
        self.action_log_y.setEnabled(self.plot_view.can_log_y)
        if not self.plot_view.can_log_x:
            self.action_log_x.setChecked(False)
            self.action_log_x.setToolTip("Log axis needs strictly positive x values")
        else:
            self.action_log_x.setToolTip("Logarithmic x-axis")
        if not self.plot_view.can_log_y:
            self.action_log_y.setChecked(False)
            self.action_log_y.setToolTip("Log axis needs strictly positive y values")
        else:
            self.action_log_y.setToolTip("Logarithmic y-axis")

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
        """Dock layout, geometry and collapsed panels from QSettings, when stored."""
        state = self._settings.value(_LAYOUT_STATE_KEY, "")
        if isinstance(state, str) and state:
            self.layout_restored = bool(
                self.restoreState(QByteArray.fromBase64(bytes(state, encoding="ascii")))
            )
        geometry = self._settings.value(_LAYOUT_GEOMETRY_KEY, "")
        if isinstance(geometry, str) and geometry:
            self.restoreGeometry(QByteArray.fromBase64(bytes(geometry, encoding="ascii")))
        self.analysis_panel.restore_splitter(self._settings)  # the ranking/summary split
        self.analysis_panel.autofit_view.set_show_top3(self._stored_show_top3())
        for name in self.docks:
            extended = self._settings.value(f"{_EXPANDED_PREFIX}{name}", None)
            try:
                if extended is not None:
                    self._expanded_extent[name] = int(str(extended))
            except ValueError:
                pass
            self.set_dock_collapsed(name, self._stored_collapsed(name), persist=False)
        self._dock_floating_panels()

    def _dock_floating_panels(self) -> int:
        """Bring every floating panel back into the layout, and say so in the status bar.

        Floating is a *session* state, not part of the stored arrangement: a saved
        floating panel reopens as a window stranded over the middle of the workspace
        (often at a position from a different screen size), which reads as "the app
        opened broken". Restoring docks it back to the area it belongs to, at the size
        Qt kept for it; the user can float it again, and it will not be persisted.
        """
        rehomed = 0
        for name, dock in self.docks.items():
            if not dock.isFloating():
                continue
            dock.setFloating(False)
            home = _DOCK_HOME[name]
            if self.dockWidgetArea(dock) != home:
                self.addDockWidget(home, dock)
            if self._collapsed.get(name, False):
                # re-docking must not silently undo a collapsed strip
                if name == "table":
                    strip = max(28, self._title_bars[name].sizeHint().height())
                    dock.setMinimumHeight(strip)
                    dock.setMaximumHeight(strip)
                else:
                    dock.setMinimumWidth(_COLLAPSED_EXTENT)
                    dock.setMaximumWidth(_COLLAPSED_EXTENT)
            rehomed += 1
        if rehomed:
            self.statusBar().showMessage("Panels docked — floating is not kept between sessions")
        return rehomed

    def _stored_collapsed(self, name: str) -> bool:
        value = self._settings.value(f"{_COLLAPSED_PREFIX}{name}", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _stored_show_top3(self) -> bool:
        value = self._settings.value(_SHOW_TOP3_KEY, False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _on_show_top3_changed(self, enabled: bool) -> None:
        """Persist the overlay choice and redraw the selected row without refitting."""
        self._settings.setValue(_SHOW_TOP3_KEY, enabled)
        self._settings.sync()
        if self.analysis.mode != MODE_AUTOFIT:
            return  # a manual fit is not part of the ranking: leave its curve alone
        row = self._selected_row()
        if row is not None:
            self._plot_candidate(row.model_id)

    def _save_layout(self) -> None:
        if not self.persist_layout:
            return  # a second project window never rewrites the remembered layout
        self._settings.setValue(
            _LAYOUT_STATE_KEY, bytes(self.saveState().toBase64().data()).decode("ascii")
        )
        self._settings.setValue(
            _LAYOUT_GEOMETRY_KEY, bytes(self.saveGeometry().toBase64().data()).decode("ascii")
        )
        for name, collapsed in self._collapsed.items():
            self._settings.setValue(f"{_COLLAPSED_PREFIX}{name}", collapsed)
            if name in self._expanded_extent:
                self._settings.setValue(f"{_EXPANDED_PREFIX}{name}", self._expanded_extent[name])
        self.analysis_panel.save_splitter(self._settings)
        self.details_dialog.save_geometry()
        if self.help_dialog is not None:
            self.help_dialog.save_geometry()
        self._settings.sync()

    def set_dock_collapsed(self, name: str, collapsed: bool, *, persist: bool = True) -> None:
        """Collapse a panel to a slim title strip (or expand it back to its size)."""
        dock = self.docks[name]
        bar = self._title_bars[name]
        if self._collapsed.get(name, False) == collapsed:
            bar.set_collapsed(collapsed)
            return
        vertical = name != "table"
        content = dock.widget()
        if collapsed:
            extent = dock.width() if vertical else dock.height()
            # Learn the expanded size only from a live window: a layout restored before
            # the window is shown has not been laid out yet (the dock may still be a
            # sliver), and overwriting the remembered extent with that sliver would lose
            # the real size the expand has to restore.
            if self.isVisible() and (
                extent > _COLLAPSED_EXTENT or name not in self._expanded_extent
            ):
                self._expanded_extent[name] = extent
            if content is not None:
                content.hide()
            if vertical:
                dock.setMinimumWidth(_COLLAPSED_EXTENT)
                dock.setMaximumWidth(_COLLAPSED_EXTENT)
                dock.resize(_COLLAPSED_EXTENT, dock.height())
            else:
                strip = max(28, bar.sizeHint().height())
                dock.setMinimumHeight(strip)
                dock.setMaximumHeight(strip)
                dock.resize(dock.width(), strip)
        else:
            if content is not None:
                content.show()
            restore = self._expanded_extent.get(name, ANALYSIS_DOCK_WIDTH if vertical else 220)
            if vertical:
                dock.setMinimumWidth(0)
                dock.setMaximumWidth(_UNBOUNDED)
                self.resizeDocks([dock], [restore], Qt.Orientation.Horizontal)
            else:
                dock.setMinimumHeight(0)
                dock.setMaximumHeight(_UNBOUNDED)
                self.resizeDocks([dock], [restore], Qt.Orientation.Vertical)
        self._collapsed[name] = collapsed
        bar.set_collapsed(collapsed)
        if persist:
            self._settings.setValue(f"{_COLLAPSED_PREFIX}{name}", collapsed)
            if name in self._expanded_extent:
                self._settings.setValue(f"{_EXPANDED_PREFIX}{name}", self._expanded_extent[name])
            self._settings.sync()

    def _reset_layout(self) -> None:
        """Forget the stored layout and re-apply the default panel arrangement."""
        for key in (_LAYOUT_STATE_KEY, _LAYOUT_GEOMETRY_KEY, "analysis/splitterSizes"):
            self._settings.remove(key)
        for name in self.docks:
            self._settings.remove(f"{_COLLAPSED_PREFIX}{name}")
            self._settings.remove(f"{_EXPANDED_PREFIX}{name}")
        self.layout_restored = False
        self._expanded_extent.clear()
        for name, dock in self.docks.items():
            self.set_dock_collapsed(name, False, persist=False)
            dock.setFloating(False)
            dock.show()
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.project_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.table_dock)
        self.resizeDocks((self.analysis_panel,), (ANALYSIS_DOCK_WIDTH,), Qt.Orientation.Horizontal)
        self.resizeDocks((self.table_dock,), (220,), Qt.Orientation.Vertical)
        self.analysis_panel.splitter.setSizes([700, 220])
        self.resize(*DEFAULT_WINDOW_SIZE)
        self._settings.sync()
        self.statusBar().showMessage("Layout reset")

    # -- actions without a controller home ---------------------------------------------------

    def _refresh_export_actions(self) -> None:
        """Export results/ranking and Details are enabled only when there is something to show."""
        has_bundle = self.analysis.current_bundle is not None
        has_rows = bool(self.analysis_panel.autofit_view.table.rows())
        self.action_export_results.setEnabled(has_bundle)
        self.action_details.setEnabled(has_bundle or has_rows)
        dataset_id = self.selection.active_id
        stored = None if dataset_id is None else self.analysis.stored_autofit(dataset_id)
        self.action_export_ranking.setEnabled(stored is not None)

    def export_results(self) -> None:
        """Export the current fit as a report (JSON / XLSX / PDF / CSV by suffix)."""
        bundle = self.analysis.current_bundle
        if bundle is None:  # the action is disabled; a programmatic call just no-ops
            return
        target = self.controller.paths.choose_export_path(
            "Export results", "CurveStudio report (*.json *.xlsx *.pdf *.csv)"
        )
        if target is None:
            return
        # the byline is read here (QSettings stays on the GUI thread); the worker gets text
        byline = report_identity(self._settings).byline()
        self._run_job(
            f"Exporting {target.name}",
            partial(
                exports.export_fit,
                target,
                bundle,
                byline=byline,
                spec=self.analysis.current_spec,
            ),
            on_result=lambda _done: self.statusBar().showMessage(f"Exported {target.name}"),
            on_error=lambda exc: self.controller.messenger.show_error("Export failed", str(exc)),
        )

    def export_ranking(self) -> None:
        """Export the active dataset's Auto-fit ranking table as formula-safe CSV."""
        dataset_id = self.selection.active_id
        result = None if dataset_id is None else self.analysis.stored_autofit(dataset_id)
        if result is None:  # the action is disabled; a programmatic call just no-ops
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
        byline = report_identity(self._settings).byline()
        self._run_job(
            f"Exporting {target.name}",
            partial(exports.export_publication_plot, target, bundle, byline=byline),
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

    def show_about_qt(self) -> None:
        """Help → About Qt: the framework's own licence dialog."""
        app = QApplication.instance()
        if app is not None:
            cast("QApplication", app).aboutQt()

    def open_preferences(self) -> None:
        """Edit → Preferences…: the report byline and the theme; OK applies the theme at once.

        The dialog itself stores both. Re-applying the theme here (without storing it a
        second time) repaints the window and moves the View → Theme check marks.
        """
        if PreferencesDialog(self, settings=self._settings).exec():
            self.apply_theme_preference(theme_preference(self._settings), persist=False)

    def open_help(self, section: str | None = None) -> None:
        """Open (or reuse) the non-modal help browser, optionally at a named section."""
        if self.help_dialog is None:
            from curvestudio_desktop.views.help_dialog import HelpDialog

            self.help_dialog = HelpDialog(self, settings=self._settings)
        if section is not None:
            self.help_dialog.select_section(section)
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    # -- Qt events ---------------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller.maybe_quit():
            self._save_layout()
            # top-level child windows: leaving them open would keep the application
            # alive after the main window has gone
            self.details_dialog.close()
            if self.help_dialog is not None:
                self.help_dialog.close()
            event.accept()
            self.closed.emit()
        else:
            event.ignore()
