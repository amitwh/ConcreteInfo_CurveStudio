"""Views: thin Qt widgets. Logic lives in :mod:`curvestudio_desktop.controllers`."""

from curvestudio_desktop.views.analysis_panel import AnalysisPanel
from curvestudio_desktop.views.data_table import DatasetTableModel, DataTableDock
from curvestudio_desktop.views.import_dialog import ClipboardImportDialog, ImportDialog
from curvestudio_desktop.views.main_window import MainWindow
from curvestudio_desktop.views.plot_view import PlotView
from curvestudio_desktop.views.project_tree import ProjectDock, ProjectTree

__all__ = [
    "AnalysisPanel",
    "ClipboardImportDialog",
    "DataTableDock",
    "DatasetTableModel",
    "ImportDialog",
    "MainWindow",
    "PlotView",
    "ProjectDock",
    "ProjectTree",
]
