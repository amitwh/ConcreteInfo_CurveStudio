"""Views: thin Qt widgets. Logic lives in :mod:`curvestudio_desktop.controllers`."""

from curvestudio_desktop.views.analysis_panel import AnalysisPanel
from curvestudio_desktop.views.autofit_view import AutoFitView
from curvestudio_desktop.views.data_table import DatasetTableModel, DataTableDock
from curvestudio_desktop.views.details_dialog import DetailsDialog
from curvestudio_desktop.views.fit_summary import FitSummaryPanel
from curvestudio_desktop.views.import_dialog import ClipboardImportDialog, ImportDialog
from curvestudio_desktop.views.main_window import MainWindow
from curvestudio_desktop.views.manual_fit_view import ManualFitView
from curvestudio_desktop.views.plot_view import PlotView
from curvestudio_desktop.views.project_tree import ProjectDock, ProjectTree
from curvestudio_desktop.views.ranking_table import RankingTableModel, RankingTableView
from curvestudio_desktop.views.results_tabs import ResultsTabs

__all__ = [
    "AnalysisPanel",
    "AutoFitView",
    "ClipboardImportDialog",
    "DataTableDock",
    "DatasetTableModel",
    "DetailsDialog",
    "FitSummaryPanel",
    "ImportDialog",
    "MainWindow",
    "ManualFitView",
    "PlotView",
    "ProjectDock",
    "ProjectTree",
    "RankingTableModel",
    "RankingTableView",
    "ResultsTabs",
]
