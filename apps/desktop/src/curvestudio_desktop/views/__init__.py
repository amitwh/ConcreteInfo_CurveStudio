"""Views: thin Qt widgets. Logic lives in :mod:`curvestudio_desktop.controllers`.

The names this package has always re-exported stay available — ``from
curvestudio_desktop.views import MainWindow`` works as before — but they resolve lazily
(PEP 562): importing the package imports no view, so start-up paths that only need one
light module (the window icon, say) do not pay for the whole window.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    from curvestudio_desktop.views.ranking_table import (
        RankingTableModel,
        RankingTableView,
    )
    from curvestudio_desktop.views.results_tabs import ResultsTabs
    from curvestudio_desktop.views.spreadsheet import SpreadsheetGrid

_EXPORTS = {
    "AnalysisPanel": ".analysis_panel",
    "AutoFitView": ".autofit_view",
    "ClipboardImportDialog": ".import_dialog",
    "DataTableDock": ".data_table",
    "DatasetTableModel": ".data_table",
    "DetailsDialog": ".details_dialog",
    "FitSummaryPanel": ".fit_summary",
    "ImportDialog": ".import_dialog",
    "MainWindow": ".main_window",
    "ManualFitView": ".manual_fit_view",
    "PlotView": ".plot_view",
    "ProjectDock": ".project_tree",
    "ProjectTree": ".project_tree",
    "RankingTableModel": ".ranking_table",
    "RankingTableView": ".ranking_table",
    "ResultsTabs": ".results_tabs",
    "SpreadsheetGrid": ".spreadsheet",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(__name__ + module, package=__name__), name)
    globals()[name] = value  # later lookups skip this hook
    return value
