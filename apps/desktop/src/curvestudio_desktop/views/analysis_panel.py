"""AnalysisPanel: the right-hand dock — mode switch, the workflow's form, the fit summary.

The dock is Auto-fit-first: the Auto-fit page is the default (and remembered in QSettings);
Manual fit appears only when the user selects it. A vertical splitter puts the workflow
pages — the ranking table's home — above a compact ``FitSummaryPanel``: the ranking is THE
primary result, so its page keeps the dominant share and the split is remembered in
QSettings. The five full result views live in the details window, not here. The mode switch
is a compact horizontal segmented control, never stacked full-width buttons. Nothing here
computes anything: the pages talk to the AnalysisController, and the shell wires their
signals to the plot, the summary and the details window.
"""

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from curvestudio_desktop.controllers.analysis import MODE_AUTOFIT, MODE_MANUAL, AnalysisController
from curvestudio_desktop.controllers.selection import SelectionController
from curvestudio_desktop.views.autofit_view import AutoFitView
from curvestudio_desktop.views.fit_summary import FitSummaryPanel
from curvestudio_desktop.views.manual_fit_view import ManualFitView

_SPLITTER_KEY = "analysis/splitterSizes"
#: A fresh install shows the ranking and a fully visible summary at 1600x920 and
#: 1280x800; the summary's own minimum height is the hard floor the splitter honours.
_DEFAULT_SPLIT = (640, 240)


class AnalysisPanel(QDockWidget):
    """Right dock: ANALYSIS — mode switch, Auto-fit or Manual fit page, fit summary."""

    def __init__(
        self,
        analysis: AnalysisController,
        selection: SelectionController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Analysis", parent)
        self.setObjectName("analysisDock")
        self._analysis = analysis
        self._split_restored = False
        self.autofit_view = AutoFitView(analysis)
        self.manual_view = ManualFitView(analysis, selection.xy)
        self.summary = FitSummaryPanel()
        self._pages = QStackedWidget()
        self._pages.addWidget(self.autofit_view)
        self._pages.addWidget(self.manual_view)
        self.autofit_button = QPushButton("Auto-fit")
        self.autofit_button.setObjectName("autofitModeButton")
        self.autofit_button.setCheckable(True)
        self.manual_button = QPushButton("Manual fit")
        self.manual_button.setObjectName("manualModeButton")
        self.manual_button.setCheckable(True)
        for button in (self.autofit_button, self.manual_button):
            button.setProperty("segment", "true")  # QSS styles the checked/unchecked pair
        group = QButtonGroup(self)
        group.addButton(self.autofit_button)
        group.addButton(self.manual_button)
        group.setExclusive(True)
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(self.autofit_button, stretch=1)
        mode_row.addWidget(self.manual_button, stretch=1)
        # the ranking table is the primary result: its page keeps the dominant share
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setObjectName("analysisSplitter")
        self.splitter.addWidget(self._pages)
        self.splitter.addWidget(self.summary)
        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setCollapsible(0, False)  # the workflow page never collapses away
        self.splitter.setCollapsible(1, True)  # the summary can, when the ranking is the focus
        self.splitter.setSizes(list(_DEFAULT_SPLIT))
        holder = QWidget(self)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(mode_row)
        layout.addWidget(self.splitter, stretch=1)
        self.setWidget(holder)
        self.autofit_button.clicked.connect(lambda: analysis.set_mode(MODE_AUTOFIT))
        self.manual_button.clicked.connect(lambda: analysis.set_mode(MODE_MANUAL))
        analysis.mode_changed.connect(self.set_mode)
        self.set_mode(analysis.mode)

    # -- splitter persistence -----------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._split_restored:
            # seed the default once the dock has its real geometry; Qt scales the
            # pair to the space available, so the ratio is what matters
            self.splitter.setSizes(list(_DEFAULT_SPLIT))

    def save_splitter(self, settings: QSettings) -> None:
        """Remember the table/tabs split (the caller syncs the settings)."""
        top, bottom = self.splitter.sizes()[:2] if len(self.splitter.sizes()) >= 2 else (0, 0)
        if top > 0:
            settings.setValue(_SPLITTER_KEY, f"{top},{bottom}")

    def restore_splitter(self, settings: QSettings) -> bool:
        """Apply a remembered split; ``False`` (nothing applied) when the value is junk."""
        stored = str(settings.value(_SPLITTER_KEY, ""))
        try:
            top, bottom = (int(part) for part in stored.split(","))
        except ValueError:
            return False
        if top <= 0 or bottom < 0:
            return False
        self._split_restored = True
        self.splitter.setSizes([top, bottom])
        return True

    # -- mode ---------------------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Show the mode's page (and check its button) without re-triggering it."""
        self.autofit_button.setChecked(mode == MODE_AUTOFIT)
        self.manual_button.setChecked(mode == MODE_MANUAL)
        self._pages.setCurrentWidget(
            self.autofit_view if mode == MODE_AUTOFIT else self.manual_view
        )

    def mode(self) -> str:
        return MODE_AUTOFIT if self._pages.currentWidget() is self.autofit_view else MODE_MANUAL
