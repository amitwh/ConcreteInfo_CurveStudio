"""AnalysisPanel: the right-hand dock with the four result tabs.

D1 builds the real panel: four tabs, each an honest "No fit yet" until a fit
exists. D2 fills them with parameters, statistics, diagnostics and the rendered
equation; nothing here pretends results exist when they do not.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QTabWidget, QVBoxLayout, QWidget

TAB_TITLES = ("Parameters", "Statistics", "Diagnostics", "Equation")
EMPTY_TEXT = "No fit yet"


class AnalysisPanel(QDockWidget):
    """Right dock: ANALYSIS tabs — Parameters / Statistics / Diagnostics / Equation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Analysis", parent)
        self.setObjectName("analysisDock")
        self._tabs = QTabWidget()
        self._labels: list[QLabel] = []
        for title in TAB_TITLES:
            label = QLabel(EMPTY_TEXT)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._labels.append(label)
            self._tabs.addTab(label, title)
        holder = QWidget(self)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._tabs)
        self.setWidget(holder)

    def tab_titles(self) -> list[str]:
        return [self._tabs.tabText(i) for i in range(self._tabs.count())]

    def tab_text_at(self, index: int) -> str:
        return self._labels[index].text()

    def count(self) -> int:
        return self._tabs.count()
