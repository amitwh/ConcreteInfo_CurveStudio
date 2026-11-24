"""ProjectDock: the left-hand tree of datasets, each listing its stored analyses.

Datasets are selectable (the plot, table and analysis dock follow); an analysis row under
its dataset restores that stored result when clicked — an Auto-fit's ranking or a fit's
tabs, without refitting anything.
"""

from collections.abc import Callable, Mapping
from typing import cast

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QWidget

from curvestudio.project import Analysis, Project
from curvestudio.tools.stats import catalogue

STAT_LABEL = "Statistics"
"""The row label's first word: a stored statistics result names its tool after it."""


def _stat_label(analysis: Analysis, titles: Mapping[str, str]) -> str:
    """A stored statistics result's row: "Statistics — <tool title>"."""
    tool = analysis.config["tool"]  # the model keeps a "stat" analysis's tool name here
    return f"{STAT_LABEL} — {titles.get(tool, tool)}"


class ProjectTree(QTreeWidget):
    """The tree itself: datasets are selectable, and each lists its analyses beneath it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._handler: Callable[[str | None], None] | None = None
        self._analysis_handler: Callable[[str], None] | None = None
        self._menu_handler: Callable[[str, QPoint], None] | None = None
        self._labels: Callable[[str], str] | None = None
        self.itemSelectionChanged.connect(self._emit_selection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._emit_menu_request)

    def on_dataset_selected(self, handler: Callable[[str | None], None]) -> None:
        self._handler = handler

    def on_analysis_selected(self, handler: Callable[[str], None]) -> None:
        """Clicking an analysis row: handler(analysis_id) — the shell restores it."""
        self._analysis_handler = handler

    def on_dataset_menu(self, handler: Callable[[str, QPoint], None]) -> None:
        """Right-click on a dataset row: handler(dataset_id, global position)."""
        self._menu_handler = handler

    def set_analysis_labeller(self, labeller: Callable[[str], str]) -> None:
        """How an analysis row is titled (the analysis controller names them)."""
        self._labels = labeller

    def refresh(self, project: Project, active_id: str | None) -> None:
        """Rebuild the tree, keeping (or restoring) the active dataset selected."""
        self.blockSignals(True)
        self.clear()
        datasets_root = QTreeWidgetItem(["Datasets"])
        self.addTopLevelItem(datasets_root)
        selected: QTreeWidgetItem | None = None
        titles: dict[str, str] | None = None
        for entry in project.datasets:
            item = QTreeWidgetItem([entry.name])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.id)
            item.setToolTip(0, f"dataset id: {entry.id}")
            datasets_root.addChild(item)
            for analysis in project.analyses:
                if analysis.dataset_id != entry.id:
                    continue
                if analysis.kind == "stat":
                    if titles is None:  # loaded once per rebuild, and only if needed
                        titles = {info.name: info.title for info in catalogue()}
                    label = _stat_label(analysis, titles)
                else:
                    label = self._labels(analysis.id) if self._labels else analysis.kind
                child = QTreeWidgetItem([label])
                child.setData(0, Qt.ItemDataRole.UserRole, analysis.id)
                child.setData(0, Qt.ItemDataRole.UserRole + 1, "analysis")
                child.setToolTip(0, f"analysis {analysis.kind}: {analysis.id}")
                item.addChild(child)
            if entry.id == active_id:
                selected = item
        datasets_root.setExpanded(True)
        self.blockSignals(False)
        if selected is not None:
            selected.setSelected(True)

    def selected_dataset_id(self) -> str | None:
        for item in self.selectedItems():
            if item.data(0, Qt.ItemDataRole.UserRole + 1) is None:
                return cast("str | None", item.data(0, Qt.ItemDataRole.UserRole))
        return None

    def selected_analysis_id(self) -> str | None:
        for item in self.selectedItems():
            if item.data(0, Qt.ItemDataRole.UserRole + 1) == "analysis":
                return cast("str | None", item.data(0, Qt.ItemDataRole.UserRole))
        return None

    def dataset_items(self) -> list[QTreeWidgetItem]:
        """The dataset rows (children of the Datasets root, in project order)."""
        root = self.topLevelItem(0)  # "Datasets"
        if root is None:
            return []
        return [item for i in range(root.childCount()) if (item := root.child(i)) is not None]

    def select_active(self, active_id: str | None) -> None:
        """Highlight ``active_id`` (no rebuild, no selection-changed signal).

        The tree is the *view* of :class:`SelectionController.active_id` — whenever
        the active dataset changes for any reason (import, click, undo), the tree
        must follow it.
        """
        self.blockSignals(True)
        for item in self.dataset_items():
            item.setSelected(item.data(0, Qt.ItemDataRole.UserRole) == active_id)
        self.blockSignals(False)

    def update_label(self, dataset_id: str, name: str) -> None:
        """Retitle one dataset row in place (a rename is not a structural change)."""
        for item in self.dataset_items():
            if item.data(0, Qt.ItemDataRole.UserRole) == dataset_id:
                item.setText(0, name)
                return

    def dataset_labels(self) -> list[str]:
        root = self.topLevelItem(0)  # "Datasets"
        if root is None:
            return []
        labels: list[str] = []
        for i in range(root.childCount()):
            child = root.child(i)
            if child is not None:
                labels.append(child.text(0))
        return labels

    def _emit_selection(self) -> None:
        analysis_id = self.selected_analysis_id()
        if analysis_id is not None:
            if self._analysis_handler is not None:
                self._analysis_handler(analysis_id)
            return
        if self._handler is not None:
            self._handler(self.selected_dataset_id())

    def _emit_menu_request(self, position: QPoint) -> None:
        if self._menu_handler is None:
            return
        item = self.itemAt(position)
        if item is None:
            return
        if item.data(0, Qt.ItemDataRole.UserRole + 1) == "analysis":
            return  # only dataset rows have a context menu
        dataset_id = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(dataset_id, str):
            self._menu_handler(dataset_id, self.viewport().mapToGlobal(position))


class ProjectDock(QDockWidget):
    """Left dock: the PROJECT tree (datasets + analyses, PRD §16.1)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Project", parent)
        self.setObjectName("projectDock")
        self.tree = ProjectTree(self)
        self.setWidget(self.tree)

    @property
    def project_tree(self) -> ProjectTree:
        return self.tree
