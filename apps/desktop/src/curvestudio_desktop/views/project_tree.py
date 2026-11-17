"""ProjectDock: the left-hand tree of datasets and their analyses."""

from collections.abc import Callable
from typing import cast

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QDockWidget, QTreeWidget, QTreeWidgetItem, QWidget

from curvestudio.project import Project


class ProjectTree(QTreeWidget):
    """The tree itself: datasets are selectable, analyses are listed read-only (D1)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectTree")
        self.setHeaderHidden(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._handler: Callable[[str | None], None] | None = None
        self._menu_handler: Callable[[str, QPoint], None] | None = None
        self.itemSelectionChanged.connect(self._emit_selection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._emit_menu_request)

    def on_dataset_selected(self, handler: Callable[[str | None], None]) -> None:
        self._handler = handler

    def on_dataset_menu(self, handler: Callable[[str, QPoint], None]) -> None:
        """Right-click on a dataset row: handler(dataset_id, global position)."""
        self._menu_handler = handler

    def refresh(self, project: Project, active_id: str | None) -> None:
        """Rebuild the tree, keeping (or restoring) the active dataset selected."""
        self.blockSignals(True)
        self.clear()
        datasets_root = QTreeWidgetItem(["Datasets"])
        analyses_root = QTreeWidgetItem(["Analyses"])
        self.addTopLevelItems([datasets_root, analyses_root])
        selected: QTreeWidgetItem | None = None
        for entry in project.datasets:
            item = QTreeWidgetItem([entry.name])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.id)
            item.setToolTip(0, f"dataset id: {entry.id}")
            datasets_root.addChild(item)
            if entry.id == active_id:
                selected = item
        for analysis in project.analyses:
            item = QTreeWidgetItem([f"{analysis.kind} — {analysis.id}"])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # fits become clickable in D2
            analyses_root.addChild(item)
        datasets_root.setExpanded(True)
        analyses_root.setExpanded(True)
        self.blockSignals(False)
        if selected is not None:
            selected.setSelected(True)

    def selected_dataset_id(self) -> str | None:
        items = self.selectedItems()
        if not items:
            return None
        return cast("str | None", items[0].data(0, Qt.ItemDataRole.UserRole))

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
        if self._handler is not None:
            self._handler(self.selected_dataset_id())

    def _emit_menu_request(self, position: QPoint) -> None:
        if self._menu_handler is None:
            return
        item = self.itemAt(position)
        if item is None:
            return
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
