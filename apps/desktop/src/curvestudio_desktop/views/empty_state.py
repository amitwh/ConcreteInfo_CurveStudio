"""EmptyStartView: the guided first screen in the plot area (design item 18).

A launch into an empty "Untitled project" used to meet a blank plot. This page stands
in for it instead: the four first moves as buttons that trigger the window's existing
actions (the same handlers, the same shortcuts as the menus), and the recent projects
from the same QSettings list the Open-recent menu reads. It knows nothing else — no
controller, no dialogs; whoever owns the actions decides what they do.
"""

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EmptyStartView(QWidget):
    """The empty project's start page: four actions and the recent projects."""

    openRecentRequested = Signal(Path)
    """A recent project was chosen (Enter or double-click): its path, for the owner to open."""

    def __init__(
        self,
        *,
        open_action: QAction,
        import_action: QAction,
        clipboard_action: QAction,
        help_action: QAction,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("emptyStart")

        title = QLabel("Get started")
        title.setObjectName("emptyStartTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        intro = QLabel("Open a project or import data to see it plotted here.")
        intro.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro.setWordWrap(True)
        intro.setStyleSheet("color: rgba(128, 128, 128, 200);")

        # The buttons carry the design's wording; what they trigger is the window's
        # existing action, so the handler and the shortcut can never drift apart.
        # They stack: a row of four would set the plot area's minimum width, and the
        # docks (not this page) own the window's width budget.
        actions = (
            ("Open a project…", "emptyOpenProjectButton", open_action),
            ("Import data…", "emptyImportButton", import_action),
            ("Paste from clipboard", "emptyPasteButton", clipboard_action),
            ("Help", "emptyHelpButton", help_action),
        )
        buttons = QVBoxLayout()
        buttons.setSpacing(6)
        for text, name, action in actions:
            button = QPushButton(text)
            button.setObjectName(name)
            button.setToolTip(action.statusTip())
            button.clicked.connect(action.trigger)
            buttons.addWidget(button)

        self._recent_title = QLabel("Recent projects")
        self._recent_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._recent_title.setStyleSheet("color: rgba(128, 128, 128, 200);")
        self._recent = QListWidget()
        self._recent.setObjectName("emptyRecentList")
        self._recent.setMaximumHeight(140)
        self._recent.setMaximumWidth(360)
        # activated (Enter or double-click), not clicked: one activation must open one
        # project, and a double-click is also two clicks
        self._recent.itemActivated.connect(self._open_item)
        self.set_recent_projects(())

        column = QVBoxLayout()
        column.addStretch(2)
        column.addWidget(title)
        column.addWidget(intro)
        column.addSpacing(12)
        column.addLayout(buttons)
        column.addSpacing(20)
        column.addWidget(self._recent_title)
        column.addWidget(self._recent)
        column.addStretch(3)
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        centered = QHBoxLayout()
        centered.addStretch(1)
        centered.addLayout(column)
        centered.addStretch(1)
        outer.addLayout(centered)
        outer.addStretch(1)

    # -- the recent projects -------------------------------------------------------------------

    def set_recent_projects(self, paths: Sequence[Path]) -> None:
        """Show ``paths`` as the recent projects (nothing hides the whole section)."""
        self._recent.clear()
        for path in paths:
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self._recent.addItem(item)
        listed = bool(paths)
        self._recent_title.setVisible(listed)
        self._recent.setVisible(listed)

    def recent_projects(self) -> list[Path]:
        """The projects listed, newest first, as given to :meth:`set_recent_projects`."""
        return [
            Path(self._recent.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self._recent.count())
        ]

    def _open_item(self, item: QListWidgetItem) -> None:
        self.openRecentRequested.emit(Path(item.data(Qt.ItemDataRole.UserRole)))
