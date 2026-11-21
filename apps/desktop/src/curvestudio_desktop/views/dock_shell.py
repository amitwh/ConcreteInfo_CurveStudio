"""A custom dock title bar: title, collapse/expand, float and close.

``QDockWidget`` gets this widget via ``setTitleBarWidget``. Collapsing is owned by
the main window (it knows the dock's orientation and size); the title bar only
emits :attr:`DockTitleBar.collapseToggled` — from its button or a title double-click
— and reflects the state through :meth:`set_collapsed`.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

from curvestudio_desktop.views.icons import icon

_BUTTON_ICON_SIZE = QSize(16, 16)


class DockTitleBar(QWidget):
    """The title row every dock gets; buttons carry icons and tooltips."""

    collapseToggled = Signal(bool)
    """The user asked to collapse (True) or expand (False) the dock."""

    def __init__(
        self,
        title: str,
        dock: QDockWidget,
        *,
        vertical: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dockTitleBar")
        self.setToolTip(title)
        self._dock = dock
        self._vertical = vertical
        self._collapsed = False

        self.label = QLabel(title, self)
        self.label.setObjectName("dockTitleLabel")
        self.collapse_button = self._button("dockCollapseButton")
        self.float_button = self._button("dockFloatButton")
        self.close_button = self._button("dockCloseButton")
        self.collapse_button.clicked.connect(self._toggle_collapse)
        self.float_button.clicked.connect(self._toggle_float)
        self.close_button.clicked.connect(dock.close)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.collapse_button)
        layout.addWidget(self.float_button)
        layout.addWidget(self.close_button)
        self.collapse_button.setToolTip("Collapse panel")
        self.float_button.setToolTip("Float panel")
        self.close_button.setToolTip("Close panel")
        self.refresh_icons()

    def _button(self, object_name: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName(object_name)
        button.setAutoRaise(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setIconSize(_BUTTON_ICON_SIZE)
        return button

    # -- state -------------------------------------------------------------------------------

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Reflect the dock's state: slim strip shows only the expand button."""
        self._collapsed = collapsed
        self.collapse_button.setToolTip("Expand panel" if collapsed else "Collapse panel")
        self.collapse_button.setIcon(icon("expand" if collapsed else "collapse"))
        # a collapsed vertical strip has no room for the title or the other buttons
        hide_extras = collapsed and self._vertical
        self.label.setVisible(not hide_extras)
        self.float_button.setVisible(not hide_extras)
        self.close_button.setVisible(not hide_extras)

    def refresh_icons(self) -> None:
        """Re-render every icon for the current theme (called after a theme switch)."""
        self.collapse_button.setIcon(icon("expand" if self._collapsed else "collapse"))
        self.float_button.setIcon(icon("float"))
        self.close_button.setIcon(icon("close"))

    # -- actions -----------------------------------------------------------------------------

    def _toggle_collapse(self) -> None:
        self.collapseToggled.emit(not self._collapsed)

    def _toggle_float(self) -> None:
        self._dock.setFloating(not self._dock.isFloating())

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.collapseToggled.emit(not self._collapsed)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


__all__ = ["DockTitleBar"]
