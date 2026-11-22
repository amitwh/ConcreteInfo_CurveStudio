"""A custom dock title bar: title, collapse/expand, float and close.

``QDockWidget`` gets this widget via ``setTitleBarWidget``. Collapsing is owned by
the main window (it knows the dock's orientation and size); the title bar only
emits :attr:`DockTitleBar.collapseToggled` — from its button or a title double-click
— and reflects the state through :meth:`set_collapsed`.

A custom title bar replaces the native one, so the window manager never sees a title
bar to drag a floating panel by. A press on a floating title bar therefore asks the
platform to start the window move (``QWindow.startSystemMove``: _NET_WM_MOVERESIZE on
X11, ``xdg_toplevel.move`` on Wayland, the native move elsewhere); a docked panel
ignores the press so ``QDockWidget`` keeps its own drag-to-dock behaviour. Moving the
window by hand from a mouse handler is not an option: Qt re-dispatches the move after
the window has shifted, and a ``QDockWidget`` that receives a move it never saw the
press for dereferences its drag state and crashes.

Double-clicking a floating title bar docks the panel back. The click pair is detected
from the presses themselves, because the platform's move grab can swallow the second
press before Qt turns it into a double-click event.
"""

from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

from curvestudio_desktop.views.icons import icon

_BUTTON_ICON_SIZE = QSize(16, 16)
_DOUBLE_CLICK_SLOP = 4  # px the pointer may drift between the two presses of a double-click
_DOUBLE_CLICK_TAIL_MS = 600  # how long the pair's trailing event may still arrive


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
        self._last_press: tuple[int, QPoint] | None = None
        self._swallow_double_click = False

        self.label = QLabel(title, self)
        self.label.setObjectName("dockTitleLabel")
        self.collapse_button = self._button("dockCollapseButton")
        self.float_button = self._button("dockFloatButton")
        self.float_button.setCheckable(True)  # checked while the panel floats
        self.close_button = self._button("dockCloseButton")
        self.collapse_button.clicked.connect(self._toggle_collapse)
        self.float_button.clicked.connect(self._toggle_float)
        self.close_button.clicked.connect(dock.close)
        # a panel can also be docked back by dragging it onto the main window or by
        # the View menu: the button must always reflect the real state
        dock.topLevelChanged.connect(self._on_floating_changed)
        self._on_floating_changed(dock.isFloating())

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
        if self._dock.isFloating():
            # back into the layout where the main window put it (never where it drifted)
            self._dock.setFloating(False)
            return
        self._dock.setFloating(True)
        self._dock.show()
        self._dock.raise_()
        self._dock.activateWindow()

    def _on_floating_changed(self, floating: bool) -> None:
        """Reflect the real state: a floating panel offers "dock", a docked one "float"."""
        self.float_button.setChecked(floating)
        self.float_button.setToolTip("Dock panel" if floating else "Float panel")
        self._last_press = None

    # -- moving a floating panel -------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton or not self._dock.isFloating():
            # docked: ignore so QDockWidget still sees the press and drags the panel
            # into another dock area (its own dock-drag handling lives in the parent)
            super().mousePressEvent(event)
            return
        if self._is_second_click(event):
            self._dock.setFloating(False)  # double-click a floating title: dock it back
            self._swallow_double_click = True  # the platform may still deliver the pair
            # the tail of this click arrives at once or not at all; never let a stale
            # flag swallow a later, unrelated double-click on the docked title bar
            QTimer.singleShot(_DOUBLE_CLICK_TAIL_MS, self._clear_swallow)
            event.accept()
            return
        handle = self._dock.windowHandle()
        if handle is not None:
            handle.startSystemMove()  # the window manager drags from here (see module note)
        event.accept()

    def _clear_swallow(self) -> None:
        self._swallow_double_click = False

    def _is_second_click(self, event: QMouseEvent) -> bool:
        """Whether this press completes a double-click on the title bar."""
        previous = self._last_press
        current = (event.timestamp(), event.globalPosition().toPoint())
        self._last_press = current
        if previous is None:
            return False
        elapsed, position = previous
        return (current[1] - position).manhattanLength() <= _DOUBLE_CLICK_SLOP and (
            current[0] - elapsed <= QApplication.doubleClickInterval()
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Keep a floating panel's moves away from QDockWidget's drag state machine.

        The platform owns the drag (``startSystemMove``); an ignored move would
        propagate to the dock, whose floating-drag handler expects the press it never
        saw and dereferences its drag state (:class:`QDockWidget` crashes there).
        """
        if self._dock.isFloating():
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dock.isFloating() and event.button() is Qt.MouseButton.LeftButton:
            event.accept()  # the press was ours; the dock must not see its release
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        if self._swallow_double_click:
            # the press pair already docked the panel; this is its tail, not a new request
            self._swallow_double_click = False
            event.accept()
            return
        if self._dock.isFloating():
            self._dock.setFloating(False)  # a floating title docks its panel back
        else:
            self.collapseToggled.emit(not self._collapsed)
        event.accept()


__all__ = ["DockTitleBar"]
