"""One window per project — *File → New project* and *Open project…* open another window.

A project owns a whole analysis surface (its dataset table, its plot, its ranking, its
running jobs), so a second project belongs in a second window rather than a tab that
would have to hide the first project's docks. Opening a project in a new window also
means the work already on screen is never replaced by an accidental Ctrl+N.

Windows are independent: each has its own controller, its own unsaved-changes prompt and
its own jobs. Only the first window writes the remembered layout to QSettings, so adding
a window never rewrites the arrangement the user set up.

The registry keeps every open window alive (a PySide top-level with no Python reference
would be collected), and forgets one when it closes.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

if TYPE_CHECKING:
    from curvestudio_desktop.views.main_window import MainWindow

WindowFactory = Callable[["WindowRegistry", bool], "MainWindow"]
"""``factory(registry, persist_layout)`` builds one fully wired project window."""


class WindowRegistry(QObject):
    """The open project windows, and the way to open one more."""

    def __init__(self, factory: WindowFactory, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._factory = factory
        self._windows: list[MainWindow] = []

    def windows(self) -> list[MainWindow]:
        """The open windows, oldest first."""
        return list(self._windows)

    def spawn(self, path: Path | None = None) -> MainWindow:
        """Open a new project window (optionally loading ``path``), and show it."""
        window = self._factory(self, not self._windows)
        self._windows.append(window)
        window.closed.connect(lambda closed=window: self.forget(closed))
        if path is not None:
            window.open_project_path(path)
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def open_from(self, source: MainWindow) -> MainWindow | None:
        """Ask for a project file in ``source`` and open it in a new window."""
        path = source.controller.paths.choose_open_path()
        if path is None:
            return None
        return self.spawn(path)

    def forget(self, window: MainWindow) -> None:
        """Drop a closed window from the registry (idempotent)."""
        if window in self._windows:
            self._windows.remove(window)


__all__ = ["WindowFactory", "WindowRegistry"]
