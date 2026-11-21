"""Widget-backed dialogs and the production implementations of the provider protocols."""

import traceback
from importlib import resources
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import Qt, qVersion
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from curvestudio_desktop import APP_NAME, __version__
from curvestudio_desktop.controllers.providers import UnsavedAnswer

_CSTUDIO_FILTER = "CurveStudio project (*.cstudio);;All files (*)"


def active_parent() -> QWidget | None:
    """The window a dialog must parent to, so it never opens behind the main window."""
    return QApplication.activeWindow()


def notice_text() -> str:
    """The bundled licence notice, read via ``importlib.resources`` (packaging-safe)."""
    try:
        return (resources.files("curvestudio_desktop") / "resources" / "NOTICE.txt").read_text(
            encoding="utf-8"
        )
    except OSError, FileNotFoundError, ModuleNotFoundError:
        return "Licensed AGPL-3.0-or-later. See the LICENSE file in the repository."


class AboutDialog(QDialog):
    """Help → About: application name, version, licence notice, stack versions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About CurveStudio")
        self._label = QLabel(self.full_text())
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextBrowserInteraction
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        close = QPushButton("Close")
        close.setObjectName("closeButton")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)

    def full_text(self) -> str:
        import PySide6

        return (
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {__version__}</p>"
            f"<p>PySide6 {PySide6.__version__} / Qt {qVersion()}</p>"
            f"<pre style='white-space: pre-wrap'>{notice_text()}</pre>"
        )


class CrashDialog(QDialog):
    """The app-level crash guard dialog: the error, its details, one-click copy."""

    def __init__(self, details: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._details = details
        self.setWindowTitle("Unexpected error")
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "CurveStudio hit an unexpected error and may be in an inconsistent state.\n"
                "Your saved files are unaffected; unsaved changes stay in memory until you quit."
            )
        )
        details_view = QPlainTextEdit(details)
        details_view.setReadOnly(True)
        layout.addWidget(details_view)
        copy = QPushButton("Copy details")
        copy.setObjectName("copyDetailsButton")
        copy.clicked.connect(self.copy_details)
        layout.addWidget(copy, 0, Qt.AlignmentFlag.AlignRight)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)

    def details(self) -> str:
        return self._details

    def copy_details(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._details)


def show_crash_dialog(details: str) -> None:
    """Open (and wait on) the crash dialog; the app exits after it is dismissed."""
    if QApplication.instance() is None:
        print(details, flush=True)
        return
    CrashDialog(details, active_parent()).exec()


class MessageBoxMessenger:
    """The production :class:`Messenger`: message boxes, never tracebacks."""

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(active_parent(), title, message)

    def show_information(self, title: str, message: str) -> None:
        QMessageBox.information(active_parent(), title, message)


class DialogConfirmationProvider:
    """The production :class:`ConfirmationProvider`: Save / Discard / Cancel."""

    def confirm_unsaved_changes(self, action: str) -> UnsavedAnswer:
        parent = active_parent()
        box = QMessageBox(parent) if parent is not None else QMessageBox()
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"Save the changes to this project before {action}?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            return UnsavedAnswer.SAVE
        if choice == QMessageBox.StandardButton.Discard:
            return UnsavedAnswer.DISCARD
        return UnsavedAnswer.CANCEL


class DialogPathProvider:
    """The production :class:`PathProvider`: the platform file dialogs."""

    def choose_open_path(self) -> Path | None:
        path, _ = QFileDialog.getOpenFileName(
            active_parent(), "Open project", "", _CSTUDIO_FILTER, "CurveStudio project (*.cstudio)"
        )
        return Path(path) if path else None

    def choose_save_path(self, suggested_name: str) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(
            active_parent(), "Save project", suggested_name, _CSTUDIO_FILTER
        )
        return Path(path) if path else None

    def choose_export_path(self, title: str, filters: str) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(active_parent(), title, "", filters)
        return Path(path) if path else None


def format_exception(
    exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
) -> str:
    """The exact text the crash guard shows and copies."""
    return "".join(traceback.format_exception(exc_type, exc, tb))
