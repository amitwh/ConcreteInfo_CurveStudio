"""Widget-backed dialogs and the production implementations of the provider protocols."""

import platform
import traceback
from importlib import resources
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import Qt, qVersion
from PySide6.QtGui import QPixmap
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
WEBSITE_URL = "https://www.concreteinfo.in"
SOURCE_URL = "https://github.com/amitwh/ConcreteInfo_CurveStudio"
LICENCE_URL = "https://www.gnu.org/licenses/agpl-3.0.html"
DESCRIPTION = "Fit a curve to paired x/y measurements and see, honestly, how well it fits."
CREDITS: tuple[str, ...] = (
    "Qt / PySide6 — LGPL-3.0",
    "NumPy, SciPy, SymPy — BSD-3-Clause",
    "pyqtgraph — MIT",
    "Inter and JetBrains Mono typefaces — SIL Open Font License 1.1 (OFL-1.1)",
    "Lucide icons — ISC",
)


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


def runtime_versions() -> dict[str, str]:
    """The versions of the stack actually imported at runtime, never hardcoded."""
    import numpy
    import PySide6
    import scipy

    return {
        "Python": platform.python_version(),
        "Qt": qVersion(),
        "PySide6": PySide6.__version__,
        "NumPy": numpy.__version__,
        "SciPy": scipy.__version__,
    }


def _logo_pixmap() -> QPixmap:
    """The staged brand logo, loaded unchanged (it is built for light surfaces)."""
    try:
        asset = (
            resources.files("curvestudio_desktop") / "resources" / "brand" / "concreteinfo-logo.png"
        )
        data = asset.read_bytes()
    except OSError, ModuleNotFoundError:
        return QPixmap()
    pixmap = QPixmap()
    if not pixmap.loadFromData(data, "PNG"):
        return QPixmap()
    return pixmap


class AboutDialog(QDialog):
    """Help → About: the brand, version, links, runtime stack and credits."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("aboutDialog")
        self.setWindowTitle("About CurveStudio")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(8)

        # the asset is designed for light surfaces: it sits on white and is never
        # inverted or recoloured, so it stays legible in the dark theme too
        self.logo_label = QLabel()
        self.logo_label.setObjectName("aboutLogo")
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setStyleSheet(
            "QLabel#aboutLogo { background-color: #ffffff; border-radius: 8px; padding: 10px; }"
        )
        logo = _logo_pixmap()
        if not logo.isNull():
            self.logo_label.setPixmap(logo)
        layout.addWidget(self.logo_label)

        self.title_label = QLabel(APP_NAME)
        self.title_label.setStyleSheet("font-size: 19px; font-weight: 600;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.version_label = QLabel(f"Version {__version__}")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version_label)

        self.description_label = QLabel(DESCRIPTION)
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.description_label)

        self.website_label = QLabel(
            f'<a href="{WEBSITE_URL}">{WEBSITE_URL.removeprefix("https://")}</a>'
            f' · <a href="{SOURCE_URL}">source code</a>'
        )
        self._make_link_label(self.website_label)
        layout.addWidget(self.website_label)

        self.licence_label = QLabel(
            f'Licensed under <a href="{LICENCE_URL}">AGPL-3.0-or-later</a>.'
        )
        self._make_link_label(self.licence_label)
        layout.addWidget(self.licence_label)

        self.stack_label = QLabel(self.version_text())
        self.stack_label.setWordWrap(True)
        self.stack_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.stack_label)

        self.credits_label = QLabel(self.credits_text())
        self.credits_label.setObjectName("aboutCredits")
        self.credits_label.setWordWrap(True)
        self.credits_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.credits_label)

        close = QPushButton("Close")
        close.setObjectName("closeButton")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _make_link_label(label: QLabel) -> None:
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def version_text(self) -> str:
        versions = runtime_versions()
        return " · ".join(
            f"{name} {versions[name]}" for name in ("Python", "Qt", "PySide6", "NumPy", "SciPy")
        )

    def credits_text(self) -> str:
        return "\n".join(CREDITS)

    def full_text(self) -> str:
        versions = runtime_versions()
        return (
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {__version__}</p>"
            f"<p>{DESCRIPTION}</p>"
            f"<p><a href='{WEBSITE_URL}'>{WEBSITE_URL}</a> · "
            f"<a href='{SOURCE_URL}'>source code</a></p>"
            f"<p>Licensed under <a href='{LICENCE_URL}'>AGPL-3.0-or-later</a>.</p>"
            f"<p>Python {versions['Python']} / PySide6 {versions['PySide6']} / "
            f"Qt {versions['Qt']} / NumPy {versions['NumPy']} / SciPy {versions['SciPy']}</p>"
            f"<p>{'<br/>'.join(CREDITS)}</p>"
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
