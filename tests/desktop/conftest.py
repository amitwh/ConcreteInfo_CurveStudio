"""Shared fixtures for the desktop tests.

The whole directory runs offscreen: ``QT_QPA_PLATFORM=offscreen`` is set before
PySide6 is imported anywhere, and the directory is skipped cleanly when PySide6
itself is unavailable (for example a core-only checkout).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed; desktop tests skipped")

from PySide6.QtCore import QSettings

from curvestudio_desktop.controllers.project import ProjectController
from curvestudio_desktop.controllers.providers import UnsavedAnswer

# Shared CSV/text snippets live in desktop_data.py (a plain module: test dirs have
# no packages), imported by whichever test needs them.


class FakeConfirmation:
    """Scripted answers to the unsaved-changes prompt, with the questions recorded."""

    def __init__(self) -> None:
        self.answers: list[UnsavedAnswer] = []
        self.actions: list[str] = []

    def confirm_unsaved_changes(self, action: str) -> UnsavedAnswer:
        self.actions.append(action)
        return self.answers.pop(0) if self.answers else UnsavedAnswer.CANCEL


class FakePathProvider:
    """Scripted file-dialog answers; an empty queue behaves like the user cancelling."""

    def __init__(self, base) -> None:
        self.base = base
        self.open_paths: list = []
        self.save_paths: list = []
        self.save_suggestions: list[str] = []
        self.export_paths: list = []
        self.export_calls: list[tuple[str, str]] = []

    def choose_open_path(self):
        return self.open_paths.pop(0) if self.open_paths else None

    def choose_save_path(self, suggested_name: str):
        self.save_suggestions.append(suggested_name)
        return self.save_paths.pop(0) if self.save_paths else None

    def choose_export_path(self, title: str, filters: str):
        self.export_calls.append((title, filters))
        return self.export_paths.pop(0) if self.export_paths else None


class RecordingMessenger:
    """Collects every error the controller would otherwise show in a message box."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, str]] = []
        self.infos: list[tuple[str, str]] = []

    def show_error(self, title: str, message: str) -> None:
        self.errors.append((title, message))

    def show_information(self, title: str, message: str) -> None:
        self.infos.append((title, message))

    def error_text(self) -> str:
        return "\n".join(message for _, message in self.errors)


@pytest.fixture()
def settings(tmp_path) -> QSettings:
    """A throwaway INI-backed QSettings, so tests never touch the user's config."""
    path = tmp_path / "settings.ini"
    return QSettings(str(path), QSettings.Format.IniFormat)


@pytest.fixture()
def confirmation() -> FakeConfirmation:
    return FakeConfirmation()


@pytest.fixture()
def paths(tmp_path) -> FakePathProvider:
    return FakePathProvider(base=tmp_path)


@pytest.fixture()
def messenger() -> RecordingMessenger:
    return RecordingMessenger()


@pytest.fixture()
def controller(
    confirmation: FakeConfirmation,
    paths: FakePathProvider,
    messenger: RecordingMessenger,
    settings: QSettings,
) -> ProjectController:
    return ProjectController(
        confirmation=confirmation,
        paths=paths,
        messenger=messenger,
        settings=settings,
    )


@pytest.fixture()
def write_file(tmp_path):
    """Factory: write text or bytes under ``tmp_path`` and return the Path."""

    def _write(name: str, content: str | bytes):
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    return _write


@pytest.fixture()
def make_window(controller, settings):
    """Factory building a MainWindow around the fixture controller (no dialogs)."""

    def _make():
        from curvestudio_desktop.views.main_window import MainWindow

        return MainWindow(controller, settings=settings)

    return _make
