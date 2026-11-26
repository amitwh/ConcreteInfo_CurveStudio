"""Shared fixtures for the desktop tests.

The whole directory runs offscreen: ``QT_QPA_PLATFORM=offscreen`` is set before
PySide6 is imported anywhere, and the directory is skipped cleanly when PySide6
itself is unavailable (for example a core-only checkout). The one network call the
app makes (its install record) is switched off for the whole directory as well, so
a test that spawns a real window can never post to the real endpoint.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# No desktop test may reach the install-record endpoint: an empty URL means "never send"
# (see curvestudio_desktop.install_record). A test that exercises the variable sets it
# itself, with monkeypatch.
os.environ.setdefault("CURVESTUDIO_INSTALL_RECORD_URL", "")

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed; desktop tests skipped")

from PySide6.QtCore import QSettings

from curvestudio_desktop import model_library
from curvestudio_desktop.controllers.project import ProjectController
from curvestudio_desktop.controllers.providers import UnsavedAnswer


@pytest.fixture(autouse=True)
def _model_library_loaded() -> None:
    """The library behind the window, loaded before any test runs.

    The app loads it in the background after the window shows; the tests assert on fully
    populated windows, so they load it up front instead (the fitting stack is imported
    around them anyway). The window-first behaviour has its own tests.
    """
    model_library.load_now()


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


@pytest.fixture(autouse=True)
def _manual_mode_by_default(settings):
    """Desktop tests default to Manual fit mode: background Auto-fit must not race them.

    The product default is Auto-fit (imports start a run by themselves), but a test that
    imports data and then saves would race the run landing in the project (and a default
    run with the full catalogue is slow). Tests that exercise the Auto-fit-first workflow
    set the mode explicitly — or remove this key to check the product default.
    """
    settings.setValue("analysis/mode", "manual")
    settings.sync()


@pytest.fixture(autouse=True)
def _preserve_model_registry():
    """Snapshot the shared model registry around every desktop test and restore it.

    Desktop tests trigger the registry's lazy plugin loading (Auto-fit enumeration, the
    model picker) while the root conftest hides the plugin entry points — that reload
    would empty the process-wide registry and leave the later in-process CLI tests with
    no models at all. Restoring the snapshot keeps whatever catalogue the session had.
    Tests that manage the registry themselves (``_clear_registry`` under
    ``real_plugins``) still work: their changes are simply undone afterwards.
    """
    from curvestudio.models import registry

    snapshot = (
        dict(registry._REGISTRY),
        dict(registry._FAMILIES),
        dict(registry._FAMILY_ERRORS),
        list(registry._LOADED_PLUGINS),
        list(registry._PLUGIN_FAILURES),
        registry._PLUGINS_DONE,
    )
    yield
    models, families, family_errors, loaded, failures, done = snapshot
    registry._REGISTRY.clear()
    registry._REGISTRY.update(models)
    registry._FAMILIES.clear()
    registry._FAMILIES.update(families)
    registry._FAMILY_ERRORS.clear()
    registry._FAMILY_ERRORS.update(family_errors)
    registry._LOADED_PLUGINS[:] = loaded
    registry._PLUGIN_FAILURES[:] = failures
    registry._PLUGINS_DONE = done


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
