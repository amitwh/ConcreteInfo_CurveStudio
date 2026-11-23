"""Multiple project windows: File → New/Open project opens another window."""

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from curvestudio_desktop import windows as windows_module
from curvestudio_desktop.controllers.project import ProjectController
from curvestudio_desktop.views.main_window import MainWindow
from curvestudio_desktop.windows import WindowRegistry


@pytest.fixture()
def registry(settings, confirmation, paths, messenger, qtbot):
    """A registry whose windows get their own controller (the real wiring, fake dialogs)."""

    def factory(window_registry: WindowRegistry, persist_layout: bool) -> MainWindow:
        controller = ProjectController(
            confirmation=confirmation,
            paths=paths,
            messenger=messenger,
            settings=settings,
        )
        window = MainWindow(
            controller, settings=settings, registry=window_registry, persist_layout=persist_layout
        )
        qtbot.addWidget(window)
        return window

    # the registry gets the tests' throwaway settings too: its install record reads them
    return WindowRegistry(factory, settings=settings)


def test_a_spawned_window_sends_the_install_record_once_it_is_shown(
    registry, settings: QSettings, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    """The window comes first; the app's one network call follows, off the start-up path."""
    asked: list[QSettings] = []
    monkeypatch.setattr(windows_module, "maybe_send_install_record", asked.append)

    registry.spawn()
    assert asked == []  # show() has run; the deferred call has not

    qtbot.waitUntil(lambda: bool(asked), timeout=2000)
    assert asked == [settings]  # through the registry's own settings


def test_new_project_opens_a_second_window_and_leaves_the_first_alone(registry) -> None:
    first = registry.spawn()
    special = first.controller  # the first window keeps this very controller
    second = registry.spawn()
    QApplication.processEvents()

    assert registry.windows() == [first, second]
    assert first.controller is not second.controller
    assert first.persist_layout is True
    assert second.persist_layout is False  # only the first window owns the layout
    assert second.controller is not None and second.controller.path is None

    # File → New project spawns another window; the first project is untouched
    first.action_new.trigger()
    QApplication.processEvents()
    assert len(registry.windows()) == 3
    assert first.controller is special
    assert first.controller.path is None


def test_a_bare_window_keeps_the_single_project_behaviour(make_window, qtbot) -> None:
    """No registry (the selftest, tests): New project still resets this very window."""
    window = make_window()
    qtbot.addWidget(window)
    assert window.registry is None
    window.action_new.trigger()
    assert window.controller.path is None
    assert window.windowTitle().startswith("ConcreteInfo CurveStudio")


def test_closing_a_window_forgets_it_but_keeps_the_others(registry) -> None:
    first = registry.spawn()
    second = registry.spawn()
    QApplication.processEvents()

    second.close()
    QApplication.processEvents()
    assert registry.windows() == [first]
    assert first.isVisible()


def test_a_secondary_window_never_rewrites_the_stored_layout(registry, settings) -> None:
    first = registry.spawn()
    second = registry.spawn()
    QApplication.processEvents()

    settings.remove("main_window/state")
    second._save_layout()
    assert not settings.contains("main_window/state")

    first._save_layout()
    assert settings.contains("main_window/state")


def test_open_project_asks_for_a_file_then_spawns_a_window(registry, paths, tmp_path) -> None:
    first = registry.spawn()
    target = tmp_path / "other.cstudio"
    paths.open_paths.append(target)

    opened = registry.open_from(first)
    QApplication.processEvents()
    assert opened is not None
    assert registry.windows() == [first, opened]
    assert paths.open_paths == []  # the dialog answer was consumed
    assert first.controller.path is None  # the asking window kept its project

    # cancelling the dialog opens nothing
    assert registry.open_from(first) is None
    assert len(registry.windows()) == 2
