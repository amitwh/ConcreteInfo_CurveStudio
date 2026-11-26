"""The guided start: what an empty "Untitled project" shows in the plot area (item 18).

Launching into an empty project meets a blank plot. The empty state stands in for it:
Open a project, Import data, Paste from clipboard and Help — each button fires the
window's existing action, so the same handler (and shortcut) runs — plus the recent
projects, from the same QSettings list the Open-recent menu reads.
"""

from pathlib import Path

import pytest
from PySide6.QtWidgets import QListWidget, QPushButton

PAIR_CSV = "age,strength\n1,2.1\n2,4.4\n3,9.2\n4,16.8\n"


def button_of(window, name: str) -> QPushButton:
    button = window.empty_state.findChild(QPushButton, name)
    assert button is not None, f"no {name} on the empty start"
    return button


def test_the_empty_state_renders_for_an_empty_project(make_window, qtbot) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    assert window.controller.project.datasets == ()
    assert window.plot_view.is_empty_page_shown()
    assert window.empty_state.isVisibleTo(window)


def test_data_in_the_project_replaces_the_empty_state(make_window, qtbot, write_file) -> None:
    window = make_window()
    qtbot.addWidget(window)
    window.show()
    assert window.import_file(write_file("pair.csv", PAIR_CSV)) is True
    assert not window.plot_view.is_empty_page_shown()
    assert not window.empty_state.isVisibleTo(window)


class _CancelledDialog:
    """Stands in for a modal dialog the handler would open: closes at once, no outcome."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    def exec(self) -> int:
        return 0


@pytest.mark.parametrize(
    ("button_name", "action_name"),
    [
        ("emptyOpenProjectButton", "action_open"),
        ("emptyImportButton", "action_import"),
        ("emptyPasteButton", "action_clipboard"),
        ("emptyHelpButton", "action_help_contents"),
    ],
)
def test_every_button_fires_the_existing_action(
    make_window, qtbot, monkeypatch, button_name: str, action_name: str
) -> None:
    window = make_window()
    qtbot.addWidget(window)
    # the real handler runs — only its modal dialog is stood down (an offscreen exec()
    # blocks forever, and the dialog's outcome is not what is under test here)
    import curvestudio_desktop.views.main_window as main_window_module

    monkeypatch.setattr(main_window_module, "ImportDialog", _CancelledDialog)
    monkeypatch.setattr(window, "make_clipboard_dialog", lambda *_a, **_k: _CancelledDialog())
    fired: list[bool] = []
    getattr(window, action_name).triggered.connect(lambda checked=False: fired.append(True))
    button_of(window, button_name).click()
    assert fired == [True], f"{button_name} did not fire {action_name}"


def test_the_recent_projects_are_the_settings_list(
    make_window, qtbot, settings, write_file
) -> None:
    first = write_file("bridge.cstudio", "x")
    second = write_file("slab.cstudio", "x")
    gone = write_file("gone.cstudio", "x")
    gone.unlink()  # the list keeps only projects that still exist on disk
    settings.setValue("recent_projects", "\n".join(str(p) for p in (first, second, gone)))
    settings.sync()
    window = make_window()
    qtbot.addWidget(window)
    assert window.empty_state.recent_projects() == [first, second]


def test_choosing_a_recent_project_opens_it(
    make_window, qtbot, settings, controller, write_file, monkeypatch
) -> None:
    target = write_file("arch.cstudio", "x")
    settings.setValue("recent_projects", str(target))
    settings.sync()
    opened: list[Path] = []
    monkeypatch.setattr(controller, "open_project", lambda path: opened.append(path))
    window = make_window()
    qtbot.addWidget(window)
    recent_list = window.empty_state.findChild(QListWidget, "emptyRecentList")
    assert recent_list is not None and recent_list.count() == 1
    recent_list.itemActivated.emit(recent_list.item(0))  # Enter or double-click
    assert opened == [target]
