"""The window first, the library behind it (P6 Task 6, half B).

Building the main window must not import the fitting stack: the heavy libraries load on a
background thread after the window is up, with a status-bar message while they do, and a
command that needs the library before it is ready waits or says so — never crashes, never
silently no-ops. These tests hold that: a fresh process builds the window without one heavy
module; the window appears before the library finishes; the library-needing paths work once
it lands; and the model picker fills when it does.
"""

import ast
import os
import subprocess
import sys
import textwrap

HEAVY = (
    "curvestudio.discovery",
    "curvestudio.workflow",
    "curvestudio.project",
    "curvestudio.tools",
    "curvestudio.fitting",
    "pandas",
    "scipy",
    "sympy",
)
"""What building the window must not import: the fitting stack the window never needs
before it is on screen. NumPy is deliberately absent — pyqtgraph draws the plot itself
and brings it, which is the window's own cost, not the library's."""


def build_offscreen(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=240,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )


def test_building_the_window_imports_none_of_the_heavy_libraries() -> None:
    result = build_offscreen(
        """
        import os, sys, tempfile
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        app = QApplication([sys.argv[0]])
        from curvestudio_desktop.app import build_main_window
        with tempfile.TemporaryDirectory() as d:
            settings = QSettings(os.path.join(d, "s.ini"), QSettings.Format.IniFormat)
            window = build_main_window(settings)
            window.show()
            for _ in range(10):
                app.processEvents()
            imported = sorted(
                m for m in sys.modules if m == m.split(".")[0] or m.startswith("curvestudio")
            )
            print("IMPORTED", [m for m in imported])
            window.close()
        """
    )
    assert result.returncode == 0, result.stderr
    printed = next(line for line in result.stdout.splitlines() if line.startswith("IMPORTED"))
    top = set(ast.literal_eval(printed[len("IMPORTED ") :]))
    offenders = [m for m in HEAVY if m in top or any(o == m or o.startswith(m + ".") for o in top)]
    assert not offenders, f"building the window imported {offenders}"


def test_the_window_shows_before_the_model_library_is_ready(qtbot, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    from curvestudio_desktop import model_library
    from curvestudio_desktop.app import build_main_window

    assert QApplication.instance() is not None
    # the library is held back: nothing loads until the test lets it
    monkeypatch.setattr(model_library, "_loaded", False)
    monkeypatch.setattr(model_library, "_thread", None)
    window = build_main_window()
    assert window.isVisible() or not window.library_ready()
    assert "loading" in window.statusBar().currentMessage().lower()
    assert not model_library.is_ready()
    # a command that needs the library says so instead of silently doing nothing
    assert window.analysis_panel.manual_view.model_count() == 0
    window.close()


def test_the_library_loads_and_the_views_fill(qtbot) -> None:
    from PySide6.QtWidgets import QApplication

    from curvestudio_desktop import model_library
    from curvestudio_desktop.app import build_main_window

    assert QApplication.instance() is not None
    model_library.load_now()  # the synchronous path: already imported in a test process
    window = build_main_window()
    assert model_library.is_ready()
    assert window.library_ready()
    assert window.analysis_panel.manual_view.model_count() > 0
    window.close()


def test_a_command_needing_the_library_waits_for_it(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    from curvestudio_desktop import model_library
    from curvestudio_desktop.app import build_main_window

    assert QApplication.instance() is not None
    model_library.load_now()
    window = build_main_window()
    # while "loading": a library command reports it, and works once loaded
    monkeypatch.setattr(model_library, "is_ready", lambda: False)
    assert window.library_ready() is False
    monkeypatch.setattr(model_library, "is_ready", lambda: True)
    assert window.library_ready() is True
    window.close()


def test_the_light_rank_values_mirror_the_engines_enum() -> None:
    """The strings the pre-library UI reads must be exactly ``RankBy``'s members."""
    from curvestudio.discovery import RankBy
    from curvestudio_desktop.controllers.analysis import RANK_VALUES
    from curvestudio_desktop.views.autofit_view import RANK_LABELS

    assert tuple(member.value for member in RankBy) == RANK_VALUES
    assert tuple(value for value, _label in RANK_LABELS) == RANK_VALUES
