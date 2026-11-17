"""The ``--selftest`` entry point and the app-level crash guard."""

import subprocess
import sys
import threading

from PySide6.QtCore import QSettings

from curvestudio_desktop import __version__


def run_selftest() -> subprocess.CompletedProcess[str]:
    import os

    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.run(
        [sys.executable, "-m", "curvestudio_desktop", "--selftest"],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )


def test_selftest_exits_zero_and_prints_ok() -> None:
    result = run_selftest()
    assert result.returncode == 0, f"selftest failed:\n{result.stdout}\n{result.stderr}"
    assert "SELFTEST OK" in result.stdout


def test_version_flag_reports_version() -> None:
    import os

    result = subprocess.run(
        [sys.executable, "-m", "curvestudio_desktop", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_selftest_settings_is_isolated_and_cleaned_up() -> None:
    """The selftest never reads or writes the user's real configuration."""
    from pathlib import Path

    from curvestudio_desktop.app import selftest_settings

    with selftest_settings() as settings:
        path = Path(settings.fileName())
        assert path.suffix == ".ini"
        settings.setValue("analysis/mode", "manual")
        settings.sync()
        assert settings.value("analysis/mode") == "manual"
        directory = path.parent
    assert not directory.exists()  # the throwaway directory went away with it


def test_build_main_window_reads_and_writes_the_injected_settings(qtbot, tmp_path) -> None:
    from curvestudio_desktop.app import build_main_window

    settings = QSettings(str(tmp_path / "selftest.ini"), QSettings.Format.IniFormat)
    settings.setValue("analysis/mode", "manual")
    settings.sync()
    window = build_main_window(settings)
    qtbot.addWidget(window)
    assert window.analysis.mode == "manual"  # the window read the injected settings
    window.analysis.set_mode("autofit")
    settings.sync()
    assert settings.value("analysis/mode") == "autofit"  # and writes back to them


def test_crash_guard_formats_and_shows_the_error() -> None:
    from curvestudio_desktop.app import install_crash_guard

    shown: list[str] = []
    original = sys.excepthook
    original_threading = threading.excepthook
    install_crash_guard(shower=shown.append)
    try:
        assert sys.excepthook is not original
        try:
            raise ValueError("boom")
        except ValueError as error:
            sys.excepthook(ValueError, error, error.__traceback__)
    finally:
        sys.excepthook = original
        threading.excepthook = original_threading
    assert len(shown) == 1
    assert "ValueError" in shown[0] and "boom" in shown[0]


def test_crash_guard_installs_a_threading_hook_too() -> None:
    from curvestudio_desktop.app import install_crash_guard

    original = sys.excepthook
    original_threading = threading.excepthook
    install_crash_guard(shower=lambda _details: None)
    try:
        assert threading.excepthook is not original_threading
    finally:
        sys.excepthook = original
        threading.excepthook = original_threading


def test_crash_guard_marshals_worker_crash_to_the_gui_thread(qtbot) -> None:
    """A threading.Thread crash must surface on the GUI thread, never on the worker."""
    from PySide6.QtWidgets import QApplication

    from curvestudio_desktop.app import install_crash_guard

    app = QApplication.instance()
    shown_on: list = []

    def shower(details: str) -> None:
        from PySide6.QtCore import QThread

        shown_on.append(QThread.currentThread())

    original = sys.excepthook
    original_threading = threading.excepthook
    install_crash_guard(shower=shower)
    try:

        def boom() -> None:
            raise RuntimeError("worker boom")

        thread = threading.Thread(target=boom)
        thread.start()
        qtbot.waitUntil(lambda: bool(shown_on), timeout=5000)
        thread.join(timeout=5)
    finally:
        sys.excepthook = original
        threading.excepthook = original_threading
    assert shown_on, "the worker crash never reached the shower"
    assert shown_on[0] is app.thread(), "the crash dialog must open on the GUI thread"


def test_crash_guard_is_not_re_entrant(qtbot, capsys) -> None:
    """A crash inside the crash dialog must not recurse into more dialogs."""
    from curvestudio_desktop.app import install_crash_guard

    calls: list[str] = []

    def recursive_shower(details: str) -> None:
        calls.append(details)
        if len(calls) == 1:  # the first show itself crashes — through the hook
            try:
                raise ValueError("nested")
            except ValueError as error:
                sys.excepthook(ValueError, error, error.__traceback__)

    original = sys.excepthook
    original_threading = threading.excepthook
    install_crash_guard(shower=recursive_shower)
    try:
        try:
            raise ValueError("first")
        except ValueError as error:
            sys.excepthook(ValueError, error, error.__traceback__)
    finally:
        sys.excepthook = original
        threading.excepthook = original_threading
    assert len(calls) == 1, "the nested crash must not open a second dialog"
    captured = capsys.readouterr()
    assert "nested" in captured.err  # it went to stderr instead


def test_crash_dialog_has_copy_details(qtbot) -> None:
    from PySide6.QtWidgets import QApplication, QPushButton

    from curvestudio_desktop.views.dialogs import CrashDialog

    dialog = CrashDialog(details="Traceback (most recent call last):\n  boom")
    qtbot.addWidget(dialog)
    button = dialog.findChild(QPushButton, "copyDetailsButton")
    assert button is not None
    dialog.copy_details()
    assert QApplication.clipboard().text().startswith("Traceback")
