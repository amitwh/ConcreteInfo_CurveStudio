"""The local crash log: one entry per crash, on this computer, never sent anywhere.

The crash guard appends every crash it handles (main thread or worker thread) to
``crash.log`` in the app's data directory, with the time in IST and the versions it ran
under. The file rotates at 1 MB. Writing it can fail (a full disk, no rights to the
folder) and never takes the dialog or the stderr line down with it.
"""

import os
import platform
import re
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PySide6.QtCore import QStandardPaths, QUrl, qVersion
from PySide6.QtWidgets import QLabel, QPushButton

import curvestudio_desktop.app as app_module
from curvestudio_desktop import __version__
from curvestudio_desktop.app import (
    CRASH_LOG_MAX_BYTES,
    CRASH_LOG_NAME,
    CRASH_LOG_ROTATED_NAME,
    install_crash_guard,
)
from curvestudio_desktop.views import dialogs as dialogs_module
from curvestudio_desktop.views.dialogs import CrashDialog

IST = timezone(timedelta(hours=5, minutes=30))
HEADER = re.compile(r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) IST · CurveStudio .+$", re.M)

Crash = Callable[[str], None]
"""Raises ``ValueError(<message>)`` through the installed main-thread hook."""


@pytest.fixture()
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The app's data directory, relocated into the test's own folder."""
    directory = tmp_path / "app-data"
    monkeypatch.setattr(app_module, "app_data_dir", lambda: directory)
    return directory


@pytest.fixture()
def crash(log_dir: Path) -> Iterator[Crash]:
    """The crash guard with a silent shower, and a way to raise through its main-thread hook."""
    original = (sys.excepthook, threading.excepthook)
    install_crash_guard(shower=lambda _details: None)

    def raise_on_the_main_thread(message: str = "boom") -> None:
        try:
            raise ValueError(message)
        except ValueError as error:
            sys.excepthook(ValueError, error, error.__traceback__)

    yield raise_on_the_main_thread
    sys.excepthook, threading.excepthook = original


def log_text(log_dir: Path) -> str:
    return (log_dir / CRASH_LOG_NAME).read_text(encoding="utf-8")


# -- the entries ---------------------------------------------------------------------------------


def test_each_crash_is_one_entry_with_its_five_parts(crash: Crash, log_dir: Path) -> None:
    crash("first")
    crash("second")

    text = log_text(log_dir)
    headers = HEADER.findall(text)
    assert len(headers) == 2, text
    for part in (
        f"CurveStudio {__version__}",
        f"Python {platform.python_version()}",
        f"Qt {qVersion()}",
    ):
        assert text.count(part) == 2, part
    # the traceback itself, once per crash, each under its own header
    assert text.count("Traceback (most recent call last):") == 2
    assert text.count("ValueError: first") == 1
    assert text.count("ValueError: second") == 1
    first, second = text.split("ValueError: first", 1)
    assert HEADER.search(first) and "ValueError: second" not in first
    assert HEADER.search(second)


def test_the_log_time_is_ist_not_the_machines_zone(crash: Crash, log_dir: Path) -> None:
    """Put the process on UTC: a log in "local" time would read five and a half hours off."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        crash()
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()

    stamp = HEADER.search(log_text(log_dir)).group("time")
    stamp = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    assert abs((datetime.now(IST) - stamp).total_seconds()) < 60


def test_a_crash_inside_the_crash_dialog_is_still_logged(log_dir: Path) -> None:
    """The dialog is not re-entered, but the second crash is still one more entry."""
    calls: list[str] = []

    def recursive_shower(details: str) -> None:
        calls.append(details)
        if len(calls) == 1:
            try:
                raise ValueError("nested")
            except ValueError as error:
                sys.excepthook(ValueError, error, error.__traceback__)

    original = (sys.excepthook, threading.excepthook)
    install_crash_guard(shower=recursive_shower)
    try:
        try:
            raise ValueError("first")
        except ValueError as error:
            sys.excepthook(ValueError, error, error.__traceback__)
    finally:
        sys.excepthook, threading.excepthook = original

    assert len(calls) == 1
    text = log_text(log_dir)
    assert len(HEADER.findall(text)) == 2
    assert "ValueError: nested" in text


def test_a_worker_thread_crash_is_logged_too(crash: Crash, log_dir: Path, qtbot) -> None:
    def boom() -> None:
        raise RuntimeError("worker boom")

    thread = threading.Thread(target=boom)
    thread.start()
    qtbot.waitUntil(
        lambda: (log_dir / CRASH_LOG_NAME).is_file() and "worker boom" in log_text(log_dir),
        timeout=5000,
    )
    thread.join(timeout=5)
    assert len(HEADER.findall(log_text(log_dir))) == 1


# -- rotation, failure and the directory ---------------------------------------------------------


def test_the_log_rotates_at_one_megabyte_replacing_the_previous_copy(
    crash: Crash, log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert CRASH_LOG_MAX_BYTES == 1_000_000
    # big enough for one entry (a traceback is a few hundred bytes), small enough to trip
    monkeypatch.setattr(app_module, "CRASH_LOG_MAX_BYTES", 2_000)
    old = "o" * 2_500
    log_dir.mkdir()  # the app creates it on demand; here a previous session already logged
    (log_dir / CRASH_LOG_NAME).write_text(old, encoding="utf-8")
    (log_dir / CRASH_LOG_ROTATED_NAME).write_text("older", encoding="utf-8")

    crash("after the rotation")

    assert (log_dir / CRASH_LOG_ROTATED_NAME).read_text(encoding="utf-8") == old  # replaces "older"
    text = log_text(log_dir)
    assert old not in text
    assert "ValueError: after the rotation" in text

    # the new file is below the limit: the next crash is appended, nothing rotates again
    crash("appended")
    assert (log_dir / CRASH_LOG_ROTATED_NAME).read_text(encoding="utf-8") == old
    text = log_text(log_dir)
    assert "ValueError: after the rotation" in text and "ValueError: appended" in text


def test_a_crash_with_an_unwritable_log_directory_still_shows_the_dialog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file where the log folder should be", encoding="utf-8")
    monkeypatch.setattr(app_module, "app_data_dir", lambda: blocker)
    shown: list[str] = []
    original = (sys.excepthook, threading.excepthook)
    install_crash_guard(shower=shown.append)
    try:
        try:
            raise ValueError("boom")
        except ValueError as error:
            sys.excepthook(ValueError, error, error.__traceback__)  # must not raise
    finally:
        sys.excepthook, threading.excepthook = original

    assert len(shown) == 1 and "ValueError: boom" in shown[0]
    captured = capsys.readouterr()
    assert "ValueError: boom" in captured.err  # the stderr line still went out
    assert "crash log" in captured.err.lower()  # and the failure is said, not swallowed
    assert blocker.read_text(encoding="utf-8") == "a file where the log folder should be"


def test_app_data_dir_is_qts_app_data_folder_created_on_demand() -> None:
    """Qt's test mode puts every standard path under ``~/.qttest``: the real one is untouched."""
    QStandardPaths.setTestModeEnabled(True)
    expected = Path(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    )
    try:
        assert expected.is_relative_to(Path.home() / ".qttest"), expected
        if expected.exists():
            shutil.rmtree(expected)
        assert app_module.app_data_dir() == expected
        assert expected.is_dir()
        assert app_module.app_data_dir() == expected  # asking again is harmless
    finally:
        QStandardPaths.setTestModeEnabled(False)
        shutil.rmtree(expected, ignore_errors=True)


# -- the dialog ----------------------------------------------------------------------------------


def test_the_crash_dialog_offers_the_log_folder_and_says_the_report_stays_local(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[QUrl] = []

    class FakeDesktopServices:
        @staticmethod
        def openUrl(url: QUrl) -> bool:
            opened.append(url)
            return True

    monkeypatch.setattr(dialogs_module, "QDesktopServices", FakeDesktopServices)
    monkeypatch.setattr(app_module, "app_data_dir", lambda: tmp_path)
    dialog = CrashDialog("Traceback (most recent call last):\n  boom")
    qtbot.addWidget(dialog)

    button = dialog.findChild(QPushButton, "openLogFolderButton")
    assert button is not None and button.text() == "Open log folder"
    note = dialog.findChild(QLabel, "crashLogNote")
    assert note is not None and note.text() == "This report stays on your computer."
    assert dialog.findChild(QPushButton, "copyDetailsButton") is not None  # still there

    button.click()
    assert [url.toLocalFile() for url in opened] == [str(tmp_path)]
