"""The one network call the desktop app makes: an anonymous install record, once per version.

On first launch, and again after an upgrade, the app posts seven short strings (a random
installation id, the app version, the operating system, its version, the architecture,
which package the app came from, the session type) and nothing else. A failure is silent
and leaves the version unrecorded, so the next launch simply tries again.

The real transport is exercised against a local ``http.server``: a test may import
``http``, the package never does (``test_desktop_offline.py`` keeps that true).
"""

import json
import platform
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from curvestudio_desktop import INSTALL_RECORD_URL, __version__, install_record
from curvestudio_desktop.install_record import (
    DEFAULT_TIMEOUT_MS,
    INSTALL_ID_KEY,
    INSTALL_RECORD_KEY,
    INSTALL_RECORD_URL_ENV,
    LAST_SENT_KEY,
    PACKAGE_KINDS,
    QtTransport,
    build_payload,
    install_id,
    install_record_enabled,
    maybe_send_install_record,
    package_kind,
    set_install_record_enabled,
)

FIELDS = frozenset(
    {"install_id", "app_version", "os_name", "os_version", "arch", "package", "session"}
)
"""The body is exactly these keys: the server refuses an unknown one with 422."""

SILENCE_SECONDS = 5.0
"""How long the silent server holds a request without answering; a client must give up first."""

Request = tuple[str, dict[str, str], int, Callable[[bool], None]]
"""``(url, body, timeout in ms, the completion callback)`` of one posted record."""


class RecordingTransport:
    """Stands in for the network: records each request, answers from the test's script."""

    def __init__(self, *, ok: bool = True, settle: bool = True) -> None:
        self.ok = ok
        self.settle = settle  # False keeps the request "in flight" until the test says otherwise
        self.requests: list[Request] = []

    def post(
        self, url: str, body: bytes, *, timeout_ms: int, on_done: Callable[[bool], None]
    ) -> None:
        self.requests.append((url, json.loads(body), timeout_ms, on_done))
        if self.settle:
            on_done(self.ok)

    def payload(self, index: int = 0) -> dict[str, str]:
        return self.requests[index][1]

    def urls(self) -> list[str]:
        return [request[0] for request in self.requests]


@pytest.fixture(autouse=True)
def _a_clean_start(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts as a fresh launch would: no variable set, no request in flight.

    The directory's conftest switches the record off through the environment; here the
    destination is the constant again, and only the fakes ever "send". A test that
    exercises the variable sets it itself.
    """
    monkeypatch.delenv(INSTALL_RECORD_URL_ENV, raising=False)
    install_record._in_flight.clear()
    yield
    install_record._in_flight.clear()


# -- what is sent, and when ----------------------------------------------------------------------


def test_a_fresh_installation_sends_one_record_with_exactly_the_listed_fields(
    settings: QSettings,
) -> None:
    transport = RecordingTransport()

    assert maybe_send_install_record(settings, transport=transport) is True

    assert len(transport.requests) == 1
    url, body, timeout_ms, _done = transport.requests[0]
    assert url == INSTALL_RECORD_URL
    assert timeout_ms == DEFAULT_TIMEOUT_MS == 5_000
    assert set(body) == FIELDS
    assert all(isinstance(value, str) for value in body.values())
    assert body["app_version"] == __version__
    # the id is a version-4 UUID, and it is now stored: the same installation answers next time
    assert uuid.UUID(body["install_id"]).version == 4
    settings.sync()
    assert settings.value(INSTALL_ID_KEY) == body["install_id"]
    assert settings.value(LAST_SENT_KEY) == __version__


def test_the_same_version_sends_nothing_on_the_next_launch(settings: QSettings) -> None:
    assert maybe_send_install_record(settings, transport=RecordingTransport()) is True

    settings.sync()
    # a fresh settings object on the same file is what the next launch reads
    next_launch = QSettings(settings.fileName(), QSettings.Format.IniFormat)
    second = RecordingTransport()
    assert maybe_send_install_record(next_launch, transport=second) is False
    assert second.requests == []


def test_a_new_version_sends_one_more_record_with_the_same_installation_id(
    settings: QSettings,
) -> None:
    old, new = RecordingTransport(), RecordingTransport()

    assert maybe_send_install_record(settings, transport=old) is True
    assert maybe_send_install_record(settings, transport=new, app_version="99.0.0") is True

    assert len(new.requests) == 1
    assert new.payload()["install_id"] == old.payload()["install_id"]
    assert new.payload()["app_version"] == "99.0.0"
    settings.sync()
    assert settings.value(LAST_SENT_KEY) == "99.0.0"


def test_while_a_record_is_in_flight_a_second_window_sends_nothing(settings: QSettings) -> None:
    """Ctrl+N before the first answer must not double the installation count."""
    slow = RecordingTransport(settle=False)

    assert maybe_send_install_record(settings, transport=slow) is True
    assert maybe_send_install_record(settings, transport=RecordingTransport()) is False
    assert len(slow.requests) == 1

    _url, _body, _timeout, answered = slow.requests[0]
    answered(True)  # the server's 2xx arrives: the version is recorded, and stays recorded
    settings.sync()
    assert settings.value(LAST_SENT_KEY) == __version__
    assert maybe_send_install_record(settings, transport=RecordingTransport()) is False


# -- when nothing is sent, nothing is built ------------------------------------------------------


def test_with_the_switch_off_no_transport_is_ever_built(
    settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_install_record_enabled(False, settings)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a transport was built although the switch is off")

    monkeypatch.setattr(install_record, "QtTransport", forbidden)

    assert install_record_enabled(settings) is False
    assert maybe_send_install_record(settings) is False
    assert settings.contains(INSTALL_ID_KEY) is False  # not even the id is created


def test_no_transport_is_built_when_the_version_is_already_recorded(
    settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.setValue(LAST_SENT_KEY, __version__)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a transport was built although the version is already recorded")

    monkeypatch.setattr(install_record, "QtTransport", forbidden)
    assert maybe_send_install_record(settings) is False


def test_an_empty_url_means_never_send_and_the_environment_variable_overrides_the_constant(
    settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = RecordingTransport()
    override = "https://example.invalid/api/v1/installs"

    # an empty URL, given or in the environment, means never send
    assert maybe_send_install_record(settings, transport=transport, url="") is False
    monkeypatch.setenv(INSTALL_RECORD_URL_ENV, "")
    assert maybe_send_install_record(settings, transport=transport) is False
    assert transport.requests == []

    # the environment variable overrides the constant
    monkeypatch.setenv(INSTALL_RECORD_URL_ENV, override)
    assert maybe_send_install_record(settings, transport=transport) is True
    assert transport.urls() == [override]

    # an explicit url wins over the environment; an unset variable falls back to the constant
    settings.remove(LAST_SENT_KEY)
    assert maybe_send_install_record(settings, transport=transport, url=override + "/x") is True
    monkeypatch.delenv(INSTALL_RECORD_URL_ENV)
    settings.remove(LAST_SENT_KEY)
    assert maybe_send_install_record(settings, transport=transport) is True
    assert transport.urls() == [override, override + "/x", INSTALL_RECORD_URL]


# -- failures ------------------------------------------------------------------------------------


def test_a_failed_post_leaves_the_version_unrecorded_so_the_next_launch_tries_again(
    settings: QSettings,
) -> None:
    failed = RecordingTransport(ok=False)
    assert maybe_send_install_record(settings, transport=failed) is True
    settings.sync()
    assert settings.value(LAST_SENT_KEY) is None
    assert install_record_enabled(settings) is True  # a failure never flips the switch

    retried = RecordingTransport()
    assert maybe_send_install_record(settings, transport=retried) is True
    assert retried.payload()["install_id"] == failed.payload()["install_id"]
    settings.sync()
    assert settings.value(LAST_SENT_KEY) == __version__


def test_a_2xx_records_the_version(settings: QSettings) -> None:
    assert maybe_send_install_record(settings, transport=RecordingTransport(ok=True)) is True
    settings.sync()
    assert settings.value(LAST_SENT_KEY) == __version__


@pytest.mark.parametrize("status", [429, 422])
def test_a_429_and_a_422_count_as_failures_so_the_version_stays_unrecorded(
    qtbot, settings: QSettings, status: int
) -> None:
    """Through the real transport: rate-limited and refused answers are failures, not records."""
    with serving(status) as (url, _received):
        assert maybe_send_install_record(settings, url=url) is True
        qtbot.waitUntil(lambda: not install_record._in_flight, timeout=10_000)
    settings.sync()
    assert settings.value(LAST_SENT_KEY) is None
    assert settings.contains(INSTALL_ID_KEY) is True  # the installation keeps its id for the retry


# -- the real transport, against a local server --------------------------------------------------


@contextmanager
def serving(status: int | None) -> Iterator[tuple[str, list[tuple[str, str | None, bytes]]]]:
    """A local server answering every POST with ``status`` and no body; ``None`` never answers."""
    received: list[tuple[str, str | None, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            received.append((self.path, self.headers.get("Content-Type"), self.rfile.read(length)))
            if status is None:
                time.sleep(SILENCE_SECONDS)  # the client must give up long before this
                return
            self.send_response(status)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            pass  # the test log does not need the server's request lines

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/installs", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post_and_wait(qtbot, url: str, body: bytes = b"{}", *, timeout_ms: int = 2000) -> bool:
    """One real POST; returns what the transport reported once the server was done."""
    transport = QtTransport(timeout_ms=timeout_ms)
    answers: list[bool] = []
    transport.post(url, body, timeout_ms=timeout_ms, on_done=answers.append)
    qtbot.waitUntil(lambda: bool(answers), timeout=10_000)
    return answers[0]


@pytest.mark.parametrize("status", [200, 204])
def test_the_real_transport_posts_json_and_reports_a_2xx(qtbot, status: int) -> None:
    body = json.dumps({"install_id": "00000000-0000-4000-8000-000000000000"}).encode()

    with serving(status) as (url, received):
        assert post_and_wait(qtbot, url, body) is True

    assert received == [("/installs", "application/json", body)]


def test_the_real_transport_reports_a_failure_on_a_timeout(qtbot) -> None:
    with serving(None) as (url, _received):
        started = time.monotonic()
        assert post_and_wait(qtbot, url, timeout_ms=300) is False
        elapsed = time.monotonic() - started
    assert elapsed < SILENCE_SECONDS  # it gave up on its own; the server never answered


def test_the_real_transport_reports_a_failure_when_nothing_listens(qtbot) -> None:
    with socket.socket() as probe:  # a port that was free a moment ago, with no listener behind it
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    assert post_and_wait(qtbot, f"http://127.0.0.1:{port}/installs") is False


def test_the_real_transport_falls_back_to_its_own_timeout(qtbot) -> None:
    """A post that names no timeout uses the transport's own (five seconds unless overridden)."""
    assert QtTransport()._timeout_ms == DEFAULT_TIMEOUT_MS
    with serving(204) as (url, _received):
        transport = QtTransport(timeout_ms=2000)
        answers: list[bool] = []
        transport.post(url, b"{}", on_done=answers.append)
        qtbot.waitUntil(lambda: bool(answers), timeout=10_000)
    assert answers == [True]


# -- the payload's words -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "machine", "xdg", "expected"),
    [
        # (os_name, arch, session)
        ("win32", "AMD64", "x11", ("windows", "x86_64", "windows")),
        ("win32", "", "wayland", ("windows", "other", "windows")),
        ("darwin", "arm64", "wayland", ("macos", "arm64", "other")),  # XDG counts on Linux only
        ("darwin", "x86_64", "", ("macos", "x86_64", "other")),
        ("linux", "x86_64", "wayland", ("linux", "x86_64", "wayland")),
        ("linux", "aarch64", "X11", ("linux", "arm64", "x11")),
        ("linux", "x86_64", "", ("linux", "x86_64", "other")),
        ("linux", "riscv64", "tty", ("linux", "other", "other")),
        ("freebsd14", "sparc", "wayland", ("other", "other", "other")),
        ("", "i686", None, ("other", "other", "other")),
    ],
)
def test_the_payload_uses_the_servers_words(
    settings: QSettings,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    xdg: str | None,
    expected: tuple[str, str, str],
) -> None:
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(platform, "release", lambda: "6.8.0-45-generic")
    monkeypatch.setattr(platform, "version", lambda: "#45-Ubuntu SMP")
    if xdg is None:
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    else:
        monkeypatch.setenv("XDG_SESSION_TYPE", xdg)

    payload = build_payload(settings)

    assert (payload["os_name"], payload["arch"], payload["session"]) == expected
    assert payload["os_version"] == "6.8.0-45-generic"
    assert payload["package"] in PACKAGE_KINDS
    assert all(isinstance(value, str) and value for value in payload.values()), payload
    assert len(json.dumps(payload).encode()) < 2048


def test_the_os_version_falls_back_and_is_cut_to_forty_characters(
    settings: QSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "release", lambda: "")
    monkeypatch.setattr(platform, "version", lambda: "#45-Ubuntu SMP")
    assert build_payload(settings)["os_version"] == "#45-Ubuntu SMP"

    monkeypatch.setattr(platform, "release", lambda: "6.8.0-45-generic-" + "x" * 60)
    assert build_payload(settings)["os_version"] == ("6.8.0-45-generic-" + "x" * 60)[:40]

    # nothing known is still a word, never an empty string
    monkeypatch.setattr(platform, "release", lambda: "")
    monkeypatch.setattr(platform, "version", lambda: "")
    assert build_payload(settings)["os_version"]


def test_an_app_version_longer_than_forty_characters_is_cut(settings: QSettings) -> None:
    payload = build_payload(settings, app_version="9." + "0" * 60)
    assert payload["app_version"] == "9." + "0" * 38


# -- the settings behind the record --------------------------------------------------------------


def test_the_installation_id_is_created_once_and_reused(settings: QSettings) -> None:
    first = install_id(settings)
    assert first.version == 4
    assert install_id(settings) == first
    settings.sync()
    reopened = QSettings(settings.fileName(), QSettings.Format.IniFormat)
    assert reopened.value(INSTALL_ID_KEY) == str(first)


def test_an_unreadable_stored_installation_id_is_replaced(settings: QSettings) -> None:
    settings.setValue(INSTALL_ID_KEY, "not a uuid")
    replaced = install_id(settings)
    assert replaced.version == 4
    settings.sync()
    assert settings.value(INSTALL_ID_KEY) == str(replaced)


def test_the_switch_is_on_by_default_and_reads_back_what_was_written(settings: QSettings) -> None:
    assert install_record_enabled(settings) is True
    assert settings.contains(INSTALL_RECORD_KEY) is False

    set_install_record_enabled(False, settings)
    assert settings.value(INSTALL_RECORD_KEY) is False
    assert install_record_enabled(settings) is False

    set_install_record_enabled(True, settings)
    assert settings.value(INSTALL_RECORD_KEY) is True
    assert install_record_enabled(settings) is True

    # a settings file edited by hand
    for value in ("false", "0", 0):
        settings.setValue(INSTALL_RECORD_KEY, value)
        assert install_record_enabled(settings) is False, value
    for value in ("true", "yes", 1):
        settings.setValue(INSTALL_RECORD_KEY, value)
        assert install_record_enabled(settings) is True, value


# -- which package the app came from -------------------------------------------------------------


def test_the_package_word_comes_from_the_file_beside_the_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_file = tmp_path / "package.txt"
    monkeypatch.setattr(install_record, "__file__", str(tmp_path / "install_record.py"))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "curvestudio-desktop"))

    def frozen(status: bool) -> None:
        monkeypatch.setattr(sys, "frozen", status, raising=False)

    # running from source: no file is the normal case
    frozen(False)
    assert package_kind() == "source"
    package_file.write_text("deb\n", encoding="utf-8")
    assert package_kind() == "deb"
    package_file.write_text("AppImage\n", encoding="utf-8")
    assert package_kind() == "appimage"
    package_file.write_text("snap", encoding="utf-8")  # not one of the words
    assert package_kind() == "other"

    # frozen (a PyInstaller build): no file is no longer "source"
    package_file.unlink()
    frozen(True)
    assert package_kind() == "other"
    package_file.write_text("windows_setup\n", encoding="utf-8")
    assert package_kind() == "windows_setup"

    # a file that cannot be read counts as absent
    package_file.unlink()
    package_file.mkdir()
    assert package_kind() == "other"
