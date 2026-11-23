"""The one network call the desktop app makes: an anonymous install record, once per version.

On first launch, and again after an upgrade to a new version, the app posts one small JSON
body to ``INSTALL_RECORD_URL``: a random installation id it generated once, the app
version, the operating system and its version, the architecture, which package the app
came from (``deb``, ``appimage``, ``tarball``, ``windows_zip``, ``windows_setup`` or
``source``) and the session type. Nothing else ever leaves the computer: no account (there
is none), no name, no project, no error report.

The rules, all enforced by :func:`maybe_send_install_record`:

* The switch ``privacy/install_record`` (Edit → Preferences → Privacy) is on unless the
  user turned it off. Off means not even a transport is built.
* ``telemetry/last_sent_version`` holds the version the server accepted. While it equals
  the running version nothing is sent. A failed post stores nothing, so the next launch
  simply tries again.
* One request per version at a time: a second project window opened before the first
  answer does not double the count.
* A failure is silent. No dialog, no log line: an install record must never be a reason
  for the app to misbehave.

This is the only module of the package that imports ``PySide6.QtNetwork``.
``tests/desktop/test_desktop_offline.py`` pins that, and that this module names no second
destination; Python's own network stacks (``http``, ``urllib``, ``socket``) stay forbidden
there. Settings are read on the GUI thread, as everywhere else in the app.
"""

import json
import os
import platform
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QByteArray, QSettings, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from curvestudio_desktop import INSTALL_RECORD_URL, __version__

INSTALL_ID_KEY = "telemetry/install_id"
LAST_SENT_KEY = "telemetry/last_sent_version"
INSTALL_RECORD_KEY = "privacy/install_record"

INSTALL_RECORD_URL_ENV = "CURVESTUDIO_INSTALL_RECORD_URL"
"""Overrides the destination (a test, a private deployment); an empty value means never send."""

DEFAULT_TIMEOUT_MS = 5_000
"""How long a request may go unanswered before the transport gives up on it."""

MAX_VERSION_LENGTH = 40
"""The server's limit for ``app_version`` and ``os_version``."""

UNKNOWN = "other"
"""What the server stores for any value outside its list. The payload never holds a blank."""

OS_NAMES = {"win32": "windows", "darwin": "macos", "linux": "linux"}
ARCHITECTURES = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}
LINUX_SESSIONS = ("x11", "wayland")
PACKAGE_KINDS = ("deb", "appimage", "tarball", "windows_setup", "windows_zip", "source", UNKNOWN)
PACKAGE_FILE = "package.txt"
"""Written by the packaging beside the frozen executable: which artifact the app came from."""


class Transport(Protocol):
    """What the install record needs from a network: one JSON POST, and one answer."""

    def post(
        self, url: str, body: bytes, *, timeout_ms: int, on_done: Callable[[bool], None]
    ) -> None:
        """Send ``body`` to ``url``; later, call ``on_done`` with whether the server said 2xx."""


class QtTransport:
    """The real transport: one ``QNetworkAccessManager``, a JSON content type, a timeout.

    Every outcome that is not a 2xx from the server reaches ``on_done`` as ``False``: a
    refused or rate-limited answer, a connection error, a request that went unanswered
    for ``timeout_ms``. ``on_done`` runs on the thread that called ``post`` (Qt delivers a
    reply on its manager's thread), the GUI thread in the app. The transport keeps itself
    and its replies alive until then, so the caller need not hold a reference.
    """

    def __init__(self, *, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self._timeout_ms = timeout_ms
        self._manager = QNetworkAccessManager()
        self._replies: set[QNetworkReply] = set()

    def post(
        self,
        url: str,
        body: bytes,
        *,
        timeout_ms: int | None = None,
        on_done: Callable[[bool], None],
    ) -> None:
        request = QNetworkRequest(QUrl(url))
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, QByteArray(b"application/json")
        )
        # An inactivity limit, counted from the start of the request: a name that never
        # resolves, a connection that never opens, a server that never answers.
        request.setTransferTimeout(self._timeout_ms if timeout_ms is None else timeout_ms)
        reply = self._manager.post(request, body)
        self._replies.add(reply)
        reply.finished.connect(lambda: self._settled(reply, on_done))

    def _settled(self, reply: QNetworkReply, on_done: Callable[[bool], None]) -> None:
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        ok = (
            reply.error() == QNetworkReply.NetworkError.NoError
            and isinstance(status, int)
            and 200 <= status < 300
        )
        self._replies.discard(reply)
        reply.deleteLater()
        on_done(ok)


def _store(settings: QSettings | None) -> QSettings:
    return settings if settings is not None else QSettings()


def _is_off(value: object) -> bool:
    """Whether a stored switch value means off: ``False`` from the dialog, ``false`` from a file."""
    return value is False or str(value).strip().lower() in {"false", "0"}


def install_record_enabled(settings: QSettings | None = None) -> bool:
    """Whether the install record may be sent: true unless the switch holds ``false``."""
    return not _is_off(_store(settings).value(INSTALL_RECORD_KEY))


def set_install_record_enabled(enabled: bool, settings: QSettings | None = None) -> None:
    """Store the switch; the Preferences dialog's OK calls this."""
    store = _store(settings)
    store.setValue(INSTALL_RECORD_KEY, bool(enabled))
    store.sync()


def install_id(settings: QSettings | None = None) -> uuid.UUID:
    """The installation's id: a version-4 UUID, created and saved the first time it is asked for."""
    store = _store(settings)
    try:
        return uuid.UUID(str(store.value(INSTALL_ID_KEY, "")))
    except ValueError:
        pass  # never created, or a value that is not a UUID: make a new one
    created = uuid.uuid4()
    store.setValue(INSTALL_ID_KEY, str(created))
    store.sync()
    return created


def _package_dir() -> Path:
    """Where ``package.txt`` lives: beside the frozen executable, otherwise beside this module."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def package_kind() -> str:
    """Which artifact the app came from, read from ``package.txt`` beside the executable.

    The packaging leaves one word there per artifact. Running from source there is no
    file, and the answer is ``source``. A frozen build without the file, or with a word
    that is not one of the server's, is ``other``.
    """
    frozen = bool(getattr(sys, "frozen", False))
    try:
        word = (_package_dir() / PACKAGE_FILE).read_text(encoding="utf-8").strip().lower()
    except OSError, UnicodeDecodeError:
        return UNKNOWN if frozen else "source"
    return word if word in PACKAGE_KINDS else UNKNOWN


def _session_type() -> str:
    if sys.platform == "linux":
        session = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
        if session in LINUX_SESSIONS:
            return session
    if sys.platform == "win32":
        return "windows"
    return UNKNOWN


def build_payload(
    settings: QSettings | None = None, *, app_version: str = __version__
) -> dict[str, str]:
    """The seven strings the server accepts, in its words: nothing outside a list, nothing blank."""
    os_version = (platform.release() or platform.version()).strip()[:MAX_VERSION_LENGTH]
    return {
        "install_id": str(install_id(settings)),
        "app_version": app_version[:MAX_VERSION_LENGTH],
        "os_name": OS_NAMES.get(sys.platform, UNKNOWN),
        "os_version": os_version or UNKNOWN,
        "arch": ARCHITECTURES.get(platform.machine().strip().lower(), UNKNOWN),
        "package": package_kind(),
        "session": _session_type(),
    }


_transport: QtTransport | None = None
"""The app's one transport, built the first time a record is actually sent."""

_in_flight: set[str] = set()
"""The versions whose record is out and unanswered, so no second one goes for the same version."""


def maybe_send_install_record(
    settings: QSettings | None = None,
    *,
    transport: Transport | None = None,
    app_version: str = __version__,
    url: str | None = None,
) -> bool:
    """Start the install record if this launch owes one; return whether a request was started.

    Call on the GUI thread. The destination is ``url``, else the ``CURVESTUDIO_INSTALL_RECORD_URL``
    environment variable, else :data:`INSTALL_RECORD_URL`; an empty one means never send. A
    request starts only when the switch is on, the version is not yet recorded and no request
    for it is unanswered. In every other case nothing is built, not even a transport. A 2xx
    records the version; any other outcome, including a transport that fails on the spot,
    is silence.
    """
    global _transport

    default_target = os.environ.get(INSTALL_RECORD_URL_ENV, INSTALL_RECORD_URL)
    target = url if url is not None else default_target
    if not target:
        return False
    store = _store(settings)
    if (
        not install_record_enabled(store)
        or store.value(LAST_SENT_KEY) == app_version
        or app_version in _in_flight
    ):
        return False
    if transport is None:
        if _transport is None:
            _transport = QtTransport()
        transport = _transport

    payload = build_payload(store, app_version=app_version)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    _in_flight.add(app_version)

    def done(accepted: bool) -> None:
        _in_flight.discard(app_version)
        if accepted:
            store.setValue(LAST_SENT_KEY, app_version)
            store.sync()

    try:
        transport.post(target, body, timeout_ms=DEFAULT_TIMEOUT_MS, on_done=done)
    except Exception:
        # The record is optional by design; a transport that cannot even start a request
        # must not reach the crash guard at start-up. The version stays unrecorded.
        _in_flight.discard(app_version)
        return False
    return True


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_TIMEOUT_MS",
    "INSTALL_ID_KEY",
    "INSTALL_RECORD_KEY",
    "INSTALL_RECORD_URL_ENV",
    "LAST_SENT_KEY",
    "LINUX_SESSIONS",
    "MAX_VERSION_LENGTH",
    "OS_NAMES",
    "PACKAGE_FILE",
    "PACKAGE_KINDS",
    "UNKNOWN",
    "QtTransport",
    "Transport",
    "build_payload",
    "install_id",
    "install_record_enabled",
    "maybe_send_install_record",
    "package_kind",
    "set_install_record_enabled",
]
