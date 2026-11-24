"""Runs upload parsing in a short-lived process, with concurrency limits and a time budget."""

import io
import json
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

from curvestudio_api.errors import ApiError
from curvestudio_api.import_worker import ImportWorker, import_process_main, unpack_payloads
from curvestudio_api.logging_setup import get_logger
from curvestudio_api.worker import STARTED

_log = get_logger("imports")

MAX_XLSX_EXPANDED_BYTES = 100 * 1024 * 1024
"""A spreadsheet whose entries declare more than this once uncompressed is refused unread."""
MAX_XLSX_ENTRIES = 10_000
STARTUP_LIMIT_S = 60.0
REAP_TIMEOUT_S = 5.0
_EOCD = b"PK\x05\x06"
_META_LIMIT = 8 * 1024 * 1024


def _too_large(message: str) -> ApiError:
    return ApiError(413, "TOO_LARGE", message, {"code": "TOO_LARGE"})


def precheck(raw: bytes, filename: str, max_rows: int) -> None:
    """Cheap refusals that need no parsing and no process.

    * ``.xlsx``: the zip's central directory is read, and a file declaring more than
      ``MAX_XLSX_EXPANDED_BYTES`` uncompressed (or more than ``MAX_XLSX_ENTRIES`` entries) is
      refused before anything is expanded.
    * ``.csv``/``.tsv``/``.txt``: more lines than ``max_rows`` plus header and a trailing
      newline is refused without looking at a single cell.

    Raises:
        ApiError: 413 ``TOO_LARGE``.
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "xlsx":
        _check_zip(raw)
    elif suffix in ("csv", "tsv", "txt"):
        lines = max(raw.count(b"\n"), raw.count(b"\r"))
        if lines > max_rows + 2:
            raise _too_large(f"the file has more than {max_rows:,} data rows")


def _check_zip(raw: bytes) -> None:
    tail = raw[-(65_535 + 22) :]
    at = tail.rfind(_EOCD)
    if at < 0 or len(tail) - at < 22:
        return  # not a zip: the parser will say so
    entries = struct.unpack_from("<H", tail, at + 10)[0]
    if entries > MAX_XLSX_ENTRIES:
        raise _too_large("the spreadsheet contains too many internal files")
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            declared = sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile, ValueError, OSError:
        return  # unreadable archive: left to the parser, which reports it
    if declared > MAX_XLSX_EXPANDED_BYTES:
        raise _too_large(
            f"the spreadsheet expands to {declared:,} bytes; the limit is "
            f"{MAX_XLSX_EXPANDED_BYTES:,} bytes uncompressed"
        )


@dataclass(frozen=True)
class ImportRun:
    """What the import process returned: metadata (JSON), the stored CSV of the fit's input
    and the stored CSV of the table's other columns (empty when the table has none)."""

    meta: dict[str, Any]
    payload: bytes
    table: bytes = b""


class ImportRunner:
    """At most ``concurrency`` imports at once in the whole API and one per user."""

    def __init__(
        self,
        *,
        concurrency: int,
        timeout_s: float,
        memory_mb: int,
        max_payload_bytes: int,
        worker: ImportWorker = import_process_main,
    ) -> None:
        self._concurrency = concurrency
        self._timeout_s = timeout_s
        self._memory_mb = memory_mb
        self._max_payload_bytes = max_payload_bytes
        """The most one result message may hold: the fit's and the table's payload together."""
        self._worker = worker
        self._active: set[uuid.UUID] = set()
        self._lock = threading.Lock()
        self._context = get_context("spawn")
        self.spawned = 0
        """Import processes started so far (for tests and diagnostics)."""

    def run(self, user_id: uuid.UUID, arguments: dict[str, Any]) -> ImportRun:
        """Parse in a child process and return its result.

        Raises:
            ApiError: 429 ``TOO_MANY_UPLOADS`` (this user already has an import running, or
                the server is at its limit), 413 ``TOO_LARGE`` (memory cap hit), 422 (timeout
                or a parser process that died).
        """
        with self._lock:
            if user_id in self._active or len(self._active) >= self._concurrency:
                raise ApiError(
                    429,
                    "TOO_MANY_UPLOADS",
                    "another upload is being processed; try again in a moment",
                    {"code": "TOO_MANY_UPLOADS"},
                )
            self._active.add(user_id)
            self.spawned += 1
        try:
            return self._run(arguments)
        finally:
            with self._lock:
                self._active.discard(user_id)

    def _run(self, arguments: dict[str, Any]) -> ImportRun:
        receiver, sender = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=self._worker,
            args=(sender, self._memory_mb, arguments),
            name="curvestudio-import",
            daemon=True,
        )
        try:
            process.start()
        finally:
            sender.close()
        started = time.monotonic()
        running_since: float | None = None
        try:
            return self._wait(process, receiver, started, running_since)
        finally:
            if process.is_alive():
                process.kill()
            process.join(REAP_TIMEOUT_S)
            process.close()
            receiver.close()

    def _wait(
        self, process: Any, receiver: Any, started: float, running_since: float | None
    ) -> ImportRun:
        meta: dict[str, Any] | None = None
        while True:
            now = time.monotonic()
            limit = (
                running_since + self._timeout_s
                if running_since is not None
                else started + STARTUP_LIMIT_S + self._timeout_s
            )
            if now > limit:
                raise ApiError(
                    422,
                    "VALIDATION_ERROR",
                    f"reading the file took longer than {self._timeout_s:g} s; send a smaller file",
                    {"code": "IMPORT_TIMEOUT"},
                )
            try:
                if receiver.poll(0.05):
                    data = receiver.recv_bytes(
                        maxlength=_META_LIMIT if meta is None else self._max_payload_bytes
                    )
                    if data == STARTED and running_since is None:
                        running_since = time.monotonic()
                    elif meta is None:
                        meta = json.loads(data)
                    else:
                        return self._finish(meta, data)
                elif not process.is_alive() and not receiver.poll(0):
                    raise EOFError
            except (EOFError, OSError, ValueError) as exc:
                _log.error("import process died", extra={"fields": {"exit": process.exitcode}})
                raise ApiError(
                    422,
                    "VALIDATION_ERROR",
                    "the file could not be read",
                    {"code": "IMPORT_FAILED"},
                ) from exc

    @staticmethod
    def _finish(meta: dict[str, Any], message: bytes) -> ImportRun:
        """The run a child reported: ``message`` holds both payloads (see ``pack_payloads``).

        A message of another shape raises :class:`ValueError`, which :meth:`_wait` reports as
        a parser that died: nothing of it is stored.
        """
        status = meta.get("status")
        if status == "too_much_memory":
            raise _too_large("the file needs more memory to read than is allowed")
        if status == "failed":
            raise ApiError(
                422, "VALIDATION_ERROR", "the file could not be read", {"code": "IMPORT_FAILED"}
            )
        payload, table = unpack_payloads(message)
        return ImportRun(meta, payload, table)
