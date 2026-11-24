"""The code that parses an upload inside its own short-lived process (``spawn`` start).

Parsing untrusted spreadsheets and tables is the one place where a small upload can cost a
lot of CPU and memory, so it never runs in the API process. The child limits its own address
space (best effort, POSIX only), reports back with JSON plus the raw stored bytes (the fit
input's and the table's, in one message), and never sends pickles: the parent must not
deserialise anything the parser could influence.
"""

import json
import struct
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Any

from curvestudio.workflow import ImportLimits, WorkflowError, import_dataset
from curvestudio_api.storage import encode_payload, encode_table, table_column_list
from curvestudio_api.worker import STARTED

MAX_ISSUE_ROWS = 100
"""Row numbers kept per data issue; the message already says what is wrong."""
MIB = 1024 * 1024
_LENGTH = struct.Struct(">Q")
"""The fit payload's length at the head of the last message; the table payload follows it."""

ImportWorker = Any
"""``Callable[[Connection, int, dict[str, Any]], None]``: see :func:`import_process_main`."""


def pack_payloads(fit: bytes, table: bytes) -> bytes:
    """The last message of an import process: the fit payload's length, then both payloads.

    Both travel in one message, so the parent's size cap covers the two together.
    """
    return _LENGTH.pack(len(fit)) + fit + table


def unpack_payloads(message: bytes) -> tuple[bytes, bytes]:
    """The fit payload and the table payload of a message from :func:`pack_payloads`.

    Raises:
        ValueError: when the message is not of that shape: shorter than its length prefix,
            or naming a fit payload longer than what follows.
    """
    if len(message) < _LENGTH.size:
        raise ValueError(f"an import result is {len(message)} byte(s) long, not at least 8")
    (fit_length,) = _LENGTH.unpack_from(message)
    held = len(message) - _LENGTH.size
    if fit_length > held:
        raise ValueError(
            f"an import result names a fit payload of {fit_length:,} byte(s) and holds {held:,}"
        )
    return message[_LENGTH.size : _LENGTH.size + fit_length], message[_LENGTH.size + fit_length :]


def _vm_size_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError, ValueError:
        return None
    return None


def limit_address_space(headroom_mb: int) -> bool:
    """Cap the process' address space at what it uses now plus ``headroom_mb``.

    Returns whether a limit was set; where the platform has no ``RLIMIT_AS`` (or no way to
    read the current size) the parser still runs, bounded by the parent's wall-clock timeout.
    """
    try:
        import resource

        current = _vm_size_bytes()
        if current is None:
            return False
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = current + headroom_mb * MIB
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except ImportError, ValueError, OSError:
        return False
    return True


def run_import(
    *,
    raw: bytes,
    filename: str,
    x_column: str | None,
    y_column: str | None,
    weight_column: str | None,
    max_bytes: int,
    max_rows: int,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Parse ``raw``; returns ``(metadata, payload, table)``: the stored CSVs of the fit's
    input and of the table's other columns.

    Both payloads are empty unless ``metadata["status"]`` is ``ok``. The status is also
    ``unusable`` (the table is fine but the chosen data cannot be fitted) or ``error`` (a
    :class:`WorkflowError`, with its ``code``). An ``ok`` metadata carries ``table_columns``:
    every column of the table with its kind and blank cells, computed here once so that no
    route has to read the stored table back.
    """
    try:
        outcome = import_dataset(
            raw,
            filename,
            x_column=x_column,
            y_column=y_column,
            weight_column=weight_column,
            limits=ImportLimits(max_bytes=max_bytes, max_rows=max_rows),
        )
    except WorkflowError as exc:
        return {"status": "error", "code": exc.code, "message": exc.message}, b"", b""
    issues = [
        {
            "code": i.code,
            "severity": i.severity,
            "message": i.message,
            "rows": list(i.rows[:MAX_ISSUE_ROWS]),
        }
        for i in outcome.issues
    ]
    if outcome.xy is None or outcome.x_column is None or outcome.y_column is None:
        return (
            {
                "status": "unusable",
                "columns": list(outcome.columns),
                "issues": issues,
            },
            b"",
            b"",
        )
    x, y, weight = outcome.x_column, outcome.y_column, outcome.weight_column
    n_rows = int(outcome.xy.x.size)
    table = (
        encode_table(outcome.dataset, x_column=x, y_column=y, weight_column=weight, n_rows=n_rows)
        or b""
    )
    meta = {
        "status": "ok",
        "name": outcome.dataset.name,
        "n_rows": n_rows,
        "columns": list(outcome.columns),
        "x_column": x,
        "y_column": y,
        "weight_column": weight,
        "issues": issues,
        "fingerprint": outcome.xy.fingerprint,
        "table_columns": table_column_list(
            list(outcome.columns),
            x_column=x,
            y_column=y,
            weight_column=weight,
            table_payload=table,
        ),
    }
    return meta, encode_payload(outcome.xy), table


def import_process_main(connection: Connection, memory_mb: int, arguments: dict[str, Any]) -> None:
    """Entry point of the import process: hello, cap memory, parse, send metadata, then both
    payloads in one message."""
    try:
        connection.send_bytes(STARTED)
        limit_address_space(memory_mb)
        try:
            meta, payload, table = run_import(**arguments)
        except MemoryError:
            meta, payload, table = {"status": "too_much_memory"}, b"", b""
        except Exception:
            traceback.print_exc(file=sys.stderr)
            meta, payload, table = {"status": "failed"}, b"", b""
        connection.send_bytes(json.dumps(meta, allow_nan=False).encode())
        connection.send_bytes(pack_payloads(payload, table))
    finally:
        connection.close()
