"""The code that parses an upload inside its own short-lived process (``spawn`` start).

Parsing untrusted spreadsheets and tables is the one place where a small upload can cost a
lot of CPU and memory, so it never runs in the API process. The child limits its own address
space (best effort, POSIX only), reports back with JSON plus the raw cleaned-data bytes, and
never sends pickles: the parent must not deserialise anything the parser could influence.
"""

import json
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Any

from curvestudio.workflow import ImportLimits, WorkflowError, import_dataset
from curvestudio_api.storage import encode_payload
from curvestudio_api.worker import STARTED

MAX_ISSUE_ROWS = 100
"""Row numbers kept per data issue; the message already says what is wrong."""
MIB = 1024 * 1024

ImportWorker = Any
"""``Callable[[Connection, int, dict[str, Any]], None]``: see :func:`import_process_main`."""


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
) -> tuple[dict[str, Any], bytes]:
    """Parse ``raw``; returns ``(metadata, payload)`` where ``payload`` is the stored CSV.

    ``metadata["status"]`` is ``ok``, ``unusable`` (the table is fine but the chosen data
    cannot be fitted) or ``error`` (a :class:`WorkflowError`, with its ``code``).
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
        return {"status": "error", "code": exc.code, "message": exc.message}, b""
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
        return {
            "status": "unusable",
            "columns": list(outcome.columns),
            "issues": issues,
        }, b""
    meta = {
        "status": "ok",
        "name": outcome.dataset.name,
        "n_rows": int(outcome.xy.x.size),
        "columns": list(outcome.columns),
        "x_column": outcome.x_column,
        "y_column": outcome.y_column,
        "weight_column": outcome.weight_column,
        "issues": issues,
        "fingerprint": outcome.xy.fingerprint,
    }
    return meta, encode_payload(outcome.xy)


def import_process_main(connection: Connection, memory_mb: int, arguments: dict[str, Any]) -> None:
    """Entry point of the import process: hello, cap memory, parse, send metadata then payload."""
    try:
        connection.send_bytes(STARTED)
        limit_address_space(memory_mb)
        try:
            meta, payload = run_import(**arguments)
        except MemoryError:
            meta, payload = {"status": "too_much_memory"}, b""
        except Exception:
            traceback.print_exc(file=sys.stderr)
            meta, payload = {"status": "failed"}, b""
        connection.send_bytes(json.dumps(meta, allow_nan=False).encode())
        connection.send_bytes(payload)
    finally:
        connection.close()
