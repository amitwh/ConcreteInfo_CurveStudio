"""The code that runs inside a job process. Top-level and picklable (``spawn`` start)."""

import json
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Any

from curvestudio.data import XY
from curvestudio.workflow import FitSpec, WorkflowError, fit_dataset
from curvestudio_api.storage import decode_payload

JobWorker = Any
"""``Callable[..., dict[str, Any]]`` taking the keyword arguments of :func:`run_fit_job`."""

STARTED = b"S"
"""First message of a job process: it is up (its imports are done) and about to work."""


def run_fit_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Fit one stored dataset. Returns ``{"ok": True, "bundle": <FitBundle JSON>}`` or, for a
    specification the engine rejects, ``{"ok": False, "code": ..., "message": ...}``.

    A fit that does not converge is not an error: it is a bundle whose status says so.
    """
    xy: XY = decode_payload(payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint)
    try:
        bundle = fit_dataset(xy, FitSpec.model_validate(spec), compact=True)
    except WorkflowError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    return {"ok": True, "bundle": bundle.model_dump(mode="json")}


def job_process_main(connection: Connection, worker: JobWorker, arguments: dict[str, Any]) -> None:
    """Entry point of one job's process: say hello, run ``worker``, send its JSON result.

    The parent counts the job's wall-clock budget from the hello, so a slow process start does
    not eat into it. An exception in ``worker`` is reported as ``INTERNAL_ERROR`` (the
    traceback goes to stderr); a hard crash shows up to the parent as a closed pipe.
    """
    try:
        connection.send_bytes(STARTED)
        try:
            result: dict[str, Any] = worker(**arguments)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            result = {
                "ok": False,
                "code": "INTERNAL_ERROR",
                "message": "the fit job failed unexpectedly",
            }
        connection.send_bytes(json.dumps(result, allow_nan=False).encode())
    finally:
        connection.close()
