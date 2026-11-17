"""The code that runs inside a job process. Top-level and picklable (``spawn`` start)."""

import json
import multiprocessing
import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from curvestudio.data import XY
from curvestudio.workflow import FitSpec, WorkflowError, fit_dataset
from curvestudio_api.storage import decode_payload

JobWorker = Any
"""``Callable[..., dict[str, Any]]`` taking the keyword arguments of :func:`run_fit_job`."""

STARTED = b"S"
"""First message of a job process: it is up (its imports are done) and about to work."""

PROGRESS = b"P"
"""Prefix of a progress message (``PROGRESS`` + JSON). A result is JSON, so it starts with ``{``."""

ProgressReporter = Callable[[dict[str, Any]], None]
"""What an Auto-fit worker calls with each progress snapshot."""


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


def _lead_own_process_group() -> bool:
    """Make this process the leader of a new session, so the parent can stop it and everything
    it started (the Auto-fit worker pool) with one ``killpg``. POSIX only; false elsewhere."""
    setsid = getattr(os, "setsid", None)
    if setsid is None:
        return False
    try:
        setsid()
    except OSError:
        return False  # the parent falls back to killing this process alone
    return True


def _exit_with_parent(leads_group: bool) -> None:
    """Die, together with every child, when the API process disappears without a clean stop.

    A job process that has children is not a daemon, so nothing else would end it (or its
    pool) after a hard kill of the API process.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()  # returns when the parent has exited
        kill_group = getattr(os, "killpg", None)
        if leads_group and kill_group is not None:
            kill_group(os.getpgrp(), signal.SIGKILL)
        os._exit(1)

    threading.Thread(target=watch, name="curvestudio-parent-watch", daemon=True).start()


def job_process_main(
    connection: Connection,
    worker: JobWorker,
    arguments: dict[str, Any],
    reports_progress: bool = False,
) -> None:
    """Entry point of one job's process: say hello, run ``worker``, send its JSON result.

    The parent counts the job's wall-clock budget from the hello, so a slow process start does
    not eat into it. An exception in ``worker`` is reported as ``INTERNAL_ERROR`` (the
    traceback goes to stderr); a hard crash shows up to the parent as a closed pipe. With
    ``reports_progress`` the worker also gets ``report=`` and every snapshot it passes is
    forwarded to the parent as a progress message.
    """
    leads_group = _lead_own_process_group()
    _exit_with_parent(leads_group)
    try:
        connection.send_bytes(STARTED)
        if reports_progress:
            arguments = {
                **arguments,
                "report": lambda snapshot: connection.send_bytes(
                    PROGRESS + json.dumps(snapshot, allow_nan=False).encode()
                ),
            }
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
