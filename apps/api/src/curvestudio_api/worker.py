"""The fit job's own code, and the wire markers shared by every job-process path.

The process machinery — warm workers, the forked children that run these functions, the
kill and recycle protocol around them — lives in :mod:`curvestudio_api.worker_pool`; the
upload parser keeps its own short-lived child in :mod:`curvestudio_api.import_worker`.
"""

from collections.abc import Callable
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
