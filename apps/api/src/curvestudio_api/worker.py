"""The code that runs inside a worker process. Top-level and picklable (``spawn`` start)."""

from pathlib import Path
from typing import Any

from curvestudio.data import XY
from curvestudio.workflow import FitSpec, WorkflowError, fit_dataset
from curvestudio_api.storage import decode_payload

JobWorker = Any
"""``Callable[..., dict[str, Any]]`` taking the keyword arguments of :func:`run_fit_job`."""


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


def run_with_marker(worker: JobWorker, marker: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Signal that the worker process is up (its imports are done), then run ``worker``.

    The parent starts the job's wall-clock budget when the marker file appears, so a slow
    process start does not eat into it.
    """
    Path(marker).touch()
    result: dict[str, Any] = worker(**arguments)
    return result
