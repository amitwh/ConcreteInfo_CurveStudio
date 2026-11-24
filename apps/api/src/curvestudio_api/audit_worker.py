"""The ``audit`` job's code, which runs inside a job process (top-level and picklable: ``spawn``
start).

:func:`run_cross_validation_job` cross-validates one stored fit on the data it was computed from.
"""

from typing import Any

from curvestudio.tools import ToolError, cross_validate
from curvestudio.workflow import FitBundle, FitSpec
from curvestudio_api.storage import decode_payload


def run_cross_validation_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
    bundle: dict[str, Any],
    fit_spec: dict[str, Any],
) -> dict[str, Any]:
    """Cross-validate one stored fit on the data it was computed from.

    ``spec`` is the job's own (the fit's id and the fold count), ``bundle`` the fit's stored
    bundle and ``fit_spec`` the specification it ran with. Returns ``{"ok": True, "cv":
    <CrossValidation JSON>}`` or, for an input the tool refuses, ``{"ok": False, "code": ...,
    "message": ...}``.
    """
    stored = FitBundle.model_validate(bundle)
    table = None
    if stored.data.subsampled:
        # A compact bundle holds only a subsample of the data it was fitted to: the fit is
        # cross-validated on the dataset's whole table, which is that data. Any other bundle
        # holds all of its data, and that is the fit's own even when it is itself a subsample of
        # the dataset (a row of an Auto-fit run on a very large dataset, opened): the tools
        # refuse any other data, so the table is not handed to them for it.
        table = decode_payload(payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint)
    try:
        cv = cross_validate(
            stored, FitSpec.model_validate(fit_spec), data=table, folds=spec["folds"]
        )
    except ToolError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    return {"ok": True, "cv": cv.model_dump(mode="json")}
