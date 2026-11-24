"""The ``audit`` job's code, which runs inside a job process (top-level and picklable: ``spawn``
start).

:func:`run_cross_validation_job` cross-validates one stored fit on the table it was fitted to.
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
    """Cross-validate one stored fit on its dataset's table.

    ``spec`` is the job's own (the fit's id and the fold count), ``bundle`` the fit's stored
    bundle and ``fit_spec`` the specification it ran with. The data are decoded as the fit job
    decodes them, so a compact bundle is cross-validated on the whole table it was fitted to.
    Returns ``{"ok": True, "cv": <CrossValidation JSON>}`` or, for an input the tool refuses,
    ``{"ok": False, "code": ..., "message": ...}``.
    """
    xy = decode_payload(payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint)
    try:
        cv = cross_validate(
            FitBundle.model_validate(bundle),
            FitSpec.model_validate(fit_spec),
            data=xy,
            folds=spec["folds"],
        )
    except ToolError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message}
    return {"ok": True, "cv": cv.model_dump(mode="json")}
