"""The fit-centric tools' code, which runs inside a job process (top-level and picklable:
``spawn`` start).

Each function starts from one stored fit, rebuilds its data and specification exactly as
:mod:`curvestudio_api.audit_worker` does, calls the core tool and sends the one
:class:`~curvestudio.tools.stats.ToolResult` back as JSON:

* :func:`run_discernibility_job` reads the fit's Auto-fit ranking and reports whether the
  leading models can be told apart;
* :func:`run_simulation_job` simulates the requested design under the fit's specification;
* :func:`run_advice_job` recommends where to measure next for the fit.

The tools read a fit, so a job whose fit was deleted never reaches them: the manager fails
that job ``FIT_DELETED`` before it starts.
"""

from typing import Any

import numpy as np

from curvestudio.data import XY
from curvestudio.discovery import AutoFitResult
from curvestudio.tools._errors import ToolError
from curvestudio.tools.advisor import AdvisorRequest, advise
from curvestudio.tools.discernibility import DiscernibilityRequest, discernibility
from curvestudio.tools.feasibility import SimulationRequest, simulate
from curvestudio.workflow import FitBundle, FitSpec, WorkflowError
from curvestudio_api.storage import decode_payload

Failure = dict[str, Any]
"""A failed job: ``{"ok": False, "code": ..., "message": ...}``."""


def _options(spec: dict[str, Any]) -> dict[str, Any]:
    """The job's own tool options: its specification without the fit it runs on."""
    return {key: value for key, value in spec.items() if key != "fit_id"}


def _fit_data(
    bundle: FitBundle, *, payload: bytes, x_name: str, y_name: str, fingerprint: str
) -> XY:
    """The data the fit was computed from.

    A compact bundle keeps only an evenly strided sample of its rows: the fit used the whole
    dataset, so that is the table read and given to the tool. Any other bundle carries every
    row the fit used — even when it is itself a subsample of the dataset (a row of an Auto-fit
    run on a very large dataset, opened) — and its data are rebuilt from the bundle with the
    fingerprint the fit recorded, which is the fingerprint the statistics layer checks.
    """
    if bundle.data.subsampled:
        return decode_payload(payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint)
    snapshot = bundle.data
    return XY(
        x=np.asarray(snapshot.x, dtype=np.float64),
        y=np.asarray(snapshot.y, dtype=np.float64),
        weights=(
            None if snapshot.weights is None else np.asarray(snapshot.weights, dtype=np.float64)
        ),
        x_name=snapshot.x_name,
        y_name=snapshot.y_name,
        fingerprint=bundle.result.provenance.dataset_fingerprint,
    )


def _failure(error: ToolError | ValueError | WorkflowError) -> Failure:
    """The job's failure for one of the core's user-facing errors.

    A :class:`~curvestudio.tools.ToolError` and a ``StatisticsError`` (a ``ValueError`` with
    the same shape) carry a ``code``; a ``WorkflowError`` carries a ``code`` and a
    ``message``. Anything else a tool raises is an unexpected failure and the job process
    reports it as ``INTERNAL_ERROR``.
    """
    code = getattr(error, "code", "INVALID_OPTION")
    message = getattr(error, "message", None)
    return {
        "ok": False,
        "code": str(code),
        "message": message if isinstance(message, str) else str(error),
    }


def run_discernibility_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
    bundle: dict[str, Any],
    fit_spec: dict[str, Any],
    ranking: dict[str, Any],
) -> dict[str, Any]:
    """Report whether the data can tell the fit's leading models apart.

    ``spec`` is the job's own (the fit's id and the report's options), ``bundle`` the fit's
    stored bundle, ``fit_spec`` the specification it ran with and ``ranking`` the Auto-fit
    result the fit was opened from. Returns ``{"ok": True, "result": <ToolResult JSON>}`` or,
    for an input the tool refuses, ``{"ok": False, "code": ..., "message": ...}``.
    """
    try:
        data = _fit_data(
            FitBundle.model_validate(bundle),
            payload=payload,
            x_name=x_name,
            y_name=y_name,
            fingerprint=fingerprint,
        )
        result = discernibility(
            data,
            AutoFitResult.model_validate(ranking),
            request=DiscernibilityRequest.model_validate(_options(spec)),
        )
    except (ToolError, ValueError, WorkflowError) as exc:
        return _failure(exc)
    return {"ok": True, "result": result.model_dump(mode="json")}


def run_simulation_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
    bundle: dict[str, Any],
    fit_spec: dict[str, Any],
) -> dict[str, Any]:
    """Simulate the job's design under the fit's specification.

    The fit supplies the model, its bounds and its starts; no stored data take part (the
    simulated datasets are generated from the stated truth and noise). ``payload``, the column
    names and the fingerprint are the job's common arguments and are not read. Returns
    ``{"ok": True, "result": <ToolResult JSON>}`` or a failure with the tool's code.
    """
    del payload, x_name, y_name, fingerprint, bundle  # the simulation invents its own data
    try:
        result = simulate(
            FitSpec.model_validate(fit_spec),
            SimulationRequest.model_validate(_options(spec)),
        )
    except (ToolError, ValueError, WorkflowError) as exc:
        return _failure(exc)
    return {"ok": True, "result": result.model_dump(mode="json")}


def run_advice_job(
    *,
    payload: bytes,
    x_name: str,
    y_name: str,
    fingerprint: str,
    spec: dict[str, Any],
    bundle: dict[str, Any],
    fit_spec: dict[str, Any],
) -> dict[str, Any]:
    """Recommend the next measurement locations for the fit.

    ``bundle`` is the stored fit, on whose data, Jacobian and covariance the advice is
    computed. Returns ``{"ok": True, "result": <ToolResult JSON>}`` or a failure with the
    tool's code.
    """
    stored = FitBundle.model_validate(bundle)
    try:
        data = _fit_data(
            stored, payload=payload, x_name=x_name, y_name=y_name, fingerprint=fingerprint
        )
        result = advise(
            data,
            FitSpec.model_validate(fit_spec),
            stored.result,
            request=AdvisorRequest.model_validate(_options(spec)),
        )
    except (ToolError, ValueError, WorkflowError) as exc:
        return _failure(exc)
    return {"ok": True, "result": result.model_dump(mode="json")}


__all__ = ["run_advice_job", "run_discernibility_job", "run_simulation_job"]
