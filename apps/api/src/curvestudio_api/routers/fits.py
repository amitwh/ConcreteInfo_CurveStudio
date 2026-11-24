"""Fit jobs, stored fits, prediction, export, and the audit of a stored fit."""

import json
import uuid
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, defer

from curvestudio.data import XY
from curvestudio.discovery import AutoFitResult
from curvestudio.tools import ToolError, all_methods_texts, audit_fit
from curvestudio.tools.cross_validation import CrossValidation
from curvestudio.workflow import FitBundle, FitSpec, Prediction, WorkflowError, predict
from curvestudio_api import audit, quotas
from curvestudio_api.deps import Context, Ctx, CurrentUser, Db, own_dataset, own_fit, own_job
from curvestudio_api.errors import ApiError
from curvestudio_api.export import XLSX_MEDIA_TYPE, csv_chunks, xlsx_bytes
from curvestudio_api.jobs import ACTIVE, Task
from curvestudio_api.logging_setup import request_id_var
from curvestudio_api.orm import Autofit, Dataset, Fit, Job, User
from curvestudio_api.schemas import (
    CrossValidationIn,
    FitAuditOut,
    FitMethodsOut,
    FitSummaryOut,
    JobCreatedOut,
    JobOut,
    PredictIn,
)
from curvestudio_api.specs import check_spec
from curvestudio_api.storage import decode_payload
from curvestudio_api.workflow_errors import api_error, tool_api_error

router = APIRouter(tags=["fits"])

MAX_ACTIVE_JOBS_PER_USER = 20
MIN_FOLDS = 2
"""The fewest folds a cross-validation has (the core's own rule). Fewer is refused before a job
is queued, with the code the tools refuse it by."""


def admit_job(db: Session, ctx: Context, user: User, check_quota: Callable[[], None]) -> None:
    """Refuse a new job when the user already has too many in flight, when ``check_quota``
    (the job kind's own quota) says no, or when the server's queue is full."""
    active = db.scalar(
        select(func.count()).select_from(Job).where(Job.user_id == user.id, Job.status.in_(ACTIVE))
    )
    if (active or 0) >= MAX_ACTIVE_JOBS_PER_USER:
        raise ApiError(
            429,
            "RATE_LIMITED",
            f"{MAX_ACTIVE_JOBS_PER_USER} jobs are already waiting or running; wait for one",
            {"code": "TOO_MANY_JOBS"},
        )
    check_quota()
    if not ctx.jobs.queue_has_room():
        raise ApiError(
            503,
            "QUEUE_FULL",
            "the server is busy: too many fits are waiting; try again shortly",
            {"code": "QUEUE_FULL"},
            {"Retry-After": "30"},
        )


def _result_id(job: Job) -> uuid.UUID | None:
    """What the job produced, once it has succeeded: the row ``result_ref`` points at. A
    cross-validation is the exception: its result is stored on the fit, which ``result_ref``
    does not point at."""
    if job.status != "succeeded":
        return None
    if job.kind == "audit":
        return uuid.UUID(job.spec["fit_id"])
    return job.result_ref


def job_out(job: Job) -> JobOut:
    return JobOut.model_validate(
        {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "dataset_id": job.dataset_id,
            "spec": job.spec,
            "result_ref": job.result_ref,
            "result_id": _result_id(job),
            "progress": job.progress,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
    )


def _summary(fit: Fit) -> FitSummaryOut:
    return FitSummaryOut.model_validate(
        {
            "id": fit.id,
            "dataset_id": fit.dataset_id,
            "status": fit.status,
            "spec": fit.spec,
            "created_at": fit.created_at,
        }
    )


def _stored_xy(dataset: Dataset) -> XY:
    return decode_payload(
        dataset.payload,
        x_name=dataset.x_column,
        y_name=dataset.y_column,
        fingerprint=dataset.fingerprint,
    )


# -- jobs ------------------------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/fits", status_code=202, response_model=JobCreatedOut)
def create_fit(
    dataset_id: uuid.UUID, spec: FitSpec, user: CurrentUser, ctx: Ctx, db: Db
) -> JobCreatedOut:
    """Queue a fit. A specification the engine rejects (bad equation, unknown model) is a 422
    here; a fit that runs but does not converge is a finished job whose fit says so."""
    dataset = own_dataset(db, user, dataset_id)
    try:
        check_spec(spec, weighted=dataset.weight_column is not None)
    except WorkflowError as exc:
        raise api_error(exc) from exc
    admit_job(db, ctx, user, lambda: quotas.check(db, ctx.settings, user.id, "fits"))
    budget = ctx.settings.job_timeout_s
    effective = spec.model_copy(update={"time_limit": min(spec.time_limit or budget, budget)})
    job = Job(
        id=uuid.uuid4(),
        user_id=user.id,
        dataset_id=dataset.id,
        kind="fit",
        status="queued",
        spec=effective.model_dump(mode="json"),
    )
    db.add(job)
    db.commit()
    ctx.jobs.submit(
        Task(
            job_id=job.id,
            user_id=user.id,
            dataset_id=dataset.id,
            request_id=request_id_var.get(),
        )
    )
    return JobCreatedOut(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
def get_job(job_id: uuid.UUID, user: CurrentUser, db: Db) -> JobOut:
    return job_out(own_job(db, user, job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobOut, tags=["jobs"])
def cancel_job(job_id: uuid.UUID, user: CurrentUser, ctx: Ctx, db: Db) -> JobOut:
    """Cancel a queued or running job. A job that already finished is a 409."""
    job = own_job(db, user, job_id)
    status = ctx.jobs.cancel(job.id)
    db.refresh(job)
    if status not in ("cancelled",):
        raise ApiError(
            409, "CONFLICT", f"the job has already finished ({status})", {"code": "JOB_FINISHED"}
        )
    return job_out(job)


# -- fits ------------------------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/fits", response_model=list[FitSummaryOut])
def list_fits(dataset_id: uuid.UUID, user: CurrentUser, db: Db) -> list[FitSummaryOut]:
    dataset = own_dataset(db, user, dataset_id)
    rows = db.scalars(
        select(Fit)
        .where(Fit.dataset_id == dataset.id, Fit.user_id == user.id)
        .options(defer(Fit.bundle))
        .order_by(Fit.created_at.desc(), Fit.id)
    )
    return [_summary(f) for f in rows]


@router.get(
    "/fits/{fit_id}",
    response_model=FitBundle,
    responses={200: {"description": "The stored fit bundle, exactly as the fit produced it."}},
)
def get_fit(fit_id: uuid.UUID, user: CurrentUser, db: Db) -> Response:
    # The stored JSON is returned as is: re-validating it would cost O(n) for no gain.
    return JSONResponse(own_fit(db, user, fit_id).bundle)


@router.delete("/fits/{fit_id}", status_code=204)
def delete_fit(fit_id: uuid.UUID, user: CurrentUser, db: Db) -> Response:
    fit = own_fit(db, user, fit_id)
    audit.record(db, "fit.delete", "fit", user_id=user.id, entity_id=fit.id)
    db.delete(fit)
    db.commit()
    return Response(status_code=204)


@router.post("/fits/{fit_id}/predict", response_model=Prediction)
def predict_fit(fit_id: uuid.UUID, body: PredictIn, user: CurrentUser, db: Db) -> Prediction:
    """ŷ at the given x (up to 100 000) from a converged fit, with an optional 95 % band."""
    fit = own_fit(db, user, fit_id)
    dataset = own_dataset(db, user, fit.dataset_id)
    try:
        return predict(
            FitBundle.model_validate(fit.bundle),
            FitSpec.model_validate(fit.spec),
            body.x,
            body.band,
            data=_stored_xy(dataset),
        )
    except WorkflowError as exc:
        raise api_error(exc) from exc


@router.get("/fits/{fit_id}/export")
def export_fit(
    fit_id: uuid.UUID,
    user: CurrentUser,
    db: Db,
    format: Annotated[Literal["json", "csv", "xlsx"], Query()] = "json",
) -> Response:
    """Download a fit as JSON (the bundle), CSV (parameters, then fitted values per row)
    or XLSX (the Excel workbook, with its plot sheet)."""
    fit = own_fit(db, user, fit_id)
    disposition = {"Content-Disposition": f'attachment; filename="fit-{fit.id}.{format}"'}
    if format == "json":
        return Response(
            json.dumps(fit.bundle, allow_nan=False),
            media_type="application/json",
            headers=disposition,
        )
    bundle = FitBundle.model_validate(fit.bundle)
    if format == "xlsx":
        return StreamingResponse(
            iter([xlsx_bytes(bundle)]),
            media_type=XLSX_MEDIA_TYPE,
            headers=disposition,
        )
    dataset = own_dataset(db, user, fit.dataset_id)
    chunks = csv_chunks(bundle, _stored_xy(dataset))
    return StreamingResponse(chunks, media_type="text/csv; charset=utf-8", headers=disposition)


# -- audit -----------------------------------------------------------------------------------------


def _stored_fit(fit: Fit) -> tuple[FitBundle, FitSpec]:
    """The fit's stored bundle and the specification it ran with, as the core's types."""
    return FitBundle.model_validate(fit.bundle), FitSpec.model_validate(fit.spec)


def _ranking_of(db: Session, fit: Fit) -> AutoFitResult | None:
    """The Auto-fit result a fit opened from a row was ranked in: ``None`` for an ordinary fit,
    and once the fit's run has been deleted."""
    if fit.autofit_id is None:
        return None
    stored = db.scalar(select(Autofit.result).where(Autofit.id == fit.autofit_id))
    return None if stored is None else AutoFitResult.model_validate(stored)


def _stored_cv(fit: Fit) -> CrossValidation | None:
    """The cross-validation stored on the fit; ``None`` until one has been run.

    ``CrossValidation`` is the statistics layer's, which the API may not import directly: it
    is taken from the tools module that produces it.
    """
    return None if fit.cv is None else CrossValidation.model_validate(fit.cv)


@router.get("/fits/{fit_id}/audit", response_model=FitAuditOut)
def get_fit_audit(fit_id: uuid.UUID, user: CurrentUser, db: Db) -> FitAuditOut:
    """What a stored fit's evidence shows, check by check: convergence, the residual tests,
    the parameters, the held-out error once the fit has been cross-validated, and a simpler
    model once the fit was ranked by an Auto-fit run. Read from the stored fit; nothing is
    refitted. A bundle with no observations is a 422 whose ``detail.code`` is ``NO_DATA``."""
    fit = own_fit(db, user, fit_id)
    bundle, spec = _stored_fit(fit)
    try:
        found = audit_fit(bundle, spec=spec, ranking=_ranking_of(db, fit), cv=_stored_cv(fit))
    except ToolError as exc:
        raise tool_api_error(exc) from exc
    return FitAuditOut.model_validate(found.model_dump(mode="json"))


@router.get("/fits/{fit_id}/methods", response_model=FitMethodsOut)
def get_fit_methods(fit_id: uuid.UUID, user: CurrentUser, db: Db) -> FitMethodsOut:
    """The methods text of a converged fit in three styles: the paragraph, the equation and
    the table, the citations, what the author must complete and what is missing. A fit that
    did not converge is a 422 whose ``detail.code`` is ``FIT_NOT_CONVERGED_INPUT``."""
    fit = own_fit(db, user, fit_id)
    bundle, spec = _stored_fit(fit)
    try:
        texts = all_methods_texts(bundle, spec=spec)
    except ToolError as exc:
        raise tool_api_error(exc) from exc
    return FitMethodsOut.model_validate(
        {style: text.model_dump(mode="json") for style, text in texts.items()}
    )


def _active_cross_validation(db: Session, user: User, fit: Fit) -> Job | None:
    """The user's queued or running cross-validation of ``fit``, when there is one."""
    for job in db.scalars(
        select(Job).where(Job.user_id == user.id, Job.kind == "audit", Job.status.in_(ACTIVE))
    ):
        if job.spec["fit_id"] == str(fit.id):
            return job
    return None


@router.post("/fits/{fit_id}/audit/cross-validation", status_code=202, response_model=JobCreatedOut)
def cross_validate_fit(
    fit_id: uuid.UUID, body: CrossValidationIn, user: CurrentUser, ctx: Ctx, db: Db
) -> JobCreatedOut:
    """Queue the cross-validation of a stored fit on the data it was computed from. The result
    is stored on the fit, where ``GET /fits/{id}/audit`` reads it: follow the job
    (``GET /jobs/{id}``), then read the audit again. The job uses no quota. One
    cross-validation of a fit runs at a time: while one is queued or running, to ask for
    another is a 409 whose ``detail.job_id`` names the job to follow. A fit that did not
    converge, or more folds than there are points, fails the job with the code the tools
    refuse it by."""
    fit = own_fit(db, user, fit_id)
    if body.folds != "loo" and body.folds < MIN_FOLDS:
        # Refused here, with the tool's own error, so that nothing is queued which cannot run.
        raise tool_api_error(
            ToolError("INVALID_FOLDS", f'folds must be at least {MIN_FOLDS}, or "loo"')
        )
    active = _active_cross_validation(db, user, fit)
    if active is not None:
        raise ApiError(
            409,
            "CONFLICT",
            "a cross-validation of this fit is already queued or running",
            {"code": "CROSS_VALIDATION_RUNNING", "job_id": str(active.id)},
        )
    admit_job(db, ctx, user, check_quota=lambda: None)  # the job uses no quota
    job = Job(
        id=uuid.uuid4(),
        user_id=user.id,
        dataset_id=fit.dataset_id,
        kind="audit",
        status="queued",
        spec={"fit_id": str(fit.id), "folds": body.folds},
    )
    db.add(job)
    db.commit()
    ctx.jobs.submit(
        Task(
            job_id=job.id,
            user_id=user.id,
            dataset_id=fit.dataset_id,
            request_id=request_id_var.get(),
            kind="audit",
        )
    )
    return JobCreatedOut(job_id=job.id)
