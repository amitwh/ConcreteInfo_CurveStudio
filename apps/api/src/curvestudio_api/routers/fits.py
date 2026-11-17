"""Fit jobs, stored fits, prediction and export."""

import json
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import defer

from curvestudio.data import XY
from curvestudio.workflow import FitBundle, FitSpec, Prediction, WorkflowError, predict
from curvestudio_api import audit
from curvestudio_api.deps import Ctx, CurrentUser, Db, own_dataset, own_fit, own_job
from curvestudio_api.errors import ApiError
from curvestudio_api.export import csv_chunks
from curvestudio_api.jobs import ACTIVE, Task
from curvestudio_api.logging_setup import request_id_var
from curvestudio_api.orm import Dataset, Fit, Job
from curvestudio_api.schemas import FitSummaryOut, JobCreatedOut, JobOut, PredictIn
from curvestudio_api.specs import check_spec
from curvestudio_api.storage import decode_payload
from curvestudio_api.workflow_errors import api_error

router = APIRouter(tags=["fits"])

MAX_ACTIVE_JOBS_PER_USER = 20


def job_out(job: Job) -> JobOut:
    return JobOut.model_validate(
        {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "dataset_id": job.dataset_id,
            "spec": job.spec,
            "result_ref": job.result_ref,
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
    format: Annotated[Literal["json", "csv"], Query()] = "json",
) -> Response:
    """Download a fit as JSON (the bundle) or CSV (parameters, then fitted values per row)."""
    fit = own_fit(db, user, fit_id)
    disposition = {"Content-Disposition": f'attachment; filename="fit-{fit.id}.{format}"'}
    if format == "json":
        return Response(
            json.dumps(fit.bundle, allow_nan=False),
            media_type="application/json",
            headers=disposition,
        )
    dataset = own_dataset(db, user, fit.dataset_id)
    bundle = FitBundle.model_validate(fit.bundle)
    chunks = csv_chunks(bundle, _stored_xy(dataset))
    return StreamingResponse(chunks, media_type="text/csv; charset=utf-8", headers=disposition)
