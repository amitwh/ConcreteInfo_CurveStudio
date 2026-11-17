"""Auto-fit: start a run, follow it through ``/jobs``, read, re-rank, export and open its rows."""

import json
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Response
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, defer

from curvestudio.discovery import (
    AutoFitError,
    AutoFitResult,
    RankBy,
    overfitting_note,
)
from curvestudio_api import audit, quotas
from curvestudio_api.autofit_plan import check_autofit, config_from_request, fit_spec_for_candidate
from curvestudio_api.deps import Ctx, CurrentUser, Db, own_autofit, own_dataset
from curvestudio_api.errors import ApiError, not_found
from curvestudio_api.export import autofit_csv
from curvestudio_api.jobs import ACTIVE, Task
from curvestudio_api.logging_setup import request_id_var
from curvestudio_api.orm import Autofit, Fit, Job
from curvestudio_api.routers.fits import admit_job
from curvestudio_api.schemas import (
    AutofitOut,
    AutoFitRequest,
    AutofitSummaryOut,
    JobCreatedOut,
    OpenRowOut,
)
from curvestudio_api.storage import decode_payload
from curvestudio_api.timeutil import utcnow
from curvestudio_api.workflow_errors import autofit_api_error

router = APIRouter(tags=["autofit"])

_FITTED = ("converged", "exact_fit")


def _summary(autofit: Autofit) -> AutofitSummaryOut:
    return AutofitSummaryOut.model_validate(
        {
            "id": autofit.id,
            "dataset_id": autofit.dataset_id,
            "created_at": autofit.created_at,
            "best_model_id": autofit.best_model_id,
            "n_candidates": autofit.n_candidates,
            "rank_by": autofit.config["rank_by"],
        }
    )


def _reranked(result: AutoFitResult, rank_by: RankBy | None) -> AutoFitResult:
    """``result`` ordered by ``rank_by`` (its own ordering when ``None``); nothing is refitted."""
    if rank_by is None or rank_by == result.config.rank_by:
        return result
    ranked = result.ranked_by(rank_by)
    return result.model_copy(
        update={
            "ranked": ranked,
            "config": result.config.model_copy(update={"rank_by": rank_by}),
            "overfitting_note": overfitting_note(ranked, rank_by),
        }
    )


def _stored(autofit: Autofit) -> AutoFitResult:
    return AutoFitResult.model_validate(autofit.result)


@router.post("/datasets/{dataset_id}/autofit", status_code=202, response_model=JobCreatedOut)
def create_autofit(
    dataset_id: uuid.UUID, body: AutoFitRequest, user: CurrentUser, ctx: Ctx, db: Db
) -> JobCreatedOut:
    """Queue an Auto-fit run: every applicable model is fitted and ranked. Follow it with
    ``GET /jobs/{id}`` (progress and the best rows so far); the finished run is an
    ``/autofits/{id}``. A request the engine refuses (too few points, unknown family, an
    unusable custom equation) is a 422 whose ``detail.code`` names the reason."""
    dataset = own_dataset(db, user, dataset_id)
    config = config_from_request(body, ctx.settings)
    xy = decode_payload(
        dataset.payload,
        x_name=dataset.x_column,
        y_name=dataset.y_column,
        fingerprint=dataset.fingerprint,
    )
    try:
        check_autofit(xy, config)
    except AutoFitError as exc:
        raise autofit_api_error(exc) from exc
    admit_job(db, ctx, user, lambda: quotas.check_autofit_runs(db, ctx.settings, user.id))
    job = Job(
        id=uuid.uuid4(),
        user_id=user.id,
        dataset_id=dataset.id,
        kind="autofit",
        status="queued",
        spec={"config": config.model_dump(mode="json"), "budget_s": ctx.settings.autofit_budget_s},
    )
    db.add(job)
    db.commit()
    ctx.jobs.submit(
        Task(
            job_id=job.id,
            user_id=user.id,
            dataset_id=dataset.id,
            request_id=request_id_var.get(),
            kind="autofit",
        )
    )
    return JobCreatedOut(job_id=job.id)


@router.get("/datasets/{dataset_id}/autofits", response_model=list[AutofitSummaryOut])
def list_autofits(dataset_id: uuid.UUID, user: CurrentUser, db: Db) -> list[AutofitSummaryOut]:
    dataset = own_dataset(db, user, dataset_id)
    rows = db.scalars(
        select(Autofit)
        .where(Autofit.dataset_id == dataset.id, Autofit.user_id == user.id)
        .options(defer(Autofit.result))
        .order_by(Autofit.created_at.desc(), Autofit.id)
    )
    return [_summary(a) for a in rows]


@router.get("/autofits/{autofit_id}", response_model=AutofitOut)
def get_autofit(
    autofit_id: uuid.UUID,
    user: CurrentUser,
    db: Db,
    rank_by: Annotated[RankBy | None, Query()] = None,
) -> AutofitOut:
    """The stored run. ``rank_by`` re-orders the rows by another criterion without refitting."""
    autofit = own_autofit(db, user, autofit_id)
    result = _reranked(_stored(autofit), rank_by)
    return AutofitOut(
        id=autofit.id,
        dataset_id=autofit.dataset_id,
        created_at=autofit.created_at,
        best_model_id=autofit.best_model_id,
        n_candidates=autofit.n_candidates,
        rank_by=result.config.rank_by,
        partial=result.cancelled,
        result=result,
    )


@router.delete("/autofits/{autofit_id}", status_code=204)
def delete_autofit(autofit_id: uuid.UUID, user: CurrentUser, db: Db) -> Response:
    autofit = own_autofit(db, user, autofit_id)
    audit.record(db, "autofit.delete", "autofit", user_id=user.id, entity_id=autofit.id)
    db.delete(autofit)
    db.commit()
    return Response(status_code=204)


@router.get("/autofits/{autofit_id}/export")
def export_autofit(
    autofit_id: uuid.UUID,
    user: CurrentUser,
    db: Db,
    format: Annotated[Literal["json", "csv"], Query()] = "csv",
    rank_by: Annotated[RankBy | None, Query()] = None,
) -> Response:
    """Download the run as one CSV table of candidates, or as JSON (the same as ``GET``)."""
    autofit = own_autofit(db, user, autofit_id)
    result = _reranked(_stored(autofit), rank_by)
    disposition = {"Content-Disposition": f'attachment; filename="autofit-{autofit.id}.{format}"'}
    if format == "json":
        return Response(
            json.dumps(result.model_dump(mode="json"), allow_nan=False),
            media_type="application/json",
            headers=disposition,
        )
    return Response(
        autofit_csv(result, result.config.rank_by),
        media_type="text/csv; charset=utf-8",
        headers=disposition,
    )


def _job_of_opened_row(
    db: Session, autofit: Autofit, candidate_id: str, fit_id: uuid.UUID
) -> uuid.UUID:
    """The ``open_row`` job that produced ``fit_id``; a finished one is recorded if it is gone
    (purged, or the fit was made by a racing request), so the caller always has a job to read."""
    job_id = db.scalar(
        select(Job.id).where(
            Job.user_id == autofit.user_id, Job.kind == "open_row", Job.result_ref == fit_id
        )
    )
    if job_id is not None:
        return job_id
    now = utcnow()
    job = Job(
        id=uuid.uuid4(),
        user_id=autofit.user_id,
        dataset_id=autofit.dataset_id,
        kind="open_row",
        status="succeeded",
        spec={"autofit_id": str(autofit.id), "candidate_id": candidate_id},
        result_ref=fit_id,
        started_at=now,
        finished_at=now,
    )
    db.add(job)
    db.commit()
    return job.id


@router.post(
    "/autofits/{autofit_id}/rows/{candidate_id}/open", status_code=202, response_model=OpenRowOut
)
def open_row(
    autofit_id: uuid.UUID, candidate_id: str, user: CurrentUser, ctx: Ctx, db: Db
) -> OpenRowOut:
    """Refit one row with the run's seed into a stored fit (``/fits/{id}``: curve, bands,
    diagnostics, predict, export). Idempotent: a row already opened answers with its fit
    (``fit_id``); one being opened answers with the job already running."""
    autofit = own_autofit(db, user, autofit_id)
    result = _stored(autofit)
    outcome = next((o for o in result.outcomes if o.candidate_id == candidate_id), None)
    if outcome is None:
        raise not_found("row")
    if outcome.outcome not in _FITTED:
        raise ApiError(
            422,
            "VALIDATION_ERROR",
            f"{candidate_id!r} was not fitted ({outcome.outcome}): there is nothing to open",
            {"code": "ROW_NOT_FITTED"},
        )

    existing = db.scalar(
        select(Fit.id).where(Fit.autofit_id == autofit.id, Fit.candidate_id == candidate_id)
    )
    if existing is not None:
        return OpenRowOut(
            job_id=_job_of_opened_row(db, autofit, candidate_id, existing), fit_id=existing
        )
    running = db.scalars(
        select(Job).where(
            Job.user_id == user.id,
            Job.dataset_id == autofit.dataset_id,
            Job.kind == "open_row",
            Job.status.in_(ACTIVE),
        )
    )
    for job in running:
        if job.spec["autofit_id"] == str(autofit.id) and job.spec["candidate_id"] == candidate_id:
            return OpenRowOut(job_id=job.id)

    admit_job(db, ctx, user, lambda: quotas.check(db, ctx.settings, user.id, "fits"))
    try:
        fit_spec = fit_spec_for_candidate(result.config, candidate_id)
    except ValidationError as exc:  # e.g. a custom equation with more parameters than FitSpec holds
        raise ApiError(
            422,
            "VALIDATION_ERROR",
            f"{candidate_id!r} cannot be opened as a fit: {exc.errors()[0]['msg']}",
            {"code": "ROW_NOT_OPENABLE"},
        ) from exc
    job = Job(
        id=uuid.uuid4(),
        user_id=user.id,
        dataset_id=autofit.dataset_id,
        kind="open_row",
        status="queued",
        spec={
            "autofit_id": str(autofit.id),
            "candidate_id": candidate_id,
            "config": autofit.config,
            "dataset_fingerprint": result.dataset_fingerprint,
            "fit_spec": fit_spec.model_dump(mode="json"),
        },
    )
    db.add(job)
    db.commit()
    ctx.jobs.submit(
        Task(
            job_id=job.id,
            user_id=user.id,
            dataset_id=autofit.dataset_id,
            request_id=request_id_var.get(),
            kind="open_row",
        )
    )
    return OpenRowOut(job_id=job.id)
