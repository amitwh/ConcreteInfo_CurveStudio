"""The statistical tools: the catalogue, running a tool, and the stored results."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from curvestudio.tools import ToolError
from curvestudio.tools.stats import FAMILIES, catalogue
from curvestudio_api import audit
from curvestudio_api.deps import Ctx, CurrentUser, Db, own_dataset
from curvestudio_api.errors import ApiError, not_found
from curvestudio_api.orm import StatResult, User
from curvestudio_api.schemas import (
    StatResultOut,
    StatResultSummaryOut,
    StatsToolsOut,
)
from curvestudio_api.stats import result_out, run_tool_for, summary_title
from curvestudio_api.workflow_errors import tool_api_error

router = APIRouter(tags=["stats"])


def _tool_error(exc: ToolError) -> ApiError:
    """An unknown tool is a 404; every other refusal is the tools' 422 (or 413 ``TOO_LARGE``)."""
    if exc.code == "UNKNOWN_TOOL":
        return not_found("tool")
    return tool_api_error(exc)


def own_result(db: Session, user: User, result_id: uuid.UUID) -> StatResult:
    """The user's stored result, or 404 (also for somebody else's)."""
    row = db.get(StatResult, result_id)
    if row is None or row.user_id != user.id:
        raise not_found("stat result")
    return row


def _out(row: StatResult) -> StatResultOut:
    result = result_out(row)
    return StatResultOut(
        id=row.id,
        tool=row.tool,
        title=result.title,
        dataset_id=row.dataset_id,
        request=row.request,
        result=result,
        created_at=row.created_at,
    )


def _summary(row: StatResult) -> StatResultSummaryOut:
    return StatResultSummaryOut(
        id=row.id,
        tool=row.tool,
        title=summary_title(row),
        dataset_id=row.dataset_id,
        created_at=row.created_at,
    )


def _run(
    db: Session,
    ctx: Ctx,
    user: User,
    dataset_id: uuid.UUID | None,
    tool: str,
    choices: dict[str, Any],
) -> StatResultOut:
    dataset = None if dataset_id is None else own_dataset(db, user, dataset_id)
    try:
        row = run_tool_for(db, ctx, user, dataset, tool, choices)
    except ToolError as exc:
        raise _tool_error(exc) from exc
    return _out(row)


@router.get("/stats/tools", response_model=StatsToolsOut)
def list_tools(user: CurrentUser) -> StatsToolsOut:
    """Every statistical tool, with the families a person meets them in."""
    return StatsToolsOut(families=list(FAMILIES), tools=list(catalogue()))


@router.post("/datasets/{dataset_id}/stats/{tool}", status_code=201, response_model=StatResultOut)
def run_on_dataset(
    dataset_id: uuid.UUID,
    tool: str,
    choices: dict[str, Any],
    user: CurrentUser,
    ctx: Ctx,
    db: Db,
) -> StatResultOut:
    """Run a tool on a dataset's columns; the choices are its request fields by name.

    A tool that needs no data is still accepted here and stored with the dataset."""
    return _run(db, ctx, user, dataset_id, tool, choices)


@router.post("/stats/{tool}", status_code=201, response_model=StatResultOut)
def run_without_dataset(
    tool: str,
    choices: dict[str, Any],
    user: CurrentUser,
    ctx: Ctx,
    db: Db,
) -> StatResultOut:
    """Run a tool that needs no data (a sample-size calculation). A tool that does need a
    dataset is a 422 ``INVALID_OPTION``."""
    return _run(db, ctx, user, None, tool, choices)


@router.get("/stats/results", response_model=list[StatResultSummaryOut])
def list_results(
    user: CurrentUser,
    db: Db,
    dataset_id: Annotated[uuid.UUID | None, Query()] = None,
) -> list[StatResultSummaryOut]:
    """The user's stored results, newest first; only one dataset's when ``dataset_id`` is given."""
    query = select(StatResult).where(StatResult.user_id == user.id)
    if dataset_id is not None:
        own_dataset(db, user, dataset_id)
        query = query.where(StatResult.dataset_id == dataset_id)
    rows = db.scalars(query.order_by(StatResult.created_at.desc(), StatResult.id))
    return [_summary(row) for row in rows]


@router.get("/stats/results/{result_id}", response_model=StatResultOut)
def get_result(result_id: uuid.UUID, user: CurrentUser, db: Db) -> StatResultOut:
    return _out(own_result(db, user, result_id))


@router.delete("/stats/results/{result_id}", status_code=204)
def delete_result(result_id: uuid.UUID, user: CurrentUser, db: Db) -> Response:
    row = own_result(db, user, result_id)
    audit.record(
        db,
        "stat.delete",
        "stat_result",
        user_id=user.id,
        entity_id=row.id,
        detail={"tool": row.tool},
    )
    db.delete(row)
    db.commit()
    return Response(status_code=204)
