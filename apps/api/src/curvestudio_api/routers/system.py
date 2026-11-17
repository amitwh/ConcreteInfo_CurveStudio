"""Liveness and readiness probes, the current user and the public model catalogue."""

from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from curvestudio.workflow import ModelInfo, WorkflowError, get_model, list_models
from curvestudio_api import __version__
from curvestudio_api.deps import Ctx, CurrentUser
from curvestudio_api.errors import ApiError
from curvestudio_api.logging_setup import get_logger
from curvestudio_api.migrations import current_revision_is_head
from curvestudio_api.schemas import HealthOut, ReadyOut, UserOut

router = APIRouter()
_log = get_logger("health")


def _database_up(ctx: Ctx) -> bool:
    try:
        with ctx.session_factory() as db:
            db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        _log.exception("database check failed")
        return False
    return True


@router.get("/health", response_model=HealthOut, tags=["system"])
def health(ctx: Ctx) -> HealthOut:
    """Liveness plus a real ``SELECT 1``; 503 when the database cannot be reached."""
    if not _database_up(ctx):
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the database is not reachable")
    return HealthOut(status="ok", database="ok", version=__version__)


@router.get("/ready", response_model=ReadyOut, tags=["system"])
def ready(ctx: Ctx) -> ReadyOut:
    """Ready to serve: database reachable, schema at the newest migration, jobs running."""
    if not _database_up(ctx):
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the database is not reachable")
    if not current_revision_is_head(ctx.engine):
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the database schema is not up to date")
    if not ctx.jobs.is_running():
        raise ApiError(503, "SERVICE_UNAVAILABLE", "the job scheduler is not running")
    return ReadyOut(status="ready")


@router.get("/me", response_model=UserOut, tags=["auth"])
def me(user: CurrentUser) -> UserOut:
    return UserOut(id=user.id, email=user.email, name=user.name, created_at=user.created_at)


@router.get("/models", response_model=list[ModelInfo], tags=["models"])
def models(
    category: Annotated[str | None, Query(max_length=100)] = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
) -> list[ModelInfo]:
    """The model catalogue, optionally filtered by category and a search text."""
    found = list_models()
    if category:
        wanted = category.strip().lower()
        found = [m for m in found if m.category.lower() == wanted]
    if q and q.strip():
        needle = q.strip().lower()
        found = [
            m for m in found if needle in " ".join((m.id, m.name, m.category, m.equation)).lower()
        ]
    return found


@router.get("/models/{model_id}", response_model=ModelInfo, tags=["models"])
def model(model_id: str) -> ModelInfo:
    try:
        return get_model(model_id)
    except WorkflowError as exc:
        raise ApiError(404, "NOT_FOUND", exc.message, {"code": exc.code}) from exc
