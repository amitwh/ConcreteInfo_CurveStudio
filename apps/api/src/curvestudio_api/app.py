"""The FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from curvestudio_api import __version__
from curvestudio_api.db import make_engine, make_session_factory
from curvestudio_api.deps import (
    AUTH_ATTEMPTS_PER_MINUTE,
    AUTH_PER_IP_PER_MINUTE,
    Context,
)
from curvestudio_api.errors import install_error_handlers
from curvestudio_api.jobs import JobManager
from curvestudio_api.logging_setup import configure_logging, get_logger
from curvestudio_api.middleware import BodyLimitMiddleware, RequestContextMiddleware
from curvestudio_api.migrations import upgrade_to_head
from curvestudio_api.ratelimit import SlidingWindowLimiter
from curvestudio_api.routers import auth, datasets, fits, system
from curvestudio_api.security import JwtKeys
from curvestudio_api.settings import Settings, load_jwt_keys
from curvestudio_api.spa import mount_spa
from curvestudio_api.worker import JobWorker, run_fit_job

API_PREFIX = "/api/v1"
JSON_BODY_LIMIT = 8 * 1024 * 1024
MULTIPART_OVERHEAD = 1024 * 1024
_log = get_logger("app")


def create_app(settings: Settings | None = None, *, job_worker: JobWorker = run_fit_job) -> FastAPI:
    """Build the application.

    Production settings without JWT keys, or keys that do not parse or do not match, fail
    here, so a misconfigured deployment never starts serving. ``job_worker`` exists for
    tests; it must be a top-level function (it runs in a spawned process).
    """
    settings = settings or Settings()
    configure_logging()
    private_pem, public_pem = load_jwt_keys(settings)
    engine = make_engine(settings)
    session_factory = make_session_factory(engine)
    jobs = JobManager(
        session_factory,
        workers=settings.job_workers,
        timeout_s=settings.job_timeout_s,
        grace_s=settings.job_grace_s,
        worker=job_worker,
    )
    context = Context(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        keys=JwtKeys(private_pem, public_pem),
        jobs=jobs,
        auth_limiter=SlidingWindowLimiter(AUTH_ATTEMPTS_PER_MINUTE),
        ip_limiter=SlidingWindowLimiter(AUTH_PER_IP_PER_MINUTE),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        upgrade_to_head(engine)
        jobs.start()
        _log.info("started", extra={"fields": {"env": settings.env, "version": __version__}})
        try:
            yield
        finally:
            jobs.stop()
            engine.dispose()

    app = FastAPI(
        title="CurveStudio API",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.state.ctx = context
    install_error_handlers(app)

    api = APIRouter(prefix=API_PREFIX)
    for module in (system, auth, datasets, fits):
        api.include_router(module.router)
    app.include_router(api)
    if settings.web_dist is not None:
        mount_spa(app, settings.web_dist)

    # Innermost first: the last one added is the outermost layer.
    app.add_middleware(
        BodyLimitMiddleware,
        default_limit=JSON_BODY_LIMIT,
        upload_limit=settings.max_upload_bytes + MULTIPART_OVERHEAD,
        upload_path=f"{API_PREFIX}/datasets",
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            allow_credentials=False,
        )
    app.add_middleware(RequestContextMiddleware, hsts=settings.env == "prod")
    return app


__all__ = ["create_app"]
