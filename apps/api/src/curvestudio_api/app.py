"""The FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from curvestudio_api import __version__
from curvestudio_api.autofit_worker import run_autofit_job
from curvestudio_api.db import make_engine, make_session_factory
from curvestudio_api.deps import (
    AUTH_ATTEMPTS_PER_MINUTE,
    AUTH_PER_IP_PER_MINUTE,
    ERROR_REPORTS_PER_HOUR,
    INSTALL_RECORDS_PER_HOUR,
    Context,
)
from curvestudio_api.errors import install_error_handlers
from curvestudio_api.housekeeping import Housekeeper
from curvestudio_api.import_worker import ImportWorker, import_process_main
from curvestudio_api.importer import ImportRunner
from curvestudio_api.jobs import JobManager
from curvestudio_api.logging_setup import LOGGER_NAME, configure_logging, get_logger
from curvestudio_api.middleware import BodyLimitMiddleware, RequestContextMiddleware
from curvestudio_api.migrations import upgrade_to_head
from curvestudio_api.modules import register_models
from curvestudio_api.modules.errors import router as error_events
from curvestudio_api.modules.errors.capture import (
    install_error_log_handler,
    set_default_error_capture_sink,
)
from curvestudio_api.modules.installs import router as installs
from curvestudio_api.ratelimit import DailyCap, SlidingWindowLimiter
from curvestudio_api.routers import auth, autofits, datasets, fits, oauth, system
from curvestudio_api.security import JwtKeys
from curvestudio_api.settings import Settings, load_jwt_keys
from curvestudio_api.spa import mount_spa
from curvestudio_api.worker import JobWorker, run_fit_job

API_PREFIX = "/api/v1"
JSON_BODY_LIMIT = 8 * 1024 * 1024
MULTIPART_OVERHEAD = 1024 * 1024
_log = get_logger("app")


def create_app(
    settings: Settings | None = None,
    *,
    job_worker: JobWorker = run_fit_job,
    autofit_worker: JobWorker = run_autofit_job,
    import_worker: ImportWorker = import_process_main,
) -> FastAPI:
    """Build the application.

    Production settings without JWT keys, or keys that do not parse or do not match, fail
    here, so a misconfigured deployment never starts serving. ``job_worker``,
    ``autofit_worker`` and ``import_worker`` exist for tests; they must be top-level functions
    (they run in spawned processes).
    """
    register_models()
    settings = settings or Settings()
    configure_logging()
    private_pem, public_pem = load_jwt_keys(settings)
    if settings.env == "prod" and settings.require_email_verification and not settings.smtp_host:
        raise RuntimeError(
            "CURVESTUDIO_SMTP_HOST is required when email verification is enabled in prod"
        )
    engine = make_engine(settings)
    session_factory = make_session_factory(engine)
    # Errors the server captures itself go to this app's database; see modules/errors/capture.
    set_default_error_capture_sink(session_factory)
    install_error_log_handler(logging.getLogger(LOGGER_NAME))
    jobs = JobManager(
        session_factory,
        workers=settings.job_workers,
        timeout_s=settings.job_timeout_s,
        grace_s=settings.job_grace_s,
        max_result_bytes=settings.job_max_result_bytes,
        queue_max=settings.job_queue_max,
        worker=job_worker,
        autofit_worker=autofit_worker,
        autofit_timeout_s=settings.autofit_timeout_s,
    )
    housekeeper = Housekeeper(session_factory, settings.housekeeping_interval_s)
    context = Context(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        keys=JwtKeys(private_pem, public_pem),
        jobs=jobs,
        auth_limiter=SlidingWindowLimiter(AUTH_ATTEMPTS_PER_MINUTE),
        ip_limiter=SlidingWindowLimiter(AUTH_PER_IP_PER_MINUTE),
        install_limiter=SlidingWindowLimiter(INSTALL_RECORDS_PER_HOUR, 3600.0),
        error_limiter=SlidingWindowLimiter(ERROR_REPORTS_PER_HOUR, 3600.0),
        imports=ImportRunner(
            concurrency=settings.import_concurrency,
            timeout_s=settings.import_timeout_s,
            memory_mb=settings.import_memory_mb,
            max_payload_bytes=max(64 * 1024 * 1024, settings.max_rows * 100),
            worker=import_worker,
        ),
        failed_logins=DailyCap(),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        upgrade_to_head(engine)
        jobs.start()
        housekeeper.start()
        _log.info("started", extra={"fields": {"env": settings.env, "version": __version__}})
        try:
            yield
        finally:
            housekeeper.stop()
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
    for module in (system, auth, oauth, datasets, fits, autofits, installs, error_events):
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
        path_limits={f"{API_PREFIX}/installs": 2048, f"{API_PREFIX}/errors": 65_536},
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            allow_credentials=False,
        )
    app.add_middleware(RequestContextMiddleware, hsts=settings.env == "prod")
    return app


__all__ = ["create_app"]
