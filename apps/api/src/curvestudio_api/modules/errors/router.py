"""``POST /api/v1/errors``: what browsers report. Anonymous, by address; the rules live in
the service and no query is written here."""

from fastapi import APIRouter, Request, Response

from curvestudio_api.deps import Ctx, Db, client_ip
from curvestudio_api.modules.errors.schemas import ErrorEventBatchIn, ErrorReportIn
from curvestudio_api.modules.errors.service import (
    DESCRIPTION_KEY,
    record_error_event,
    route_template,
)
from curvestudio_api.timeutil import utcnow

router = APIRouter(prefix="/errors", tags=["errors"])


@router.post("", status_code=204, summary="Report one error event, or a batch of up to ten")
def report_errors(request: Request, ctx: Ctx, db: Db, body: ErrorReportIn) -> Response:
    """Record what a browser reported.

    An address may report twenty times an hour. A repeat of an error already stored is rolled
    up. What the user typed (``description``) is kept in the event's context. Nothing in a
    report identifies an account. The answer is 204 with no body.
    """
    ctx.error_limiter.check(client_ip(request))
    events = body.events if isinstance(body, ErrorEventBatchIn) else [body]
    now = utcnow()
    for event in events:
        context = dict(event.context)
        if event.description is not None and event.description.strip():
            context[DESCRIPTION_KEY] = event.description
        record_error_event(
            db,
            source="client",
            category=event.category,
            severity=event.severity,
            error_type=event.error_type,
            message=event.message,
            stack=event.stack,
            route=None if event.route is None else route_template(event.route),
            request_id=event.request_id,
            user_agent=event.user_agent,
            app_version=event.app_version,
            context=context or None,
            console_log=event.console_log,
            now=now,
        )
    db.commit()
    return Response(status_code=204)
