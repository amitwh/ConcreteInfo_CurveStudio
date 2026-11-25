"""ASGI middleware: request id + access log + security headers + request statistics, and
request body limits."""

import logging
import re
import time
import uuid
from collections.abc import Mapping
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from curvestudio_api import API_PREFIX
from curvestudio_api.errors import ApiError, envelope
from curvestudio_api.logging_setup import get_logger, log_fields, request_id_var
from curvestudio_api.modules.errors.capture import CAPTURED_FIELD, capture_request_error
from curvestudio_api.modules.errors.service import route_template

# The client-side PDF renderer needs three narrow allowances:
#   * `'wasm-unsafe-eval'` instantiates its WebAssembly text-layout engine (compilation
#     only — not JavaScript `eval`), so the no-unsafe-eval posture is preserved;
#   * `worker-src blob:` lets it start its font-subsetting worker from a blob URL;
#   * `connect-src data:` lets it read the engine's inline data-URL payload.
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; font-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; script-src 'self' 'wasm-unsafe-eval'; "
    "worker-src 'self' blob:; connect-src 'self' data: blob:"
)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": CSP,
}
HSTS = "max-age=31536000; includeSubDomains"
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
# The liveness and readiness probes. The image's HEALTHCHECK calls /api/v1/health every 30 s
# (~2,880 probes a day; a failing probe still surfaces through the health route's own error
# log), so their request lines go to DEBUG to keep the log clear of probe noise.
HEALTH_PROBE_PATHS = frozenset((f"{API_PREFIX}/health", f"{API_PREFIX}/ready"))
_log = get_logger("http")


class RequestContextMiddleware:
    """Outermost layer: assigns ``request_id``, adds headers, logs one line per request and
    turns an unhandled exception into the ``INTERNAL_ERROR`` envelope."""

    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = Headers(scope=scope).get("x-request-id")
        request_id = supplied if supplied and _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        request_id_var.set(request_id)
        started = time.perf_counter()
        state: dict[str, Any] = {"status": 500, "started": False}

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["started"] = True
                state["status"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in SECURITY_HEADERS.items():
                    headers.setdefault(name, value)
                if self.hsts:
                    headers.setdefault("Strict-Transport-Security", HSTS)
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception as exc:
            # Recorded first; the record then says whether that worked, so the log handler
            # writes it only when this did not. Neither step may change the response.
            captured = _capture(scope, exc)
            _log.exception("unhandled error", extra=log_fields(**{CAPTURED_FIELD: captured}))
            if state["started"]:
                raise
            response = JSONResponse(
                envelope("INTERNAL_ERROR", "an internal error occurred"), status_code=500
            )
            await response(scope, receive, send_with_headers)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            _record(scope, state["status"], duration_ms)
            _log.log(
                logging.DEBUG if scope["path"] in HEALTH_PROBE_PATHS else logging.INFO,
                "request",
                extra=log_fields(
                    method=scope["method"],
                    path=scope["path"],
                    status=state["status"],
                    duration_ms=duration_ms,
                ),
            )


def _capture(scope: Scope, exc: Exception) -> bool:
    """Record ``exc`` as a fatal server event: ``True`` when a row was written.

    Never raises: a scope without the application's ``Context`` (a bare ASGI app in a test)
    records nothing, and the hook itself swallows its own failures. The route is stored as its
    template; a request that matched no route has none.
    """
    context = getattr(getattr(scope.get("app"), "state", None), "ctx", None)
    factory = getattr(context, "session_factory", None)
    if factory is None:
        return False
    route = matched_route(scope)
    return capture_request_error(
        factory,
        exc=exc,
        method=scope["method"],
        route=route_template(route) if route is not None else None,
        request_id=request_id_var.get(),
        user_agent=Headers(scope=scope).get("user-agent") or "",
    )


def is_api_request(scope: Scope) -> bool:
    """Whether the request addresses the API, as opposed to the web app's own pages and
    assets, which one catch-all route serves."""
    path = scope.get("path") or ""
    return path == API_PREFIX or path.startswith(f"{API_PREFIX}/")


def matched_route(scope: Scope) -> str | None:
    """The route a request matched, as its template: ``/api/v1/datasets/{id}``.

    ``None`` when it matched none. This FastAPI keeps an included router's routes nested, so a
    matched route's own path has no ``API_PREFIX`` in it; for a request under the prefix it is
    put back. A route mounted with its full path (a test's scratch route) already has it.
    """
    template = getattr(scope.get("route"), "path_format", None)
    if template is None:
        return None
    if is_api_request(scope) and not template.startswith(f"{API_PREFIX}/"):
        template = f"{API_PREFIX}{template}"
    return template


def _record(scope: Scope, status: int, duration_ms: float) -> None:
    """Count the finished API request in the per-route statistics.

    The request is already answered, so nothing here may change the response: never raises.
    A scope without the application's ``Context`` (a bare ASGI app in a test) counts nothing,
    and neither does a request that matched no route or is not under the API: the web app's
    pages and assets are served by one catch-all route, and mixed with API calls their timings
    would say nothing about the API. The route is stored as its template.
    """
    try:
        context = getattr(getattr(scope.get("app"), "state", None), "ctx", None)
        stats = getattr(context, "stats", None)
        route = matched_route(scope) if is_api_request(scope) else None
        if stats is not None and route is not None:
            stats.record(route, duration_ms, error=status >= 500)
    except Exception:
        _log.warning("request statistics were not recorded", exc_info=True)


class BodyLimitMiddleware:
    """Rejects oversized request bodies while they stream in, before anything is parsed.

    A declared ``Content-Length`` over the limit is refused without reading a byte; a body
    that exceeds it anyway (chunked, or a lying header) stops at the limit with a 413.

    A path in ``path_limits`` gets that limit, whatever the method; ``upload_limit`` applies to
    POSTs on ``upload_path``; every other request gets ``default_limit``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        default_limit: int,
        upload_limit: int,
        upload_path: str,
        path_limits: Mapping[str, int] | None = None,
    ) -> None:
        self.app = app
        self.default_limit = default_limit
        self.upload_limit = upload_limit
        self.upload_path = upload_path
        self.path_limits: Mapping[str, int] = path_limits or {}

    def _limit(self, scope: Scope) -> int:
        path = scope["path"]
        if path in self.path_limits:
            return self.path_limits[path]
        if scope["method"] == "POST" and path == self.upload_path:
            return self.upload_limit
        return self.default_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._limit(scope)
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            response = JSONResponse(
                envelope("TOO_LARGE", _too_large_message(limit), {"code": "TOO_LARGE"}),
                status_code=413,
            )
            await response(scope, receive, send)
            return
        seen = 0

        async def counting_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise ApiError(
                        413, "TOO_LARGE", _too_large_message(limit), {"code": "TOO_LARGE"}
                    )
            return message

        await self.app(scope, counting_receive, send)


def _too_large_message(limit: int) -> str:
    return f"the request body is larger than the limit of {limit:,} bytes"
