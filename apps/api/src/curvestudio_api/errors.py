"""The error envelope ``{"error": CODE, "message": str, "detail": {}}`` and its handlers."""

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

STATUS_CODES: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    503: "SERVICE_UNAVAILABLE",
}


class ErrorBody(BaseModel):
    """The JSON body of every error response."""

    error: str
    message: str
    detail: dict[str, Any]


class ApiError(HTTPException):
    """An error the API raises on purpose. ``code`` is the envelope's ``error`` field."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=dict(headers or {}))
        self.code = code
        self.message = message
        self.extra = detail or {}


def not_found(what: str) -> ApiError:
    """The 404 used for a missing resource *and* for another user's resource."""
    return ApiError(404, "NOT_FOUND", f"{what} not found")


def envelope(code: str, message: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": code, "message": message, "detail": detail or {}}


def internal_error_response() -> JSONResponse:
    return JSONResponse(envelope("INTERNAL_ERROR", "an internal error occurred"), status_code=500)


async def _http_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    if isinstance(exc, ApiError):
        code, message, detail = exc.code, exc.message, exc.extra
    else:
        code = STATUS_CODES.get(exc.status_code, "INTERNAL_ERROR")
        message = str(exc.detail)
        detail = {}
    return JSONResponse(
        envelope(code, message, detail), status_code=exc.status_code, headers=exc.headers
    )


async def _validation_error(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Only where/what/kind: pydantic's "input" would echo the offending value (a password).
    problems = [
        {"loc": [str(part) for part in err["loc"]], "msg": err["msg"], "type": err["type"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        envelope("VALIDATION_ERROR", "the request is not valid", {"errors": problems}),
        status_code=422,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Route every error response through the envelope."""
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
