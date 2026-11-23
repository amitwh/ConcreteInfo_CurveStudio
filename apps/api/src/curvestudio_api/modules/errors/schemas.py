"""What the network may send about an error: one event, or a batch of one to ten."""

from typing import Any, Literal

from pydantic import Field, field_validator

from curvestudio_api.modules.errors.service import (
    MAX_APP_VERSION,
    MAX_CONSOLE_LOG,
    MAX_CONTEXT_BYTES,
    MAX_ERROR_TYPE,
    MAX_MESSAGE,
    MAX_REQUEST_ID,
    MAX_ROUTE,
    MAX_STACK,
    MAX_USER_AGENT,
    context_bytes,
)
from curvestudio_api.schemas import _Body

MAX_DESCRIPTION = 1_000
"""The most of what the user typed that one report carries."""


class ErrorEventIn(_Body):
    """One error as a browser reports it.

    There is no ``source`` field and an unknown field is refused, so a report can never pass
    itself off as the server's. Nothing here identifies an account.
    """

    category: Literal["crash", "ui", "network"]
    severity: Literal["error", "fatal"] = "error"
    error_type: str = Field(min_length=1, max_length=MAX_ERROR_TYPE)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE)
    stack: str | None = Field(default=None, max_length=MAX_STACK)
    route: str | None = Field(default=None, max_length=MAX_ROUTE)
    user_agent: str | None = Field(default=None, max_length=MAX_USER_AGENT)
    app_version: str | None = Field(default=None, max_length=MAX_APP_VERSION)
    request_id: str | None = Field(default=None, max_length=MAX_REQUEST_ID)
    console_log: str | None = Field(default=None, max_length=MAX_CONSOLE_LOG)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("context")
    @classmethod
    def _context_fits(cls, value: dict[str, Any]) -> dict[str, Any]:
        if context_bytes(value) > MAX_CONTEXT_BYTES:
            raise ValueError(
                f"the context is larger than {MAX_CONTEXT_BYTES:,} bytes once serialised"
            )
        return value


class ErrorEventBatchIn(_Body):
    """Several events in one report."""

    events: list[ErrorEventIn] = Field(min_length=1, max_length=10)


ErrorReportIn = ErrorEventBatchIn | ErrorEventIn
"""The body of ``POST /api/v1/errors``: a batch when it has an ``events`` key, else one event."""
