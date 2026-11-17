"""Mapping of the workflow layer's :class:`WorkflowError` onto HTTP errors."""

from curvestudio.workflow import WorkflowError
from curvestudio_api.errors import ApiError


def api_error(exc: WorkflowError) -> ApiError:
    """413 ``TOO_LARGE`` for size limits, otherwise 422 ``VALIDATION_ERROR``; the workflow
    code is always in ``detail.code``."""
    detail = {"code": exc.code}
    if exc.code == "TOO_LARGE":
        return ApiError(413, "TOO_LARGE", exc.message, detail)
    return ApiError(422, "VALIDATION_ERROR", exc.message, detail)
