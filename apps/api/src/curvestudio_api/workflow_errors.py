"""Mapping of the workflow layer's :class:`WorkflowError` and the tools layer's
:class:`ToolError` onto HTTP errors."""

from curvestudio.discovery import AutoFitError
from curvestudio.tools import ToolError
from curvestudio.workflow import WorkflowError
from curvestudio_api.errors import ApiError


def _refused(code: str, message: str) -> ApiError:
    """413 ``TOO_LARGE`` for size limits, otherwise 422 ``VALIDATION_ERROR``; the engine's code
    is always in ``detail.code``."""
    detail = {"code": code}
    if code == "TOO_LARGE":
        return ApiError(413, "TOO_LARGE", message, detail)
    return ApiError(422, "VALIDATION_ERROR", message, detail)


def api_error(exc: WorkflowError) -> ApiError:
    """A request the workflow refuses, with the workflow code in ``detail.code``."""
    return _refused(exc.code, exc.message)


def tool_api_error(exc: ToolError) -> ApiError:
    """A request a tool refuses, answered as the workflow's equivalent is.

    The tool's code and message, and the status ``api_error`` gives that code: only
    ``TOO_LARGE`` is not a 422. A tool code the workflow has no equivalent for
    (``INVALID_FOLDS``, ``NO_DATA``, ...) is a 422 ``VALIDATION_ERROR`` whose ``detail.code``
    names it.
    """
    return _refused(exc.code, exc.message)


def autofit_api_error(exc: AutoFitError) -> ApiError:
    """422 ``VALIDATION_ERROR`` for an Auto-fit request the engine refuses; the Auto-fit code
    (``TOO_FEW_POINTS``, ``UNKNOWN_FAMILY``, ...) is in ``detail.code``."""
    return ApiError(422, "VALIDATION_ERROR", str(exc), {"code": exc.code})
