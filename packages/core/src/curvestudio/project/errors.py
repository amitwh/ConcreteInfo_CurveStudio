"""Errors raised by the ``.cstudio`` project format.

Loading treats the archive as untrusted input: every failure reaches the caller as a
:class:`ProjectError` with a stable code, never as a traceback of another type. The
original exception, when there is one, is chained as the cause for debugging.
"""

from enum import StrEnum


class ProjectErrorCode(StrEnum):
    """Stable failure codes.

    The first five are load failures; ``UNSUPPORTED_DATA`` is raised while saving when
    the project holds something the format cannot represent exactly, and ``SAVE_FAILED``
    when the target file cannot be written.
    """

    NOT_A_PROJECT = "NOT_A_PROJECT"
    UNSAFE_ARCHIVE = "UNSAFE_ARCHIVE"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    CORRUPT = "CORRUPT"
    UNSUPPORTED_DATA = "UNSUPPORTED_DATA"
    SAVE_FAILED = "SAVE_FAILED"


class ProjectError(Exception):
    """A project could not be saved or loaded.

    ``code`` is stable and machine-readable; ``message`` is meant for people.
    """

    def __init__(self, code: ProjectErrorCode | str, message: str) -> None:
        super().__init__(message)
        self.code = ProjectErrorCode(code)
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
