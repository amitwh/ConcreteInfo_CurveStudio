"""The one exception the workflow layer raises for problems the caller can fix."""


class WorkflowError(Exception):
    """A user-caused problem, with a stable machine-readable ``code``.

    Codes: ``TOO_LARGE``, ``UNREADABLE``, ``UNSUPPORTED_FORMAT`` (import); ``INVALID_SPEC``,
    ``INVALID_EQUATION``, ``UNKNOWN_MODEL`` (fit specification); ``FIT_NOT_CONVERGED``,
    ``MODEL_MISMATCH``, ``DATA_MISMATCH``, ``DATA_REQUIRED`` (prediction). Numerical outcomes
    (a fit that does not converge, a statistic that is undefined) are not errors: they are
    reported in the returned bundle.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)  # args must be the constructor's: keeps it picklable
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
