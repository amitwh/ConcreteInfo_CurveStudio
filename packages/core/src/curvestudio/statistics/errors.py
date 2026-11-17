"""Error type shared by every statistics routine."""


class StatisticsError(ValueError):
    """A statistic cannot be computed for the given fit, data or settings.

    ``code`` is a stable machine-readable identifier (``FIT_NOT_CONVERGED``,
    ``DATA_MISMATCH``, ``MODEL_MISMATCH``, ``NO_COVARIANCE``, ``NO_DEGREES_OF_FREEDOM``, ...)
    so callers and UIs can react without parsing the message.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)  # args=(code, message): survives pickling
        self.code = code

    def __str__(self) -> str:
        return str(self.args[1])
