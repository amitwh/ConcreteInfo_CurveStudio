"""Errors raised by the data layer."""


class DataError(Exception):
    """A data problem that the caller can report to the user.

    ``code`` is a stable machine-readable identifier (for example ``MISSING_VALUE``).
    ``rows`` holds 0-based positions in the original table of the offending rows,
    or an empty tuple when the error is not tied to specific rows.
    """

    def __init__(self, code: str, message: str, rows: tuple[int, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.rows = rows

    def __str__(self) -> str:
        suffix = f" (rows {', '.join(map(str, self.rows))})" if self.rows else ""
        return f"{self.code}: {super().__str__()}{suffix}"
