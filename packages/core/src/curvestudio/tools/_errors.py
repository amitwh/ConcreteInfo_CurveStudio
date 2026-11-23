"""The one exception the tools raise for an input no check can run on."""


class ToolError(Exception):
    """An input a tool cannot audit, with a stable machine-readable ``code``.

    Codes: ``NO_DATA`` (the bundle carries no observations). A numerical shortfall, a fit
    that did not converge, a statistic that is undefined, is not an error: the tool reports
    it in its result, check by check.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)  # args must be the constructor's: keeps it picklable
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
