"""The one exception the tools raise for an input no check can run on."""


class ToolError(Exception):
    """An input a tool cannot work on, with a stable machine-readable ``code``.

    The codes the tools raise themselves, each with what it means:

    - ``NO_DATA``: the bundle carries no observations, or a Jacobian is empty.
    - ``NO_STATISTICS``: the fit has no statistics for the methods text to write.
    - ``FIT_NOT_CONVERGED_INPUT``: cross-validation was asked of a fit that did not converge.
    - ``DATA_REQUIRED``: a compact bundle holds only a subsample, and no full data were given.
    - ``DATA_MISMATCH``: the data given are not the ones the fit was computed from.
    - ``MODEL_MISMATCH``: the specification is not the one the fit was computed with.
    - ``INVALID_FOLDS``: a number of folds the cross-validation cannot use.
    - ``INVALID_LEVEL``: an interval level outside (0, 1).
    - ``RANK_DEFICIENT_DESIGN``: the columns of a Jacobian are dependent, or not all finite.
    - ``NO_DEGREES_OF_FREEDOM``: a fit leaves no residual direction beside its parameters.
    - ``INTEGRATION_NOT_CONVERGED``: a quadrature could not reach its tolerance, so the value
      it stopped at is not returned as if it were exact.

    What the workflow refuses in a specification (``INVALID_SPEC``, ``UNKNOWN_MODEL``,
    ``INVALID_EQUATION``) or the core in a setting (``INVALID_SEED``, ...) is raised again
    with its own code. A fit that did not converge, a statistic that is undefined, a
    numerical shortfall a check can name, is not an error: the tool reports it in its
    result, check by check. ``INTEGRATION_NOT_CONVERGED`` is the one raised, and the audit
    reports it the same way, as the reason a check fell back on another rule.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)  # args must be the constructor's: keeps it picklable
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
