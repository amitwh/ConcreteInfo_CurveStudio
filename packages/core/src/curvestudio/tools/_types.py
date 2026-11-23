"""What the audit returns: a verdict per check, the check, and the audit as a whole.

:mod:`curvestudio.tools.audit` re-exports every class here; import them from there.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Verdict(StrEnum):
    """What one check's evidence says. ``NOT_APPLICABLE``: the check had no number to read."""

    PASS = "pass"
    CAUTION = "caution"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class AuditCheck(BaseModel):
    """One check of the audit.

    ``title`` states what was found, phrased for the verdict. ``statistic`` is the number
    the verdict turned on, as text, and ``threshold`` the rule that turned it. ``evidence``
    says what was read, or (for ``NOT_APPLICABLE``) why nothing could be. ``explanation``
    is one or two plain sentences on what the check is about, and ``next_step`` what to do
    about a ``caution`` or ``fail``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    verdict: Verdict
    statistic: str | None
    threshold: str | None
    evidence: str
    explanation: str
    next_step: str | None


class SimplerModel(BaseModel):
    """A ranked model that fits as well with fewer parameters.

    ``n_parameters`` is the count the ranking charges: fitted parameters plus free-frequency
    shape parameters. ``delta_aicc`` is its distance from the best model's AICc.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    name: str
    n_parameters: int
    delta_aicc: float | None
    cv_rmse: float | None


class ExcludedRow(BaseModel):
    """A row of the bundle's data that takes part in no check, and why. ``index`` is 0-based."""

    model_config = ConfigDict(frozen=True)

    index: int
    reason: str


class FitAudit(BaseModel):
    """The audit of one fit.

    ``checks`` come in a fixed order: convergence, the fit itself, the residual checks, the
    parameters, the data size, cross-validation and a simpler model. ``overall`` is the
    worst of them. ``summary`` is one to three sentences for a banner; ``methods_fragment``
    one or two a methods paragraph can carry as it stands. ``simpler_model`` is set only
    when that check reads ``caution``. ``n_observations`` counts the rows that took part;
    ``excluded`` lists the rest.
    """

    model_config = ConfigDict(frozen=True)

    checks: tuple[AuditCheck, ...]
    overall: Verdict
    summary: str
    methods_fragment: str
    simpler_model: SimplerModel | None
    n_observations: int
    n_parameters: int
    excluded: tuple[ExcludedRow, ...]
