"""What the checks share: what they read, the record that runs one, and readers of numbers.

The readers look up what the fitting core already computed (a diagnostic test, the summary's
reason for being absent, the rows a fit used) so that no check recomputes it.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from curvestudio.diagnostics import DiagnosticTest, FitReport
from curvestudio.discovery import AutoFitResult, RankedFit
from curvestudio.fitting import FitResult
from curvestudio.statistics import CrossValidation
from curvestudio.tools._thresholds import EXACT_FIT_RATIO, OFFSET_FLOOR_RATIO
from curvestudio.tools._types import AuditCheck, ExcludedRow, Verdict
from curvestudio.tools._wording import capped
from curvestudio.workflow import DataSnapshot, FitBundle, FitSpec


@dataclass(frozen=True)
class Inputs:
    """Everything the checks read, resolved once."""

    bundle: FitBundle
    result: FitResult
    spec: FitSpec | None
    ranking: AutoFitResult | None
    cv: CrossValidation | None
    n: int
    k: int
    summary: FitReport | None
    active_rows: list[int]
    simplest: RankedFit | None
    simplest_reason: str | None


@dataclass(frozen=True)
class Check:
    """One check: the text that never changes, and the function that runs it."""

    id: str
    untested: str
    about: str
    run: Callable[[Inputs], AuditCheck]


def audit_check(
    check: Check,
    verdict: Verdict,
    title: str,
    evidence: str,
    *,
    statistic: str | None = None,
    threshold: str | None = None,
    next_step: str | None = None,
) -> AuditCheck:
    """A check's result: its own text goes with the ``Check`` that produced it."""
    return AuditCheck(
        id=check.id,
        title=title,
        verdict=verdict,
        statistic=statistic,
        threshold=threshold,
        evidence=evidence,
        explanation=check.about,
        next_step=next_step,
    )


def skipped(check: Check, reason: str, next_step: str | None = None) -> AuditCheck:
    """The check as it reads when it had no number to run on."""
    return audit_check(check, Verdict.NOT_APPLICABLE, check.untested, reason, next_step=next_step)


def title_for(titles: tuple[str, str, str], verdict: Verdict) -> str:
    """The title for a verdict, from its ``(pass, caution, fail)`` forms."""
    return titles[(Verdict.PASS, Verdict.CAUTION, Verdict.FAIL).index(verdict)]


def p_verdict(p: float, caution: float, fail: float) -> Verdict:
    if p < fail:
        return Verdict.FAIL
    return Verdict.CAUTION if p < caution else Verdict.PASS


def is_exact(inp: Inputs) -> bool:
    """Whether the residuals are zero to machine precision, as the core's diagnostics judge it."""
    stats = inp.result.stats
    if stats is None:
        return False
    if stats.rmse == 0.0:
        return True
    y = inp.bundle.data.y
    if not y:
        return False
    scale = max(max(y) - min(y), OFFSET_FLOOR_RATIO * max(abs(value) for value in y))
    return scale > 0.0 and stats.rmse <= EXACT_FIT_RATIO * scale


def where(inp: Inputs, positions: Sequence[int], points: int) -> str:
    """Where flagged points are: rows of the data when the data are the points, else positions."""
    if len(inp.active_rows) == points:
        rows = [str(inp.active_rows[position] + 1) for position in positions]
        return capped(rows, "row")
    counted = [str(position + 1) for position in positions]
    return (
        f"{capped(counted, 'point')}, counted from 1 among the points the fit used, in data order"
    )


def core_test(summary: FitReport, name: str) -> DiagnosticTest | None:
    """The diagnostic test the core calls ``name``, when the summary holds it."""
    return next((test for test in summary.diagnostics.tests if test.name == name), None)


def no_summary(inp: Inputs) -> str:
    return inp.bundle.summary_reason or "the fit has no summary"


def read_test(
    inp: Inputs, check: Check, name: str, label: str
) -> AuditCheck | tuple[DiagnosticTest, float]:
    """The core's test ``name`` with its p-value, or the check that says why it cannot be read."""
    if inp.summary is None:
        return skipped(check, no_summary(inp))
    test = core_test(inp.summary, name)
    if test is None:
        return skipped(check, f"the {label} test is not in the summary")
    if test.status == "not_applicable":
        return skipped(check, test.reason or f"the {label} test did not apply")
    if test.p_value is None:
        return skipped(check, f"the {label} test returned no p-value")
    return test, test.p_value


def split_rows(data: DataSnapshot) -> tuple[tuple[ExcludedRow, ...], list[int]]:
    """The rows the audit leaves out, each with its reason, and the rows it uses, in order.

    Data without a weights column use every row. Where weights exist, a row whose weight is
    0 takes no part in the fit and one whose weight is blank cannot.
    """
    if data.weights is None:
        return (), list(range(len(data.x)))
    excluded: list[ExcludedRow] = []
    active: list[int] = []
    for row in range(len(data.x)):
        weight = data.weights[row] if row < len(data.weights) else None
        if weight is None or math.isnan(weight):
            excluded.append(ExcludedRow(index=row, reason="weight is missing"))
        elif weight == 0.0:
            excluded.append(ExcludedRow(index=row, reason="weight is zero"))
        else:
            active.append(row)
    return tuple(excluded), active
