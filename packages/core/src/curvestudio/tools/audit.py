"""The fit audit: check by check, what a finished fit's evidence shows and what to do next.

:func:`audit_fit` reads a :class:`~curvestudio.workflow.FitBundle` and returns a
:class:`FitAudit`. It fits nothing, reads and writes no file and touches no network. Every
number comes from the fitting core (``bundle.summary.diagnostics`` and
``bundle.summary.residuals``); the one new computation is the residual-trend F test.

A verdict is a statement of evidence, never of proof. A check never passes on a number it
does not have: without a p-value, a covariance or a summary it reads ``not_applicable`` and
gives the reason. Rows the audit cannot use are listed in the result, never dropped.

The module layout follows the seams: :mod:`._thresholds` holds every number a rule turns on,
:mod:`._types` what the audit returns, :mod:`._wording` the fixed advice, :mod:`._common` what
the checks share, and the checks sit in :mod:`._checks_fit`, :mod:`._checks_residuals`,
:mod:`._trend` and :mod:`._checks_selection`. This module re-exports the public names.
"""

from collections.abc import Mapping, Sequence

from curvestudio.discovery import AutoFitResult, RankedFit
from curvestudio.fitting import FitResult, FitStatus
from curvestudio.statistics import CrossValidation
from curvestudio.tools._checks_fit import CONVERGED, IDENTIFIABLE, PERFECT_FIT, POINTS
from curvestudio.tools._checks_residuals import (
    CONSTANT_VARIANCE,
    INDEPENDENCE,
    INFLUENCE,
    NORMALITY,
)
from curvestudio.tools._checks_selection import (
    CROSS_VALIDATION,
    SIMPLER_MODEL,
    charged,
    expected_cv_ratio,
    simplest,
)
from curvestudio.tools._common import Check, Inputs, core_test, skipped, split_rows
from curvestudio.tools._errors import ToolError
from curvestudio.tools._thresholds import (
    AUTOCORRELATION_CAUTION_P,
    CHANCE_CAUTION_Q,
    CHANCE_FAIL_Q,
    COOKS_CAUTION_REGION,
    COOKS_CHANCE_SHARE,
    COOKS_FAIL_REGION,
    CV_CAUTION_RATIO,
    CV_FAIL_RATIO,
    DW_CAUTION_P,
    DW_FAIL_P,
    MIN_POINTS_PER_PARAMETER_CAUTION,
    MIN_POINTS_PER_PARAMETER_FAIL,
    NORMALITY_CAUTION_P,
    NORMALITY_FAIL_P,
    NOT_CONVERGED,
    SIMPLE_DELTA_AICC,
    SMALL_SAMPLE_N,
    TREND_CAUTION_P,
    TREND_EXTRA_TERMS,
    TREND_FAIL_P,
    TREND_MIN_RESIDUAL_DF,
    TREND_MODEL_TERMS,
    VARIANCE_CAUTION_P,
    VARIANCE_FAIL_P,
    WIDE_INTERVAL_SHARE,
)
from curvestudio.tools._trend import TREND, Trend, residual_trend
from curvestudio.tools._types import (
    AuditCheck,
    ExcludedRow,
    FitAudit,
    SimplerModel,
    Verdict,
)
from curvestudio.tools._wording import joined, needs_look, num
from curvestudio.workflow import FitBundle, FitSpec

__all__ = [
    "AUTOCORRELATION_CAUTION_P",
    "CHANCE_CAUTION_Q",
    "CHANCE_FAIL_Q",
    "COOKS_CAUTION_REGION",
    "COOKS_CHANCE_SHARE",
    "COOKS_FAIL_REGION",
    "CV_CAUTION_RATIO",
    "CV_FAIL_RATIO",
    "DW_CAUTION_P",
    "DW_FAIL_P",
    "MIN_POINTS_PER_PARAMETER_CAUTION",
    "MIN_POINTS_PER_PARAMETER_FAIL",
    "NORMALITY_CAUTION_P",
    "NORMALITY_FAIL_P",
    "SIMPLE_DELTA_AICC",
    "SMALL_SAMPLE_N",
    "TREND_CAUTION_P",
    "TREND_EXTRA_TERMS",
    "TREND_FAIL_P",
    "TREND_MIN_RESIDUAL_DF",
    "TREND_MODEL_TERMS",
    "VARIANCE_CAUTION_P",
    "VARIANCE_FAIL_P",
    "WIDE_INTERVAL_SHARE",
    "AuditCheck",
    "ExcludedRow",
    "FitAudit",
    "SimplerModel",
    "Verdict",
    "audit_fit",
    "expected_cv_ratio",
]

CHECKS: tuple[Check, ...] = (
    CONVERGED,
    PERFECT_FIT,
    NORMALITY,
    INDEPENDENCE,
    CONSTANT_VARIANCE,
    INFLUENCE,
    TREND,
    IDENTIFIABLE,
    POINTS,
    CROSS_VALIDATION,
    SIMPLER_MODEL,
)
"""Every check, in the order the audit reports it."""


def _run_checks(inp: Inputs) -> tuple[AuditCheck, ...]:
    """Every check in order. Once the fit has not converged, the rest only say so."""
    converged = inp.result.status is FitStatus.CONVERGED
    return tuple(
        check.run(inp) if converged or check is CONVERGED else skipped(check, NOT_CONVERGED)
        for check in CHECKS
    )


def _overall(checks: Sequence[AuditCheck]) -> Verdict:
    verdicts = {check.verdict for check in checks}
    for verdict in (Verdict.FAIL, Verdict.CAUTION, Verdict.PASS):
        if verdict in verdicts:
            return verdict
    return Verdict.NOT_APPLICABLE


def _summary(checks: Sequence[AuditCheck], result: FitResult) -> str:
    """One to three sentences: what happened and, if anything was flagged, what to do first."""
    if result.status is not FitStatus.CONVERGED:
        reason = result.termination_reason
        head = f"The fit did not converge ({reason})," if reason else "The fit did not converge,"
        return f"{head} so there is nothing to audit yet. {checks[0].next_step}"
    failed = [check for check in checks if check.verdict is Verdict.FAIL]
    cautioned = [check for check in checks if check.verdict is Verdict.CAUTION]
    missing = [check for check in checks if check.verdict is Verdict.NOT_APPLICABLE]
    if failed:
        count = "1 check failed" if len(failed) == 1 else f"{len(failed)} checks failed"
        look = f", {needs_look(len(cautioned))}" if cautioned else ""
        return f"{count}{look}. {failed[0].title}. {failed[0].next_step}"
    if cautioned:
        first = cautioned[0]
        return f"No check failed; {needs_look(len(cautioned))}. {first.title}. {first.next_step}"
    text = "Every check that could run passed."
    if missing:
        count = "1 could not be run" if len(missing) == 1 else f"{len(missing)} could not be run"
        text += f" {count}: {', '.join(check.id for check in missing)}."
    return text


_CLAUSES: dict[str, dict[Verdict, str]] = {
    "residual_trend": {
        Verdict.PASS: "no trend in x (p = {p})",
        Verdict.CAUTION: "a possible trend in x (p = {p})",
        Verdict.FAIL: "a trend in x (p = {p})",
    },
    "constant_variance": {
        Verdict.PASS: "constant variance (Breusch-Pagan p = {p})",
        Verdict.CAUTION: "possibly changing variance (Breusch-Pagan p = {p})",
        Verdict.FAIL: "variance that changes with the fitted value (Breusch-Pagan p = {p})",
    },
    "residual_independence": {
        Verdict.PASS: "no autocorrelation (Ljung-Box p = {p})",
        Verdict.CAUTION: "possible autocorrelation (Ljung-Box p = {p})",
        Verdict.FAIL: "autocorrelation (Ljung-Box p = {p})",
    },
    "residual_normality": {
        Verdict.PASS: "Shapiro-Wilk p = {p}",
        Verdict.CAUTION: "Shapiro-Wilk p = {p}, a possible departure from normality",
        Verdict.FAIL: "Shapiro-Wilk p = {p}, a departure from normality",
    },
}
"""How each residual check reads in a methods sentence, by verdict."""
_TEST_OF = {
    "residual_normality": "shapiro_wilk",
    "residual_independence": "ljung_box",
    "constant_variance": "breusch_pagan",
}


def _residual_p(inp: Inputs, check_id: str) -> float | None:
    """The p-value a residual check turned on, read again from where the check read it."""
    summary = inp.summary
    if summary is None:
        return None
    if check_id == "residual_trend":
        trend = residual_trend(inp, summary)
        return trend.p_value if isinstance(trend, Trend) else None
    test = core_test(summary, _TEST_OF[check_id])
    return None if test is None else test.p_value


def _methods_fragment(inp: Inputs, checks: Mapping[str, AuditCheck]) -> str:
    """One or two sentences a methods paragraph can carry as it stands."""
    if inp.result.status is not FitStatus.CONVERGED:
        return "The fit did not converge, so no residual checks apply."
    clauses: dict[str, str] = {}
    for check_id, forms in _CLAUSES.items():
        verdict = checks[check_id].verdict
        p = _residual_p(inp, check_id)
        if verdict is not Verdict.NOT_APPLICABLE and p is not None:
            clauses[check_id] = forms[verdict].format(p=num(p))
    shape = [
        clauses[check_id]
        for check_id in ("residual_trend", "constant_variance", "residual_independence")
        if check_id in clauses
    ]
    normal = clauses.get("residual_normality")
    if shape and normal:
        sentence = f"Residuals showed {joined(shape)}; {normal}."
    elif shape:
        sentence = f"Residuals showed {joined(shape)}."
    elif normal:
        sentence = f"Only the normality test could be run: {normal}."
    elif checks[PERFECT_FIT.id].verdict is Verdict.CAUTION:
        sentence = (
            "The model passes through every point exactly, so the residual tests carry no "
            "information."
        )
    else:
        sentence = "No residual test could be run on this fit."
    if checks[CROSS_VALIDATION.id].verdict is Verdict.NOT_APPLICABLE:
        ran = "did not complete" if inp.cv is not None else "was not run"
        sentence += f" Cross-validation {ran}."
    return sentence


def _simpler_model(row: RankedFit) -> SimplerModel:
    return SimplerModel(
        model_id=row.model_id,
        name=row.name,
        n_parameters=charged(row),
        delta_aicc=row.delta_aicc,
        cv_rmse=row.metrics.cv_rmse,
    )


def audit_fit(
    bundle: FitBundle,
    *,
    spec: FitSpec | None = None,
    ranking: AutoFitResult | None = None,
    cv: CrossValidation | None = None,
) -> FitAudit:
    """Audit a finished fit, check by check.

    Args:
        bundle: the fit with its data snapshot and, when it converged, its summary.
        spec: the specification the fit ran with. Its bounds are the ones checked; without
            it only a registered model's declared bounds are known.
        ranking: an Auto-fit result for the same dataset, to look for a simpler model.
        cv: a cross-validation of the same model on the same data, to set its held-out error
            against the in-sample one.

    Returns:
        The checks in a fixed order with their overall verdict, a summary, a methods
        fragment and the rows that took no part.

    Raises:
        ToolError: ``NO_DATA`` when the bundle carries no observations at all.
    """
    data = bundle.data
    if not data.x:
        raise ToolError(
            "NO_DATA", "the bundle carries no observations, so there is nothing to audit"
        )
    result = bundle.result
    excluded, active_rows = split_rows(data)
    stats = result.stats
    n = stats.n if stats is not None else len(result.fitted) - len(excluded)
    fewest, fewest_reason = simplest(ranking, result.model_id)
    inputs = Inputs(
        bundle=bundle,
        result=result,
        spec=spec,
        ranking=ranking,
        cv=cv,
        n=n,
        k=len(result.parameters),
        summary=bundle.summary,
        active_rows=active_rows,
        simplest=fewest,
        simplest_reason=fewest_reason,
    )
    checks = _run_checks(inputs)
    by_id = {check.id: check for check in checks}
    simpler = (
        _simpler_model(fewest)
        if fewest is not None and by_id[SIMPLER_MODEL.id].verdict is Verdict.CAUTION
        else None
    )
    return FitAudit(
        checks=checks,
        overall=_overall(checks),
        summary=_summary(checks, result),
        methods_fragment=_methods_fragment(inputs, by_id),
        simpler_model=simpler,
        n_observations=n,
        n_parameters=len(result.parameters),
        excluded=excluded,
    )
