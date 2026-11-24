"""The fit audit: check by check, what a finished fit's evidence shows and what to do next.

:func:`audit_fit` reads a :class:`~curvestudio.workflow.FitBundle` and returns a
:class:`FitAudit`. It fits nothing, reads and writes no file and touches no network. Every
number comes from the fitting core (``bundle.summary.diagnostics`` and
``bundle.summary.residuals``); the new computations are the residual-trend F test and the
exact Durbin-Watson p-values, which need the fit's own Jacobian.

A verdict is a statement of evidence, never of proof. A check never passes on a number it
does not have: without a p-value, a covariance or a summary it reads ``not_applicable`` and
gives the reason. Rows the audit cannot use are listed in the result, never dropped.

The module layout follows the seams: :mod:`._thresholds` holds every number a rule turns on,
:mod:`._types` what the audit returns, :mod:`._wording` the fixed advice, :mod:`._common` what
the checks share, and the checks sit in :mod:`._checks_fit`, :mod:`._checks_residuals`,
:mod:`._trend` and :mod:`._checks_selection`. This module re-exports the public names.
"""

from collections.abc import Mapping, Sequence

from curvestudio.data import XY
from curvestudio.data.errors import DataError
from curvestudio.diagnostics import FitReport
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
    AUTOCORRELATION_FAIL_P,
    CHANCE_CAUTION_Q,
    CHANCE_FAIL_Q,
    COOKS_CAUTION_REGION,
    COOKS_CHANCE_SHARE,
    COOKS_FAIL_REGION,
    CV_CAUTION_RATIO,
    CV_FAIL_RATIO,
    DW_CAUTION_P,
    DW_FAIL_P,
    DW_MIN_RESIDUAL_DF,
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
from curvestudio.tools.durbin_watson import (
    DW_EXACT_MAX_N,
    DurbinWatsonTest,
    durbin_watson_test,
)
from curvestudio.workflow import FitBundle, FitSpec, WorkflowError, fit_view, fitter_for

__all__ = [
    "AUTOCORRELATION_CAUTION_P",
    "AUTOCORRELATION_FAIL_P",
    "CHANCE_CAUTION_Q",
    "CHANCE_FAIL_Q",
    "COOKS_CAUTION_REGION",
    "COOKS_CHANCE_SHARE",
    "COOKS_FAIL_REGION",
    "CV_CAUTION_RATIO",
    "CV_FAIL_RATIO",
    "DW_CAUTION_P",
    "DW_FAIL_P",
    "DW_MIN_RESIDUAL_DF",
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
        # By the plain title each check carries when it could not run, never by its id.
        names = "; ".join(check.title for check in missing)
        text += f" {count} ({names})."
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
        Verdict.PASS: "no autocorrelation ({test} p = {p})",
        Verdict.CAUTION: "possible autocorrelation ({test} p = {p})",
        Verdict.FAIL: "autocorrelation ({test} p = {p})",
    },
    "residual_normality": {
        Verdict.PASS: "Shapiro-Wilk p = {p}",
        Verdict.CAUTION: "Shapiro-Wilk p = {p}, a possible departure from normality",
        Verdict.FAIL: "Shapiro-Wilk p = {p}, a departure from normality",
    },
}
"""How each residual check reads in a methods sentence, by verdict. The independence
clause names the test that decided it: the exact Durbin-Watson test when it ran, else
Ljung-Box."""
_TEST_OF = {
    "residual_normality": ("shapiro_wilk", "Shapiro-Wilk"),
    "residual_independence": ("ljung_box", "Ljung-Box"),
    "constant_variance": ("breusch_pagan", "Breusch-Pagan"),
}


def _residual_test(inp: Inputs, check_id: str) -> tuple[str, float] | None:
    """The test a residual check turned on and its p-value, read again from where the check
    read them."""
    summary = inp.summary
    if summary is None:
        return None
    if check_id == "residual_trend":
        trend = residual_trend(inp, summary)
        return ("F test", trend.p_value) if isinstance(trend, Trend) else None
    if check_id == "residual_independence" and inp.durbin_watson is not None:
        exact = inp.durbin_watson
        return "Durbin-Watson", min(exact.positive, exact.negative)
    test = core_test(summary, _TEST_OF[check_id][0])
    if test is None or test.p_value is None:
        return None
    return _TEST_OF[check_id][1], test.p_value


def _methods_fragment(inp: Inputs, checks: Mapping[str, AuditCheck]) -> str:
    """One or two sentences a methods paragraph can carry as it stands."""
    if inp.result.status is not FitStatus.CONVERGED:
        return "The fit did not converge, so no residual checks apply."
    clauses: dict[str, str] = {}
    for check_id, forms in _CLAUSES.items():
        verdict = checks[check_id].verdict
        found = _residual_test(inp, check_id)
        if verdict is not Verdict.NOT_APPLICABLE and found is not None:
            clauses[check_id] = forms[verdict].format(p=num(found[1]), test=found[0])
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


_WORKFLOW_REASONS = {
    "DATA_REQUIRED": (
        "the bundle holds only a subsample of the data and the full data were not given"
    ),
    "MODEL_MISMATCH": "the specification does not match the fit",
    "DATA_MISMATCH": "the data given are not the ones the fit was computed from",
}
"""What the workflow's refusals mean for the exact Durbin-Watson test; any other of its
codes reads as the workflow refusing the request."""
_TOOL_REASONS = {
    "RANK_DEFICIENT_DESIGN": "the parameters are not separated at the estimates",
    "INTEGRATION_NOT_CONVERGED": "its p-value integral did not reach its tolerance",
    "NO_DEGREES_OF_FREEDOM": "the fit has no residual degrees of freedom",
}
"""The same for the refusals of the test itself."""


def _exact_durbin_watson(
    bundle: FitBundle,
    spec: FitSpec | None,
    n: int,
    k: int,
    summary: FitReport | None,
    data: XY | None,
) -> tuple[DurbinWatsonTest | None, str | None]:
    """The exact Durbin-Watson test of the fit, or ``None`` and why it was not run.

    Both are ``None`` when there is nothing to add: the fit did not converge, or the
    independence check already says that it has no Durbin-Watson statistic to test.
    """
    result = bundle.result
    if spec is None:
        return None, "the exact Durbin-Watson test needs the fit's settings"
    if result.status is not FitStatus.CONVERGED:
        return None, None
    if n > DW_EXACT_MAX_N:
        return None, (
            f"the exact test is computed for at most {DW_EXACT_MAX_N} points and this fit has {n}"
        )
    if result.covariance is None:
        return None, "the fit has no covariance matrix"
    if n < k + DW_MIN_RESIDUAL_DF:
        return None, (
            f"this fit has {n} active points, fewer than its {k} parameters plus "
            f"{DW_MIN_RESIDUAL_DF}"
        )
    carried = None if summary is None else core_test(summary, "durbin_watson")
    if carried is None or carried.statistic is None:
        return None, None
    try:
        # The specification's weight choice first: a fit that ignored its weights recorded
        # the fingerprint of the data without them.
        fitted = None if data is None else fitter_for(spec, data, result=result)[1]
        view = fit_view(bundle, spec, data=fitted)
    except WorkflowError as exc:
        reason = _WORKFLOW_REASONS.get(exc.code, "the workflow refused the request")
        return None, f"{reason} ({exc.code}: {exc.message})"
    except DataError as exc:
        # The exact test needs the fit's data rebuilt, and the data layer refuses a missing
        # value (a blank weight, which no import lets through but a bundle can carry). The
        # audit still answers, on its other rule.
        return None, f"the fit's data hold a value the exact test cannot use ({exc.code})"
    try:
        return durbin_watson_test(carried.statistic, view.weighted_jacobian), None
    except ToolError as exc:
        return None, _TOOL_REASONS.get(exc.code, f"the test was refused ({exc.code})")


def audit_fit(
    bundle: FitBundle,
    *,
    spec: FitSpec | None = None,
    ranking: AutoFitResult | None = None,
    cv: CrossValidation | None = None,
    data: XY | None = None,
) -> FitAudit:
    """Audit a finished fit, check by check.

    Args:
        bundle: the fit with its data snapshot and, when it converged, its summary.
        spec: the specification the fit ran with. Its bounds are the ones checked; without
            it only a registered model's declared bounds are known. Given, it also lets the
            exact Durbin-Watson test decide the independence of the residuals.
        ranking: an Auto-fit result for the same dataset, to look for a simpler model.
        cv: a cross-validation of the same model on the same data, to set its held-out error
            against the in-sample one.
        data: the data the fit was computed from, as they were given to it. The exact
            Durbin-Watson test needs every point, so a compact bundle, which holds only a
            subsample, must be given them here. The specification's weight choice is applied
            before they are compared, so the data as uploaded are right for a fit that
            ignored its weights too.

    Returns:
        The checks in a fixed order with their overall verdict, a summary, a methods
        fragment and the rows that took no part.

    Raises:
        ToolError: ``NO_DATA`` when the bundle carries no observations at all.
    """
    snapshot = bundle.data
    if not snapshot.x:
        raise ToolError(
            "NO_DATA", "the bundle carries no observations, so there is nothing to audit"
        )
    result = bundle.result
    excluded, active_rows = split_rows(snapshot)
    stats = result.stats
    n = stats.n if stats is not None else len(result.fitted) - len(excluded)
    fewest, fewest_reason = simplest(ranking, result.model_id)
    summary = bundle.summary
    exact, exact_reason = _exact_durbin_watson(
        bundle, spec, n, len(result.parameters), summary, data
    )
    inputs = Inputs(
        bundle=bundle,
        result=result,
        spec=spec,
        ranking=ranking,
        cv=cv,
        n=n,
        k=len(result.parameters),
        summary=summary,
        active_rows=active_rows,
        simplest=fewest,
        simplest_reason=fewest_reason,
        durbin_watson=exact,
        durbin_watson_reason=exact_reason,
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
