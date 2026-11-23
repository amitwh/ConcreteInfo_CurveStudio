"""Checks on the fit itself and its parameters.

Convergence, an exact fit, whether each parameter is pinned down, and whether there are
enough points for them all.
"""

import math

from curvestudio.fitting import FitStatus, ParameterEstimate
from curvestudio.models import get_model
from curvestudio.tools._common import Check, Inputs, audit_check, is_exact, skipped
from curvestudio.tools._thresholds import (
    AT_BOUND,
    BOUND_RTOL,
    MIN_POINTS_PER_PARAMETER_CAUTION,
    MIN_POINTS_PER_PARAMETER_FAIL,
    NO_STATISTICS,
    POLYNOMIAL_ID,
    UNBOUNDED,
    WIDE_INTERVAL_SHARE,
)
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import (
    EXACT_NEXT_STEP,
    IDENTIFIABLE_NEXT_STEP,
    POINTS_NEXT_STEP,
    RESTART,
    named,
    num,
)


def check_converged(inp: Inputs) -> AuditCheck:
    result = inp.result
    status = result.status.value
    if result.termination_reason:
        evidence = f"Status {status}: {result.termination_reason}"
    else:
        evidence = f"Status {status}, with no reason given"
    if result.status is FitStatus.CONVERGED:
        return audit_check(
            CONVERGED, Verdict.PASS, "The fit converged", evidence, threshold="status converged"
        )
    return audit_check(
        CONVERGED,
        Verdict.FAIL,
        "The fit did not converge",
        evidence,
        threshold="status converged",
        next_step=RESTART,
    )


def check_perfect_fit(inp: Inputs) -> AuditCheck:
    stats = inp.result.stats
    if stats is None:
        return skipped(PERFECT_FIT, NO_STATISTICS)
    exact = is_exact(inp)
    if exact:
        verdict, title = Verdict.CAUTION, "The model passes through every point exactly"
        y = inp.bundle.data.y
        evidence = (
            f"Sum of squared residuals {num(stats.sse)}, RMSE {num(stats.rmse)} against a "
            f"data range of {num(max(y) - min(y))}: the residuals are zero to machine "
            "precision."
        )
    else:
        verdict, title = Verdict.PASS, "The model does not pass through every point"
        evidence = f"Sum of squared residuals {num(stats.sse)} (RMSE {num(stats.rmse)})."
    return audit_check(
        PERFECT_FIT,
        verdict,
        title,
        evidence,
        statistic=f"SSE = {num(stats.sse)}",
        threshold="SSE = 0 (to machine precision)",
        next_step=EXACT_NEXT_STEP if exact else None,
    )


def effective_bounds(inp: Inputs) -> tuple[dict[str, tuple[float, float]], bool]:
    """The bounds each parameter was fitted within, and whether they are known at all.

    The spec's bounds win. For a registered model the rest are its declared ones. Without a
    spec nothing is known for a custom equation or an id the registry does not have; a
    polynomial has none by nature.
    """
    bounds: dict[str, tuple[float, float]] = {}
    polynomial = POLYNOMIAL_ID.fullmatch(inp.result.model_id) is not None
    known = polynomial
    if not polynomial:
        try:
            model = get_model(inp.result.model_id)
        except KeyError:
            model = None
        if model is not None:
            known = True
            bounds = {p.name: (p.lower, p.upper) for p in model.parameters}
    if inp.spec is not None:
        bounds.update(
            {
                name: (
                    -math.inf if lower is None else lower,
                    math.inf if upper is None else upper,
                )
                for name, (lower, upper) in inp.spec.bounds.items()
            }
        )
    return bounds, known or inp.spec is not None


def parameter_finding(
    parameter: ParameterEstimate, bounds: tuple[float, float], reported: bool
) -> tuple[str, str]:
    """How one parameter stands and the sentence that says so.

    ``"bound"``: it sits on a bound. ``"weak"``: its interval spans its whole allowed range,
    its standard error exceeds its value, or either could not be determined. ``"fine"``:
    nothing of the kind, and no sentence.
    """
    name = parameter.name
    lower, upper = bounds
    ci_low, ci_high = parameter.ci_low, parameter.ci_high
    near = [
        bound
        for bound in bounds
        if math.isfinite(bound)
        and abs(parameter.value - bound) <= BOUND_RTOL * max(1.0, abs(bound))
    ]
    if near:
        return (
            "bound",
            f"{name} sits at a bound: estimate {num(parameter.value)} against the bound "
            f"{num(near[0])}.",
        )
    if reported:
        return "bound", f"{name} sits at a bound (the fit reported it there)."
    two_sided = math.isfinite(lower) and math.isfinite(upper)
    unbounded = not math.isfinite(lower) and not math.isfinite(upper)
    if two_sided and (ci_low is None or ci_high is None):
        missing = "standard error" if parameter.std_err is None else "interval"
        return "weak", f"{name}: its {missing} could not be determined."
    if two_sided and ci_low is not None and ci_high is not None:
        covered = (min(ci_high, upper) - max(ci_low, lower)) / (upper - lower)
        if covered >= WIDE_INTERVAL_SHARE:
            return (
                "weak",
                f"{name} has an interval from {num(ci_low)} to {num(ci_high)} covering "
                f"{num(covered * 100)} % of its allowed range {num(lower)} to "
                f"{num(upper)}.",
            )
    if unbounded and parameter.std_err is not None and parameter.std_err > abs(parameter.value):
        return (
            "weak",
            f"{name} has a standard error of {num(parameter.std_err)}, larger than its "
            f"value of {num(parameter.value)}.",
        )
    if parameter.std_err is None:
        return "weak", f"{name}: its standard error could not be determined."
    return "fine", ""


def check_parameters_identifiable(inp: Inputs) -> AuditCheck:
    result = inp.result
    bounds, known = effective_bounds(inp)
    reported = {
        warning.removeprefix(AT_BOUND).strip()
        for warning in result.warnings
        if warning.startswith(AT_BOUND)
    }
    at_bound: list[str] = []
    concerned: list[str] = []
    findings: list[str] = []
    for parameter in result.parameters:
        standing, finding = parameter_finding(
            parameter, bounds.get(parameter.name, UNBOUNDED), parameter.name in reported
        )
        if standing == "bound":
            at_bound.append(parameter.name)
        elif standing == "weak":
            concerned.append(parameter.name)
        if finding:
            findings.append(finding)
    no_covariance = result.covariance is None
    if no_covariance:
        findings.append("The fit has no covariance matrix, so parameter uncertainty is undefined.")
    if not known:
        findings.append("Parameters were checked without the fit's bounds.")
    statistic = (
        f"{len(at_bound)} of {len(result.parameters)} parameters at a bound, "
        f"{len(concerned)} poorly determined"
    )
    threshold = (
        f"within {num(BOUND_RTOL)} x max(1, |bound|) of a bound: failure; interval "
        f"covering >= {num(WIDE_INTERVAL_SHARE * 100)} % of the allowed range, or standard "
        "error > |value|: caution"
    )
    if at_bound:
        verdict, title = Verdict.FAIL, f"{named(at_bound, 'sits', 'sit')} at a bound"
        next_step = IDENTIFIABLE_NEXT_STEP
    elif concerned or no_covariance:
        verdict = Verdict.CAUTION
        title = (
            f"{named(concerned, 'is', 'are')} not pinned down by these data"
            if concerned
            else "Parameter uncertainty has no covariance matrix behind it"
        )
        next_step = IDENTIFIABLE_NEXT_STEP
    else:
        evidence = "Every parameter has a standard error and none sits at a bound."
        findings.insert(0, evidence)
        verdict, title, next_step = (
            Verdict.PASS,
            "No parameter sits at a bound or is poorly determined",
            None,
        )
    return audit_check(
        IDENTIFIABLE,
        verdict,
        title,
        " ".join(findings),
        statistic=statistic,
        threshold=threshold,
        next_step=next_step,
    )


def check_points_per_parameter(inp: Inputs) -> AuditCheck:
    n, k = inp.n, inp.k
    if k == 0:
        return skipped(POINTS, "the fit has no parameters to count points for")
    per = n / k
    per_text = f"{per:.0f}" if per >= 10 else f"{per:.1f}".rstrip("0").rstrip(".")
    statistic = f"{n} points for {k} parameters ({per_text} per parameter)"
    threshold = (
        f">= {MIN_POINTS_PER_PARAMETER_CAUTION} per parameter pass; "
        f">= {MIN_POINTS_PER_PARAMETER_FAIL} caution; "
        f"below {MIN_POINTS_PER_PARAMETER_FAIL} fail"
    )
    if per >= MIN_POINTS_PER_PARAMETER_CAUTION:
        verdict = Verdict.PASS
        title = f"There are at least {MIN_POINTS_PER_PARAMETER_CAUTION} points per parameter"
    elif per < MIN_POINTS_PER_PARAMETER_FAIL:
        verdict = Verdict.FAIL
        title = f"There are fewer than {MIN_POINTS_PER_PARAMETER_FAIL} points per parameter"
    else:
        verdict = Verdict.CAUTION
        title = f"There are fewer than {MIN_POINTS_PER_PARAMETER_CAUTION} points per parameter"
    return audit_check(
        POINTS,
        verdict,
        title,
        f"{statistic}.",
        statistic=statistic,
        threshold=threshold,
        next_step=None if verdict is Verdict.PASS else POINTS_NEXT_STEP,
    )


CONVERGED = Check(
    "converged",
    "Convergence of the fit was not assessed",
    "A fit that stopped before it converged has no usable estimates, so nothing else can be "
    "judged until it does.",
    check_converged,
)
PERFECT_FIT = Check(
    "perfect_fit",
    "Whether the fit is exact was not assessed",
    "A model with as many parameters as points passes through every point exactly, and that "
    "says nothing about the data. An exact fit on more points leaves no scatter for the "
    "residual tests.",
    check_perfect_fit,
)
IDENTIFIABLE = Check(
    "parameters_identifiable",
    "The parameters were not checked",
    "A parameter the data do not constrain can take many values with little change in the "
    "fit, so its estimate and interval mean little.",
    check_parameters_identifiable,
)
POINTS = Check(
    "points_per_parameter",
    "Points per parameter were not counted",
    "Each parameter needs several points to be estimated with a usable interval. With few "
    "points per parameter the fit follows noise and the intervals widen.",
    check_points_per_parameter,
)
