"""The fit audit: check by check, what a finished fit's evidence shows and what to do next.

:func:`audit_fit` reads a :class:`~curvestudio.workflow.FitBundle` and returns a
:class:`FitAudit`. It fits nothing, reads and writes no file and touches no network. Every
number comes from the fitting core (``bundle.summary.diagnostics`` and
``bundle.summary.residuals``); the one new computation is the residual-trend F test, which
the core does not have.

A verdict is a statement of evidence, never of proof. A check never passes on a number it
does not have: without a p-value, a covariance or a summary it reads ``not_applicable`` and
gives the reason. Rows the audit cannot use are listed in the result, never dropped.
"""

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import stats as scipy_stats

from curvestudio.diagnostics import DiagnosticTest, FitReport
from curvestudio.discovery import AutoFitResult, RankedFit
from curvestudio.fitting import FitResult, FitStatus, ParameterEstimate
from curvestudio.models import get_model
from curvestudio.statistics import CrossValidation
from curvestudio.tools._errors import ToolError
from curvestudio.workflow import DataSnapshot, FitBundle, FitSpec

FloatArray = NDArray[np.float64]

# -- thresholds ---------------------------------------------------------------------------------
# Each is the rule it stands for; the checks below read no other number of this kind.

NORMALITY_CAUTION_P = 0.05
"""Shapiro-Wilk p below this, but not below ``NORMALITY_FAIL_P``, reads ``caution``."""
NORMALITY_FAIL_P = 0.01
"""Shapiro-Wilk p below this reads ``fail``."""
SMALL_SAMPLE_N = 20
"""With fewer points than this the normality evidence says the test rarely detects anything."""
VARIANCE_CAUTION_P = 0.05
"""Breusch-Pagan p below this, but not below ``VARIANCE_FAIL_P``, reads ``caution``."""
VARIANCE_FAIL_P = 0.01
"""Breusch-Pagan p below this reads ``fail``."""
TREND_CAUTION_P = 0.05
"""Residual-trend p below this, but not below ``TREND_FAIL_P``, reads ``caution``."""
TREND_FAIL_P = 0.01
"""Residual-trend p below this reads ``fail``."""
INFLUENCE_CAUTION_COUNT = 1
"""From this many points with a Cook's distance above 4/n the check reads ``caution``."""
INFLUENCE_FAIL_COUNT = 3
"""From this many points with a Cook's distance above 4/n the check reads ``fail``."""
INFLUENCE_FAIL_SHARE = 0.10
"""Points with a Cook's distance above 4/n taking at least this share of all points: ``fail``."""
LEVERAGE_CAUTION_SHARE = 0.20
"""High-leverage points taking at least this share of all points are called many in the
evidence; leverage alone never changes the verdict."""
MIN_POINTS_PER_PARAMETER_CAUTION = 10
"""Fewer points per parameter than this, but at least ``MIN_POINTS_PER_PARAMETER_FAIL``:
``caution``."""
MIN_POINTS_PER_PARAMETER_FAIL = 5
"""Fewer points per parameter than this reads ``fail``."""
CV_CAUTION_RATIO = 1.1
"""A cross-validation RMSE above this multiple of the in-sample RMSE reads ``caution``."""
CV_FAIL_RATIO = 1.5
"""A cross-validation RMSE above this multiple of the in-sample RMSE reads ``fail``."""
SIMPLE_DELTA_AICC = 2.0
"""Ranked models within this AICc of the best count as fitting as well as it."""
WIDE_INTERVAL_SHARE = 0.99
"""A confidence interval covering at least this share of the parameter's allowed range reads
``caution``."""
DW_CAUTION_P = 0.05
"""Ljung-Box p below this, but not below ``DW_FAIL_P``, reads ``caution`` for independence.
This task decides on Ljung-Box; the Durbin-Watson statistic itself gets no verdict."""
DW_FAIL_P = 0.01
"""Ljung-Box p below this reads ``fail`` for residual independence."""
AUTOCORRELATION_CAUTION_P = 0.05
"""Ljung-Box p below this, but not below ``DW_FAIL_P``, reads ``caution`` for independence."""

_BOUND_RTOL = 1e-8
"""A parameter within this relative distance of a finite bound sits on it: the fitting
layer's own rule, by which it writes ``PARAMETER_AT_BOUND``."""
_EXACT_FIT_RATIO = 1e-10
"""A fit whose RMSE is at most this fraction of the data's scale is exact: the diagnostics
layer's own rule, by which it withholds the residual tests."""
_OFFSET_FLOOR_RATIO = 1e-4
"""Rounding in a large ``y`` offset reaches this fraction of ``max|y|``; it floors the scale."""
_TREND_MIN_POINTS = 8
"""The residual-trend F test needs this many points for its denominator ``n - 3`` to say
anything."""
_AT_BOUND = "PARAMETER_AT_BOUND: "
_LISTED_MAX = 8
"""The evidence names at most this many flagged points; beyond that it gives the count of the
rest, so a sentence stays readable however many points are flagged."""
_POLYNOMIAL_ID = re.compile(r"poly(\d+)(_origin)?")
"""Model ids ``fit_polynomial`` writes (``poly2``, ``poly2_origin``); a polynomial has no
bounds."""

_NOT_CONVERGED = "the fit did not converge"
_NO_STATISTICS = "the fit has no statistics"
_UNBOUNDED = (-math.inf, math.inf)

_RESTART = "Check the starting values and the bounds, or try a model with fewer parameters."
_EXACT_NEXT_STEP = (
    "There is no scatter to estimate uncertainty from, so report no intervals or p-values "
    "for this fit."
)
_NORMALITY_NEXT_STEP = (
    "If a few points cause it, try a robust loss (Huber or soft-L1). Otherwise say in your "
    "write-up that the intervals rely on an approximation."
)
_INDEPENDENCE_NEXT_STEP = (
    "Plot the residuals against x in data order. If they wander, the model is missing a "
    "shape; if they alternate, look at how the data were taken."
)
_VARIANCE_NEXT_STEP = (
    "If the spread grows with the signal, use weights from replicates or a robust loss, and say so."
)
_TREND_NEXT_STEP = (
    "The residuals show a curved trend, so the model is missing a shape the data contain. "
    "Try a model with one more parameter."
)
_IDENTIFIABLE_NEXT_STEP = (
    "A parameter at a bound or with a wide interval is not pinned down by these data. Fix "
    "the bound if it is wrong, take more points where the model is sensitive, or use a model "
    "with fewer parameters."
)
_POINTS_NEXT_STEP = "Take more points, or use a simpler model."
_CV_NEXT_STEP = (
    "The fit describes noise as well as signal. A simpler model, or one with fewer "
    "parameters, will predict better."
)
_SIMPLER_NEXT_STEP = (
    "Report the simpler model unless the extra parameter answers a question of your own."
)


# -- result types -------------------------------------------------------------------------------


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


# -- what the checks read -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Inputs:
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
class _Check:
    """One check: the text that never changes, and the function that runs it."""

    id: str
    untested: str
    about: str
    run: Callable[[_Inputs], AuditCheck]


def _num(value: float) -> str:
    """A number for a sentence: at most three significant digits, never ``-0``."""
    text = f"{value:.3g}"
    return "0" if text == "-0" else text


def _p_rule(caution: float, fail: float) -> str:
    return (
        f"p >= {_num(caution)} pass; {_num(fail)} <= p < {_num(caution)} caution; "
        f"p < {_num(fail)} fail"
    )


def _p_verdict(p: float, caution: float, fail: float) -> Verdict:
    if p < fail:
        return Verdict.FAIL
    return Verdict.CAUTION if p < caution else Verdict.PASS


def _joined(parts: Sequence[str]) -> str:
    """``a``, ``a and b`` or ``a, b and c``."""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _listing(items: Sequence[str], noun: str) -> str:
    """``row 13``, ``rows 13 and 41`` or ``rows 1, 2 and 3``."""
    return f"{noun} {items[0]}" if len(items) == 1 else f"{noun}s {_joined(items)}"


def _subjects(count: int, noun: str, one: str, many: str) -> str:
    """``1 point has`` or ``3 points have``: the noun with its verb for ``count``."""
    return f"1 {noun} {one}" if count == 1 else f"{count} {noun}s {many}"


def _named(names: Sequence[str], one: str, many: str) -> str:
    """``Parameter a sits`` or ``Parameters a and b sit``."""
    return f"{_listing(names, 'Parameter')} {one if len(names) == 1 else many}"


def _needs_look(count: int) -> str:
    """``1 needs a look`` or ``3 need a look``."""
    return "1 needs a look" if count == 1 else f"{count} need a look"


def _test(summary: FitReport, name: str) -> DiagnosticTest | None:
    """The diagnostic test the core calls ``name``, when the summary holds it."""
    return next((test for test in summary.diagnostics.tests if test.name == name), None)


def _no_summary(inp: _Inputs) -> str:
    return inp.bundle.summary_reason or "the fit has no summary"


def _split_rows(data: DataSnapshot) -> tuple[tuple[ExcludedRow, ...], list[int]]:
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


def _audit_check(
    check: _Check,
    verdict: Verdict,
    title: str,
    evidence: str,
    *,
    statistic: str | None = None,
    threshold: str | None = None,
    next_step: str | None = None,
) -> AuditCheck:
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


def _skipped(check: _Check, reason: str, next_step: str | None = None) -> AuditCheck:
    return _audit_check(check, Verdict.NOT_APPLICABLE, check.untested, reason, next_step=next_step)


def _title(titles: tuple[str, str, str], verdict: Verdict) -> str:
    """The title for a verdict, from its ``(pass, caution, fail)`` forms."""
    return titles[(Verdict.PASS, Verdict.CAUTION, Verdict.FAIL).index(verdict)]


def _is_exact(inp: _Inputs) -> bool:
    """Whether the residuals are zero to machine precision, as the core's diagnostics judge it."""
    stats = inp.result.stats
    if stats is None:
        return False
    if stats.rmse == 0.0:
        return True
    y = inp.bundle.data.y
    if not y:
        return False
    scale = max(max(y) - min(y), _OFFSET_FLOOR_RATIO * max(abs(value) for value in y))
    return scale > 0.0 and stats.rmse <= _EXACT_FIT_RATIO * scale


def _capped(numbers: Sequence[str], noun: str) -> str:
    """``rows 7 and 13``, or the first ``_LISTED_MAX`` of a longer list and how many follow."""
    if len(numbers) <= _LISTED_MAX:
        return _listing(numbers, noun)
    return f"{_listing(numbers[:_LISTED_MAX], noun)}, and {len(numbers) - _LISTED_MAX} more"


def _where(inp: _Inputs, positions: Sequence[int], points: int) -> str:
    """Where flagged points are: rows of the data when the data are the points, else positions."""
    if len(inp.active_rows) == points:
        rows = [str(inp.active_rows[position] + 1) for position in positions]
        return _capped(rows, "row")
    counted = [str(position + 1) for position in positions]
    return (
        f"{_capped(counted, 'point')}, counted from 1 among the points the fit used, in data order"
    )


def _read_test(
    inp: _Inputs, check: _Check, name: str, label: str
) -> AuditCheck | tuple[DiagnosticTest, float]:
    """The core's test ``name`` with its p-value, or the check that says why it cannot be read."""
    if inp.summary is None:
        return _skipped(check, _no_summary(inp))
    test = _test(inp.summary, name)
    if test is None:
        return _skipped(check, f"the {label} test is not in the summary")
    if test.status == "not_applicable":
        return _skipped(check, test.reason or f"the {label} test did not apply")
    if test.p_value is None:
        return _skipped(check, f"the {label} test returned no p-value")
    return test, test.p_value


@dataclass(frozen=True)
class _Trend:
    """The residual-trend F test: adding x and x² to an intercept-only fit of the residuals."""

    statistic: float
    p_value: float
    df_error: int


def _trend_test(x: FloatArray, weighted: FloatArray) -> _Trend | None:
    """The F test of a quadratic trend in the residuals, with x scaled to [0, 1].

    ``None`` when the denominator is zero: a quadratic in x then accounts for the whole
    spread of the residuals and the statistic is undefined.
    """
    count = int(x.size)
    scaled = (x - float(x.min())) / float(x.max() - x.min())
    design = np.column_stack([np.ones(count), scaled, scaled * scaled])
    coefficients = np.linalg.lstsq(design, weighted, rcond=None)[0]
    left = weighted - design @ coefficients
    sse_quadratic = float(left @ left)
    centred = weighted - float(weighted.mean())
    sse_intercept = float(centred @ centred)
    if sse_quadratic <= 0.0:
        return None
    df_error = count - 3
    statistic = max(((sse_intercept - sse_quadratic) / 2.0) / (sse_quadratic / df_error), 0.0)
    return _Trend(
        statistic=statistic,
        p_value=float(scipy_stats.f.sf(statistic, 2, df_error)),
        df_error=df_error,
    )


def _residual_trend(summary: FitReport) -> _Trend | str:
    """The trend test on the summary's residuals, or the reason it cannot run."""
    x = np.asarray(summary.residuals.x, dtype=np.float64)
    weighted = np.asarray(summary.residuals.weighted_residuals, dtype=np.float64)
    count = int(x.size)
    if count < _TREND_MIN_POINTS:
        return f"only {count} points; the test needs at least {_TREND_MIN_POINTS}"
    if not (bool(np.all(np.isfinite(x))) and bool(np.all(np.isfinite(weighted)))):
        return "the residuals are not all finite"
    if float(x.max()) == float(x.min()):
        return "all x values are equal, so the residuals have no order in x"
    trend = _trend_test(x, weighted)
    if trend is None:
        return (
            "a quadratic in x accounts for the whole spread of the residuals, so the F "
            "statistic is undefined"
        )
    return trend


def _effective_bounds(inp: _Inputs) -> tuple[dict[str, tuple[float, float]], bool]:
    """The bounds each parameter was fitted within, and whether they are known at all.

    The spec's bounds win. For a registered model the rest are its declared ones. Without a
    spec nothing is known for a custom equation or an id the registry does not have; a
    polynomial has none by nature.
    """
    bounds: dict[str, tuple[float, float]] = {}
    polynomial = _POLYNOMIAL_ID.fullmatch(inp.result.model_id) is not None
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


def _charged(row: RankedFit) -> int:
    """The parameter count the ranking charges: fitted parameters plus shape parameters."""
    return row.metrics.k + row.shape_parameters


def _simplest(ranking: AutoFitResult | None, model_id: str) -> tuple[RankedFit | None, str | None]:
    """The ranked model with the fewest parameters among those within ``SIMPLE_DELTA_AICC`` of
    the best that are neither degenerate nor redundant, or the reason there is none.

    Ties go to the lower cross-validation error (a missing one counts as worse), then to the
    lower rank.
    """
    if ranking is None or not any(row.model_id == model_id for row in ranking.ranked):
        return None, "no Auto-fit ranking for this dataset"
    usable = [
        row
        for row in ranking.ranked
        if row.delta_aicc is not None
        and row.delta_aicc <= SIMPLE_DELTA_AICC
        and not row.degenerate
        and row.redundant_with is None
    ]
    if not usable:
        return None, (
            f"every model within {_num(SIMPLE_DELTA_AICC)} AICc of the best is degenerate "
            "or redundant"
        )
    fewest = min(
        usable,
        key=lambda row: (
            _charged(row),
            row.metrics.cv_rmse is None,
            row.metrics.cv_rmse or 0.0,
            row.rank,
        ),
    )
    return fewest, None


# -- the checks, in the order the audit reports them --------------------------------------------


def _check_converged(inp: _Inputs) -> AuditCheck:
    result = inp.result
    status = result.status.value
    if result.termination_reason:
        evidence = f"Status {status}: {result.termination_reason}"
    else:
        evidence = f"Status {status}, with no reason given"
    if result.status is FitStatus.CONVERGED:
        return _audit_check(
            _CONVERGED, Verdict.PASS, "The fit converged", evidence, threshold="status converged"
        )
    return _audit_check(
        _CONVERGED,
        Verdict.FAIL,
        "The fit did not converge",
        evidence,
        threshold="status converged",
        next_step=_RESTART,
    )


def _check_perfect_fit(inp: _Inputs) -> AuditCheck:
    stats = inp.result.stats
    if stats is None:
        return _skipped(_PERFECT_FIT, _NO_STATISTICS)
    exact = _is_exact(inp)
    if exact:
        verdict, title = Verdict.CAUTION, "The model passes through every point exactly"
        y = inp.bundle.data.y
        evidence = (
            f"Sum of squared residuals {_num(stats.sse)}, RMSE {_num(stats.rmse)} against a "
            f"data range of {_num(max(y) - min(y))}: the residuals are zero to machine "
            "precision."
        )
    else:
        verdict, title = Verdict.PASS, "The model does not pass through every point"
        evidence = f"Sum of squared residuals {_num(stats.sse)} (RMSE {_num(stats.rmse)})."
    return _audit_check(
        _PERFECT_FIT,
        verdict,
        title,
        evidence,
        statistic=f"SSE = {_num(stats.sse)}",
        threshold="SSE = 0 (to machine precision)",
        next_step=_EXACT_NEXT_STEP if exact else None,
    )


def _check_residual_normality(inp: _Inputs) -> AuditCheck:
    found = _read_test(inp, _NORMALITY, "shapiro_wilk", "Shapiro-Wilk")
    if isinstance(found, AuditCheck):
        return found
    test, p = found
    verdict = _p_verdict(p, NORMALITY_CAUTION_P, NORMALITY_FAIL_P)
    evidence = f"Shapiro-Wilk p = {_num(p)}"
    if inp.n < SMALL_SAMPLE_N:
        evidence += "; with fewer than 20 points this test rarely detects anything"
    statistic = f"p = {_num(p)}"
    if test.statistic is not None:
        statistic = f"W = {_num(test.statistic)}, {statistic}"
    return _audit_check(
        _NORMALITY,
        verdict,
        _title(_NORMALITY_TITLES, verdict),
        evidence,
        statistic=statistic,
        threshold=_p_rule(NORMALITY_CAUTION_P, NORMALITY_FAIL_P),
        next_step=None if verdict is Verdict.PASS else _NORMALITY_NEXT_STEP,
    )


def _check_residual_independence(inp: _Inputs) -> AuditCheck:
    found = _read_test(inp, _INDEPENDENCE, "ljung_box", "Ljung-Box")
    if isinstance(found, AuditCheck):
        return found
    test, p = found
    verdict = _p_verdict(p, AUTOCORRELATION_CAUTION_P, DW_FAIL_P)
    evidence = f"Ljung-Box p = {_num(p)}, residuals taken in ascending x."
    durbin_watson = None if inp.summary is None else _test(inp.summary, "durbin_watson")
    if durbin_watson is None:
        evidence += " The Durbin-Watson statistic is not in the summary."
    elif durbin_watson.statistic is not None:
        evidence += (
            f" Durbin-Watson statistic {_num(durbin_watson.statistic)}, which is reported "
            "here without a verdict of its own."
        )
    else:
        why = durbin_watson.reason or "the test did not apply"
        evidence += f" The Durbin-Watson statistic is not available ({why})."
    statistic = f"p = {_num(p)}"
    if test.statistic is not None:
        lags = "" if test.df is None else f"({_num(test.df)})"
        statistic = f"Q{lags} = {_num(test.statistic)}, {statistic}"
    return _audit_check(
        _INDEPENDENCE,
        verdict,
        _title(_INDEPENDENCE_TITLES, verdict),
        evidence,
        statistic=statistic,
        threshold=_p_rule(AUTOCORRELATION_CAUTION_P, DW_FAIL_P),
        next_step=None if verdict is Verdict.PASS else _INDEPENDENCE_NEXT_STEP,
    )


def _check_constant_variance(inp: _Inputs) -> AuditCheck:
    found = _read_test(inp, _CONSTANT_VARIANCE, "breusch_pagan", "Breusch-Pagan")
    if isinstance(found, AuditCheck):
        return found
    test, p = found
    verdict = _p_verdict(p, VARIANCE_CAUTION_P, VARIANCE_FAIL_P)
    statistic = f"p = {_num(p)}"
    if test.statistic is not None:
        statistic = f"BP = {_num(test.statistic)}, {statistic}"
    return _audit_check(
        _CONSTANT_VARIANCE,
        verdict,
        _title(_CONSTANT_VARIANCE_TITLES, verdict),
        f"Breusch-Pagan p = {_num(p)}",
        statistic=statistic,
        threshold=_p_rule(VARIANCE_CAUTION_P, VARIANCE_FAIL_P),
        next_step=None if verdict is Verdict.PASS else _VARIANCE_NEXT_STEP,
    )


def _check_influential_points(inp: _Inputs) -> AuditCheck:
    if inp.summary is None:
        return _skipped(_INFLUENCE, _no_summary(inp))
    if _is_exact(inp):
        return _skipped(_INFLUENCE, "the fit is exact, so no point stands out from the others")
    residuals = inp.summary.residuals
    influential = list(residuals.flagged_high_influence)
    count = len(influential)
    if count >= INFLUENCE_FAIL_COUNT or count >= INFLUENCE_FAIL_SHARE * inp.n:
        verdict = Verdict.FAIL
    elif count >= INFLUENCE_CAUTION_COUNT:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.PASS
    points = len(residuals.x)
    cooks = _num(residuals.cooks_threshold)
    leverage = _num(residuals.leverage_threshold)
    who = _subjects(count, "point", "has", "have")
    cook_cut = f"the threshold 4/n = {cooks}"
    if count:
        sentences = [
            f"{who} a Cook's distance above {cook_cut}: {_where(inp, influential, points)}."
        ]
    else:
        sentences = [f"No point has a Cook's distance above {cook_cut}."]
    flagged = set(influential)
    leverage_only = [i for i in residuals.flagged_high_leverage if i not in flagged]
    if leverage_only:
        sentences.append(
            f"{_subjects(len(leverage_only), 'point', 'has', 'have')} high leverage but "
            f"little influence (leverage above the threshold 2k/n = {leverage}): "
            f"{_where(inp, leverage_only, points)}."
        )
        if len(leverage_only) >= LEVERAGE_CAUTION_SHARE * inp.n:
            sentences.append(
                f"That is at least {_num(LEVERAGE_CAUTION_SHARE * 100)} % of the points."
            )
    threshold = (
        f"Cook's distance > 4/n = {cooks}: {INFLUENCE_CAUTION_COUNT} point is a caution, "
        f"{INFLUENCE_FAIL_COUNT} points or {_num(INFLUENCE_FAIL_SHARE * 100)} % of the "
        f"points a failure; leverage > 2k/n = {leverage} is reported only"
    )
    statistic = f"{count} of {points} points with a Cook's distance above {cooks}"
    next_step = None
    if verdict is Verdict.PASS:
        title = "No point has a large influence on the fit"
    else:
        title = f"{who} a large influence on the fit"
        them = "it" if count == 1 else "them"
        next_step = (
            f"Look at {_where(inp, influential, points)}. Refit without {them} to see what "
            "changes. Do not delete data to improve a fit."
        )
    return _audit_check(
        _INFLUENCE,
        verdict,
        title,
        " ".join(sentences),
        statistic=statistic,
        threshold=threshold,
        next_step=next_step,
    )


def _check_residual_trend(inp: _Inputs) -> AuditCheck:
    if inp.summary is None:
        return _skipped(_TREND, _no_summary(inp))
    if _is_exact(inp):
        return _skipped(_TREND, "the fit is exact, so its residuals carry no trend")
    trend = _residual_trend(inp.summary)
    if not isinstance(trend, _Trend):
        return _skipped(_TREND, trend)
    verdict = _p_verdict(trend.p_value, TREND_CAUTION_P, TREND_FAIL_P)
    return _audit_check(
        _TREND,
        verdict,
        _title(_TREND_TITLES, verdict),
        "Residuals regressed on 1, x and x² over "
        f"{trend.df_error + 3} points, x scaled to [0, 1]: p = {_num(trend.p_value)}.",
        statistic=f"F(2, {trend.df_error}) = {_num(trend.statistic)}, p = {_num(trend.p_value)}",
        threshold=_p_rule(TREND_CAUTION_P, TREND_FAIL_P),
        next_step=None if verdict is Verdict.PASS else _TREND_NEXT_STEP,
    )


def _parameter_finding(
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
        and abs(parameter.value - bound) <= _BOUND_RTOL * max(1.0, abs(bound))
    ]
    if near:
        return (
            "bound",
            f"{name} sits at a bound: estimate {_num(parameter.value)} against the bound "
            f"{_num(near[0])}.",
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
                f"{name} has an interval from {_num(ci_low)} to {_num(ci_high)} covering "
                f"{_num(covered * 100)} % of its allowed range {_num(lower)} to "
                f"{_num(upper)}.",
            )
    if unbounded and parameter.std_err is not None and parameter.std_err > abs(parameter.value):
        return (
            "weak",
            f"{name} has a standard error of {_num(parameter.std_err)}, larger than its "
            f"value of {_num(parameter.value)}.",
        )
    if parameter.std_err is None:
        return "weak", f"{name}: its standard error could not be determined."
    return "fine", ""


def _check_parameters_identifiable(inp: _Inputs) -> AuditCheck:
    result = inp.result
    bounds, known = _effective_bounds(inp)
    reported = {
        warning.removeprefix(_AT_BOUND).strip()
        for warning in result.warnings
        if warning.startswith(_AT_BOUND)
    }
    at_bound: list[str] = []
    concerned: list[str] = []
    findings: list[str] = []
    for parameter in result.parameters:
        standing, finding = _parameter_finding(
            parameter, bounds.get(parameter.name, _UNBOUNDED), parameter.name in reported
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
        f"within {_num(_BOUND_RTOL)} x max(1, |bound|) of a bound: failure; interval "
        f"covering >= {_num(WIDE_INTERVAL_SHARE * 100)} % of the allowed range, or standard "
        "error > |value|: caution"
    )
    if at_bound:
        verdict, title = Verdict.FAIL, f"{_named(at_bound, 'sits', 'sit')} at a bound"
        next_step = _IDENTIFIABLE_NEXT_STEP
    elif concerned or no_covariance:
        verdict = Verdict.CAUTION
        title = (
            f"{_named(concerned, 'is', 'are')} not pinned down by these data"
            if concerned
            else "Parameter uncertainty has no covariance matrix behind it"
        )
        next_step = _IDENTIFIABLE_NEXT_STEP
    else:
        evidence = "Every parameter has a standard error and none sits at a bound."
        findings.insert(0, evidence)
        verdict, title, next_step = (
            Verdict.PASS,
            "No parameter sits at a bound or is poorly determined",
            None,
        )
    return _audit_check(
        _IDENTIFIABLE,
        verdict,
        title,
        " ".join(findings),
        statistic=statistic,
        threshold=threshold,
        next_step=next_step,
    )


def _check_points_per_parameter(inp: _Inputs) -> AuditCheck:
    n, k = inp.n, inp.k
    if k == 0:
        return _skipped(_POINTS, "the fit has no parameters to count points for")
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
    return _audit_check(
        _POINTS,
        verdict,
        title,
        f"{statistic}.",
        statistic=statistic,
        threshold=threshold,
        next_step=None if verdict is Verdict.PASS else _POINTS_NEXT_STEP,
    )


def _check_cross_validation(inp: _Inputs) -> AuditCheck:
    stats = inp.result.stats
    if stats is None:
        return _skipped(_CROSS_VALIDATION, _NO_STATISTICS)
    cv = inp.cv
    if cv is None:
        return _skipped(
            _CROSS_VALIDATION,
            "cross-validation has not been run",
            "Run cross-validation to check this.",
        )
    if cv.rmse is None:
        return _skipped(
            _CROSS_VALIDATION,
            f"cross-validation is incomplete: {cv.n_failed} of {cv.folds} folds failed, so "
            "there is no held-out error to compare",
            "Run cross-validation with fewer folds, or on more points, so that every fold "
            "can be fitted.",
        )
    if cv.rmse == stats.rmse:
        ratio = 1.0
    elif stats.rmse == 0.0:
        ratio = math.inf
    else:
        ratio = cv.rmse / stats.rmse
    if ratio <= CV_CAUTION_RATIO:
        verdict = Verdict.PASS
    elif ratio <= CV_FAIL_RATIO:
        verdict = Verdict.CAUTION
    else:
        verdict = Verdict.FAIL
    compared = f"In-sample RMSE {_num(stats.rmse)}, cross-validation RMSE {_num(cv.rmse)}"
    if math.isfinite(ratio):
        evidence = f"{compared}: a ratio of {_num(ratio)}."
        statistic = f"CV RMSE / RMSE = {_num(ratio)}"
    else:
        evidence = f"{compared}, against an in-sample error of zero."
        statistic = "CV RMSE / RMSE is unbounded: the in-sample RMSE is 0"
    return _audit_check(
        _CROSS_VALIDATION,
        verdict,
        _title(_CROSS_VALIDATION_TITLES, verdict),
        evidence,
        statistic=statistic,
        threshold=(
            f"ratio <= {_num(CV_CAUTION_RATIO)} pass; <= {_num(CV_FAIL_RATIO)} caution; "
            f"above {_num(CV_FAIL_RATIO)} fail"
        ),
        next_step=None if verdict is Verdict.PASS else _CV_NEXT_STEP,
    )


def _check_simpler_model(inp: _Inputs) -> AuditCheck:
    fewest = inp.simplest
    if fewest is None or inp.ranking is None:
        return _skipped(
            _SIMPLER_MODEL, inp.simplest_reason or "no Auto-fit ranking for this dataset"
        )
    audited = next(row for row in inp.ranking.ranked if row.model_id == inp.result.model_id)
    here, there = _charged(audited), _charged(fewest)
    delta = fewest.delta_aicc if fewest.delta_aicc is not None else 0.0
    within = f"within {_num(SIMPLE_DELTA_AICC)} AICc of the best"
    statistic = f"{here} parameters here; the simplest {within} has {there} ({fewest.model_id})"
    threshold = f"delta AICc <= {_num(SIMPLE_DELTA_AICC)}"
    if fewest.model_id == inp.result.model_id:
        verdict, title = Verdict.PASS, "This is the simplest model within 2 AICc of the best"
        evidence = (
            f"This is the simplest model {within}: {here} parameters, "
            f"{_num(delta)} AICc from the best."
        )
    elif there > here:
        verdict, title = Verdict.PASS, "No simpler model fits as well"
        evidence = (
            "No model within 2 AICc of the best that is not degenerate or redundant has "
            f"fewer parameters than this one; {fewest.name} ({fewest.model_id}) has {there}."
        )
    elif there == here:
        verdict, title = Verdict.PASS, "An equally simple model fits as well"
        evidence = (
            f"{fewest.name} ({fewest.model_id}) has the same number of parameters ({there}) "
            f"and lies {_num(delta)} AICc from the best."
        )
    else:
        verdict, title = Verdict.CAUTION, "A simpler model fits as well"
        evidence = (
            f"{fewest.name} ({fewest.model_id}) reaches the same fit with {there} parameters "
            f"instead of {here} and lies {_num(delta)} AICc from the best."
        )
        if fewest.metrics.cv_rmse is not None:
            evidence += f" Its cross-validation RMSE is {_num(fewest.metrics.cv_rmse)}."
    return _audit_check(
        _SIMPLER_MODEL,
        verdict,
        title,
        evidence,
        statistic=statistic,
        threshold=threshold,
        next_step=_SIMPLER_NEXT_STEP if verdict is Verdict.CAUTION else None,
    )


_NORMALITY_TITLES = (
    "Residuals are consistent with a normal distribution",
    "Residuals may depart from a normal distribution",
    "Residuals depart from a normal distribution",
)
_INDEPENDENCE_TITLES = (
    "Residuals show no autocorrelation in x order",
    "Residuals may be autocorrelated in x order",
    "Residuals are autocorrelated in x order",
)
_CONSTANT_VARIANCE_TITLES = (
    "Residual variance is consistent with being constant",
    "Residual variance may change with the fitted value",
    "Residual variance changes with the fitted value",
)
_TREND_TITLES = (
    "Residuals show no trend in x",
    "Residuals may show a trend in x",
    "Residuals show a trend in x",
)
_CROSS_VALIDATION_TITLES = (
    "Held-out error is close to the in-sample error",
    "Held-out error is somewhat above the in-sample error",
    "Held-out error is far above the in-sample error",
)

_CONVERGED = _Check(
    "converged",
    "Convergence of the fit was not assessed",
    "A fit that stopped before it converged has no usable estimates, so nothing else can be "
    "judged until it does.",
    _check_converged,
)
_PERFECT_FIT = _Check(
    "perfect_fit",
    "Whether the fit is exact was not assessed",
    "A model with as many parameters as points passes through every point exactly, and that "
    "says nothing about the data. An exact fit on more points leaves no scatter for the "
    "residual tests.",
    _check_perfect_fit,
)
_NORMALITY = _Check(
    "residual_normality",
    "Normality of the residuals was not tested",
    "Intervals and p-values assume the residuals are roughly normal. The Shapiro-Wilk test "
    "looks at the shape of their distribution, not at whether the model is right.",
    _check_residual_normality,
)
_INDEPENDENCE = _Check(
    "residual_independence",
    "Independence of the residuals was not tested",
    "Points taken in sequence often carry information from their neighbours. The Ljung-Box "
    "test asks whether residuals close together in x are also alike.",
    _check_residual_independence,
)
_CONSTANT_VARIANCE = _Check(
    "constant_variance",
    "Constancy of the residual variance was not tested",
    "Intervals and tests assume the residuals scatter equally at every fitted value. The "
    "Breusch-Pagan test asks whether their spread grows with the fitted value.",
    _check_constant_variance,
)
_INFLUENCE = _Check(
    "influential_points",
    "Influence of single points was not assessed",
    "One point can pull a fitted curve towards itself. Cook's distance measures how much the "
    "fit would move if the point were left out; leverage, how far a point's x sits from the "
    "others.",
    _check_influential_points,
)
_TREND = _Check(
    "residual_trend",
    "A trend of the residuals in x was not tested",
    "A model of the right shape leaves residuals scattered evenly about zero at every x. A "
    "curve left in the residuals is a shape the data contain and the model lacks.",
    _check_residual_trend,
)
_IDENTIFIABLE = _Check(
    "parameters_identifiable",
    "The parameters were not checked",
    "A parameter the data do not constrain can take many values with little change in the "
    "fit, so its estimate and interval mean little.",
    _check_parameters_identifiable,
)
_POINTS = _Check(
    "points_per_parameter",
    "Points per parameter were not counted",
    "Each parameter needs several points to be estimated with a usable interval. With few "
    "points per parameter the fit follows noise and the intervals widen.",
    _check_points_per_parameter,
)
_CROSS_VALIDATION = _Check(
    "cross_validation",
    "Predictive error was not assessed",
    "Cross-validation refits the model on subsets of the data and scores the points left "
    "out. A held-out error well above the fitted error means the model follows noise.",
    _check_cross_validation,
)
_SIMPLER_MODEL = _Check(
    "simpler_model",
    "Simpler models were not compared",
    "Of the models that fit about equally well, the one with the fewest parameters is usually "
    "the better description. A difference of 2 AICc or less is usually taken as no "
    "difference.",
    _check_simpler_model,
)

_CHECKS = (
    _CONVERGED,
    _PERFECT_FIT,
    _NORMALITY,
    _INDEPENDENCE,
    _CONSTANT_VARIANCE,
    _INFLUENCE,
    _TREND,
    _IDENTIFIABLE,
    _POINTS,
    _CROSS_VALIDATION,
    _SIMPLER_MODEL,
)
"""Every check, in the order the audit reports it."""


# -- the audit as a whole -----------------------------------------------------------------------


def _run_checks(inp: _Inputs) -> tuple[AuditCheck, ...]:
    """Every check in order. Once the fit has not converged, the rest only say so."""
    converged = inp.result.status is FitStatus.CONVERGED
    return tuple(
        check.run(inp) if converged or check is _CONVERGED else _skipped(check, _NOT_CONVERGED)
        for check in _CHECKS
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
        look = f", {_needs_look(len(cautioned))}" if cautioned else ""
        return f"{count}{look}. {failed[0].title}. {failed[0].next_step}"
    if cautioned:
        first = cautioned[0]
        return f"No check failed; {_needs_look(len(cautioned))}. {first.title}. {first.next_step}"
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


def _residual_p(inp: _Inputs, check_id: str) -> float | None:
    """The p-value a residual check turned on, read again from where the check read it."""
    summary = inp.summary
    if summary is None:
        return None
    if check_id == "residual_trend":
        trend = _residual_trend(summary)
        return trend.p_value if isinstance(trend, _Trend) else None
    test = _test(summary, _TEST_OF[check_id])
    return None if test is None else test.p_value


def _methods_fragment(inp: _Inputs, checks: Mapping[str, AuditCheck]) -> str:
    """One or two sentences a methods paragraph can carry as it stands."""
    if inp.result.status is not FitStatus.CONVERGED:
        return "The fit did not converge, so no residual checks apply."
    clauses: dict[str, str] = {}
    for check_id, forms in _CLAUSES.items():
        verdict = checks[check_id].verdict
        p = _residual_p(inp, check_id)
        if verdict is not Verdict.NOT_APPLICABLE and p is not None:
            clauses[check_id] = forms[verdict].format(p=_num(p))
    shape = [
        clauses[check_id]
        for check_id in ("residual_trend", "constant_variance", "residual_independence")
        if check_id in clauses
    ]
    normal = clauses.get("residual_normality")
    if shape and normal:
        sentence = f"Residuals showed {_joined(shape)}; {normal}."
    elif shape:
        sentence = f"Residuals showed {_joined(shape)}."
    elif normal:
        sentence = f"Only the normality test could be run: {normal}."
    elif checks[_PERFECT_FIT.id].verdict is Verdict.CAUTION:
        sentence = (
            "The model passes through every point exactly, so the residual tests carry no "
            "information."
        )
    else:
        sentence = "No residual test could be run on this fit."
    if checks[_CROSS_VALIDATION.id].verdict is Verdict.NOT_APPLICABLE:
        ran = "did not complete" if inp.cv is not None else "was not run"
        sentence += f" Cross-validation {ran}."
    return sentence


def _simpler_model(row: RankedFit) -> SimplerModel:
    return SimplerModel(
        model_id=row.model_id,
        name=row.name,
        n_parameters=_charged(row),
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
    excluded, active_rows = _split_rows(data)
    stats = result.stats
    n = stats.n if stats is not None else len(result.fitted) - len(excluded)
    simplest, simplest_reason = _simplest(ranking, result.model_id)
    inputs = _Inputs(
        bundle=bundle,
        result=result,
        spec=spec,
        ranking=ranking,
        cv=cv,
        n=n,
        k=len(result.parameters),
        summary=bundle.summary,
        active_rows=active_rows,
        simplest=simplest,
        simplest_reason=simplest_reason,
    )
    checks = _run_checks(inputs)
    by_id = {check.id: check for check in checks}
    simpler = (
        _simpler_model(simplest)
        if simplest is not None and by_id[_SIMPLER_MODEL.id].verdict is Verdict.CAUTION
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
