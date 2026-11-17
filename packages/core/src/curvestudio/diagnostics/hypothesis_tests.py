"""Statistical tests on the residuals of a fit: normality, heteroscedasticity, autocorrelation.

Every test is reported as a :class:`DiagnosticTest` that names the null hypothesis it
addresses, the assumptions it rests on and what it cannot show.  A test is evidence about
one assumption only: a p-value above the significance level never shows that the
assumption holds, and no test result certifies the fit.

The routines work on a *weighted* residual vector ``r = sqrt(w) * (y - fitted)``.  The
order of that vector matters for the autocorrelation tests, so the caller decides it:
:func:`run_diagnostics` sorts by ascending ``x`` (ties broken by the residual value)
before calling them.

Data that cannot support a test is never an exception: constant residuals, non-finite
values or a sample that is too small give ``status="not_applicable"`` with a ``reason``.
Only a programming error - residuals and fitted values of different lengths - is raised.
"""

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import special as scipy_special
from scipy import stats as scipy_stats

from curvestudio.fitting import FitStatus
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.statistics import FitView, StatisticsError

FloatArray = NDArray[np.float64]

NOTE = "A test result is evidence about one assumption; it never certifies the model as valid."
"""The caveat every result carries: a test informs, it does not certify."""

_SHAPIRO_MIN_N = 3
"""The Shapiro-Wilk test needs at least this many residuals."""

_ANDERSON_MIN_N = 8
"""The Anderson-Darling test needs at least this many residuals."""

_SHAPIRO_LARGE_N = 5000
"""Above this many residuals SciPy's Shapiro-Wilk p-value is an approximation."""

_EXACT_FIT_RATIO = 1e-10
"""The fit counts as exact when the residuals are this small relative to the data scale."""

_OFFSET_FLOOR_RATIO = 1e-4
"""Rounding in a large ``y`` offset reaches this fraction of ``max|y|``; it floors the scale."""

_CONSTANT_FITTED_RATIO = 1e-12
"""Fitted values spanning less than this fraction of their magnitude count as constant."""

_NORMALITY_NULL = (
    "The residuals are drawn from a normal distribution (mean and variance estimated)."
)

_METADATA: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "shapiro_wilk": (
        _NORMALITY_NULL,
        (
            "The observations are independent of each other.",
            "The residuals come from a single continuous population.",
            "The sample is a random sample of that population.",
        ),
        (
            "The test looks at the shape of the residual distribution only; it says nothing "
            "about independence.",
            "It reacts to departures from normality in the tails and has little power in "
            "small samples.",
            "A p-value above the significance level does not show that the assumption holds.",
        ),
    ),
    "anderson_darling": (
        _NORMALITY_NULL,
        (
            "The observations are independent of each other.",
            "The residuals come from a single continuous population.",
            "The sample is a random sample of that population.",
        ),
        (
            "The statistic weights the tails more heavily than the Shapiro-Wilk statistic, "
            "so it reacts sooner to a heavy tail.",
            "The p-value uses the Stephens (1974) asymptotic approximation and is only a "
            "rough guide in small samples.",
            "With a large sample the p-value can be small even when the departure from "
            "normality is of no practical size.",
        ),
    ),
    "breusch_pagan": (
        "The residual variance does not depend on the fitted values (the residuals are "
        "homoscedastic).",
        (
            "The errors are independent of the regressors (exogeneity).",
            "The regressors of the auxiliary regression are not perfectly collinear.",
            "The Koenker form does not require the errors to be normally distributed.",
        ),
        (
            "Only a variance that grows linearly with the fitted values is detected; other "
            "patterns of heteroscedasticity pass unnoticed.",
            "A misspecified mean also gives a significant result, so the test does not "
            "isolate the variance.",
            "With a large sample small departures are significant, so the size of the "
            "statistic matters as much as its p-value.",
        ),
    ),
    "ljung_box": (
        "The residuals are serially independent up to the tested lag (no autocorrelation).",
        (
            "The residuals are serially uncorrelated up to the tested lag under the null.",
            "The observations are equally spaced in the ordering used.",
            "The model is correctly specified; a misspecified mean leaves autocorrelation "
            "in the residuals.",
        ),
        (
            "The result depends on the lag count; a large lag spends degrees of freedom on "
            "high-order structure.",
            "The chi-square distribution is an asymptotic approximation.",
            "The ordering is the ascending-x order, so the test assumes x is a sensible "
            "time-like order.",
        ),
    ),
    "durbin_watson": (
        "The residuals are serially uncorrelated (no first-order autocorrelation).",
        (
            "The residuals are in the order in which they were observed (ascending x here).",
            "The errors are independent apart from any first-order dependence.",
        ),
        (
            "Only a statistic is produced: the Durbin-Watson statistic has no exact p-value.",
            "Only first-order autocorrelation is detected.",
            "The statistic depends on the order of the observations.",
        ),
    ),
}

_TIED_X_LIMITATION = (
    "The data contain duplicated x values, so the ascending-x order is not unique; ties are "
    "broken by the residual value, which is deterministic but arbitrary, so the result "
    "depends on that choice."
)

_TEST_NAMES = (
    "shapiro_wilk",
    "anderson_darling",
    "breusch_pagan",
    "ljung_box",
    "durbin_watson",
)
"""The tests :func:`run_diagnostics` runs, in the order it reports them."""


class DiagnosticTest(BaseModel):
    """One statistical test applied to the residuals of a fit.

    ``statistic``, ``p_value`` and ``df`` are ``None`` when the test does not produce them:
    a test that does not apply, or a statistic-only test such as the Durbin-Watson.
    ``status`` is ``"ok"`` when the test was computed and ``"not_applicable"`` when the
    data cannot support it; ``reason`` then says why.  ``reason`` may also carry a note on
    how a test was computed (a clamped lag, for instance) while ``status`` is ``"ok"``.

    ``null_hypothesis`` states what the test assumes, ``assumptions`` what it needs for
    the result to mean anything and ``limitations`` what it cannot show.  ``note`` is the
    same on every result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    statistic: FiniteFloat | None
    p_value: FiniteFloat | None
    df: FiniteFloat | None
    status: Literal["ok", "not_applicable"]
    reason: str | None
    null_hypothesis: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    note: str


class DiagnosticsReport(BaseModel):
    """The tests run on one fit, in the order they were run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tests: tuple[DiagnosticTest, ...]


def _as_vector(values: object, label: str) -> FloatArray:
    """The residuals as a one-dimensional float array.

    Raises:
        StatisticsError: ``INVALID_ARGUMENT`` when the input is not one-dimensional.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise StatisticsError(
            "INVALID_ARGUMENT",
            f"the {label} must be a one-dimensional vector, got shape {array.shape}",
        )
    return array


def _unusable(values: FloatArray) -> str | None:
    """Why the residuals cannot support a test, or ``None`` when they can."""
    if values.size == 0:
        return "there are no residuals"
    if not bool(np.all(np.isfinite(values))):
        return "the residuals contain non-finite values"
    if float(np.max(values)) == float(np.min(values)):
        return "the residuals are constant, so the test statistic is not defined"
    return None


def _metadata(name: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    null_hypothesis, assumptions, limitations = _METADATA[name]
    return null_hypothesis, assumptions, limitations


def _not_applicable(
    name: str, reason: str, extra_limitations: tuple[str, ...] = ()
) -> DiagnosticTest:
    null_hypothesis, assumptions, limitations = _metadata(name)
    return DiagnosticTest(
        name=name,
        statistic=None,
        p_value=None,
        df=None,
        status="not_applicable",
        reason=reason,
        null_hypothesis=null_hypothesis,
        assumptions=assumptions,
        limitations=(*limitations, *extra_limitations),
        note=NOTE,
    )


def _ok(
    name: str,
    *,
    statistic: float,
    p_value: float | None,
    df: float | None,
    reason: str | None = None,
    extra_limitations: tuple[str, ...] = (),
) -> DiagnosticTest:
    null_hypothesis, assumptions, limitations = _metadata(name)
    return DiagnosticTest(
        name=name,
        statistic=statistic,
        p_value=p_value,
        df=df,
        status="ok",
        reason=reason,
        null_hypothesis=null_hypothesis,
        assumptions=assumptions,
        limitations=(*limitations, *extra_limitations),
        note=NOTE,
    )


def shapiro_wilk(residuals: FloatArray) -> DiagnosticTest:
    """Shapiro-Wilk test for normality of the residuals.

    Needs at least 3 residuals.  Above 5000 residuals SciPy's p-value is an
    approximation, which the result then states as a limitation.
    """
    values = _as_vector(residuals, "residuals")
    unusable = _unusable(values)
    if unusable is not None:
        return _not_applicable("shapiro_wilk", unusable)
    n = int(values.size)
    if n < _SHAPIRO_MIN_N:
        return _not_applicable(
            "shapiro_wilk",
            f"the Shapiro-Wilk test needs at least {_SHAPIRO_MIN_N} residuals, got {n}",
        )
    with warnings.catch_warnings():
        # SciPy warns for n > 5000; the same caveat is stated in the limitations.
        warnings.simplefilter("ignore", UserWarning)
        result = scipy_stats.shapiro(values)
    extra = (
        ("Above 5000 residuals the SciPy p-value is an approximation and may be inaccurate.",)
        if n > _SHAPIRO_LARGE_N
        else ()
    )
    return _ok(
        "shapiro_wilk",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        df=None,
        extra_limitations=extra,
    )


def _anderson_darling_statistic(values: FloatArray) -> tuple[float, float]:
    """The small-sample adjusted ``A*^2`` and its Stephens (1974) p-value.

    The logarithms of the normal CDF are taken with ``log_ndtr`` so that a very extreme
    residual gives a large finite statistic instead of an underflowed quantile.  The
    caller guarantees at least 8 non-constant residuals, so the spread is positive.
    """
    n = int(values.size)
    ordered = np.sort(values)
    z = (ordered - float(np.mean(ordered))) / float(np.std(ordered, ddof=1))
    index = np.arange(1, n + 1, dtype=np.float64)
    total = float(
        np.sum(
            (2.0 * index - 1.0) / n * (scipy_special.log_ndtr(z) + scipy_special.log_ndtr(-z)[::-1])
        )
    )
    a2 = -n - total
    adjusted = a2 * (1.0 + 0.75 / n + 2.25 / n**2)
    return adjusted, _anderson_darling_pvalue(adjusted)


def _anderson_darling_pvalue(adjusted: float) -> float:
    """The Stephens (1974) piecewise p-value of the adjusted Anderson-Darling statistic."""
    if 0.0 <= adjusted < 0.200:
        return float(1.0 - np.exp(-13.436 + 101.14 * adjusted - 223.73 * adjusted**2))
    if adjusted < 0.340:
        return float(1.0 - np.exp(-8.318 + 42.796 * adjusted - 59.938 * adjusted**2))
    if adjusted < 0.600:
        return float(np.exp(0.9177 - 4.279 * adjusted - 1.38 * adjusted**2))
    if adjusted <= 13.0:
        return float(np.exp(1.2937 - 5.709 * adjusted + 0.0186 * adjusted**2))
    return 0.0


def anderson_darling(residuals: FloatArray) -> DiagnosticTest:
    """Anderson-Darling test for normality of the residuals.

    Needs at least 8 residuals.  The statistic is the small-sample adjusted ``A*^2``
    (D'Agostino and Stephens 1986); its p-value is the Stephens (1974) piecewise
    approximation, which has no degrees of freedom.
    """
    values = _as_vector(residuals, "residuals")
    unusable = _unusable(values)
    if unusable is not None:
        return _not_applicable("anderson_darling", unusable)
    n = int(values.size)
    if n < _ANDERSON_MIN_N:
        return _not_applicable(
            "anderson_darling",
            f"the Anderson-Darling test needs at least {_ANDERSON_MIN_N} residuals, got {n}",
        )
    adjusted, p_value = _anderson_darling_statistic(values)
    return _ok(
        "anderson_darling",
        statistic=adjusted,
        p_value=p_value,
        df=None,
    )


def breusch_pagan(residuals: FloatArray, fitted: FloatArray) -> DiagnosticTest:
    """Koenker studentized Breusch-Pagan test for heteroscedasticity.

    ``residuals`` is the weighted residual vector and ``fitted`` the fitted values of the
    same points.  The statistic is ``n * R**2`` of the auxiliary regression of the squared
    residuals on ``[1, fitted]``, compared with a chi-square with 1 degree of freedom.
    The Koenker form does not need the errors to be normal.

    Raises:
        StatisticsError: ``DATA_MISMATCH`` when the residuals and the fitted values have
            different lengths.
    """
    values = _as_vector(residuals, "residuals")
    fitted_values = _as_vector(fitted, "fitted values")
    if fitted_values.size != values.size:
        raise StatisticsError(
            "DATA_MISMATCH",
            f"there are {values.size} residuals but {fitted_values.size} fitted values",
        )
    unusable = _unusable(values)
    if unusable is not None:
        return _not_applicable("breusch_pagan", unusable)
    if not bool(np.all(np.isfinite(fitted_values))):
        return _not_applicable("breusch_pagan", "the fitted values contain non-finite values")
    n = int(values.size)
    if n < 3:
        return _not_applicable(
            "breusch_pagan",
            f"the Breusch-Pagan test needs at least 3 residuals, got {n}",
        )
    fitted_scale = max(1.0, float(np.max(np.abs(fitted_values))))
    if float(np.ptp(fitted_values)) <= _CONSTANT_FITTED_RATIO * fitted_scale:
        return _not_applicable(
            "breusch_pagan",
            "the fitted values are constant, so the auxiliary regression has no regressor",
        )
    squared = values**2
    design = np.column_stack([np.ones(n), fitted_values])
    solution, _residuals, rank, _singular = np.linalg.lstsq(design, squared, rcond=None)
    if int(rank) < design.shape[1]:
        return _not_applicable(
            "breusch_pagan",
            "the regressors of the auxiliary regression are collinear, so it is not defined",
        )
    leftover = squared - design @ solution
    sum_squares = float(leftover @ leftover)
    total = float(np.sum((squared - float(np.mean(squared))) ** 2))
    if total <= 0.0:
        return _not_applicable(
            "breusch_pagan",
            "the squared residuals are constant, so the auxiliary regression explains nothing",
        )
    statistic = max(0.0, n * (1.0 - sum_squares / total))
    return _ok(
        "breusch_pagan",
        statistic=statistic,
        p_value=float(scipy_stats.chi2.sf(statistic, 1)),
        df=1.0,
    )


def ljung_box(
    residuals: FloatArray, lags: int | None = None, *, tied_x: bool = False
) -> DiagnosticTest:
    """Ljung-Box test for serial correlation of the residuals.

    ``lags`` defaults to ``min(10, n // 5)`` and is clamped to ``1 <= lags <= n - 1``; a
    clamp is stated in ``reason``.  ``tied_x`` says that the ordering used has duplicated
    x values, which makes that order ambiguous.  The statistic is
    ``Q = n (n + 2) sum(rho_k**2 / (n - k))`` over ``k = 1..lags``, with the ``rho_k`` the
    autocorrelations of the demeaned residuals, and it is compared with a chi-square with
    ``lags`` degrees of freedom.
    """
    values = _as_vector(residuals, "residuals")
    unusable = _unusable(values)
    if unusable is not None:
        return _not_applicable("ljung_box", unusable)
    n = int(values.size)
    requested = min(10, n // 5) if lags is None else int(lags)
    used = max(1, min(requested, n - 1))
    reason = None
    if used != requested:
        reason = (
            f"the lag count was clamped from {requested} to {used}; it must lie between 1 "
            f"and {n - 1} for {n} residuals"
        )
    centred = values - float(np.mean(values))
    denominator = float(centred @ centred)
    total = 0.0
    for lag in range(1, used + 1):
        rho = float(centred[: n - lag] @ centred[lag:]) / denominator
        total += rho**2 / (n - lag)
    statistic = n * (n + 2) * total
    return _ok(
        "ljung_box",
        statistic=statistic,
        p_value=float(scipy_stats.chi2.sf(statistic, used)),
        df=float(used),
        reason=reason,
        extra_limitations=(_TIED_X_LIMITATION,) if tied_x else (),
    )


def durbin_watson(residuals: FloatArray, *, tied_x: bool = False) -> DiagnosticTest:
    """Durbin-Watson statistic for first-order autocorrelation of the residuals.

    The statistic is ``sum((r[t] - r[t-1])**2) / sum(r[t]**2)``.  It has no exact
    p-value, so only the statistic is reported.  ``tied_x`` says that the ordering used
    has duplicated x values, which makes that order ambiguous.
    """
    values = _as_vector(residuals, "residuals")
    unusable = _unusable(values)
    if unusable is not None:
        return _not_applicable("durbin_watson", unusable)
    differences = np.diff(values)
    denominator = float(values @ values)
    extra = (_TIED_X_LIMITATION,) if tied_x else ()
    return _ok(
        "durbin_watson",
        statistic=float((differences @ differences) / denominator),
        p_value=None,
        df=None,
        extra_limitations=extra,
    )


def run_diagnostics(view: FitView, lags: int | None = None) -> DiagnosticsReport:
    """Run the five residual tests on a converged fit.

    The weighted residuals ``r = sqrt(w) * (y - fitted)`` are sorted by ascending ``x``
    (ties are broken by the residual value) so that the autocorrelation tests see an order
    that does not depend on the order of the input rows, and every test is run on them.
    ``lags`` is passed to the Ljung-Box test.

    Raises:
        StatisticsError: ``FIT_NOT_CONVERGED`` when the fit did not converge.
    """
    if view.result.status is not FitStatus.CONVERGED:
        raise StatisticsError(
            "FIT_NOT_CONVERGED",
            f"diagnostics need a converged fit; the status is {view.result.status.value!r}",
        )
    weighted = np.sqrt(view.w) * view.residuals
    order = np.lexsort((weighted, view.x))
    residuals = weighted[order]
    fitted = view.fitted[order]
    tied = bool(np.unique(view.x).size != view.x.size)

    if _is_exact_fit(view):
        reason = (
            "the fit is exact (the residuals are zero to machine precision), so the tests "
            "carry no information"
        )
        return DiagnosticsReport(tests=tuple(_not_applicable(name, reason) for name in _TEST_NAMES))

    return DiagnosticsReport(
        tests=(
            shapiro_wilk(residuals),
            anderson_darling(residuals),
            breusch_pagan(residuals, fitted),
            ljung_box(residuals, lags, tied_x=tied),
            durbin_watson(residuals, tied_x=tied),
        )
    )


def _is_exact_fit(view: FitView) -> bool:
    """Whether the residuals are zero to machine precision relative to the data scale."""
    data_scale = max(float(np.ptp(view.y)), _OFFSET_FLOOR_RATIO * float(np.max(np.abs(view.y))))
    residual_scale = float(np.max(np.abs(view.residuals)))
    return residual_scale <= _EXACT_FIT_RATIO * data_scale
