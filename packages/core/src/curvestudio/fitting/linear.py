"""Linear and polynomial least-squares regression (OLS and WLS).

Polynomials are linear in their coefficients, so the optimum is found directly and the
result is exact to rounding, with no starting values or iteration. The numerics follow the
usual advice for ill-conditioned problems: the weighted design matrix has each column scaled
to unit maximum, ``np.linalg.lstsq`` (SVD based) solves it, and the covariance is built from
the same SVD. The normal equations ``X'WX`` are never formed, because that squares the
condition number and loses digits on exactly the data (high degree, wide x range) where
accuracy matters.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from curvestudio.data import XY
from curvestudio.fitting._uncertainty import correlation as _correlation
from curvestudio.fitting._uncertainty import finite_or_none as _finite_or_none
from curvestudio.fitting.metrics import FitStatistics, fit_statistics
from curvestudio.fitting.result import (
    FitResult,
    FitStatus,
    ParameterEstimate,
    collect_provenance,
)

FloatArray = NDArray[np.float64]

HIGH_ORDER_DEGREE = 6
"""Degree from which a polynomial is flagged as unstable outside the data range."""

ILL_CONDITIONED_LIMIT = 1e12
"""Condition-number limit above which digits are visibly at risk.

Applied to the column-scaled design *and* to the unscaled one. Column scaling lets the
solver cope, but it hides the case where the coefficients themselves span many orders of
magnitude (x from 1e-9 to 1e9 at degree 3): the scaled problem looks tame while the small
coefficients are lost to rounding, so the unscaled condition number is the honest alarm.
"""

_DIGITS = 6


def _term(power: int) -> str:
    return "x" if power == 1 else f"x**{power}"


def _equation(powers: range, values: list[float] | None) -> str:
    """Render ``y = b0 + b1*x + ...``; with ``values`` the numbers replace the names."""
    parts: list[str] = []
    for i, power in enumerate(powers):
        if values is None:
            coef = f"b{power}"
        else:
            text = f"{values[i]:.{_DIGITS}g}"
            coef = f"({text})" if values[i] < 0 else text
        parts.append(coef if power == 0 else f"{coef}*{_term(power)}")
    return "y = " + " + ".join(parts)


def fit_polynomial(
    data: XY,
    degree: int,
    *,
    through_origin: bool = False,
    confidence_level: float = 0.95,
) -> FitResult:
    """Fit ``y = b0 + b1*x + ... + b{degree}*x**degree`` by (weighted) least squares.

    ``degree=1`` is simple linear regression. Weights come from ``data.weights`` and
    multiply squared residuals (``sum(w * r**2)`` is minimised); points with weight 0
    contribute nothing, so statistics, degrees of freedom and the covariance are all
    computed on the active (``w > 0``) points, while ``fitted``/``residuals`` cover every
    input point. With ``through_origin=True`` the intercept is fixed at 0 and only
    ``b1..b{degree}`` are estimated; R2 is then the uncentred form.

    Standard errors are ``sqrt(diag(s2 * inv(X'WX)))`` with ``s2 = sse / df``; confidence
    intervals use the Student-t quantile with ``df`` degrees of freedom. Exact-fit data
    (``sse == 0``) gives zero covariance and zero-width intervals, never NaN, and an
    undefined (``None``) correlation.

    A problem that cannot be solved is reported through :attr:`FitResult.status`, not
    raised: ``INVALID_INPUT`` when there are no more active points than parameters, and
    ``FAILED`` when the design is rank deficient (for example fewer distinct x values than
    ``degree + 1``) or overflows. Warnings: ``HIGH_ORDER_POLYNOMIAL`` for ``degree >= 6``
    and ``ILL_CONDITIONED`` when the condition number of the column-scaled or of the unscaled
    weighted design exceeds 1e12.

    Raises:
        ValueError: ``degree`` is not an int >= 0 (>= 1 through the origin) or
            ``confidence_level`` is not strictly between 0 and 1 (programmer errors).
    """
    min_degree = 1 if through_origin else 0
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < min_degree:
        raise ValueError(f"degree must be an int >= {min_degree}, got {degree!r}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level!r}")

    powers = range(min_degree, degree + 1)
    k = len(powers)
    names = [f"b{p}" for p in powers]
    weighted = data.weights is not None
    provenance = collect_provenance(data.fingerprint, seed=0)
    equation = _equation(powers, None)
    model_id = f"poly{degree}_origin" if through_origin else f"poly{degree}"
    warnings: list[str] = []
    if degree >= HIGH_ORDER_DEGREE:
        warnings.append("HIGH_ORDER_POLYNOMIAL: unstable outside the data range")

    def build(
        status: FitStatus,
        reason: str,
        *,
        parameters: tuple[ParameterEstimate, ...] = (),
        covariance: list[list[float]] | None = None,
        correlation: list[list[float]] | None = None,
        stats: FitStatistics | None = None,
        fitted: list[float] | None = None,
        residuals: list[float] | None = None,
        equation_fitted: str | None = None,
    ) -> FitResult:
        return FitResult(
            model_id=model_id,
            equation=equation,
            equation_fitted=equation_fitted,
            parameters=parameters,
            covariance=covariance,
            correlation=correlation,
            stats=stats,
            fitted=fitted or [],
            residuals=residuals or [],
            status=status,
            termination_reason=reason,
            method="wls" if weighted else "ols",
            loss="linear",
            initial_params={},
            n_starts=1,
            warnings=tuple(warnings),
            provenance=provenance,
        )

    x = np.asarray(data.x, dtype=np.float64)
    y = np.asarray(data.y, dtype=np.float64)
    w = np.ones_like(x) if data.weights is None else np.asarray(data.weights, dtype=np.float64)
    active = w > 0.0
    n_active = int(np.count_nonzero(active))
    if n_active <= k:
        return build(
            FitStatus.INVALID_INPUT,
            f"{n_active} active observation(s) for {k} parameter(s): need more observations "
            "than parameters",
        )

    x_a, y_a, w_a = x[active], y[active], w[active]
    with np.errstate(over="ignore", invalid="ignore"):
        design = x_a[:, None] ** np.array(list(powers), dtype=np.float64)
    if not bool(np.all(np.isfinite(design))):
        return build(
            FitStatus.FAILED,
            "design matrix overflows (x**degree is not finite); rescale x or lower the degree",
        )

    root_w = np.sqrt(w_a)
    scaled = design * root_w[:, None]
    col_scale = np.max(np.abs(scaled), axis=0)
    col_scale[col_scale == 0.0] = 1.0  # an all-zero column stays zero and shows as rank loss
    scaled /= col_scale

    coef_s, _, rank, sing = np.linalg.lstsq(scaled, y_a * root_w, rcond=None)
    cond = float(sing[0] / sing[-1]) if sing[-1] > 0.0 else math.inf
    if cond > ILL_CONDITIONED_LIMIT or np.linalg.cond(design * root_w[:, None]) > (
        ILL_CONDITIONED_LIMIT
    ):
        warnings.append("ILL_CONDITIONED")
    coef = coef_s / col_scale

    full_coef = np.concatenate([np.zeros(1), coef]) if through_origin else coef
    fitted_all = np.polynomial.polynomial.polyval(x, full_coef)
    finite_fit = bool(np.all(np.isfinite(coef)) and np.all(np.isfinite(fitted_all)))
    fitted = fitted_all.tolist() if finite_fit else None
    residuals = (y - fitted_all).tolist() if finite_fit else None

    if int(rank) < k:
        reason = (
            f"design matrix is rank deficient (rank {int(rank)} < {k} parameters): "
            "too few distinct x values for this degree, or x is numerically degenerate"
        )
        params = (
            tuple(
                ParameterEstimate(name=n, value=float(v)) for n, v in zip(names, coef, strict=True)
            )
            if finite_fit
            else ()
        )
        return build(
            FitStatus.FAILED, reason, parameters=params, fitted=fitted, residuals=residuals
        )
    if not finite_fit:
        return build(FitStatus.FAILED, "solution is not finite (numerical overflow)")

    stats = fit_statistics(
        y_a, fitted_all[active], k, weights=w_a if weighted else None, through_origin=through_origin
    )

    # inv(A'A) = V diag(1/s^2) V' from the SVD of the scaled design, then undo the scaling.
    _, svals, vt = np.linalg.svd(scaled, full_matrices=False)
    cov_unit = (vt.T / svals**2) @ vt / np.outer(col_scale, col_scale)

    if stats.sse == 0.0:
        covariance: list[list[float]] | None = [[0.0] * k for _ in range(k)]
        correlation: list[list[float]] | None = None
        std_err = np.zeros(k)
    else:
        cov = (stats.sse / stats.df) * cov_unit
        covariance = _finite_or_none(cov)
        std_err = np.sqrt(np.diag(cov)) if covariance is not None else np.full(k, np.nan)
        correlation = _correlation(cov, std_err) if covariance is not None else None

    estimates: list[ParameterEstimate] = []
    half_width = scipy_stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, stats.df)
    for name, value, se in zip(names, coef.tolist(), std_err.tolist(), strict=True):
        if math.isfinite(se):
            margin = float(half_width) * se
            estimates.append(
                ParameterEstimate(
                    name=name,
                    value=value,
                    std_err=se,
                    ci_low=value - margin,
                    ci_high=value + margin,
                )
            )
        else:
            estimates.append(ParameterEstimate(name=name, value=value))

    return build(
        FitStatus.CONVERGED,
        "linear least squares (QR/SVD) solved",
        parameters=tuple(estimates),
        covariance=covariance,
        correlation=correlation,
        stats=stats,
        fitted=fitted,
        residuals=residuals,
        equation_fitted=_equation(powers, coef.tolist()),
    )
