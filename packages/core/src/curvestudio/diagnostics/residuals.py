"""Residual set and influence measures: everything a residual-diagnostics plot panel needs.

All quantities are computed on the *active* points (weight > 0) of a :class:`FitView`, in
input order, from the weighted residuals ``r = sqrt(w) * e`` and the weighted Jacobian
``J_w = sqrt(w) * J``. Leverage comes from the thin SVD of ``J_w`` (never from the normal
equations, which square the condition number). For a nonlinear model the Jacobian is
evaluated at the estimate, so leverage, Cook's distance and PRESS are the usual linearised
(first-order) measures; for polynomials they are exact.

A point with leverage 1 fixes its own fitted value, so its standardized residual and
Cook's distance are not defined: those entries are ``None`` (never NaN) and the point is
flagged as both high-leverage and high-influence.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm

from curvestudio.statistics import FitView, StatisticsError

FloatArray = NDArray[np.float64]

_LEVERAGE_ONE_TOLERANCE = 1e-10
"""``1 - h`` at or below this counts as ``h == 1`` (the SVD returns 1 to ~1e-16)."""

_ACF_BOUND_Z = 1.96
"""Two-sided 95 % normal quantile used for the white-noise band of the ACF."""


class ResidualSet(BaseModel):
    """Per-point residual diagnostics and plot-ready series for one fit.

    Per-point lists (``x`` to ``scale_location``) hold the active points in input order and
    have equal length. ``standardized``, ``studentized_external``, ``cooks_distance`` and
    ``scale_location`` are ``None`` where undefined (leverage-one points, a zero residual
    variance, or too few degrees of freedom). ``qq_theoretical``/``qq_ordered`` pair the
    normal quantiles at the Blom positions ``(i - 0.375)/(n + 0.25)`` with the sorted
    weighted residuals. ``hist_edges``/``hist_counts`` histogram the weighted residuals
    (Freedman-Diaconis bins, Sturges when the IQR is 0, or ``bins`` equal bins).

    ``acf`` is the demeaned, biased autocorrelation of the weighted residuals taken in
    ascending-x order (ties keep input order) at lags ``0..nlags``; it is empty when the
    residuals have no variance. ``acf_bound`` is ``1.96/sqrt(n)``. ``flagged_high_leverage``
    (``h > 2k/n``) and ``flagged_high_influence`` (Cook's ``> 4/n``) index the active arrays;
    leverage-one points are in both.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: list[float]
    fitted: list[float]
    residuals: list[float]
    weighted_residuals: list[float]
    standardized: list[float | None]
    studentized_external: list[float | None]
    leverage: list[float]
    cooks_distance: list[float | None]
    scale_location: list[float | None]
    qq_theoretical: list[float]
    qq_ordered: list[float]
    hist_edges: list[float]
    hist_counts: list[int]
    acf: list[float]
    acf_bound: float
    leverage_threshold: float
    cooks_threshold: float
    flagged_high_leverage: list[int]
    flagged_high_influence: list[int]


@dataclass(frozen=True)
class Influence:
    """Output of :func:`influence_measures`; ``None`` marks an undefined entry."""

    leverage: FloatArray
    standardized: list[float | None]
    studentized_external: list[float | None]
    cooks_distance: list[float | None]
    scale_location: list[float | None]
    leverage_one: list[int]


def leverage(weighted_jacobian: FloatArray) -> FloatArray:
    """Diagonal of ``J_w (J_w' J_w)^-1 J_w'`` from the thin SVD of ``J_w``.

    A rank-deficient Jacobian is projected onto its numerical column space (singular
    values below ``max(shape) * eps * s_max`` are dropped), so the sum is its rank.
    """
    jacobian = np.asarray(weighted_jacobian, dtype=np.float64)
    u, s, _ = np.linalg.svd(jacobian, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros(jacobian.shape[0])
    rank = int(np.count_nonzero(s > max(jacobian.shape) * np.finfo(np.float64).eps * s[0]))
    return np.clip(np.sum(u[:, :rank] ** 2, axis=1), 0.0, 1.0)


def _undefined(n: int) -> list[float | None]:
    return [None] * n


def influence_measures(
    weighted_residuals: FloatArray, weighted_jacobian: FloatArray, df: int, s2: float
) -> Influence:
    """Leverage, standardized/externally studentized residuals and Cook's distance.

    ``standardized = r / (s sqrt(1 - h))``; ``studentized_external = standardized *
    sqrt((df - 1) / (df - standardized^2))`` (undefined for ``df < 2`` or when the
    denominator is not positive); ``cooks = standardized^2 h / (k (1 - h))``;
    ``scale_location = sqrt(|standardized|)``. Everything is ``None`` for a point with
    ``h == 1`` and, for the whole set, when ``s2 <= 0`` (a perfect fit has no scale).
    """
    r = np.asarray(weighted_residuals, dtype=np.float64)
    h = leverage(weighted_jacobian)
    n, k = weighted_jacobian.shape
    one = (1.0 - h) <= _LEVERAGE_ONE_TOLERANCE
    leverage_one = [int(i) for i in np.flatnonzero(one)]
    if s2 <= 0.0:
        none = _undefined(n)
        return Influence(h, none, list(none), list(none), list(none), leverage_one)

    standardized = np.full(n, np.nan)
    standardized[~one] = r[~one] / (np.sqrt(s2) * np.sqrt(1.0 - h[~one]))
    cooks = np.full(n, np.nan)
    cooks[~one] = standardized[~one] ** 2 * h[~one] / (k * (1.0 - h[~one]))

    external = np.full(n, np.nan)
    if df >= 2:
        denominator = df - standardized**2  # NaN (leverage one) compares False below
        valid = denominator > 0.0
        external[valid] = standardized[valid] * np.sqrt((df - 1) / denominator[valid])
    scale_location = np.sqrt(np.abs(standardized))

    def listed(values: FloatArray) -> list[float | None]:
        return [None if np.isnan(v) else float(v) for v in values]

    return Influence(
        h,
        listed(standardized),
        listed(external),
        listed(cooks),
        listed(scale_location),
        leverage_one,
    )


def _autocorrelation(r_in_x_order: FloatArray, nlags: int) -> list[float]:
    centred = r_in_x_order - r_in_x_order.mean()
    variance_sum = float(np.dot(centred, centred))
    if variance_sum == 0.0:
        return []
    return [
        float(np.dot(centred[: centred.size - k], centred[k:]) / variance_sum)
        for k in range(nlags + 1)
    ]


def residual_set(view: FitView, nlags: int | None = None, bins: int | None = None) -> ResidualSet:
    """Build the :class:`ResidualSet` of a converged fit.

    Args:
        view: the fit seen through its active points.
        nlags: highest ACF lag; default ``min(int(10 log10 n), n - 1)``, larger values are
            clamped to ``n - 1``.
        bins: number of equal-width histogram bins; default Freedman-Diaconis.

    Raises:
        StatisticsError: ``NO_DEGREES_OF_FREEDOM`` (no residual variance exists) or
            ``INVALID_ARGUMENT`` (``nlags < 0`` or ``bins < 1``).
    """
    if view.s2 is None:
        raise StatisticsError(
            "NO_DEGREES_OF_FREEDOM",
            f"the fit has {view.df} residual degrees of freedom; residual diagnostics need "
            "at least one",
        )
    if nlags is not None and nlags < 0:
        raise StatisticsError("INVALID_ARGUMENT", f"nlags must be >= 0, got {nlags}")
    if bins is not None and bins < 1:
        raise StatisticsError("INVALID_ARGUMENT", f"bins must be >= 1, got {bins}")

    n = int(view.x.size)
    k = len(view.names)
    weighted = np.sqrt(view.w) * view.residuals
    influence = influence_measures(weighted, view.weighted_jacobian, view.df, view.s2)
    h = influence.leverage

    ordered = np.sort(weighted)
    blom = (np.arange(1, n + 1) - 0.375) / (n + 0.25)

    q1, q3 = np.percentile(weighted, [25.0, 75.0])
    if bins is not None:
        edges = np.histogram_bin_edges(weighted, bins=bins)
    else:
        edges = np.histogram_bin_edges(weighted, bins="fd" if q3 > q1 else "sturges")
    counts, _ = np.histogram(weighted, bins=edges)

    default_lags = int(10.0 * np.log10(n)) if n > 1 else 0
    lags = min(default_lags if nlags is None else nlags, n - 1)
    acf = _autocorrelation(weighted[np.argsort(view.x, kind="stable")], lags)

    leverage_threshold = 2.0 * k / n
    cooks_threshold = 4.0 / n
    one = set(influence.leverage_one)
    high_leverage = [i for i in range(n) if i in one or h[i] > leverage_threshold]
    high_influence = [
        i
        for i, d in enumerate(influence.cooks_distance)
        if i in one or (d is not None and d > cooks_threshold)
    ]

    return ResidualSet(
        x=view.x.tolist(),
        fitted=view.fitted.tolist(),
        residuals=view.residuals.tolist(),
        weighted_residuals=weighted.tolist(),
        standardized=influence.standardized,
        studentized_external=influence.studentized_external,
        leverage=h.tolist(),
        cooks_distance=influence.cooks_distance,
        scale_location=influence.scale_location,
        qq_theoretical=norm.ppf(blom).tolist(),
        qq_ordered=ordered.tolist(),
        hist_edges=edges.tolist(),
        hist_counts=[int(c) for c in counts],
        acf=acf,
        acf_bound=_ACF_BOUND_Z / float(np.sqrt(n)),
        leverage_threshold=leverage_threshold,
        cooks_threshold=cooks_threshold,
        flagged_high_leverage=high_leverage,
        flagged_high_influence=high_influence,
    )


def press_rmse(view: FitView) -> float | None:
    """Leave-one-out prediction error ``sqrt(mean((e / (1 - h))^2))`` in the units of y.

    ``e`` is the unweighted residual and ``h`` the weighted leverage, so for a linear
    (polynomial) fit this is the exact leave-one-out RMSE, weighted fits included. For a
    nonlinear model it is the *linearised* approximation (Jacobian at the estimate).
    ``None`` when any point has leverage 1 (its leave-one-out error is not defined).
    """
    h = leverage(view.weighted_jacobian)
    if bool(np.any((1.0 - h) <= _LEVERAGE_ONE_TOLERANCE)):
        return None
    return float(np.sqrt(np.mean((view.residuals / (1.0 - h)) ** 2)))
