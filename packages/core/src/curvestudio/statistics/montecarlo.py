"""Monte-Carlo propagation of parameter uncertainty.

The delta method (:mod:`curvestudio.statistics.bands`) describes the uncertainty of a
derived quantity ``g(theta)`` by a normal distribution built from the first-order Taylor
expansion of ``g`` at the estimates. That is exact for a quantity linear in the
parameters and only approximate for a nonlinear one, and it says nothing about the
*shape* of the distribution. Monte-Carlo propagation replaces the approximation with the
computation: the parameters are drawn from their asymptotic normal distribution
``N(theta-hat, C)``, the quantity is evaluated on every draw, and the spread of the
results is reported.

Every random number comes from ``numpy.random.default_rng(seed)`` and the seed is stored
in the result, so the same seed always reproduces the same numbers.
"""

import math
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.statistics.bands import Band
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.view import FitView

FloatArray = NDArray[np.float64]

_PSD_TOLERANCE = 1e-12
"""Relative size of a negative eigenvalue a covariance may carry before it is refused: an
eigenvalue below ``-1e-12 * max_eigenvalue`` is a real failure, not rounding."""


class Propagation(BaseModel):
    """The distribution of a derived quantity under the parameter uncertainty of a fit.

    ``n`` is the number of draws that were taken and ``n_nonfinite`` how many of them did
    not give a finite result (those are left out of every statistic and counted, never
    dropped silently). ``mean`` and ``std`` describe the finite results, ``ci_low`` and
    ``ci_high`` are the central ``level`` percentiles of them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    seed: int
    level: FiniteFloat
    mean: FiniteFloat
    std: FiniteFloat
    ci_low: FiniteFloat
    ci_high: FiniteFloat
    n_nonfinite: int


def _check_level(level: float) -> None:
    if not (np.isfinite(level) and 0.0 < level < 1.0):
        raise StatisticsError(
            "INVALID_LEVEL", f"level must lie strictly between 0 and 1, got {level}"
        )


def _check_x(x: FloatArray) -> FloatArray:
    points = np.asarray(x, dtype=np.float64)
    if points.ndim != 1 or points.size == 0:
        raise StatisticsError("INVALID_X", "x must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(points)):
        raise StatisticsError("INVALID_X", "x must contain only finite values")
    return points


def _check_n(n: int) -> None:
    if n <= 0:
        raise StatisticsError("INVALID_N", f"the number of draws must be positive, got {n}")


def _psd_factor(covariance: FloatArray) -> FloatArray:
    """A factor ``L`` with ``L L'`` equal to the positive semi-definite part of ``covariance``.

    The covariance is symmetrised first (a covariance is symmetric; one that is not is
    rounded) and its eigenvalues are clipped at zero. A clipping larger than
    ``1e-12 * max_eigenvalue`` means the covariance is not positive semi-definite: it is
    refused rather than quietly altered.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise StatisticsError("INVALID_COVARIANCE", "the covariance holds a non-finite value")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if smallest < -_PSD_TOLERANCE * max(largest, 0.0):
        raise StatisticsError(
            "COVARIANCE_NOT_PSD",
            f"the covariance is not positive semi-definite (smallest eigenvalue "
            f"{smallest:.6g}, largest {largest:.6g}); it is refused rather than altered",
        )
    # eigh returns the eigenvalues in ascending order, so the columns of eigenvectors are
    # scaled by the square roots of the (clipped) eigenvalues.
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def sample_parameters(view: FitView, n: int = 10000, seed: int = 0) -> FloatArray:
    """Draw ``n`` parameter vectors from the asymptotic normal distribution of the fit.

    The draws are ``theta-hat + L z`` with ``z ~ N(0, I)`` and ``L L' = C``, so their
    sample covariance is the covariance of the fit.

    Raises:
        StatisticsError: ``NO_DEGREES_OF_FREEDOM`` / ``NO_COVARIANCE`` (the fit gives no
            parameter uncertainty); ``INVALID_COVARIANCE`` (the covariance holds a
            non-finite value); ``COVARIANCE_NOT_PSD`` (the covariance is not positive
            semi-definite); ``INVALID_N`` (``n`` is not positive).
    """
    _check_n(n)
    factor = _psd_factor(view.require_covariance())
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, factor.shape[0]))
    return np.asarray(view.theta + noise @ factor.T, dtype=np.float64)


def propagate(
    view: FitView,
    func: Callable[[Mapping[str, float]], float],
    n: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> Propagation:
    """Propagate the parameter uncertainty of a fit through ``func``.

    ``func`` is called once per draw with a mapping of parameter name to value and
    returns the derived quantity. A draw that does not give a finite number -- because
    ``func`` returned NaN or infinity, or because it raised, the quantity being undefined
    for that parameter set -- is counted in ``n_nonfinite`` and left out of the
    statistics. When more than half of the draws fail the propagation is refused as
    unstable rather than reported from the few that survived.

    The reported interval is the central ``level`` interval of the propagated
    distribution (the percentiles of the finite results). For a quantity linear in the
    parameters that distribution is normal, and the interval then coincides with
    ``mean +- z * std``.

    Raises:
        StatisticsError: as :func:`sample_parameters`, plus ``INVALID_LEVEL`` (``level``
            not in (0, 1)) and ``PROPAGATION_UNSTABLE`` (too few draws gave a finite
            result).
    """
    _check_level(level)
    samples = sample_parameters(view, n=n, seed=seed)
    values = np.empty(samples.shape[0], dtype=np.float64)
    n_nonfinite = 0
    first_failure: BaseException | None = None
    for index, row in enumerate(samples):
        try:
            value = float(func(dict(zip(view.names, row.tolist(), strict=True))))
        except Exception as exc:  # the quantity is undefined for this parameter set
            value = math.nan
            if first_failure is None:
                first_failure = exc
        if not math.isfinite(value):
            n_nonfinite += 1
        values[index] = value

    finite = values[np.isfinite(values)]
    detail = f"; the first failure was {first_failure!r}" if first_failure else ""
    if finite.size * 2 < samples.shape[0]:
        raise StatisticsError(
            "PROPAGATION_UNSTABLE",
            f"{n_nonfinite} of {samples.shape[0]} draws did not give a finite result"
            f"{detail}; more than half of the draws failed, so the propagation is not defined",
        )
    if finite.size < 2:
        raise StatisticsError(
            "PROPAGATION_UNSTABLE",
            f"only {finite.size} of {samples.shape[0]} draws gave a finite result; a spread "
            "cannot be reported from so few values",
        )

    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return Propagation(
        n=int(samples.shape[0]),
        seed=seed,
        level=level,
        mean=float(finite.mean()),
        std=float(finite.std(ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
        n_nonfinite=n_nonfinite,
    )


def _responses(view: FitView, points: FloatArray, samples: FloatArray) -> FloatArray:
    """The model values at ``points`` for every parameter draw, shape ``(points, draws)``."""
    if view.model is None:
        # A polynomial is linear in its coefficients, so its Jacobian is the design
        # matrix and the values follow from a single matrix product.
        return view.jacobian_at(points) @ samples.T
    columns = [
        view.model.evaluate(points, **dict(zip(view.names, row.tolist(), strict=True)))
        for row in samples
    ]
    return np.asarray(columns, dtype=np.float64).T


def monte_carlo_band(
    view: FitView,
    x: FloatArray,
    n: int = 5000,
    level: float = 0.95,
    seed: int = 0,
) -> Band:
    """Monte-Carlo percentile band for the mean response at ``x``.

    The parameters are drawn from ``N(theta-hat, C)`` and the model is evaluated on every
    draw, so the band follows the curvature of the model in its parameters, where the
    delta-method band of :func:`~curvestudio.statistics.confidence_band` only linearises
    it. The band carries parameter uncertainty only: no residual noise is added, so it
    is a band on the mean response and stays well inside the prediction band.

    The bounds are the central ``level`` percentiles of the draws. No Student-t
    correction is applied (the draws are normal by construction), so the band is slightly
    narrower than the t-based delta-method band of the same level.

    Raises:
        StatisticsError: as :func:`sample_parameters`, plus ``INVALID_LEVEL`` (``level``
            not in (0, 1)), ``INVALID_X`` (empty, non-1-D or non-finite ``x``, or a model
            that is not finite there) and ``PROPAGATION_UNSTABLE`` (more than half of the
            draws gave no finite model value at one of the requested ``x``).
    """
    _check_level(level)
    points = _check_x(x)
    samples = sample_parameters(view, n=n, seed=seed)
    fitted = view.predict(points)
    if not np.all(np.isfinite(fitted)):
        raise StatisticsError(
            "INVALID_X",
            "the model is not finite (outside its domain) at one or more of the requested x",
        )

    responses = _responses(view, points, samples)
    usable = np.count_nonzero(np.isfinite(responses), axis=1)
    if np.any(usable * 2 < samples.shape[0]):
        worst = int(np.argmin(usable))
        raise StatisticsError(
            "PROPAGATION_UNSTABLE",
            f"at x = {points[worst]:g} only {usable[worst]} of {samples.shape[0]} draws gave "
            "a finite model value; more than half failed, so the band is not defined there",
        )

    values = np.where(np.isfinite(responses), responses, np.nan)
    alpha = (1.0 - level) / 2.0
    lower = np.nanquantile(values, alpha, axis=1)
    upper = np.nanquantile(values, 1.0 - alpha, axis=1)

    dropped = int(samples.shape[0] - usable.min())
    extra = (
        f" Up to {dropped} of the {samples.shape[0]} draws were not finite at some x and "
        "were left out of the percentiles there."
        if dropped
        else ""
    )
    note = (
        f"Monte-Carlo {level:.0%} percentile band for the mean response, from {samples.shape[0]} "
        "parameter draws taken from N(theta-hat, C) and evaluated on the model: it follows the "
        "curvature of the model in its parameters and carries parameter uncertainty only (no "
        "residual noise). No Student-t correction is applied (the draws are normal by "
        "construction), so it is narrower than the t-based delta-method band of the same level. "
        f"The coverage is pointwise, not simultaneous over the curve.{extra}"
    )
    return Band(
        kind="monte_carlo",
        level=level,
        x=points.tolist(),
        fit=fitted.tolist(),
        lower=np.asarray(lower, dtype=np.float64).tolist(),
        upper=np.asarray(upper, dtype=np.float64).tolist(),
        extrapolated=((points < view.x.min()) | (points > view.x.max())).tolist(),
        df=view.df,
        note=note,
    )
