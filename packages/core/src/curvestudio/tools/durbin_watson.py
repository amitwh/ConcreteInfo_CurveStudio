"""The Durbin-Watson statistic with p-values that are exact for the fitted model.

The core computes ``d = sum((e[t] - e[t-1])**2) / sum(e[t]**2)`` and stops there: the
distribution of ``d`` depends on the design matrix of the fitted model, which the published
bound-and-limit tables do not know. This module computes that distribution.

Method. The errors, as the fitted model sees them, are the observations projected on the
space orthogonal to the design matrix. With ``Q`` an orthonormal basis of that space (the
SVD of the Jacobian gives it), ``A`` the tridiagonal matrix of the numerator (``2`` on the
diagonal, ``1`` at both ends, ``-1`` beside it) and ``B = Q' A Q``, the statistic is
distributed as ``sum(lambda_i z_i**2) / sum(z_i**2)`` for independent standard-normal ``z``,
where ``lambda`` are the ``n - k`` eigenvalues of ``B``. Each tail of ``d`` is then the
distribution function of a central quadratic form in normal variables, which Imhof's integral
gives (J. P. Imhof, Biometrika 48, 1961).

A nonlinear model enters through the Jacobian at its estimates: a linearisation.
"""

import math
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import integrate, optimize
from scipy.integrate import IntegrationWarning

from curvestudio.tools._errors import ToolError

FloatArray = NDArray[np.float64]

DW_EXACT_MAX_N = 1000
"""Above this many active points the exact distribution is not computed: the
eigen-decomposition grows with the cube of the number of points."""

RANK_RTOL = 1e-10
"""A column of the Jacobian counts as dependent when its singular value is below this
fraction of the largest."""

_TOLERANCE = 1e-13
"""Absolute and relative error asked of every quadrature."""
_NEGLIGIBLE = 1e-10
"""The plain quadrature stops, and the tail quadrature begins, where the envelope
``1 / (u rho(u))`` of Imhof's integrand has fallen below this."""
_LIMIT = 500
"""The most subintervals the plain quadrature may bisect its range into."""
_HEAD_OSCILLATIONS = 60
"""The most periods of Imhof's oscillation the plain quadrature resolves in the head."""
_CHUNK_OSCILLATIONS = 80
"""The most it resolves in one interval of the tail; past that the Fourier quadrature takes
over."""
_TAIL_BOUND_T = 40.0
"""Beyond a point whose one-sided chance is at most ``e**-TAIL_BOUND_T`` by Laurent and
Massart's bound on a sum of weighted chi-squares, the distribution function is 0 or 1."""


class DurbinWatsonTest(BaseModel):
    """The Durbin-Watson statistic with the two p-values of its exact distribution.

    ``df`` is ``n - k``. ``positive`` is the chance of a statistic this low or lower, which
    positive autocorrelation gives; ``negative`` the chance of one this high or higher.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    statistic: float
    df: int
    positive: float
    negative: float


class DurbinWatsonCritical(BaseModel):
    """The two critical values of the Durbin-Watson statistic for one design.

    At ``lower`` the chance of a statistic that low is ``alpha``; at ``upper`` the chance of
    one that high is ``alpha``.
    """

    model_config = ConfigDict(frozen=True)

    lower: float
    upper: float
    alpha: float
    df: int


def quadratic_form_cdf(weights: FloatArray, x: float) -> float:
    """The chance that ``sum(w_i z_i**2) <= x`` for independent standard-normal ``z_i``.

    Imhof's integral: ``F(x) = 1/2 - (1/pi) * integral of sin(theta(u)) / (u rho(u)) du``
    over ``u > 0``, with ``theta(u) = (sum of arctan(w_i u) - x u) / 2`` and
    ``rho(u) = product of (1 + w_i**2 u**2)**(1/4)``. At ``u = 0`` the integrand has the
    removable limit ``(sum of w_i - x) / 2``. A tail is its own integral: its digits do not
    depend on subtracting the other tail from one.

    Args:
        weights: the coefficients of the form; a zero weight takes no part.
        x: the point, which may have either sign.

    Returns:
        A probability between 0 and 1, right to about ``1e-14`` in absolute terms: one far
        below that has few digits of its own.

    Raises:
        ValueError: when a weight or ``x`` is not finite.
        ToolError: ``INTEGRATION_NOT_CONVERGED`` when the quadrature cannot reach its
            tolerance: the value it stopped at is not returned as if it were exact. This
            happens for a tail that decays slowly and oscillates for long, as with a single
            weight and a point far into a tail.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not bool(np.isfinite(w).all()):
        raise ValueError("every weight must be finite")
    if not math.isfinite(x):
        raise ValueError("x must be finite")
    w = w[w != 0.0]
    if w.size == 0:
        return 1.0 if x >= 0.0 else 0.0
    # The chance is unchanged when the weights and x are scaled by one positive number.
    largest = float(np.abs(w).max())
    w = w / largest
    x = x / largest
    if x <= 0.0 and bool((w > 0.0).all()):
        return 0.0
    if x >= 0.0 and bool((w < 0.0).all()):
        return 1.0
    # Laurent and Massart: the chance that the form exceeds its mean by this much, or falls
    # as far below, is at most e**-TAIL_BOUND_T.
    reach = float(np.abs(w).sum()) + 2.0 * math.sqrt(_TAIL_BOUND_T * float((w * w).sum()))
    reach += 2.0 * _TAIL_BOUND_T
    if x >= reach:
        return 1.0
    if x <= -reach:
        return 0.0
    # The head runs to where the envelope has fallen and every arctan has levelled off; with
    # ``x`` it also stops where its oscillation grows past what the plain quadrature resolves.
    upper = min(_where_envelope_falls(w, _NEGLIGIBLE), 8.0 / float(np.abs(w).min()))
    if x != 0.0:
        upper = min(upper, _HEAD_OSCILLATIONS * 4.0 * math.pi / abs(x))
    value = _imhof_integral(w, x, upper)
    return min(1.0, max(0.0, 0.5 - value / math.pi))


def durbin_watson_test(statistic: float, jacobian: FloatArray) -> DurbinWatsonTest | None:
    """The exact two-sided p-values of a Durbin-Watson statistic for one fitted model.

    Args:
        statistic: the Durbin-Watson statistic of the fit's residuals, taken in ascending x.
        jacobian: the Jacobian of the model at its estimates over the active points, an
            ``(n, k)`` array with its intercept column when the model has one.

    Returns:
        The test, or ``None`` when ``jacobian`` has more than ``DW_EXACT_MAX_N`` rows.

    Raises:
        ValueError: when ``statistic`` is not finite or ``jacobian`` is not two-dimensional.
        ToolError: ``NO_DATA`` for an empty Jacobian; ``RANK_DEFICIENT_DESIGN`` when the
            columns are dependent (by ``RANK_RTOL``) or not all finite;
            ``NO_DEGREES_OF_FREEDOM`` when there are no more rows than columns;
            ``INTEGRATION_NOT_CONVERGED`` as for :func:`quadratic_form_cdf`.
    """
    if not math.isfinite(statistic):
        raise ValueError("the statistic must be finite")
    eigenvalues = _eigenvalues(jacobian)
    if eigenvalues is None:
        return None
    return _test_at(statistic, eigenvalues)


def durbin_watson_critical_values(
    jacobian: FloatArray, *, alpha: float = 0.05
) -> DurbinWatsonCritical | None:
    """The exact critical values of the Durbin-Watson statistic for one fitted model.

    The distribution is not symmetric about 2 for a general design, so each value is the root
    of its own tail.

    Args:
        jacobian: as for :func:`durbin_watson_test`.
        alpha: the one-sided chance of each critical value.

    Returns:
        The critical values, or ``None`` above ``DW_EXACT_MAX_N`` rows.

    Raises:
        ValueError: when ``alpha`` does not lie strictly between 0 and 0.5, or as for
            :func:`durbin_watson_test`.
        ToolError: as for :func:`durbin_watson_test`.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between 0 and 0.5")
    eigenvalues = _eigenvalues(jacobian)
    if eigenvalues is None:
        return None
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if smallest == largest:
        # One residual direction: the statistic is that eigenvalue and never moves.
        return DurbinWatsonCritical(lower=smallest, upper=largest, alpha=alpha, df=1)
    lower = optimize.brentq(
        lambda value: quadratic_form_cdf(eigenvalues - value, 0.0) - alpha, smallest, largest
    )
    upper = optimize.brentq(
        lambda value: quadratic_form_cdf(value - eigenvalues, 0.0) - alpha, smallest, largest
    )
    return DurbinWatsonCritical(lower=lower, upper=upper, alpha=alpha, df=int(eigenvalues.size))


def _eigenvalues(jacobian: FloatArray) -> FloatArray | None:
    """The eigenvalues of ``Q' A Q``, ``Q`` a basis of the space orthogonal to ``jacobian``."""
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("the Jacobian must be two-dimensional, with a row per point")
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        raise ToolError("NO_DATA", "the Jacobian is empty, so there are no residuals to test")
    if rows > DW_EXACT_MAX_N:
        return None
    if not bool(np.isfinite(matrix).all()):
        raise ToolError(
            "RANK_DEFICIENT_DESIGN",
            "the Jacobian is not finite at every point, so its columns cannot be separated",
        )
    left, singular, _ = np.linalg.svd(matrix, full_matrices=True)
    independent = int(np.count_nonzero(singular > RANK_RTOL * singular[0]))
    if independent < columns:
        raise ToolError(
            "RANK_DEFICIENT_DESIGN",
            f"only {independent} of the {columns} columns of the Jacobian are independent",
        )
    if rows <= columns:
        raise ToolError(
            "NO_DEGREES_OF_FREEDOM",
            f"{rows} points leave no residual direction beside {columns} parameters",
        )
    basis = left[:, columns:]
    restricted = basis.T @ _apply_difference_matrix(basis)
    return np.linalg.eigvalsh((restricted + restricted.T) / 2.0)


def _apply_difference_matrix(vectors: FloatArray) -> FloatArray:
    """``A @ vectors`` with ``A`` multiplied through array shifts and never formed: ``2`` on
    the diagonal, ``1`` at both ends, ``-1`` beside it."""
    diagonal = np.full(vectors.shape[0], 2.0, dtype=np.float64)
    diagonal[0] = diagonal[-1] = 1.0
    product = vectors * diagonal[:, None]
    product[1:] -= vectors[:-1]
    product[:-1] -= vectors[1:]
    return product


def _test_at(statistic: float, eigenvalues: FloatArray) -> DurbinWatsonTest:
    positive = quadratic_form_cdf(eigenvalues - statistic, 0.0)
    negative = quadratic_form_cdf(statistic - eigenvalues, 0.0)
    return DurbinWatsonTest(
        statistic=float(statistic), df=int(eigenvalues.size), positive=positive, negative=negative
    )


def _where_envelope_falls(weights: FloatArray, level: float) -> float:
    """The smallest ``u`` at which ``1 / (u rho(u))`` is below ``level``, by bisection on
    ``log u``: the envelope falls strictly, from infinity at ``u = 0`` to zero."""
    low, high = -40.0, 40.0
    target = -math.log(level)
    while high - low > 1e-12:
        middle = 0.5 * (low + high)
        scaled = weights * math.exp(middle)
        height = 0.25 * float(np.log1p(scaled * scaled).sum()) + middle
        if height < target:
            low = middle
        else:
            high = middle
    return math.exp(high)


def _phase_and_envelope(weights: FloatArray, u: float) -> tuple[float, float]:
    """``(sum of arctan(w_i u)) / 2`` and ``exp(-log rho(u)) / u``, the latter zero where
    ``rho(u)`` is beyond the range of a float."""
    scaled = weights * u
    log_rho = 0.25 * float(np.log1p(scaled * scaled).sum())
    if not log_rho < 700.0:
        return 0.0, 0.0
    return 0.5 * float(np.arctan(scaled).sum()), math.exp(-log_rho) / u


def _mass_below_tolerance(weights: FloatArray, u: float) -> bool:
    """Whether at most the tolerance of the integrand's envelope is left beyond ``u``.

    The envelope is ``1 / (u rho(u))`` and ``rho(u)`` is at least
    ``(product of |w_i u|)**(1/2)``, so its integral beyond ``u`` is at most
    ``(2/m) (product of |w_i|)^(-1/2) u^(-m/2)``.
    """
    count = weights.size
    log_bound = math.log(2.0 / count) - 0.5 * float(np.log(np.abs(weights)).sum())
    log_bound -= 0.5 * count * math.log(u)
    return log_bound < math.log(_TOLERANCE)


def _plain(integrand: Callable[[float], float], low: float, high: float) -> float:
    return float(
        integrate.quad(integrand, low, high, epsabs=_TOLERANCE, epsrel=_TOLERANCE, limit=_LIMIT)[0]
    )


def _imhof_integral(weights: FloatArray, x: float, upper: float) -> float:
    """``integral of sin(theta(u)) / (u rho(u)) du`` from 0 to infinity, in parts.

    Up to ``upper`` the plain adaptive quadrature suffices. Beyond it the phase is what is
    left of ``sum of arctan(w_i u)`` (a slow drift) minus ``x u / 2``: without ``x`` the
    integrand is smooth to infinity and the infinite-range quadrature takes it; with ``x``
    :func:`_oscillating_tail` takes it. A quadrature that cannot reach the tolerance is an
    error, never a value passed off as exact.
    """
    half_x = 0.5 * x

    def integrand(u: float) -> float:
        if u == 0.0:
            return 0.5 * (float(weights.sum()) - x)
        phase, envelope = _phase_and_envelope(weights, u)
        if envelope == 0.0:
            return 0.0
        return math.sin(phase - half_x * u) * envelope

    def sine_of_phase(u: float) -> float:
        """The factor of ``cos(omega u)`` in the tail: ``sin(phase)`` times the envelope."""
        phase, envelope = _phase_and_envelope(weights, u)
        return math.sin(phase) * envelope

    def cosine_of_phase(u: float) -> float:
        """The factor of ``sin(omega u)`` in the tail: ``cos(phase)`` times the envelope."""
        phase, envelope = _phase_and_envelope(weights, u)
        return math.cos(phase) * envelope

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            total = _plain(integrand, 0.0, upper)
            if x == 0.0:
                total += _plain(integrand, upper, math.inf)
            else:
                total += _oscillating_tail(
                    weights, x, upper, integrand, sine_of_phase, cosine_of_phase
                )
        except IntegrationWarning as exc:
            raise ToolError(
                "INTEGRATION_NOT_CONVERGED", f"the p-value integral did not converge: {exc}"
            ) from exc
    return total


def _oscillating_tail(
    weights: FloatArray,
    x: float,
    start: float,
    integrand: Callable[[float], float],
    sine_of_phase: Callable[[float], float],
    cosine_of_phase: Callable[[float], float],
) -> float:
    """The tail of Imhof's integral once the term ``x u`` keeps it oscillating.

    A point far out in a tail draws its whole (tiny) chance from where the oscillation has
    switched on, far from the origin, so the tail cannot be dropped or taken in one stroke.
    The plain quadrature walks out in intervals that double, each of which it resolves, for
    as long as more than the tolerance of the envelope is left beyond. Where an interval
    would hold too many oscillations for it, the Fourier quadrature sums the rest from
    there: by then the start is more than a hundred of that quadrature's own intervals out,
    so its factor varies slowly across each of them.
    """
    total = 0.0
    begin = start
    period = 4.0 * math.pi / abs(x)
    while not _mass_below_tolerance(weights, begin):
        if begin > _CHUNK_OSCILLATIONS * period:
            # sin(phase - omega u) = sin(phase) cos(omega u) - cos(phase) sin(omega u), with
            # omega of either sign: only the sine changes with it.
            omega = 0.5 * abs(x)
            cosine = _fourier(sine_of_phase, "cos", begin, omega)
            sine = _fourier(cosine_of_phase, "sin", begin, omega)
            return total + cosine - math.copysign(1.0, x) * sine
        total += _plain(integrand, begin, 2.0 * begin)
        begin *= 2.0
    return total


def _fourier(factor: Callable[[float], float], kernel: str, start: float, omega: float) -> float:
    """``integral of factor(u) times the kernel from start to infinity``, by the weighted
    quadrature, which extrapolates over its own intervals of ``pi / omega``."""
    return float(
        integrate.quad(
            factor,
            start,
            math.inf,
            weight=kernel,
            wvar=omega,
            epsabs=_TOLERANCE,
            epsrel=_TOLERANCE,
            limlst=200,
            limit=200,
        )[0]
    )
