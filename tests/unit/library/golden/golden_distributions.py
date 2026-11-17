"""Independent NumPy reference implementations of the distribution family.

Each callable is written from the mathematical definition of the distribution (not from the
expression string of the model) and takes ``x`` and the parameters by keyword. Where
SciPy implements the density, the reference goes through that implementation, so the
comparison checks the model's own formula rather than repeating it.

Pattern for a new family: add ``golden/golden_<family>.py`` (any ``golden_*.py`` is
discovered) with ``GOLDEN: dict[str, Callable[..., np.ndarray]]`` keyed by model id.
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.special as sps
import scipy.stats as st
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _lognormal(x: FloatArray, *, a: float, m: float, s: float) -> FloatArray:
    """The log-normal density with median ``m`` and log-standard deviation ``s``."""
    return np.asarray(a * st.lognorm.pdf(np.asarray(x, dtype=np.float64), s=s, scale=m))


def _gamma(x: FloatArray, *, a: float, k: float, t: float) -> FloatArray:
    """The gamma density kernel ``x**(k-1)*exp(-x/t)``, scaled by ``a``.

    The kernel is the normalised gamma density times its normalising factor
    ``Gamma(k)*t**k``.
    """
    density = st.gamma.pdf(np.asarray(x, dtype=np.float64), a=k, scale=t)
    return np.asarray(a * math.gamma(k) * t**k * density)


def _weibull(x: FloatArray, *, a: float, k: float, t: float) -> FloatArray:
    """The Weibull density with shape ``k`` and scale ``t``."""
    return np.asarray(a * st.weibull_min.pdf(np.asarray(x, dtype=np.float64), c=k, scale=t))


def _cauchy(x: FloatArray, *, a: float, m: float, g: float) -> FloatArray:
    """The Cauchy density with location ``m`` and scale ``g``."""
    return np.asarray(a * st.cauchy.pdf(np.asarray(x, dtype=np.float64), loc=m, scale=g))


def _levy(x: FloatArray, *, a: float, c: float) -> FloatArray:
    """The Levy density with location zero and scale ``c``.

    The classic Levy density is the Levy-stable distribution with ``alpha = 1/2`` and
    ``beta = 1``, which is how SciPy offers it.
    """
    return np.asarray(a * st.levy_stable.pdf(np.asarray(x, dtype=np.float64), 0.5, 1.0, scale=c))


def _rayleigh(x: FloatArray, *, a: float, s: float) -> FloatArray:
    """The Rayleigh density with scale ``s``."""
    return np.asarray(a * st.rayleigh.pdf(np.asarray(x, dtype=np.float64), scale=s))


def _maxwell(x: FloatArray, *, a: float, s: float) -> FloatArray:
    """The Maxwell-Boltzmann density with scale ``s``."""
    return np.asarray(a * st.maxwell.pdf(np.asarray(x, dtype=np.float64), scale=s))


def _beta(x: FloatArray, *, a: float, p: float, q: float) -> FloatArray:
    """The beta density kernel ``x**(p-1)*(1-x)**(q-1)``, scaled by ``a``.

    The kernel is the normalised beta density times the beta function ``B(p,q)``.
    """
    density = st.beta.pdf(np.asarray(x, dtype=np.float64), a=p, b=q)
    return np.asarray(a * sps.beta(p, q) * density)


def _logistic(x: FloatArray, *, a: float, m: float, s: float) -> FloatArray:
    """The logistic density with location ``m`` and scale ``s``."""
    return np.asarray(a * st.logistic.pdf(np.asarray(x, dtype=np.float64), loc=m, scale=s))


def _gumbel(x: FloatArray, *, a: float, m: float, t: float) -> FloatArray:
    """The Gumbel (type I extreme value) density with location ``m`` and scale ``t``."""
    return np.asarray(a * st.gumbel_r.pdf(np.asarray(x, dtype=np.float64), loc=m, scale=t))


def _laplace(x: FloatArray, *, a: float, m: float, t: float) -> FloatArray:
    """The Laplace density with location ``m`` and scale ``t``."""
    return np.asarray(a * st.laplace.pdf(np.asarray(x, dtype=np.float64), loc=m, scale=t))


def _inverse_gaussian(x: FloatArray, *, a: float, m: float, lam: float) -> FloatArray:
    """The inverse Gaussian (Wald) density with mean ``m`` and shape ``lam``.

    The density is the first passage time of a Brownian motion with drift: its mean is
    ``m`` and its variance ``m**3/lam``.
    """
    xs = np.asarray(x, dtype=np.float64)
    deviation = (xs - m) / np.sqrt(xs)
    density = np.sqrt(lam / (2.0 * math.pi)) * np.exp(-0.5 * lam * deviation**2 / m**2) / xs**1.5
    return np.asarray(a * density)


def _normal(x: FloatArray, *, a: float, m: float, s: float) -> FloatArray:
    """The normal density with mean ``m`` and standard deviation ``s``."""
    return np.asarray(a * st.norm.pdf(np.asarray(x, dtype=np.float64), loc=m, scale=s))


def _log_logistic(x: FloatArray, *, a: float, m: float, k: float) -> FloatArray:
    """The log-logistic (Fisk) density with scale ``m`` and shape ``k``."""
    xs = np.asarray(x, dtype=np.float64)
    u = xs / m
    return np.asarray(a * k * u ** (k - 1.0) / (m * (1.0 + u**k) ** 2))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "distribution_lognormal": _lognormal,
    "distribution_gamma": _gamma,
    "distribution_weibull": _weibull,
    "distribution_cauchy": _cauchy,
    "distribution_levy": _levy,
    "distribution_rayleigh": _rayleigh,
    "distribution_maxwell_boltzmann": _maxwell,
    "distribution_beta": _beta,
    "distribution_logistic": _logistic,
    "distribution_gumbel": _gumbel,
    "distribution_laplace": _laplace,
    "distribution_inverse_gaussian": _inverse_gaussian,
    "distribution_normal": _normal,
    "distribution_log_logistic": _log_logistic,
}
