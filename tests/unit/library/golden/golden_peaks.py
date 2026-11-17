"""Independent NumPy reference implementations of the peak family.

Each callable is written from the mathematical definition of the profile (not from the
expression string of the model) and takes ``x`` and the parameters by keyword.
``test_catalogue_golden`` compares it with the model at the example parameters, so a sign
slip, a wrong power or a swapped parameter in an expression is caught.

Pattern for a new family: add ``golden/golden_<family>.py`` (any ``golden_*.py`` is
discovered) with ``GOLDEN: dict[str, Callable[..., np.ndarray]]`` keyed by model id.
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.special as sps
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
"""Full width at half maximum of a Gaussian divided by its standard deviation."""


def _gaussian_offset(x: FloatArray, *, c: float, a: float, mu: float, s: float) -> FloatArray:
    """A Gaussian of height ``a`` and standard deviation ``s`` on the baseline ``c``."""
    z = (np.asarray(x, dtype=np.float64) - mu) / s
    return np.asarray(c + a * np.exp(-0.5 * z**2))


def _lorentzian_centered(x: FloatArray, *, a: float, g: float) -> FloatArray:
    """A Lorentzian of height ``a`` and half width at half maximum ``g``, centred on zero."""
    u = np.asarray(x, dtype=np.float64) / g
    return np.asarray(a / (1.0 + u**2))


def _lorentzian(x: FloatArray, *, a: float, mu: float, g: float) -> FloatArray:
    """A Lorentzian of height ``a`` and half width at half maximum ``g`` centred on ``mu``."""
    u = (np.asarray(x, dtype=np.float64) - mu) / g
    return np.asarray(a / (1.0 + u**2))


def _lorentzian_offset(x: FloatArray, *, c: float, a: float, mu: float, g: float) -> FloatArray:
    """A Lorentzian peak on the baseline ``c``."""
    u = (np.asarray(x, dtype=np.float64) - mu) / g
    return np.asarray(c + a / (1.0 + u**2))


def _pearson_vii(x: FloatArray, *, a: float, mu: float, g: float, nu: float) -> FloatArray:
    """The Pearson VII profile: a peak of height ``a`` whose tails fall off as a power law."""
    u = (np.asarray(x, dtype=np.float64) - mu) / g
    return np.asarray(a * (1.0 + u**2) ** (-nu))


def _pseudo_voigt(x: FloatArray, *, a: float, mu: float, w: float, eta: float) -> FloatArray:
    """A Gaussian and a Lorentzian of the same full width at half maximum, mixed by ``eta``."""
    xs = np.asarray(x, dtype=np.float64)
    sigma = w / _FWHM_PER_SIGMA
    gauss = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    half = w / 2.0
    lorentz = 1.0 / (1.0 + ((xs - mu) / half) ** 2)
    return np.asarray(a * ((1.0 - eta) * gauss + eta * lorentz))


def _voigt(x: FloatArray, *, a: float, mu: float, sigma: float, gamma: float) -> FloatArray:
    """The area-normalised Voigt profile, written with the Faddeeva function."""
    xs = np.asarray(x, dtype=np.float64)
    z = (xs - mu + 1j * gamma) / (sigma * math.sqrt(2.0))
    profile = np.real(sps.wofz(z)) / (sigma * math.sqrt(2.0 * math.pi))
    return np.asarray(a * profile)


def _emg(x: FloatArray, *, a: float, mu: float, sigma: float, tau: float) -> FloatArray:
    """The exponentially modified Gaussian, written with the normal cumulative function.

    A Gaussian of mean ``mu`` and standard deviation ``sigma`` convolved with an
    exponential of mean ``tau``; the closed form uses the normal CDF.
    """
    xs = np.asarray(x, dtype=np.float64)
    z = (xs - mu - sigma**2 / tau) / sigma
    cdf = sps.ndtr(z)
    density = np.exp(-(xs - mu) / tau + sigma**2 / (2.0 * tau**2)) * cdf / tau
    return np.asarray(a * density)


def _skew_normal(x: FloatArray, *, a: float, mu: float, s: float, alpha: float) -> FloatArray:
    """The skew-normal density of Azzalini: ``2/s`` times the normal density times Phi.

    The CDF is written with ``erf``, the way the model writes it: in the far left tail
    the difference between a CDF and its complement loses its digits, so the two
    implementations have to round alike to be compared.
    """
    xs = np.asarray(x, dtype=np.float64)
    z = (xs - mu) / s
    normal = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + sps.erf(alpha * (xs - mu) / (math.sqrt(2.0) * s)))
    return np.asarray(a * 2.0 * normal * cdf / s)


def _split_gaussian(x: FloatArray, *, a: float, mu: float, s1: float, s2: float) -> FloatArray:
    """A Gaussian peak with the standard deviation ``s1`` below ``mu`` and ``s2`` above it."""
    xs = np.asarray(x, dtype=np.float64)
    left = np.where(xs < mu, (xs - mu) ** 2 / (2.0 * s1**2), 0.0)
    right = np.where(xs >= mu, (xs - mu) ** 2 / (2.0 * s2**2), 0.0)
    return np.asarray(a * np.exp(-(left + right)))


def _power_tail(x: FloatArray, *, a: float, mu: float, w: float, b: float) -> FloatArray:
    """A peak whose height falls off as the power ``b`` of the distance from ``mu``."""
    u = np.abs((np.asarray(x, dtype=np.float64) - mu) / w)
    return np.asarray(a / (1.0 + u) ** b)


def _two_gaussian(
    x: FloatArray, *, a1: float, mu1: float, s1: float, a2: float, d: float, s2: float
) -> FloatArray:
    """Two Gaussian peaks; the second one stands at ``mu1 + d``."""
    xs = np.asarray(x, dtype=np.float64)
    first = a1 * np.exp(-0.5 * ((xs - mu1) / s1) ** 2)
    second = a2 * np.exp(-0.5 * ((xs - (mu1 + d)) / s2) ** 2)
    return np.asarray(first + second)


def _three_gaussian(
    x: FloatArray,
    *,
    a1: float,
    mu1: float,
    s1: float,
    a2: float,
    d2: float,
    s2: float,
    a3: float,
    d3: float,
    s3: float,
) -> FloatArray:
    """Three Gaussian peaks, in order of position."""
    xs = np.asarray(x, dtype=np.float64)
    first = a1 * np.exp(-0.5 * ((xs - mu1) / s1) ** 2)
    second = a2 * np.exp(-0.5 * ((xs - (mu1 + d2)) / s2) ** 2)
    third = a3 * np.exp(-0.5 * ((xs - (mu1 + d2 + d3)) / s3) ** 2)
    return np.asarray(first + second + third)


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "peak_gaussian_offset": _gaussian_offset,
    "lorentzian_centered": _lorentzian_centered,
    "peak_lorentzian": _lorentzian,
    "peak_lorentzian_offset": _lorentzian_offset,
    "peak_pearson_vii": _pearson_vii,
    "peak_pseudo_voigt": _pseudo_voigt,
    "peak_voigt": _voigt,
    "peak_emg": _emg,
    "peak_skew_normal": _skew_normal,
    "peak_split_gaussian": _split_gaussian,
    "peak_power_tail": _power_tail,
    "peak_two_gaussian": _two_gaussian,
    "peak_three_gaussian": _three_gaussian,
}
