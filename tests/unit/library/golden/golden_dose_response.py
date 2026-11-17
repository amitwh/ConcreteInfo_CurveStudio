"""Independent NumPy reference implementations of the dose-response family.

Each callable is written from the mathematical definition (not from the expression string of
the model) and takes ``x`` and the parameters by keyword. The log-logistic models use the dose
``x`` directly, exactly as the catalogue expressions do; no model is fitted on a transformed
response scale.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

FloatArray = NDArray[np.float64]


def _ll2(x: FloatArray, *, e: float, b: float) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.power(x / e, b)))


def _ll3(x: FloatArray, *, d: float, e: float, b: float) -> FloatArray:
    return np.asarray(d / (1.0 + np.power(x / e, b)))


def _four_pl(x: FloatArray, *, c: float, d: float, e: float, b: float) -> FloatArray:
    return np.asarray(c + (d - c) / (1.0 + np.power(x / e, b)))


def _five_pl(x: FloatArray, *, c: float, d: float, e: float, b: float, f: float) -> FloatArray:
    return np.asarray(c + (d - c) / np.power(1.0 + np.power(x / e, b), f))


def _weibull1_3(x: FloatArray, *, d: float, e: float, b: float) -> FloatArray:
    return np.asarray(d * np.exp(-np.power(x / e, b)))


def _emax(x: FloatArray, *, e0: float, emax: float, ec50: float) -> FloatArray:
    return np.asarray(e0 + emax * x / (ec50 + x))


def _inhibitory_emax(x: FloatArray, *, e0: float, imax: float, ic50: float) -> FloatArray:
    return np.asarray(e0 - imax * x / (ic50 + x))


def _brain_cousens(
    x: FloatArray, *, b: float, c: float, d: float, e: float, f: float
) -> FloatArray:
    return np.asarray(c + (d - c + f * x) / (1.0 + np.power(x / e, b)))


def _cedergreen(
    x: FloatArray, *, a: float, b: float, c: float, d: float, e: float, f: float
) -> FloatArray:
    return np.asarray(c + (d - c + f * np.exp(-1.0 / np.power(x, a))) / (1.0 + np.power(x / e, b)))


def _lognormal(x: FloatArray, *, c: float, d: float, mu: float, sigma: float) -> FloatArray:
    z = (np.log(x) - mu) / (sigma * np.sqrt(2.0))
    return np.asarray(c + (d - c) * 0.5 * (1.0 + erf(z)))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "dose_response_ll2": _ll2,
    "dose_response_ll3": _ll3,
    "dose_response_4pl": _four_pl,
    "dose_response_5pl": _five_pl,
    "dose_response_weibull1_3": _weibull1_3,
    "dose_response_emax": _emax,
    "dose_response_inhibitory_emax": _inhibitory_emax,
    "dose_response_brain_cousens": _brain_cousens,
    "dose_response_cedergreen": _cedergreen,
    "dose_response_lognormal": _lognormal,
}
