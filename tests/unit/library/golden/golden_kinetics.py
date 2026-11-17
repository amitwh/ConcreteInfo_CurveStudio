"""Independent NumPy reference implementations of the kinetics family.

Each callable is written from the mathematical definition (not from the expression
string of the model) and takes ``x`` and the parameters by keyword.
``test_catalogue_golden`` compares it with the model at the example parameters, so a
sign slip, a wrong power or a swapped parameter in an expression is caught.

The rate laws are written as the equations they come from:

* a first-order conversion is ``X = xinf*(1 - exp(-k*t))``;
* an n-th order conversion follows ``(1-X)**(1-n) = 1 + (n-1)*k*t`` with the half-time
  fixing ``(n-1)*k = (2**(n-1) - 1)/thalf``;
* the Bateman pair solves ``A -> B -> C`` with the rates ``k1`` and ``k1 + g``;
* the Deal-Grove film follows ``x**2 + A*x = B*(t + tau)``;
* the Cabrera-Mott film follows the logarithmic law ``x = k*log(1 + t/tau)``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

FloatArray = NDArray[np.float64]


def _first_order_conversion(x: FloatArray, *, xinf: float, k: float) -> FloatArray:
    return np.asarray(xinf * (1.0 - np.exp(-k * x)))


def _second_order_conversion(x: FloatArray, *, thalf: float) -> FloatArray:
    return np.asarray(x / (x + thalf))


def _nth_order_conversion(x: FloatArray, *, thalf: float, n: float) -> FloatArray:
    # (1 - X)**(1 - n) = 1 + (n - 1)*k*t and (n - 1)*k = (2**(n - 1) - 1)/thalf
    coefficient = (2.0 ** (n - 1.0) - 1.0) / thalf
    return np.asarray(1.0 - np.power(1.0 + coefficient * x, 1.0 / (1.0 - n)))


def _haldane(x: FloatArray, *, vmax: float, km: float, ki: float) -> FloatArray:
    return np.asarray(vmax * x / (km + x + x**2 / ki))


def _langmuir_hinshelwood(x: FloatArray, *, a: float, b: float, d: float) -> FloatArray:
    coverage_a = 1.0 / (1.0 + b * x)
    coverage_b = 1.0 / (1.0 + (b + d) * x)
    return np.asarray(a * x * x * coverage_a * coverage_b)


def _dissociative_adsorption(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # a dissociative adsorption covers two sites: the coverage follows sqrt(pressure)
    pressure = np.sqrt(x)
    return np.asarray(a * pressure / (1.0 + b * pressure))


def _bateman_intermediate(x: FloatArray, *, a0: float, k1: float, g: float) -> FloatArray:
    # [B] = a0 * k1/(k2 - k1) * (exp(-k1*t) - exp(-k2*t)) with k2 = k1 + g
    k2 = k1 + g
    return np.asarray(a0 * k1 / (k2 - k1) * (np.exp(-k1 * x) - np.exp(-k2 * x)))


def _bateman_product(x: FloatArray, *, a0: float, k1: float, g: float) -> FloatArray:
    # [C] = a0*(k2*(1 - exp(-k1*t)) - k1*(1 - exp(-k2*t)))/(k2 - k1) with k2 = k1 + g
    k2 = k1 + g
    return np.asarray(a0 * (k2 * (1.0 - np.exp(-k1 * x)) - k1 * (1.0 - np.exp(-k2 * x))) / g)


def _parabolic_oxidation(x: FloatArray, *, b: float, tau: float) -> FloatArray:
    return np.asarray(np.sqrt(2.0 * b * (x + tau)))


def _deal_grove(x: FloatArray, *, A: float, B: float, tau: float) -> FloatArray:
    # x**2 + A*x = B*(t + tau): the positive root of the quadratic
    return np.asarray((-A + np.sqrt(A**2 + 4.0 * B * (x + tau))) / 2.0)


def _cabrera_mott_log(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # x = k*log(1 + t/tau) with k = a and tau = 1/b, written with log1p
    return np.asarray(a * np.log1p(b * x))


def _power_exponential(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a * x**b * np.exp(-c * x))


def _butler_volmer(x: FloatArray, *, j0: float, a: float, b: float) -> FloatArray:
    return np.asarray(j0 * (np.exp(a * x) - np.exp(-b * x)))


def _two_exponential_decay(x: FloatArray, *, a: float, b: float, c: float, g: float) -> FloatArray:
    return np.asarray(a * np.exp(-b * x) + c * np.exp(-(b + g) * x))


def _error_function_diffusion(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    return np.asarray(a - b * erf(c * x))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "kinetics_first_order_conversion": _first_order_conversion,
    "kinetics_second_order_conversion": _second_order_conversion,
    "kinetics_nth_order_conversion": _nth_order_conversion,
    "kinetics_haldane_substrate_inhibition": _haldane,
    "kinetics_langmuir_hinshelwood_bimolecular": _langmuir_hinshelwood,
    "kinetics_dissociative_adsorption": _dissociative_adsorption,
    "kinetics_bateman_intermediate": _bateman_intermediate,
    "kinetics_bateman_product": _bateman_product,
    "kinetics_parabolic_oxidation": _parabolic_oxidation,
    "kinetics_deal_grove_oxidation": _deal_grove,
    "kinetics_cabrera_mott_oxidation": _cabrera_mott_log,
    "kinetics_power_exponential": _power_exponential,
    "kinetics_butler_volmer": _butler_volmer,
    "kinetics_two_exponential_decay": _two_exponential_decay,
    "kinetics_error_function_diffusion": _error_function_diffusion,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "kinetics_haldane_substrate_inhibition": lambda x, *, vmax, km, ki: np.asarray(
        km + x + x**2 / ki
    ),
    "kinetics_langmuir_hinshelwood_bimolecular": lambda x, *, a, b, d: np.asarray(
        (1.0 + b * x) * (1.0 + (b + d) * x)
    ),
    "kinetics_bateman_intermediate": lambda x, *, a0, k1, g: np.asarray(g * np.ones_like(x)),
    "kinetics_bateman_product": lambda x, *, a0, k1, g: np.asarray(g * np.ones_like(x)),
    "kinetics_nth_order_conversion": lambda x, *, thalf, n: np.asarray(
        1.0 + (2.0 ** (n - 1.0) - 1.0) * x / thalf
    ),
}
