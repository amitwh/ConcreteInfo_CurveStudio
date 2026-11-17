"""Independent NumPy reference implementations of the physics family.

Each callable is written from the mathematical definition (not from the expression
string of the model) and takes ``x`` and the parameters by keyword.
``test_catalogue_golden`` compares it with the model at the example parameters, so a
sign slip, a wrong power or a swapped parameter in an expression is caught.

The definitions used are the usual ones: the Arrhenius and Eyring rate laws, the
Einstein heat capacity, the Antoine vapour pressure, the one- and two-term Sellmeier
and the Cauchy dispersion formulas, the Stefan-Boltzmann and Planck radiation laws,
the Vogel-Fulcher-Tammann viscosity, the Debye dielectric relaxation, the
Maxwell-Boltzmann speed distribution and the Fermi-Dirac and Bose-Einstein
occupations.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _arrhenius(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # k = A*exp(-E/(R*T)) with the activation temperature b = E/R
    return np.asarray(a * np.exp(-b / x))


def _eyring(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # k = (kB*T/h)*exp(DS/R)*exp(-DH/(R*T))
    return np.asarray(a * x * np.exp(-b / x))


def _einstein(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # C = 3R*(theta/T)**2*exp(theta/T)/(exp(theta/T) - 1)**2, written with expm1
    ratio = b / x
    return np.asarray(a * ratio**2 * np.exp(ratio) / np.expm1(ratio) ** 2)


def _antoine(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # ln(P) = a - b/(T + c)
    return np.asarray(np.exp(a - b / (c + x)))


def _sellmeier(x: FloatArray, *, b: float, c: float) -> FloatArray:
    # n**2 = 1 + B*lambda**2/(lambda**2 - C)
    return np.asarray(np.sqrt(1.0 + b * x**2 / (x**2 - c)))


def _cauchy(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # n = A + B/lambda**2 + C/lambda**4
    return np.asarray(a + b / x**2 + c / x**4)


def _wien(x: FloatArray, *, b: float) -> FloatArray:
    # lambda_max*T = b
    return np.asarray(b / x)


def _stefan_boltzmann(x: FloatArray, *, a: float) -> FloatArray:
    # P = sigma*T**4
    return np.asarray(a * x**4)


def _planck(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # B_lambda(T) = (2*h*c**2/lambda**5)/(exp(h*c/(lambda*k*T)) - 1)
    return np.asarray(a / (np.exp(b / x) - 1.0))


def _vogel_fulcher_tammann(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # eta = eta0*exp(B/(T - T0)) with eta0 = exp(a)
    return np.asarray(np.exp(a) * np.exp(b / (x - c)))


def _debye_dielectric(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # eps(w) = eps_inf + deps/(1 + (w*tau)**2) with deps = b/c and tau = 1/c
    increment = b / c**2
    return np.asarray(a + increment / (1.0 + (x / c) ** 2))


def _maxwell_boltzmann(x: FloatArray, *, a: float, b: float) -> FloatArray:
    # f(v) proportional to v**2*exp(-m*v**2/(2*k*T))
    return np.asarray(a * x**2 * np.exp(-b * x**2))


def _lorentzian(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # the height of a damped oscillator: A/(1 + ((x - x0)/gamma)**2)
    return np.asarray(a / (1.0 + ((x - c) / b) ** 2))


def _sellmeier_two(x: FloatArray, *, b: float, c: float, d: float, g: float) -> FloatArray:
    # n**2 = 1 + B1*lambda**2/(lambda**2 - C1) + B2*lambda**2/(lambda**2 - C2), C2 = c + g
    squared = x**2
    return np.asarray(np.sqrt(1.0 + b * squared / (squared - c) + d * squared / (squared - c - g)))


def _fermi_dirac(x: FloatArray, *, c: float, d: float) -> FloatArray:
    # f = 1/(exp((E - mu)/(k*T)) + 1)
    return np.asarray(1.0 / (np.exp((x - c) / d) + 1.0))


def _bose_einstein(x: FloatArray, *, c: float, d: float) -> FloatArray:
    # n = 1/(exp((E - mu)/(k*T)) - 1)
    return np.asarray(1.0 / (np.exp((x - c) / d) - 1.0))


def _moseley(x: FloatArray, *, a: float, c: float) -> FloatArray:
    # nu = K*(Z - sigma)**2
    return np.asarray(a * np.square(x - c))


def _van_der_waals(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # P = n*R*T/(V - n*b) - a*n**2/V**2
    return np.asarray(a / (x - b) - c / x**2)


def _coulomb(x: FloatArray, *, a: float) -> FloatArray:
    # the field of a point source falls with the square of the distance
    return np.asarray(a / x**2)


def _lorentz_oscillator(x: FloatArray, *, a: float, b: float, c: float) -> FloatArray:
    # eps = eps_inf + S/(omega0**2 - omega**2)
    return np.asarray(a + b / (c**2 - x**2))


GOLDEN: dict[str, Callable[..., FloatArray]] = {
    "physics_arrhenius": _arrhenius,
    "physics_eyring": _eyring,
    "physics_einstein_heat_capacity": _einstein,
    "physics_antoine": _antoine,
    "physics_sellmeier_1term": _sellmeier,
    "physics_cauchy_dispersion": _cauchy,
    "physics_wien_displacement": _wien,
    "physics_stefan_boltzmann": _stefan_boltzmann,
    "physics_planck_radiance": _planck,
    "physics_vogel_fulcher_tammann": _vogel_fulcher_tammann,
    "physics_debye_dielectric": _debye_dielectric,
    "physics_maxwell_boltzmann_speed": _maxwell_boltzmann,
    "physics_lorentzian_resonance": _lorentzian,
    "physics_sellmeier_2term": _sellmeier_two,
    "physics_fermi_dirac_occupation": _fermi_dirac,
    "physics_bose_einstein_occupation": _bose_einstein,
    "physics_moseley_law": _moseley,
    "physics_van_der_waals": _van_der_waals,
    "physics_coulomb_field": _coulomb,
    "physics_lorentz_oscillator": _lorentz_oscillator,
}

DENOMINATORS: dict[str, Callable[..., FloatArray]] = {
    "physics_sellmeier_1term": lambda x, *, b, c: np.asarray(x**2 - c),
    "physics_sellmeier_2term": lambda x, *, b, c, d, g: np.asarray((x**2 - c) * (x**2 - c - g)),
    "physics_antoine": lambda x, *, a, b, c: np.asarray(c + x),
    "physics_vogel_fulcher_tammann": lambda x, *, a, b, c: np.asarray(x - c),
    "physics_van_der_waals": lambda x, *, a, b, c: np.asarray(x - b),
    "physics_lorentz_oscillator": lambda x, *, a, b, c: np.asarray(c**2 - x**2),
}
