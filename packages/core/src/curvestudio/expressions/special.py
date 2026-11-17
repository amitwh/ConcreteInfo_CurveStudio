"""SymPy functions that SymPy itself lacks, with analytic derivatives and NumPy callables.

* :class:`voigt` is the area-normalised Voigt profile (the convolution of a Gaussian with
  standard deviation ``sigma`` and a Lorentzian with scale ``gamma``).
* ``min`` and ``max`` (classes :class:`_Min`, :class:`_Max`) are two-argument functions that
  compile to ``numpy.minimum`` / ``numpy.maximum``, so they broadcast however deeply they
  are nested. Their derivative selects the active argument; at an exact tie each argument
  gets ``0.5`` (the mid-point of the sub-gradient), see :class:`_OrderStep`.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.special as sps
import sympy
from numpy.typing import NDArray

_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)

_FAR_FIELD_RADIUS = 10.0
"""``|z|`` from which the Faddeeva derivatives use their asymptotic series (see below)."""

_SERIES_TERMS = 30


def _series_coefficients(extra: Callable[[int], float]) -> NDArray[np.float64]:
    """``c_n * extra(n)`` with ``c_n = (2n+1)!! / 2**n`` (c_0 = 1), low order first."""
    coefficients = []
    c = 1.0
    for n in range(_SERIES_TERMS):
        if n:
            c *= (2 * n + 1) / 2.0
        coefficients.append(c * extra(n))
    return np.array(coefficients)


_P_COEFFICIENTS = _series_coefficients(lambda n: 1.0)
_R_COEFFICIENTS = _series_coefficients(lambda n: 2.0 * n + 2.0)


def voigt_value(x: Any, sigma: Any, gamma: Any) -> NDArray[np.float64]:
    """Area-normalised Voigt profile (``scipy.special.voigt_profile``)."""
    return np.asarray(sps.voigt_profile(x, sigma, gamma), dtype=np.float64)


def voigt_partial_value(x: Any, sigma: Any, gamma: Any, index: Any) -> NDArray[np.float64]:
    """Partial derivative of :func:`voigt_value` with respect to argument ``index`` (0, 1 or 2).

    With ``w`` the Faddeeva function and ``z = (x + i gamma) / (sigma sqrt 2)`` the profile is
    ``V = Re w(z) / (sigma sqrt(2 pi))`` and ``w'(z) = -2 z w(z) + 2i / sqrt(pi)``. The partials are
    ``V_x = Re w' / (2 sigma^2 sqrt(pi))``, ``V_gamma = -Im w' / (2 sigma^2 sqrt(pi))`` and (the
    heat equation of Gaussian smoothing) ``V_sigma = sigma V_xx = Re w'' / (2 sigma^2 sqrt(2 pi))``
    with ``w'' = -2 w - 2 z w'``.

    For ``|z| >= 10`` the two terms of ``w'`` cancel almost completely, so the partials come from
    the asymptotic series ``w'(z) = -(i / sqrt(pi)) sum_n (2n+1)!! / 2^n z^-(2n+2)`` instead,
    written in ``s = x + i gamma`` so nothing is divided by ``sigma`` (``sigma = 0`` gives the
    Lorentzian partials exactly). Both branches agree to better than 1e-6 relative error
    wherever the partial is not negligible.
    """
    which = int(index)
    xs, sig, gam = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (x, sigma, gamma))
    )
    with np.errstate(all="ignore"):
        s = xs + 1j * gam
        far = (s.real**2 + s.imag**2) >= 2.0 * _FAR_FIELD_RADIUS**2 * sig**2

        # Far field: series in u = 2 sigma^2 / s^2 (|u| <= 1/100 here).
        u = np.where(far, 2.0 * sig**2 / s**2, 0.0)
        p = np.polynomial.polynomial.polyval(u, _P_COEFFICIENTS)
        if which == 0:
            far_value = (p / s**2).imag / math.pi
        elif which == 2:
            far_value = (p / s**2).real / math.pi
        else:
            r = np.polynomial.polynomial.polyval(u, _R_COEFFICIENTS)
            far_value = -sig * (r / s**3).imag / math.pi

        # Near field: direct evaluation through the Faddeeva function.
        safe_sigma = np.where(far, 1.0, sig)
        z = np.where(far, 1.0 + 0.0j, s / (safe_sigma * _SQRT2))
        w = sps.wofz(z)
        w1 = -2.0 * z * w + 2j / _SQRT_PI
        if which == 0:
            near_value = w1.real / (2.0 * safe_sigma**2 * _SQRT_PI)
        elif which == 2:
            near_value = -w1.imag / (2.0 * safe_sigma**2 * _SQRT_PI)
        else:
            w2 = -2.0 * w - 2.0 * z * w1
            near_value = w2.real / (2.0 * safe_sigma**2 * _SQRT_2PI)
        out = np.where(far, far_value, near_value)
    return np.asarray(out, dtype=np.float64)


class voigt_partial(sympy.Function):  # type: ignore[misc]  # SymPy is untyped
    """``d voigt(x, sigma, gamma) / d argument[index]``; ``index`` is 0, 1 or 2."""

    nargs = 4

    def _eval_is_real(self) -> bool:
        return True


class voigt(sympy.Function):  # type: ignore[misc]  # SymPy is untyped
    """Area-normalised Voigt profile ``voigt(x, sigma, gamma)``, differentiable in all arguments.

    Defined for ``sigma > 0`` and ``gamma >= 0`` (``sigma = 0`` is the Lorentzian of scale
    ``gamma``, ``gamma = 0`` the Gaussian of standard deviation ``sigma``); other values give
    non-finite results, never an exception.
    """

    nargs = 3

    def fdiff(self, argindex: int = 1) -> Any:
        if argindex not in (1, 2, 3):
            raise sympy.ArgumentIndexError(self, argindex)
        x, sigma, gamma = self.args
        return voigt_partial(x, sigma, gamma, sympy.Integer(argindex - 1))

    def _eval_is_real(self) -> bool:
        return True


class order_step(sympy.Function):  # type: ignore[misc]  # SymPy is untyped
    """``1`` for a positive argument, ``0`` for a negative one and ``0.5`` at exactly zero.

    The derivative selector of ``min`` and ``max``: ``d min(a, b) / da = order_step(b - a)``.
    It is piecewise constant, so its own derivative is zero.
    """

    nargs = 1

    @classmethod
    def eval(cls, argument: Any) -> Any:
        if argument.is_Number:
            if argument > 0:
                return sympy.Integer(1)
            return sympy.Integer(0) if argument < 0 else sympy.Rational(1, 2)
        return None

    def fdiff(self, argindex: int = 1) -> Any:
        return sympy.Integer(0)

    def _eval_is_real(self) -> bool:
        return True


class _Min(sympy.Function):  # type: ignore[misc]  # SymPy is untyped
    """``min(a, b)``; ``d/da = order_step(b - a)`` and ``d/db = order_step(a - b)``."""

    nargs = 2

    @classmethod
    def eval(cls, a: Any, b: Any) -> Any:
        if a == b:
            return a
        if a.is_Number and b.is_Number:
            return a if a < b else b
        return None

    def fdiff(self, argindex: int = 1) -> Any:
        a, b = self.args
        if argindex == 1:
            return order_step(b - a)
        if argindex == 2:
            return order_step(a - b)
        raise sympy.ArgumentIndexError(self, argindex)

    def _eval_is_real(self) -> bool:
        return True


class _Max(sympy.Function):  # type: ignore[misc]  # SymPy is untyped
    """``max(a, b)``; ``d/da = order_step(a - b)`` and ``d/db = order_step(b - a)``."""

    nargs = 2

    @classmethod
    def eval(cls, a: Any, b: Any) -> Any:
        if a == b:
            return a
        if a.is_Number and b.is_Number:
            return a if a > b else b
        return None

    def fdiff(self, argindex: int = 1) -> Any:
        a, b = self.args
        if argindex == 1:
            return order_step(a - b)
        if argindex == 2:
            return order_step(b - a)
        raise sympy.ArgumentIndexError(self, argindex)

    def _eval_is_real(self) -> bool:
        return True


# SymPy prints a function by its class name, which must be the name of the compiled callable.
_Min.__name__ = "min"
_Max.__name__ = "max"
minimum_function = _Min
maximum_function = _Max


def _order_step_value(argument: Any) -> Any:
    return np.heaviside(argument, 0.5)


NUMPY_CALLABLES: dict[str, Callable[..., Any]] = {
    "voigt": voigt_value,
    "voigt_partial": voigt_partial_value,
    "min": np.minimum,
    "max": np.maximum,
    "order_step": _order_step_value,
}
"""Namespace entries ``sympy.lambdify`` needs to evaluate the functions above."""
