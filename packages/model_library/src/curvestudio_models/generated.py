"""Generated model families, built lazily by id and registered with ``register_family``.

Nothing here is compiled until a model is asked for (``get_model``); the id table itself is
cheap. Every generated model is fitted on the **original response scale**: a TableCurve-style
``y = 1/(a + b*ln(x))`` is written as that expression, never fitted as a line in ``1/y``, so its
AICc and BIC stay comparable with every other candidate. Straight-line fits in a transformed
response are used only for the *initial guess*.

Stable ids (an id never changes meaning; ``k`` is the parameter count)
----------------------------------------------------------------------

=============================  =====================================================  ==========
id                             equation                                               k
=============================  =====================================================  ==========
``rat_{m}_{n}``                ``(a0 + .. + am*x**m) / (1 + c1*x + .. + cn*x**n)``    m + n + 1
``invpoly_{d}``                ``1 / (a0 + a1*x + .. + ad*x**d)``                     d + 1
``recip_{d}``                  ``a0 + a1/x + .. + ad/x**d``                           d + 1
``exp_sum_{n}``                ``sum a_i*exp(-r_i*x)``, ``r_i = r_(i-1) + d_i``       2n
``exp_sum_{n}_offset``         the same plus a constant ``c``                         2n + 1
``gauss_sum_{n}``              ``sum a_i*exp(-(x-mu_i)**2/(2*s_i**2))``, ordered mu   3n
``pow_sum_{n}``                ``sum a_i*x**p_i``, ``p_i = p_(i-1) + d_i``             2n
``fourier_{n}``                ``a0 + sum a_i*cos(i*w*x) + b_i*sin(i*w*x)``           2n + 2
``tc_{y}_{x}``                 ``y = T(a + b*f(x))``                                   2
``tc3_{y}_{x}_{x2}``           ``y = T(a + b*f(x) + c*g(x))``                          3
=============================  =====================================================  ==========

Ranges: ``rat`` m, n >= 1 with m + n + 1 <= 10; ``invpoly`` d = 2..9; ``recip`` d = 3..9;
``exp_sum`` and ``exp_sum_offset`` n = 2, 3; ``gauss_sum`` and ``pow_sum`` n = 2, 3; ``fourier``
n = 1..6. ``y`` is the transform ``T`` of the linear predictor: ``id`` (``eta``), ``inv``
(``1/eta``), ``exp`` (``exp(eta)``), ``sqr`` (``eta**2``) or ``sqrtof`` (``sqrt(eta)``). ``x`` and
``x2`` name basis functions: ``x`` (x), ``inv`` (1/x), ``ln`` (ln x), ``sqrt`` (sqrt x),
``sqr`` (x**2) and ``expneg`` (exp(-x)) are the *base* set; ``cube`` (x**3), ``invsqr`` (1/x**2),
``lnsq`` (ln(x)**2), ``xlnx`` (x*ln(x)) and ``invsqrt`` (1/sqrt x) extend it. ``tc3`` takes every
unordered pair of two different basis functions, first the one that comes earlier in the order
above. ``sqr`` bounds ``a >= 0`` (``(-eta)**2 = eta**2``: the sign is otherwise not identifiable).

Listings (``ids(exhaustive)``): the default excludes every model with more than 5 parameters and
every ``tc3`` model that uses an extended basis function; exhaustive raises the parameter cap to
10 (``fourier_5``, ``fourier_6`` and the longest ``rat`` / ``invpoly`` / ``recip`` ids are
therefore reachable only through their id).

Not generated, because an existing model is the same curve (asking for the id raises ``KeyError``
naming it): ``rat_1_1``, ``rat_2_1``, ``rat_1_2`` (``rational_*``); ``rat_0_n`` is ``invpoly_n``
(``m`` starts at 1); ``invpoly_1``, ``recip_1``, ``recip_2`` and the ``tc`` / ``tc3`` ids listed in
``_EQUAL_TO``. Polynomials are not generated at all (``fit_polynomial``).

Exchangeable terms (sums of exponentials, Gaussians and powers) use an ordered parametrisation
(``r2 = r1 + d2`` with ``d2 >= 0``) so a fit has one label order, not ``n!`` equal optima.
"""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, partial
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import (
    ModelExample,
    ModelSpec,
    Parameter,
    register_family,
    registry,
    rules_guess,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]

DEFAULT_PARAMETER_CAP = 5
EXHAUSTIVE_PARAMETER_CAP = 10

_NO_LITERATURE = (
    "no literature reference: generated family; see the module docstring of "
    "curvestudio_models.generated for the generation rule"
)
_BAKER = (
    "Baker G.A., Graves-Morris P. (1996) Pade Approximants, 2nd ed., Cambridge University Press"
)
_TINY = 1e-9
"""Positive lower bound for widths and frequencies (zero would divide by zero)."""


class GeneratedInfo(NamedTuple):
    """What the UI shows about a generated id without building (or compiling) the model."""

    family: str
    k: int
    description: str


@dataclass(frozen=True)
class _Member:
    """One generated id: its table row and the callable that builds the spec."""

    id: str
    family: str
    k: int
    description: str
    build: Callable[[], ModelSpec]
    long_tail: bool = False


# -- numerics shared by the initial guesses ----------------------------------------------


def _clean(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Finite ``(x, y)`` pairs ordered by ``x``; fresh arrays."""
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n = min(xa.size, ya.size)
    xa, ya = xa[:n], ya[:n]
    keep = np.isfinite(xa) & np.isfinite(ya)
    order = np.argsort(xa[keep], kind="stable")
    return xa[keep][order], ya[keep][order]


def _scale(values: FloatArray) -> float:
    top = float(np.max(np.abs(values))) if values.size else 0.0
    return top if top > 0.0 else 1.0


def _span(x: FloatArray) -> float:
    if x.size and float(np.ptp(x)) > 0.0:
        return float(np.ptp(x))
    return _scale(x)


def _least_squares(
    columns: Sequence[FloatArray], target: FloatArray, weights: FloatArray
) -> FloatArray | None:
    """Weighted linear least squares ``target ~ columns``; ``None`` if not determined.

    Columns are scaled to unit length first so ``x**3`` and ``1`` do not fight over the
    tolerance of the rank decision.
    """
    if target.size < len(columns):
        return None
    design = np.column_stack(columns)
    root = np.sqrt(weights)
    norms = np.linalg.norm(design * root[:, None], axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        return None
    scaled = design / norms
    coef, _, rank, _ = np.linalg.lstsq(scaled * root[:, None], target * root, rcond=None)
    if rank < design.shape[1] or not np.all(np.isfinite(coef)):
        return None
    return np.asarray(coef / norms, dtype=np.float64)


def _guess_from(
    parameters: tuple[Parameter, ...], estimate: Estimator
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """``initial_guess`` from an estimator on clean data; values are moved inside the bounds.

    ``estimate`` may return non-finite values (they become 0) and may leave parameters out
    of its answer; the result always names every parameter with a finite value strictly
    inside its bounds, whatever the data.
    """
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _clean(x, y)
        with np.errstate(all="ignore"):
            estimates = estimate(xs, ys)
        values = {
            p.name: (v if math.isfinite(v := float(estimates.get(p.name, 0.0))) else 0.0)
            for p in parameters
        }
        return rules_guess(values, bounds)(xs, ys)

    return guess


def _ones(n: int) -> FloatArray:
    return np.ones(n, dtype=np.float64)


# -- applicability ----------------------------------------------------------------------


def _count(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


_X_RULES = {
    "any": "",
    "nonzero": "x must not be zero (the model divides by a power of x)",
    "nonneg": "x must be non-negative",
    "positive": "x must be strictly positive (the model takes ln(x), a power or a root of x)",
}
_MIN_DENOMINATOR = 1e-9


def _domain_check(
    x_rule: str, y_rule: str = "any"
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, the ``x`` rule of the basis and the ``y`` rule of the transform.

    ``y_rule`` is ``any``, ``nonzero_one_sign`` (``y = 1/eta``: no pole between points),
    ``positive`` (``y = exp(eta)``) or ``non_negative`` (``y`` is a square or a square root).
    """

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        for label, values in (("x", xa), ("y", ya)):
            if (n := int(np.count_nonzero(~np.isfinite(values)))) > 0:
                reasons.append(
                    f"{label} must be finite: found {_count(n, 'value')} that are NaN or infinite"
                )
        fx, fy = xa[np.isfinite(xa)], ya[np.isfinite(ya)]
        bad = {
            "nonzero": int(np.count_nonzero(fx == 0.0)),
            "nonneg": int(np.count_nonzero(fx < 0.0)),
            "positive": int(np.count_nonzero(fx <= 0.0)),
        }.get(x_rule, 0)
        if bad:
            reasons.append(f"{_X_RULES[x_rule]}: found {_count(bad, 'offending value')}")
        reasons.extend(_y_reasons(y_rule, fy))
        return reasons

    return check


def _y_reasons(y_rule: str, y: FloatArray) -> list[str]:
    if not y.size:
        return []
    if y_rule == "positive" and bool(np.any(y <= 0.0)):
        return [
            "y must be strictly positive: the model is exp(...), which is never zero or negative"
        ]
    if y_rule == "non_negative" and bool(np.any(y < 0.0)):
        return ["y must not be negative: the model is a square or a square root"]
    if y_rule == "nonzero_one_sign":
        reasons = []
        if bool(np.any(y == 0.0)):
            reasons.append("y must not be zero: the model is the reciprocal of a finite number")
        if bool(np.any(y > 0.0)) and bool(np.any(y < 0.0)):
            reasons.append(
                "y changes sign: the denominator would have to pass through zero (a pole) "
                "inside the data range"
            )
        if float(np.max(np.abs(y))) >= 1.0 / _MIN_DENOMINATOR:
            reasons.append("|y| is so large that the denominator would be a pole at the data")
        return reasons
    return []


# -- shared builders --------------------------------------------------------------------


def _poly(names: Sequence[str], first_power: int = 0, variable: str = "x") -> str:
    """``n0 + n1*x + n2*x**2`` for names given from ``first_power`` upwards."""
    terms = []
    for power, name in enumerate(names, start=first_power):
        if power == 0:
            terms.append(name)
        elif power == 1:
            terms.append(f"{name}*{variable}")
        else:
            terms.append(f"{name}*{variable}**{power}")
    return "+".join(terms)


def _indexed(prefix: str, first: int, last: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(first, last + 1)]


def _spec(
    member: str,
    *,
    name: str,
    category: str,
    description: str,
    expression: str,
    parameters: tuple[Parameter, ...],
    estimate: Estimator,
    example: ModelExample,
    domain_check: Callable[[FloatArray, FloatArray], list[str]],
    assumptions: str,
    references: tuple[str, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        id=member,
        name=name,
        category=category,
        description=description,
        expression=expression,
        parameters=parameters,
        initial_guess=_guess_from(parameters, estimate),
        domain_check=domain_check,
        references=(*references, _NO_LITERATURE),
        assumptions=assumptions,
        example=example,
    )


def _round(value: float) -> float:
    """Four significant digits, so an example reads like a hand-picked one."""
    return float(f"{value:.4g}")


# -- rat_{m}_{n}: rational functions ----------------------------------------------------

_RAT_NUMERATOR = (1.0, 2.0, -1.5, 1.2, -1.0, 0.8, -0.6, 0.5, -0.4)
_RAT_DENOMINATOR = (0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06)
_REWEIGHTING_STEPS = 4


def _rat_estimator(m: int, n: int) -> Estimator:
    """Equation-error fit ``y = sum a_i x^i - sum c_j x^j y`` with Sanathanan-Koerner weights."""
    names = [*_indexed("a", 0, m), *_indexed("c", 1, n)]
    first_den = m + 1

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        columns = [x**i for i in range(m + 1)] + [-(x**j) * y for j in range(1, n + 1)]
        weights = np.ones(y.size)
        best: FloatArray | None = None
        for _ in range(1 + _REWEIGHTING_STEPS):
            coef = _least_squares(columns, y, weights)
            if coef is None:
                break
            denominator = 1.0 + sum(coef[first_den + j - 1] * x**j for j in range(1, n + 1))
            if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
                break
            best = coef
            weights = 1.0 / denominator**2
        out = dict.fromkeys(names, 0.0)
        out["a0"] = float(y[0]) if y.size else 0.0
        if best is not None:
            # Denominator coefficients are bounded below by zero: clip, then refit the numerator.
            den_coef = np.maximum(best[first_den:], 0.0)
            denominator = 1.0 + sum(den_coef[j - 1] * x**j for j in range(1, n + 1))
            numerator = _least_squares(
                [x**i for i in range(first_den)], y * denominator, 1.0 / denominator**2
            )
            best = np.concatenate([best[:first_den] if numerator is None else numerator, den_coef])
            out = dict(zip(names, (float(c) for c in best), strict=True))
        return out

    return estimate


def _rat(m: int, n: int) -> ModelSpec:
    numerator, denominator = _indexed("a", 0, m), _indexed("c", 1, n)
    parameters = (
        Parameter("a0", "Numerator constant term: the value of y at x = 0"),
        *(Parameter(f"a{i}", f"Numerator coefficient of x**{i}") for i in range(1, m + 1)),
        *(
            Parameter(
                f"c{j}",
                f"Denominator coefficient of x**{j} (non-negative, no pole for x >= 0)",
                lower=0.0,
            )
            for j in range(1, n + 1)
        ),
    )
    params = {
        **{f"a{i}": _RAT_NUMERATOR[i] for i in range(m + 1)},
        **{f"c{j}": _RAT_DENOMINATOR[j - 1] for j in range(1, n + 1)},
    }
    return _spec(
        f"rat_{m}_{n}",
        name=f"Rational [{m}/{n}]",
        category="rational",
        description=_rat_description(m, n),
        expression=f"({_poly(numerator)})/(1+{_poly(denominator, first_power=1)})",
        parameters=parameters,
        estimate=_rat_estimator(m, n),
        example=ModelExample(params, x_range=(0.0, 3.0), n=80),
        domain_check=_domain_check("nonneg"),
        assumptions=(
            "x is non-negative and the denominator coefficients are non-negative, so the "
            f"denominator never vanishes; y behaves like x**{m - n} at large x."
        ),
        references=(_BAKER,),
    )


def _rat_description(m: int, n: int) -> str:
    return (
        f"Rational function: a degree-{m} polynomial in x over 1 plus a degree-{n} polynomial "
        f"without constant term ({m + n + 1} parameters)."
    )


# -- invpoly_{d}: reciprocal of a polynomial --------------------------------------------

_INVPOLY_COEFFICIENTS = (2.0, 0.6, 0.4, -0.2, 0.1, -0.06, 0.04, -0.02, 0.01, -0.005)


def _invpoly_estimator(d: int) -> Estimator:
    """``1/y`` is a polynomial in ``x``; weights ``y**4`` (delta method for ``1/y``)."""
    names = _indexed("a", 0, d)

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        keep = y != 0.0
        xk, yk = x[keep], y[keep]
        coef = _least_squares(
            [xk**i for i in range(d + 1)], 1.0 / yk, (yk / _scale(yk) if yk.size else yk) ** 4
        )
        if coef is None:
            peak = float(y[int(np.argmax(np.abs(y)))]) if y.size else 1.0
            return {**dict.fromkeys(names, 0.0), "a0": 1.0 / peak if peak else 1.0}
        return dict(zip(names, (float(c) for c in coef), strict=True))

    return estimate


def _invpoly(d: int) -> ModelSpec:
    names = _indexed("a", 0, d)
    return _spec(
        f"invpoly_{d}",
        name=f"Reciprocal of a degree-{d} polynomial",
        category="rational",
        description=_invpoly_description(d),
        expression=f"1/({_poly(names)})",
        parameters=tuple(
            Parameter(n, f"Coefficient of x**{i} of the denominator polynomial")
            for i, n in enumerate(names)
        ),
        estimate=_invpoly_estimator(d),
        example=ModelExample(
            dict(zip(names, _INVPOLY_COEFFICIENTS, strict=False)), x_range=(-1.0, 1.0), n=80
        ),
        domain_check=_domain_check("any", "nonzero_one_sign"),
        assumptions=(
            "The denominator polynomial keeps one sign and never reaches zero inside the data "
            "range (y has no pole there); errors are additive on y, not on 1/y."
        ),
    )


def _invpoly_description(d: int) -> str:
    return f"Reciprocal of a degree-{d} polynomial in x: y = 1/(a0 + a1*x + ... + a{d}*x**{d})."


# -- recip_{d}: inverse-power series ----------------------------------------------------

_RECIP_COEFFICIENTS = (1.0, 2.0, -1.5, 1.2, -1.0, 0.8, -0.6, 0.5, -0.4, 0.3)


def _recip_estimator(d: int) -> Estimator:
    names = _indexed("a", 0, d)

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        keep = x != 0.0
        u = 1.0 / x[keep]
        coef = _least_squares([u**i for i in range(d + 1)], y[keep], _ones(u.size))
        if coef is None:
            return {**dict.fromkeys(names, 0.0), "a0": float(np.mean(y)) if y.size else 0.0}
        return dict(zip(names, (float(c) for c in coef), strict=True))

    return estimate


def _recip(d: int) -> ModelSpec:
    names = _indexed("a", 0, d)
    terms = ["a0", "a1/x", *(f"a{i}/x**{i}" for i in range(2, d + 1))]
    return _spec(
        f"recip_{d}",
        name=f"Inverse-power series of degree {d}",
        category="rational",
        description=_recip_description(d),
        expression="+".join(terms),
        parameters=(
            Parameter("a0", "Level that y approaches as |x| grows (the horizontal asymptote)"),
            *(Parameter(f"a{i}", f"Coefficient of 1/x**{i}") for i in range(1, d + 1)),
        ),
        estimate=_recip_estimator(d),
        example=ModelExample(
            dict(zip(names, _RECIP_COEFFICIENTS, strict=False)), x_range=(0.4, 4.0), n=80
        ),
        domain_check=_domain_check("nonzero"),
        assumptions=(
            "x is never zero; y is a polynomial in 1/x with constant-variance errors on y."
        ),
    )


def _recip_description(d: int) -> str:
    return f"Inverse-power series of degree {d}: y = a0 + a1/x + ... + a{d}/x**{d}."


# -- sums of exponentials, Gaussians and powers (ordered terms) -------------------------

_EXP_SUM_AMPLITUDES = (2.0, 1.2, 0.8)
_EXP_SUM_RATES = {2: (0.25, 1.5), 3: (0.2, 0.8, 4.0)}  # r1, then the gaps d_i
_EXP_SUM_RANGE = {2: (0.0, 8.0), 3: (0.0, 6.0)}
_RATE_GRID = 14


def _exp_sum_names(n: int, offset: bool) -> list[str]:
    names = ["a1", "r1"]
    for i in range(2, n + 1):
        names += [f"a{i}", f"d{i}"]
    return [*names, "c"] if offset else names


def _exp_sum_estimator(n: int, offset: bool) -> Estimator:
    """Variable projection on a rate grid: amplitudes (and offset) by linear least squares."""
    names = _exp_sum_names(n, offset)

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        out = dict.fromkeys(names, 0.0)
        if x.size == 0:
            return out
        rates = [float(r) for r in np.geomspace(0.05, 40.0, _RATE_GRID) / _span(x)]
        best: tuple[float, tuple[float, ...], FloatArray] | None = None
        for combo in itertools.combinations(rates, n):
            columns = [np.exp(-r * x) for r in combo] + ([np.ones_like(x)] if offset else [])
            coef = _least_squares(columns, y, _ones(y.size))
            if coef is None:
                continue
            sse = float(np.sum((y - np.column_stack(columns) @ coef) ** 2))
            if best is None or sse < best[0]:
                best = (sse, combo, coef)
        if best is None:
            out["a1"], out["r1"] = float(np.mean(y)), 1.0 / _span(x)
            return out
        _, combo, coef = best
        out["a1"], out["r1"] = float(coef[0]), float(combo[0])
        for i in range(2, n + 1):
            out[f"a{i}"], out[f"d{i}"] = float(coef[i - 1]), float(combo[i - 1] - combo[i - 2])
        if offset:
            out["c"] = float(coef[n])
        return out

    return estimate


def _exp_sum(n: int, offset: bool) -> ModelSpec:
    names = _exp_sum_names(n, offset)
    terms = ["a1*exp(-r1*x)"]
    rate = "r1"
    for i in range(2, n + 1):
        rate += f"+d{i}"
        terms.append(f"a{i}*exp(-({rate})*x)")
    if offset:
        terms.append("c")
    parameters = [
        Parameter("a1", "Amplitude of the slowest exponential term"),
        Parameter("r1", "Decay rate of the slowest term (non-negative, per unit x)", lower=0.0),
    ]
    for i in range(2, n + 1):
        parameters += [
            Parameter(f"a{i}", f"Amplitude of exponential term {i}"),
            Parameter(
                f"d{i}",
                f"Extra decay rate of term {i} over term {i - 1} (non-negative): "
                f"r{i} = r{i - 1} + d{i}, which keeps the rates ordered",
                lower=0.0,
            ),
        ]
    if offset:
        parameters.append(Parameter("c", "Offset: the value y approaches at large x"))
    r1, *gaps = _EXP_SUM_RATES[n]
    values = {"r1": r1, **{f"d{i}": g for i, g in zip(range(2, n + 1), gaps, strict=True)}}
    values.update({f"a{i}": _EXP_SUM_AMPLITUDES[i - 1] for i in range(1, n + 1)})
    if offset:
        values["c"] = 0.5
    assert set(values) == set(names)
    return _spec(
        _exp_sum_id(n, offset),
        name=f"Sum of {n} exponentials" + (" with offset" if offset else ""),
        category="exponential",
        description=_exp_sum_description(n, offset),
        expression="+".join(terms),
        parameters=tuple(parameters),
        estimate=_exp_sum_estimator(n, offset),
        example=ModelExample(values, x_range=_EXP_SUM_RANGE[n], n=120),
        domain_check=_domain_check("any"),
        assumptions=(
            f"y is a sum of {n} decaying exponentials with non-negative rates "
            "(multi-compartment decay); the rates are ordered r1 <= r2 so each curve has one "
            "parameter set, and they must be separable on the data scale to be identified."
        ),
    )


def _exp_sum_id(n: int, offset: bool) -> str:
    return f"exp_sum_{n}" + ("_offset" if offset else "")


def _exp_sum_description(n: int, offset: bool) -> str:
    return (
        f"Sum of {n} decaying exponentials with ordered rates"
        + (" plus a constant offset" if offset else "")
        + f" ({len(_exp_sum_names(n, offset))} parameters)."
    )


_GAUSS_AMPLITUDES = (2.0, 1.5, 1.0)
_GAUSS_CENTRES = {2: (3.0, (4.0,)), 3: (2.0, (3.5, 4.5))}
_GAUSS_WIDTHS = (1.0, 1.3, 0.9)
_GAUSS_RANGE = {2: (-1.0, 12.0), 3: (-2.0, 14.0)}
_CENTRE_GRID = 12


def _gauss_names(n: int) -> list[str]:
    names = ["a1", "mu1", "s1"]
    for i in range(2, n + 1):
        names += [f"a{i}", f"d{i}", f"s{i}"]
    return names


def _gauss_sum_estimator(n: int) -> Estimator:
    """Variable projection on a grid of centres and one common width (linear amplitudes)."""
    names = _gauss_names(n)

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        out = dict.fromkeys(names, 0.0)
        if x.size == 0:
            return out
        span = _span(x)
        centres = [float(c) for c in np.linspace(float(x[0]), float(x[-1]), _CENTRE_GRID)]
        best: tuple[float, tuple[float, ...], float, FloatArray] | None = None
        for width in (span / (6.0 * n), span / (3.0 * n), span / (1.5 * n)):
            for combo in itertools.combinations(centres, n):
                columns = [np.exp(-((x - c) ** 2) / (2.0 * width**2)) for c in combo]
                coef = _least_squares(columns, y, _ones(y.size))
                if coef is None:
                    continue
                sse = float(np.sum((y - np.column_stack(columns) @ coef) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, combo, width, coef)
        if best is None:
            out["a1"], out["mu1"], out["s1"] = float(np.max(y)), float(np.median(x)), span / 4.0
            for i in range(2, n + 1):
                out[f"s{i}"], out[f"d{i}"] = span / 4.0, span / 4.0
            return out
        _, combo, width, coef = best
        out["a1"], out["mu1"], out["s1"] = float(coef[0]), float(combo[0]), width
        for i in range(2, n + 1):
            out[f"a{i}"], out[f"d{i}"] = float(coef[i - 1]), float(combo[i - 1] - combo[i - 2])
            out[f"s{i}"] = width
        return out

    return estimate


def _gauss_sum(n: int) -> ModelSpec:
    terms = ["a1*exp(-(x-mu1)**2/(2*s1**2))"]
    centre = "mu1"
    parameters = [
        Parameter("a1", "Height of the left-most peak (negative for a dip)"),
        Parameter("mu1", "Centre of the left-most peak"),
        Parameter("s1", "Standard deviation of the left-most peak (positive)", lower=0.0),
    ]
    for i in range(2, n + 1):
        centre += f"+d{i}"
        terms.append(f"a{i}*exp(-(x-({centre}))**2/(2*s{i}**2))")
        parameters += [
            Parameter(f"a{i}", f"Height of peak {i}"),
            Parameter(
                f"d{i}",
                f"Distance of peak {i} to the right of peak {i - 1} (non-negative): "
                f"mu{i} = mu{i - 1} + d{i}, which keeps the peaks ordered",
                lower=0.0,
            ),
            Parameter(f"s{i}", f"Standard deviation of peak {i} (positive)", lower=0.0),
        ]
    mu1, gaps = _GAUSS_CENTRES[n]
    values = {"mu1": mu1}
    for i in range(1, n + 1):
        values[f"a{i}"] = _GAUSS_AMPLITUDES[i - 1]
        values[f"s{i}"] = _GAUSS_WIDTHS[i - 1]
        if i >= 2:
            values[f"d{i}"] = gaps[i - 2]
    return _spec(
        f"gauss_sum_{n}",
        name=f"Sum of {n} Gaussian peaks",
        category="peak",
        description=_gauss_description(n),
        expression="+".join(terms),
        parameters=tuple(parameters),
        estimate=_gauss_sum_estimator(n),
        example=ModelExample(values, x_range=_GAUSS_RANGE[n], n=140),
        domain_check=_domain_check("any"),
        assumptions=(
            f"y is a sum of {n} Gaussian peaks on a zero baseline; the peaks are ordered by "
            "centre (mu1 <= mu2) so each curve has one parameter set, and they must be resolved "
            "by the data to be identified."
        ),
    )


def _gauss_description(n: int) -> str:
    return f"Sum of {n} Gaussian peaks with ordered centres ({3 * n} parameters)."


_POW_AMPLITUDES = (2.0, -0.15, 0.02)
_POW_EXPONENTS = {2: (0.5, (1.5,)), 3: (0.5, (1.0, 1.5))}
_POW_GRID = tuple(float(p) for p in np.arange(-3.0, 3.01, 0.5))


def _pow_names(n: int) -> list[str]:
    names = ["a1", "p1"]
    for i in range(2, n + 1):
        names += [f"a{i}", f"d{i}"]
    return names


def _pow_sum_estimator(n: int) -> Estimator:
    """Variable projection on a grid of exponents; amplitudes by least squares."""
    names = _pow_names(n)

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        out = dict.fromkeys(names, 0.0)
        keep = x > 0.0
        xk, yk = x[keep], y[keep]
        out["a1"] = float(np.mean(y)) if y.size else 0.0
        best: tuple[float, tuple[float, ...], FloatArray] | None = None
        for combo in itertools.combinations(_POW_GRID, n):
            columns = [xk**p for p in combo]
            coef = _least_squares(columns, yk, _ones(yk.size))
            if coef is None:
                continue
            sse = float(np.sum((yk - np.column_stack(columns) @ coef) ** 2))
            if best is None or sse < best[0]:
                best = (sse, combo, coef)
        if best is None:
            return out
        _, combo, coef = best
        out["a1"], out["p1"] = float(coef[0]), float(combo[0])
        for i in range(2, n + 1):
            out[f"a{i}"], out[f"d{i}"] = float(coef[i - 1]), float(combo[i - 1] - combo[i - 2])
        return out

    return estimate


def _pow_sum(n: int) -> ModelSpec:
    terms = ["a1*x**p1"]
    exponent = "p1"
    parameters = [
        Parameter("a1", "Coefficient of the lowest power"),
        Parameter("p1", "Exponent of the lowest power term"),
    ]
    for i in range(2, n + 1):
        exponent += f"+d{i}"
        terms.append(f"a{i}*x**({exponent})")
        parameters += [
            Parameter(f"a{i}", f"Coefficient of power term {i}"),
            Parameter(
                f"d{i}",
                f"Exponent of term {i} above that of term {i - 1} (non-negative): "
                f"p{i} = p{i - 1} + d{i}, which keeps the exponents ordered",
                lower=0.0,
            ),
        ]
    p1, gaps = _POW_EXPONENTS[n]
    values = {"p1": p1}
    for i in range(1, n + 1):
        values[f"a{i}"] = _POW_AMPLITUDES[i - 1]
        if i >= 2:
            values[f"d{i}"] = gaps[i - 2]
    return _spec(
        f"pow_sum_{n}",
        name=f"Sum of {n} power laws",
        category="power",
        description=_pow_description(n),
        expression="+".join(terms),
        parameters=tuple(parameters),
        estimate=_pow_sum_estimator(n),
        example=ModelExample(values, x_range=(0.5, 4.0), n=100),
        domain_check=_domain_check("positive"),
        assumptions=(
            f"x is strictly positive; y is a sum of {n} power laws whose exponents are ordered "
            "(p1 <= p2) so each curve has one parameter set; the terms must be distinguishable "
            "over the data range."
        ),
    )


def _pow_description(n: int) -> str:
    return f"Sum of {n} power laws with ordered exponents ({2 * n} parameters)."


_FOURIER_FREQUENCY = 1.3
_FOURIER_RANGE = (0.0, 10.0)
_FREQUENCY_GRID = 400


def _fourier_names(n: int) -> list[str]:
    names = ["a0", "w"]
    for i in range(1, n + 1):
        names += [f"a{i}", f"b{i}"]
    return names


def _fourier_estimator(n: int) -> Estimator:
    """Variable projection on a frequency grid: the coefficients are linear for a fixed ``w``."""
    names = _fourier_names(n)

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        out = dict.fromkeys(names, 0.0)
        out["a0"] = float(np.mean(y)) if y.size else 0.0
        span = _span(x)
        out["w"] = 2.0 * math.pi / span
        steps = np.diff(np.unique(x))
        step = float(np.median(steps)) if steps.size else span
        low, high = math.pi / span, math.pi / (n * step)
        if x.size == 0 or not high > low:
            return out
        best: tuple[float, float, FloatArray] | None = None
        for w in (float(v) for v in np.geomspace(low, high, _FREQUENCY_GRID)):
            columns = [np.ones_like(x)]
            for i in range(1, n + 1):
                columns += [np.cos(i * w * x), np.sin(i * w * x)]
            coef = _least_squares(columns, y, _ones(y.size))
            if coef is None:
                continue
            sse = float(np.sum((y - np.column_stack(columns) @ coef) ** 2))
            if best is None or sse < best[0]:
                best = (sse, w, coef)
        if best is None:
            return out
        _, w, coef = best
        out["w"] = w
        out["a0"] = float(coef[0])
        for i in range(1, n + 1):
            out[f"a{i}"], out[f"b{i}"] = float(coef[2 * i - 1]), float(coef[2 * i])
        return out

    return estimate


def _fourier(n: int) -> ModelSpec:
    terms = [
        "a0",
        *(
            f"a{i}*cos({i}*w*x)+b{i}*sin({i}*w*x)" if i > 1 else "a1*cos(w*x)+b1*sin(w*x)"
            for i in range(1, n + 1)
        ),
    ]
    parameters = [
        Parameter("a0", "Mean level of the oscillation"),
        Parameter("w", "Base angular frequency (positive, radians per unit x)", lower=_TINY),
    ]
    for i in range(1, n + 1):
        parameters += [
            Parameter(f"a{i}", f"Cosine amplitude of harmonic {i}"),
            Parameter(f"b{i}", f"Sine amplitude of harmonic {i}"),
        ]
    values = {"a0": 1.0, "w": _FOURIER_FREQUENCY}
    for i in range(1, n + 1):
        values[f"a{i}"] = 1.5 / i * (1.0 if i % 2 else -1.0)
        values[f"b{i}"] = 1.0 / i
    return _spec(
        f"fourier_{n}",
        name=f"Fourier series, {n} harmonic" + ("s" if n > 1 else ""),
        category="harmonic",
        description=_fourier_description(n),
        expression="+".join(terms),
        parameters=tuple(parameters),
        estimate=_fourier_estimator(n),
        example=ModelExample(values, x_range=_FOURIER_RANGE, n=160),
        domain_check=_domain_check("any"),
        assumptions=(
            "y is periodic in x with period 2*pi/w; the data must cover at least one period "
            "and be sampled finely enough for the highest harmonic."
        ),
    )


def _fourier_description(n: int) -> str:
    return (
        f"Fourier series with {n} harmonic{'s' if n > 1 else ''} and a free base frequency "
        f"({2 * n + 2} parameters)."
    )


# -- tc_{y}_{x} and tc3_{y}_{x}_{x2}: TableCurve-style linear predictors ----------------


class _Basis(NamedTuple):
    token: str
    label: str  # f(x) as text
    fn: Callable[[FloatArray], FloatArray]
    x_rule: str  # key of _X_RULES

    def term(self, coefficient: str) -> str:
        """``coefficient * f(x)`` as text (``c/x`` for reciprocals)."""
        if self.label.startswith("1/"):
            return f"{coefficient}/{self.label[2:]}"
        return f"{coefficient}*{self.label}"


_BASE = (
    _Basis("x", "x", lambda x: x, "any"),
    _Basis("inv", "1/x", lambda x: 1.0 / x, "nonzero"),
    _Basis("ln", "ln(x)", np.log, "positive"),
    _Basis("sqrt", "sqrt(x)", np.sqrt, "nonneg"),
    _Basis("sqr", "x**2", lambda x: x**2, "any"),
    _Basis("expneg", "exp(-x)", lambda x: np.exp(-x), "any"),
)
_EXTRA = (
    _Basis("cube", "x**3", lambda x: x**3, "any"),
    _Basis("invsqr", "1/x**2", lambda x: 1.0 / x**2, "nonzero"),
    _Basis("lnsq", "ln(x)**2", lambda x: np.log(x) ** 2, "positive"),
    _Basis("xlnx", "x*ln(x)", lambda x: x * np.log(x), "positive"),
    _Basis("invsqrt", "1/sqrt(x)", lambda x: 1.0 / np.sqrt(x), "positive"),
)
_BASES = {b.token: b for b in (*_BASE, *_EXTRA)}
_BASE_TOKENS = frozenset(b.token for b in _BASE)
_RULE_ORDER = ("any", "nonneg", "nonzero", "positive")  # strongest rule wins


class _Transform(NamedTuple):
    token: str
    wrap: str  # ``{e}`` is the linear predictor
    y_rule: str
    eta_mid: float  # level of the example's linear predictor
    linearise: Callable[[FloatArray], tuple[FloatArray, FloatArray, BoolArray]]


def _lin_id(y: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
    return y, _ones(y.size), np.ones(y.size, dtype=bool)


def _lin_inv(y: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
    ok = y != 0.0
    z = np.where(ok, 1.0 / np.where(ok, y, 1.0), 0.0)
    return z, (y / _scale(y)) ** 4, ok


def _lin_exp(y: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
    ok = y > 0.0
    return np.log(np.where(ok, y, 1.0)), (y / _scale(y)) ** 2, ok


def _lin_sqr(y: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
    ok = y >= 0.0
    return np.sqrt(np.where(ok, y, 0.0)), np.abs(y) / _scale(y), ok


def _lin_sqrtof(y: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
    ok = y >= 0.0
    z = np.where(ok, y, 0.0) ** 2
    return z, 1.0 / np.maximum((y / _scale(y)) ** 2, 1e-4), ok


_TRANSFORMS = {
    t.token: t
    for t in (
        _Transform("id", "{e}", "any", 2.0, _lin_id),
        _Transform("inv", "1/({e})", "nonzero_one_sign", 2.0, _lin_inv),
        _Transform("exp", "exp({e})", "positive", 1.0, _lin_exp),
        _Transform("sqr", "({e})**2", "non_negative", 2.0, _lin_sqr),
        _Transform("sqrtof", "sqrt({e})", "non_negative", 2.0, _lin_sqrtof),
    )
}

_COEFFICIENTS = ("b", "c")


def _eta(bases: tuple[_Basis, ...], *, spaced: bool) -> str:
    glue = " + " if spaced else "+"
    terms = [t.term(c) for t, c in zip(bases, _COEFFICIENTS, strict=False)]
    return glue.join(["a", *terms])


def _tc_category(transform: _Transform, bases: tuple[_Basis, ...]) -> str:
    if transform.token == "exp":
        return "exponential"
    if transform.token == "inv":
        return "rational"
    if transform.token in ("sqr", "sqrtof"):
        return "power"
    tokens = {b.token for b in bases}
    if tokens & {"ln", "lnsq", "xlnx"}:
        return "logarithmic"
    if "expneg" in tokens:
        return "exponential"
    if tokens & {"inv", "invsqr", "invsqrt"}:
        return "rational"
    return "power" if "sqrt" in tokens else "polynomial"


def _tc_estimator(transform: _Transform, bases: tuple[_Basis, ...]) -> Estimator:
    names = ["a", *_COEFFICIENTS[: len(bases)]]

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        out = dict.fromkeys(names, 0.0)
        columns = [np.ones_like(x), *(b.fn(x) for b in bases)]
        z, weights, ok = transform.linearise(y)
        for column in columns:
            ok = ok & np.isfinite(column)
        coef = _least_squares([c[ok] for c in columns], z[ok], weights[ok])
        if coef is None:
            out["a"] = _constant_level(transform.token, float(np.mean(y)) if y.size else 0.0)
            return out
        if transform.token == "sqr" and coef[0] < 0.0:
            coef = -coef  # (-eta)**2 = eta**2; the bound a >= 0 picks one sign
        return dict(zip(names, (float(c) for c in coef), strict=True))

    return estimate


def _constant_level(token: str, mean: float) -> float:
    """The linear-predictor level whose transform equals ``mean`` (0 where that is undefined)."""
    if token == "inv":
        return 1.0 / mean if mean else 1.0
    if token == "exp":
        return math.log(mean) if mean > 0.0 else 0.0
    if token == "sqr":
        return math.sqrt(mean) if mean > 0.0 else 0.0
    if token == "sqrtof":
        return mean * mean
    return mean


def _tc_example(transform: _Transform, bases: tuple[_Basis, ...]) -> ModelExample:
    """Linear predictor held inside ``eta_mid +- 0.9`` over the example range (no pole, root or
    overflow), with coefficients rounded to four digits."""
    x = np.linspace(1.0, 6.0, 60)
    weights = (0.55, -0.35) if len(bases) == 2 else (0.7,)
    values: dict[str, float] = {}
    intercept = transform.eta_mid
    for basis, name, weight in zip(bases, _COEFFICIENTS, weights, strict=False):
        column = basis.fn(x)
        coefficient = _round(weight / float(np.ptp(column)))
        values[name] = coefficient
        intercept -= coefficient * float(np.mean(column))
    values["a"] = _round(intercept)
    if transform.token == "sqr" and values["a"] < 0.0:
        values = {name: -v for name, v in values.items()}
    return ModelExample(values, x_range=(1.0, 6.0), n=60)


def _tc(transform_token: str, bases: tuple[_Basis, ...], member: str) -> ModelSpec:
    transform = _TRANSFORMS[transform_token]
    k = 1 + len(bases)
    parameters = [
        Parameter(
            "a",
            "Constant term of the linear predictor"
            + (
                " (non-negative: the sign of the predictor is not identifiable)"
                if transform_token == "sqr"
                else ""
            ),
            lower=0.0 if transform_token == "sqr" else -math.inf,
        ),
        *(
            Parameter(name, f"Coefficient of {basis.label} in the linear predictor")
            for name, basis in zip(_COEFFICIENTS, bases, strict=False)
        ),
    ]
    x_rule = max((b.x_rule for b in bases), key=_RULE_ORDER.index)
    pretty = transform.wrap.format(e=_eta(bases, spaced=True))
    return _spec(
        member,
        name=f"y = {pretty}",
        category=_tc_category(transform, bases),
        description=_tc_description(transform_token, bases),
        expression=transform.wrap.format(e=_eta(bases, spaced=False)),
        parameters=tuple(parameters),
        estimate=_tc_estimator(transform, bases),
        example=_tc_example(transform, bases),
        domain_check=_domain_check(x_rule, transform.y_rule),
        assumptions=(
            f"Fitted on the original y scale (the transform is inside the expression). "
            f"{_X_RULES[x_rule] or 'x may be any real number'}"
            f"; {_y_assumption(transform.y_rule)} {k} parameters."
        ),
    )


def _y_assumption(y_rule: str) -> str:
    return {
        "any": "y may be any real number.",
        "nonzero_one_sign": "y keeps one sign and never reaches zero.",
        "positive": "y is strictly positive.",
        "non_negative": "y is not negative.",
    }[y_rule]


def _tc_description(transform_token: str, bases: tuple[_Basis, ...]) -> str:
    transform = _TRANSFORMS[transform_token]
    pretty = transform.wrap.format(e=_eta(bases, spaced=True))
    return f"TableCurve-style generated model: y = {pretty} ({1 + len(bases)} parameters)."


# Models that would repeat an existing curve; asking for them names the original.
_EQUAL_TO = {
    "rat_1_1": "rational_1_1",
    "rat_2_1": "rational_2_1",
    "rat_1_2": "rational_1_2",
    "invpoly_1": "rational_reciprocal_linear",
    "recip_1": "rational_reciprocal_offset",
    "recip_2": "rational_reciprocal_quadratic_offset",
    "tc_id_x": "a degree-1 polynomial (fit_polynomial)",
    "tc_id_inv": "rational_reciprocal_offset",
    "tc_id_ln": "log_natural",
    "tc_inv_x": "rational_reciprocal_linear",
    "tc_inv_inv": "rational_x_over_linear",
    "tc_inv_sqr": "rational_agnesi",
    "tc_exp_x": "exp_growth",
    "tc_exp_ln": "power",
    "tc3_id_x_sqr": "a degree-2 polynomial (fit_polynomial)",
    "tc3_id_inv_invsqr": "rational_reciprocal_quadratic_offset",
    "tc3_inv_x_sqr": "invpoly_2",
    "tc3_exp_x_sqr": "gaussian",
    "tc3_exp_ln_lnsq": "a log-normal peak",
    "tc3_exp_x_ln": "exponential_power_growth",
}


# -- the id table ------------------------------------------------------------------------


def _members() -> list[_Member]:
    members: list[_Member] = []

    def add(
        member_id: str,
        family: str,
        k: int,
        description: str,
        build: Callable[[], ModelSpec],
        long_tail: bool = False,
    ) -> None:
        if member_id not in _EQUAL_TO:
            members.append(_Member(member_id, family, k, description, build, long_tail))

    for m in range(1, 9):
        for n in range(1, 10 - m):
            add(
                f"rat_{m}_{n}",
                "rat",
                m + n + 1,
                _rat_description(m, n),
                partial(_rat, m, n),
            )
    for d in range(1, 10):
        add(f"invpoly_{d}", "invpoly", d + 1, _invpoly_description(d), partial(_invpoly, d))
    for d in range(1, 10):
        add(f"recip_{d}", "recip", d + 1, _recip_description(d), partial(_recip, d))
    for n in (2, 3):
        for offset in (False, True):
            add(
                _exp_sum_id(n, offset),
                "exp_sum",
                len(_exp_sum_names(n, offset)),
                _exp_sum_description(n, offset),
                partial(_exp_sum, n, offset),
            )
    for n in (2, 3):
        add(f"gauss_sum_{n}", "gauss_sum", 3 * n, _gauss_description(n), partial(_gauss_sum, n))
    for n in (2, 3):
        add(f"pow_sum_{n}", "pow_sum", 2 * n, _pow_description(n), partial(_pow_sum, n))
    for n in range(1, 7):
        add(f"fourier_{n}", "fourier", 2 * n + 2, _fourier_description(n), partial(_fourier, n))
    for t in _TRANSFORMS:
        for basis in _BASE:
            member = f"tc_{t}_{basis.token}"
            add(
                member,
                "tc",
                2,
                _tc_description(t, (basis,)),
                partial(_tc, t, (basis,), member),
            )
    ordered = (*_BASE, *_EXTRA)
    for t in _TRANSFORMS:
        for first, second in itertools.combinations(ordered, 2):
            member = f"tc3_{t}_{first.token}_{second.token}"
            extended = not {first.token, second.token} <= _BASE_TOKENS
            add(
                member,
                "tc3",
                3,
                _tc_description(t, (first, second)),
                partial(_tc, t, (first, second), member),
                long_tail=extended,
            )
    return members


@cache
def _table() -> dict[str, _Member]:
    return {m.id: m for m in _members()}


_PREFIXES = (
    ("rat_", "rat"),
    ("invpoly_", "invpoly"),
    ("recip_", "recip"),
    ("exp_sum_", "exp_sum"),
    ("gauss_sum_", "gauss_sum"),
    ("pow_sum_", "pow_sum"),
    ("fourier_", "fourier"),
    ("tc_", "tc"),
    ("tc3_", "tc3"),
)


def ids(exhaustive: bool = False) -> list[str]:
    """Generated ids in stable order.

    The default lists models with at most :data:`DEFAULT_PARAMETER_CAP` parameters and leaves
    out the ``tc3`` models that use an extended basis function; ``exhaustive`` raises the cap
    to :data:`EXHAUSTIVE_PARAMETER_CAP` and includes those.
    """
    cap = EXHAUSTIVE_PARAMETER_CAP if exhaustive else DEFAULT_PARAMETER_CAP
    return [m.id for m in _table().values() if m.k <= cap and (exhaustive or not m.long_tail)]


def generated_info(model_id: str) -> GeneratedInfo:
    """``(family, k, description)`` of a generated id, without building the model.

    Works for every generated id, including those not listed by :func:`ids`.

    Raises:
        KeyError: ``model_id`` is not a generated id (the message names the existing model
            when the id was left out because it repeats one).
    """
    member = _table().get(model_id)
    if member is None:
        raise KeyError(_unknown(model_id))
    return GeneratedInfo(member.family, member.k, member.description)


def build(model_id: str) -> ModelSpec:
    """The spec of a generated id (compiled on first use, like any ``ModelSpec``).

    Raises:
        KeyError: ``model_id`` is not a generated id.
    """
    member = _table().get(model_id)
    if member is None:
        raise KeyError(_unknown(model_id))
    return member.build()


def _unknown(model_id: str) -> str:
    if model_id in _EQUAL_TO:
        return f"{model_id!r} is not generated: it is the same curve as {_EQUAL_TO[model_id]}"
    return f"{model_id!r} is not a generated model id"


def register_generated() -> None:
    """Register every generated family with the core registry (idempotent).

    Raises:
        ValueError: a family prefix is already taken by something else (from the registry).
    """
    for prefix, family in _PREFIXES:
        existing = registry._FAMILIES.get(prefix)
        if existing is not None and existing.build is _builders[family]:
            continue
        register_family(prefix, _id_lister(family), _builders[family])


def _id_lister(family: str) -> Callable[[bool], list[str]]:
    def listing(exhaustive: bool) -> list[str]:
        return [i for i in ids(exhaustive) if _table()[i].family == family]

    return listing


def _family_builder(family: str) -> Callable[[str], ModelSpec]:
    def build_member(model_id: str) -> ModelSpec:
        member = _table().get(model_id)
        if member is None or member.family != family:
            raise KeyError(_unknown(model_id))
        return member.build()

    return build_member


_builders = {family: _family_builder(family) for _, family in _PREFIXES}
