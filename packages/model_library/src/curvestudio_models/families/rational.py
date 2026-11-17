"""Rational models: a polynomial (or a constant) divided by a polynomial in ``x``.

Every model here is fitted on the original response scale; reciprocal forms such as
``1/(a+b*x)`` are *not* fitted as a straight line in ``1/y``. Their initial guesses do use
that straight line (or, for the Pade-type quotients, the equation-error form
``y*(1+c*x) = a+b*x``), which is exact for noiseless data and a sound start for noisy data.

Denominator coefficients that must stay non-negative for the curve to have no pole on
``x >= 0`` are bounded below by zero, and the example ranges keep clear of every pole.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from curvestudio.models import ModelExample, ModelSpec, Parameter, rules_guess

FloatArray = NDArray[np.float64]
Estimator = Callable[[FloatArray, FloatArray], dict[str, float]]

# References are cited only where the book treats the model; no chapter or page is given
# because none was verified.
_BAKER = (
    "Baker G.A., Graves-Morris P. (1996) Pade Approximants, 2nd ed., Cambridge University Press"
)
_DRAPER_SMITH = "Draper N.R., Smith H. (1998) Applied Regression Analysis, 3rd ed., Wiley"
_NUMERICAL_RECIPES = (
    "Press W.H., Teukolsky S.A., Vetterling W.T., Flannery B.P. (2007) Numerical Recipes: "
    "The Art of Scientific Computing, 3rd ed., Cambridge University Press "
    "(general linear least squares)"
)


# -- initial guesses --------------------------------------------------------------------


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
    """Largest magnitude of ``values``, or 1 when there is none or all are zero."""
    top = float(np.max(np.abs(values))) if values.size else 0.0
    return top if top > 0.0 else 1.0


def _span(x: FloatArray) -> float:
    """Extent of ``x``; its magnitude (or 1) when ``x`` is constant."""
    if x.size and float(np.ptp(x)) > 0.0:
        return float(np.ptp(x))
    return _scale(x)


def _least_squares(
    columns: Sequence[FloatArray], target: FloatArray, weights: FloatArray
) -> FloatArray | None:
    """Weighted linear least squares ``target ~ columns``; ``None`` if not determined.

    Columns are scaled to unit length first so ``x**2`` and ``1`` do not fight over the
    tolerance of the rank decision.
    """
    if target.size < len(columns):
        return None
    design = np.column_stack(columns)
    root = np.sqrt(weights)
    with np.errstate(all="ignore"):
        norms = np.linalg.norm(design * root[:, None], axis=0)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
            return None
        scaled = design / norms
        coef, _, rank, _ = np.linalg.lstsq(scaled * root[:, None], target * root, rcond=None)
    if rank < design.shape[1] or not np.all(np.isfinite(coef)):
        return None
    return np.asarray(coef / norms, dtype=np.float64)


def _reciprocal_weights(y: FloatArray) -> FloatArray:
    """Weights ``y**4`` (delta method for ``1/y``), scaled to a maximum of 1."""
    scaled = y / _scale(y)
    return np.asarray(scaled**4, dtype=np.float64)


def _bespoke_guess(
    parameters: tuple[Parameter, ...], estimate: Estimator
) -> Callable[[FloatArray, FloatArray], dict[str, float]]:
    """``initial_guess`` from an estimator on clean data; values are moved inside the bounds.

    ``estimate`` always returns finite values for every parameter (its own fallback covers
    degenerate data); :func:`rules_guess` with literal numbers then keeps them strictly
    inside the parameter bounds.
    """
    bounds = {p.name: (p.lower, p.upper) for p in parameters}

    def guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
        xs, ys = _clean(x, y)
        with np.errstate(all="ignore"):  # overflow on extreme data falls back, see _least_squares
            estimates = estimate(xs, ys)
        values = {k: (v if math.isfinite(v) else 0.0) for k, v in estimates.items()}
        return rules_guess(values, bounds)(xs, ys)

    return guess


def _reciprocal_offset(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``y = a + b*u`` with ``u = 1/x``: exactly linear in the parameters."""
    keep = x != 0.0
    mean = float(np.mean(y)) if y.size else 0.0
    coef = _least_squares([np.ones(int(keep.sum())), 1.0 / x[keep]], y[keep], np.ones(keep.sum()))
    return {"a": mean, "b": 0.0} if coef is None else {"a": coef[0], "b": coef[1]}


def _reciprocal_quadratic_offset(x: FloatArray, y: FloatArray) -> dict[str, float]:
    keep = x != 0.0
    u = 1.0 / x[keep]
    mean = float(np.mean(y)) if y.size else 0.0
    coef = _least_squares([np.ones(u.size), u, u**2], y[keep], np.ones(u.size))
    if coef is None:
        return {"a": mean, "b": 0.0, "c": 0.0}
    return {"a": coef[0], "b": coef[1], "c": coef[2]}


def _reciprocal_linear(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``1/y = a + b*x`` weighted by ``y**4``."""
    keep = y != 0.0
    coef = _least_squares(
        [np.ones(int(keep.sum())), x[keep]], 1.0 / y[keep], _reciprocal_weights(y[keep])
    )
    if coef is None:
        peak = float(y[int(np.argmax(np.abs(y)))]) if y.size else 1.0
        return {"a": 1.0 / peak if peak else 1.0, "b": 0.0}
    return {"a": coef[0], "b": coef[1]}


def _x_over_linear(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``x/y = a + b*x`` (Hanes-Woolf form) weighted by ``y**4 / x**2``."""
    keep = (y != 0.0) & (x != 0.0)
    xk, yk = x[keep], y[keep]
    coef = _least_squares(
        [np.ones(xk.size), xk], xk / yk, _reciprocal_weights(yk) / np.maximum(xk**2, 1e-300)
    )
    if coef is None:
        top = _scale(y)
        return {"a": _span(x) / top, "b": 1.0 / top}
    return {"a": coef[0], "b": coef[1]}


_REWEIGHTING_STEPS = 4
"""Sanathanan-Koerner refinements of the equation-error fit."""


def _pade(num_degree: int, den_degree: int) -> Estimator:
    """Estimator for ``(sum a_i x^i) / (1 + sum c_j x^j)``.

    ``y = sum a_i x^i - sum c_j x^j y`` is linear in all coefficients and exact for noiseless
    data. With noise it multiplies the error by the denominator, so the fit is repeated with
    weights ``1/denominator**2`` (Sanathanan-Koerner), which turns the equation error into the
    error of the quotient itself; a step that produces a non-positive denominator on the data is
    discarded. Parameter names follow the order ``a, b, c, d`` of the model.
    """
    names = ["a", "b", "c", "d"][: num_degree + 1 + den_degree]
    first_den = num_degree + 1

    def estimate(x: FloatArray, y: FloatArray) -> dict[str, float]:
        columns = [x**i for i in range(num_degree + 1)] + [
            -(x**j) * y for j in range(1, den_degree + 1)
        ]
        weights = np.ones(y.size)
        best: FloatArray | None = None
        for _ in range(1 + _REWEIGHTING_STEPS):
            coef = _least_squares(columns, y, weights)
            if coef is None:
                break
            denominator = 1.0 + sum(
                coef[first_den + j - 1] * x**j for j in range(1, den_degree + 1)
            )
            if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
                break
            best = coef
            weights = 1.0 / denominator**2
        out = dict.fromkeys(names, 0.0)
        out["a"] = float(y[0]) if y.size else 0.0
        if best is not None:
            # The model bounds the denominator coefficients below by zero. Noise can push a
            # weakly determined one negative; clip it and refit the numerator for that denominator.
            denominator_coef = np.maximum(best[first_den:], 0.0)
            denominator = 1.0 + sum(
                denominator_coef[j - 1] * x**j for j in range(1, den_degree + 1)
            )
            numerator = _least_squares(
                [x**i for i in range(first_den)], y * denominator, 1.0 / denominator**2
            )
            best = np.concatenate(
                [best[:first_den] if numerator is None else numerator, denominator_coef]
            )
            out = dict(zip(names, (float(c) for c in best), strict=True))
        return out

    return estimate


def _agnesi(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """``1/y = 1/a + (b/a) * x**2`` weighted by ``y**4``."""
    keep = y != 0.0
    coef = _least_squares(
        [np.ones(int(keep.sum())), x[keep] ** 2], 1.0 / y[keep], _reciprocal_weights(y[keep])
    )
    peak = float(y[int(np.argmax(np.abs(y)))]) if y.size else 1.0
    fallback = {"a": peak or 1.0, "b": 1.0 / _span(x) ** 2}
    if coef is None or coef[0] == 0.0:
        return fallback
    return {"a": 1.0 / coef[0], "b": coef[1] / coef[0]}


# -- applicability ----------------------------------------------------------------------


def _count(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


_MIN_DENOMINATOR = 1e-9
"""Smallest denominator magnitude the data may imply for ``y = 1/denominator``."""


def _reciprocal_reasons(y: FloatArray) -> list[str]:
    """Why ``y = 1/(a+b*x)`` cannot describe ``y``: a pole inside the data range.

    ``y`` is the reciprocal of the denominator, so the denominator changes sign exactly when
    ``y`` does, is zero where ``y`` is infinite, and reaches ``1/|y|`` in magnitude.
    """
    reasons: list[str] = []
    if int(np.count_nonzero(y == 0.0)):
        reasons.append("y must not be zero: 1/(a+b*x) never reaches zero")
    if bool(np.any(y > 0.0)) and bool(np.any(y < 0.0)):
        reasons.append(
            "y changes sign: the denominator a+b*x would have to pass through zero (a pole) "
            "inside the data range"
        )
    if float(np.max(np.abs(y))) >= 1.0 / _MIN_DENOMINATOR:
        reasons.append(
            f"|y| reaches {float(np.max(np.abs(y))):.3g}, which needs a denominator below "
            f"{_MIN_DENOMINATOR:g} (a pole at the data)"
        )
    return reasons


def _domain_check(
    *,
    x_nonzero_for: str | None = None,
    x_non_negative: bool = False,
    reciprocal_denominator: bool = False,
) -> Callable[[FloatArray, FloatArray], list[str]]:
    """Applicability: finite data, plus ``x != 0`` or ``x >= 0`` where the model needs it."""

    def check(x: FloatArray, y: FloatArray) -> list[str]:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        reasons: list[str] = []
        if (n := int(np.count_nonzero(~np.isfinite(xa)))) > 0:
            reasons.append(f"x must be finite: found {_count(n, 'value')} that are NaN or infinite")
        if (n := int(np.count_nonzero(~np.isfinite(ya)))) > 0:
            reasons.append(f"y must be finite: found {_count(n, 'value')} that are NaN or infinite")
        finite_x = xa[np.isfinite(xa)]
        if x_nonzero_for is not None and (n := int(np.count_nonzero(finite_x == 0.0))) > 0:
            reasons.append(f"x must not be zero for {x_nonzero_for}: found {_count(n, 'zero')}")
        if x_non_negative and (n := int(np.count_nonzero(finite_x < 0.0))) > 0:
            reasons.append(
                "x must be non-negative (the denominator coefficients are bounded below by "
                f"zero so the curve has no pole for x >= 0): found {_count(n, 'negative value')}"
            )
        finite = np.isfinite(ya)
        if reciprocal_denominator and finite.any():
            reasons.extend(_reciprocal_reasons(ya[finite]))
        return reasons

    return check


# -- the models -------------------------------------------------------------------------

_TINY = 1e-12
"""Positive lower bound for ``a`` and ``b`` of ``x/(a+b*x)``: at ``a = b = 0`` and ``x = 0`` the
model is 0/0 (NaN), so both are kept strictly positive."""

_P_RECIPROCAL_OFFSET = (
    Parameter("a", "Level that y approaches as |x| grows (the horizontal asymptote)"),
    Parameter("b", "Coefficient of 1/x: y departs from the asymptote by b/x"),
)
_P_RECIPROCAL_LINEAR = (
    Parameter("a", "Reciprocal of y at x = 0 (the denominator's constant term)"),
    Parameter("b", "Change of 1/y per unit x (slope of the denominator)"),
)
_P_X_OVER_LINEAR = (
    Parameter("a", "Initial slope reciprocal: y is about x/a for small x", lower=_TINY),
    Parameter("b", "Reciprocal of the plateau: y approaches 1/b at large x", lower=_TINY),
)
_P_QUADRATIC_OFFSET = (
    Parameter("a", "Level that y approaches as |x| grows (the horizontal asymptote)"),
    Parameter("b", "Coefficient of 1/x (first-order departure from the asymptote)"),
    Parameter("c", "Coefficient of 1/x**2 (second-order departure from the asymptote)"),
)
_P_1_1 = (
    Parameter("a", "Value of y at x = 0 (numerator constant term)"),
    Parameter("b", "Numerator slope: contribution of x to the numerator"),
    Parameter("c", "Denominator slope: how fast the curve saturates (non-negative)", lower=0.0),
)
_P_2_1 = (
    Parameter("a", "Value of y at x = 0 (numerator constant term)"),
    Parameter("b", "Numerator coefficient of x"),
    Parameter("c", "Numerator coefficient of x**2"),
    Parameter("d", "Denominator slope: how strongly the curve is damped (non-negative)", lower=0.0),
)
_P_AGNESI = (
    Parameter("a", "Height of the curve at x = 0"),
    Parameter("b", "Width control: y falls to a/2 at x = 1/sqrt(b) (non-negative)", lower=0.0),
)
_P_1_2 = (
    Parameter("a", "Value of y at x = 0 (numerator constant term)"),
    Parameter("b", "Numerator slope: contribution of x to the numerator"),
    Parameter("c", "Denominator coefficient of x (non-negative)", lower=0.0),
    Parameter("d", "Denominator coefficient of x**2 (non-negative)", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="rational_reciprocal_offset",
        name="Reciprocal with offset",
        category="rational",
        description="Rectangular hyperbola: a constant plus a term inversely proportional to x.",
        example=ModelExample({"a": 2.0, "b": 3.0}, x_range=(0.5, 10.0)),
        expression="a+b/x",
        parameters=_P_RECIPROCAL_OFFSET,
        initial_guess=_bespoke_guess(_P_RECIPROCAL_OFFSET, _reciprocal_offset),
        domain_check=_domain_check(x_nonzero_for="b/x"),
        references=(_NUMERICAL_RECIPES,),
        assumptions=(
            "x is never zero; y is linear in 1/x with constant-variance errors on y "
            "(fitted on the original scale)."
        ),
    ),
    ModelSpec(
        id="rational_reciprocal_linear",
        name="Reciprocal of a straight line",
        category="rational",
        description="The reciprocal of a linear function of x: y = 1/(a + b*x).",
        example=ModelExample({"a": 0.5, "b": 0.25}, x_range=(0.0, 10.0)),
        expression="1/(a+b*x)",
        parameters=_P_RECIPROCAL_LINEAR,
        initial_guess=_bespoke_guess(_P_RECIPROCAL_LINEAR, _reciprocal_linear),
        domain_check=_domain_check(reciprocal_denominator=True),
        references=(_DRAPER_SMITH,),
        assumptions=(
            "The denominator a + b*x does not change sign or reach zero inside the data range, "
            "so y keeps one sign and never reaches zero; the parameters are unbounded because the "
            "curve may rise (b < 0) or fall (b > 0). Errors are additive on y (not on 1/y)."
        ),
    ),
    ModelSpec(
        id="rational_x_over_linear",
        name="x over a straight line",
        category="rational",
        description="Hyperbolic saturation written as y = x/(a + b*x), zero at x = 0.",
        example=ModelExample({"a": 0.5, "b": 0.2}, x_range=(0.0, 20.0)),
        expression="x/(a+b*x)",
        parameters=_P_X_OVER_LINEAR,
        initial_guess=_bespoke_guess(_P_X_OVER_LINEAR, _x_over_linear),
        domain_check=_domain_check(x_non_negative=True),
        references=("Michaelis L., Menten M.L. (1913) Biochem. Z. 49:333-369",),
        assumptions=(
            "x is non-negative and y rises hyperbolically from zero to the plateau 1/b; the same "
            "curve as Michaelis-Menten with vmax = 1/b and km = a/b."
        ),
    ),
    ModelSpec(
        id="rational_reciprocal_quadratic_offset",
        name="Inverse-power series",
        category="rational",
        description="A constant plus terms in 1/x and 1/x**2: y = a + b/x + c/x**2.",
        example=ModelExample({"a": 1.0, "b": 2.0, "c": 3.0}, x_range=(0.4, 8.0)),
        expression="a+b/x+c/x**2",
        parameters=_P_QUADRATIC_OFFSET,
        initial_guess=_bespoke_guess(_P_QUADRATIC_OFFSET, _reciprocal_quadratic_offset),
        domain_check=_domain_check(x_nonzero_for="b/x and c/x**2"),
        references=(_NUMERICAL_RECIPES,),
        assumptions=(
            "x is never zero; y is a quadratic polynomial in 1/x with constant-variance errors."
        ),
    ),
    ModelSpec(
        id="rational_1_1",
        name="Rational [1/1]",
        category="rational",
        description="Ratio of two straight lines: y = (a + b*x)/(1 + c*x).",
        example=ModelExample({"a": 1.0, "b": 2.0, "c": 0.5}, x_range=(0.0, 10.0)),
        expression="(a+b*x)/(1+c*x)",
        parameters=_P_1_1,
        initial_guess=_bespoke_guess(_P_1_1, _pade(1, 1)),
        domain_check=_domain_check(x_non_negative=True),
        references=(_BAKER,),
        assumptions=(
            "x is non-negative and c >= 0, so the denominator never vanishes; y tends to b/c "
            "at large x."
        ),
    ),
    ModelSpec(
        id="rational_2_1",
        name="Rational [2/1]",
        category="rational",
        description="Quadratic over linear: y = (a + b*x + c*x**2)/(1 + d*x).",
        example=ModelExample({"a": 1.0, "b": 0.5, "c": 0.3, "d": 0.4}, x_range=(0.0, 8.0)),
        expression="(a+b*x+c*x**2)/(1+d*x)",
        parameters=_P_2_1,
        initial_guess=_bespoke_guess(_P_2_1, _pade(2, 1)),
        domain_check=_domain_check(x_non_negative=True),
        references=(_BAKER,),
        assumptions=(
            "x is non-negative and d >= 0, so the denominator never vanishes; y grows "
            "linearly at large x with slope c/d."
        ),
    ),
    ModelSpec(
        id="rational_agnesi",
        name="Witch of Agnesi (scaled)",
        category="rational",
        description="Symmetric bell-shaped curve with heavy tails: y = a/(1 + b*x**2).",
        example=ModelExample({"a": 5.0, "b": 0.8}, x_range=(-4.0, 4.0)),
        expression="a/(1+b*x**2)",
        parameters=_P_AGNESI,
        initial_guess=_bespoke_guess(_P_AGNESI, _agnesi),
        domain_check=_domain_check(),
        references=(
            "Agnesi M.G. (1748) Instituzioni analitiche ad uso della gioventu italiana, Milan",
        ),
        assumptions=(
            "The curve is centred on x = 0 and symmetric; shift x first if the peak is "
            "elsewhere. Errors are additive on y."
        ),
        aliases=("witch_of_agnesi",),
    ),
    ModelSpec(
        id="rational_1_2",
        name="Rational [1/2]",
        category="rational",
        description="Linear over quadratic: y = (a + b*x)/(1 + c*x + d*x**2).",
        example=ModelExample({"a": 1.0, "b": 3.0, "c": 0.4, "d": 0.15}, x_range=(0.0, 12.0)),
        expression="(a+b*x)/(1+c*x+d*x**2)",
        parameters=_P_1_2,
        initial_guess=_bespoke_guess(_P_1_2, _pade(1, 2)),
        domain_check=_domain_check(x_non_negative=True),
        references=(_BAKER,),
        assumptions=(
            "x is non-negative and c, d >= 0, so the denominator never vanishes; y decays "
            "like b/(d*x) at large x and may pass through a single maximum."
        ),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # x/(a + b*x) with a = km/vmax and b = 1/vmax is vmax*x/(km + x).
    ("rational_x_over_linear", "michaelis_menten"),
    # Declared ahead of the saturation and peak families; ignored until both ids exist (the
    # ids are provisional: the owner of those families corrects them if they differ).
    # (a + b*x)/(1 + c*x) = b/c + (a - b/c)/(1 + c*x): a hyperbola with an offset.
    ("rational_1_1", "rectangular_hyperbola_offset"),
    # a/(1 + b*x**2) is a Lorentzian centred at x = 0 with half-width 1/sqrt(b).
    ("rational_agnesi", "lorentzian_centered"),
)
