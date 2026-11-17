"""Durability models: chloride ingress, carbonation, absorption and gas permeability.

The chloride profile is the complementary-error-function solution of Fick's second law for a
semi-infinite medium with a constant surface concentration. Its diffusion term is the product
``D_t = D * t_exposure`` (m²): a single profile constrains only that product, so fitting D and
the exposure time separately would be a reparametrisation with an unidentifiable scale and the
model documents the product instead. Carbonation follows the usual power law in time,
sorptivity and absorption the root-time capillary law, and permeability the Klinkenberg
slip correction. Every parameter carries its unit and x is the physical quantity named in the
model's assumptions.
"""

import math

import numpy as np
from scipy.special import erfc, erfinv

from curvestudio.models import ModelExample, ModelSpec, Parameter
from curvestudio_engineering.families._common import (
    FloatArray,
    bespoke_guess,
    domain_check,
    least_squares,
    log_fit,
    refine_around,
    scale,
    span,
    unexplained,
)
from curvestudio_engineering.notice import engineering_model

_CRANK = "Crank J. (1975) The Mathematics of Diffusion, 2nd ed., Oxford University Press"
_COLLEPARDI = (
    "Collepardi M., Marcialis A., Turriziani R. (1970) The kinetics of penetration of "
    "chloride ions into the concrete, Il Cemento 67(4):157-164"
)
_PAPADAKIS = (
    "Papadakis V.G., Vayenas C.G., Fardis M.N. (1991) Fundamental modeling and experimental "
    "investigation of concrete carbonation, ACI Materials Journal 88(4):363-373"
)
_HALL = (
    "Hall C. (1989) Water sorptivity of mortars and concretes: a review, Magazine of Concrete "
    "Research 41(147):51-61"
)
_ASTM_C1585 = (
    "ASTM C1585 Standard Test Method for Measurement of Rate of Absorption of Water by "
    "Hydraulic-Cement Concretes"
)
_KLINKENBERG = (
    "Klinkenberg L.J. (1941) The permeability of porous media to liquids and gases, Drilling "
    "and Production Practice, American Petroleum Institute"
)
_NEVILLE = "Neville A.M. (2011) Properties of Concrete, 5th ed., Pearson Education"

_COARSE_STEPS = 41
_FINE_STEPS = 21

_ERFC_HALF = float(erfinv(0.5))
"""The argument at which ``erfc`` equals one half; used to read a diffusion scale off the data."""


# -- initial guesses --------------------------------------------------------------------


def _chloride_line(
    x: FloatArray, y: FloatArray, depth_scale: float
) -> tuple[float, float, float] | None:
    """Best ``C_i`` and ``C_s`` for one ``D_t``; ``(score, C_i, C_s)`` or ``None``.

    With ``D_t`` fixed the profile is linear in ``C_i`` and ``C_s``, so a two-column least
    squares solves both exactly.
    """
    if not (math.isfinite(depth_scale) and depth_scale > 0.0):
        return None
    with np.errstate(all="ignore"):
        segment = erfc(x / (2.0 * np.sqrt(depth_scale)))
    coefficients = least_squares([1.0 - segment, segment], y, np.ones(x.size))
    if coefficients is None:
        return None
    initial, surface = float(coefficients[0]), float(coefficients[1])
    curve = initial + (surface - initial) * segment
    return unexplained(x, y, curve), initial, surface


def _chloride_profile_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Read a diffusion scale from the half-concentration depth, then scan and refine it."""
    if x.size < 3:
        return {"C_i": 0.05, "C_s": 0.5, "D_t": 100.0}
    lowest = float(np.min(y))
    highest = float(np.max(y))
    if not highest > lowest:
        return {"C_i": lowest, "C_s": highest, "D_t": max(span(x) ** 2, 1e-12)}
    level = 0.5 * lowest + 0.5 * highest
    below = np.nonzero(y <= level)[0]
    crossing = float(np.median(x))
    if below.size and int(below[0]) > 0:
        i = int(below[0])
        y0, y1 = float(y[i - 1]), float(y[i])
        x0, x1 = float(x[i - 1]), float(x[i])
        if y1 != y0:
            crossing = x0 + (level - y0) / (y1 - y0) * (x1 - x0)
    if crossing > 0.0:
        depth_scale = (crossing / (2.0 * _ERFC_HALF)) ** 2
    else:
        depth_scale = max(span(x) ** 2, 1e-12)
    best: tuple[float, float, float, float] | None = None
    candidates = np.geomspace(max(depth_scale * 0.1, 1e-300), depth_scale * 10.0, _COARSE_STEPS)
    for stage in range(3):
        for candidate in candidates:
            scored = _chloride_line(x, y, float(candidate))
            if scored is None:
                continue
            score, initial, surface = scored
            if best is None or score < best[0]:
                best = (score, float(candidate), initial, surface)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.10 if stage == 0 else 0.02, steps=_FINE_STEPS)
    if best is not None:
        return {"C_i": best[2], "C_s": best[3], "D_t": best[1]}
    return {"C_i": lowest, "C_s": highest, "D_t": max(depth_scale, 1e-12)}


def _carbonation_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-log line ``log y = log k + n*log x`` weighted by ``y**2``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    line = log_fit(np.log(xp), yp, 1.0) if xp.size >= 2 else None
    if line is not None and line[0] > 0.0:
        coefficient = math.exp(min(line[1], 700.0))
        if math.isfinite(coefficient) and coefficient > 0.0:
            return {"k": coefficient, "n": line[0]}
    top = float(np.max(y)) if y.size else 0.0
    middle = float(np.median(xp)) if xp.size else 1.0
    return {
        "k": (top if top > 0.0 else scale(y)) / max(math.sqrt(abs(middle)), 1e-12),
        "n": 0.5,
    }


def _sorptivity_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Ordinary least squares ``y = A + S*sqrt(x)``."""
    non_negative = x >= 0.0
    xk, yk = x[non_negative], y[non_negative]
    if xk.size >= 2:
        coefficients = least_squares([np.ones(xk.size), np.sqrt(xk)], yk, np.ones(xk.size))
        if coefficients is not None:
            offset, sorptivity = float(coefficients[0]), float(coefficients[1])
            if math.isfinite(offset) and math.isfinite(sorptivity):
                return {"S": max(sorptivity, 0.0), "A": max(offset, 0.0)}
    top = float(np.max(y)) if y.size else 0.0
    return {"S": top if top > 0.0 else scale(y), "A": 0.0}


def _water_absorption_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Ordinary least squares ``y = A + S*sqrt(x) + c*x``."""
    non_negative = x >= 0.0
    xk, yk = x[non_negative], y[non_negative]
    if xk.size >= 3:
        coefficients = least_squares([np.ones(xk.size), np.sqrt(xk), xk], yk, np.ones(xk.size))
        if coefficients is not None:
            offset, sorptivity, linear = (float(c) for c in coefficients)
            if all(math.isfinite(v) for v in (offset, sorptivity, linear)):
                return {
                    "A": max(offset, 0.0),
                    "S": max(sorptivity, 0.0),
                    "c": max(linear, 0.0),
                }
    top = float(np.max(y)) if y.size else 0.0
    return {
        "A": 0.0,
        "S": top if top > 0.0 else scale(y),
        "c": 0.0,
    }


def _permeability_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Line ``y = k_l + (k_l*b)/x``: the intercept is ``k_l`` and the slope ``k_l*b``."""
    positive = x > 0.0
    xk, yk = x[positive], y[positive]
    if xk.size >= 2:
        coefficients = least_squares([np.ones(xk.size), 1.0 / xk], yk, np.ones(xk.size))
        if coefficients is not None:
            intrinsic, slope = float(coefficients[0]), float(coefficients[1])
            if math.isfinite(intrinsic) and intrinsic > 0.0 and math.isfinite(slope):
                return {"k_l": intrinsic, "b": max(slope / intrinsic, 0.0)}
    middle = float(np.median(xk)) if xk.size else 1.0
    return {"k_l": float(np.median(y)) if y.size else 1.0, "b": abs(middle)}


# -- the models -------------------------------------------------------------------------

_P_CHLORIDE = (
    Parameter(
        "C_i", "Initial chloride content of the concrete", "% by mass of concrete", lower=0.0
    ),
    Parameter("C_s", "Surface chloride content held constant", "% by mass of concrete", lower=0.0),
    Parameter(
        "D_t",
        "Diffusion-time product D*t_exposure (fitted as one parameter)",
        "mm² (= D·t_exposure)",
        lower=0.0,
    ),
)
_P_CARBONATION = (
    Parameter("k", "Carbonation coefficient", "mm/year^n", lower=0.0),
    Parameter("n", "Time exponent (about 0.5 for diffusion control)", "dimensionless", lower=0.0),
)
_P_SORPTIVITY = (
    Parameter("S", "Sorptivity (capillary absorption coefficient)", "mm/h^0.5", lower=0.0),
    Parameter("A", "Initial (surface) absorption at zero time", "mm", lower=0.0),
)
_P_WATER_ABSORPTION = (
    Parameter("A", "Initial (surface) absorption at zero time", "mm", lower=0.0),
    Parameter("S", "Capillary absorption coefficient", "mm/h^0.5", lower=0.0),
    Parameter("c", "Slow linear absorption term", "mm/h", lower=0.0),
)
_P_PERMEABILITY = (
    Parameter("k_l", "Intrinsic (liquid-equivalent) permeability", "10^-16 m²", lower=0.0),
    Parameter("b", "Klinkenberg slip factor", "bar", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    engineering_model(
        id="engineering_durability_chloride_profile",
        name="Chloride concentration profile",
        category="engineering.durability",
        description=(
            "Chloride content against depth from the complementary-error-function solution."
        ),
        expression="C_i+(C_s-C_i)*erfc(x/(2*sqrt(D_t)))",
        parameters=_P_CHLORIDE,
        initial_guess=bespoke_guess(_P_CHLORIDE, _chloride_profile_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_CRANK, _COLLEPARDI),
        assumptions=(
            "One-dimensional Fickian diffusion in a semi-infinite, homogeneous concrete from a "
            "constant surface concentration; x is depth in millimetres (0 at the exposed "
            "surface). Only the product D_t = D*t_exposure of the effective diffusion "
            "coefficient and the exposure time is identifiable from one profile, so it is the "
            "fitted parameter, in mm²; report D by fixing the exposure time. Binding, cracks "
            "and convection are outside the model."
        ),
        example=ModelExample({"C_i": 0.05, "C_s": 0.5, "D_t": 100.0}, x_range=(0.0, 50.0)),
    ),
    engineering_model(
        id="engineering_durability_carbonation",
        name="Carbonation depth development",
        category="engineering.durability",
        description="Carbonation depth as a power law in exposure time.",
        expression="k*x**n",
        parameters=_P_CARBONATION,
        initial_guess=bespoke_guess(_P_CARBONATION, _carbonation_guess),
        domain_check=domain_check(
            x_positive_for="k*x**n (its parameter derivative is not defined at x <= 0)"
        ),
        references=(_PAPADAKIS, _NEVILLE),
        assumptions=(
            "One concrete, one relative-humidity and CO2 environment, measured from the same "
            "surface; x is exposure time in years, k the carbonation coefficient and n about "
            "0.5 when CO2 diffusion controls the depth. Rain, cracks and a changing environment "
            "break the single power law."
        ),
        example=ModelExample({"k": 2.5, "n": 0.55}, x_range=(0.5, 50.0)),
    ),
    engineering_model(
        id="engineering_durability_sorptivity",
        name="Water sorptivity",
        category="engineering.durability",
        description="Cumulative capillary water absorption against the square root of time.",
        expression="S*sqrt(x)+A",
        parameters=_P_SORPTIVITY,
        initial_guess=bespoke_guess(_P_SORPTIVITY, _sorptivity_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_HALL, _ASTM_C1585),
        assumptions=(
            "One-dimensional capillary absorption into a dry, preconditioned specimen; x is "
            "time in hours, S the sorptivity in mm/h^0.5 and A the initial surface absorption "
            "in mm. The root-time law holds while the wetting front advances and the specimen "
            "stays unsaturated."
        ),
        example=ModelExample({"S": 0.25, "A": 0.3}, x_range=(0.25, 24.0)),
    ),
    engineering_model(
        id="engineering_durability_water_absorption",
        name="Water absorption",
        category="engineering.durability",
        description=(
            "Absorption with an initial fill, a capillary root-time term and a slow linear term."
        ),
        expression="A+S*sqrt(x)+c*x",
        parameters=_P_WATER_ABSORPTION,
        initial_guess=bespoke_guess(_P_WATER_ABSORPTION, _water_absorption_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_HALL, _NEVILLE),
        assumptions=(
            "Immersion of a dry specimen; x is immersion time in hours, A the initial surface "
            "absorption, S the capillary term and c a slow linear uptake by air displacement "
            "or continued saturation. The three terms are empirical: water absorption is not "
            "strictly separable into them."
        ),
        example=ModelExample({"A": 0.2, "S": 0.3, "c": 0.005}, x_range=(0.25, 48.0)),
    ),
    engineering_model(
        id="engineering_durability_permeability",
        name="Klinkenberg gas permeability",
        category="engineering.durability",
        description=(
            "Apparent gas permeability against mean pressure with the Klinkenberg slip correction."
        ),
        expression="k_l*(1+b/x)",
        parameters=_P_PERMEABILITY,
        initial_guess=bespoke_guess(_P_PERMEABILITY, _permeability_guess),
        domain_check=domain_check(x_positive_for="b/x"),
        references=(_KLINKENBERG, _NEVILLE),
        assumptions=(
            "Steady-state gas flow through a dry specimen at one confining pressure; x is the "
            "mean gas pressure in bar, k_l the intrinsic permeability in 10^-16 m² and b the "
            "slip factor in bar. Extrapolating 1/x to zero pressure is only valid while the "
            "slip correction is linear; moisture and microcracks change the answer."
        ),
        example=ModelExample({"k_l": 1.0, "b": 0.5}, x_range=(0.5, 4.0)),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # k*x**n is the core power law with the same parameters.
    ("engineering_durability_carbonation", "power"),
    # k_l*(1+b/x) is a+b/x with a = k_l and b = k_l*b.
    ("engineering_durability_permeability", "rational_reciprocal_offset"),
)
