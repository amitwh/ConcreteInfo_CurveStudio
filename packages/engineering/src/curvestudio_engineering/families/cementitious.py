"""Cement and supplementary-cementitious-material models.

Cement strength development follows the CEB-FIP exponential form; the replacement
relationships express the strength of a blend at a fixed age relative to the plain binder;
the pozzolanic reaction degree and the heat of hydration follow saturating kinetics. Every
model is fitted on its original scale; x is always the ageing time, the replacement fraction
or the reaction time, and each parameter carries its unit.

None of these is a code acceptance equation: the replacement relations are fitted to one
binder, one age and one set of curing conditions at a time.
"""

import math

import numpy as np

from curvestudio.models import ModelExample, ModelSpec, Parameter
from curvestudio_engineering.families._common import (
    FloatArray,
    bespoke_guess,
    domain_check,
    limits_above,
    log_fit,
    refine_around,
    scale,
    unexplained,
    weighted_line,
)
from curvestudio_engineering.notice import engineering_model

_NEVILLE = "Neville A.M. (2011) Properties of Concrete, 5th ed., Pearson Education"
_MEHTA_MONTEIRO = (
    "Mehta P.K., Monteiro P.J.M. (2014) Concrete: Microstructure, Properties, and Materials, "
    "4th ed., McGraw-Hill"
)
_TAYLOR = "Taylor H.F.W. (1997) Cement Chemistry, 2nd ed., Thomas Telford"
_CEB_FIP = (
    "Comité Euro-International du Béton (1993) CEB-FIP Model Code 1990: Design Code, Thomas Telford"
)
_LOTHENBACH = (
    "Lothenbach B., Scrivener K., Hooton R.D. (2011) Supplementary cementitious materials, "
    "Cement and Concrete Research 41(12):1244-1256"
)

_COARSE_STEPS = 41
_FINE_STEPS = 21


# -- initial guesses --------------------------------------------------------------------


def _cement_strength_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Line ``log y = (log f_28 + s) - s*sqrt(28/x)`` weighted by ``y**2``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    line = None
    if xp.size >= 2:
        line = weighted_line(np.sqrt(28.0 / xp), np.log(yp), yp**2)
    if line is not None and line[0] < 0.0:
        coefficient = -line[0]
        strength_28 = math.exp(min(line[1] - coefficient, 700.0))
        if math.isfinite(strength_28) and strength_28 > 0.0 and math.isfinite(coefficient):
            return {"f_28": strength_28, "s": coefficient}
    top = float(np.max(y)) if y.size else 0.0
    return {"f_28": top if top > 0.0 else scale(y), "s": 0.25}


def _fly_ash_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Line ``log(1-y) = log a + b*log x`` weighted by ``(1-y)**2``."""
    keep = (x > 0.0) & (x < 1.0) & (y > 0.0) & (y < 1.0)
    if int(np.count_nonzero(keep)) >= 2:
        line = log_fit(np.log(x[keep]), 1.0 - y[keep], 1.0)
        if line is not None and line[0] > 0.0:
            loss = math.exp(min(line[1], 700.0))
            if math.isfinite(loss) and 0.0 < loss < 1.0:
                return {"a": loss, "b": line[0]}
    return {"a": 0.3, "b": 0.7}


def _ggbs_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Line ``x/(1-y) = b/a + x/a``: the slope is ``1/a`` and the intercept ``b/a``."""
    keep = (x > 0.0) & (y < 1.0)
    xk, yk = x[keep], y[keep]
    if xk.size >= 2:
        denominator = 1.0 - yk
        line = weighted_line(xk, xk / denominator, np.ones(xk.size))
        if line is not None and line[0] > 0.0 and line[1] >= 0.0:
            loss = 1.0 / line[0]
            half_level = line[1] * loss
            if math.isfinite(loss) and 0.0 < loss < 1.0 and math.isfinite(half_level):
                return {"a": loss, "b": half_level}
    return {"a": 0.35, "b": 0.3}


def _pozzolanic_score(
    x: FloatArray, y: FloatArray, ultimate: float
) -> tuple[float, float, float] | None:
    """Fit the Hill form for one reaction-degree limit through ``log(y/(R_u-y))``."""
    residual = ultimate - y
    keep = (residual > 0.0) & (y > 0.0) & (x > 0.0)
    if int(np.count_nonzero(keep)) < 2:
        return None
    line = weighted_line(
        np.log(x[keep]), np.log(y[keep] / residual[keep]), np.ones(int(np.count_nonzero(keep)))
    )
    if line is None or line[0] <= 0.0:
        return None
    exponent, intercept = line
    half_time = math.exp(min(max(-intercept / exponent, -700.0), 700.0))
    if not (math.isfinite(half_time) and half_time > 0.0):
        return None
    powered = np.power(x, exponent)
    curve = ultimate * powered / (np.power(half_time, exponent) + powered)
    return unexplained(x, y, curve), exponent, half_time


def _pozzolanic_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the reaction-degree limit (at most 1) and refine it twice."""
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    candidates = limits_above(
        y, factor=min(2.0, 1.0 / top) if top > 0.0 else 2.0, steps=_COARSE_STEPS
    )
    for stage in range(3):
        for ultimate in candidates:
            if ultimate > 1.0:
                continue
            scored = _pozzolanic_score(x, y, float(ultimate))
            if scored is None:
                continue
            score, exponent, half_time = scored
            if best is None or score < best[0]:
                best = (score, float(ultimate), exponent, half_time)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.12 if stage == 0 else 0.03, steps=_FINE_STEPS)
    if best is not None:
        return {"R_u": best[1], "k": best[3], "n": best[2]}
    middle = float(np.median(x)) if x.size else 1.0
    return {"R_u": min(top * 1.1, 0.99) if top > 0.0 else 0.85, "k": middle, "n": 1.0}


def _hydration_heat_score(
    x: FloatArray, y: FloatArray, ultimate: float
) -> tuple[float, float, float] | None:
    """Fit ``Q_u*exp(-(tau/x)**beta)`` for one ultimate heat through ``log(-log(y/Q_u))``."""
    keep = (y > 0.0) & (x > 0.0)
    if int(np.count_nonzero(keep)) < 2:
        return None
    ratio = y[keep] / ultimate
    keep2 = ratio < 1.0
    if int(np.count_nonzero(keep2)) < 2:
        return None
    xk = x[keep][keep2]
    transformed = -np.log(ratio[keep2])
    keep3 = transformed > 0.0
    if int(np.count_nonzero(keep3)) < 2:
        return None
    line = weighted_line(
        np.log(xk[keep3]), np.log(transformed[keep3]), np.ones(int(np.count_nonzero(keep3)))
    )
    if line is None or line[0] >= 0.0:
        return None
    exponent = -line[0]
    time_scale = math.exp(min(max(line[1] / exponent, -700.0), 700.0))
    if not (math.isfinite(time_scale) and time_scale > 0.0):
        return None
    curve = ultimate * np.exp(-np.power(time_scale / x, exponent))
    return unexplained(x, y, curve), time_scale, exponent


def _hydration_heat_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the ultimate heat above the data and refine the limit twice."""
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    candidates = limits_above(y, steps=_COARSE_STEPS)
    for stage in range(3):
        for ultimate in candidates:
            scored = _hydration_heat_score(x, y, float(ultimate))
            if scored is None:
                continue
            score, time_scale, exponent = scored
            if best is None or score < best[0]:
                best = (score, float(ultimate), time_scale, exponent)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.12 if stage == 0 else 0.03, steps=_FINE_STEPS)
    if best is not None:
        return {"Q_u": best[1], "tau": best[2], "beta": best[3]}
    middle = float(np.median(x)) if x.size else 1.0
    return {"Q_u": top * 1.2 if top > 0.0 else scale(y), "tau": middle, "beta": 1.0}


# -- the models -------------------------------------------------------------------------

_P_CEMENT_STRENGTH = (
    Parameter("f_28", "Compressive strength at 28 days", "MPa", lower=0.0),
    Parameter("s", "Cement-type coefficient of the CEB-FIP curve", "dimensionless", lower=0.0),
)
_P_FLY_ASH = (
    Parameter(
        "a", "Relative strength loss at full replacement", "dimensionless", lower=0.0, upper=1.0
    ),
    Parameter("b", "Curvature of the replacement relationship", "dimensionless", lower=0.0),
)
_P_GGBS = (
    Parameter(
        "a",
        "Maximum relative strength loss approached at high replacement",
        "dimensionless",
        lower=0.0,
        upper=1.0,
    ),
    Parameter(
        "b", "Replacement level at which half of the loss is reached", "dimensionless", lower=0.0
    ),
)
_P_POZZOLANIC = (
    Parameter(
        "R_u",
        "Ultimate pozzolanic reaction degree (at most 1)",
        "dimensionless",
        lower=0.0,
        upper=1.0,
    ),
    Parameter("k", "Time at which half of the reaction degree is reached", "days", lower=0.0),
    Parameter("n", "Hill exponent of the reaction-degree curve", "dimensionless", lower=0.0),
)
_P_HYDRATION_HEAT = (
    Parameter("Q_u", "Ultimate heat of hydration", "J/g", lower=0.0),
    Parameter("tau", "Time scale of the heat release", "hours", lower=0.0),
    Parameter("beta", "Shape exponent of the heat-release curve", "dimensionless", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    engineering_model(
        id="engineering_cementitious_cement_strength",
        name="Cement strength development (CEB-FIP)",
        category="engineering.cementitious",
        description="Mortar or concrete strength against age on the CEB-FIP exponential curve.",
        expression="f_28*exp(s*(1-sqrt(28/x)))",
        parameters=_P_CEMENT_STRENGTH,
        initial_guess=bespoke_guess(_P_CEMENT_STRENGTH, _cement_strength_guess),
        domain_check=domain_check(x_positive_for="sqrt(28/x)"),
        references=(_CEB_FIP, _NEVILLE),
        assumptions=(
            "One cement type and one curing temperature; x is age in days, f_28 the 28-day "
            "strength and s the cement-type coefficient (about 0.2 for rapid-hardening, 0.25 "
            "for normal and 0.38 for slow-hardening cement in CEB-FIP)."
        ),
        example=ModelExample({"f_28": 42.5, "s": 0.25}, x_range=(1.0, 90.0)),
    ),
    engineering_model(
        id="engineering_cementitious_fly_ash_replacement",
        name="Strength against fly-ash replacement",
        category="engineering.cementitious",
        description=(
            "Relative strength of a fly-ash blend falling as a power of the replacement level."
        ),
        expression="1-a*x**b",
        parameters=_P_FLY_ASH,
        initial_guess=bespoke_guess(_P_FLY_ASH, _fly_ash_guess),
        domain_check=domain_check(x_non_negative=True, x_below=("replacement fraction", 1.0)),
        references=(_LOTHENBACH, _NEVILLE),
        assumptions=(
            "A fixed age, water/binder ratio and total binder content; x is the fly-ash mass "
            "fraction of the binder, y the strength of the blend relative to the plain binder "
            "(1 at zero replacement). The relation does not transfer between ages: a fly-ash "
            "blend gains relative strength with time."
        ),
        example=ModelExample({"a": 0.5, "b": 0.8}, x_range=(0.1, 0.5)),
    ),
    engineering_model(
        id="engineering_cementitious_ggbs_replacement",
        name="Strength against GGBS replacement",
        category="engineering.cementitious",
        description="Relative strength of a GGBS blend falling hyperbolically with replacement.",
        expression="1-a*x/(b+x)",
        parameters=_P_GGBS,
        initial_guess=bespoke_guess(_P_GGBS, _ggbs_guess),
        domain_check=domain_check(x_non_negative=True, x_below=("replacement fraction", 1.0)),
        references=(_LOTHENBACH, _NEVILLE),
        assumptions=(
            "A fixed age, water/binder ratio and total binder content; x is the GGBS mass "
            "fraction of the binder, y the strength relative to the plain binder, a the maximum "
            "relative loss approached at high replacement and b the replacement of the half-loss."
        ),
        example=ModelExample({"a": 0.4, "b": 0.3}, x_range=(0.1, 0.7)),
    ),
    engineering_model(
        id="engineering_cementitious_pozzolanic_reactivity",
        name="Pozzolanic reaction degree",
        category="engineering.cementitious",
        description="Pozzolanic reaction degree rising as a Hill-type saturation with time.",
        expression="R_u*x**n/(k**n+x**n)",
        parameters=_P_POZZOLANIC,
        initial_guess=bespoke_guess(_P_POZZOLANIC, _pozzolanic_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_TAYLOR, _MEHTA_MONTEIRO),
        assumptions=(
            "One pozzolan, one lime or cement environment and one temperature; x is reaction "
            "time in days, R_u the ultimate reaction degree (bounded at 1), k the half-reaction "
            "time and n the Hill exponent. Reaction degree is measured independently (for "
            "example by selective dissolution or calorimetry), not inferred from strength."
        ),
        example=ModelExample({"R_u": 0.85, "k": 28.0, "n": 1.6}, x_range=(3.0, 180.0)),
    ),
    engineering_model(
        id="engineering_cementitious_hydration_heat",
        name="Hydration heat development",
        category="engineering.cementitious",
        description="Cumulative heat of hydration rising exponentially towards its ultimate value.",
        expression="Q_u*exp(-(tau/x)**beta)",
        parameters=_P_HYDRATION_HEAT,
        initial_guess=bespoke_guess(_P_HYDRATION_HEAT, _hydration_heat_guess),
        domain_check=domain_check(x_positive_for="tau/x"),
        references=(_TAYLOR, _MEHTA_MONTEIRO),
        assumptions=(
            "Isothermal calorimetry of one cement at one temperature; x is age in hours, Q_u "
            "the ultimate heat in J/g of cement, tau the time scale and beta the shape exponent. "
            "The curve ignores the sulphate-depletion peak of real calorimetry."
        ),
        example=ModelExample({"Q_u": 350.0, "tau": 12.0, "beta": 0.9}, x_range=(1.0, 72.0)),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = ()
"""The cementitious models have no reparametrisation partner in the shipped catalogue yet."""
