"""Strength models: compressive strength against age and against mix proportions.

The age models are the exponential, hyperbolic (the ACI 209 form), logistic and inverse forms
people actually report; the mix-proportion models are Abrams' water/cement law, Bolomey's
linear law, Feret's compacity law and two replacement-level relations. Every model is fitted
on the original strength scale, and each parameter documents its unit in its ``unit`` field.

Two of these are reparametrisations of models in the general catalogue and are declared in
``EQUIVALENT_PAIRS``: the hyperbolic form with a characteristic time is Michaelis-Menten, the
ACI 209 form is the core's ``rational_x_over_linear``, the logistic form is the core's
``logistic3``, Abrams' law is an exponential in the water/cement ratio (``exp_growth``), and
the cementitious-content curve is the core's exponential with offset.
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
    span,
    three_point_decay,
    unexplained,
    weighted_line,
)
from curvestudio_engineering.notice import engineering_model

_NEVILLE = "Neville A.M. (2011) Properties of Concrete, 5th ed., Pearson Education"
_MEHTA_MONTEIRO = (
    "Mehta P.K., Monteiro P.J.M. (2014) Concrete: Microstructure, Properties, and Materials, "
    "4th ed., McGraw-Hill"
)
_ACI_209 = (
    "ACI Committee 209 (1992) Prediction of Creep, Shrinkage, and Temperature Effects in "
    "Concrete Structures (ACI 209R-92), American Concrete Institute"
)
_ABRAMS = (
    "Abrams D.A. (1918) Design of Concrete Mixtures, Bulletin No. 1, Structural Materials "
    "Research Laboratory, Lewis Institute, Chicago"
)
_FERET = (
    "Feret R. (1892) Sur la compacité des mortiers hydrauliques, Annales des Ponts et Chaussées"
)
_CEB_FIP = (
    "Comité Euro-International du Béton (1993) CEB-FIP Model Code 1990: Design Code, Thomas Telford"
)
_VERHULST = (
    "Verhulst P.F. (1838) Notice sur la loi que la population poursuit dans son accroissement, "
    "Correspondance Mathématique et Physique 10:113-121"
)
_LOTHENBACH = (
    "Lothenbach B., Scrivener K., Hooton R.D. (2011) Supplementary cementitious materials, "
    "Cement and Concrete Research 41(12):1244-1256"
)

_RATE_LOW = 1e-3
_RATE_HIGH = 1e3
"""Believable strength-gain rates in units of ``1 / x-span`` for the logistic guess."""


# -- initial guesses --------------------------------------------------------------------


def _exponential_age_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the asymptote above the data; at the true one ``log(1-y/f_cu)`` is linear in x.

    The scan is robust to noise in a way a three-point formula is not: every point enters
    the linear fit (weighted by ``(1-y/f_cu)**2``, the delta-method weight of the logarithm)
    and the curve's own residual decides between candidates.
    """
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float] | None = None
    if x.size >= 3 and top > 0.0:
        candidates = limits_above(y, factor=8.0, steps=41)
        for stage in range(3):
            for limit in candidates:
                residual = 1.0 - y / limit
                keep = residual > 0.0
                if int(np.count_nonzero(keep)) < 3:
                    continue
                line = weighted_line(x[keep], np.log(residual[keep]), residual[keep] ** 2)
                if line is None or line[0] >= 0.0:
                    continue
                rate = -line[0]
                curve = limit * (1.0 - np.exp(-rate * x))
                score = unexplained(x, y, curve)
                if best is None or score < best[0]:
                    best = (score, float(limit), rate)
            if best is None:
                break
            candidates = refine_around(best[1], spread=0.15 if stage == 0 else 0.03, steps=21)
    if best is not None:
        return {"f_cu": best[1], "k": best[2]}
    return {
        "f_cu": top if top > 0.0 else scale(y),
        "k": 1.0 / max(span(x), 1e-12),
    }


def _hyperbolic_age_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Hanes-Woolf line ``x/y = k/f_cu + x/f_cu`` weighted by ``y**4/x**2``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    if xp.size >= 2:
        line = weighted_line(xp, xp / yp, yp**4 / xp**2)
        if line is not None and line[0] > 0.0 and line[1] > 0.0:
            f_cu = 1.0 / line[0]
            return {"f_cu": f_cu, "k": line[1] * f_cu}
    top = float(np.max(y)) if y.size else 0.0
    return {
        "f_cu": (top if top > 0.0 else scale(y)) * 1.2,
        "k": float(np.median(x)) if x.size else 1.0,
    }


def _logistic_age_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the upper plateau; for each candidate the logit line gives ``r`` and ``t_0``.

    A plateau guessed too high flattens the logit slope and biases both the rate and the
    midpoint, so the plateau is chosen by the logistic curve's own residual rather than by
    a fixed margin over the largest observation.
    """
    if not y.size:
        return {"f_cu": 1.0, "r": 1.0, "t_0": 0.0}
    top = float(np.max(y))
    x_span = span(x)
    best: tuple[float, float, float, float] | None = None
    if x.size >= 3 and top > 0.0:
        candidates = limits_above(y, factor=4.0, steps=41)
        for stage in range(3):
            for ceiling in candidates:
                p = y / ceiling
                band = (p > 0.02) & (p < 0.98)
                if int(np.count_nonzero(band)) < 2:
                    continue
                pb = p[band]
                line = weighted_line(
                    x[band], np.log(pb / (1.0 - pb)), (ceiling * pb * (1.0 - pb)) ** 2
                )
                if line is None or line[0] <= 0.0:
                    continue
                rate = line[0]
                midpoint = -line[1] / line[0]
                curve = ceiling / (1.0 + np.exp(-rate * (x - midpoint)))
                score = unexplained(x, y, curve)
                if best is None or score < best[0]:
                    best = (score, float(ceiling), rate, midpoint)
            if best is None:
                break
            candidates = refine_around(best[1], spread=0.15 if stage == 0 else 0.03, steps=21)
    if best is not None:
        rate = min(max(best[2], _RATE_LOW / x_span), _RATE_HIGH / x_span)
        return {"f_cu": best[1], "r": rate, "t_0": best[3]}
    ceiling = top + 0.05 * float(np.ptp(y)) if top > 0.0 else scale(y)
    middle = float(x[int(np.argmin(np.abs(y - 0.5 * ceiling)))])
    return {"f_cu": ceiling, "r": 8.0 / x_span, "t_0": middle}


def _aci209_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Reciprocal line ``1/y = a*(1/x) + b`` weighted by ``y**4``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    if xp.size >= 2:
        line = weighted_line(1.0 / xp, 1.0 / yp, yp**4)
        if line is not None and line[0] > 0.0 and line[1] > 0.0:
            return {"a": line[0], "b": line[1]}
    top = float(np.max(y)) if y.size else 0.0
    return {
        "a": span(x) if x.size else 1.0,
        "b": 1.0 / (top if top > 0.0 else scale(y)),
    }


def _abrams_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-linear line ``log y = log A - (log B) * x`` weighted by ``y**2``."""
    line = log_fit(x, y, 1.0)
    if line is not None:
        slope, intercept = line
        amplitude = math.exp(min(intercept, 700.0))
        base = math.exp(min(max(-slope, 0.0), 700.0))
        if math.isfinite(amplitude) and amplitude > 0.0 and math.isfinite(base):
            return {"A": amplitude, "B": max(base, 1.0 + 1e-9)}
    top = float(np.max(y)) if y.size else 0.0
    return {"A": top if top > 0.0 else scale(y), "B": 2.0}


def _bolomey_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Ordinary least squares ``y = K*x - K*a``: the slope is K and the intercept -K*a."""
    if x.size >= 2:
        line = weighted_line(x, y, np.ones(x.size))
        if line is not None and line[0] > 0.0:
            offset = -line[1] / line[0]
            if math.isfinite(offset):
                return {"K": line[0], "a": max(offset, 0.0)}
    top = float(np.max(np.abs(y))) if y.size else 0.0
    return {
        "K": (top if top > 0.0 else 1.0) / max(span(x), 1e-12),
        "a": 0.0,
    }


def _feret_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Least squares through the origin ``y = K*x**2``: ``K = sum(y*x**2) / sum(x**4)``."""
    keep = (x != 0.0) & np.isfinite(x) & np.isfinite(y)
    xk, yk = x[keep], y[keep]
    if xk.size >= 1:
        denominator = float(np.sum(xk**4))
        numerator = float(np.sum(yk * xk**2))
        if math.isfinite(denominator) and denominator > 0.0 and math.isfinite(numerator):
            stiffness = numerator / denominator
            if math.isfinite(stiffness) and stiffness > 0.0:
                return {"K": stiffness}
    top = float(np.max(y)) if y.size else 0.0
    return {"K": (top if top > 0.0 else 1.0) / max(span(x) ** 2, 1e-12)}


def _cementitious_content_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Three-point offset exponential: ``f_0 + f_u*(1-exp(-k*x))``."""
    three = three_point_decay(x, y)
    if three is not None:
        limit, rate, amplitude = three
        if limit > 0.0 and rate > 0.0 and amplitude > 0.0:
            return {"f_0": max(limit - amplitude, 0.0), "f_u": amplitude, "k": rate}
    top = float(np.max(y)) if y.size else 0.0
    bottom = float(np.min(y)) if y.size else 0.0
    f_0 = max(bottom, 0.0)
    f_u = top - f_0
    if f_u <= 0.0:
        f_u = scale(y)
    return {"f_0": f_0, "f_u": f_u, "k": 1.0 / max(span(x), 1e-12)}


def _scm_replacement_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-linear line ``log y = log f_0 + b*log(1-x)`` weighted by ``y**2``."""
    keep = (x < 1.0) & (y > 0.0)
    xk, yk = x[keep], y[keep]
    if xk.size >= 2:
        line = log_fit(np.log(1.0 - xk), yk, 1.0)
        if line is not None:
            slope, intercept = line
            amplitude = math.exp(min(intercept, 700.0))
            if math.isfinite(amplitude) and amplitude > 0.0 and slope >= 0.0:
                return {"f_0": amplitude, "b": slope}
    top = float(np.max(y)) if y.size else 0.0
    return {"f_0": top if top > 0.0 else scale(y), "b": 0.5}


# -- the models -------------------------------------------------------------------------

_P_EXPONENTIAL_AGE = (
    Parameter(
        "f_cu", "Asymptotic compressive strength as age grows without limit", "MPa", lower=0.0
    ),
    Parameter("k", "Rate constant of the exponential strength gain", "1/day", lower=0.0),
)
_P_HYPERBOLIC_AGE = (
    Parameter(
        "f_cu", "Asymptotic compressive strength as age grows without limit", "MPa", lower=0.0
    ),
    Parameter("k", "Age at which half of the asymptotic strength is reached", "days", lower=0.0),
)
_P_LOGISTIC_AGE = (
    Parameter("f_cu", "Upper plateau of the S-curve", "MPa", lower=0.0),
    Parameter("r", "Steepness of the S-curve at its midpoint", "1/day", lower=0.0),
    Parameter("t_0", "Age at the midpoint of the S-curve", "days"),
)
_P_ACI209 = (
    Parameter("a", "Empirical time constant of the hyperbolic form", "days", lower=0.0),
    Parameter("b", "Reciprocal of the asymptotic strength", "1/MPa", lower=0.0),
)
_P_ABRAMS = (
    Parameter("A", "Strength extrapolated to a water/cement ratio of zero", "MPa", lower=0.0),
    Parameter(
        "B",
        "Base of the exponential decay, greater than one so strength falls as water/cement rises",
        "dimensionless",
        lower=1.0,
    ),
)
_P_BOLOMEY = (
    Parameter("K", "Strength gained per unit of cement/water ratio", "MPa", lower=0.0),
    Parameter(
        "a",
        "Cement/water ratio at which the linear law gives zero strength",
        "dimensionless",
        lower=0.0,
    ),
)
_P_FERET = (
    Parameter("K", "Feret coefficient: strength at unit paste compacity", "MPa", lower=0.0),
)
_P_CEMENTITIOUS_CONTENT = (
    Parameter("f_0", "Strength at zero cementitious content (intercept)", "MPa", lower=0.0),
    Parameter(
        "f_u", "Additional strength reached as content grows without limit", "MPa", lower=0.0
    ),
    Parameter("k", "Rate constant against cementitious content", "m³/kg", lower=0.0),
)
_P_SCM = (
    Parameter("f_0", "Strength of the plain binder at the same age", "MPa", lower=0.0),
    Parameter("b", "Sensitivity of strength to the replacement level", "dimensionless", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    engineering_model(
        id="engineering_strength_exponential_age",
        name="Exponential strength development with age",
        category="engineering.strength",
        description=(
            "Compressive strength rising towards an ultimate value at a constant fractional rate."
        ),
        expression="f_cu*(1-exp(-k*x))",
        parameters=_P_EXPONENTIAL_AGE,
        initial_guess=bespoke_guess(_P_EXPONENTIAL_AGE, _exponential_age_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_NEVILLE, _CEB_FIP),
        assumptions=(
            "Constant moist curing at one temperature; strength starts at zero and approaches "
            "f_cu asymptotically with age x in days; the rate k does not change with age."
        ),
        example=ModelExample({"f_cu": 45.0, "k": 0.15}, x_range=(1.0, 60.0)),
    ),
    engineering_model(
        id="engineering_strength_hyperbolic_age",
        name="Hyperbolic strength development with age",
        category="engineering.strength",
        description="Compressive strength rising hyperbolically to an ultimate value with age.",
        expression="f_cu*x/(k+x)",
        parameters=_P_HYPERBOLIC_AGE,
        initial_guess=bespoke_guess(_P_HYPERBOLIC_AGE, _hyperbolic_age_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_NEVILLE, _ACI_209),
        assumptions=(
            "Strength is zero at zero age and follows a rectangular hyperbola; k is the age "
            "at which half of f_cu is reached; one curing temperature."
        ),
        example=ModelExample({"f_cu": 50.0, "k": 10.0}, x_range=(1.0, 90.0)),
    ),
    engineering_model(
        id="engineering_strength_logistic_age",
        name="Logistic strength development with age",
        category="engineering.strength",
        description="S-shaped compressive strength gain from zero to an upper plateau with age.",
        expression="f_cu/(1+exp(-r*(x-t_0)))",
        parameters=_P_LOGISTIC_AGE,
        initial_guess=bespoke_guess(_P_LOGISTIC_AGE, _logistic_age_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_VERHULST, _NEVILLE),
        assumptions=(
            "Symmetric logistic gain with midpoint t_0 and steepness r; the lower plateau is "
            "zero and the curve is symmetric about t_0, which real early-age data need not be."
        ),
        example=ModelExample({"f_cu": 50.0, "r": 0.12, "t_0": 14.0}, x_range=(1.0, 90.0)),
    ),
    engineering_model(
        id="engineering_strength_aci209",
        name="ACI 209 hyperbolic strength form",
        category="engineering.strength",
        description="The ACI 209 hyperbolic strength-age form t/(a + b*t).",
        expression="x/(a+b*x)",
        parameters=_P_ACI209,
        initial_guess=bespoke_guess(_P_ACI209, _aci209_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_ACI_209, _NEVILLE),
        assumptions=(
            "The empirical constants a and b of ACI 209R-92 for one cement type and curing "
            "regime; the asymptote is 1/b and the initial slope is 1/a; age x in days."
        ),
        example=ModelExample({"a": 8.0, "b": 0.02}, x_range=(1.0, 90.0)),
    ),
    engineering_model(
        id="engineering_strength_abrams",
        name="Abrams' water/cement law",
        category="engineering.strength",
        description="Compressive strength falling exponentially as the water/cement ratio rises.",
        expression="A/B**x",
        parameters=_P_ABRAMS,
        initial_guess=bespoke_guess(_P_ABRAMS, _abrams_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_ABRAMS, _NEVILLE),
        assumptions=(
            "Abrams' law for fully compacted concrete at a fixed age; x is the water/cement "
            "ratio by mass, A the extrapolated strength at x = 0 and B > 1 the decay base."
        ),
        example=ModelExample({"A": 120.0, "B": 6.5}, x_range=(0.35, 0.65)),
    ),
    engineering_model(
        id="engineering_strength_bolomey",
        name="Bolomey's linear strength law",
        category="engineering.strength",
        description="Compressive strength as a straight line in the cement/water ratio.",
        expression="K*(x-a)",
        parameters=_P_BOLOMEY,
        initial_guess=bespoke_guess(_P_BOLOMEY, _bolomey_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_NEVILLE, _MEHTA_MONTEIRO),
        assumptions=(
            "Bolomey's linear relation at a fixed age and consistency; x is the cement/water "
            "ratio by mass and a the ratio at which the line gives zero strength (about 0.5 "
            "for ordinary portland cement concrete)."
        ),
        example=ModelExample({"K": 22.0, "a": 0.45}, x_range=(1.4, 2.6)),
    ),
    engineering_model(
        id="engineering_strength_feret",
        name="Feret's compacity law",
        category="engineering.strength",
        description="Compressive strength proportional to the square of the paste compacity.",
        expression="K*x**2",
        parameters=_P_FERET,
        initial_guess=bespoke_guess(_P_FERET, _feret_guess),
        domain_check=domain_check(x_non_negative=True, x_below=("paste compacity", 1.0)),
        references=(_FERET, _NEVILLE),
        assumptions=(
            "Feret's law with the classical exponent 2; x is the volume fraction of cement in "
            "the cement + water + air paste (the compacity), so it lies between 0 and 1; "
            "entrapped air and the aggregate are ignored."
        ),
        example=ModelExample({"K": 400.0}, x_range=(0.25, 0.42)),
    ),
    engineering_model(
        id="engineering_strength_cementitious_content",
        name="Strength against cementitious content",
        category="engineering.strength",
        description="Compressive strength rising with a saturating exponential in binder content.",
        expression="f_0+f_u*(1-exp(-k*x))",
        parameters=_P_CEMENTITIOUS_CONTENT,
        initial_guess=bespoke_guess(_P_CEMENTITIOUS_CONTENT, _cementitious_content_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_NEVILLE, _MEHTA_MONTEIRO),
        assumptions=(
            "At a fixed water/cement ratio, aggregate and age, strength rises from f_0 towards "
            "f_0 + f_u as cementitious content x (kg/m³) increases; the curve saturates "
            "because the paste volume, not the cement mass alone, controls strength."
        ),
        example=ModelExample({"f_0": 10.0, "f_u": 50.0, "k": 0.005}, x_range=(150.0, 450.0)),
    ),
    engineering_model(
        id="engineering_strength_scm_replacement",
        name="Strength against SCM replacement level",
        category="engineering.strength",
        description="Strength of a blended binder falling as the SCM replacement fraction rises.",
        expression="f_0*(1-x)**b",
        parameters=_P_SCM,
        initial_guess=bespoke_guess(_P_SCM, _scm_replacement_guess),
        domain_check=domain_check(x_non_negative=True, x_below=("replacement fraction", 1.0)),
        references=(_LOTHENBACH, _NEVILLE),
        assumptions=(
            "A fixed age, water/binder ratio and total binder content; x is the supplementary "
            "cementitious material mass fraction of the binder and b the sensitivity. Early "
            "ages and slow pozzolans need their own fit: one curve does not cover all ages."
        ),
        example=ModelExample({"f_0": 55.0, "b": 0.35}, x_range=(0.05, 0.6)),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # f_cu*x/(k+x) is vmax*x/(km+x) with vmax = f_cu and km = k.
    ("engineering_strength_hyperbolic_age", "michaelis_menten"),
    # f_cu*x/(k+x) is x/(a+b*x) with a = k/f_cu and b = 1/f_cu.
    ("engineering_strength_hyperbolic_age", "rational_x_over_linear"),
    # The ACI 209 form x/(a+b*x) is the core rational x over a straight line.
    ("engineering_strength_aci209", "rational_x_over_linear"),
    # Both hyperbolic forms describe the same curve with a different parameter meaning.
    ("engineering_strength_hyperbolic_age", "engineering_strength_aci209"),
    # f_cu/(1+exp(-r*(x-t_0))) is the core logistic3 with k = f_cu and x0 = t_0.
    ("engineering_strength_logistic_age", "logistic3"),
    # A/B**x is a*exp(b*x) with a = A and b = -ln(B).
    ("engineering_strength_abrams", "exp_growth"),
    # f_0 + f_u*(1-exp(-k*x)) is the core exponential with offset.
    ("engineering_strength_cementitious_content", "exp_offset"),
    ("engineering_strength_cementitious_content", "exp_decay_offset"),
)
