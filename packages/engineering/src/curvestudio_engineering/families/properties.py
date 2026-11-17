"""Concrete-property models: workability, non-destructive strength, deformation and maturity.

The families here are the standard empirical relations of concrete technology: slump against
admixture dosage, ultrasonic pulse velocity against strength (exponential and shifted power),
elastic modulus against compressive strength, hyperbolic shrinkage, creep, strength against
maturity and the Arrhenius temperature response. Every model is fitted on its original scale
and each parameter carries its unit.

The elastic-modulus power law is a reparametrisation of the core ``power`` model and is
declared in ``EQUIVALENT_PAIRS``; the maturity curve is the core exponential with offset.
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
    unexplained,
    weighted_line,
)
from curvestudio_engineering.notice import engineering_model

_NEVILLE = "Neville A.M. (2011) Properties of Concrete, 5th ed., Pearson Education"
_RAMACHANDRAN = (
    "Ramachandran V.S. (1995) Concrete Admixtures Handbook: Properties, Science, and "
    "Technology, 2nd ed., Noyes Publications"
)
_MALHOTRA_CARINO = (
    "Malhotra V.M., Carino N.J. (2004) Handbook on Nondestructive Testing of Concrete, "
    "2nd ed., CRC Press"
)
_ACI_318 = (
    "ACI Committee 318 (2019) Building Code Requirements for Structural Concrete (ACI 318-19) "
    "and Commentary, American Concrete Institute"
)
_EN_1992 = (
    "European Committee for Standardization (2004) Eurocode 2: Design of concrete structures "
    "- Part 1-1: General rules and rules for buildings (EN 1992-1-1), CEN"
)
_ACI_209 = (
    "ACI Committee 209 (1992) Prediction of Creep, Shrinkage, and Temperature Effects in "
    "Concrete Structures (ACI 209R-92), American Concrete Institute"
)
_BAZANT_BAWEJA = (
    "Bažant Z.P., Baweja S. (1995) Creep and shrinkage prediction model for analysis and "
    "design of concrete structures - Model B3, Materials and Structures 28:357-365"
)
_NEVILLE_DILGER_BROOKS = (
    "Neville A.M., Dilger W.H., Brooks J.J. (1983) Creep of Plain and Structural Concrete, "
    "Construction Press"
)
_FREIESLEBEN_HANSEN = (
    "Freiesleben Hansen P., Pedersen E.J. (1977) Maturity computer for controlled curing and "
    "hardening of concrete, Nordisk Betong 1:19-34"
)
_ASTM_C1074 = "ASTM C1074 Standard Practice for Estimating Concrete Strength by the Maturity Method"
_ARRHENIUS = (
    "Arrhenius S. (1889) Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker "
    "durch Säuren, Zeitschrift für Physikalische Chemie 4:226-248"
)

_COARSE_STEPS = 41
"""Candidates in a first grid scan over one shape parameter."""
_FINE_STEPS = 21
"""Candidates in each refinement scan."""


# -- initial guesses --------------------------------------------------------------------


def _stretched_saturation_score(
    x: FloatArray, y: FloatArray, limit: float
) -> tuple[float, float, float] | None:
    """Fit ``limit*(1-exp(-(x/k)**b))`` for one limit; ``(score, b, k)`` or ``None``.

    For the true limit, ``log(-log(1-y/limit))`` is exactly linear in ``log x``, so the
    linear fit is both the estimator and the criterion for choosing the limit.
    """
    residual = 1.0 - y / limit
    keep = residual > 0.0
    if int(np.count_nonzero(keep)) < 3:
        return None
    xk = x[keep]
    w = -np.log(residual[keep])
    keep2 = w > 0.0
    if int(np.count_nonzero(keep2)) < 3:
        return None
    line = weighted_line(np.log(xk[keep2]), np.log(w[keep2]), np.ones(int(np.count_nonzero(keep2))))
    if line is None or line[0] <= 0.0:
        return None
    shape, intercept = line
    scale_parameter = math.exp(min(max(-intercept / shape, -700.0), 700.0))
    if not (math.isfinite(scale_parameter) and scale_parameter > 0.0):
        return None
    curve = limit * (1.0 - np.exp(-np.power(x / scale_parameter, shape)))
    return unexplained(x, y, curve), shape, scale_parameter


def _slump_dosage_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Grid the asymptote, then refine it twice; the transform is linear at the true limit."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    candidates = limits_above(yp, steps=_COARSE_STEPS)
    for _stage in range(3):
        for limit in candidates:
            scored = _stretched_saturation_score(xp, yp, float(limit))
            if scored is None:
                continue
            score, shape, scale_parameter = scored
            if best is None or score < best[0]:
                best = (score, float(limit), shape, scale_parameter)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.12 if _stage == 0 else 0.03, steps=_FINE_STEPS)
    if best is not None:
        return {"S_u": best[1], "k": best[3], "b": best[2]}
    return {
        "S_u": top if top > 0.0 else scale(y),
        "k": float(np.median(xp)) if xp.size else 1.0,
        "b": 1.0,
    }


def _through_origin_exponential(
    x: FloatArray, y: FloatArray, rate: float
) -> tuple[float, float] | None:
    """Best amplitude and score of ``a*(exp(rate*x)-1)`` for one rate."""
    with np.errstate(all="ignore"):
        powered = np.exp(rate * x) - 1.0
        denominator = float(np.sum(powered * powered))
        numerator = float(np.sum(y * powered))
    if not (math.isfinite(denominator) and denominator > 0.0 and math.isfinite(numerator)):
        return None
    amplitude = numerator / denominator
    if not (math.isfinite(amplitude) and amplitude > 0.0):
        return None
    return unexplained(x, y, amplitude * powered), amplitude


def _upv_exponential_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the rate (the amplitude is linear for a fixed rate) and refine it twice."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    base_rate = 1.0 / max(span(xp), 1e-12)
    line = log_fit(xp, yp, 1.0) if xp.size >= 2 else None
    if line is not None and line[0] > 0.0:
        base_rate = line[0]
    best: tuple[float, float, float] | None = None
    candidates = np.geomspace(0.2 * base_rate, 5.0 * base_rate, _COARSE_STEPS)
    for stage in range(3):
        for rate in candidates:
            scored = _through_origin_exponential(xp, yp, float(rate))
            if scored is None:
                continue
            score, amplitude = scored
            if best is None or score < best[0]:
                best = (score, float(rate), amplitude)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.10 if stage == 0 else 0.02, steps=_FINE_STEPS)
    if best is not None:
        return {"a": best[2], "b": best[1]}
    top = float(np.max(y)) if y.size else 0.0
    return {"a": top if top > 0.0 else scale(y), "b": base_rate}


def _shifted_power_score(
    x: FloatArray, y: FloatArray, offset: float
) -> tuple[float, float, float] | None:
    """Log-log fit of ``a*(x-offset)**b`` for one offset; ``(score, a, b)`` or ``None``."""
    shifted = x - offset
    keep = shifted > 0.0
    if int(np.count_nonzero(keep)) < 2:
        return None
    line = log_fit(np.log(shifted[keep]), y[keep], 1.0)
    if line is None or line[0] <= 0.0:
        return None
    exponent, intercept = line
    amplitude = math.exp(min(intercept, 700.0))
    if not (math.isfinite(amplitude) and amplitude > 0.0):
        return None
    curve = amplitude * np.power(shifted, exponent)
    return unexplained(x, y, curve), amplitude, exponent


def _upv_power_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the velocity offset below the smallest x, then refine it twice."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    if xp.size >= 3:
        floor = 0.9 * float(np.min(xp))
        candidates = np.linspace(0.0, floor, _COARSE_STEPS)
        for stage in range(3):
            for offset in candidates:
                scored = _shifted_power_score(xp, yp, float(offset))
                if scored is None:
                    continue
                score, amplitude, exponent = scored
                if best is None or score < best[0]:
                    best = (score, float(offset), amplitude, exponent)
            if best is None:
                break
            candidates = np.linspace(
                max(0.0, best[1] - (0.10 if stage == 0 else 0.03) * floor),
                best[1] + (0.10 if stage == 0 else 0.03) * floor,
                _FINE_STEPS,
            )
    if best is not None:
        return {"a": best[2], "b": best[3], "V_0": best[1]}
    return {
        "a": top if top > 0.0 else scale(y),
        "b": 1.0,
        "V_0": 0.0,
    }


def _elastic_modulus_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-log line ``log y = log a + b*log x`` weighted by ``y**2``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    line = log_fit(np.log(xp), yp, 1.0) if xp.size >= 2 else None
    if line is not None and line[0] > 0.0:
        exponent, intercept = line
        amplitude = math.exp(min(intercept, 700.0))
        if math.isfinite(amplitude) and amplitude > 0.0:
            return {"a": amplitude, "b": exponent}
    top = float(np.max(y)) if y.size else 0.0
    middle = float(np.median(xp)) if xp.size else 1.0
    amplitude = (top if top > 0.0 else scale(y)) / max(math.sqrt(abs(middle)), 1e-12)
    return {"a": amplitude, "b": 0.5}


def _shrinkage_score(
    x: FloatArray, y: FloatArray, exponent: float
) -> tuple[float, float, float] | None:
    """Fit ``e_u*(x/(k+x))**m`` for one m through the linear form ``1/y**(1/m)``."""
    powered = np.power(y, 1.0 / exponent)
    keep = np.isfinite(powered) & (powered > 0.0) & (x > 0.0)
    if int(np.count_nonzero(keep)) < 2:
        return None
    line = weighted_line(1.0 / x[keep], 1.0 / powered[keep], np.ones(int(np.count_nonzero(keep))))
    if line is None:
        return None
    slope, intercept = line
    if not (math.isfinite(intercept) and intercept > 0.0 and math.isfinite(slope) and slope > 0.0):
        return None
    effective = 1.0 / intercept
    time_scale = slope * effective
    ultimate = effective**exponent
    if not (math.isfinite(time_scale) and time_scale > 0.0 and math.isfinite(ultimate)):
        return None
    curve = ultimate * np.power(x / (time_scale + x), exponent)
    return unexplained(x, y, curve), ultimate, time_scale


def _shrinkage_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the curvature exponent and refine it twice."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    candidates = np.geomspace(0.2, 3.0, _COARSE_STEPS)
    for stage in range(3):
        for exponent in candidates:
            scored = _shrinkage_score(xp, yp, float(exponent))
            if scored is None:
                continue
            score, ultimate, time_scale = scored
            if best is None or score < best[0]:
                best = (score, float(exponent), ultimate, time_scale)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.10 if stage == 0 else 0.02, steps=_FINE_STEPS)
    if best is not None:
        return {"e_u": best[2], "k": best[3], "m": best[1]}
    return {
        "e_u": top if top > 0.0 else scale(y),
        "k": float(np.median(xp)) if xp.size else 1.0,
        "m": 1.0,
    }


def _creep_score(
    x: FloatArray, y: FloatArray, ultimate: float
) -> tuple[float, float, float] | None:
    """Fit ``phi_u*x**psi/(d+x**psi)`` for one asymptote through ``log(y/(phi_u-y))``."""
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
    time_scale = math.exp(min(max(-intercept, -700.0), 700.0))
    if not (math.isfinite(time_scale) and time_scale > 0.0):
        return None
    powered = np.power(x, exponent)
    curve = ultimate * powered / (time_scale + powered)
    return unexplained(x, y, curve), exponent, time_scale


def _creep_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the creep asymptote above the data and refine it twice."""
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    candidates = limits_above(y, steps=_COARSE_STEPS)
    for stage in range(3):
        for ultimate in candidates:
            scored = _creep_score(x, y, float(ultimate))
            if scored is None:
                continue
            score, exponent, time_scale = scored
            if best is None or score < best[0]:
                best = (score, float(ultimate), exponent, time_scale)
        if best is None:
            break
        candidates = refine_around(best[1], spread=0.12 if stage == 0 else 0.03, steps=_FINE_STEPS)
    if best is not None:
        return {"phi_u": best[1], "psi": best[2], "d": best[3]}
    return {
        "phi_u": (top if top > 0.0 else scale(y)) * 1.2,
        "psi": 0.5,
        "d": max(span(x), 1e-12),
    }


def _maturity_strength_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Scan the plateau; at the true one ``log(f_u-y)`` is linear in x.

    ``y = f_u*(1-exp(-(x-M_0)/k))`` gives ``f_u - y = f_u*exp(M_0/k)*exp(-x/k)``, so the
    logarithm is a line in x with slope ``-1/k``; the offset M_0 follows from its intercept.
    Using every point this way is far steadier under noise than a three-point formula.
    """
    top = float(np.max(y)) if y.size else 0.0
    best: tuple[float, float, float, float] | None = None
    if x.size >= 3 and top > 0.0:
        candidates = limits_above(y, factor=8.0, steps=41)
        for stage in range(3):
            for plateau in candidates:
                residual = plateau - y
                keep = residual > 0.0
                if int(np.count_nonzero(keep)) < 3:
                    continue
                line = weighted_line(x[keep], np.log(residual[keep]), residual[keep] ** 2)
                if line is None or line[0] >= 0.0:
                    continue
                time_constant = -1.0 / line[0]
                amplitude = math.exp(min(line[1], 700.0))
                if not (math.isfinite(time_constant) and time_constant > 0.0):
                    continue
                if not (math.isfinite(amplitude) and amplitude > 0.0):
                    continue
                offset = time_constant * math.log(amplitude / plateau)
                curve = plateau * (1.0 - np.exp(-(x - offset) / time_constant))
                score = unexplained(x, y, curve)
                if best is None or score < best[0]:
                    best = (score, float(plateau), time_constant, offset)
            if best is None:
                break
            candidates = refine_around(best[1], spread=0.15 if stage == 0 else 0.03, steps=21)
    if best is not None:
        return {"f_u": best[1], "k": best[2], "M_0": max(best[3], 0.0)}
    return {
        "f_u": top if top > 0.0 else scale(y),
        "k": max(span(x), 1e-12),
        "M_0": 0.0,
    }


def _arrhenius_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-linear line ``log y = log f_ref + b/293.15 - b/x`` weighted by ``y**2``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    line = log_fit(1.0 / xp, yp, 1.0) if xp.size >= 2 else None
    if line is not None and line[0] < 0.0:
        activation = -line[0]
        reference = math.exp(min(line[1] - activation / 293.15, 700.0))
        if math.isfinite(reference) and reference > 0.0 and math.isfinite(activation):
            return {"f_ref": reference, "b": activation}
    top = float(np.max(y)) if y.size else 0.0
    return {"f_ref": top if top > 0.0 else scale(y), "b": 1000.0}


# -- the models -------------------------------------------------------------------------

_P_SLUMP = (
    Parameter("S_u", "Slump approached as the dosage grows without limit", "mm", lower=0.0),
    Parameter(
        "k", "Dosage at the characteristic point of the response", "% by mass of binder", lower=0.0
    ),
    Parameter("b", "Shape exponent of the response", "dimensionless", lower=0.0),
)
_P_UPV_EXPONENTIAL = (
    Parameter("a", "Pre-exponential amplitude of the relation", "MPa", lower=0.0),
    Parameter("b", "Rate of strength growth with pulse velocity", "per km/s", lower=0.0),
)
_P_UPV_POWER = (
    Parameter("a", "Scale of the shifted power law", "MPa/(km/s)^b", lower=0.0),
    Parameter("b", "Exponent of the shifted power law", "dimensionless", lower=0.0),
    Parameter(
        "V_0",
        "Pulse velocity at which the strength extrapolates to zero",
        "km/s",
        lower=0.0,
        upper=10.0,
    ),
)
_P_ELASTIC_MODULUS = (
    Parameter("a", "Modulus at unit compressive strength", "MPa^(1-b)", lower=0.0),
    Parameter("b", "Exponent of the modulus-strength power law", "dimensionless", lower=0.0),
)
_P_SHRINKAGE = (
    Parameter("e_u", "Ultimate drying shrinkage", "microstrain", lower=0.0),
    Parameter("k", "Time scale of the shrinkage development", "days", lower=0.0),
    Parameter("m", "Curvature exponent of the hyperbolic-power form", "dimensionless", lower=0.0),
)
_P_CREEP = (
    Parameter("phi_u", "Ultimate creep coefficient", "dimensionless", lower=0.0),
    Parameter("psi", "Time exponent of the creep development", "dimensionless", lower=0.0),
    Parameter("d", "Time scale of the creep development", "days^psi", lower=0.0),
)
_P_MATURITY = (
    Parameter("f_u", "Strength approached as maturity grows without limit", "MPa", lower=0.0),
    Parameter(
        "k",
        "Maturity constant: the maturity increment that closes 1-1/e of the remaining gap",
        "°C·h",
        lower=0.0,
    ),
    Parameter(
        "M_0", "Maturity below which the curve extrapolates to zero strength", "°C·h", lower=0.0
    ),
)
_P_ARRHENIUS = (
    Parameter("f_ref", "Strength at the 20 °C reference temperature", "MPa", lower=0.0),
    Parameter("b", "Apparent activation temperature E_a/R", "K", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    engineering_model(
        id="engineering_properties_slump_dosage",
        name="Slump against admixture dosage",
        category="engineering.properties",
        description=(
            "Slump rising with a stretched-exponential saturation as the admixture dosage "
            "increases."
        ),
        expression="S_u*(1-exp(-(x/k)**b))",
        parameters=_P_SLUMP,
        initial_guess=bespoke_guess(_P_SLUMP, _slump_dosage_guess),
        domain_check=domain_check(x_positive_for="(x/k)**b"),
        references=(_RAMACHANDRAN, _NEVILLE),
        assumptions=(
            "One admixture, one binder and one mixing procedure; x is the dosage as a "
            "percentage of the binder mass and S_u the slump the response approaches; the "
            "stretched exponential allows the S-shaped or saturating response real admixtures "
            "show, with b fitted rather than assumed."
        ),
        example=ModelExample({"S_u": 200.0, "k": 0.8, "b": 1.5}, x_range=(0.2, 2.0)),
    ),
    engineering_model(
        id="engineering_properties_upv_strength_exponential",
        name="UPV-strength relation (exponential)",
        category="engineering.properties",
        description=(
            "Compressive strength as an exponential of ultrasonic pulse velocity, through zero."
        ),
        expression="a*(exp(b*x)-1)",
        parameters=_P_UPV_EXPONENTIAL,
        initial_guess=bespoke_guess(_P_UPV_EXPONENTIAL, _upv_exponential_guess),
        domain_check=domain_check(x_positive_for="x"),
        references=(_MALHOTRA_CARINO, _NEVILLE),
        assumptions=(
            "An empirical calibration for one concrete mix, aggregate and moisture state; x is "
            "the ultrasonic pulse velocity in km/s and strength is forced through zero at zero "
            "velocity. The relation is not transferable between mixes without recalibration."
        ),
        example=ModelExample({"a": 0.5, "b": 0.9}, x_range=(3.5, 4.5)),
    ),
    engineering_model(
        id="engineering_properties_upv_strength_power",
        name="UPV-strength relation (shifted power)",
        category="engineering.properties",
        description=(
            "Compressive strength as a power of pulse velocity above a percolation threshold."
        ),
        expression="a*(x-V_0)**b",
        parameters=_P_UPV_POWER,
        initial_guess=bespoke_guess(_P_UPV_POWER, _upv_power_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_MALHOTRA_CARINO, _NEVILLE),
        assumptions=(
            "A calibration for one concrete; x is the ultrasonic pulse velocity in km/s and V_0 "
            "the velocity below which the extrapolated strength is zero. V_0 is bounded above "
            "at 10 km/s, well beyond any concrete, so the fractional power stays real."
        ),
        example=ModelExample({"a": 5.0, "b": 2.3, "V_0": 2.0}, x_range=(3.2, 4.6)),
    ),
    engineering_model(
        id="engineering_properties_elastic_modulus",
        name="Elastic modulus against compressive strength",
        category="engineering.properties",
        description="Secant modulus of elasticity as a power law in compressive strength.",
        expression="a*x**b",
        parameters=_P_ELASTIC_MODULUS,
        initial_guess=bespoke_guess(_P_ELASTIC_MODULUS, _elastic_modulus_guess),
        domain_check=domain_check(x_positive_for="x**b"),
        references=(_ACI_318, _EN_1992, _NEVILLE),
        assumptions=(
            "Normal-weight concrete with ordinary aggregates; x is the compressive strength in "
            "MPa and y the secant modulus in MPa. ACI 318 uses a fixed exponent of 0.5 "
            "(4700*sqrt(f'c)); Eurocode 2 uses 0.3 on the mean strength. The fitted exponent "
            "lets the data choose between them."
        ),
        example=ModelExample({"a": 4700.0, "b": 0.5}, x_range=(15.0, 60.0)),
    ),
    engineering_model(
        id="engineering_properties_shrinkage_hyperbolic",
        name="Drying-shrinkage development",
        category="engineering.properties",
        description=(
            "Drying shrinkage rising towards its ultimate value as a hyperbolic power of time."
        ),
        expression="e_u*(x/(k+x))**m",
        parameters=_P_SHRINKAGE,
        initial_guess=bespoke_guess(_P_SHRINKAGE, _shrinkage_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_ACI_209, _BAZANT_BAWEJA),
        assumptions=(
            "One member size, relative humidity and curing history; x is drying time in days "
            "from the end of curing, e_u the ultimate shrinkage in microstrain, k the time "
            "scale and m the curvature. The model does not separate autogenous from drying "
            "shrinkage or include a temperature history."
        ),
        example=ModelExample({"e_u": 600.0, "k": 20.0, "m": 1.15}, x_range=(1.0, 365.0)),
    ),
    engineering_model(
        id="engineering_properties_creep",
        name="Creep coefficient development",
        category="engineering.properties",
        description="Creep coefficient rising towards its ultimate value with loading duration.",
        expression="phi_u*x**psi/(d+x**psi)",
        parameters=_P_CREEP,
        initial_guess=bespoke_guess(_P_CREEP, _creep_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_ACI_209, _NEVILLE_DILGER_BROOKS, _BAZANT_BAWEJA),
        assumptions=(
            "Constant stress and one loading age, member size, relative humidity and mix; x is "
            "the loading duration in days, phi_u the ultimate creep coefficient, psi the time "
            "exponent and d the time scale in days^psi. Drying creep and basic creep are not "
            "separated."
        ),
        example=ModelExample({"phi_u": 2.5, "psi": 0.8, "d": 10.0}, x_range=(1.0, 365.0)),
    ),
    engineering_model(
        id="engineering_properties_maturity_strength",
        name="Strength against maturity",
        category="engineering.properties",
        description="Compressive strength rising exponentially with the maturity index.",
        expression="f_u*(1-exp(-(x-M_0)/k))",
        parameters=_P_MATURITY,
        initial_guess=bespoke_guess(_P_MATURITY, _maturity_strength_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_FREIESLEBEN_HANSEN, _ASTM_C1074),
        assumptions=(
            "Strength-maturity curve for one concrete and cement; x is the maturity index in "
            "°C·h (Freiesleben Hansen-Pedersen equivalent age at the reference temperature), "
            "k the maturity constant in the same unit and M_0 the maturity at which the curve "
            "extrapolates to zero strength. The same maturity index is assumed to reproduce "
            "strength under any temperature history."
        ),
        example=ModelExample({"f_u": 60.0, "k": 2500.0, "M_0": 200.0}, x_range=(500.0, 20000.0)),
    ),
    engineering_model(
        id="engineering_properties_arrhenius_equivalent_age",
        name="Arrhenius temperature-strength response",
        category="engineering.properties",
        description="Strength at a fixed age as an Arrhenius function of the curing temperature.",
        expression="f_ref*exp(-b*(1/x-1/293.15))",
        parameters=_P_ARRHENIUS,
        initial_guess=bespoke_guess(_P_ARRHENIUS, _arrhenius_guess),
        domain_check=domain_check(x_positive_for="1/x"),
        references=(_ARRHENIUS, _FREIESLEBEN_HANSEN),
        assumptions=(
            "Isothermal curing at one temperature for a fixed age; x is the absolute curing "
            "temperature in kelvin, f_ref the strength at 20 °C (293.15 K) and b = E_a/R the "
            "apparent activation temperature. Outside the normal curing range (about 5-45 °C) "
            "the single-activation-energy form breaks down."
        ),
        example=ModelExample({"f_ref": 40.0, "b": 2500.0}, x_range=(278.15, 318.15)),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # a*x**b is the core power law with the same parameters.
    ("engineering_properties_elastic_modulus", "power"),
    # f_u*(1-exp(-k*(x-M_0))) is a+b*exp(c*x) with a=f_u, b=-f_u*exp(k*M_0), c=-k.
    ("engineering_properties_maturity_strength", "exp_offset"),
)
