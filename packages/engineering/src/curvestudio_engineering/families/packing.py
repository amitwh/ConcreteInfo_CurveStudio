"""Particle-packing models: cumulative percent passing against particle size.

Fuller's curve uses the classical exponent 1/2, the modified Fuller curve fits the exponent,
Andreasen's distribution uses the classical exponent 1/3, and the modified Andreasen
(Dinger-Funk) distribution adds a minimum particle size. x is the particle size in the same
unit as the size parameters (mm in the examples), y the cumulative percent passing.

The modified Fuller curve is a power law in disguise and is declared equivalent to the core
``power`` model in ``EQUIVALENT_PAIRS``; fixing an exponent to its classical value is what
keeps Fuller and Andreasen distinct curves rather than the same model twice.
"""

import math

import numpy as np

from curvestudio.models import ModelExample, ModelSpec, Parameter
from curvestudio_engineering.families._common import (
    FloatArray,
    bespoke_guess,
    domain_check,
    log_fit,
    scale,
)
from curvestudio_engineering.notice import engineering_model

_FULLER = (
    "Fuller W.B., Thompson S.E. (1907) The laws of proportioning concrete, Transactions of "
    "the American Society of Civil Engineers 59:67-143"
)
_ANDREASEN = (
    "Andreasen A.H.M., Andersen J. (1930) Über die Beziehung zwischen Kornabstufung und "
    "Zwischenraum in Produkten aus losen Körnern, Kolloid-Zeitschrift 50:217-228"
)
_FUNK_DINGER = (
    "Funk J.E., Dinger D.R. (1994) Predictive Process Control of Crowded Particulate "
    "Suspensions, Kluwer Academic Publishers"
)

_COARSE_STEPS = 25
_FINE_STEPS = 15


# -- initial guesses --------------------------------------------------------------------


def _fuller_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Read D from each point of ``y = 100*sqrt(x/D)`` and take the median."""
    keep = (x > 0.0) & (y > 0.0)
    xk, yk = x[keep], y[keep]
    if xk.size:
        with np.errstate(all="ignore"):
            diameter = xk * np.square(100.0 / yk)
        valid = diameter[np.isfinite(diameter) & (diameter > 0.0)]
        if valid.size:
            return {"D": float(np.median(valid))}
    top = float(np.max(x)) if x.size else 0.0
    return {"D": top if top > 0.0 else 1.0}


def _andreasen_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Read D from each point of ``y = 100*(x/D)**(1/3)`` and take the median."""
    keep = (x > 0.0) & (y > 0.0)
    xk, yk = x[keep], y[keep]
    if xk.size:
        with np.errstate(all="ignore"):
            diameter = xk * np.power(100.0 / yk, 3.0)
        valid = diameter[np.isfinite(diameter) & (diameter > 0.0)]
        if valid.size:
            return {"D": float(np.median(valid))}
    top = float(np.max(x)) if x.size else 0.0
    return {"D": top if top > 0.0 else 1.0}


def _modified_fuller_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Log-log line ``log(y/100) = q*log(x/D)`` weighted by ``(y/100)**2``."""
    positive = (x > 0.0) & (y > 0.0)
    xp, yp = x[positive], y[positive]
    line = log_fit(np.log(xp), yp / 100.0, 1.0) if xp.size >= 2 else None
    if line is not None and line[0] > 0.0:
        exponent, intercept = line
        diameter = math.exp(min(max(-intercept / exponent, -700.0), 700.0))
        if math.isfinite(diameter) and diameter > 0.0:
            return {"D": diameter, "q": exponent}
    top = float(np.max(x)) if x.size else 0.0
    return {"D": top if top > 0.0 else 1.0, "q": 0.45}


def _modified_andreasen_score(
    x: FloatArray, y: FloatArray, minimum: float, exponent: float
) -> tuple[float, float] | None:
    """Score one ``(d_min, n)`` pair; ``(spread, d_span)`` or ``None``.

    With ``d_min`` and ``n`` fixed, ``(x**n - d_min**n) / (y/100)`` must be the constant
    ``D**n - d_min**n``. The normalised variance of that quantity is the criterion, and the
    mean gives ``D`` (hence ``d_span = D - d_min``).
    """
    if not (math.isfinite(minimum) and minimum >= 0.0 and math.isfinite(exponent)):
        return None
    if exponent <= 0.0:
        return None
    with np.errstate(all="ignore"):
        powered = np.power(x, exponent)
        base = math.pow(minimum, exponent) if minimum > 0.0 else 0.0
        numerator = powered - base
    keep = (numerator > 0.0) & (y > 0.0) & (y < 100.0)
    if int(np.count_nonzero(keep)) < 3:
        return None
    fraction = y[keep] / 100.0
    values = numerator[keep] / fraction
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        return None
    mean_value = float(np.mean(values))
    if not (math.isfinite(mean_value) and mean_value > 0.0):
        return None
    spread = float(np.mean(np.square(values - mean_value))) / mean_value**2
    if not math.isfinite(spread):
        return None
    constant = mean_value + base
    if constant <= 0.0:
        return None
    top = constant ** (1.0 / exponent)
    if not (math.isfinite(top) and top > minimum):
        return None
    return spread, top - minimum


def _modified_andreasen_guess(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Two-dimensional scan over the minimum size and the exponent, refined twice."""
    keep = (x > 0.0) & (y > 0.0) & (y < 100.0)
    xk, yk = x[keep], y[keep]
    top = float(np.max(x)) if x.size else 0.0
    best: tuple[float, float, float, float] | None = None
    if xk.size >= 3:
        floor = float(np.min(xk))
        minima = np.linspace(0.0, 0.9 * floor, _COARSE_STEPS)
        exponents = np.geomspace(0.15, 1.5, _COARSE_STEPS)
        for stage in range(3):
            for minimum in minima:
                for exponent in exponents:
                    scored = _modified_andreasen_score(xk, yk, float(minimum), float(exponent))
                    if scored is None:
                        continue
                    spread, span_length = scored
                    if best is None or spread < best[0]:
                        best = (spread, float(minimum), float(exponent), span_length)
            if best is None:
                break
            width = (0.10 if stage == 0 else 0.02) * floor
            minima = np.linspace(
                max(0.0, best[1] - width), min(0.95 * floor, best[1] + width), _FINE_STEPS
            )
            relative = 0.10 if stage == 0 else 0.02
            exponents = np.geomspace(
                best[2] * (1.0 - relative), best[2] * (1.0 + relative), _FINE_STEPS
            )
    if best is not None:
        return {"d_min": best[1], "d_span": best[3], "n": best[2]}
    length = top if top > 0.0 else scale(x)
    return {"d_min": 0.0, "d_span": length, "n": 0.5}


# -- the models -------------------------------------------------------------------------

_P_FULLER = (Parameter("D", "Maximum particle size (100 % passing)", "mm", lower=0.0),)
_P_MODIFIED_FULLER = (
    Parameter("D", "Maximum particle size at which passing reaches 100 %", "mm", lower=0.0),
    Parameter("q", "Grading exponent (about 0.5 for Fuller)", "dimensionless", lower=0.0),
)
_P_ANDREASEN = (Parameter("D", "Maximum particle size (100 % passing)", "mm", lower=0.0),)
_P_MODIFIED_ANDREASEN = (
    Parameter("d_min", "Minimum particle size of the distribution", "mm", lower=0.0),
    Parameter(
        "d_span", "Difference between the maximum and minimum particle size", "mm", lower=0.0
    ),
    Parameter("n", "Distribution exponent", "dimensionless", lower=0.0),
)

MODELS: tuple[ModelSpec, ...] = (
    engineering_model(
        id="engineering_packing_fuller",
        name="Fuller curve",
        category="engineering.packing",
        description=(
            "Fuller's ideal grading: cumulative percent passing as the square root of the "
            "size ratio."
        ),
        expression="100*sqrt(x/D)",
        parameters=_P_FULLER,
        initial_guess=bespoke_guess(_P_FULLER, _fuller_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_FULLER,),
        assumptions=(
            "Dense, angular to rounded aggregate graded for minimum void content; x is the "
            "particle size in mm and D the maximum size, so the curve is used for x <= D. "
            "Fuller's parabola is a target grading, not a measured size distribution."
        ),
        example=ModelExample({"D": 20.0}, x_range=(0.075, 20.0)),
    ),
    engineering_model(
        id="engineering_packing_modified_fuller",
        name="Modified Fuller curve",
        category="engineering.packing",
        description="Fuller grading with a fitted exponent instead of the classical one half.",
        expression="100*(x/D)**q",
        parameters=_P_MODIFIED_FULLER,
        initial_guess=bespoke_guess(_P_MODIFIED_FULLER, _modified_fuller_guess),
        domain_check=domain_check(
            x_positive_for="(x/D)**q (its parameter derivative is not defined at x <= 0)"
        ),
        references=(_FULLER, _FUNK_DINGER),
        assumptions=(
            "A target grading with the exponent q left free (typically 0.4-0.5); x and D are "
            "particle sizes in mm. A smaller q fills the coarse end and a larger q the fine "
            "end; the curve says nothing about particle shape or compaction energy."
        ),
        example=ModelExample({"D": 20.0, "q": 0.45}, x_range=(0.075, 20.0)),
    ),
    engineering_model(
        id="engineering_packing_andreasen",
        name="Andreasen distribution",
        category="engineering.packing",
        description="Andreasen's distribution with the classical exponent one third.",
        expression="100*(x/D)**(1/3)",
        parameters=_P_ANDREASEN,
        initial_guess=bespoke_guess(_P_ANDREASEN, _andreasen_guess),
        domain_check=domain_check(x_non_negative=True),
        references=(_ANDREASEN, _FUNK_DINGER),
        assumptions=(
            "An ideal continuous grading from zero size to D with the classical exponents "
            "around 1/3; x is the particle size in mm and D the maximum size. The distribution "
            "is unbounded below zero, so the smallest measured size fixes the useful range."
        ),
        example=ModelExample({"D": 20.0}, x_range=(0.075, 20.0)),
    ),
    engineering_model(
        id="engineering_packing_modified_andreasen",
        name="Modified Andreasen (Dinger-Funk) distribution",
        category="engineering.packing",
        description="Andreasen distribution with a minimum particle size and a fitted exponent.",
        expression="100*(x**n-d_min**n)/((d_min+d_span)**n-d_min**n)",
        parameters=_P_MODIFIED_ANDREASEN,
        initial_guess=bespoke_guess(_P_MODIFIED_ANDREASEN, _modified_andreasen_guess),
        domain_check=domain_check(
            x_positive_for="x**n (its parameter derivative is not defined at x <= 0)"
        ),
        references=(_ANDREASEN, _FUNK_DINGER),
        assumptions=(
            "The Dinger-Funk modification of Andreasen's distribution: x is the particle size "
            "in mm, d_min the minimum size (0 % passing), d_min + d_span the maximum size "
            "(100 % passing) and n the distribution exponent. It assumes a continuous "
            "distribution and ignores packing kinetics and particle shape."
        ),
        example=ModelExample({"d_min": 0.075, "d_span": 19.925, "n": 0.45}, x_range=(0.15, 20.0)),
    ),
)

EQUIVALENT_PAIRS: tuple[tuple[str, str], ...] = (
    # 100*(x/D)**q is a*x**b with a = 100/D**q and b = q.
    ("engineering_packing_modified_fuller", "power"),
)
