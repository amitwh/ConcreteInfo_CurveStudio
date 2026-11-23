"""The residual-trend check: does a shape the fitted model cannot represent remain in the residuals?

This is the audit's one new computation. The weighted residuals are regressed on Legendre
terms the model cannot represent and the block is F-tested. A polynomial's residuals are
orthogonal to every Legendre degree up to its own, so only the degrees beyond it are tested;
any other model is tested on degrees 1 to ``TREND_MODEL_TERMS`` against a constant.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from curvestudio.diagnostics import FitReport
from curvestudio.tools._common import (
    Check,
    Inputs,
    audit_check,
    is_exact,
    no_summary,
    p_verdict,
    skipped,
    title_for,
)
from curvestudio.tools._thresholds import (
    POLYNOMIAL_ID,
    TREND_CAUTION_P,
    TREND_EXTRA_TERMS,
    TREND_FAIL_P,
    TREND_MIN_RESIDUAL_DF,
    TREND_MODEL_TERMS,
)
from curvestudio.tools._types import AuditCheck, Verdict
from curvestudio.tools._wording import (
    TREND_NEXT_STEP,
    degree_name,
    num,
    p_threshold,
    terms,
)
from curvestudio.workflow import PolynomialRef


@dataclass(frozen=True)
class Trend:
    """The residual-trend F test of Legendre terms the fitted model cannot represent."""

    statistic: float
    p_value: float
    df_error: int
    tested: tuple[int, ...]
    beyond: str
    """What the tested terms are measured beyond: the fitted polynomial, or the model."""


def fitted_polynomial(inp: Inputs) -> tuple[int, bool] | None:
    """Degree and origin flag of the fitted polynomial, or ``None`` for any other model.

    The result's model id says both (``poly2``, ``poly1_origin``). A polynomial in the spec,
    when one is given, decides the degree.
    """
    matched = POLYNOMIAL_ID.fullmatch(inp.result.model_id)
    if matched is None:
        return None
    degree = int(matched.group(1))
    if inp.spec is not None and isinstance(inp.spec.model, PolynomialRef):
        degree = inp.spec.model.degree
    return degree, matched.group(2) is not None


def trend_terms(inp: Inputs) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Degrees tested, baseline degrees, and how many terms the fitted model already occupies.

    A polynomial of degree ``d`` is tested on ``d + 1`` and ``d + 2`` with no baseline (and
    on the constant term too when the fit goes through the origin, since such a fit has
    none). Any other model is tested on degrees 1 to ``TREND_MODEL_TERMS`` against a constant.
    """
    polynomial = fitted_polynomial(inp)
    if polynomial is None:
        return tuple(range(1, TREND_MODEL_TERMS + 1)), (0,), 0
    degree, origin = polynomial
    tested = list(range(degree + 1, degree + 1 + TREND_EXTRA_TERMS))
    if origin:
        tested.insert(0, 0)
    return tuple(tested), (), degree if origin else degree + 1


def residual_trend(inp: Inputs, summary: FitReport) -> Trend | str:
    """The trend test on the summary's residuals, or the reason it cannot run."""
    x = np.asarray(summary.residuals.x, dtype=np.float64)
    weighted = np.asarray(summary.residuals.weighted_residuals, dtype=np.float64)
    if x.size == 0:
        return "there are no residuals"
    if not (bool(np.all(np.isfinite(x))) and bool(np.all(np.isfinite(weighted)))):
        return "the residuals are not all finite"
    if float(x.max()) == float(x.min()):
        return "all x values are equal, so the residuals have no order in x"
    tested, baseline, own = trend_terms(inp)
    df_error = int(x.size) - len(tested) - len(baseline) - own
    if df_error < TREND_MIN_RESIDUAL_DF:
        plural = "degree" if df_error == 1 else "degrees"
        return (
            f"only {df_error} residual {plural} of freedom; the test needs at least "
            f"{TREND_MIN_RESIDUAL_DF}"
        )
    scaled = 2.0 * (x - float(x.min())) / float(x.max() - float(x.min())) - 1.0
    degrees = sorted({*tested, *baseline})
    vander = np.polynomial.legendre.legvander(scaled, max(degrees))[:, degrees]

    def leftover(columns: Sequence[int]) -> float:
        """Sum of squares left of the residuals after fitting the Legendre ``columns``."""
        if not columns:
            return float(weighted @ weighted)
        design = vander[:, list(columns)]
        left = weighted - design @ np.linalg.lstsq(design, weighted, rcond=None)[0]
        return float(left @ left)

    sse_baseline = leftover(range(len(baseline)))
    sse_full = leftover(range(len(degrees)))
    if sse_full <= 0.0:
        return (
            "the tested terms account for the whole spread of the residuals, so the F "
            "statistic is undefined"
        )
    count = len(tested)
    statistic = max(((sse_baseline - sse_full) / count) / (sse_full / df_error), 0.0)
    polynomial = fitted_polynomial(inp)
    if polynomial is None:
        beyond = f"the fitted {inp.result.model_id} model"
    else:
        degree, origin = polynomial
        beyond = f"the fitted {degree_name(degree)}" + (" through the origin" if origin else "")
    return Trend(
        statistic=statistic,
        p_value=float(scipy_stats.f.sf(statistic, count, df_error)),
        df_error=df_error,
        tested=tested,
        beyond=beyond,
    )


def check_residual_trend(inp: Inputs) -> AuditCheck:
    if inp.summary is None:
        return skipped(TREND, no_summary(inp))
    if is_exact(inp):
        return skipped(TREND, "the fit is exact, so its residuals carry no trend")
    trend = residual_trend(inp, inp.summary)
    if not isinstance(trend, Trend):
        return skipped(TREND, trend)
    verdict = p_verdict(trend.p_value, TREND_CAUTION_P, TREND_FAIL_P)
    points = len(inp.summary.residuals.x)
    names = terms(trend.tested)
    return audit_check(
        TREND,
        verdict,
        title_for(TREND_TITLES, verdict),
        f"Weighted residuals regressed on {names} terms over {points} points, x scaled to "
        f"[-1, 1]: p = {num(trend.p_value)}.",
        statistic=(
            f"F({len(trend.tested)}, {trend.df_error}) = {num(trend.statistic)}, "
            f"p = {num(trend.p_value)}, for {names} terms beyond {trend.beyond}"
        ),
        threshold=p_threshold(TREND_CAUTION_P, TREND_FAIL_P),
        next_step=None if verdict is Verdict.PASS else TREND_NEXT_STEP,
    )


TREND_TITLES = (
    "Residuals show no trend in x",
    "Residuals may show a trend in x",
    "Residuals show a trend in x",
)

TREND = Check(
    "residual_trend",
    "A trend of the residuals in x was not tested",
    "A model of the right shape leaves residuals scattered evenly about zero at every x. "
    "Terms the model cannot represent, left in the residuals, are a shape the data contain "
    "and the model lacks.",
    check_residual_trend,
)
