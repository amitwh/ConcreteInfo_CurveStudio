"""Pointwise confidence and prediction bands by the delta method.

For a fitted curve ``f(x; theta)`` with parameter covariance ``C`` (already scaled by the
residual variance, as stored on the fit), the standard error of the mean response at ``x0``
is ``sqrt(g' C g)`` with ``g = d f / d theta`` at ``x0``. A *confidence* band covers the
mean curve; a *prediction* band covers a new observation, adding the observation variance
``s2 / new_weight``. Both use the Student-t quantile with the fit's residual degrees of
freedom. The bands are pointwise, not simultaneous.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import stats as sps

from curvestudio.statistics._checks import check_level, check_x
from curvestudio.statistics.errors import StatisticsError
from curvestudio.statistics.view import FitView

FloatArray = NDArray[np.float64]

_POINTWISE_NOTE = (
    "Pointwise {what} band from the delta method (linearised model, Student-t quantile with "
    "{df} degrees of freedom). The {level:.0%} coverage holds at each x separately, not "
    "simultaneously over the whole curve; accuracy depends on the model being close to linear "
    "in its parameters near the estimates."
)


class Band(BaseModel):
    """A pointwise band around the fitted curve at the requested ``x`` values.

    ``extrapolated[i]`` is true where ``x[i]`` lies outside the range of the active
    (weight > 0) fitted points, where the band relies on the model form alone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["confidence", "prediction", "monte_carlo"]
    level: float
    x: list[float]
    fit: list[float]
    lower: list[float]
    upper: list[float]
    extrapolated: list[bool]
    df: int
    pointwise: bool = True
    note: str


def _check_new_weight(new_weight: float | FloatArray, n_points: int) -> FloatArray:
    weights = np.asarray(new_weight, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full(n_points, float(weights))
    if weights.shape != (n_points,):
        raise StatisticsError(
            "INVALID_WEIGHT",
            f"new_weight must be a scalar or have one value per x ({n_points}), "
            f"got shape {weights.shape}",
        )
    if not (np.all(np.isfinite(weights)) and np.all(weights > 0.0)):
        raise StatisticsError("INVALID_WEIGHT", "new_weight must be finite and greater than zero")
    return weights


def _build_band(
    view: FitView,
    x: FloatArray,
    level: float,
    kind: Literal["confidence", "prediction"],
    added_variance: FloatArray | None,
) -> Band:
    cov = view.require_covariance()
    fitted = view.predict(x)
    jac = view.jacobian_at(x)
    if not (np.all(np.isfinite(fitted)) and np.all(np.isfinite(jac))):
        raise StatisticsError(
            "INVALID_X",
            "the model is not finite (outside its domain) at one or more of the requested x",
        )
    # g' C g per point; rounding can leave a tiny negative for a (near) zero covariance
    variance = np.maximum(np.einsum("ij,jk,ik->i", jac, cov, jac), 0.0)
    if added_variance is not None:
        variance = variance + added_variance
    half_width = float(sps.t.ppf(1.0 - (1.0 - level) / 2.0, view.df)) * np.sqrt(variance)
    what = "confidence (mean response)" if kind == "confidence" else "prediction (new observation)"
    return Band(
        kind=kind,
        level=level,
        x=x.tolist(),
        fit=fitted.tolist(),
        lower=(fitted - half_width).tolist(),
        upper=(fitted + half_width).tolist(),
        extrapolated=((x < view.x.min()) | (x > view.x.max())).tolist(),
        df=view.df,
        note=_POINTWISE_NOTE.format(what=what, df=view.df, level=level),
    )


def confidence_band(view: FitView, x: FloatArray, level: float = 0.95) -> Band:
    """Pointwise confidence band for the mean response at ``x``.

    Raises:
        StatisticsError: ``INVALID_LEVEL`` (``level`` not in (0, 1)); ``INVALID_X`` (empty,
            non-1-D, non-finite ``x``, or a model that is not finite there);
            ``NO_DEGREES_OF_FREEDOM`` / ``NO_COVARIANCE`` (the fit gives no uncertainty).
    """
    check_level(level)
    points = check_x(x)
    return _build_band(view, points, level, "confidence", None)


def prediction_band(
    view: FitView, x: FloatArray, level: float = 0.95, new_weight: float | FloatArray = 1.0
) -> Band:
    """Pointwise prediction band for a new observation at ``x``.

    ``new_weight`` is the weight of the new observation (scalar, or one per ``x``): its
    error variance is ``s2 / new_weight``, matching how the fit weighted its own data.

    Raises:
        StatisticsError: as :func:`confidence_band`, plus ``INVALID_WEIGHT`` (``new_weight``
            not finite and positive, or of the wrong shape).
    """
    check_level(level)
    points = check_x(x)
    weights = _check_new_weight(new_weight, points.size)
    view.require_covariance()  # raises NO_DEGREES_OF_FREEDOM / NO_COVARIANCE before s2 is used
    assert view.s2 is not None  # df > 0 was just ensured
    return _build_band(view, points, level, "prediction", view.s2 / weights)
