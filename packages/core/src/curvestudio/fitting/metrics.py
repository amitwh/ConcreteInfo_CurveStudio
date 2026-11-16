"""Goodness-of-fit statistics for a fitted curve.

Every statistic is computed from plain arrays so it can be checked by hand and reused by
any fitting back end. The function never lets a NumPy ``RuntimeWarning`` escape: degenerate
cases (noiseless data, constant ``y``, no degrees of freedom) are resolved explicitly and
reported as ``None`` where a number would be meaningless, never as ``nan``.
"""

import math
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

FloatArray = NDArray[np.float64]

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
"""A float that pydantic refuses to accept as NaN or +/-inf, so JSON stays standard."""


class FitStatistics(BaseModel):
    """Goodness-of-fit summary. ``None`` means "not defined", never "not computed".

    ``n`` is the number of observations passed in; the engines pass the active
    (weight > 0) points only, so zero-weight points are excluded from ``n`` and from
    ``df = n - k`` in every fit result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    k: int
    df: int
    sse: FiniteFloat
    mse: FiniteFloat
    rmse: FiniteFloat
    mae: FiniteFloat
    r2: FiniteFloat | None
    adj_r2: FiniteFloat | None
    residual_se: FiniteFloat | None
    r2_definition: Literal["centered", "uncentered"]


def _as_vector(name: str, values: NDArray[np.float64] | list[float]) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def fit_statistics(
    y: FloatArray,
    fitted: FloatArray,
    n_params: int,
    weights: FloatArray | None = None,
    through_origin: bool = False,
) -> FitStatistics:
    """Compute fit statistics for observations ``y`` against model values ``fitted``.

    Weights multiply squared residuals, so ``sse = sum(w * r**2)`` and the total sum of
    squares is taken about the *weighted* mean. ``mae`` follows the same convention as
    ``mse``: ``sum(w * |r|) / n``. Zero weights are allowed in ``weights`` (they add
    nothing to any sum); the engines pass only the active (``w > 0``) points, so ``n``
    counts active observations in every fit result.

    ``r2`` is ``1 - sse/sst`` with ``sst`` centred on the (weighted) mean. With
    ``through_origin=True`` the model has no free intercept, the centred form can go
    negative or be misleading, so the uncentred ``sst = sum(w * y**2)`` is used and
    ``r2_definition`` says so. The adjusted R2 then uses ``n`` instead of ``n - 1`` in
    ``1 - (1 - r2) * n / df``, because no degree of freedom was spent on the mean.

    Degenerate cases: when ``sst == 0`` the ratio is undefined; a perfect fit
    (``sse == 0``) is reported as ``r2 = 1`` and an imperfect one as ``None``.
    ``residual_se`` and ``adj_r2`` are ``None`` when ``df <= 0``.

    Raises:
        ValueError: inputs are not 1-D, equal-length and finite, ``n < 1``,
            ``n_params < 0``, weights are negative / non-finite / all zero, or a sum
            overflows.
    """
    y_arr = _as_vector("y", y)
    fitted_arr = _as_vector("fitted", fitted)
    if y_arr.shape != fitted_arr.shape:
        raise ValueError(f"y and fitted differ in length: {y_arr.size} vs {fitted_arr.size}")
    n = int(y_arr.size)
    if n < 1:
        raise ValueError("at least one observation is required")
    if n_params < 0:
        raise ValueError(f"n_params must be >= 0, got {n_params}")

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _as_vector("weights", weights)
        if w.shape != y_arr.shape:
            raise ValueError(f"weights length {w.size} does not match y length {n}")
        if np.any(w < 0):
            raise ValueError("weights must be >= 0")
        if not np.any(w > 0):
            raise ValueError("weights must not all be zero")

    with np.errstate(over="ignore", invalid="ignore"):
        resid = y_arr - fitted_arr
        sse = float(np.sum(w * resid * resid))
        abs_sum = float(np.sum(w * np.abs(resid)))
        active = w > 0
        if through_origin:
            sst = float(np.sum(w * y_arr * y_arr))
            degenerate = not np.any(y_arr[active] != 0.0)
        else:
            y_active = y_arr[active]
            # Test for constant y directly: a rounded weighted mean would leave a tiny
            # positive sst and turn a perfect fit into a 0/0-like ratio.
            degenerate = bool(np.all(y_active == y_active[0]))
            sst = (
                0.0
                if degenerate
                else float(np.sum(w * (y_arr - np.average(y_arr, weights=w)) ** 2))
            )
    if not (math.isfinite(sse) and math.isfinite(abs_sum) and math.isfinite(sst)):
        raise ValueError("sums of squares are not finite (overflow); check the input scale")

    df = n - n_params
    r2: float | None = None
    if not degenerate and sst > 0.0:
        r2 = 1.0 - sse / sst
    elif sse == 0.0:
        r2 = 1.0

    adj_r2: float | None = None
    if r2 is not None and df > 0:
        dof_total = n if through_origin else n - 1
        adj_r2 = 1.0 - (1.0 - r2) * dof_total / df
    residual_se = math.sqrt(sse / df) if df > 0 else None

    mse = sse / n
    return FitStatistics(
        n=n,
        k=n_params,
        df=df,
        sse=sse,
        mse=mse,
        rmse=math.sqrt(mse),
        mae=abs_sum / n,
        r2=r2,
        adj_r2=adj_r2,
        residual_se=residual_se,
        r2_definition="uncentered" if through_origin else "centered",
    )
