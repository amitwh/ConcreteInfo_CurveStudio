"""Result containers of the workflow layer. All are JSON-safe: undefined means ``None``."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from curvestudio.diagnostics import FitReport
from curvestudio.fitting import FitResult
from curvestudio.fitting.metrics import FiniteFloat
from curvestudio.statistics import Band

CURVE_POINTS = 200
CURVE_MARGIN = 0.05
BAND_LEVEL = 0.95
COMPACT_MAX_POINTS = 5000
MAX_PREDICT_POINTS = 100_000


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CurveSamples(_Frozen):
    """The fitted curve sampled at ``CURVE_POINTS`` x values; every ``y`` is finite."""

    x: list[FiniteFloat]
    y: list[FiniteFloat]


class DataSnapshot(_Frozen):
    """The fit input a bundle was computed from, so a bundle stands on its own.

    ``n_total`` is the number of rows the fit used. With ``subsampled`` the arrays hold an
    evenly strided subset (first and last row included) and are for plotting only.
    """

    x: list[FiniteFloat]
    y: list[FiniteFloat]
    weights: list[FiniteFloat] | None
    x_name: str
    y_name: str
    n_total: int
    subsampled: bool = False


class FitBundle(_Frozen):
    """One fit with everything a UI or report needs.

    ``result.status`` says whether the fit converged. When it did not (or a quantity is not
    defined) the matching field is ``None`` and the ``*_reason`` field says why; a curve is
    never substituted for a failed fit. ``summary`` is the ``describe_fit`` report
    (information criteria, residual diagnostics, tests); ``curve`` covers the data range
    extended by 5 % each side; the bands are pointwise at the curve's x values.
    """

    result: FitResult
    summary: FitReport | None
    summary_reason: str | None
    curve: CurveSamples | None
    curve_reason: str | None
    confidence_band: Band | None
    prediction_band: Band | None
    bands_reason: str | None
    equation_text: str
    warnings: tuple[str, ...]
    data: DataSnapshot


class Prediction(_Frozen):
    """Predicted values at the requested ``x`` (same order and length).

    A row is ``None`` where it is undefined: a non-finite ``x``, or a model that has no
    real value there. ``lower``/``upper`` are ``None`` when no band was requested or the
    band is undefined altogether (``band_reason`` says why); individual rows may be ``None``.
    ``extrapolated[i]`` is true where ``x[i]`` lies outside the fitted data's range.
    """

    x: list[FiniteFloat | None]
    y: list[FiniteFloat | None]
    band: Literal["confidence", "prediction"] | None
    level: float
    lower: list[FiniteFloat | None] | None
    upper: list[FiniteFloat | None] | None
    extrapolated: list[bool]
    band_reason: str | None
