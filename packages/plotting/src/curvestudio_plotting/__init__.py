"""CurveStudio publication plotting: Matplotlib figures built from a :class:`FitBundle`.

Only :class:`matplotlib.figure.Figure` objects are created (never ``pyplot`` global
state), so figures render identically headless, in the desktop app and inside
reports. A fit that did not converge is never drawn as a curve. ``Figure`` is
re-exported because it is the return type of both figure builders, so consumers
never need matplotlib imports of their own.
"""

from matplotlib.figure import Figure

from curvestudio_plotting.figures import (
    FIT_FIGURE_DPI,
    FIT_FIGURE_SIZE,
    RESIDUAL_FIGURE_SIZE,
    WATERMARK_ALPHA,
    WATERMARK_TEXT,
    FigureFormat,
    fit_figure,
    residual_figure,
    save_figure,
)
from curvestudio_plotting.palette import (
    BAND_CONFIDENCE_FILL,
    BAND_PREDICTION_FILL,
    DATA_COLOR,
    MODEL_COLORS,
    band_fill,
    curve_style,
    data_color,
    model_color,
)

__all__ = [
    "BAND_CONFIDENCE_FILL",
    "BAND_PREDICTION_FILL",
    "DATA_COLOR",
    "FIT_FIGURE_DPI",
    "FIT_FIGURE_SIZE",
    "MODEL_COLORS",
    "RESIDUAL_FIGURE_SIZE",
    "WATERMARK_ALPHA",
    "WATERMARK_TEXT",
    "Figure",
    "FigureFormat",
    "band_fill",
    "curve_style",
    "data_color",
    "fit_figure",
    "model_color",
    "residual_figure",
    "save_figure",
]
