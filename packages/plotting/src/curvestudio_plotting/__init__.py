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
    FigureFormat,
    fit_figure,
    residual_figure,
    save_figure,
)

__all__ = [
    "FIT_FIGURE_DPI",
    "FIT_FIGURE_SIZE",
    "RESIDUAL_FIGURE_SIZE",
    "Figure",
    "FigureFormat",
    "fit_figure",
    "residual_figure",
    "save_figure",
]
