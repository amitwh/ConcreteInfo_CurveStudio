"""CurveStudio publication plotting: Matplotlib figures built from a :class:`FitBundle`.

Only :class:`matplotlib.figure.Figure` objects are created (never ``pyplot`` global
state), so figures render identically headless, in the desktop app and inside
reports. A fit that did not converge is never drawn as a curve. ``Figure`` is
re-exported because it is the return type of both figure builders, so consumers
never need matplotlib imports of their own.

The figure half resolves lazily (PEP 562): importing this package imports only the
light palette, and ``matplotlib``/``figures`` arrive at the first attribute that
needs them — the desktop's plot widget takes its colours from here long before any
report is built, and must not pay for Matplotlib to do it.
"""

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from curvestudio_plotting.figures import (
        FIT_FIGURE_DPI,
        FIT_FIGURE_SIZE,
        RESIDUAL_FIGURE_SIZE,
        WATERMARK_ALPHA,
        WATERMARK_TEXT,
        FigureFormat,
        add_byline,
        fit_figure,
        residual_figure,
        save_figure,
    )

_FIGURE_EXPORTS = {
    "FIT_FIGURE_DPI": ".figures",
    "FIT_FIGURE_SIZE": ".figures",
    "RESIDUAL_FIGURE_SIZE": ".figures",
    "WATERMARK_ALPHA": ".figures",
    "WATERMARK_TEXT": ".figures",
    "Figure": ("matplotlib.figure", "Figure"),
    "FigureFormat": ".figures",
    "add_byline": ".figures",
    "fit_figure": ".figures",
    "residual_figure": ".figures",
    "save_figure": ".figures",
}

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
    "add_byline",
    "band_fill",
    "curve_style",
    "data_color",
    "fit_figure",
    "model_color",
    "residual_figure",
    "save_figure",
]


def __getattr__(name: str):
    where = _FIGURE_EXPORTS.get(name)
    if where is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    if isinstance(where, tuple):
        module_name, attribute = where
        value = getattr(import_module(module_name), attribute)
    else:
        value = getattr(import_module(__name__ + where, package=__name__), name)
    globals()[name] = value  # later lookups skip this hook
    return value
