"""Matplotlib figures of a fit: publication output shared by the app and the reports.

Every figure is built from a :class:`~curvestudio.workflow.FitBundle` with
:class:`matplotlib.figure.Figure` objects only — no ``pyplot``, no global figure
manager, no GUI backend — so the module is safe to import in a headless worker.
A fit that did not converge is drawn honestly: the data, plus a prominent banner;
never a curve that could pass for a result. Undefined diagnostics render as an
explicit ``n/a`` panel instead of empty axes or NaN artefacts.
"""

import re
import threading
from pathlib import Path
from typing import Any, BinaryIO, Literal

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.transforms import Affine2D

from curvestudio.diagnostics import ResidualSet
from curvestudio.fitting import FitStatus
from curvestudio.statistics import Band
from curvestudio.workflow import FitBundle
from curvestudio_plotting.palette import (
    band_fill,
    curve_style,
    data_color,
)

FigureFormat = Literal["png", "svg", "pdf"]

FIT_FIGURE_SIZE: tuple[float, float] = (6.4, 4.2)
FIT_FIGURE_DPI = 150.0
RESIDUAL_FIGURE_SIZE: tuple[float, float] = (6.4, 6.2)

# The publication figure is always light (white page): rank 1's curve, neutral data and
# neutral bands, so an exported report matches the app without carrying a theme switch.
_DATA_COLOR = data_color(dark=False)
_CURVE_COLOR, _CURVE_WIDTH = curve_style(1)
_CONFIDENCE_COLOR = band_fill("confidence", dark=False)
_PREDICTION_COLOR = band_fill("prediction", dark=False)
_BANNER_COLOR = "#b2182b"
_NA_COLOR = "0.4"

#: The faint brand mark on every published figure. ``curvestudio_reporting`` carries the
#: same literal separately: the import contracts forbid plotting importing reporting, so
#: the two packages duplicate the text by design rather than sharing it.
WATERMARK_TEXT = "ConcreteInfo CurveStudio"
WATERMARK_ALPHA = 0.22
_WATERMARK_COLOR = "#464646"
_WATERMARK_FONT_SIZE = 6

_BYLINE_COLOR = "#666666"
#: The words :func:`add_byline` puts in front of the identity. ``curvestudio_reporting`` carries
#: the same literal separately (``PREPARED_BY``): the import contracts forbid plotting
#: importing reporting, so the two packages duplicate the text by design rather than sharing it.
_BYLINE_PREFIX = "Prepared by "
_BYLINE_FONT_SIZE = 7
_BYLINE_TOP_IN = 0.03
"""The byline's top edge below the figure's top edge, in inches (about 2 pt)."""
_BYLINE_BAND_IN = 0.27
"""The band :func:`add_byline` takes from the layout: the top offset, one 7 pt line and
about a line of clear space under it. In inches, so both figure heights get the same spacing."""

PANEL_TITLES = ("Residuals vs fitted", "Residuals vs x", "Normal Q-Q", "Histogram")

_RC_LOCK = threading.Lock()
"""Serialises the global rcParams change in :func:`save_figure` (thread-safe saves)."""

_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
"""Same set as the reports' ``sanitise`` (XML-illegal control characters), defined here
because the import contracts forbid plotting from importing the reporting package."""


def _clean(text: str) -> str:
    """Strip control characters a font has no glyph for (they render as boxes)."""
    return _XML_ILLEGAL.sub("", text)


def _watermark(fig: Figure) -> None:
    """Draw the faint brand mark in the figure's bottom-right corner.

    It lives on the :class:`~matplotlib.figure.Figure` (not an axes), so every format
    :func:`save_figure` writes and every report image built from the figure inherit it.
    """
    fig.text(
        0.995,
        0.006,
        WATERMARK_TEXT,
        ha="right",
        va="bottom",
        fontsize=_WATERMARK_FONT_SIZE,
        color=_WATERMARK_COLOR,
        alpha=WATERMARK_ALPHA,
    )


def add_byline(fig: Figure, byline: str) -> None:
    """Print "Prepared by <byline>" as one small muted line along the figure's top edge.

    ``byline`` is the identity text alone (a name, an organisation, or both), as the
    reports' ``byline`` argument is; the words in front of it are added here.

    The figure must use a constrained layout (both builders do), which leaves a plain
    :meth:`~matplotlib.figure.Figure.text` no room of its own: the byline reserves the
    top band from the layout engine and draws inside it, and the axes and their titles
    settle below. A plain constrained layout pins an automatic suptitle
    (``fit_figure(title=…)``) a fixed distance under the figure's top edge while
    reserving its room under the layout rect; shifting the title by the band keeps both
    true, with the title directly above the axes instead of inside the band. Call it
    once the figure is built and before :func:`save_figure`.

    A blank ``byline`` draws nothing and reserves nothing.
    """
    identity = " ".join(_clean(byline).split())  # one line, whatever line breaks were pasted in
    if not identity:
        return
    engine = fig.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        raise TypeError("add_byline needs a figure built with a constrained layout")
    height_in = fig.get_figheight()
    band = _BYLINE_BAND_IN / height_in
    fig.text(
        0.5,
        1 - _BYLINE_TOP_IN / height_in,
        f"{_BYLINE_PREFIX}{identity}",
        ha="center",
        va="top",
        fontsize=_BYLINE_FONT_SIZE,
        color=_BYLINE_COLOR,
        parse_math=False,
    )
    # two private reads, neither with a public getter in this matplotlib: the suptitle
    # and the engine's compressed flag
    suptitle = fig._suptitle  # type: ignore[attr-defined]
    compressed = engine._compress  # type: ignore[attr-defined]
    if suptitle is not None and not compressed:
        # the compressed variant places the title from the layout itself, below the band
        suptitle.set_transform(Affine2D().translate(0, -band) + suptitle.get_transform())
    engine.set(rect=(0, 0, 1, 1 - band))


def fit_figure(
    bundle: FitBundle,
    *,
    show_confidence: bool = True,
    show_prediction: bool = False,
    title: str | None = None,
    size: tuple[float, float] = FIT_FIGURE_SIZE,
    dpi: float = FIT_FIGURE_DPI,
) -> Figure:
    """The fit as a publication figure: data points, curve and bands.

    The fitted curve is drawn only when ``bundle.curve`` is defined (converged fit,
    finite over the plotted range); bands only when the bundle carries them. For a
    fit that did not converge the figure shows the data and an unambiguous
    ``FIT DID NOT CONVERGE: <reason>`` banner — and never a curve.
    """
    fig = Figure(figsize=size, dpi=dpi, layout="constrained")
    ax = fig.subplots()
    data = bundle.data
    label = "data (subsampled)" if data.subsampled else "data"
    ax.scatter(data.x, data.y, s=14, color=_DATA_COLOR, label=label, zorder=2)
    ax.set_xlabel(_clean(data.x_name), parse_math=False)
    ax.set_ylabel(_clean(data.y_name), parse_math=False)
    if title is not None:
        fig.suptitle(_clean(title), parse_math=False)

    if bundle.result.status is not FitStatus.CONVERGED:
        banner = _clean(f"FIT DID NOT CONVERGE: {bundle.result.termination_reason}")
        ax.set_title("FIT DID NOT CONVERGE", color=_BANNER_COLOR, fontweight="bold", fontsize=10)
        ax.text(
            0.5,
            0.5,
            banner,
            parse_math=False,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            wrap=True,
            color="white",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": _BANNER_COLOR, "alpha": 0.9},
            zorder=5,
        )
        ax.legend()
        _watermark(fig)
        return fig

    ax.set_title(_clean(bundle.equation_text), fontsize=9, parse_math=False)
    if bundle.curve is not None:
        ax.plot(
            bundle.curve.x,
            bundle.curve.y,
            color=_CURVE_COLOR,
            linewidth=_CURVE_WIDTH,
            label="fitted curve",
            zorder=3,
        )
    if show_confidence and bundle.confidence_band is not None:
        _draw_band(ax, bundle.confidence_band, _CONFIDENCE_COLOR)
    if show_prediction and bundle.prediction_band is not None:
        _draw_band(ax, bundle.prediction_band, _PREDICTION_COLOR)
    ax.legend()
    _watermark(fig)
    return fig


def _draw_band(ax: Axes, band: Band, color: str) -> None:
    ax.fill_between(
        band.x,
        band.lower,
        band.upper,
        color=color,
        linewidth=0,
        label=f"{band.level:.0%} {band.kind} band",
        zorder=1,
    )


def residual_figure(bundle: FitBundle) -> Figure:
    """The residual diagnostics of a converged fit as a 2x2 panel set.

    Panels: residuals vs fitted, residuals vs x, normal Q-Q, histogram of the
    weighted residuals — all from the bundle's diagnostics data, recomputed
    nowhere. A panel whose data is undefined shows ``n/a`` (and the reason in a
    small caption) instead of empty axes.
    """
    fig = Figure(figsize=RESIDUAL_FIGURE_SIZE, dpi=FIT_FIGURE_DPI, layout="constrained")
    grid = fig.subplots(2, 2)
    axes = list(grid.flat)
    residuals = None if bundle.summary is None else bundle.summary.residuals
    reason = bundle.summary_reason or "the residual diagnostics are not defined"
    x_name = bundle.data.x_name
    drawers = (
        _panel_residuals_vs_fitted,
        _panel_residuals_vs_x,
        _panel_qq,
        _panel_histogram,
    )
    for ax, title, draw in zip(axes, PANEL_TITLES, drawers, strict=True):
        ax.set_title(title, fontsize=10)
        if residuals is None:
            _na_panel(ax, reason)
        else:
            draw(ax, residuals, x_name)
    _watermark(fig)
    return fig


def _panel_residuals_vs_fitted(ax: Axes, residuals: ResidualSet, x_name: str) -> None:
    del x_name  # this panel's horizontal axis is the fitted value, not x
    if residuals.fitted:
        ax.scatter(residuals.fitted, residuals.residuals, s=12, color=_DATA_COLOR)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
    else:
        _na_panel(ax, "no fitted values")
    ax.set_xlabel("fitted")
    ax.set_ylabel("residual")


def _panel_residuals_vs_x(ax: Axes, residuals: ResidualSet, x_name: str) -> None:
    if residuals.x:
        ax.scatter(residuals.x, residuals.residuals, s=12, color=_DATA_COLOR)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
    else:
        _na_panel(ax, "no observations")
    ax.set_xlabel(_clean(x_name), parse_math=False)
    ax.set_ylabel("residual")


def _panel_qq(ax: Axes, residuals: ResidualSet, x_name: str) -> None:
    del x_name  # the Q-Q panel's axes are quantiles
    if residuals.qq_theoretical and residuals.qq_ordered:
        ax.scatter(residuals.qq_theoretical, residuals.qq_ordered, s=12, color=_DATA_COLOR)
        low = min(min(residuals.qq_theoretical), min(residuals.qq_ordered))
        high = max(max(residuals.qq_theoretical), max(residuals.qq_ordered))
        ax.plot([low, high], [low, high], "--", color="0.6", linewidth=0.8)
    else:
        _na_panel(ax, "the normal quantiles are not defined")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("ordered weighted residuals")


def _panel_histogram(ax: Axes, residuals: ResidualSet, x_name: str) -> None:
    del x_name  # the histogram's horizontal axis is the weighted residuals
    edges, counts = residuals.hist_edges, residuals.hist_counts
    if len(edges) >= 2:
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        ax.bar(centers, list(counts), width=widths, color=_DATA_COLOR, alpha=0.7, linewidth=0)
    else:
        _na_panel(ax, "the residual histogram is not defined")
    ax.set_xlabel("weighted residuals")
    ax.set_ylabel("count")


def _na_panel(ax: Axes, reason: str) -> None:
    ax.text(
        0.5,
        0.55,
        "n/a",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color=_NA_COLOR,
    )
    ax.text(
        0.5,
        0.3,
        reason,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color=_NA_COLOR,
        wrap=True,
    )


def save_figure(fig: Figure, target: Path | BinaryIO, format: FigureFormat) -> None:
    """Write ``fig`` to ``target`` as byte-reproducible PNG, SVG or PDF.

    The output carries no generation timestamp and SVG text is rendered as glyph
    paths, so the same figure always saves to identical bytes — a report written
    twice differs nowhere. Timestamped metadata is removed per format
    (``Date``/``CreationDate`` set to ``None``), the SVG hash salt is fixed
    (matplotlib would otherwise salt element ids with a fresh UUID on every save)
    and the producer is a constant.
    """
    if format not in ("png", "svg", "pdf"):
        raise ValueError(f"unsupported figure format {format!r}; use 'png', 'svg' or 'pdf'")
    metadata: dict[str, Any]
    if format == "png":
        metadata = {"Software": "CurveStudio"}
    elif format == "svg":
        metadata = {"Creator": "CurveStudio", "Date": None}
    else:
        metadata = {"Creator": "CurveStudio", "CreationDate": None, "ModDate": None}
    with _RC_LOCK, matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "CurveStudio"}):
        fig.savefig(target, format=format, metadata=metadata)
