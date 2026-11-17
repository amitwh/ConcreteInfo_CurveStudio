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

from curvestudio.diagnostics import ResidualSet
from curvestudio.fitting import FitStatus
from curvestudio.statistics import Band
from curvestudio.workflow import FitBundle

FigureFormat = Literal["png", "svg", "pdf"]

FIT_FIGURE_SIZE: tuple[float, float] = (6.4, 4.2)
FIT_FIGURE_DPI = 150.0
RESIDUAL_FIGURE_SIZE: tuple[float, float] = (6.4, 6.2)

_DATA_COLOR = "#1f77b4"
_CURVE_COLOR = "#d62728"
_CONFIDENCE_COLOR = "#1f77b4"
_PREDICTION_COLOR = "#7f7f7f"
_BANNER_COLOR = "#b2182b"
_NA_COLOR = "0.4"

PANEL_TITLES = ("Residuals vs fitted", "Residuals vs x", "Normal Q-Q", "Histogram")

_RC_LOCK = threading.Lock()
"""Serialises the global rcParams change in :func:`save_figure` (thread-safe saves)."""

_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
"""Same set as the reports' ``sanitise`` (XML-illegal control characters), defined here
because the import contracts forbid plotting from importing the reporting package."""


def _clean(text: str) -> str:
    """Strip control characters a font has no glyph for (they render as boxes)."""
    return _XML_ILLEGAL.sub("", text)


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
        return fig

    ax.set_title(_clean(bundle.equation_text), fontsize=9, parse_math=False)
    if bundle.curve is not None:
        ax.plot(
            bundle.curve.x,
            bundle.curve.y,
            color=_CURVE_COLOR,
            linewidth=1.6,
            label="fitted curve",
            zorder=3,
        )
    if show_confidence and bundle.confidence_band is not None:
        _draw_band(ax, bundle.confidence_band, _CONFIDENCE_COLOR)
    if show_prediction and bundle.prediction_band is not None:
        _draw_band(ax, bundle.prediction_band, _PREDICTION_COLOR)
    ax.legend()
    return fig


def _draw_band(ax: Axes, band: Band, color: str) -> None:
    ax.fill_between(
        band.x,
        band.lower,
        band.upper,
        color=color,
        alpha=0.2,
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
