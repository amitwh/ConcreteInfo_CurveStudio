"""PlotView: the interactive pyqtgraph scatter of the active dataset.

Large datasets stay cheap: the drawn scatter passes through unchanged up to
``SCATTER_DISPLAY_MAX_POINTS`` samples and above that is a min/max envelope
(the web workspace's ``decimateMinMax`` port — bucketed by x-value, so safe for
an unsorted scatter, unlike pyqtgraph's auto peak mode, which presumes uniform x
spacing); fitted curves enable pyqtgraph's auto downsampling, which their
uniform-grid samples never trip. Updates reuse the drawn plot items instead of
removing and re-adding them. Whatever is decimated is said so under the plot
("showing X of N points"). A log axis is refused (with a reason) while the data
contain values ≤ 0; export goes through ``pyqtgraph.exporters`` with the crosshair
hidden. Publication-quality Matplotlib export arrives with D3.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from curvestudio_desktop.theme import current_scheme, mono_font

if TYPE_CHECKING:
    import numpy as np
    import pyqtgraph as pg

    from curvestudio.tools.stats import Plot

_DARK_PLOT_BACKGROUND = "#1e1e1e"
_DARK_PLOT_FOREGROUND = "#d4d4d4"
#: Bands sit below the curves so a fill never tints the line it belongs to.
_BAND_Z = 0.5
_CURVE_Z = 2.0
#: The in-app brand mark: the same text (and faintness) as the published figures.
_WATERMARK_Z = 10.0
#: The display budget of the CPU-raster symbol scatter, above which the drawn scatter
#: becomes a min/max envelope (see :func:`_decimate_min_max`) — the desktop counterpart
#: of the web workspace's ``GL_DISPLAY_MAX_POINTS`` (200k, just under the WebGL knee;
#: here every symbol is blitted individually, so the budget is far tighter, and a
#: ~1400 px plot at symbol size 5 saturates far below it anyway). ``scatter_data`` and
#: the data table always hold every sample; only the drawn subset shrinks, and the line
#: under the plot says so.
SCATTER_DISPLAY_MAX_POINTS = 20_000


def _watermark_html() -> str:
    """The plot watermark, from the reporting layer's constants (imported on first use:
    the plotting package pulls the fitting stack through its figures module)."""
    from curvestudio_plotting import WATERMARK_ALPHA, WATERMARK_TEXT

    return (
        f'<span style="color: rgba(70,70,70,{WATERMARK_ALPHA:g}); font-size: 7pt;">'
        f"{WATERMARK_TEXT}</span>"
    )


_EQUATION_LINE_WIDTH = 52
_EQUATION_MAX_LINES = 4


def _decimate_min_max(
    x: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """The drawn subset of ``(x, y)``: a min/max envelope within ``max_points`` samples.

    The desktop port of the web workspace's ``decimateMinMax`` (apps/web/src/
    workspace/traces.ts): the drawable x-range is split into ``ceil(max_points / 2)``
    buckets and every bucket contributes its least-y and greatest-y sample (once, when
    one sample wins both); the whole range's least-x and greatest-x samples are always
    kept, so the drawn x-span never shrinks, and a spike inside one bucket cannot
    vanish. Buckets are by x-value, not input order, so an unsorted scatter decimates
    just as correctly — the property pyqtgraph's auto peak mode lacks. Among equal
    extremes the first sample wins, which keeps the output deterministic. Non-finite
    samples are skipped: such a sample draws no marker anyway.
    """
    import numpy as np

    if x.size <= max_points:
        return x, y
    finite = np.nonzero(np.isfinite(x) & np.isfinite(y))[0]
    if finite.size == 0:
        return x[:0], y[:0]
    fx = x[finite]
    fy = y[finite]
    buckets = max(1, int(np.ceil(max_points / 2)))
    width = (float(fx.max()) - float(fx.min())) / buckets
    if width > 0.0:
        bucket = np.clip(((fx - fx.min()) / width).astype(np.int64), 0, buckets - 1)
    else:
        bucket = np.zeros(fx.size, dtype=np.int64)
    # the least-y / greatest-y sample per bucket, by position in ``finite``: sort by
    # bucket then y; each bucket's block gives its min-y at the head and max-y at the tail
    order = np.lexsort((fy, bucket))
    grouped = bucket[order]
    group_edges = np.flatnonzero(np.diff(grouped))
    starts = np.concatenate(([0], group_edges + 1))
    ends = np.concatenate((group_edges, [grouped.size - 1]))
    kept = np.union1d(
        finite[order[np.union1d(starts, ends)]],  # each bucket's min-y / max-y sample
        finite[[int(np.argmin(fx)), int(np.argmax(fx))]],  # the range's least-x / greatest-x
    )
    return x[kept], y[kept]


def _wrap_equation(
    text: str, width: int = _EQUATION_LINE_WIDTH, max_lines: int = _EQUATION_MAX_LINES
) -> list[str]:
    """Wrap a fitted equation at its `` + ``/`` - `` terms, capped with an honest ellipsis.

    A degree-8 polynomial runs to 120+ characters; the plot caption shows the first
    lines and an ellipsis rather than silently clipping a line mid-term. The complete
    equation is always in the details window and the reports.
    """
    terms: list[str] = []
    for part in re.split(r"(?= [+-] )", text):
        terms.append(part)
    lines: list[str] = []
    current = ""
    for term in terms:
        if current and len(current) + len(term) > width:
            lines.append(current)
            current = term.lstrip()
        else:
            current += term
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = f"{lines[-1].rstrip()} …"
    return lines


@dataclass(frozen=True)
class OverlayCurve:
    """One model curve in a multi-curve overlay: label, samples and 1-based display position.

    ``position`` is the row's ordinal in the displayed ranking — never its tied
    ``RankedFit.rank`` — so two rows sharing a rank still get different colours.
    """

    name: str
    x: Sequence[float]
    y: Sequence[float]
    position: int


class PlotView(QWidget):
    """Scatter plot with labelled axes, crosshair readout and log-axis toggles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        # pyqtgraph is the one heavy import the window itself needs to draw: it lands
        # here, on first construction, not at module import.
        import pyqtgraph as pg

        super().__init__(parent)
        self._x_name, self._y_name = "x", "y"
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._log_x = False
        self._log_y = False
        self._can_log_x = False
        self._can_log_y = False
        self._plot = pg.PlotWidget(self)
        self._plot.setMouseEnabled(x=True, y=True)
        self._scatter: pg.PlotDataItem | None = None
        self._curves: list[pg.PlotDataItem] = []
        self._curve_positions: list[int] = []
        self._curve_names: list[str] = []
        self._legend: pg.LegendItem | None = None
        self._legend_names: list[str] = []
        self._confidence: pg.FillBetweenItem | None = None
        self._prediction: pg.FillBetweenItem | None = None
        # the band fills' backing lower/upper curves, kept for in-place updates
        self._confidence_curves: tuple[pg.PlotDataItem, pg.PlotDataItem] | None = None
        self._prediction_curves: tuple[pg.PlotDataItem, pg.PlotDataItem] | None = None
        self._band_x: Sequence[float] | None = None
        self._band_confidence: tuple[Sequence[float], Sequence[float]] | None = None
        self._band_prediction: tuple[Sequence[float], Sequence[float]] | None = None
        # what the drawn scatter honestly holds: (shown, total) when it is a decimated
        # subset of the dataset, None when every sample is drawn
        self._display_shown: int | None = None
        self._display_total: int | None = None
        # the statistical plot's items, and the view state they displaced while shown
        self._stats_items: list[Any] = []
        self._stats_saved_log: tuple[bool, bool] | None = None
        self._stats_equation_was_visible = False
        empty_label = QLabel("Import data to see it plotted")
        empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_page: QWidget = empty_label
        # a non-modal inline notice over the plot (a ranking row with no sampled curve)
        self._hint = QLabel(self._plot)
        self._hint.setStyleSheet(
            "QLabel { background: rgba(90, 90, 90, 210); color: white;"
            " border-radius: 4px; padding: 4px 10px; }"
        )
        self._hint.hide()
        # the selected model's equation, pinned to the plot's top-left corner: visible
        # where the curve is, instead of a trip to the details window to read it
        self._equation = QLabel(self._plot)
        self._equation_text: tuple[str, str, int] | None = None
        self._equation.setObjectName("plotEquation")
        self._equation.setTextFormat(Qt.TextFormat.RichText)
        self._equation.setWordWrap(True)
        self._equation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._equation.hide()
        self._plot.installEventFilter(self)
        self._readout = QLabel("", self)
        self._readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._readout.setFont(mono_font(9))
        stack = QStackedLayout()
        stack.addWidget(self._plot)
        stack.addWidget(self._empty_page)
        stack.setCurrentWidget(self._empty_page)
        self._stack = stack
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(stack)
        outer.addWidget(self._readout)
        self._install_crosshair()
        # The faint brand mark is a scene item (not a QLabel overlay) so both exporters
        # — ImageExporter's PNG and the QSvgGenerator SVG, which serialise the scene —
        # render it. NoButton keeps it out of the crosshair's way.
        # empty until first asked: the watermark's constants live in the plotting
        # package, whose import would pull the whole fitting stack onto the window's
        # path for the sake of one label
        self._watermark = pg.LabelItem(text="", parent=self._plot.graphicsItem())
        self._watermark.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-8, -6))
        self._watermark.setZValue(_WATERMARK_Z)
        self._watermark.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # pyqtgraph's auto ("peak") downsampling stays off at the PlotItem level: peak
        # decimation presumes uniform x spacing and would distort an unsorted scatter.
        # The scatter decimates itself instead (an unsorted-safe min/max envelope above
        # SCATTER_DISPLAY_MAX_POINTS); fitted curves enable it per item after being
        # added — they are uniform-grid samples, so the presumption holds there.
        # clip-to-view stays on.
        self._plot.getPlotItem().setDownsampling(auto=False)
        self._plot.getPlotItem().setClipToView(True)
        self.apply_scheme()

    def apply_scheme(self, scheme: str | None = None) -> None:
        import pyqtgraph as pg

        """Repaint the plot surface, viewbox, axes and labels for the light/dark scheme.

        ``apply_theme`` reconfigures pyqtgraph's *defaults*, but a PlotWidget built before
        the switch keeps the background it was created with. This is the explicit repaint
        the theme action calls, so a dark UI never surrounds a white plot.
        """
        resolved = scheme if scheme is not None else current_scheme()
        dark = resolved == "dark"
        background = _DARK_PLOT_BACKGROUND if dark else "w"
        foreground = _DARK_PLOT_FOREGROUND if dark else "k"
        self._plot.setBackground(background)
        self._plot.getPlotItem().getViewBox().setBackgroundColor(background)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)
            axis.label.setDefaultTextColor(QColor(foreground))
        self._scheme = resolved
        from curvestudio_plotting.palette import data_color

        # Data points stay neutral in either theme; bands pick the theme's neutral fill.
        if self._scatter is not None:
            neutral = data_color(dark=dark)
            self._scatter.setSymbolBrush(pg.mkBrush(neutral))
            self._scatter.setSymbolPen(pg.mkPen(neutral))
        self._draw_bands()

    def plot_background(self) -> str:
        """The drawn plot background as ``#rrggbb`` (what the theme tests assert)."""
        return str(self._plot.backgroundBrush().color().name())

    def watermark_text(self) -> str:
        from curvestudio_plotting import WATERMARK_TEXT

        self._fill_watermark()
        """The brand mark drawn faintly in the plot's bottom-right corner (also exported)."""
        return WATERMARK_TEXT

    def _fill_watermark(self) -> None:
        """Set the watermark label's text; the plotting package is imported only now."""
        if self._watermark.text.startswith("<span"):
            return
        self._watermark.setText(_watermark_html())

    # -- data --------------------------------------------------------------------------------

    def set_empty_page(self, page: QWidget) -> None:
        """Swap the page shown while no data is plotted (the guided empty start).

        The window owns that page — the actions and recent projects on it are the
        shell's business, not the plot's — but the stack (and the no-data logic that
        drives it) stays here, so the swap is an explicit hand-off, once, at build.
        """
        was_empty = self._stack.currentWidget() is self._empty_page
        self._stack.removeWidget(self._empty_page)
        self._empty_page.deleteLater()
        self._empty_page = page
        self._stack.addWidget(page)
        if was_empty:
            self._stack.setCurrentWidget(page)

    def is_empty_page_shown(self) -> bool:
        """Whether the no-data page stands in for the plot right now."""
        return self._stack.currentWidget() is self._empty_page

    def set_data(self, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
        import pyqtgraph as pg

        from curvestudio_plotting.palette import data_color

        """Replace the scatter with ``(x, y)``, labelled from the column names.

        The drawn scatter holds every sample up to ``SCATTER_DISPLAY_MAX_POINTS``;
        above that it draws a min/max envelope and says so under the plot. The item
        itself is reused when one is already drawn — a dataset switch or a re-selection
        swaps the arrays in place instead of removing and re-adding the item.
        """
        self._x_name, self._y_name = x_name, y_name
        self._x, self._y = x, y
        shown_x, shown_y = self._display_arrays(x, y)
        neutral = data_color(dark=self._scheme == "dark")
        if self._scatter is None:
            self._scatter = pg.PlotDataItem(
                shown_x,
                shown_y,
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolBrush=pg.mkBrush(neutral),
                symbolPen=pg.mkPen(neutral),
            )
            # clip-to-view comes from the PlotItem config in __init__: configuring the
            # item before adding it breaks pyqtgraph 0.14's itemChange path
            self._plot.addItem(self._scatter)
        else:
            self._scatter.setData(shown_x, shown_y)
        self._plot.setLabel("bottom", x_name)
        self._plot.setLabel("left", y_name)
        self._stack.setCurrentWidget(self._plot)
        self._recompute_log_availability(x, y)
        self.reset_view()
        self._refresh_readout()

    def _display_arrays(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The arrays the scatter draws: full up to the display budget, else the
        min/max envelope; records the honest (shown, total) counts alongside."""
        if x.size > SCATTER_DISPLAY_MAX_POINTS:
            shown_x, shown_y = _decimate_min_max(x, y, SCATTER_DISPLAY_MAX_POINTS)
            self._display_shown = int(shown_x.size)
            self._display_total = int(x.size)
            return shown_x, shown_y
        self._display_shown = None
        self._display_total = None
        return x, y

    def clear_data(self) -> None:
        if self._scatter is not None:
            self._plot.removeItem(self._scatter)
            self._scatter = None
        self.clear_fit()
        self._x = self._y = None
        self._display_shown = None
        self._display_total = None
        self._refresh_readout()
        self._stack.setCurrentWidget(self._empty_page)
        self._can_log_x = self._can_log_y = False
        self._set_log(x=False, y=False)

    def scatter_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The full dataset arrays — never the decimated drawn subset — or ``(None, None)``.

        The drawn scatter is a min/max envelope above ``SCATTER_DISPLAY_MAX_POINTS``
        (see :meth:`is_downsampling`); these arrays always hold every sample.
        """
        return self._x, self._y

    def axis_labels(self) -> tuple[str, str]:
        return self._x_name, self._y_name

    # -- fitted curve and bands ---------------------------------------------------------------

    def set_fit(
        self,
        x: Sequence[float],
        y: Sequence[float],
        confidence: tuple[Sequence[float], Sequence[float]] | None = None,
        prediction: tuple[Sequence[float], Sequence[float]] | None = None,
        *,
        position: int = 1,
        name: str = "fitted curve",
    ) -> None:
        """Draw one fitted curve (with optional bands) over the scatter.

        The curve takes the colour of its 1-based **display position** in the ranking
        (position 1 is the brand accent); bands are neutral translucent fills, never a
        model colour, so a band can never be read as another curve. Callers never call
        this for a fit that did not converge — there is no curve to draw then (see
        :meth:`clear_fit`). One curve is the one-curve overlay: the same item reuse
        applies.
        """
        self.set_fit_overlay(
            [OverlayCurve(name=name, x=x, y=y, position=position)],
            band_x=x,
            confidence=confidence,
            prediction=prediction,
        )

    def set_fit_overlay(
        self,
        curves: Sequence[OverlayCurve],
        *,
        band_x: Sequence[float] | None = None,
        confidence: tuple[Sequence[float], Sequence[float]] | None = None,
        prediction: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        """Draw several model curves, each in its display-position colour, with a legend.

        Bands belong to the selected model only and stay neutral; ``band_x`` is that
        model's curve x (it need not be the first curve in ``curves``). Curve items are
        reused when the overlay keeps the same number of curves — every ranking-row
        click redraws through here, and remove+re-add churn is pure cost then; a
        genuine shape change (a different curve count) rebuilds them.
        """
        import numpy as np
        import pyqtgraph as pg

        from curvestudio_plotting.palette import curve_style

        reuse = len(self._curves) == len(curves)
        if reuse:
            # the items are reused, but a redraw is still a new selection: drop the
            # previous row's caption and inline notice exactly as clear_fit would
            self._hint.hide()
            self.clear_equation()
        else:
            self.clear_fit()
        self._band_x = list(band_x) if band_x is not None else None
        self._band_confidence = confidence
        self._band_prediction = prediction
        if reuse:
            for item, entry in zip(self._curves, curves, strict=True):
                colour, width = curve_style(entry.position)
                item.setData(
                    np.asarray(entry.x, dtype=np.float64),
                    np.asarray(entry.y, dtype=np.float64),
                )
                item.setPen(pg.mkPen(colour, width=width))
        else:
            fresh: list[pg.PlotDataItem] = []
            for entry in curves:
                colour, width = curve_style(entry.position)
                item = pg.PlotDataItem(
                    np.asarray(entry.x, dtype=np.float64),
                    np.asarray(entry.y, dtype=np.float64),
                    pen=pg.mkPen(colour, width=width),
                )
                item.setZValue(_CURVE_Z)
                self._plot.addItem(item)
                # auto ("peak") downsampling draws a curve holding far more samples than
                # the view can show as a min/max envelope; engine curves are 200
                # uniform-grid samples and pass through untouched. Set after addItem:
                # PlotItem applies its own (off) downsampling config when an item is
                # added, which would override an item-level setting made before.
                item.setDownsampling(auto=True, method="peak")
                fresh.append(item)
            self._curves = fresh
        self._curve_positions = [entry.position for entry in curves]
        self._curve_names = [entry.name for entry in curves]
        self._draw_bands()
        self._rebuild_legend()

    def clear_fit(self) -> None:
        """Remove every curve, the bands and the legend (a dataset switch, or no curve)."""
        for item in (*self._curves, self._confidence, self._prediction):
            if item is not None:
                self._plot.removeItem(item)
        self._curves.clear()
        self._curve_positions.clear()
        self._curve_names.clear()
        self._confidence = None
        self._prediction = None
        self._confidence_curves = None
        self._prediction_curves = None
        self._band_x = None
        self._band_confidence = None
        self._band_prediction = None
        self._rebuild_legend()
        self._hint.hide()
        self.clear_equation()

    # -- legend -------------------------------------------------------------------------------

    def _rebuild_legend(self) -> None:
        """Name the models when more than one curve is drawn; hide it otherwise."""
        if len(self._curves) < 2:
            self._legend_names = []
            if self._legend is not None:
                self._legend.clear()
                self._legend.setVisible(False)
            return
        if self._legend is None:
            self._legend = self._plot.getPlotItem().addLegend(offset=(10, 10))
        self._legend.clear()
        for item, name in zip(self._curves, self._curve_names, strict=True):
            self._legend.addItem(item, name)
        self._legend.setVisible(True)
        self._legend_names = list(self._curve_names)

    def legend_names(self) -> list[str]:
        """The legend's model labels (empty when the legend is hidden)."""
        return list(self._legend_names)

    def drawn_curve_colors(self) -> list[str]:
        """The drawn curves' pen colours as ``#rrggbb``, in draw order."""
        colors: list[str] = []
        for item in self._curves:
            pen = item.opts.get("pen")
            if pen is not None and pen.style() != Qt.PenStyle.NoPen:
                colors.append(pen.color().name())
        return colors

    def drawn_curve_positions(self) -> list[int]:
        """The 1-based display position of each drawn curve, in draw order."""
        return list(self._curve_positions)

    # -- inline hint -------------------------------------------------------------------------

    def show_hint(self, text: str) -> None:
        """A one-line notice centred at the top of the plot area.

        Used when honesty beats drawing: a ranking row outside the sampled top-N has
        no stored curve, and saying so inline beats silently clearing the overlay.
        """
        self._hint.setText(text)
        self._hint.adjustSize()
        self._hint.show()
        self._position_hint()

    def hint_text(self) -> str:
        return self._hint.text()

    def hint_visible(self) -> bool:
        return not self._hint.isHidden()

    def _position_hint(self) -> None:
        """Keep the hint centred at the top of the plot as it resizes."""
        if self._hint.isHidden():
            return
        left = max(8, (self._plot.width() - self._hint.width()) // 2)
        self._hint.move(left, 8)

    # -- the selected model's equation on the plot ---------------------------------------------

    def set_equation(self, model: str, equation: str, *, position: int = 1) -> None:
        """Caption the plot with the selected model's fitted equation.

        ``position`` is the model's 1-based display position, so the equation takes the
        same colour as its curve (never the tied rank's colour). The engine stores ``**``;
        it is shown as ``^``, the way it is written in the details window and reports.
        """
        from curvestudio.workflow import format_equation

        display = format_equation(equation)
        if not display:
            self.clear_equation()
            return
        # a generated family is named by its template, which carries the engine's **
        self._equation_text = (format_equation(model), display, position)
        self._render_equation()
        self._position_equation()
        self._equation.show()

    def clear_equation(self) -> None:
        self._equation_text = None
        self._equation.setText("")
        self._equation.hide()

    def equation_text(self) -> str | None:
        """The caption as ``"model\\nequation"`` text (``None`` when nothing is shown)."""
        if self._equation_text is None:
            return None
        model, display, _position = self._equation_text
        return f"{model}\n{display}"

    def equation_visible(self) -> bool:
        return not self._equation.isHidden()

    def _render_equation(self) -> None:
        from curvestudio_plotting.palette import curve_style

        if self._equation_text is None:
            return
        model, display, position = self._equation_text
        colour, _width = curve_style(position)
        dark = getattr(self, "_scheme", None) == "dark"
        border = "#4a4643" if dark else "#e3e3e3"
        background = "rgba(28, 25, 23, 0.90)" if dark else "rgba(255, 255, 255, 0.93)"
        text = "#f1f1f1" if dark else "#0d0b09"
        self._equation.setStyleSheet(
            f"QLabel#plotEquation {{ background: {background}; color: {text};"
            f" border: 1px solid {border}; border-radius: 4px; padding: 3px 7px; }}"
        )
        lines = _wrap_equation(display)
        self._equation.setText(
            f'<div style="font-size:8.5pt;">{escape(model)}</div>'
            f"<div style=\"color:{colour}; font-family:'JetBrains Mono'; font-size:8.5pt;\">"
            f"{'<br/>'.join(escape(line) for line in lines)}</div>"
        )

    def _position_equation(self) -> None:
        """Pin the caption to the plot's top-left corner, clear of the y-axis labels."""
        if self._equation.isHidden():
            return
        axis = self._plot.getPlotItem().getAxis("left")
        left = max(10, int(axis.width()) + 6) if axis is not None else 10
        width = max(220, min(520, int((self._plot.width() - left) * 0.78)))
        self._equation.setFixedWidth(width)
        self._equation.adjustSize()
        self._equation.move(left, 8)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._plot and event.type() is QEvent.Type.Resize:
            self._position_hint()
            self._position_equation()
        return super().eventFilter(watched, event)

    def fit_curve_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        import numpy as np

        """The first drawn curve's arrays, or ``None`` when no curve is shown."""
        if not self._curves:
            return None
        x, y = self._curves[0].getData()
        if x is None or y is None:
            return None
        return np.asarray(x), np.asarray(y)

    def band_colors(self) -> list[str]:
        """The neutral band fills as ``#rrggbb`` (alpha dropped), confidence then prediction."""
        fills = (self._confidence, self._prediction)
        return [item.brush().color().name() for item in fills if item is not None]

    def _draw_bands(self) -> None:
        """(Re)draw the neutral bands for the current theme; no-op without band data."""
        from curvestudio_plotting.palette import band_fill

        dark = self._scheme == "dark"
        self._confidence, self._confidence_curves = self._sync_band(
            self._confidence,
            self._confidence_curves,
            self._band_confidence,
            band_fill("confidence", dark=dark),
        )
        self._prediction, self._prediction_curves = self._sync_band(
            self._prediction,
            self._prediction_curves,
            self._band_prediction,
            band_fill("prediction", dark=dark),
        )

    def _sync_band(
        self,
        fill: pg.FillBetweenItem | None,
        curves: tuple[pg.PlotDataItem, pg.PlotDataItem] | None,
        band: tuple[Sequence[float], Sequence[float]] | None,
        colour: str,
    ) -> tuple[pg.FillBetweenItem | None, tuple[pg.PlotDataItem, pg.PlotDataItem] | None]:
        """One band's items matching ``band``: a fill that is already drawn is updated
        in place (new arrays via its curves, new theme brush), one that appears for the
        first time is built. Returns the fill and the lower/upper curve pair backing it.
        """
        import numpy as np
        import pyqtgraph as pg

        if band is None:
            if fill is not None:
                self._plot.removeItem(fill)
            return None, None
        if fill is None or curves is None:
            lower = pg.PlotDataItem()
            upper = pg.PlotDataItem()
            fill = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(colour))
            fill.setZValue(_BAND_Z)
            self._plot.addItem(fill)
        else:
            lower, upper = curves
            # the theme may have changed since the fill was drawn
            fill.setBrush(pg.mkBrush(colour))
        band_x = np.asarray(self._band_x, dtype=np.float64)
        lower.setData(band_x, np.asarray(band[0], dtype=np.float64))
        upper.setData(band_x, np.asarray(band[1], dtype=np.float64))
        return fill, (lower, upper)

    # -- log axes ----------------------------------------------------------------------------

    @property
    def can_log_x(self) -> bool:
        return self._can_log_x

    @property
    def can_log_y(self) -> bool:
        return self._can_log_y

    @property
    def log_x_enabled(self) -> bool:
        return self._log_x

    @property
    def log_y_enabled(self) -> bool:
        return self._log_y

    def set_log_x(self, enabled: bool) -> None:
        if enabled and not self._can_log_x:
            return
        self._set_log(x=enabled, y=self._log_y)

    def set_log_y(self, enabled: bool) -> None:
        if enabled and not self._can_log_y:
            return
        self._set_log(x=self._log_x, y=enabled)

    def _set_log(self, *, x: bool, y: bool) -> None:
        self._log_x, self._log_y = x, y
        self._plot.getPlotItem().setLogMode(x=x, y=y)

    def _recompute_log_availability(self, x: np.ndarray, y: np.ndarray) -> None:
        import numpy as np

        finite_x = x[np.isfinite(x)]
        finite_y = y[np.isfinite(y)]
        self._can_log_x = bool(finite_x.size and (finite_x > 0).all())
        self._can_log_y = bool(finite_y.size and (finite_y > 0).all())
        if not self._can_log_x and self._log_x:
            self._set_log(x=False, y=self._log_y)
        if not self._can_log_y and self._log_y:
            self._set_log(x=self._log_x, y=False)

    # -- navigation --------------------------------------------------------------------------

    def reset_view(self) -> None:
        """Fit the full data range back into the view."""
        self._plot.autoRange()

    def view_ranges(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if self._scatter is None:
            return None
        x_range, y_range = self._plot.viewRange()
        return (float(x_range[0]), float(x_range[1])), (float(y_range[0]), float(y_range[1]))

    def is_downsampling(self) -> bool:
        """Whether the drawn scatter is a decimated subset of the dataset.

        Above ``SCATTER_DISPLAY_MAX_POINTS`` samples the view draws a min/max envelope
        (see :func:`_decimate_min_max`) instead of every sample; this method and the
        "showing X of N points" line under the plot say so, and :meth:`scatter_data`
        still returns the full arrays. pyqtgraph's own auto peak mode never touches the
        scatter (it presumes uniform x spacing); fitted curves enable it, but their
        200-sample uniform grids never trip it either.
        """
        return self._display_shown is not None

    # -- crosshair ---------------------------------------------------------------------------

    def _install_crosshair(self) -> None:
        import pyqtgraph as pg

        self._v_line = pg.InfiniteLine(angle=90, movable=False)
        self._h_line = pg.InfiniteLine(angle=0, movable=False)
        self._v_line.hide()
        self._h_line.hide()
        self._plot.addItem(self._v_line, ignoreBounds=True)
        self._plot.addItem(self._h_line, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            self._plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

    def _on_mouse_moved(self, event: Sequence[QPointF]) -> None:
        position = event[0]
        if self._plot.sceneBoundingRect().contains(position):
            self.show_crosshair_at(position)
        else:
            self.hide_crosshair()

    def show_crosshair_at(self, position: QPointF) -> None:
        """Move the crosshair to ``position`` (scene coordinates) and update the readout."""
        point = self._plot.plotItem.vb.mapSceneToView(position)
        self._v_line.setPos(point.x())
        self._h_line.setPos(point.y())
        self._v_line.show()
        self._h_line.show()
        # a log axis plots log10(value); the readout reports the data value itself
        x_value = 10.0 ** point.x() if self._log_x else point.x()
        y_value = 10.0 ** point.y() if self._log_y else point.y()
        self._readout.setText(f"{self._x_name} = {x_value:.6g}    {self._y_name} = {y_value:.6g}")

    def hide_crosshair(self) -> None:
        """Drop the crosshair lines; the readout falls back to the decimation note."""
        self._v_line.hide()
        self._h_line.hide()
        self._refresh_readout()

    def _refresh_readout(self) -> None:
        """The line under the plot: crosshair coordinates while they track the mouse,
        otherwise the decimation note when the drawn scatter is a subset."""
        if self._v_line.isVisible():
            return
        if self._display_shown is not None:
            self._readout.setText(
                f"showing {self._display_shown:,} of {self._display_total:,} points"
            )
        else:
            self._readout.setText("")

    def crosshair_visible(self) -> bool:
        return bool(self._v_line.isVisible() and self._h_line.isVisible())

    def scene_position_of(self, x: float, y: float) -> QPointF:
        """The scene position of a data (view) coordinate — the inverse crosshair path."""
        return cast(QPointF, self._plot.plotItem.vb.mapViewToScene(QPointF(x, y)))

    def crosshair_text(self) -> str:
        return self._readout.text()

    # -- statistical plots -------------------------------------------------------------------

    def show_stats_plot(self, plot: Plot) -> None:
        from curvestudio_desktop.views.stats_plots import draw_stats_plot

        """Show one statistical plot model in place of the dataset and the fitted curve.

        The scatter, every curve and its bands, the equation caption and the inline hint
        are hidden — never removed — and the log axes are switched off, because a
        statistical plot brings its own axes. :meth:`clear_stats_plot` puts them back.
        """
        self.clear_stats_plot()
        self._stats_saved_log = (self._log_x, self._log_y)
        self._stats_equation_was_visible = not self._equation.isHidden()
        self._set_log(x=False, y=False)
        for item in self._data_items():
            item.hide()
        self._hint.hide()
        # the dataset (and its decimation note) is off-stage while a stats plot shows
        self._readout.setText("")
        if self._stats_equation_was_visible:
            self._equation.hide()
        self._legend_names = []
        self._stack.setCurrentWidget(self._plot)
        self._stats_items = draw_stats_plot(self._plot.getPlotItem(), plot)
        self._plot.autoRange()

    def clear_stats_plot(self) -> None:
        """Remove the statistical plot's items and show the dataset and the curve again."""
        plot_item = self._plot.getPlotItem()
        for item in self._stats_items:
            if item in plot_item.items:
                plot_item.removeItem(item)
            else:
                # a note is parented to the plot item rather than added to its items
                item.setParentItem(None)
                item.deleteLater()
        self._stats_items = []
        for item in self._data_items():
            item.show()
        if self._stats_equation_was_visible:
            self._equation.show()
        self._stats_equation_was_visible = False
        if self._stats_saved_log is not None:
            log_x, log_y = self._stats_saved_log
            self._set_log(x=log_x, y=log_y)
            self._stats_saved_log = None
        self._refresh_readout()
        legend = plot_item.legend
        if legend is not None:
            legend.clear()
            legend.setVisible(False)
        self._rebuild_legend()
        if self._scatter is None:
            self._stack.setCurrentWidget(self._empty_page)
        else:
            self.reset_view()

    def stats_items(self) -> list[Any]:
        """The items the statistical plot drew (empty when none is shown)."""
        return list(self._stats_items)

    def has_stats_plot(self) -> bool:
        """Whether a statistical plot is shown in place of the data."""
        return bool(self._stats_items)

    def _data_items(self) -> list[Any]:
        """The dataset, the fitted curves and their bands: what a stats plot hides."""
        items: list[Any] = []
        if self._scatter is not None:
            items.append(self._scatter)
        items.extend(self._curves)
        for band in (self._confidence, self._prediction):
            if band is not None:
                items.append(band)
        return items

    # -- export ------------------------------------------------------------------------------

    def export_png(self, target: Path) -> bool:
        """Export the plot as PNG via ``pyqtgraph.exporters.ImageExporter``."""
        if self._scatter is None and not self._stats_items:
            return False
        from pyqtgraph.exporters import ImageExporter

        visible = self.crosshair_visible()
        self._v_line.hide()
        self._h_line.hide()
        try:
            ImageExporter(self._plot.plotItem).export(str(target))
        finally:
            if visible:
                self._v_line.show()
                self._h_line.show()
        return True

    def export_svg(self, target: Path) -> bool:
        """Export the plot as vector SVG via Qt's QSvgGenerator.

        pyqtgraph 0.14.0's own SVGExporter raises ``ValueError`` on ordinary plot
        content (its path parser cannot handle comma-less SVG path tokens), so the
        vector export renders the scene through Qt's generator instead.
        """
        if self._scatter is None and not self._stats_items:
            return False
        from PySide6.QtCore import QRectF, QSize
        from PySide6.QtGui import QPainter
        from PySide6.QtSvg import QSvgGenerator

        source = self._plot.scene().itemsBoundingRect()
        if source.isEmpty():
            return False
        width, height = source.width(), source.height()
        generator = QSvgGenerator()
        generator.setFileName(str(target))
        generator.setSize(QSize(max(1, int(width)), max(1, int(height))))
        generator.setViewBox(QRectF(0.0, 0.0, width, height))
        painter = QPainter()
        painter.begin(generator)
        visible = self.crosshair_visible()
        self._v_line.hide()
        self._h_line.hide()
        try:
            self._plot.scene().render(painter, QRectF(0.0, 0.0, width, height), source)
        finally:
            painter.end()
            if visible:
                self._v_line.show()
                self._h_line.show()
        return True
