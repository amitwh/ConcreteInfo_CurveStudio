"""PlotView: the interactive pyqtgraph scatter of the active dataset.

Clip-to-view keeps large datasets interactive; auto ("peak") downsampling is
deliberately off — peak decimation assumes sorted x values and silently distorts
an unsorted scatter. A log axis is refused (with a reason) while the data contain
values ≤ 0; export goes through ``pyqtgraph.exporters`` with the crosshair hidden.
Publication-quality Matplotlib export arrives with D3.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from curvestudio_desktop.theme import current_scheme, mono_font
from curvestudio_plotting.palette import band_fill, curve_style, data_color

_DARK_PLOT_BACKGROUND = "#1e1e1e"
_DARK_PLOT_FOREGROUND = "#d4d4d4"
#: Bands sit below the curves so a fill never tints the line it belongs to.
_BAND_Z = 0.5
_CURVE_Z = 2.0


@dataclass(frozen=True)
class OverlayCurve:
    """One model curve in a multi-curve overlay: its label, samples and 1-based rank."""

    name: str
    x: Sequence[float]
    y: Sequence[float]
    rank: int


class PlotView(QWidget):
    """Scatter plot with labelled axes, crosshair readout and log-axis toggles."""

    def __init__(self, parent: QWidget | None = None) -> None:
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
        self._curve_ranks: list[int] = []
        self._curve_names: list[str] = []
        self._legend: pg.LegendItem | None = None
        self._legend_names: list[str] = []
        self._confidence: pg.FillBetweenItem | None = None
        self._prediction: pg.FillBetweenItem | None = None
        self._band_x: Sequence[float] | None = None
        self._band_confidence: tuple[Sequence[float], Sequence[float]] | None = None
        self._band_prediction: tuple[Sequence[float], Sequence[float]] | None = None
        self._empty_label = QLabel("Import data to see it plotted")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # a non-modal inline notice over the plot (a ranking row with no sampled curve)
        self._hint = QLabel(self._plot)
        self._hint.setStyleSheet(
            "QLabel { background: rgba(90, 90, 90, 210); color: white;"
            " border-radius: 4px; padding: 4px 10px; }"
        )
        self._hint.hide()
        self._plot.installEventFilter(self)
        self._readout = QLabel("", self)
        self._readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._readout.setFont(mono_font(9))
        stack = QStackedLayout()
        stack.addWidget(self._plot)
        stack.addWidget(self._empty_label)
        stack.setCurrentWidget(self._empty_label)
        self._stack = stack
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(stack)
        outer.addWidget(self._readout)
        self._install_crosshair()
        # auto "peak" downsampling is off: peak decimation assumes sorted x and
        # distorts an unsorted scatter. clip-to-view stays on.
        self._plot.getPlotItem().setDownsampling(auto=False)
        self._plot.getPlotItem().setClipToView(True)
        self.apply_scheme()

    def apply_scheme(self, scheme: str | None = None) -> None:
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
        # Data points stay neutral in either theme; bands pick the theme's neutral fill.
        if self._scatter is not None:
            neutral = data_color(dark=dark)
            self._scatter.setSymbolBrush(pg.mkBrush(neutral))
            self._scatter.setSymbolPen(pg.mkPen(neutral))
        self._draw_bands()

    def plot_background(self) -> str:
        """The drawn plot background as ``#rrggbb`` (what the theme tests assert)."""
        return str(self._plot.backgroundBrush().color().name())

    # -- data --------------------------------------------------------------------------------

    def set_data(self, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
        """Replace the scatter with ``(x, y)``, labelled from the column names."""
        self._x_name, self._y_name = x_name, y_name
        self._x, self._y = x, y
        if self._scatter is not None:
            self._plot.removeItem(self._scatter)
        neutral = data_color(dark=self._scheme == "dark")
        self._scatter = pg.PlotDataItem(
            x,
            y,
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(neutral),
            symbolPen=pg.mkPen(neutral),
        )
        # clip-to-view comes from the PlotItem config in __init__: configuring the
        # item before adding it breaks pyqtgraph 0.14's itemChange path
        self._plot.addItem(self._scatter)
        self._plot.setLabel("bottom", x_name)
        self._plot.setLabel("left", y_name)
        self._stack.setCurrentWidget(self._plot)
        self._recompute_log_availability(x, y)
        self.reset_view()

    def clear_data(self) -> None:
        if self._scatter is not None:
            self._plot.removeItem(self._scatter)
            self._scatter = None
        self.clear_fit()
        self._x = self._y = None
        self._stack.setCurrentWidget(self._empty_label)
        self._can_log_x = self._can_log_y = False
        self._set_log(x=False, y=False)

    def scatter_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The full plotted arrays (never a downsampled view), or ``(None, None)``."""
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
        rank: int = 1,
        name: str = "fitted curve",
    ) -> None:
        """Draw one fitted curve (with optional bands) over the scatter.

        The curve takes its rank's colour from the shared palette (rank 1 is the brand
        accent); bands are neutral translucent fills, never a model colour, so a band can
        never be read as another curve. Callers never call this for a fit that did not
        converge — there is no curve to draw then (see :meth:`clear_fit`).
        """
        self.clear_fit()
        self._band_x = list(x)
        self._band_confidence = confidence
        self._band_prediction = prediction
        colour, width = curve_style(rank)
        item = pg.PlotDataItem(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            pen=pg.mkPen(colour, width=width),
        )
        item.setZValue(_CURVE_Z)
        self._plot.addItem(item)
        self._curves.append(item)
        self._curve_ranks.append(rank)
        self._curve_names.append(name)
        self._draw_bands()
        self._rebuild_legend()

    def set_fit_overlay(
        self,
        curves: Sequence[OverlayCurve],
        *,
        band_x: Sequence[float] | None = None,
        confidence: tuple[Sequence[float], Sequence[float]] | None = None,
        prediction: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        """Draw several model curves, each in its rank colour, with a named legend.

        Bands belong to the selected model only and stay neutral; ``band_x`` is that
        model's curve x (it need not be the first curve in ``curves``).
        """
        self.clear_fit()
        self._band_x = list(band_x) if band_x is not None else None
        self._band_confidence = confidence
        self._band_prediction = prediction
        for entry in curves:
            colour, width = curve_style(entry.rank)
            item = pg.PlotDataItem(
                np.asarray(entry.x, dtype=np.float64),
                np.asarray(entry.y, dtype=np.float64),
                pen=pg.mkPen(colour, width=width),
            )
            item.setZValue(_CURVE_Z)
            self._plot.addItem(item)
            self._curves.append(item)
            self._curve_ranks.append(entry.rank)
            self._curve_names.append(entry.name)
        self._draw_bands()
        self._rebuild_legend()

    def clear_fit(self) -> None:
        """Remove every curve, the bands and the legend (a dataset switch, or no curve)."""
        for item in (*self._curves, self._confidence, self._prediction):
            if item is not None:
                self._plot.removeItem(item)
        self._curves.clear()
        self._curve_ranks.clear()
        self._curve_names.clear()
        self._confidence = None
        self._prediction = None
        self._band_x = None
        self._band_confidence = None
        self._band_prediction = None
        self._rebuild_legend()
        self._hint.hide()

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

    def drawn_curve_ranks(self) -> list[int]:
        """The rank each drawn curve belongs to (1-based), in draw order."""
        return list(self._curve_ranks)

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

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._plot and event.type() is QEvent.Type.Resize:
            self._position_hint()
        return super().eventFilter(watched, event)

    def fit_curve_data(self) -> tuple[np.ndarray, np.ndarray] | None:
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
        for item in (self._confidence, self._prediction):
            if item is not None:
                self._plot.removeItem(item)
        self._confidence = None
        self._prediction = None
        if self._band_x is None:
            return
        dark = self._scheme == "dark"
        if self._band_confidence is not None:
            self._confidence = self._fill_between(
                self._band_x, self._band_confidence, band_fill("confidence", dark=dark)
            )
            self._plot.addItem(self._confidence)
        if self._band_prediction is not None:
            self._prediction = self._fill_between(
                self._band_x, self._band_prediction, band_fill("prediction", dark=dark)
            )
            self._plot.addItem(self._prediction)

    def _fill_between(
        self,
        x: Sequence[float],
        band: tuple[Sequence[float], Sequence[float]],
        fill: str,
    ) -> pg.FillBetweenItem:
        lower = pg.PlotDataItem(
            np.asarray(x, dtype=np.float64), np.asarray(band[0], dtype=np.float64)
        )
        upper = pg.PlotDataItem(
            np.asarray(x, dtype=np.float64), np.asarray(band[1], dtype=np.float64)
        )
        band_item = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(fill))
        band_item.setZValue(_BAND_Z)
        return band_item

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
        """Whether the scatter is being decimated for drawing (truthful: never here).

        Auto peak downsampling is disabled in this view — it assumes sorted x values
        and would distort an unsorted scatter. :meth:`scatter_data` therefore always
        reports the full point count.
        """
        return False

    # -- crosshair ---------------------------------------------------------------------------

    def _install_crosshair(self) -> None:
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
        """Drop the crosshair lines and clear the readout (mouse left the plot)."""
        self._v_line.hide()
        self._h_line.hide()
        self._readout.setText("")

    def crosshair_visible(self) -> bool:
        return bool(self._v_line.isVisible() and self._h_line.isVisible())

    def scene_position_of(self, x: float, y: float) -> QPointF:
        """The scene position of a data (view) coordinate — the inverse crosshair path."""
        return cast(QPointF, self._plot.plotItem.vb.mapViewToScene(QPointF(x, y)))

    def crosshair_text(self) -> str:
        return self._readout.text()

    # -- export ------------------------------------------------------------------------------

    def export_png(self, target: Path) -> bool:
        """Export the plot as PNG via ``pyqtgraph.exporters.ImageExporter``."""
        if self._scatter is None:
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
        if self._scatter is None:
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
