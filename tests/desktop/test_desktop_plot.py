"""PlotView: scatter, log-axis rules, crosshair, reset and export — offscreen."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QLabel

from curvestudio_desktop.views.plot_view import OverlayCurve, PlotView


@pytest.fixture()
def plot(qtbot) -> PlotView:
    view = PlotView()
    qtbot.addWidget(view)
    return view


def test_empty_plot_is_ready(plot: PlotView) -> None:
    assert plot.scatter_data() == (None, None)
    assert plot.can_log_x is False and plot.can_log_y is False
    assert plot.axis_labels() == ("x", "y")


def test_set_data_scatter_and_labels(plot: PlotView) -> None:
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "age", "load")
    x, y = plot.scatter_data()
    assert x is not None and y is not None
    assert list(x) == [1.0, 2.0, 3.0]
    assert plot.axis_labels() == ("age", "load")
    assert plot.can_log_x is True and plot.can_log_y is True


def test_nonpositive_data_blocks_log_axes(plot: PlotView) -> None:
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([-4.0, 0.0, 8.0]), "x", "y")
    assert plot.can_log_x is True
    assert plot.can_log_y is False
    plot.set_data(np.array([-1.0, 2.0, 3.0]), np.array([1.0, 4.0, 8.0]), "x", "y")
    assert plot.can_log_x is False
    assert plot.can_log_y is True


def test_log_toggles_apply_and_reset(plot: PlotView) -> None:
    plot.set_data(np.array([1.0, 10.0, 100.0]), np.array([1.0, 10.0, 100.0]), "x", "y")
    plot.set_log_x(True)
    plot.set_log_y(True)
    assert plot.log_x_enabled is True and plot.log_y_enabled is True
    plot.set_log_x(False)
    assert plot.log_x_enabled is False


def test_crosshair_readout_formats_coordinates(plot: PlotView) -> None:

    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "age", "load")
    plot.resize(600, 400)
    plot.show_crosshair_at(plot.scene_position_of(1.5, 3.0))
    text = plot.crosshair_text()
    assert "1.5" in text and "3" in text


def test_log_axis_crosshair_readout_shows_data_values(plot: PlotView) -> None:
    """On log axes the crosshair must read out data values (10**view), not log10."""
    plot.set_data(np.array([1.0, 10.0, 100.0]), np.array([1.0, 10.0, 100.0]), "x", "y")
    plot.resize(600, 400)
    plot.set_log_x(True)
    plot.set_log_y(True)
    # view space is log10 of the data: the data point (10, 100) sits at view (1, 2)
    plot.show_crosshair_at(plot.scene_position_of(1.0, 2.0))
    text = plot.crosshair_text()
    assert "= 10" in text, text
    assert "= 100" in text, text


def test_crosshair_hides_when_mouse_leaves_the_plot(plot: PlotView) -> None:
    from PySide6.QtCore import QPointF

    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    plot.resize(600, 400)
    plot.show_crosshair_at(plot.scene_position_of(1.5, 3.0))
    assert plot.crosshair_visible() is True
    plot._on_mouse_moved([QPointF(-500.0, -500.0)])  # a move outside the plot scene
    assert plot.crosshair_visible() is False
    assert plot.crosshair_text() == ""


def test_reset_view_reranges(plot: PlotView) -> None:
    plot.set_data(np.array([0.0, 100.0]), np.array([0.0, 50.0]), "x", "y")
    plot.reset_view()
    ranges = plot.view_ranges()
    assert ranges is not None


def test_export_png_and_svg(plot: PlotView, tmp_path) -> None:
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    png = tmp_path / "plot.png"
    svg = tmp_path / "plot.svg"
    plot.export_png(png)
    plot.export_svg(svg)
    assert png.is_file() and png.stat().st_size > 0
    assert svg.is_file() and svg.stat().st_size > 0


def test_export_refused_without_data(plot: PlotView, tmp_path) -> None:
    assert plot.export_png(tmp_path / "empty.png") is False
    assert not (tmp_path / "empty.png").exists()


def test_watermark_is_a_faint_scene_brand_mark(plot: PlotView, tmp_path) -> None:
    """The mark lives in the scene (so both exports carry it) and never grabs the mouse."""
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    plot.resize(640, 480)
    assert plot.watermark_text() == "ConcreteInfo CurveStudio"
    mark = plot._watermark
    assert mark.isVisible()
    assert mark.parentItem() is plot._plot.graphicsItem()
    assert mark.acceptedMouseButtons() == Qt.MouseButton.NoButton
    assert mark.zValue() > 2.0, "the mark sits above the traces"
    # the SVG serialises the scene, so the mark is in the exported file
    svg = tmp_path / "watermarked.svg"
    assert plot.export_svg(svg) is True
    assert "ConcreteInfo CurveStudio" in svg.read_text(encoding="utf-8")
    # it belongs to the widget, not the data: clearing the dataset keeps it visible
    plot.clear_data()
    assert mark.isVisible()
    assert plot.watermark_text() == "ConcreteInfo CurveStudio"


def test_export_png_hides_the_crosshair(plot: PlotView, tmp_path, monkeypatch) -> None:
    import pyqtgraph.exporters as exporters_module

    seen: dict[str, bool] = {}

    class RecordingExporter:
        def __init__(self, item: object) -> None:
            self.item = item

        def export(self, path: str) -> None:
            seen["v"] = plot.crosshair_visible()

    monkeypatch.setattr(exporters_module, "ImageExporter", RecordingExporter)
    plot.set_data(np.array([1.0, 2.0]), np.array([2.0, 4.0]), "x", "y")
    plot.resize(600, 400)
    plot.show_crosshair_at(plot.scene_position_of(1.5, 3.0))
    assert plot.crosshair_visible() is True
    assert plot.export_png(tmp_path / "crosshair-free.png") is True
    assert seen == {"v": False}, "the crosshair lines must be hidden during export"
    assert plot.crosshair_visible() is True  # restored afterwards


def test_large_scatter_is_not_peak_downsampled(plot: PlotView) -> None:
    """Peak downsampling presumes uniform x spacing — an unsorted scatter must not use it.

    Updated for D3: the drawn scatter is still never pyqtgraph-downsampled, but above
    SCATTER_DISPLAY_MAX_POINTS it is drawn as an unsorted-safe min/max envelope, the
    view says so via is_downsampling(), and scatter_data() keeps the full 60k arrays.
    """
    from curvestudio_desktop.views.plot_view import SCATTER_DISPLAY_MAX_POINTS

    x = np.arange(60_000, dtype=float)
    shuffled = x.copy()
    shuffled[::2], shuffled[1::2] = x[1::2], x[::2]  # unsorted but same values
    plot.set_data(shuffled, shuffled * 2.0, "x", "y")
    assert plot._scatter.opts["autoDownsample"] is False  # no pyqtgraph peak mode
    assert plot.is_downsampling() is True  # the drawn scatter is the min/max envelope
    drawn_x, _drawn_y = plot._scatter.getData()
    assert len(drawn_x) <= SCATTER_DISPLAY_MAX_POINTS + 2
    shown_x, shown_y = plot.scatter_data()
    assert shown_x is not None and shown_y is not None
    assert len(shown_x) == 60_000 and len(shown_y) == 60_000


def test_large_dataset_set_data_decimates_the_drawn_scatter(plot: PlotView) -> None:
    """A 100k dataset updates and the drawn scatter is the min/max envelope, honestly labelled."""
    from curvestudio_desktop.views.plot_view import SCATTER_DISPLAY_MAX_POINTS

    rng = np.random.default_rng(20261126)
    x = rng.uniform(0.0, 500.0, 100_000)
    y = rng.normal(20.0, 5.0, 100_000)
    y[12345] = 500.0  # a spike inside one bucket: decimation must not erase it
    plot.set_data(x, y, "age", "load")
    assert plot.is_downsampling() is True
    drawn_x, drawn_y = plot._scatter.getData()
    assert drawn_x.size <= SCATTER_DISPLAY_MAX_POINTS + 2
    # the envelope keeps the drawn x-span and the y extremes exactly
    assert drawn_x.min() == x.min() and drawn_x.max() == x.max()
    assert drawn_y.min() == y.min() and drawn_y.max() == y.max()
    full_x, full_y = plot.scatter_data()
    assert full_x is not None and full_y is not None
    assert len(full_x) == 100_000 and len(full_y) == 100_000
    # the line under the plot says what is drawn
    text = plot.crosshair_text()
    assert text.startswith("showing ") and text.endswith(f"of {100_000:,} points"), text
    # a dataset switch reuses the drawn item instead of removing and re-adding it
    scatter = plot._scatter
    plot.set_data(x * 2.0, y, "age", "load")
    assert plot._scatter is scatter
    assert plot.is_downsampling() is True


def test_the_decimation_note_yields_to_and_returns_from_the_crosshair(plot: PlotView) -> None:
    """The "showing X of N points" line makes way for the crosshair and comes back."""
    rng = np.random.default_rng(20261126)
    plot.set_data(rng.uniform(0.0, 500.0, 100_000), rng.normal(20.0, 5.0, 100_000), "x", "y")
    plot.resize(600, 400)
    note = plot.crosshair_text()
    assert note.startswith("showing ")
    plot.show_crosshair_at(plot.scene_position_of(250.0, 20.0))
    assert "x =" in plot.crosshair_text() and "showing" not in plot.crosshair_text()
    plot.hide_crosshair()
    assert plot.crosshair_text() == note


def test_set_fit_reuses_the_drawn_items(plot: PlotView) -> None:
    """Repeated set_fit calls with the same shape update in place — no item churn."""
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    plot.set_fit(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 8.0],
        ([1.0, 2.0, 3.0], [1.0, 3.0, 7.0]),
        ([1.0, 2.0, 3.0], [0.0, 2.0, 6.0]),
        position=2,
        name="power law",
    )
    curve = plot._curves[0]
    confidence, prediction = plot._confidence, plot._prediction
    assert confidence is not None and prediction is not None
    plot_item = plot._plot.getPlotItem()
    for _ in range(5):
        plot.set_fit(
            [1.0, 2.0, 3.0],
            [3.0, 5.0, 9.0],
            ([1.0, 2.0, 3.0], [2.0, 4.0, 8.0]),
            ([1.0, 2.0, 3.0], [1.0, 3.0, 7.0]),
            position=3,
            name="quadratic",
        )
    assert plot._curves == [curve]  # the item object is reused, not rebuilt
    assert plot._confidence is confidence and plot._prediction is prediction
    # the plot holds exactly two data items: the scatter and the one fit curve
    assert len(plot_item.curves) == 2
    drawn = plot.fit_curve_data()
    assert drawn is not None
    assert list(drawn[1]) == [3.0, 5.0, 9.0]  # the new arrays are what is drawn
    assert plot.drawn_curve_positions() == [3]
    assert plot.legend_names() == []
    # the bands updated in place too
    assert plot.band_colors() == [confidence.brush().color().name(), prediction.brush().color().name()]
    # dropping the bands removes their fills without orphaning anything
    plot.set_fit([1.0, 2.0, 3.0], [3.0, 5.0, 9.0])
    assert plot._confidence is None and plot._prediction is None
    assert plot.band_colors() == []
    assert plot._curves == [curve]


def test_set_fit_overlay_reuses_curves_while_the_shape_matches(plot: PlotView) -> None:
    """An overlay of the same size reuses its items; a size change rebuilds them."""
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    three = [
        OverlayCurve("linear", [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 1),
        OverlayCurve("power", [1.0, 2.0, 3.0], [2.0, 5.0, 9.0], 2),
        OverlayCurve("quadratic", [1.0, 2.0, 3.0], [2.0, 6.0, 12.0], 3),
    ]
    plot.set_fit_overlay(three)
    items = list(plot._curves)
    assert plot.legend_names() == ["linear", "power", "quadratic"]
    plot.set_fit_overlay(
        [
            OverlayCurve("poly", [1.0, 2.0, 3.0], [2.0, 4.0, 7.0], 2),
            OverlayCurve("exp", [1.0, 2.0, 3.0], [2.0, 4.5, 10.0], 4),
            OverlayCurve("log", [1.0, 2.0, 3.0], [2.0, 3.5, 4.0], 1),
        ]
    )
    assert plot._curves == items  # same three items, updated in place
    assert plot.drawn_curve_positions() == [2, 4, 1]
    assert plot.legend_names() == ["poly", "exp", "log"]
    # a genuine shape change rebuilds: three curves then one
    plot.set_fit([1.0, 2.0, 3.0], [2.0, 4.0, 8.0])
    assert len(plot._curves) == 1 and plot._curves[0] not in items
    assert plot.legend_names() == []
    plot.set_fit_overlay(three)
    assert len(plot._curves) == 3 and all(item not in items for item in plot._curves)
    assert plot.legend_names() == ["linear", "power", "quadratic"]


def test_a_redraw_drops_the_previous_row_hint_and_equation(plot: PlotView) -> None:
    """Reusing the curve items must not carry over the previous selection's notices."""
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    plot.set_fit([1.0, 2.0, 3.0], [2.0, 4.0, 8.0])
    plot.set_equation("Power", "y = 1.2*x**0.5")
    plot.show_hint("curve not sampled — Open full results")
    assert plot.hint_visible() and plot.equation_visible()
    plot.set_fit([1.0, 2.0, 3.0], [3.0, 5.0, 9.0])  # same shape: items reused
    assert not plot.hint_visible()
    assert not plot.equation_visible()


def test_small_datasets_render_as_before(plot: PlotView) -> None:
    """A few-thousand-point dataset keeps the exact item structure and visual config."""
    from curvestudio_plotting.palette import band_fill, curve_style, data_color

    x = np.arange(5_000, dtype=float)
    y = (x * 0.5) + 1.0
    plot.set_data(x, y, "age", "load")
    scatter = plot._scatter
    assert scatter.opts["symbol"] == "o"
    assert scatter.opts["symbolSize"] == 5
    assert scatter.opts["autoDownsample"] is False  # pyqtgraph peak stays off for the scatter
    neutral = data_color(dark=False)
    assert scatter.opts["symbolBrush"].color().name() == QColor(neutral).name()
    shown_x, shown_y = scatter.getData()
    assert shown_x.size == 5_000 and shown_y.size == 5_000  # every sample drawn
    assert plot.is_downsampling() is False
    assert plot.crosshair_text() == ""  # no decimation note under the budget
    plot.set_fit(
        [0.0, 5_000.0],
        [1.0, 2_501.0],
        ([0.0, 5_000.0], [-1.0, 1_499.0]),
        ([0.0, 5_000.0], [-2.0, 1_500.0]),
        position=1,
        name="fitted curve",
    )
    curve = plot._curves[0]
    colour, width = curve_style(1)
    assert curve.opts["pen"].color().name() == QColor(colour).name()
    assert curve.opts["pen"].widthF() == width
    assert curve.zValue() == 2.0
    assert curve.opts["autoDownsample"] is True  # curves may auto-downsample (never trips here)
    assert curve.getData()[0].size == 2  # and the small curve passes through untouched
    confidence, prediction = plot._confidence, plot._prediction
    assert confidence is not None and prediction is not None
    assert confidence.zValue() == 0.5
    assert plot.band_colors() == [
        band_fill("confidence", dark=False)[:7].lower(),
        band_fill("prediction", dark=False)[:7].lower(),
    ]
    assert plot.legend_names() == []
    assert plot.fit_curve_data() is not None


def test_the_selected_equation_is_captioned_on_the_plot(plot: PlotView, qtbot) -> None:
    """The curve's equation is shown where the curve is, never with the engine's ``**``."""
    plot.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    plot.set_fit([1.0, 2.0, 3.0], [2.0, 4.0, 8.0])
    plot.set_equation("Quadratic", "y = 0.5 + 2.0*x + 0.09*x**2", position=1)
    qtbot.wait(10)
    assert plot.equation_visible()
    text = plot.equation_text()
    assert text is not None
    assert text.startswith("Quadratic\n")
    assert "^2" in text and "**" not in text
    # the caption never leaves the plot's rect, even when the plot is narrow
    for width in (420, 900):
        plot.resize(width, 500)
        QApplication.processEvents()
        label = plot.findChild(QLabel, "plotEquation")
        assert label is not None
        assert label.geometry().right() <= plot.width()
        assert label.geometry().bottom() <= plot.height()


def test_clearing_the_fit_clears_the_equation(plot: PlotView) -> None:
    plot.set_equation("Power", "y = 1.2*x**0.5")
    assert plot.equation_visible()
    plot.clear_fit()
    assert not plot.equation_visible()
    assert plot.equation_text() is None
