"""PlotView: scatter, log-axis rules, crosshair, reset and export — offscreen."""

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from curvestudio_desktop.views.plot_view import PlotView


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
    """Peak downsampling assumes sorted x — an unsorted scatter must not use it."""
    x = np.arange(60_000, dtype=float)
    shuffled = x.copy()
    shuffled[::2], shuffled[1::2] = x[1::2], x[::2]  # unsorted but same values
    plot.set_data(shuffled, shuffled * 2.0, "x", "y")
    assert plot.is_downsampling() is False
    shown_x, shown_y = plot.scatter_data()
    assert shown_x is not None and shown_y is not None
    assert len(shown_x) == 60_000 and len(shown_y) == 60_000


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
