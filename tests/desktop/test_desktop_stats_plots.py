"""The five plot models drawn on a pyqtgraph PlotItem, and the PlotView that shows them.

Every plot model is the core's own: the tests give ``draw_stats_plot`` one of each kind and
read back the items it drew (a histogram's bar graph and one line per overlay, a box plot's
one box per group, a line plot's legend entry per series). ``PlotView.show_stats_plot``
hides the dataset and the fitted curve — never removes them — and ``clear_stats_plot``
brings them back with the curve intact.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import pytest

from curvestudio.tools.stats import (
    BoxPlot,
    HistogramPlot,
    MultiLinePlot,
    QQPlot,
    ScatterPlot,
)
from curvestudio.tools.stats._display import Band, BoxGroup, Line, Series, XYSeries
from curvestudio_desktop.views.plot_view import PlotView
from curvestudio_desktop.views.stats_plots import NOTE_NAME, draw_stats_plot


@pytest.fixture()
def plot_widget(qtbot) -> pg.PlotWidget:
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.show()
    return widget


@pytest.fixture()
def plot_item(plot_widget: pg.PlotWidget) -> pg.PlotItem:
    # the fixture depends on the widget so the plot item's C++ tree outlives the fixture
    return plot_widget.getPlotItem()


def _plot_data(items: list[Any]) -> list[pg.PlotDataItem]:
    return [item for item in items if isinstance(item, pg.PlotDataItem)]


def _bars(items: list[Any]) -> list[pg.BarGraphItem]:
    return [item for item in items if isinstance(item, pg.BarGraphItem)]


def test_a_scatter_draws_its_points_its_lines_and_its_band(plot_item: pg.PlotItem) -> None:
    plot = ScatterPlot(
        title="Residuals against fitted",
        note="445 points shown",
        x_label="Fitted",
        y_label="Residual",
        x=(1.0, 2.0, 3.0),
        y=(0.1, -0.2, 0.3),
        lines=(Line(x0=0.0, y0=0.0, x1=4.0, y1=0.0, label="zero"),),
        band=Band(x=(1.0, 2.0, 3.0), lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0)),
    )
    items = draw_stats_plot(plot_item, plot)

    assert plot_item.titleLabel.text == plot.title
    assert plot_item.getAxis("bottom").labelText == "Fitted"
    assert plot_item.getAxis("left").labelText == "Residual"
    assert len(_plot_data(items)) == 2  # the points and the reference line
    assert any(isinstance(item, pg.FillBetweenItem) for item in items)
    notes = [item for item in items if getattr(item, "objectName", lambda: "")() == NOTE_NAME]
    assert len(notes) == 1
    assert notes[0].text == "445 points shown"


def test_a_histogram_draws_one_bar_graph_and_one_line_per_overlay(
    plot_item: pg.PlotItem,
) -> None:
    plot = HistogramPlot(
        title="Distribution of strength",
        note="bins of 5 MPa",
        x_label="Strength",
        edges=(0.0, 5.0, 10.0, 15.0),
        counts=(2.0, 5.0, 3.0),
        overlays=(Series(label="normal", values=(0.1, 0.2, 0.1)),),
    )
    items = draw_stats_plot(plot_item, plot)

    assert plot_item.titleLabel.text == plot.title
    assert plot_item.getAxis("bottom").labelText == "Strength"
    assert plot_item.getAxis("left").labelText == "Count"
    assert len(_bars(items)) == 1
    assert len(_plot_data(items)) == len(plot.overlays)


def test_a_q_q_plot_draws_its_points_and_the_reference_line(plot_item: pg.PlotItem) -> None:
    plot = QQPlot(
        title="Normal Q-Q",
        theoretical=(-1.0, 0.0, 1.0),
        ordered=(-1.2, 0.1, 0.9),
        line=Line(x0=-1.0, y0=-1.2, x1=1.0, y1=0.9),
    )
    items = draw_stats_plot(plot_item, plot)

    assert plot_item.titleLabel.text == plot.title
    assert len(_plot_data(items)) == 2


def test_a_box_plot_draws_one_box_per_group_with_medians_whiskers_and_outliers(
    plot_item: pg.PlotItem,
) -> None:
    plot = BoxPlot(
        title="Strength by plant",
        note="one point beyond each fence",
        y_label="Strength",
        boxes=(
            BoxGroup(
                label="Plant A",
                q1=20.0,
                median=25.0,
                q3=30.0,
                lower_whisker=18.0,
                upper_whisker=32.0,
                outliers=(10.0,),
            ),
            BoxGroup(
                label="Plant B",
                q1=22.0,
                median=27.0,
                q3=33.0,
                lower_whisker=21.0,
                upper_whisker=35.0,
                outliers=(),
            ),
        ),
    )
    items = draw_stats_plot(plot_item, plot)

    assert plot_item.titleLabel.text == plot.title
    assert plot_item.getAxis("left").labelText == "Strength"
    assert len(_bars(items)) == len(plot.boxes)  # one box per group
    # every group has a median line and a whisker item; only the first has an outlier item
    data_items = _plot_data(items)
    assert len(data_items) == len(plot.boxes) * 2 + 1
    assert sum(1 for item in data_items if item.opts.get("symbol") == "o") == 1
    assert plot_item.getAxis("bottom")._tickLevels == [[(0.0, "Plant A"), (1.0, "Plant B")]]


def test_several_lines_draw_a_line_and_a_legend_entry_per_series(
    plot_item: pg.PlotItem,
) -> None:
    plot = MultiLinePlot(
        title="Growth over time",
        x_label="Curing day",
        y_label="Strength",
        series=(
            XYSeries(label="Mix 1", x=(1.0, 2.0, 3.0), y=(5.0, 8.0, 11.0)),
            XYSeries(label="Mix 2", x=(1.0, 2.0, 3.0), y=(4.0, 7.0, 9.0)),
        ),
        x_labels=("one", "two", "three"),
    )
    items = draw_stats_plot(plot_item, plot)

    assert plot_item.titleLabel.text == plot.title
    assert len(_plot_data(items)) == len(plot.series)
    legend = plot_item.legend
    assert legend is not None
    assert len(legend.items) == len(plot.series)
    assert plot_item.getAxis("bottom")._tickLevels == [[(1.0, "one"), (2.0, "two"), (3.0, "three")]]


def test_an_unknown_kind_is_refused(plot_item: pg.PlotItem) -> None:
    class Unknown:
        kind = "radar"
        title = "Not a plot the tool layer makes"

    with pytest.raises(TypeError, match="radar"):
        draw_stats_plot(plot_item, Unknown())  # type: ignore[arg-type]


# -- the view -------------------------------------------------------------------------------


@pytest.fixture()
def view(qtbot) -> PlotView:
    plot = PlotView()
    qtbot.addWidget(plot)
    plot.show()
    return plot


def _points(plot: PlotView) -> Any:
    return plot._scatter


def _curve_item(plot: PlotView) -> Any:
    return plot._curves[0]


def test_show_stats_plot_hides_the_data_and_the_curve_and_clear_brings_them_back(
    view: PlotView,
) -> None:
    view.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    view.set_fit([1.0, 2.0, 3.0], [2.0, 4.0, 8.0])
    plot = ScatterPlot(
        title="A result",
        x_label="x",
        y_label="y",
        x=(1.0, 2.0, 3.0),
        y=(2.0, 4.0, 8.0),
    )

    view.show_stats_plot(plot)

    assert view.has_stats_plot()
    assert not _points(view).isVisible()
    assert not _curve_item(view).isVisible()
    assert not view.log_x_enabled and not view.log_y_enabled

    view.clear_stats_plot()

    assert not view.has_stats_plot()
    assert _points(view).isVisible()
    assert _curve_item(view).isVisible()
    curve = view.fit_curve_data()
    assert curve is not None
    assert list(curve[1]) == [2.0, 4.0, 8.0]


def test_show_stats_plot_turns_a_log_axis_off_and_clear_puts_it_back(view: PlotView) -> None:
    view.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    view.set_log_y(True)
    plot = ScatterPlot(title="A result", x_label="x", y_label="y", x=(1.0,), y=(2.0,))

    view.show_stats_plot(plot)
    assert not view.log_y_enabled

    view.clear_stats_plot()
    assert view.log_y_enabled


def test_the_stats_plot_is_exported_and_the_dataset_comes_back_after(
    view: PlotView, tmp_path: Path
) -> None:
    view.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]), "x", "y")
    plot = HistogramPlot(
        title="A result",
        x_label="x",
        edges=(0.0, 1.0, 2.0),
        counts=(1.0, 2.0),
    )
    view.show_stats_plot(plot)

    png = tmp_path / "stats.png"
    svg = tmp_path / "stats.svg"
    assert view.export_png(png) is True
    assert view.export_svg(svg) is True
    assert png.stat().st_size > 0
    assert svg.stat().st_size > 0

    view.clear_stats_plot()
    x, y = view.scatter_data()
    assert x is not None and y is not None
    assert list(x) == [1.0, 2.0, 3.0]


def test_a_stats_plot_exports_even_with_no_dataset_behind_it(
    view: PlotView, tmp_path: Path
) -> None:
    view.show_stats_plot(
        ScatterPlot(title="A result", x_label="x", y_label="y", x=(1.0, 2.0), y=(2.0, 1.0))
    )
    assert view.export_png(tmp_path / "only.png") is True
    assert view.export_svg(tmp_path / "only.svg") is True
