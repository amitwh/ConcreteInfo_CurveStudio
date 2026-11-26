"""The five statistical plot models, drawn on a pyqtgraph ``PlotItem``.

A tool returns its plots as plain blocks (:mod:`curvestudio.tools.stats._display`): a
scatter with reference lines and a band, a histogram with overlays, a Q-Q plot, a box per
group, or several lines. This module is the desktop's one translator from those blocks to
drawable items, so the desktop and the web show the same plot the same way. It draws on a
plot item it is given and returns every item it added, which is what lets a view take the
plot down again without touching anything it did not draw.

The axes are the block's own labels, the title is the plot item's title and a block's note
is a label pinned under the plot. A box plot is drawn from lines and bars rather than a
composite item: pyqtgraph has no box-plot item whose parts are data coordinates, and the
parts are what the result's blocks already hold (quartiles, median, Tukey's whiskers and
the outliers beyond them).
"""

from typing import TYPE_CHECKING, Any

from curvestudio_desktop.theme import current_scheme

if TYPE_CHECKING:
    import pyqtgraph as pg

    from curvestudio.tools.stats import (
        BoxPlot,
        HistogramPlot,
        MultiLinePlot,
        Plot,
        QQPlot,
        ScatterPlot,
    )
    from curvestudio.tools.stats._display import Line

NOTE_NAME = "statsPlotNote"
"""The object name of the label a plot's note is drawn in."""

_MUTED = "#8C8C8C"
"""The neutral pen of a reference line and of the box-plot furniture."""
_LINE_WIDTH = 1.2
_BOX_WIDTH = 0.6
_CAP_WIDTH = 0.25
_NOTE_Z = 20.0


def draw_stats_plot(plot_item: pg.PlotItem, plot: Plot) -> list[Any]:
    """Draw ``plot`` on ``plot_item`` and return every item that was added.

    The item's title, the axes' labels and the block's note are set from the block. An
    item a previous plot left behind is not remembered: the caller removes what it is
    given, which is why the return value is the plot's whole footprint on the item.
    """
    _reset(plot_item)
    dark = current_scheme() == "dark"
    if plot.kind == "scatter":
        items = _scatter(plot_item, plot, dark=dark)
    elif plot.kind == "histogram":
        items = _histogram(plot_item, plot, dark=dark)
    elif plot.kind == "qq":
        items = _qq(plot_item, plot)
    elif plot.kind == "box":
        items = _box(plot_item, plot, dark=dark)
    elif plot.kind == "lines":
        items = _lines(plot_item, plot)
    else:
        kind = getattr(plot, "kind", None)
        raise TypeError(f"unknown plot kind: {kind!r}: no renderer draws it")
    plot_item.setTitle(plot.title)
    note = _note(plot_item, plot)
    if note is not None:
        items.append(note)
    return items


def _reset(plot_item: pg.PlotItem) -> None:
    """Give the item a clean state: no category ticks and no legend from an earlier plot."""
    plot_item.getAxis("bottom").setTicks(None)
    legend = plot_item.legend
    if legend is not None:
        legend.clear()
        legend.setVisible(False)


def _note(plot_item: pg.PlotItem, plot: Plot) -> pg.LabelItem | None:
    import pyqtgraph as pg

    """The block's note as a faint label pinned under the plot, or nothing when it has none.

    It is parented to the plot item (not added to its item list), so the view takes it down
    by detaching it: the plot item never owns a label it did not draw.
    """
    if not plot.note:
        return None
    label = pg.LabelItem(text=plot.note, parent=plot_item)
    label.setObjectName(NOTE_NAME)
    label.setZValue(_NOTE_Z)
    label.anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(8, -4))
    return label


def _points(x: Any, y: Any, colour: str, *, size: int = 6) -> pg.PlotDataItem:
    import numpy as np
    import pyqtgraph as pg

    """Points in ``colour``, with no joining line."""
    return pg.PlotDataItem(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        pen=None,
        symbol="o",
        symbolSize=size,
        symbolBrush=pg.mkBrush(colour),
        symbolPen=pg.mkPen(colour),
    )


def _reference_line(plot_item: pg.PlotItem, line: Line) -> pg.PlotDataItem:
    import numpy as np
    import pyqtgraph as pg

    """One straight reference line, in the neutral colour every band and rule shares."""
    item = pg.PlotDataItem(
        np.asarray([line.x0, line.x1], dtype=np.float64),
        np.asarray([line.y0, line.y1], dtype=np.float64),
        pen=pg.mkPen(_MUTED, width=_LINE_WIDTH),
    )
    plot_item.addItem(item)
    return item


def _scatter(plot_item: pg.PlotItem, plot: ScatterPlot, *, dark: bool) -> list[Any]:
    import numpy as np
    import pyqtgraph as pg

    from curvestudio_plotting.palette import data_color

    points = _points(plot.x, plot.y, data_color(dark=dark))
    plot_item.addItem(points)
    items: list[Any] = [points]
    for line in plot.lines:
        items.append(_reference_line(plot_item, line))
    if plot.band is not None:
        lower = pg.PlotDataItem(
            np.asarray(plot.band.x, dtype=np.float64),
            np.asarray(plot.band.lower, dtype=np.float64),
        )
        upper = pg.PlotDataItem(
            np.asarray(plot.band.x, dtype=np.float64),
            np.asarray(plot.band.upper, dtype=np.float64),
        )
        from curvestudio_plotting.palette import band_fill

        band = pg.FillBetweenItem(
            lower, upper, brush=pg.mkBrush(band_fill("confidence", dark=dark))
        )
        band.setZValue(0.5)
        plot_item.addItem(band)
        items.append(band)
    plot_item.setLabel("bottom", plot.x_label)
    plot_item.setLabel("left", plot.y_label)
    return items


def _histogram(plot_item: pg.PlotItem, plot: HistogramPlot, *, dark: bool) -> list[Any]:
    import numpy as np
    import pyqtgraph as pg

    edges = np.asarray(plot.edges, dtype=np.float64)
    counts = np.asarray(plot.counts, dtype=np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    from curvestudio_plotting.palette import data_color

    bars = pg.BarGraphItem(
        x=centres,
        height=counts,
        width=np.diff(edges),
        brush=pg.mkBrush(data_color(dark=dark)),
        pen=pg.mkPen(None),
    )
    plot_item.addItem(bars)
    items: list[Any] = [bars]
    for index, overlay in enumerate(plot.overlays):
        from curvestudio_plotting.palette import curve_style

        colour, width = curve_style(index + 1)
        line = pg.PlotDataItem(
            centres,
            np.asarray(overlay.values, dtype=np.float64),
            pen=pg.mkPen(colour, width=width),
        )
        plot_item.addItem(line)
        items.append(line)
    plot_item.setLabel("bottom", plot.x_label)
    plot_item.setLabel("left", plot.y_label)
    return items


def _qq(plot_item: pg.PlotItem, plot: QQPlot) -> list[Any]:

    from curvestudio_plotting.palette import curve_style

    points = _points(plot.theoretical, plot.ordered, curve_style(1)[0])
    plot_item.addItem(points)
    items: list[Any] = [points, _reference_line(plot_item, plot.line)]
    plot_item.setLabel("bottom", "Theoretical quantiles")
    plot_item.setLabel("left", "Ordered values")
    return items


def _box(plot_item: pg.PlotItem, plot: BoxPlot, *, dark: bool) -> list[Any]:
    import pyqtgraph as pg

    """One box per group: a bar from Q1 to Q3, its median, Tukey's whiskers, its outliers."""
    from curvestudio_plotting.palette import curve_style, data_color

    neutral = data_color(dark=dark)

    accent, _width = curve_style(1)
    items: list[Any] = []
    ticks: list[tuple[float, str]] = []
    for index, box in enumerate(plot.boxes):
        x = float(index)
        ticks.append((x, box.label))
        body = pg.BarGraphItem(
            x=[x],
            y0=[box.q1],
            height=[box.q3 - box.q1],
            width=_BOX_WIDTH,
            brush=pg.mkBrush(neutral),
            pen=pg.mkPen(_MUTED),
        )
        median = pg.PlotDataItem(
            [x - _BOX_WIDTH / 2.0, x + _BOX_WIDTH / 2.0],
            [box.median, box.median],
            pen=pg.mkPen(accent, width=2.0),
        )
        whiskers = pg.PlotDataItem(
            x=[
                x,
                x,
                x - _CAP_WIDTH,
                x + _CAP_WIDTH,
                x,
                x,
                x - _CAP_WIDTH,
                x + _CAP_WIDTH,
            ],
            y=[
                box.q1,
                box.lower_whisker,
                box.lower_whisker,
                box.lower_whisker,
                box.q3,
                box.upper_whisker,
                box.upper_whisker,
                box.upper_whisker,
            ],
            connect="pairs",
            pen=pg.mkPen(_MUTED, width=_LINE_WIDTH),
        )
        items.extend((body, median, whiskers))
        for item in (body, median, whiskers):
            plot_item.addItem(item)
        if box.outliers:
            outliers = _points([x] * len(box.outliers), box.outliers, neutral, size=4)
            plot_item.addItem(outliers)
            items.append(outliers)
    plot_item.getAxis("bottom").setTicks([ticks])
    plot_item.setLabel("bottom", "Group")
    plot_item.setLabel("left", plot.y_label)
    return items


def _lines(plot_item: pg.PlotItem, plot: MultiLinePlot) -> list[Any]:
    import numpy as np
    import pyqtgraph as pg

    """One line per series, with a legend entry naming each, and the category tick texts."""
    legend = plot_item.legend if plot_item.legend is not None else plot_item.addLegend()
    legend.setVisible(True)
    items: list[Any] = []
    for index, series in enumerate(plot.series):
        from curvestudio_plotting.palette import curve_style

        colour, width = curve_style(index + 1)
        line = pg.PlotDataItem(
            np.asarray(series.x, dtype=np.float64),
            np.asarray(series.y, dtype=np.float64),
            pen=pg.mkPen(colour, width=width),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(colour),
            symbolPen=pg.mkPen(colour),
        )
        plot_item.addItem(line)
        items.append(line)
        legend.addItem(line, series.label)
    plot_item.setLabel("bottom", plot.x_label)
    plot_item.setLabel("left", plot.y_label)
    ticks = _category_ticks(plot)
    if ticks:
        plot_item.getAxis("bottom").setTicks([ticks])
    return items


def _category_ticks(plot: MultiLinePlot) -> list[tuple[float, str]]:
    """``x_labels`` read at the x values the series use (or at 0, 1, … when there are none)."""
    labels = plot.x_labels
    if labels is None:
        return []
    if plot.series:
        xs = [float(x) for x in plot.series[0].x]
    else:
        xs = [float(index) for index in range(len(labels))]
    return [(x, labels[index]) for index, x in enumerate(xs) if index < len(labels)]


__all__ = ["NOTE_NAME", "draw_stats_plot"]
