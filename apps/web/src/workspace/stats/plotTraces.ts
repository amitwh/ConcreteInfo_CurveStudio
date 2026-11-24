/**
 * Pure builders from a statistical result's plot block to the Plotly traces and layout a
 * `PlotChart` draws. A kind this view has no builder for returns `null`: the caller says so
 * in one line rather than mounting an empty frame.
 */

import type { PlotlyLayout, PlotlyTrace } from "plotly.js-dist-min";
import type {
  BoxPlot,
  HistogramPlot,
  MultiLinePlot,
  QQPlot,
  ScatterPlot,
  StatPlot,
} from "../../api/client";
import { ACCENT, BAND_CONFIDENCE_FILL, DATA_COLOR, MODEL_COLORS } from "../traces";

export interface PlotDrawing {
  traces: PlotlyTrace[];
  layout: Partial<PlotlyLayout>;
}

const POINT_COLOR = DATA_COLOR.light;
const BAND_FILL = BAND_CONFIDENCE_FILL.light;

/** A categorical colour, wrapping past the palette's eight entries. */
function seriesColor(position: number): string {
  return MODEL_COLORS[position % MODEL_COLORS.length] as string;
}

function scatterDrawing(plot: ScatterPlot): PlotDrawing {
  const traces: PlotlyTrace[] = [
    {
      type: "scatter",
      mode: "markers",
      name: "Points",
      x: plot.x,
      y: plot.y,
      marker: { color: POINT_COLOR, size: 6, opacity: 0.75 },
    },
  ];
  plot.lines.forEach((line, position) => {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: line.label ?? `Line ${position + 1}`,
      x: [line.x0, line.x1],
      y: [line.y0, line.y1],
      line: { color: ACCENT, width: 1.5, dash: "dash" },
    });
  });
  if (plot.band != null) {
    const { x, lower, upper } = plot.band;
    // A band is one closed polygon: the upper edge left to right, the lower edge back.
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Interval",
      x: [...x, ...[...x].reverse()],
      y: [...upper, ...[...lower].reverse()],
      fill: "toself",
      fillcolor: BAND_FILL,
      line: { color: "transparent", width: 0 },
      hoverinfo: "skip",
      showlegend: false,
    });
  }
  return {
    traces,
    layout: {
      title: { text: plot.title },
      xaxis: { title: { text: plot.x_label } },
      yaxis: { title: { text: plot.y_label } },
      showlegend: true,
    },
  };
}

/** The centre of each bin, from the edges that bound it. */
function binCentres(edges: number[], count: number): number[] {
  return Array.from(
    { length: count },
    (_, index) => ((edges[index] ?? 0) + (edges[index + 1] ?? 0)) / 2,
  );
}

/** The width of each bin, from its two edges. */
function binWidths(edges: number[], count: number): number[] {
  return Array.from(
    { length: count },
    (_, index) => (edges[index + 1] ?? 0) - (edges[index] ?? 0),
  );
}

function histogramDrawing(plot: HistogramPlot): PlotDrawing {
  const centres = binCentres(plot.edges, plot.counts.length);
  const traces: PlotlyTrace[] = [
    {
      type: "bar",
      name: "Count",
      x: centres,
      y: plot.counts,
      width: binWidths(plot.edges, plot.counts.length),
      marker: { color: POINT_COLOR, opacity: 0.55 },
    },
  ];
  plot.overlays.forEach((series, position) => {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: series.label,
      x: centres,
      y: series.values,
      line: { color: seriesColor(position), width: 2 },
    });
  });
  return {
    traces,
    layout: {
      title: { text: plot.title },
      xaxis: { title: { text: plot.x_label } },
      yaxis: { title: { text: plot.y_label } },
      showlegend: true,
    },
  };
}

function qqDrawing(plot: QQPlot): PlotDrawing {
  return {
    traces: [
      {
        type: "scatter",
        mode: "markers",
        name: "Values",
        x: plot.theoretical,
        y: plot.ordered,
        marker: { color: POINT_COLOR, size: 6, opacity: 0.75 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: plot.line.label ?? "Reference line",
        x: [plot.line.x0, plot.line.x1],
        y: [plot.line.y0, plot.line.y1],
        line: { color: ACCENT, width: 1.5, dash: "dash" },
      },
    ],
    layout: {
      title: { text: plot.title },
      xaxis: { title: { text: "Theoretical quantile" } },
      yaxis: { title: { text: "Ordered value" } },
      showlegend: true,
    },
  };
}

function boxDrawing(plot: BoxPlot): PlotDrawing {
  const traces: PlotlyTrace[] = [];
  plot.boxes.forEach((group, position) => {
    // The box is drawn from the server's quartiles and Tukey whiskers, not refitted here.
    traces.push({
      type: "box",
      name: group.label,
      x: [group.label],
      q1: group.q1,
      median: group.median,
      q3: group.q3,
      lowerfence: group.lower_whisker,
      upperfence: group.upper_whisker,
      boxpoints: false,
      marker: { color: seriesColor(position) },
    });
    if (group.outliers.length > 0) {
      traces.push({
        type: "scatter",
        mode: "markers",
        name: `${group.label} outliers`,
        x: group.outliers.map(() => group.label),
        y: group.outliers,
        marker: { color: POINT_COLOR, size: 7, line: { width: 1 } },
        showlegend: false,
      });
    }
  });
  return {
    traces,
    layout: {
      title: { text: plot.title },
      yaxis: { title: { text: plot.y_label } },
      showlegend: false,
    },
  };
}

function linesDrawing(plot: MultiLinePlot): PlotDrawing {
  const traces: PlotlyTrace[] = plot.series.map((series, position) => ({
    type: "scatter",
    mode: "lines",
    name: series.label,
    x: series.x,
    y: series.y,
    line: { color: seriesColor(position), width: 2 },
  }));
  const xaxis: NonNullable<PlotlyLayout["xaxis"]> = {
    title: { text: plot.x_label },
  };
  if (plot.x_labels != null) {
    // The x values are positions; the labels are what the reader is meant to see.
    xaxis.tickvals = plot.series[0]?.x ?? plot.x_labels.map((_, position) => position);
    xaxis.ticktext = plot.x_labels;
  }
  return {
    traces,
    layout: {
      title: { text: plot.title },
      xaxis,
      yaxis: { title: { text: plot.y_label } },
      showlegend: true,
    },
  };
}

/**
 * The traces and layout for one plot block, or `null` for a kind this view cannot draw.
 */
export function plotTraces(plot: StatPlot): PlotDrawing | null {
  switch (plot.kind) {
    case "scatter":
      return scatterDrawing(plot);
    case "histogram":
      return histogramDrawing(plot);
    case "qq":
      return qqDrawing(plot);
    case "box":
      return boxDrawing(plot);
    case "lines":
      return linesDrawing(plot);
    default:
      return null;
  }
}
