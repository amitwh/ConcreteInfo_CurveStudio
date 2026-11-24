/**
 * The plots of a statistical result: every kind becomes the traces Plotly draws, a chooser
 * names several plots, and a kind the renderer cannot draw says so instead of an empty frame.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type {
  BoxPlot,
  HistogramPlot,
  MultiLinePlot,
  QQPlot,
  ScatterPlot,
  StatPlot,
} from "../../api/client";
import { FIXTURE_RESULT } from "./fixture";
import { StatsPlots } from "./StatsPlots";

type Trace = {
  type?: string;
  mode?: string;
  name?: string;
  x?: ReadonlyArray<number | string | null>;
  y?: ReadonlyArray<number | null>;
  fill?: string;
  fillcolor?: string;
  q1?: number;
  median?: number;
  q3?: number;
  lowerfence?: number;
  upperfence?: number;
};

type Layout = {
  title?: { text: string };
  xaxis?: { title?: { text: string }; ticktext?: ReadonlyArray<string> };
  yaxis?: { title?: { text: string } };
};

const captured = vi.hoisted(() => ({
  traces: [] as Trace[],
  layout: {} as Layout,
  ariaLabel: "",
  calls: 0,
}));

// jsdom cannot host a real Plotly graph; capture what the plots builder hands the chart.
vi.mock("../../components/PlotChart", () => ({
  PlotChart: ({
    traces,
    layout,
    ariaLabel,
  }: {
    traces: Trace[];
    layout: Layout;
    ariaLabel: string;
  }) => {
    captured.traces = traces;
    captured.layout = layout;
    captured.ariaLabel = ariaLabel;
    captured.calls += 1;
    return <div data-testid="plot" />;
  },
}));

beforeEach(() => {
  captured.traces = [];
  captured.layout = {};
  captured.ariaLabel = "";
  captured.calls = 0;
});

function plot<T extends StatPlot>(kind: T["kind"]): T {
  const found = FIXTURE_RESULT.plots.find((item) => item.kind === kind);
  expect(found, `the fixture has no ${kind} plot`).toBeDefined();
  return found as T;
}

function renderOne(one: StatPlot): void {
  render(<StatsPlots plots={[one]} index={0} onSelect={() => {}} />);
}

describe("StatsPlots — one plot of each kind", () => {
  it("draws a scatter's points and lines, and its band as a filled polygon", () => {
    const scatter = plot<ScatterPlot>("scatter");
    renderOne(scatter);

    const points = captured.traces.find((trace) => trace.mode === "markers");
    expect(points?.x).toEqual(scatter.x);
    expect(points?.y).toEqual(scatter.y);

    const line = captured.traces.find((trace) => trace.mode === "lines");
    expect(line?.x).toEqual([scatter.lines[0]?.x0, scatter.lines[0]?.x1]);

    const band = captured.traces.find((trace) => trace.fill === "toself");
    expect(band, "the band is a filled polygon").toBeDefined();
    expect(band?.fillcolor).toBeTruthy();
    expect(band?.x).toHaveLength((scatter.band?.x.length ?? 0) * 2);

    expect(captured.layout.title?.text).toBe(scatter.title);
    expect(captured.layout.xaxis?.title?.text).toBe(scatter.x_label);
    expect(captured.layout.yaxis?.title?.text).toBe(scatter.y_label);
    expect(screen.getByText(scatter.note as string)).toBeTruthy();
  });

  it("draws a histogram as one bar trace on the bin edges, and one line per overlay", () => {
    const histogram = plot<HistogramPlot>("histogram");
    renderOne(histogram);

    const bars = captured.traces.filter((trace) => trace.type === "bar");
    expect(bars).toHaveLength(1);
    expect(bars[0]?.y).toEqual(histogram.counts);

    const overlays = captured.traces.filter((trace) => trace.mode === "lines");
    expect(overlays).toHaveLength(histogram.overlays.length);
    expect(overlays[0]?.name).toBe(histogram.overlays[0]?.label);
    expect(overlays[0]?.y).toEqual(histogram.overlays[0]?.values);

    expect(captured.layout.xaxis?.title?.text).toBe(histogram.x_label);
    expect(captured.layout.yaxis?.title?.text).toBe(histogram.y_label);
  });

  it("draws a Q-Q plot as points and the reference line", () => {
    const qq = plot<QQPlot>("qq");
    renderOne(qq);

    const points = captured.traces.find((trace) => trace.mode === "markers");
    expect(points?.x).toEqual(qq.theoretical);
    expect(points?.y).toEqual(qq.ordered);

    const reference = captured.traces.find((trace) => trace.mode === "lines");
    expect(reference?.x).toEqual([qq.line.x0, qq.line.x1]);
    expect(reference?.y).toEqual([qq.line.y0, qq.line.y1]);
  });

  it("draws a box plot as one box per group, with its quartiles and its outliers", () => {
    const box = plot<BoxPlot>("box");
    renderOne(box);

    const boxes = captured.traces.filter((trace) => trace.type === "box");
    expect(boxes).toHaveLength(box.boxes.length);
    boxes.forEach((trace, index) => {
      const group = box.boxes[index];
      expect(trace.name).toBe(group?.label);
      expect(trace.q1).toBe(group?.q1);
      expect(trace.median).toBe(group?.median);
      expect(trace.q3).toBe(group?.q3);
      expect(trace.lowerfence).toBe(group?.lower_whisker);
      expect(trace.upperfence).toBe(group?.upper_whisker);
    });

    const outliers = captured.traces.filter((trace) => trace.mode === "markers");
    expect(outliers).toHaveLength(1);
    expect(outliers[0]?.y).toEqual([9.5]);

    expect(captured.layout.yaxis?.title?.text).toBe(box.y_label);
  });

  it("draws several lines, one per series, with x_labels as the tick texts", () => {
    const lines = plot<MultiLinePlot>("lines");
    renderOne(lines);

    const drawn = captured.traces.filter((trace) => trace.mode === "lines");
    expect(drawn).toHaveLength(lines.series.length);
    expect(drawn.map((trace) => trace.name)).toEqual(lines.series.map((series) => series.label));
    expect(captured.layout.xaxis?.ticktext).toEqual(lines.x_labels);
    expect(captured.layout.xaxis?.title?.text).toBe(lines.x_label);
  });
});

describe("StatsPlots — several plots", () => {
  it("names them in a chooser above the figure", () => {
    const onSelect = vi.fn();
    const several = FIXTURE_RESULT.plots.slice(0, 3);
    render(<StatsPlots plots={several} index={0} onSelect={onSelect} />);

    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual(several.map((item) => item.title));
    expect(tabs[0]?.getAttribute("aria-selected")).toBe("true");

    fireEvent.click(tabs[2] as HTMLElement);
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it("shows a single plot without a chooser", () => {
    renderOne(plot<ScatterPlot>("scatter"));
    expect(screen.queryByRole("tablist")).toBeNull();
  });
});

describe("StatsPlots — a kind it cannot draw", () => {
  it("says so in one line instead of an empty frame", () => {
    const unknown = {
      kind: "heatmap",
      title: "A plot of a kind not drawn here",
      note: null,
    } as unknown as StatPlot;
    render(<StatsPlots plots={[unknown]} index={0} onSelect={() => {}} />);

    expect(screen.getByText(/cannot be drawn/)).toBeTruthy();
    expect(screen.queryByTestId("plot")).toBeNull();
  });
});
