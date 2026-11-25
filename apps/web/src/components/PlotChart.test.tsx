import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const plotly = vi.hoisted(() => ({
  react: vi.fn((..._args: unknown[]) => Promise.resolve()),
  purge: vi.fn(),
  resize: vi.fn(),
}));

vi.mock("../lib/plotly-bundle", () => ({
  default: {
    react: plotly.react,
    purge: plotly.purge,
    Plots: { resize: plotly.resize },
    downloadImage: vi.fn(),
  },
}));

import { PlotChart } from "./PlotChart";

/** Captures each observer so a test can report a container width. */
const observers: ResizeObserverCallback[] = [];

beforeEach(() => {
  observers.length = 0;
  // The suite's fake ResizeObserver is installed non-configurably, so assign over it
  // rather than defining the property again.
  window.ResizeObserver = class {
    constructor(callback: ResizeObserverCallback) {
      observers.push(callback);
    }
    observe(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
  plotly.react.mockClear();
});

function reportWidth(width: number): void {
  act(() => {
    for (const callback of observers) {
      callback([{ contentRect: { width } } as ResizeObserverEntry], {} as ResizeObserver);
    }
  });
}

/** The config object handed to Plotly.react as its fourth argument. */
function lastConfig(): { displayModeBar?: boolean | "hover" | "static" } {
  const call = plotly.react.mock.calls.at(-1);
  return (call?.[3] ?? {}) as { displayModeBar?: boolean | "hover" | "static" };
}

type Annotation = { text: string; font?: { color?: string } };

/** The layout handed to Plotly.react as its third argument. */
function lastLayout(): { annotations?: Annotation[] } {
  const call = plotly.react.mock.calls.at(-1);
  return (call?.[2] ?? {}) as { annotations?: Annotation[] };
}

function renderChart(dark = false, annotations?: Annotation[]) {
  return render(
    <PlotChart
      traces={[{ type: "scatter", x: [1, 2], y: [3, 4] }]}
      layout={{ title: { text: "Curve" }, ...(annotations ? { annotations } : {}) }}
      ariaLabel="Curve chart"
      dark={dark}
    />,
  );
}

describe("PlotChart config", () => {
  it("keeps Plotly's hover modebar on a wide container", async () => {
    renderChart();
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());
    reportWidth(900);
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());
    expect(lastConfig().displayModeBar).toBe("hover");
  });

  it("hides the modebar when the container is narrow", async () => {
    renderChart();
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());
    reportWidth(360);
    await waitFor(() => expect(lastConfig().displayModeBar).toBe(false));
  });

  it("restores the hover modebar when the container widens again", async () => {
    renderChart();
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());
    reportWidth(360);
    await waitFor(() => expect(lastConfig().displayModeBar).toBe(false));
    reportWidth(900);
    await waitFor(() => expect(lastConfig().displayModeBar).toBe("hover"));
  });
});

describe("PlotChart watermark", () => {
  it("adds exactly one faint watermark in light mode", async () => {
    renderChart();
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());

    const annotations = lastLayout().annotations ?? [];
    expect(annotations).toHaveLength(1);
    expect(annotations[0]).toMatchObject({
      text: "ConcreteInfo CurveStudio",
      xref: "paper",
      yref: "paper",
      x: 0.995,
      y: 0.012,
      xanchor: "right",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 10, color: "rgba(70,70,70,0.20)" },
    });
  });

  it("uses the dark watermark colour in dark mode", async () => {
    renderChart(true);
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());
    reportWidth(360);
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());

    const annotations = lastLayout().annotations ?? [];
    expect(annotations.filter((a) => a.text === "ConcreteInfo CurveStudio")).toHaveLength(1);
    expect(annotations.at(-1)?.font?.color).toBe("rgba(241,241,241,0.22)");
  });

  it("preserves existing annotations and appends the watermark last", async () => {
    renderChart(false, [{ text: "Fit: y = 2.35*x^0.89" }]);
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());

    const annotations = lastLayout().annotations ?? [];
    expect(annotations).toHaveLength(2);
    expect(annotations[0]?.text).toBe("Fit: y = 2.35*x^0.89");
    expect(annotations.at(-1)?.text).toBe("ConcreteInfo CurveStudio");
  });

  it("does not duplicate a watermark the caller already supplied", async () => {
    renderChart(false, [{ text: "ConcreteInfo CurveStudio" }]);
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());

    const annotations = lastLayout().annotations ?? [];
    expect(annotations).toHaveLength(1);
    expect(annotations[0]?.text).toBe("ConcreteInfo CurveStudio");
  });
});
