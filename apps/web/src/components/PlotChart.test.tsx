import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const plotly = vi.hoisted(() => ({
  react: vi.fn((..._args: unknown[]) => Promise.resolve()),
  purge: vi.fn(),
  resize: vi.fn(),
}));

vi.mock("plotly.js-dist-min", () => ({
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

function renderChart() {
  return render(
    <PlotChart
      traces={[{ type: "scatter", x: [1, 2], y: [3, 4] }]}
      layout={{ title: { text: "Curve" } }}
      ariaLabel="Curve chart"
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
