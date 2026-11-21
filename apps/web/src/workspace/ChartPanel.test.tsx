import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { ChartPanel } from "./ChartPanel";
import { modelColor } from "./traces";

type FitBundle = components["schemas"]["FitBundle"];
type RankedFit = components["schemas"]["RankedFit"];
type RowCurve = components["schemas"]["RowCurve"];
type Trace = { mode?: string; fill?: string; name?: string; line?: { color?: string } };

const captured = vi.hoisted(() => ({ traces: [] as Trace[] }));

// jsdom cannot host a real Plotly graph; capture the traces the panel builds.
vi.mock("../components/PlotChart", () => ({
  PlotChart: ({ traces }: { traces: Trace[] }) => {
    captured.traces = traces;
    return <div data-testid="plot" />;
  },
  downloadPlotPng: () => {},
}));

const dataset = {
  id: "ds-1",
  name: "growth.csv",
  n_rows: 2,
  columns: ["x", "y"],
  x_column: "x",
  y_column: "y",
  weight_column: null,
  issues: [],
  created_at: "2026-11-17T05:00:00Z",
  points: { n_total: 2, x: [1, 2], y: [2, 4], w: null },
};

const RANKED = [1, 2, 3].map(
  (rank) =>
    ({
      rank,
      model_id: `m${rank}`,
      name: `Model ${rank}`,
      status: "converged",
      metrics: { k: 2, n: 10 },
    }) as unknown as RankedFit,
);

function storedCurves(): Record<string, RowCurve> {
  const curves: Record<string, RowCurve> = {};
  for (const row of RANKED) {
    curves[row.model_id] = { x: [0, 1], y: [2, row.rank] } as unknown as RowCurve;
  }
  return curves;
}

describe("ChartPanel show top 3", () => {
  it("enables the toggle only when at least two stored curves can be overlaid", () => {
    const bundle = {
      result: { status: "converged" },
      curve: { x: [0, 1], y: [2, 1] },
      confidence_band: null,
      prediction_band: null,
    } as unknown as FitBundle;
    const { unmount } = render(
      <ChartPanel dataset={dataset} bundle={bundle} dark={false} />,
    );
    expect(
      (screen.getByRole("checkbox", { name: /three best-ranked/i }) as HTMLInputElement).disabled,
    ).toBe(true);
    unmount();
    render(
      <ChartPanel
        dataset={dataset}
        bundle={bundle}
        dark={false}
        rankingRows={RANKED}
        storedCurves={storedCurves()}
        selectedRowId="m1"
      />,
    );
    expect(
      (screen.getByRole("checkbox", { name: /three best-ranked/i }) as HTMLInputElement).disabled,
    ).toBe(false);
  });

  it("gives tied ranks (1, 2, 2) three distinct overlay colours by position", async () => {
    const tied = [
      { rank: 1, model_id: "a", name: "Model a", status: "converged", metrics: { k: 2, n: 10 } },
      { rank: 2, model_id: "b", name: "Model b", status: "converged", metrics: { k: 2, n: 10 } },
      { rank: 2, model_id: "c", name: "Model c", status: "converged", metrics: { k: 2, n: 10 } },
    ] as unknown as RankedFit[];
    const curves: Record<string, RowCurve> = {};
    for (const row of tied) {
      curves[row.model_id] = { x: [0, 1], y: [2, 1] } as unknown as RowCurve;
    }
    const bundle = {
      result: { status: "converged" },
      curve: { x: [0, 1], y: [2, 1] },
      confidence_band: null,
      prediction_band: null,
    } as unknown as FitBundle;
    render(
      <ChartPanel
        dataset={dataset}
        bundle={bundle}
        dark={false}
        rankingRows={tied}
        storedCurves={curves}
        selectedRowId="a"
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox", { name: /three best-ranked/i }));
    const curvesShown = captured.traces.filter(
      (trace) => trace.mode === "lines" && trace.fill === undefined,
    );
    expect(curvesShown).toHaveLength(3);
    expect(curvesShown.map((trace) => trace.line?.color)).toEqual([
      modelColor(1),
      modelColor(2),
      modelColor(3),
    ]);
    expect(new Set(curvesShown.map((trace) => trace.line?.color)).size).toBe(3);
    expect(curvesShown.map((trace) => trace.name)).toEqual(["Model a", "Model b", "Model c"]);
  });

  it("colours the single curve by the selected row's position, not its tied rank", () => {
    const tied = [
      { rank: 1, model_id: "a", name: "Model a", status: "converged", metrics: { k: 2, n: 10 } },
      { rank: 2, model_id: "b", name: "Model b", status: "converged", metrics: { k: 2, n: 10 } },
      { rank: 2, model_id: "c", name: "Model c", status: "converged", metrics: { k: 2, n: 10 } },
    ] as unknown as RankedFit[];
    const bundle = {
      result: { status: "converged" },
      curve: { x: [0, 1], y: [2, 1] },
      confidence_band: null,
      prediction_band: null,
    } as unknown as FitBundle;
    render(
      <ChartPanel dataset={dataset} bundle={bundle} dark={false} rankingRows={tied} selectedRowId="c" />,
    );
    const curve = captured.traces.find((trace) => trace.name === "Fitted curve");
    expect(curve?.line?.color).toBe(modelColor(3)); // position 3, though its rank is 2
  });

  it("toggles on and off without losing the chart", async () => {
    const bundle = {
      result: { status: "converged" },
      curve: { x: [0, 1], y: [2, 1] },
      confidence_band: null,
      prediction_band: null,
    } as unknown as FitBundle;
    render(
      <ChartPanel
        dataset={dataset}
        bundle={bundle}
        dark={false}
        rankingRows={RANKED}
        storedCurves={storedCurves()}
        selectedRowId="m1"
      />,
    );
    const toggle = screen.getByRole("checkbox", { name: /three best-ranked/i });
    const user = userEvent.setup();
    await user.click(toggle);
    expect((toggle as HTMLInputElement).checked).toBe(true);
    expect(screen.getByTestId("plot")).toBeTruthy();
  });
});

describe("ChartPanel banners for problem fits", () => {
  it("renders NO duplicate banner for a non-converged fit (ResultsTabs owns the one banner)", () => {
    const bundle = { result: { status: "not_converged" } } as unknown as FitBundle;
    render(<ChartPanel dataset={dataset} bundle={bundle} dark={false} />);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByTestId("plot")).toBeTruthy();
  });

  it("keeps the warning when a fit converged but no curve could be sampled", () => {
    const bundle = {
      result: { status: "converged" },
      curve: null,
      curve_reason: "the fit range lies outside the data",
    } as unknown as FitBundle;
    render(<ChartPanel dataset={dataset} bundle={bundle} dark={false} />);
    const warning = screen.getByText(/no curve could be sampled/);
    expect(warning.textContent).toContain("the fit range lies outside the data");
  });
});
