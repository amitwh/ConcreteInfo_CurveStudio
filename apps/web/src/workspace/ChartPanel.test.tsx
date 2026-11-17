import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { ChartPanel } from "./ChartPanel";

type FitBundle = components["schemas"]["FitBundle"];

// jsdom cannot host a real Plotly graph; the unit under test is the banner logic.
vi.mock("../components/PlotChart", () => ({
  PlotChart: () => <div data-testid="plot" />,
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
