/** The parameters table's display-only sorting; the engine's row order is the default. */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { components } from "../../api/schema";
import { ParametersTab } from "./ParametersTab";

type FitBundle = components["schemas"]["FitBundle"];

function bundle(parameters: Array<Record<string, unknown>>): FitBundle {
  return {
    result: {
      model_id: "m",
      equation: "y = a * x",
      equation_fitted: null,
      parameters,
      covariance: null,
      correlation: null,
      stats: null,
      fitted: [],
      residuals: [],
      status: "converged",
      termination_reason: "converged",
      method: "lm",
      loss: "linear",
      initial_params: {},
      n_starts: 1,
      warnings: [],
      provenance: {
        software_versions: {},
        dataset_fingerprint: "fp",
        created_at: "2026-11-17T05:00:00Z",
        seed: 0,
      },
    },
    summary: null,
    summary_reason: null,
    curve: null,
    curve_reason: null,
    confidence_band: null,
    prediction_band: null,
    bands_reason: null,
    equation_text: "y = a * x",
    warnings: [],
    data: { x: [0, 1], y: [1, 2], weights: null },
  } as unknown as FitBundle;
}

const PARAMETERS = [
  { name: "tau", value: 12.5, std_err: 1.25, ci_low: 10.0, ci_high: 15.0 },
  { name: "a", value: 3.1, std_err: 0.4, ci_low: 2.3, ci_high: 3.9 },
  { name: "b", value: 0.8, std_err: null, ci_low: null, ci_high: null },
];

function rowNames(): string[] {
  return screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("rowheader")[0]?.textContent ?? "");
}

describe("ParametersTab sorting", () => {
  it("sorts by estimate, reverses, then returns to the engine's order", async () => {
    const user = userEvent.setup();
    render(<ParametersTab bundle={bundle(PARAMETERS)} />);
    expect(rowNames()).toEqual(["tau", "a", "b"]); // the engine's order
    await user.click(screen.getByRole("button", { name: "Estimate" }));
    expect(rowNames()).toEqual(["b", "a", "tau"]); // ascending value
    await user.click(screen.getByRole("button", { name: "Estimate" }));
    expect(rowNames()).toEqual(["tau", "a", "b"]); // descending
    await user.click(screen.getByRole("button", { name: "Estimate" }));
    expect(rowNames()).toEqual(["tau", "a", "b"]); // back to the engine's order
  });

  it("sorts by name and keeps undefined standard errors last", async () => {
    const user = userEvent.setup();
    render(<ParametersTab bundle={bundle(PARAMETERS)} />);
    await user.click(screen.getByRole("button", { name: "Parameter" }));
    expect(rowNames()).toEqual(["a", "b", "tau"]);
    await user.click(screen.getByRole("button", { name: "Std. error" }));
    // ascending std_err with b's undefined value at the bottom
    expect(rowNames()).toEqual(["a", "tau", "b"]);
    await user.click(screen.getByRole("button", { name: "Std. error" }));
    // descending: undefined still last
    expect(rowNames()).toEqual(["tau", "a", "b"]);
  });

  it("sorts the confidence interval column by its lower end", async () => {
    const user = userEvent.setup();
    render(<ParametersTab bundle={bundle(PARAMETERS)} />);
    await user.click(screen.getByRole("button", { name: "95% CI" }));
    expect(rowNames()).toEqual(["a", "tau", "b"]); // by ci_low, b undefined last
  });
});
