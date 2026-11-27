/**
 * The opened-result modal: it reuses the tabs' own rendering (equation, statistics,
 * parameters) inside the house modal, says so while the fit is in flight, and surfaces
 * a refused open's error.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { FitBundle } from "../../api/client";
import { OpenResultModal } from "./OpenResultModal";

/** Only the fields the three reused tabs touch. */
const BUNDLE = {
  result: {
    status: "converged",
    termination_reason: "converged",
    model_id: "exp_decay",
    equation: "y = a * exp(-b * x)",
    equation_fitted: "y = 2.05 * exp(-0.32 * x)",
    method: "trf",
    loss: "linear",
    n_starts: 3,
    provenance: { seed: 7 },
    parameters: [
      { name: "a", value: 2.05, std_err: 0.01, ci_low: 2.03, ci_high: 2.07 },
    ],
    stats: {
      n: 40,
      k: 2,
      df: 38,
      r2: 0.99,
      adj_r2: 0.98,
      rmse: 0.05,
      sse: 0.1,
      mae: 0.04,
      residual_se: 0.05,
    },
  },
  equation_text: "y = a * exp(-b * x)",
  curve: null,
  confidence_band: null,
  prediction_band: null,
  summary: {
    criteria: { aic: -120, aicc: -119.8, bic: -117, convention: "as printed in reports" },
    cv: null,
    result_summary: {},
    diagnostics: { tests: [] },
    warnings: [],
  },
} as unknown as FitBundle;

describe("OpenResultModal", () => {
  it("presents the opened result with the tabs' own rendering", () => {
    render(
      <OpenResultModal name="Exponential decay" bundle={BUNDLE} busy={false} error={null} onClose={() => {}} />,
    );
    const dialog = screen.getByRole("dialog", { name: "Opened result — Exponential decay" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    // EquationTab, StatisticsTab and ParametersTab, exactly as the results panel shows them.
    expect(within(dialog).getByText("Model equation")).toBeTruthy();
    expect(within(dialog).getAllByText("R²").length).toBeGreaterThan(0);
    expect(
      within(dialog).getByRole("columnheader", { name: "Parameter" }),
    ).toBeTruthy();
    expect(within(dialog).getByText("Adjusted R²")).toBeTruthy();
  });

  it("says the fit is in flight while the open request runs", () => {
    render(
      <OpenResultModal name="Quadratic" bundle={null} busy={true} error={null} onClose={() => {}} />,
    );
    expect(
      screen.getByText(/Fitting this model — the result opens here when it is ready/),
    ).toBeTruthy();
    expect(screen.queryByText("Model equation")).toBeNull();
  });

  it("shows a refused open's error instead of the results", () => {
    render(
      <OpenResultModal
        name="Quadratic"
        bundle={null}
        busy={false}
        error="the row could not be opened"
        onClose={() => {}}
      />,
    );
    expect(screen.getByRole("alert")).toBeTruthy();
    expect(screen.queryByText("Model equation")).toBeNull();
  });

  it("closes through the modal's own controls", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <OpenResultModal name="Quadratic" bundle={null} busy={true} error={null} onClose={onClose} />,
    );
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
