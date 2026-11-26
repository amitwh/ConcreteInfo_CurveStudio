/** The fit history's IST timestamps: every entry carries `DD/MM/YYYY HH:mm`. */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { FitSummaryOut } from "../../api/client";
import { HistoryTab } from "./HistoryTab";

function fit(overrides: Record<string, unknown> = {}): FitSummaryOut {
  return {
    id: "fit-1",
    dataset_id: "ds-1",
    created_at: "2026-11-25T09:05:00Z", // 14:35 IST that day
    status: "converged",
    spec: { model: { kind: "registered", id: "exp_decay" } },
    ...overrides,
  } as FitSummaryOut;
}

describe("HistoryTab", () => {
  it("shows each stored fit's time as DD/MM/YYYY HH:mm in IST", () => {
    render(
      <HistoryTab fits={[fit()]} activeFitId="fit-1" onLoad={vi.fn()} onDelete={vi.fn()} />,
    );
    // 09:05 UTC is 14:35 the same day at +05:30
    expect(screen.getByText("25/11/2026 14:35")).toBeTruthy();
  });

  it("carries the full time in the row's tooltip too", () => {
    render(
      <HistoryTab fits={[fit()]} activeFitId="" onLoad={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(screen.getByTitle("exp_decay · 25/11/2026 14:35")).toBeTruthy();
  });

  it("loads a fit on click and deletes after the inline confirm", async () => {
    const onLoad = vi.fn();
    const onDelete = vi.fn();
    render(<HistoryTab fits={[fit()]} activeFitId="" onLoad={onLoad} onDelete={onDelete} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /exp_decay/ }));
    expect(onLoad).toHaveBeenCalledWith("fit-1");
    await user.click(screen.getByRole("button", { name: "Delete fit fit-1" }));
    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledWith("fit-1");
  });
});
