import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { JobStatus } from "./JobStatus";
import type { JobState } from "../hooks/useJobPolling";
import { ENGINE_SENTENCES } from "../messages";

const succeeded: JobState = { phase: "succeeded", status: "succeeded", fitId: "fit-1" };

describe("JobStatus aria-live text", () => {
  it("announces 'Fit finished.' when the job succeeded (default)", () => {
    render(<JobStatus state={succeeded} onCancel={vi.fn()} />);
    expect(screen.getByRole("status").textContent).toContain("Fit finished.");
  });

  it("announces 'Fit did not converge.' when the loaded fit did not converge", () => {
    render(<JobStatus state={succeeded} onCancel={vi.fn()} succeededText="Fit did not converge." />);
    expect(screen.getByRole("status").textContent).toContain("Fit did not converge.");
    expect(screen.getByRole("status").textContent).not.toContain("Fit finished.");
  });

  it("announces a failed job as its sentence, with the server's message after it", () => {
    const failed: JobState = {
      phase: "failed",
      status: "failed",
      code: "TIMEOUT",
      message: "the fit did not finish within 30 s",
    };
    render(<JobStatus state={failed} onCancel={vi.fn()} />);
    const text = screen.getByRole("status").textContent ?? "";
    expect(text).toContain(ENGINE_SENTENCES["TIMEOUT"]);
    expect(text).toContain("the fit did not finish within 30 s");
    expect(text).not.toContain("TIMEOUT"); // the sentence stands in for the bare code
  });
});
