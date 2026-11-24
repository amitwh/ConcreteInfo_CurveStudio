import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { StatResultSummaryOut } from "../../api/client";
import { StatsHistory } from "./StatsHistory";

const OLDER: StatResultSummaryOut = {
  id: "r-1",
  tool: "descriptives",
  title: "Descriptive statistics",
  dataset_id: "ds-1",
  created_at: "2026-11-17T03:30:00Z", // 09:00 IST
};

const NEWER: StatResultSummaryOut = {
  id: "r-2",
  tool: "welch_t",
  title: "Two-sample comparison of means",
  dataset_id: "ds-1",
  created_at: "2026-11-17T10:15:00Z", // 15:45 IST
};

describe("StatsHistory", () => {
  it("lists the results newest first with their IST date and time", () => {
    const { container } = render(
      <StatsHistory results={[OLDER, NEWER]} onSelect={() => {}} onDelete={() => {}} />,
    );
    const rows = Array.from(container.querySelectorAll(".history__load"));
    expect(rows).toHaveLength(2);
    expect(rows[0]?.textContent).toContain("Two-sample comparison of means");
    expect(rows[0]?.textContent).toContain("17/11/2026, 15:45 IST");
    expect(rows[1]?.textContent).toContain("Descriptive statistics");
    expect(rows[1]?.textContent).toContain("17/11/2026, 09:00 IST");
  });

  it("opens a stored result", () => {
    const onSelect = vi.fn();
    render(<StatsHistory results={[OLDER]} onSelect={onSelect} onDelete={() => {}} />);
    fireEvent.click(screen.getByText("Descriptive statistics"));
    expect(onSelect).toHaveBeenCalledWith("r-1");
  });

  it("asks before deleting a result", () => {
    const onDelete = vi.fn();
    render(<StatsHistory results={[OLDER]} onSelect={() => {}} onDelete={onDelete} />);
    fireEvent.click(screen.getByRole("button", { name: "Delete result r-1" }));
    expect(onDelete).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledWith("r-1");
  });

  it("renders nothing when there are no stored results", () => {
    const { container } = render(
      <StatsHistory results={[]} onSelect={() => {}} onDelete={() => {}} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
