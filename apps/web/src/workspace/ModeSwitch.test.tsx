import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModeSwitch } from "./ModeSwitch";
import { loadMode } from "./mode";

describe("ModeSwitch", () => {
  it("defaults to Auto-fit on a first visit (no stored preference)", () => {
    expect(loadMode("user-1")).toBe("auto");
    render(<ModeSwitch mode="auto" onModeChange={() => {}} autofitStatus="idle" />);
    const auto = screen.getByRole("button", { name: "Auto-fit" });
    expect(auto.getAttribute("aria-pressed")).toBe("true");
  });

  it("switches to Manual fit and persists the choice per user", async () => {
    const onModeChange = vi.fn();
    render(<ModeSwitch mode="auto" onModeChange={onModeChange} autofitStatus="idle" />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Manual fit" }));
    expect(onModeChange).toHaveBeenCalledWith("manual");
  });

  it("restores a stored manual preference and keeps users separate", () => {
    localStorage.setItem("curvestudio.mode.user-1", "manual");
    expect(loadMode("user-1")).toBe("manual");
    expect(loadMode("user-2")).toBe("auto");
    localStorage.setItem("curvestudio.mode.user-1", "auto");
    expect(loadMode("user-1")).toBe("auto");
  });

  it("ignores a corrupted stored value", () => {
    localStorage.setItem("curvestudio.mode.user-1", "telepathy");
    expect(loadMode("user-1")).toBe("auto");
  });

  it("falls back to auto with no session (null user id)", () => {
    expect(loadMode(null)).toBe("auto");
  });

  it("restores a stored statistics preference", () => {
    localStorage.setItem("curvestudio.mode.user-1", "stats");
    expect(loadMode("user-1")).toBe("stats");
  });

  it("offers three modes, marks Statistics pressed and describes it", () => {
    render(<ModeSwitch mode="stats" onModeChange={() => {}} autofitStatus="idle" />);
    expect(screen.getAllByRole("button")).toHaveLength(3);
    const stats = screen.getByRole("button", { name: "Statistics" });
    expect(stats.getAttribute("aria-pressed")).toBe("true");
    expect(stats.getAttribute("title")).toBe("summarise columns and test differences");
    expect(screen.getByText("Statistics: idle")).toBeTruthy();
  });

  it("reports a switch to Statistics", async () => {
    const onModeChange = vi.fn();
    render(<ModeSwitch mode="auto" onModeChange={onModeChange} autofitStatus="idle" />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Statistics" }));
    expect(onModeChange).toHaveBeenCalledWith("stats");
  });
});
