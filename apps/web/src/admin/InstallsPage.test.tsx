import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { InstallSummaryOut } from "./adminApi";
import { InstallsPage } from "./InstallsPage";

/** The twelve Mondays ending with the one 23/11/2026 starts. */
const MONDAYS = [
  "2026-09-07",
  "2026-09-14",
  "2026-09-21",
  "2026-09-28",
  "2026-10-05",
  "2026-10-12",
  "2026-10-19",
  "2026-10-26",
  "2026-11-02",
  "2026-11-09",
  "2026-11-16",
  "2026-11-23",
];

const INSTALLS: InstallSummaryOut = {
  total: 128,
  new_30d: 14,
  latest_release: "0.11.1",
  latest_share: 0.58,
  by_version: [
    { name: "0.11.1", count: 74 },
    { name: "0.11.0", count: 40 },
    { name: "other", count: 14 },
  ],
  by_os: [
    { name: "linux", count: 70 },
    { name: "windows", count: 50 },
    { name: "other", count: 8 },
  ],
  by_arch: [
    { name: "x86_64", count: 100000 },
    { name: "arm64", count: 28 },
  ],
  by_package: [
    { name: "deb", count: 60 },
    { name: "appimage", count: 10 },
  ],
  by_session: [
    { name: "wayland", count: 40 },
    { name: "x11", count: 30 },
  ],
  new_per_week: MONDAYS.map((week_start, index) => ({ week_start, count: index })),
};

const NOTHING: InstallSummaryOut = {
  ...INSTALLS,
  total: 0,
  new_30d: 0,
  latest_release: "0.12.0.dev3",
  latest_share: 0,
  by_version: [],
  by_os: [],
  by_arch: [],
  by_package: [],
  by_session: [],
  new_per_week: [],
};

function group(name: string): HTMLElement {
  return screen.getByRole("group", { name });
}

function cells(caption: string): string[] {
  return within(screen.getByRole("table", { name: caption }))
    .getAllByRole("cell")
    .map((cell) => cell.textContent);
}

describe("InstallsPage", () => {
  it("shows loading, then the four figures beside what they count", async () => {
    vi.spyOn(api, "request").mockResolvedValue(INSTALLS);
    renderWithQuery(<InstallsPage />);
    expect(screen.getByRole("status").textContent).toContain("Loading the install figures");
    await screen.findByRole("group", { name: "Desktop installs" });

    expect(within(group("Desktop installs")).getByText("128")).toBeTruthy();
    expect(within(group("Desktop installs")).getByText("Distinct installations ever seen.")).toBeTruthy();
    expect(within(group("New in 30 days")).getByText("14")).toBeTruthy();
    expect(within(group("New in 30 days")).getByText("Installations first seen in the last 30 days.")).toBeTruthy();
    expect(within(group("Latest version")).getByText("0.11.1")).toBeTruthy();
    const share = group("On the latest version");
    expect(within(share).getByText("58%")).toBeTruthy();
    expect(within(share).getByText(/Installations whose last reported version is 0\.11\.1/)).toBeTruthy();
  });

  it("says no installation has reported the latest version when its share is 0", async () => {
    vi.spyOn(api, "request").mockResolvedValue(NOTHING);
    renderWithQuery(<InstallsPage />);
    await screen.findByRole("group", { name: "On the latest version" });
    const share = group("On the latest version");
    expect(within(share).getByText("0%")).toBeTruthy();
    const explanation = within(share).getByText(/No installation has reported this version/);
    expect(explanation.textContent).toContain("0.12.0.dev3");
  });

  it("lists the five breakdowns, most first, with 'other' spelled out", async () => {
    vi.spyOn(api, "request").mockResolvedValue(INSTALLS);
    renderWithQuery(<InstallsPage />);
    await screen.findByRole("table", { name: "By version" });

    expect(cells("By version")).toEqual(["0.11.1", "74", "0.11.0", "40", "Other or unknown", "14"]);
    expect(cells("By operating system")).toEqual(["linux", "70", "windows", "50", "Other or unknown", "8"]);
    expect(cells("By architecture")).toEqual(["x86_64", "1,00,000", "arm64", "28"]);
    expect(cells("By package")).toEqual(["deb", "60", "appimage", "10"]);
    expect(cells("By session")).toEqual(["wayland", "40", "x11", "30"]);
    expect(
      within(screen.getByRole("table", { name: "By version" })).getAllByRole("columnheader").map((th) => th.textContent),
    ).toEqual(["Version", "Installations"]);
    expect(
      within(screen.getByRole("table", { name: "By operating system" })).getAllByRole("columnheader").map((th) => th.textContent),
    ).toEqual(["Operating system", "Installations"]);
  });

  it("lists the twelve weeks, each by the Monday it starts on", async () => {
    vi.spyOn(api, "request").mockResolvedValue(INSTALLS);
    renderWithQuery(<InstallsPage />);
    await screen.findByRole("table", { name: "New installations per week" });
    const weeks = screen.getByRole("table", { name: "New installations per week" });
    const rows = within(weeks).getAllByRole("row");
    expect(rows).toHaveLength(13);
    const cellsOf = cells("New installations per week");
    expect(cellsOf.slice(0, 6)).toEqual(["Monday 07/09/2026", "0", "Monday 14/09/2026", "1", "Monday 21/09/2026", "2"]);
    expect(cellsOf.slice(-2)).toEqual(["Monday 23/11/2026", "11"]);
  });

  it("shows an empty state in every table when no installation has been seen", async () => {
    vi.spyOn(api, "request").mockResolvedValue(NOTHING);
    renderWithQuery(<InstallsPage />);
    await screen.findByRole("heading", { name: "Installs" });
    expect(await screen.findAllByText("No installation has been seen yet.")).toHaveLength(6);
    expect(screen.queryByRole("table")).toBeNull();
    expect(within(group("Desktop installs")).getByText("0")).toBeTruthy();
  });

  it("shows an error with a Retry that fetches the figures again", async () => {
    const user = userEvent.setup();
    const spy = vi
      .spyOn(api, "request")
      .mockRejectedValueOnce(new ApiError(503, "UNAVAILABLE", "the server is away", null))
      .mockResolvedValue(INSTALLS);
    renderWithQuery(<InstallsPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server is away");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table", { name: "By version" })).toBeTruthy();
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy).toHaveBeenLastCalledWith("/admin/installs/summary");
  });
});
