import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PlotlyLayout, PlotlyTrace } from "plotly.js-dist-min";
import { api, ApiError } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { InstallSummaryOut, OverviewOut } from "./adminApi";
import { OverviewPage } from "./OverviewPage";

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

vi.mock("../components/ThemeToggle", () => ({
  useTheme: () => ({ theme: "light", toggle: vi.fn() }),
}));

const INSTALLS: InstallSummaryOut = {
  total: 128,
  new_30d: 14,
  latest_release: "0.11.1",
  latest_share: 0.58,
  by_version: [],
  by_os: [],
  by_arch: [],
  by_package: [],
  by_session: [],
  new_per_week: [],
};

/** Ninety IST days ending 23/11/2026, oldest first: the 26th of August is the first. */
const SERIES = Array.from({ length: 90 }, (_, index) => {
  const day = new Date(Date.UTC(2026, 7, 26) + index * 86_400_000).toISOString().slice(0, 10);
  return { day, signups: index % 3, active: index };
});

const OVERVIEW: OverviewOut = {
  accounts: { total: 128, verified: 100, new_30d: 14 },
  active_users: { d1: 3, d7: 14, d30: 62 },
  installs: INSTALLS,
  open_errors: 4,
  autofit_runs_24h: 27,
  series: SERIES,
};

beforeEach(() => {
  plotly.react.mockClear();
});

function group(name: string): HTMLElement {
  return screen.getByRole("group", { name });
}

describe("OverviewPage", () => {
  it("shows loading, then every figure beside what it counts", async () => {
    let finish!: (value: unknown) => void;
    vi.spyOn(api, "request").mockImplementation(() => new Promise((resolve) => (finish = resolve)));
    renderWithQuery(<OverviewPage />);
    expect(screen.getByRole("status").textContent).toContain("Loading the overview");
    expect(screen.queryByRole("group")).toBeNull();

    await act(async () => {
      finish(OVERVIEW);
    });
    // the answer arrives on a later tick than the heading, which is there from the start
    await screen.findByRole("group", { name: "Accounts" });
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByRole("heading", { name: "Overview" })).toBeTruthy();

    const accounts = group("Accounts");
    expect(within(accounts).getByText("128")).toBeTruthy();
    expect(within(accounts).getByText("Registered accounts.")).toBeTruthy();
    const verified = group("Verified");
    expect(within(verified).getByText("100")).toBeTruthy();
    expect(within(verified).getByText("Accounts whose e-mail address was confirmed.")).toBeTruthy();

    // one label, two figures: accounts first, then installations
    const [newAccounts, newInstalls] = screen.getAllByRole("group", { name: "New in 30 days" });
    expect(within(newAccounts as HTMLElement).getByText("14")).toBeTruthy();
    expect(within(newAccounts as HTMLElement).getByText("Accounts created in the last 30 days.")).toBeTruthy();
    expect(within(newInstalls as HTMLElement).getByText("Installations first seen in the last 30 days.")).toBeTruthy();

    const active = group("Active users");
    expect(within(active).getByText("3")).toBeTruthy();
    expect(within(active).getByText("1 day")).toBeTruthy();
    expect(within(active).getByText("14")).toBeTruthy();
    expect(within(active).getByText("7 days")).toBeTruthy();
    expect(within(active).getByText("62")).toBeTruthy();
    expect(within(active).getByText("30 days")).toBeTruthy();
    expect(
      within(active).getByText(
        "Accounts with at least one signed-in request in the last 1, 7 and 30 days (today counts).",
      ),
    ).toBeTruthy();

    const installs = group("Desktop installs");
    expect(within(installs).getByText("128")).toBeTruthy();
    expect(within(installs).getByText("Distinct installations ever seen.")).toBeTruthy();
    expect(within(group("Latest version")).getByText("0.11.1")).toBeTruthy();

    const onLatest = group("On the latest version");
    expect(within(onLatest).getByText("58%")).toBeTruthy();
    const caveat = within(onLatest).getByText(/Installations whose last reported version is 0\.11\.1/);
    expect(caveat.textContent).toContain(
      "The console cannot tell how many are still in use: a desktop installation reports only on its first launch and after an upgrade.",
    );

    const open = group("Open errors");
    expect(within(open).getByText("4")).toBeTruthy();
    expect(within(open).getByText("Error events not yet acknowledged, resolved or ignored.")).toBeTruthy();
    const autofit = group("Auto-fit runs in 24 hours");
    expect(within(autofit).getByText("27")).toBeTruthy();
    expect(
      within(autofit).getByText("Auto-fit runs started in the last 24 hours, all accounts together."),
    ).toBeTruthy();
  });

  it("charts the series as two lines over the IST days, with a summary in words", async () => {
    vi.spyOn(api, "request").mockResolvedValue(OVERVIEW);
    renderWithQuery(<OverviewPage />);
    await waitFor(() => expect(plotly.react).toHaveBeenCalled());
    // Plotly.react(root, traces, layout, config)
    const [, traces, layout] = plotly.react.mock.calls.at(-1) as [
      HTMLElement,
      PlotlyTrace[],
      Partial<PlotlyLayout>,
    ];

    expect(traces).toHaveLength(2);
    const [signups, active] = traces;
    expect(signups?.x).toHaveLength(90);
    expect(active?.x).toHaveLength(90);
    expect(signups?.x?.[0]).toBe("2026-08-26");
    expect(signups?.x?.[89]).toBe("2026-11-23");
    expect(signups?.y).toEqual(SERIES.map((point) => point.signups));
    expect(active?.y).toEqual(SERIES.map((point) => point.active));
    expect(signups?.name).toBe("Sign-ups");
    expect(active?.name).toBe("Active accounts");
    expect(layout.xaxis?.type).toBe("date");
    expect(layout.xaxis?.tickformat).toBe("%d/%m/%Y");

    const summary = screen.getByText(/90 days from 26\/08\/2026 to 23\/11\/2026/);
    expect(summary.textContent).toContain(`${fmtTotalSignups()} accounts were created`);
    expect(summary.textContent).toContain("89 were active on 23/11/2026");
  });

  it("says so when there is no day to chart", async () => {
    vi.spyOn(api, "request").mockResolvedValue({ ...OVERVIEW, series: [] });
    renderWithQuery(<OverviewPage />);
    expect(await screen.findByText("There is nothing to chart yet.")).toBeTruthy();
    expect(plotly.react).not.toHaveBeenCalled();
  });

  it("shows an error with a Retry that fetches the overview again", async () => {
    const user = userEvent.setup();
    const spy = vi
      .spyOn(api, "request")
      .mockRejectedValueOnce(new ApiError(500, "INTERNAL_ERROR", "the server failed", null))
      .mockResolvedValue(OVERVIEW);
    renderWithQuery(<OverviewPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server failed");
    expect(screen.queryByRole("group")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("group", { name: "Accounts" })).toBeTruthy();
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy).toHaveBeenLastCalledWith("/admin/overview");
  });
});

/** Sign-ups over the series, summed the way the page words its summary. */
function fmtTotalSignups(): number {
  return SERIES.reduce((sum, point) => sum + point.signups, 0);
}
