import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { RouteLatencyOut, SystemOut } from "./adminApi";
import { SystemPage } from "./SystemPage";

const SYSTEM: SystemOut = {
  version: "0.11.1",
  current_release: "0.11.1",
  started_at: "2026-11-22T21:02:03+00:00",
  uptime_s: 11520.4,
  jobs: { running: true, workers: 2, queue_depth: 0, queue_max: 100 },
  database: { size_bytes: null, migration_head: "0012_request_stats_hourly" },
};

const DAY: RouteLatencyOut[] = [
  { route_template: "/api/v1/me", count: 1200, errors: 3, p50_ms: 12, p95_ms: 140, max_ms: 1204 },
  { route_template: "/api/v1/datasets/{id}", count: 40, errors: 0, p50_ms: 8, p95_ms: 30, max_ms: 77 },
];
const WEEK: RouteLatencyOut[] = [
  { route_template: "/api/v1/auth/login", count: 99, errors: 1, p50_ms: 210, p95_ms: 900, max_ms: 3001 },
];

/** A spy on the client's one request method, with its call record typed. */
function spyRequest() {
  return vi.spyOn(api, "request");
}
type RequestSpy = ReturnType<typeof spyRequest>;

/** Answers like the API, by path; any other path fails the test. */
function serve(responses: Record<string, unknown>): RequestSpy {
  return spyRequest().mockImplementation(async (path) => {
    const body = responses[path];
    if (body === undefined) throw new Error(`unexpected request ${path}`);
    return body;
  });
}

function group(name: string): HTMLElement {
  return screen.getByRole("group", { name });
}

function latencyRows(): string[][] {
  return within(screen.getByRole("table", { name: "Request latency by route" }))
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell").map((cell) => cell.textContent));
}

describe("SystemPage", () => {
  it("shows loading, then each figure beside what it is", async () => {
    spyRequest().mockResolvedValue(SYSTEM);
    renderWithQuery(<SystemPage />);
    // two fetches, two loading lines: the system figures and the latency
    const loading = screen.getAllByRole("status").map((line) => line.textContent);
    expect(loading).toContain("Loading the system figures…");
    expect(loading).toContain("Loading the request latency…");
    await screen.findByRole("group", { name: "API version" });

    const version = group("API version");
    expect(within(version).getByText("0.11.1")).toBeTruthy();
    expect(within(version).getByText("The version of the API serving this console.")).toBeTruthy();

    const release = group("Current release");
    expect(within(release).getByText("0.11.1")).toBeTruthy();
    expect(within(release).getByText("The API's own version: no release is set.")).toBeTruthy();

    const uptime = group("Uptime");
    expect(within(uptime).getByText("3 h 12 min")).toBeTruthy();
    expect(within(uptime).getByText("Since the API started on 23/11/2026 02:32.")).toBeTruthy();

    const jobs = group("Jobs");
    expect(within(jobs).getByText("Running")).toBeTruthy();
    expect(within(jobs).getByText("2 workers; 0 of 100 queue slots used.")).toBeTruthy();

    const database = group("Database");
    expect(within(database).getByText("Not reported")).toBeTruthy();
    expect(
      within(database).getByText(
        "The size is not reported on this database. Schema revision 0012_request_stats_hourly.",
      ),
    ).toBeTruthy();
  });

  it("names a release that is not the API's own version, and a database size when there is one", async () => {
    spyRequest().mockResolvedValue({
      ...SYSTEM,
      current_release: "0.12.0",
      jobs: { running: false, workers: 0, queue_depth: 12, queue_max: 100 },
      database: { size_bytes: 125_000_000, migration_head: "0012_request_stats_hourly" },
    });
    renderWithQuery(<SystemPage />);
    await screen.findByRole("group", { name: "Current release" });
    expect(
      within(group("Current release")).getByText("The release the console counts as the latest."),
    ).toBeTruthy();
    expect(within(group("Jobs")).getByText("Stopped")).toBeTruthy();
    expect(within(group("Jobs")).getByText("0 workers; 12 of 100 queue slots used.")).toBeTruthy();
    expect(within(group("Database")).getByText("125 MB")).toBeTruthy();
    expect(
      within(group("Database")).getByText("Size of the database. Schema revision 0012_request_stats_hourly."),
    ).toBeTruthy();
  });

  it("reads a single worker and a single queue slot in the singular", async () => {
    spyRequest().mockResolvedValue({
      ...SYSTEM,
      jobs: { running: true, workers: 1, queue_depth: 0, queue_max: 1 },
    });
    renderWithQuery(<SystemPage />);
    await screen.findByRole("group", { name: "Jobs" });
    expect(within(group("Jobs")).getByText("1 worker; 0 of 1 queue slot used.")).toBeTruthy();
  });

  it("tables the last day's latency per route, and switches periods on request", async () => {
    const user = userEvent.setup();
    const spy = serve({
      "/admin/system": SYSTEM,
      "/admin/system/latency?days=1": DAY,
      "/admin/system/latency?days=7": WEEK,
      "/admin/system/latency?days=30": [],
    });
    renderWithQuery(<SystemPage />);
    await screen.findByRole("table", { name: "Request latency by route" });

    expect(
      within(screen.getByRole("table", { name: "Request latency by route" }))
        .getAllByRole("columnheader")
        .map((th) => th.textContent),
    ).toEqual(["Route", "Requests", "Errors", "Median", "95th percentile", "Slowest"]);
    expect(latencyRows()).toEqual([
      ["/api/v1/me", "1,200", "3", "12 ms", "140 ms", "1,204 ms"],
      ["/api/v1/datasets/{id}", "40", "0", "8 ms", "30 ms", "77 ms"],
    ]);
    expect(screen.getByText("Route templates only, never raw paths. A restart loses at most the last hour of figures.")).toBeTruthy();
    expect((screen.getByRole("radio", { name: "1 day" }) as HTMLInputElement).checked).toBe(true);

    await user.click(screen.getByRole("radio", { name: "7 days" }));
    await waitFor(() => expect(spy).toHaveBeenLastCalledWith("/admin/system/latency?days=7"));
    await waitFor(() =>
      expect(latencyRows()).toEqual([["/api/v1/auth/login", "99", "1", "210 ms", "900 ms", "3,001 ms"]]),
    );

    await user.click(screen.getByRole("radio", { name: "30 days" }));
    expect(await screen.findByText("No requests were recorded in this period.")).toBeTruthy();
    expect(screen.getByText("Route templates only, never raw paths. A restart loses at most the last hour of figures.")).toBeTruthy();
  });

  it("shows an error with a Retry that fetches the system figures again", async () => {
    const user = userEvent.setup();
    let systemCalls = 0;
    const spy = spyRequest().mockImplementation(async (path) => {
      if (path === "/admin/system") {
        // the first ask fails; the Retry's ask succeeds
        systemCalls += 1;
        if (systemCalls === 1) throw new ApiError(500, "INTERNAL_ERROR", "the server failed", null);
        return SYSTEM;
      }
      if (path === "/admin/system/latency?days=1") return DAY;
      throw new Error(`unexpected request ${path}`);
    });
    renderWithQuery(<SystemPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server failed");
    // the latency figures are their own fetch and are not held up by the failure
    expect(await screen.findByRole("table", { name: "Request latency by route" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("group", { name: "API version" })).toBeTruthy();
    expect(spy.mock.calls.filter(([path]) => path === "/admin/system")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("shows an error and a Retry for the latency figures on their own", async () => {
    const user = userEvent.setup();
    let latencyCalls = 0;
    const spy = spyRequest().mockImplementation(async (path) => {
      if (path === "/admin/system") return SYSTEM;
      if (path === "/admin/system/latency?days=1") {
        // the first ask fails; the Retry's ask succeeds
        latencyCalls += 1;
        if (latencyCalls === 1) {
          throw new ApiError(500, "INTERNAL_ERROR", "the figures are away", null);
        }
        return DAY;
      }
      throw new Error(`unexpected request ${path}`);
    });
    renderWithQuery(<SystemPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the figures are away");
    expect(screen.getByRole("group", { name: "API version" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table", { name: "Request latency by route" })).toBeTruthy();
    expect(spy.mock.calls.filter(([path]) => path === "/admin/system")).toHaveLength(1);
  });
});
