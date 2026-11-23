import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import {
  audit,
  installsSummary,
  overview,
  plans,
  system,
  systemLatency,
  updatePlan,
} from "./adminApi";

const OVERVIEW = { accounts: { total: 3, verified: 2, new_30d: 1 }, series: [] };
const SUMMARY = { total: 12, new_30d: 2, latest_release: "0.11.1", latest_share: 0.5 };
const PAGE = { items: [], total: 0, page: 1, page_size: 25 };
const SYSTEM = { version: "0.11.1", current_release: "0.11.1", uptime_s: 5 };
const LATENCY = [{ route_template: "/me", count: 9, errors: 0, p50_ms: 12, p95_ms: 40, max_ms: 91 }];
const PLAN = { name: "free", limits: { max_datasets: 25, max_dataset_bytes: 1, max_fits: 5, max_autofits_per_day: 2 } };

afterEach(() => {
  vi.restoreAllMocks();
});

/** The one call the function made: its path, and the request init (or undefined). */
function lastCall(): { path: string; init: RequestInit | undefined } {
  const call = vi.mocked(api.request).mock.calls.at(-1);
  return { path: call?.[0] ?? "", init: call?.[1] };
}

describe("adminApi — one typed function per console call", () => {
  it("overview fetches GET /admin/overview and returns what the client returned", async () => {
    vi.spyOn(api, "request").mockResolvedValue(OVERVIEW);
    await expect(overview()).resolves.toBe(OVERVIEW);
    expect(vi.mocked(api.request)).toHaveBeenCalledTimes(1);
    expect(lastCall()).toEqual({ path: "/admin/overview", init: undefined });
  });

  it("installsSummary fetches GET /admin/installs/summary", async () => {
    vi.spyOn(api, "request").mockResolvedValue(SUMMARY);
    await expect(installsSummary()).resolves.toBe(SUMMARY);
    expect(lastCall()).toEqual({ path: "/admin/installs/summary", init: undefined });
  });

  it("system fetches GET /admin/system", async () => {
    vi.spyOn(api, "request").mockResolvedValue(SYSTEM);
    await expect(system()).resolves.toBe(SYSTEM);
    expect(lastCall()).toEqual({ path: "/admin/system", init: undefined });
  });

  it("plans fetches GET /admin/plans", async () => {
    vi.spyOn(api, "request").mockResolvedValue([PLAN]);
    await expect(plans()).resolves.toEqual([PLAN]);
    expect(lastCall()).toEqual({ path: "/admin/plans", init: undefined });
  });
});

describe("adminApi.audit — the filters become a query string with no empty parameters", () => {
  it("sends only the page when nothing else is filtered", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await audit({ page: 2 });
    expect(lastCall()).toEqual({ path: "/admin/audit?page=2", init: undefined });
  });

  it("sends every filter under the API's names, in order", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await audit({
      userId: "6e0f2f4a-1f7c-4b4e-9a4a-0c7d2f9b1c11",
      action: "user.role_change",
      from: "2026-11-01",
      to: "2026-11-18",
      page: 1,
      pageSize: 50,
    });
    expect(lastCall().path).toBe(
      "/admin/audit?user_id=6e0f2f4a-1f7c-4b4e-9a4a-0c7d2f9b1c11&action=user.role_change&from=2026-11-01&to=2026-11-18&page=1&page_size=50",
    );
  });

  it("leaves out an empty filter and escapes what it sends", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await audit({ userId: "", action: "a b&c=d", page: 1, pageSize: undefined });
    expect(lastCall().path).toBe("/admin/audit?action=a+b%26c%3Dd&page=1");
  });

  it("returns the page the client returned", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await expect(audit({ page: 1 })).resolves.toBe(PAGE);
  });
});

describe("adminApi.systemLatency — the period is the query", () => {
  it("asks for the period it is given", async () => {
    vi.spyOn(api, "request").mockResolvedValue(LATENCY);
    await expect(systemLatency(7)).resolves.toBe(LATENCY);
    expect(lastCall()).toEqual({ path: "/admin/system/latency?days=7", init: undefined });
  });

  it("defaults to the last day", async () => {
    vi.spyOn(api, "request").mockResolvedValue([]);
    await systemLatency();
    expect(lastCall().path).toBe("/admin/system/latency?days=1");
  });
});

describe("adminApi.updatePlan", () => {
  it("PATCHes the plan with the limits as the body and returns the plan", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PLAN);
    await expect(updatePlan("free", PLAN.limits)).resolves.toBe(PLAN);
    const { path, init } = lastCall();
    expect(path).toBe("/admin/plans/free");
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(String(init?.body))).toEqual({ limits: PLAN.limits });
  });

  it("escapes the plan's name in the path", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PLAN);
    await updatePlan("pro trial", PLAN.limits);
    expect(lastCall().path).toBe("/admin/plans/pro%20trial");
  });
});
