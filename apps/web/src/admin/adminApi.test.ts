import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import {
  audit,
  errorBundle,
  errorDetail,
  errors,
  installsSummary,
  overview,
  plans,
  system,
  systemLatency,
  updateErrorStatus,
  updatePlan,
  updateUser,
  userDetail,
  users,
  deleteUser,
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

describe("adminApi — accounts", () => {
  const USER_ID = "6e0f2f4a-1f7c-4b4e-9a4a-0c7d2f9b1c11";
  const USER = {
    id: USER_ID,
    email: "meera@example.test",
    name: "Meera Nair",
    role: "member",
    plan: "pro",
    plan_override: "pro",
    is_active: true,
    email_verified: true,
    created_at: "2026-11-20T10:00:00+05:30",
    last_login_at: null,
    last_active_on: "2026-11-22",
    usage: { datasets: 3, dataset_bytes: 4000, fits: 2, autofits_today: 1 },
  };

  it("users sends only the page when nothing is filtered", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await expect(users({ page: 1 })).resolves.toBe(PAGE);
    expect(lastCall()).toEqual({ path: "/admin/users?page=1", init: undefined });
  });

  it("users sends every filter under the API's names, in order, escaping the search", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await users({ q: "meera+1@example.test", plan: "pro", role: "admin", active: true, page: 3, pageSize: 50 });
    expect(lastCall().path).toBe(
      "/admin/users?q=meera%2B1%40example.test&plan=pro&role=admin&active=true&page=3&page_size=50",
    );
  });

  it("users keeps an inactive filter, which is false and not empty", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await users({ active: false, page: 1 });
    expect(lastCall().path).toBe("/admin/users?active=false&page=1");
  });

  it("users leaves out an empty search", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await users({ q: "  ", plan: undefined, page: 1 });
    expect(lastCall().path).toBe("/admin/users?page=1");
  });

  it("userDetail fetches GET /admin/users/{id}", async () => {
    const detail = { user: USER, audit: [] };
    vi.spyOn(api, "request").mockResolvedValue(detail);
    await expect(userDetail(USER_ID)).resolves.toBe(detail);
    expect(lastCall()).toEqual({ path: `/admin/users/${USER_ID}`, init: undefined });
  });

  it("updateUser PATCHes exactly the change it is given and returns the account", async () => {
    vi.spyOn(api, "request").mockResolvedValue(USER);
    await expect(updateUser(USER_ID, { role: "admin" })).resolves.toBe(USER);
    const { path, init } = lastCall();
    expect(path).toBe(`/admin/users/${USER_ID}`);
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(String(init?.body))).toEqual({ role: "admin" });
  });

  it("updateUser sends a null plan_override as null, which clears it", async () => {
    vi.spyOn(api, "request").mockResolvedValue(USER);
    await updateUser(USER_ID, { plan_override: null });
    expect(JSON.parse(String(lastCall().init?.body))).toEqual({ plan_override: null });
  });

  it("deleteUser sends a DELETE with no body and resolves on the empty 204", async () => {
    vi.spyOn(api, "request").mockResolvedValue(undefined);
    await expect(deleteUser(USER_ID)).resolves.toBeUndefined();
    const { path, init } = lastCall();
    expect(path).toBe(`/admin/users/${USER_ID}`);
    expect(init?.method).toBe("DELETE");
    expect(init?.body).toBeUndefined();
  });
});

describe("adminApi — error events", () => {
  const EVENT_ID = "0b1f0d3e-9a6c-4b2e-8f1a-2c3d4e5f6a7b";
  const EVENT = {
    id: EVENT_ID,
    source: "client",
    category: "crash",
    severity: "fatal",
    error_type: "TypeError",
    message: "Cannot read properties of undefined",
    fingerprint: "3f9a1c2b7d0e",
    route: "/fits/new",
    method: null,
    status_code: null,
    request_id: null,
    app_version: "0.11.1",
    status: "new",
    occurrence_count: 4,
    first_seen_at: "2026-11-20T10:00:00+05:30",
    last_seen_at: "2026-11-23T14:05:00+05:30",
  };

  it("errors sends the status and the page, nothing else", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await expect(errors({ status: "acknowledged", page: 1 })).resolves.toBe(PAGE);
    expect(lastCall()).toEqual({ path: "/admin/errors?status=acknowledged&page=1", init: undefined });
  });

  it("errors sends the category, the page and the page size under the API's names", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await errors({ status: "new", category: "network", page: 2, pageSize: 25 });
    expect(lastCall().path).toBe("/admin/errors?status=new&category=network&page=2&page_size=25");
  });

  it("errors leaves the category out for all of them", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    await errors({ status: "ignored", page: 1 });
    expect(lastCall().path).toBe("/admin/errors?status=ignored&page=1");
  });

  it("errorDetail fetches GET /admin/errors/{id}", async () => {
    const detail = { ...EVENT, stack: "TypeError: …", context_json: null, console_log: null, user_agent: null };
    vi.spyOn(api, "request").mockResolvedValue(detail);
    await expect(errorDetail(EVENT_ID)).resolves.toBe(detail);
    expect(lastCall()).toEqual({ path: `/admin/errors/${EVENT_ID}`, init: undefined });
  });

  it("updateErrorStatus PATCHes {status} and returns the event", async () => {
    vi.spyOn(api, "request").mockResolvedValue(EVENT);
    await expect(updateErrorStatus(EVENT_ID, "resolved")).resolves.toBe(EVENT);
    const { path, init } = lastCall();
    expect(path).toBe(`/admin/errors/${EVENT_ID}`);
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(String(init?.body))).toEqual({ status: "resolved" });
  });

  it("errorBundle fetches the document as text through api.text, never through request", async () => {
    const request = vi.spyOn(api, "request").mockResolvedValue(PAGE);
    const text = vi.spyOn(api, "text").mockResolvedValue("# Error event");
    await expect(errorBundle(EVENT_ID)).resolves.toBe("# Error event");
    expect(text).toHaveBeenCalledTimes(1);
    expect(text).toHaveBeenCalledWith(`/admin/errors/${EVENT_ID}/bundle`);
    expect(request).not.toHaveBeenCalled();
  });

  it("escapes an id in the path", async () => {
    vi.spyOn(api, "request").mockResolvedValue(PAGE);
    vi.spyOn(api, "text").mockResolvedValue("");
    await userDetail("a b");
    await errorDetail("c d");
    await errorBundle("e f");
    expect(vi.mocked(api.request).mock.calls.map(([path]) => path)).toEqual([
      "/admin/users/a%20b",
      "/admin/errors/c%20d",
    ]);
    expect(vi.mocked(api.text).mock.calls[0]?.[0]).toBe("/admin/errors/e%20f/bundle");
  });
});
