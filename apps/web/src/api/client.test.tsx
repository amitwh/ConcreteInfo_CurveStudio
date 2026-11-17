import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { API_BASE, ApiError, api } from "./client";

/** The fetch mock: replace `globalThis.fetch` per test. */
let fetchMock: ReturnType<typeof vi.fn>;

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function bearer(init?: RequestInit): string | null {
  const headers = init?.headers;
  if (headers instanceof Headers) return headers.get("Authorization");
  return (headers as Record<string, string> | undefined)?.Authorization ?? null;
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  api.resetForTests();
});

describe("error envelope parsing", () => {
  it("turns the API envelope into an ApiError", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(404, {
        error: "NOT_FOUND",
        message: "dataset not found",
        detail: { code: "NOT_FOUND" },
      }),
    );
    const err = await api.request("/datasets/xyz").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.status).toBe(404);
    expect(apiErr.code).toBe("NOT_FOUND");
    expect(apiErr.message).toBe("dataset not found");
    expect(apiErr.detail).toEqual({ code: "NOT_FOUND" });
  });

  it("normalises FastAPI 422 validation bodies to VALIDATION_ERROR", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(422, {
        detail: [
          { loc: ["body", "x"], msg: "field required", type: "missing" },
        ],
      }),
    );
    const err = await api
      .request<{ never: never }>("/datasets", { method: "POST" })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.status).toBe(422);
    expect(apiErr.code).toBe("VALIDATION_ERROR");
    expect(apiErr.message).toContain("field required");
  });

  it("reports a network failure as a NETWORK ApiError without inventing a status", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    const err = await api.request("/datasets").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const apiErr = err as ApiError;
    expect(apiErr.status).toBe(0);
    expect(apiErr.code).toBe("NETWORK");
  });
});

describe("401 refresh handling", () => {
  it("refreshes once for parallel 401s (single flight) and retries both", async () => {
    api.setTokens("a1", "r1");
    const refreshBodies: string[] = [];
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/auth/refresh")) {
        refreshBodies.push(String(init?.body));
        return jsonResponse(200, { access_token: "a2", refresh_token: "r2" });
      }
      if (bearer(init) === "Bearer a2") return jsonResponse(200, []);
      return jsonResponse(401, { error: "UNAUTHORIZED", message: "expired", detail: {} });
    });

    const [first, second] = await Promise.all([
      api.request("/datasets"),
      api.request("/models"),
    ]);
    expect(first).toEqual([]);
    expect(second).toEqual([]);
    expect(refreshBodies).toEqual(['{"refresh_token":"r1"}']);
    // rotation: the NEW refresh token is the one stored
    expect(localStorage.getItem("curvestudio.refresh")).toBe("r2");
  });

  it("keeps the rotated token for later requests", async () => {
    api.setTokens("a1", "r1");
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/auth/refresh")) {
        return jsonResponse(200, { access_token: "a2", refresh_token: "r2" });
      }
      if (bearer(init) === "Bearer a2") return jsonResponse(200, [{ id: "m1" }]);
      return jsonResponse(401, { error: "UNAUTHORIZED", message: "expired", detail: {} });
    });
    await api.request("/models");
    const again = await api.request("/models");
    expect(again).toEqual([{ id: "m1" }]);
    // exactly one refresh across both requests: second request uses the live a2
    const refreshCalls = fetchMock.mock.calls.filter(([u]) =>
      String(u).endsWith("/auth/refresh"),
    );
    expect(refreshCalls).toHaveLength(1);
  });

  it("logs out when the refresh token is rejected", async () => {
    api.setTokens("a1", "r1");
    const onAuthFailure = vi.fn();
    api.onAuthFailure = onAuthFailure;
    fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/auth/refresh")) {
        return jsonResponse(401, { error: "UNAUTHORIZED", message: "revoked", detail: {} });
      }
      return jsonResponse(401, { error: "UNAUTHORIZED", message: "expired", detail: {} });
    });
    const err = await api.request("/datasets").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(401);
    expect(onAuthFailure).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem("curvestudio.refresh")).toBeNull();
  });

  it("does not try to refresh the refresh endpoint itself", async () => {
    api.setTokens("a1", "r1");
    fetchMock.mockResolvedValue(
      jsonResponse(401, { error: "UNAUTHORIZED", message: "revoked", detail: {} }),
    );
    const err = await api.request("/auth/refresh", { method: "POST" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    const calls = fetchMock.mock.calls.map(([u]) => String(u));
    expect(calls).toEqual([`${API_BASE}/auth/refresh`]);
  });
});

describe("login and register store tokens (the register → login → me flow)", () => {
  /**
   * The mock mirrors the server's auth contract, not just happy-path responses:
   * `/me` rejects with the 401 envelope unless the request carries the live bearer
   * token. A login that fails to store its tokens therefore cannot pass — the very
   * next request 401s, exactly as it did in the browser.
   */
  function authServer(): void {
    const user = { id: "u1", email: "a@b.co", name: "", created_at: "2026-11-17T05:00:00Z" };
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/auth/register")) return jsonResponse(201, user);
      if (url.endsWith("/auth/login"))
        return jsonResponse(200, {
          access_token: "a1",
          refresh_token: "r1",
          expires_in: 900,
          token_type: "bearer",
        });
      if (url.endsWith("/me")) {
        return bearer(init) === "Bearer a1"
          ? jsonResponse(200, user)
          : jsonResponse(401, { error: "UNAUTHORIZED", message: "not authenticated", detail: {} });
      }
      return jsonResponse(404, { error: "NOT_FOUND", message: "no such route", detail: {} });
    });
  }

  it("stores access and refresh tokens on login; the next request carries the bearer", async () => {
    authServer();
    await api.login("a@b.co", "longenoughpass");
    expect(localStorage.getItem("curvestudio.refresh")).toBe("r1");
    const me = await api.me();
    expect(me.email).toBe("a@b.co");
    // One /me call, answered 200: no 401 → no refresh detour means the bearer was attached.
    const meCalls = fetchMock.mock.calls.filter(([u]) => String(u).endsWith("/me"));
    expect(meCalls).toHaveLength(1);
  });

  it("ends the register flow authenticated (register, then the auto-login stores tokens)", async () => {
    authServer();
    await api.register("a@b.co", "longenoughpass", "A");
    await api.login("a@b.co", "longenoughpass");
    const me = await api.me();
    expect(me.email).toBe("a@b.co");
    const meCalls = fetchMock.mock.calls.filter(([u]) => String(u).endsWith("/me"));
    expect(meCalls).toHaveLength(1);
  });

  it("cannot reach /me without a preceding login (the server rejects it with 401)", async () => {
    authServer();
    const err = await api.me().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(401);
  });
});

describe("abort support", () => {
  it("passes the signal through and rethrows abort errors unchanged", async () => {
    const controller = new AbortController();
    controller.abort();
    fetchMock.mockImplementation((_input: RequestInfo | URL, init?: RequestInit) => {
      if (init?.signal?.aborted) {
        return Promise.reject(new DOMException("aborted", "AbortError"));
      }
      return Promise.resolve(jsonResponse(200, []));
    });
    const err = await api
      .request("/datasets", { signal: controller.signal })
      .catch((e: unknown) => e);
    expect(err).not.toBeInstanceOf(ApiError);
    expect((err as DOMException).name).toBe("AbortError");
  });
});
