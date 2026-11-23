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

describe("the profile endpoints (the /me contract)", () => {
  const ME = {
    id: "u1",
    email: "a@b.co",
    name: "A",
    created_at: "2026-11-17T05:00:00+05:30",
    email_verified: true,
    role: "member",
    preferences: { error_reports: true, report_byline: false },
    last_login_at: null,
    plan: {
      name: "Free",
      max_datasets: 10,
      max_dataset_bytes: 20_000_000,
      max_fits: 100,
      max_autofits_per_day: 5,
    },
    usage: { datasets: 1, dataset_bytes: 2000, fits: 2, autofits_today: 0 },
  };

  /** The method and parsed JSON body of the one request that reached `path`. */
  function sentTo(path: string): { url: string; method: string; body: unknown; auth: string | null } {
    const calls = fetchMock.mock.calls.filter(([url]) => String(url) === `${API_BASE}${path}`);
    expect(calls).toHaveLength(1);
    const [url, init] = calls[0] as [RequestInfo | URL, RequestInit | undefined];
    return {
      url: String(url),
      method: init?.method ?? "GET",
      body: typeof init?.body === "string" ? JSON.parse(init.body) : null,
      auth: bearer(init),
    };
  }

  it("me() parses MeOut", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, ME));
    await expect(api.me()).resolves.toEqual(ME);
  });

  it("refreshToken is the stored one, or null without a session", () => {
    expect(api.refreshToken).toBeNull();
    api.setTokens("a1", "r1");
    expect(api.refreshToken).toBe("r1");
  });

  it("updateMe PATCHes /me with the name, authenticated", async () => {
    api.setTokens("a1", "r1");
    fetchMock.mockResolvedValue(jsonResponse(200, ME));
    await expect(api.updateMe("Amit")).resolves.toEqual(ME);
    const sent = sentTo("/me");
    expect(sent.method).toBe("PATCH");
    expect(sent.body).toEqual({ name: "Amit" });
    expect(sent.auth).toBe("Bearer a1");
  });

  it("updatePreferences PUTs exactly the keys it is given", async () => {
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(200, ME)));
    await api.updatePreferences({ report_byline: true });
    const one = sentTo("/me/preferences");
    expect(one.method).toBe("PUT");
    expect(one.body).toEqual({ report_byline: true });

    // a key that is not named is not sent at all, so the server keeps its value
    fetchMock.mockClear();
    await api.updatePreferences({ error_reports: false });
    expect(sentTo("/me/preferences").body).toEqual({ error_reports: false });
  });

  it("changePassword sends both passwords with the stored refresh token", async () => {
    api.setTokens("a1", "r1");
    fetchMock.mockResolvedValue(jsonResponse(200, ME));
    await expect(api.changePassword("old-pass-123", "new-pass-456")).resolves.toEqual(ME);
    const sent = sentTo("/me/password");
    expect(sent.method).toBe("POST");
    expect(sent.body).toEqual({
      current_password: "old-pass-123",
      new_password: "new-pass-456",
      refresh_token: "r1",
    });
    expect(sent.auth).toBe("Bearer a1");
  });

  it("revokeAllSessions sends the stored refresh token", async () => {
    api.setTokens("a1", "r1");
    fetchMock.mockResolvedValue(jsonResponse(200, ME));
    await expect(api.revokeAllSessions()).resolves.toEqual(ME);
    const sent = sentTo("/me/sessions/revoke-all");
    expect(sent.method).toBe("POST");
    expect(sent.body).toEqual({ refresh_token: "r1" });
  });

  it("deleteMe sends {confirm} on a DELETE and resolves on the empty 204", async () => {
    api.setTokens("a1", "r1");
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    await expect(api.deleteMe("amit")).resolves.toBeUndefined();
    const sent = sentTo("/me");
    expect(sent.method).toBe("DELETE");
    expect(sent.body).toEqual({ confirm: "amit" });
    expect(sent.auth).toBe("Bearer a1");
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

describe("onServerError (a 5xx is handed to the error reporter before the error is thrown)", () => {
  const DATASETS = `${API_BASE}/datasets`;

  /** A Response as fetch hands one back: carrying the final URL and the server's headers. */
  function serverAnswer(status: number, url: string, requestId: string | null): Response {
    const response = jsonResponse(status, {
      error: "INTERNAL_ERROR",
      message: "the server tripped",
      detail: {},
    });
    Object.defineProperty(response, "url", { value: url });
    if (requestId !== null) response.headers.set("x-request-id", requestId);
    return response;
  }

  /**
   * An XMLHttpRequest as the client drives it, answered by the test: jsdom's own would
   * put a real request on the wire.
   */
  class FakeXHR {
    status = 0;
    response = "";
    responseURL = "";
    upload: { onprogress: ((event: { lengthComputable: boolean }) => void) | null } = {
      onprogress: null,
    };
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    private headers: Record<string, string> = {};

    open(_method: string, url: string): void {
      this.responseURL = url;
    }
    setRequestHeader(): void {}
    send(): void {}
    abort(): void {}
    addEventListener(): void {}
    removeEventListener(): void {}
    getResponseHeader(name: string): string | null {
      return this.headers[name.toLowerCase()] ?? null;
    }
    reply(status: number, body: string, headers: Record<string, string>): void {
      this.status = status;
      this.response = body;
      this.headers = headers;
      this.onload?.();
    }
  }

  it.each([500, 503])("is called for a %i with the response's URL and request id", async (status) => {
    const onServerError = vi.fn();
    api.onServerError = onServerError;
    fetchMock.mockResolvedValue(serverAnswer(status, DATASETS, "req-7f3a"));
    await expect(api.request("/datasets")).rejects.toMatchObject({ status });
    expect(onServerError).toHaveBeenCalledTimes(1);
    expect(onServerError).toHaveBeenCalledWith(
      { status, url: DATASETS, requestId: "req-7f3a" },
      "/datasets",
    );
  });

  it("is called with a null request id when the server sent no header", async () => {
    const onServerError = vi.fn();
    api.onServerError = onServerError;
    fetchMock.mockResolvedValue(serverAnswer(500, DATASETS, null));
    await expect(api.request("/datasets")).rejects.toMatchObject({ status: 500 });
    expect(onServerError).toHaveBeenCalledWith(
      { status: 500, url: DATASETS, requestId: null },
      "/datasets",
    );
  });

  it.each([404, 422, 429])("is not called for a %i", async (status) => {
    const onServerError = vi.fn();
    api.onServerError = onServerError;
    fetchMock.mockResolvedValue(serverAnswer(status, DATASETS, "req-1"));
    await expect(api.request("/datasets")).rejects.toMatchObject({ status });
    expect(onServerError).not.toHaveBeenCalled();
  });

  it("leaves the error the caller receives unchanged", async () => {
    const answer = (): Response => serverAnswer(503, DATASETS, "req-1");
    fetchMock.mockImplementation(() => Promise.resolve(answer()));
    const withoutCallback = await api.request("/datasets").catch((e: unknown) => e);
    api.onServerError = vi.fn();
    const withCallback = await api.request("/datasets").catch((e: unknown) => e);
    expect(withCallback).toBeInstanceOf(ApiError);
    expect(withCallback).toEqual(withoutCallback);
    expect((withCallback as ApiError).message).toBe("the server tripped");
  });

  it("resetForTests clears it", () => {
    api.onServerError = vi.fn();
    api.resetForTests();
    expect(api.onServerError).toBeNull();
  });

  /** Route the client's uploads into a fake XHR; `next()` is the one it opened last. */
  function fakeUploads(): () => FakeXHR {
    const opened: FakeXHR[] = [];
    vi.stubGlobal(
      "XMLHttpRequest",
      class extends FakeXHR {
        constructor() {
          super();
          opened.push(this);
        }
      },
    );
    return (): FakeXHR => {
      const xhr = opened[0];
      expect(xhr, "the client opened one upload").toBeDefined();
      return xhr as FakeXHR;
    };
  }

  it("uploadDataset reports the same statuses with what the XMLHttpRequest has", async () => {
    const onServerError = vi.fn();
    api.onServerError = onServerError;
    const xhr = fakeUploads();
    const upload = api.uploadDataset(
      new File(["a,b\n1,2\n"], "data.csv", { type: "text/csv" }),
      {},
    );
    xhr().reply(
      503,
      JSON.stringify({ error: "UNAVAILABLE", message: "try later", detail: {} }),
      { "x-request-id": "req-up" },
    );
    await expect(upload).rejects.toMatchObject({ status: 503, code: "UNAVAILABLE" });
    expect(onServerError).toHaveBeenCalledTimes(1);
    expect(onServerError).toHaveBeenCalledWith(
      { status: 503, url: `${API_BASE}/datasets`, requestId: "req-up" },
      "/datasets",
    );
  });

  it("uploadDataset says nothing for a 413", async () => {
    const onServerError = vi.fn();
    api.onServerError = onServerError;
    const xhr = fakeUploads();
    const upload = api.uploadDataset(
      new File(["a,b\n1,2\n"], "data.csv", { type: "text/csv" }),
      {},
    );
    xhr().reply(413, JSON.stringify({ error: "TOO_LARGE", message: "too big", detail: {} }), {});
    await expect(upload).rejects.toMatchObject({ status: 413 });
    expect(onServerError).not.toHaveBeenCalled();
  });

  it("download reports the same statuses", async () => {
    const onServerError = vi.fn();
    api.onServerError = onServerError;
    const url = `${API_BASE}/fits/f1/export`;
    fetchMock.mockResolvedValue(serverAnswer(500, url, "req-dl"));
    await expect(api.download("/fits/f1/export")).rejects.toMatchObject({ status: 500 });
    expect(onServerError).toHaveBeenCalledWith(
      { status: 500, url, requestId: "req-dl" },
      "/fits/f1/export",
    );
  });
});
