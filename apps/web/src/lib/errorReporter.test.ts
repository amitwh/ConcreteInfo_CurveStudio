import { createHash } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import pkg from "../../package.json";
import {
  errorReportsEnabled,
  eventFromError,
  installErrorReporter,
  reportApiError,
  reportUiError,
  scrubSecrets,
  scrubText,
  sendDescription,
  setConsentResolver,
  setErrorReportsEnabled,
  topFrame,
  type ErrorReportEvent,
} from "./errorReporter";

const ORIGIN = "http://localhost:3000";
const API = `${ORIGIN}/api/v1`;
const ENDPOINT = `${API}/errors`;
const JWT =
  "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhYmMifQ.c2lnbmF0dXJlLXZhbHVl";
const STACK = [
  "TypeError: boom",
  "    at render (http://localhost:3000/assets/index-abc.js:1:234)",
  "    at App (http://localhost:3000/assets/index-abc.js:1:20)",
].join("\n");

/** SHA-256 hex, as the server computes the fingerprint. */
function sha(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

/** The fingerprint inputs the server hashes for `boom()` reported by the two paths. */
const UI_BOOM = sha("ui|TypeError|boom|index-abc.js:1|/").slice(0, 12);
const CRASH_BOOM = sha("crash|TypeError|boom|index-abc.js:1|/").slice(0, 12);

/** The prefix the reporter stamps on a console line at the faked clock (14:03:11 IST). */
const STAMP = "[14:03:11] [ERROR] ";
/** 19 characters of stamp and 479 of text: eight such lines are 3,991 characters, so a buffer of them fills the 4,000 cap almost exactly. */
const FULL_LINE = "c".repeat(479);

function byteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

let fetchMock: Mock<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>;
let beaconMock: Mock<(url: string | URL, data?: BodyInit | null) => boolean>;
/** console.error and console.warn as silent spies; the reporter wraps them, so these are what the test reads. */
let errorSpy: Mock;
let warnSpy: Mock;
let uninstall: () => void;

interface Request {
  url: string;
  body: string;
  events: ErrorReportEvent[];
  keepalive: boolean | undefined;
}

/** Every POST the reporter made, parsed. */
function requests(): Request[] {
  return fetchMock.mock.calls.map(([url, init]) => {
    const body = String(init?.body);
    return {
      url: String(url),
      body,
      events: (JSON.parse(body) as { events: ErrorReportEvent[] }).events,
      keepalive: init?.keepalive,
    };
  });
}

/** The one event of the one request the reporter has made so far. */
function sentEvent(): ErrorReportEvent {
  const [request] = requests();
  expect(request, "exactly one request so far").toBeDefined();
  expect(request?.events).toHaveLength(1);
  return request?.events[0] as ErrorReportEvent;
}

async function beaconEvents(): Promise<ErrorReportEvent[]> {
  expect(beaconMock).toHaveBeenCalledTimes(1);
  const [, data] = beaconMock.mock.calls[0] as [string | URL, Blob];
  return (JSON.parse(await data.text()) as { events: ErrorReportEvent[] }).events;
}

/** A flush that lets its promises settle and its timers fire. */
async function settle(ms = 0): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
}

/** A TypeError with a real-looking stack; `n` keeps events distinct. */
function boom(n = ""): TypeError {
  const error = new TypeError(`boom${n}`);
  error.stack = STACK.replace("boom", `boom${n}`);
  return error;
}

function scriptError(error: Error = boom(), message = error.message): void {
  window.dispatchEvent(new ErrorEvent("error", { error, message }));
}

function rejection(reason: unknown): void {
  window.dispatchEvent(
    new PromiseRejectionEvent("unhandledrejection", { promise: Promise.resolve(), reason }),
  );
}

function pageHide(): void {
  window.dispatchEvent(new Event("pagehide"));
}

/** A fetch that stays pending until the test answers it: it is the reporter's `fetch` from here on. */
function pendingFetch(): {
  fetch: Mock;
  resolve: (response: Response) => void;
} {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((done) => {
    resolve = done;
  });
  const fetch = vi.fn(() => promise);
  vi.stubGlobal("fetch", fetch);
  return { fetch, resolve };
}

function answer(status: number): Response {
  return new Response(null, { status });
}

/** A fetch that cannot reach the server at all; it is the reporter's `fetch` from here on. */
function failingFetch(): Mock {
  const fetch = vi.fn(() => Promise.reject(new TypeError("fetch failed")));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

beforeEach(() => {
  // Only what the reporter uses, so the real digest (a macrotask) can still settle.
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
  vi.setSystemTime(new Date("2026-11-23T08:33:11Z")); // 14:03:11 IST
  fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>();
  fetchMock.mockImplementation(() => Promise.resolve(answer(204)));
  vi.stubGlobal("fetch", fetchMock);
  beaconMock = vi.fn(() => true);
  Object.defineProperty(navigator, "sendBeacon", { configurable: true, value: beaconMock });
  warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
  errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  uninstall = installErrorReporter({ apiBase: API });
});

afterEach(() => {
  uninstall();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  delete (navigator as { sendBeacon?: unknown }).sendBeacon;
  window.history.replaceState(null, "", "/");
});

describe("consent", () => {
  it("is on for a visitor until switched off, and the choice persists", () => {
    expect(errorReportsEnabled()).toBe(true);
    setErrorReportsEnabled(false);
    expect(errorReportsEnabled()).toBe(false);
    expect(localStorage.getItem("curvestudio.errorReports")).toBe("off");
    setErrorReportsEnabled(true);
    expect(errorReportsEnabled()).toBe(true);
    expect(localStorage.getItem("curvestudio.errorReports")).toBeNull();
  });

  it("with consent off nothing is collected and nothing is sent", async () => {
    setErrorReportsEnabled(false);
    scriptError();
    rejection(new Error("late"));
    console.error("logged while off");
    reportApiError({ status: 503, url: `${API}/datasets`, requestId: null }, "/datasets");
    await expect(reportUiError(boom(), { componentStack: null })).resolves.toBe("");
    await expect(sendDescription("abc123def456", "a note")).resolves.toBe(false);
    await settle(20_000);
    pageHide();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(beaconMock).not.toHaveBeenCalled();

    // what was said while off was never held: a later report does not carry it
    setErrorReportsEnabled(true);
    scriptError();
    await settle(10_000);
    expect(sentEvent().console_log).toBe("");
  });

  it("a signed-in resolver overrides the visitor's setting, either way", async () => {
    setErrorReportsEnabled(false);
    setConsentResolver(() => true);
    scriptError();
    await settle(10_000);
    expect(requests()).toHaveLength(1);

    fetchMock.mockClear();
    setErrorReportsEnabled(true);
    setConsentResolver(() => false);
    scriptError(boom("2"));
    await settle(10_000);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("a resolver that returns null hands the decision back to the visitor's setting", async () => {
    setConsentResolver(() => null);
    setErrorReportsEnabled(false);
    scriptError();
    await settle(10_000);
    expect(fetchMock).not.toHaveBeenCalled();

    setErrorReportsEnabled(true);
    scriptError(boom("2"));
    await settle(10_000);
    expect(requests()).toHaveLength(1);
  });
});

describe("hooks", () => {
  it("a script error is one crash event built from the Error", async () => {
    scriptError();
    await settle(10_000);
    const event = sentEvent();
    expect(requests()[0]?.url).toBe(ENDPOINT);
    expect(event).toMatchObject({
      category: "crash",
      severity: "error",
      error_type: "TypeError",
      message: "boom",
      stack: STACK,
      route: "/",
      user_agent: navigator.userAgent,
      app_version: pkg.version,
      console_log: "",
      context: { path: "/", viewport: { width: window.innerWidth, height: window.innerHeight }, online: navigator.onLine },
    });

    // the category is part of the fingerprint: the same Error caught by the boundary is
    // another event, with another reference
    const reference = await reportUiError(boom(), { componentStack: null });
    expect(reference).toBe(UI_BOOM);
    expect(reference).not.toBe(CRASH_BOOM);
    await settle(10_000);
    expect(requests().map((request) => request.events[0]?.category)).toEqual(["crash", "ui"]);
  });

  it("a rejected promise that carries an Error is a crash event", async () => {
    rejection(new RangeError("bad range"));
    await settle(10_000);
    expect(sentEvent()).toMatchObject({
      category: "crash",
      severity: "error",
      error_type: "RangeError",
      message: "bad range",
    });
  });

  it("a rejection whose reason is not an Error is described, not dropped", async () => {
    rejection("a plain string reason");
    await settle(10_000);
    expect(sentEvent()).toMatchObject({ category: "crash", error_type: "NonError", message: "a plain string reason" });

    fetchMock.mockClear();
    rejection({ code: "E_AGAIN", message: "it broke" });
    await settle(10_000);
    expect(sentEvent()).toMatchObject({ category: "crash", message: "it broke" });
  });

  it("a script error with no Error object is built from what the event carries", async () => {
    window.dispatchEvent(
      new ErrorEvent("error", { message: "Script error.", filename: "http://localhost:3000/x.js", lineno: 7, colno: 3 }),
    );
    await settle(10_000);
    const event = sentEvent();
    expect(event).toMatchObject({ category: "crash", message: "Script error." });
    expect(event?.stack).toContain("x.js:7:3");
  });
});

describe("one page counts each error once", () => {
  it("the same error twice on one page is sent once", async () => {
    scriptError();
    scriptError();
    await settle(10_000);
    expect(requests()).toHaveLength(1);
    expect(requests()[0]?.events).toHaveLength(1);
  });

  it("a different message is a different error", async () => {
    scriptError(boom());
    scriptError(boom("2"));
    await settle(10_000);
    expect(requests()).toHaveLength(1);
    expect(requests()[0]?.events.map((event) => event.message)).toEqual(["boom", "boom2"]);
  });
});

describe("queue and flush", () => {
  it("a flush starts as soon as five are queued", async () => {
    for (let n = 1; n <= 4; n++) scriptError(boom(String(n)));
    expect(fetchMock).not.toHaveBeenCalled();
    scriptError(boom("5"));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(requests()[0]?.events).toHaveLength(5);
    expect(requests()[0]?.keepalive).toBe(true);
  });

  it("otherwise ten seconds after the first", async () => {
    scriptError();
    await settle(9_999);
    expect(fetchMock).not.toHaveBeenCalled();
    await settle(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("the clock restarts for what is left after a flush", async () => {
    for (let n = 1; n <= 6; n++) scriptError(boom(String(n)));
    expect(requests()).toHaveLength(1);
    await settle(9_999);
    expect(requests()).toHaveLength(1);
    await settle(1);
    expect(requests()).toHaveLength(2);
    expect(requests()[1]?.events.map((event) => event.message)).toEqual(["boom6"]);
  });

  it("one flush runs at a time", async () => {
    const { fetch, resolve } = pendingFetch();
    for (let n = 1; n <= 10; n++) scriptError(boom(String(n)));
    expect(fetch).toHaveBeenCalledTimes(1);
    resolve(answer(204));
    await settle();
    expect(fetch).toHaveBeenCalledTimes(2);
    const bodies = fetch.mock.calls.map(([, init]) => JSON.parse(String(init?.body)) as { events: unknown[] });
    expect(bodies.map((body) => body.events.length)).toEqual([5, 5]);
  });

  it("the queue never holds more than twenty; the oldest is dropped", async () => {
    pendingFetch();
    for (let n = 1; n <= 30; n++) scriptError(boom(String(n)));
    // 1–5 are in the flush; 6–10 make way for the newest.
    pageHide();
    const messages = (await beaconEvents()).map((event) => event.message);
    expect(messages).toHaveLength(20);
    expect(messages[0]).toBe("boom11");
    expect(messages[19]).toBe("boom30");
  });

  it("a batch stays under the server's 64 KB", async () => {
    // Five events at the caps (2,000 + 6,000 + about 4,000 characters) are about 61 KB
    // together: four go in the first request, the fifth follows.
    for (let n = 1; n <= 5; n++) {
      const error = new Error(`m${n}`.padEnd(2_000, "m"));
      error.stack = STACK.replace("boom", error.message).padEnd(20_000, "s");
      for (let line = 0; line < 20; line++) console.error(FULL_LINE);
      scriptError(error);
    }
    expect(requests()).toHaveLength(1);
    const first = requests()[0] as Request;
    expect(first.events).toHaveLength(4);
    expect(byteLength(first.body)).toBeLessThan(64_000);
    await settle(10_000);
    expect(requests()).toHaveLength(2);
    expect(requests()[1]?.events).toHaveLength(1);
    expect(byteLength(requests()[1]?.body ?? "")).toBeLessThan(64_000);
  });

  it("pagehide sends what is queued with sendBeacon and keeps nothing back", async () => {
    scriptError();
    scriptError(boom("2"));
    pageHide();
    const events = await beaconEvents();
    expect(beaconMock.mock.calls[0]?.[0]).toBe(ENDPOINT);
    expect(events.map((event) => event.message)).toEqual(["boom", "boom2"]);
    await settle(20_000);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("a POST that is refused or fails", () => {
  it("a network error drops its batch: nothing is retried and nothing is enqueued", async () => {
    const fetch = failingFetch();
    for (let n = 1; n <= 5; n++) scriptError(boom(String(n)));
    await settle(60_000);
    expect(fetch).toHaveBeenCalledTimes(1);
    pageHide();
    expect(beaconMock).not.toHaveBeenCalled();
  });

  it.each([429, 413, 422])("a %i drops its batch: nothing is retried and nothing is enqueued", async (status) => {
    fetchMock.mockImplementation(() => Promise.resolve(answer(status)));
    for (let n = 1; n <= 5; n++) scriptError(boom(String(n)));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await settle(60_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    pageHide();
    expect(beaconMock).not.toHaveBeenCalled();
  });

  it("the reporter never logs, and never reports its own failure", async () => {
    const fetch = failingFetch();
    scriptError();
    await settle(60_000);
    // the failed POST did not become an unhandled rejection, so no second event, no second POST
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(errorSpy).toHaveBeenCalledTimes(0);
    expect(warnSpy).toHaveBeenCalledTimes(0);
  });
});

describe("reportApiError", () => {
  const datasets = `${API}/datasets`;

  it("reports a same-origin 5xx that carries no X-Request-Id", async () => {
    reportApiError({ status: 503, url: datasets, requestId: null }, "/datasets");
    await settle(10_000);
    const event = sentEvent();
    expect(event).toMatchObject({
      category: "network",
      severity: "error",
      message: "the server answered 503 for /datasets",
      context: { path: "/datasets", status: 503 },
    });
    expect(event).not.toHaveProperty("request_id");
  });

  it("skips a same-origin 5xx that carries X-Request-Id: the server holds that record", async () => {
    reportApiError({ status: 500, url: datasets, requestId: "req-1" }, "/datasets");
    await settle(20_000);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports a cross-origin 5xx even with the header, and passes the id along", async () => {
    reportApiError(
      { status: 502, url: "https://api.example.com/api/v1/datasets", requestId: "req-123" },
      "/datasets",
    );
    await settle(10_000);
    expect(sentEvent()).toMatchObject({
      category: "network",
      request_id: "req-123",
      message: "the server answered 502 for /datasets",
    });
  });

  it.each([404, 429, 422])("does not report a %i", async (status) => {
    reportApiError({ status, url: datasets, requestId: null }, "/datasets");
    await settle(20_000);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("names the API path without its query and without its ids", async () => {
    reportApiError({ status: 500, url: `${datasets}/3f2b8a1e-0c4d-4e6f-8a9b-1c2d3e4f5a6b/rows?limit=5`, requestId: null }, "/datasets/3f2b8a1e-0c4d-4e6f-8a9b-1c2d3e4f5a6b/rows?limit=5");
    await settle(10_000);
    expect(sentEvent().context).toMatchObject({ path: "/datasets/{id}/rows" });
  });
});

describe("reportUiError and sendDescription", () => {
  it("reportUiError is a fatal ui event with the component stack in the context, and answers with the reference", async () => {
    const componentStack = "\n    at Child (http://localhost:3000/assets/index-abc.js:1:9)\n    at App";
    const error = new TypeError("render failed");
    error.stack = "TypeError: render failed"; // no frames: the reference below is then fixed
    const reference = await reportUiError(error, { componentStack });
    expect(reference).toBe(sha("ui|TypeError|render failed||/").slice(0, 12));
    await settle(10_000);
    expect(sentEvent()).toMatchObject({
      category: "ui",
      severity: "fatal",
      error_type: "TypeError",
      message: "render failed",
      context: { componentStack },
    });
  });

  it("the same error reported twice is one event with one reference", async () => {
    const first = await reportUiError(boom(), { componentStack: null });
    const second = await reportUiError(boom(), { componentStack: null });
    expect(first).toBe(UI_BOOM);
    expect(second).toBe(first);
    await settle(10_000);
    expect(requests()).toHaveLength(1);
    expect(requests()[0]?.events).toHaveLength(1);
  });

  it("sendDescription is its own event, carries the reference and bypasses the dedupe", async () => {
    await expect(sendDescription("abc123def456", "I clicked Save")).resolves.toBe(true);
    await expect(sendDescription("abc123def456", "I clicked Save")).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(requests()[1]?.events[0]).toMatchObject({
      category: "ui",
      severity: "error",
      error_type: "UserDescription",
      message: "I clicked Save",
      context: { reference: "abc123def456" },
    });
    expect(requests()[1]?.events[0]).not.toHaveProperty("description");
  });

  it("sendDescription sends nothing for a blank note, and says so when the server refuses", async () => {
    await expect(sendDescription("abc123def456", "   ")).resolves.toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
    fetchMock.mockImplementation(() => Promise.resolve(answer(429)));
    await expect(sendDescription("abc123def456", "a note")).resolves.toBe(false);
  });
});

describe("what leaves the browser", () => {
  it("carries no query string and no account identifier, and shows a UUID in the path as {id}", async () => {
    const uuid = "3f2b8a1e-0c4d-4e6f-8a9b-1c2d3e4f5a6b";
    window.history.replaceState(null, "", `/datasets/${uuid}/edit?token=secret-token#section`);
    scriptError();
    await settle(10_000);
    const request = requests()[0] as Request;
    const event = request.events[0] as ErrorReportEvent;
    expect(event.route).toBe("/datasets/{id}/edit");
    expect(event.context).toMatchObject({ path: "/datasets/{id}/edit" });
    expect(Object.keys(event).sort()).toEqual(
      ["app_version", "category", "console_log", "context", "error_type", "message", "route", "severity", "stack", "user_agent"],
    );
    expect(request.body).not.toContain("secret-token");
    expect(request.body).not.toContain("token=");
    expect(request.body).not.toContain(uuid);
    expect(request.body).not.toContain("@");
  });

  it("a JWT and an Authorization credential planted in a message, a stack and a console line come out as the server would keep them", async () => {
    const bearer = "Authorization: Bearer opaque.K3y-value_creds";
    const error = new Error(`${bearer} was refused; token=${JWT}`);
    error.stack = `Error: x\n    at f (http://localhost:3000/x.js:1:2)\n    token=${JWT}`;
    console.error("refresh failed for", bearer, JWT);
    scriptError(error);
    await settle(10_000);
    const event = sentEvent();
    const body = requests()[0]?.body ?? "";
    for (const secret of ["opaque.K3y-value_creds", JWT, "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"]) {
      expect(body).not.toContain(secret);
    }
    expect(event.message).toBe("Authorization: Bearer [REDACTED] was refused; token=[REDACTED]");
    expect(event.stack).toBe("Error: x\n    at f (http://localhost:3000/x.js:1:2)\n    token=[REDACTED]");
    expect(event.console_log).toBe(
      "[14:03:11] [ERROR] refresh failed for Authorization: Bearer [REDACTED] [REDACTED_JWT]",
    );
  });

  it("a named secret and an address planted in a message, a stack, a console line and a context value never leave", async () => {
    console.error("login as bob@example.org with password=hunter2");
    const error = new Error("sign-in failed for alice@example.org: secret=squirrel");
    error.stack = "Error: sign-in failed\n    at submit (app.js:3:4)\n    api_key=k3y-4live";
    await reportUiError(error, { componentStack: 'at Form (app.js:9:1) for "token": "t0k3n-value"' });
    await settle(10_000);
    const event = sentEvent();
    const body = requests()[0]?.body ?? "";
    for (const planted of ["hunter2", "squirrel", "k3y-4live", "t0k3n-value", "bob@example.org", "alice@example.org"]) {
      expect(body, planted).not.toContain(planted);
    }
    expect(event.message).toBe("sign-in failed for [REDACTED_EMAIL]: secret=[REDACTED]");
    expect(event.stack).toBe("Error: sign-in failed\n    at submit (app.js:3:4)\n    api_key=[REDACTED]");
    expect(event.console_log).toBe("[14:03:11] [ERROR] login as [REDACTED_EMAIL] with password=[REDACTED]");
    expect(event.context).toMatchObject({ componentStack: 'at Form (app.js:9:1) for "token": "[REDACTED]"' });
  });

  it("the reference is computed on the scrubbed text, so it is the server's", async () => {
    const error = new Error("login failed: password=hunter2 for user 7");
    error.stack = "Error: login failed";
    const reference = await reportUiError(error, { componentStack: null });
    expect(reference).toBe(sha("ui|Error|login failed: password=[REDACTED] for user 7||/").slice(0, 12));
  });

  it("cuts the stack at 6,000 characters and the console at 4,000, keeping the newest lines", async () => {
    const error = new Error("m".repeat(5_000));
    error.stack = STACK.replace("boom", error.message).padEnd(20_000, "s");
    for (let line = 0; line < 20; line++) console.error(FULL_LINE);
    scriptError(error);
    await settle(10_000);
    const event = sentEvent();
    expect(event?.message).toHaveLength(2_000);
    expect(event?.stack).toHaveLength(6_000);
    expect(event?.stack.startsWith("TypeError: mmmm")).toBe(true);
    expect(event?.console_log.length ?? 0).toBeLessThanOrEqual(4_000);
    expect(event?.console_log.startsWith(STAMP)).toBe(true);
    expect(event?.console_log.endsWith(FULL_LINE)).toBe(true);
  });
});

describe("console buffer", () => {
  it("keeps the last twenty lines at warning or above, each stamped in IST", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "info").mockImplementation(() => {});
    for (let n = 0; n < 25; n++) {
      console.log(`l${n}`);
      console.info(`i${n}`);
      if (n % 2 === 0) console.error(`e${n}`);
      else console.warn(`w${n}`);
    }
    scriptError();
    await settle(10_000);
    const lines = (sentEvent().console_log ?? "").split("\n");
    expect(lines).toHaveLength(20);
    expect(lines[0]).toBe("[14:03:11] [WARN] w5");
    expect(lines[19]).toBe("[14:03:11] [ERROR] e24");
    expect(lines.join("\n")).not.toMatch(/[li]\d+/);
  });

  it("summarises a logged Error and never reads into a logged object", async () => {
    console.error(new TypeError("bad value"), { token: "hidden", rows: [[1, 2]] }, 42, null);
    scriptError();
    await settle(10_000);
    const log = sentEvent().console_log ?? "";
    expect(log).toBe("[14:03:11] [ERROR] TypeError: bad value [object Object] 42 null");
  });
});

describe("pure functions", () => {
  it("topFrame is where the stack fails, read as the server reads it", () => {
    // the server's own outputs
    expect(
      topFrame("TypeError: x is undefined\n    at render (https://app.example/assets/index-abc123.js:1:2345)"),
    ).toBe("index-abc123.js:1");
    expect(topFrame("at f (http://dev.local:3000/assets/app.js:1:2)")).toBe("app.js:1");
    expect(topFrame("at f (webpack://app/./src/x.js:1:1)")).toBe("x.js:1");
    expect(topFrame("Error: boom\n    at save (app.js:12:5)")).toBe("app.js:12");
    expect(
      topFrame(
        'Traceback (most recent call last):\n  File "/srv/app/curvestudio_api/jobs.py", line 265, in _run\n    File "C:\\srv\\app\\worker.py", line 10, in run\nRuntimeError: boom\n',
      ),
    ).toBe("jobs.py:265");
    for (const noFrame of ["TypeError: x is undefined", "Error at 12:30:00 for file.json: see log", "version 1.2:3", ""]) {
      expect(topFrame(noFrame), noFrame).toBe("");
    }
    expect(topFrame(STACK)).toBe("index-abc.js:1");
    expect(topFrame("render@https://app.example.com/assets/x.js:12:3")).toBe("x.js:12");
    // a dotted host with a port reads like a frame, but is where the app was served from
    expect(topFrame("at f (https://app.example.com:8443/assets/x.js:5:1)")).toBe("x.js:5");
    // Python's `$` also matches before a newline that ends the text, so to its `\w$` a line
    // that ends in a word character and a newline ends in a word character: the frame after
    // it is passed over, and the Python frame before it is what remains
    expect(topFrame('  File "/x.py", line 5\napp.js:12:5')).toBe("x.py:5");
    expect(topFrame('  File "/x.py", line 5\napp.js:12:5\n    at g (b.js:3:4)')).toBe("b.js:3");
    expect(topFrame("line 5\napp.js:12:5")).toBe("");
  });

  it("scrubSecrets is the server's scrub, and what it leaves is final to the server", () => {
    // each pair is the server's own output for that input
    const pinned: Array<[string, string]> = [
      ["login failed: password=hunter2 for user 7", "login failed: password=[REDACTED] for user 7"],
      [
        'body was {"client_secret": "s3cr3t value", "email": "bob@example.org"}',
        'body was {"client_secret": "[REDACTED]", "email": "[REDACTED_EMAIL]"}',
      ],
      ["request had token: abc def and status_code=500", "request had token: [REDACTED] def and status_code=[REDACTED]"],
      [
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sigpart went out",
        "Authorization: Bearer [REDACTED] went out",
      ],
      ["fetch failed for monkey=banana and keyboard=azerty", "fetch failed for monkey=[REDACTED] and keyboard=azerty"],
      [
        "smtp said: recipient a.b+c@sub.example.co.in rejected, see user@host",
        "smtp said: recipient [REDACTED_EMAIL] rejected, see user@host",
      ],
      ["Authorization: Basic dXNlcjpwYXNz", "Authorization: Basic [REDACTED]"],
      ['"Authorization": "Bearer abc.def.ghi"', '"Authorization": "Bearer [REDACTED]"'],
      ["authorization=Bearer xyz123", "authorization=Bearer [REDACTED]"],
      ["Authorization: raw-credential", "Authorization: [REDACTED]"],
      ["fetch(Authorization: Bearer s3cr3t-access-token-value) failed", "fetch(Authorization: Bearer [REDACTED]) failed"],
      ["access_token=abc api_key: def client_secret='g h'", "access_token=[REDACTED] api_key: [REDACTED] client_secret='[REDACTED]'"],
      ['{"password": "hunter2", "user": "bob"}', '{"password": "[REDACTED]", "user": "bob"}'],
      ["PASSWORD = Hunter2!", "PASSWORD = [REDACTED]"],
      ['no account for "email": "bob@example.org"', 'no account for "email": "[REDACTED_EMAIL]"'],
      [`fetch failed for ${JWT} at 12:00`, "fetch failed for [REDACTED_JWT] at 12:00"],
      // the keyword's `i` is `İ` or `ı` to Python's IGNORECASE, and so here
      ["authorızatıon: Bearer x", "authorızatıon: Bearer [REDACTED]"],
      // nothing here names a secret, and no address does
      ["username=bob email=bob color=red limit: 10 keyboard: yes", "username=bob email=bob color=red limit: 10 keyboard: yes"],
      ["see user@host for details", "see user@host for details"],
      // a bearer token with no `Authorization` before it is a credential all the same; a short
      // word after "bearer" is a word
      ["got Bearer abc123.def456-ghi back", "got Bearer [REDACTED] back"],
      ["the bearer of bad news", "the bearer of bad news"],
      ["Bearer [REDACTED]", "Bearer [REDACTED]"],
      // a word glued to a JWT: the first pass reads it as the scheme, the next takes the rest
      [
        "Authorization:client_secreteyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sigpart\t/ = ",
        "Authorization:[REDACTED]\t[REDACTED] = ",
      ],
    ];
    for (const [input, output] of pinned) {
      expect(scrubSecrets(input), input).toBe(output);
      // the server scrubs what it is given: its scrub must find nothing left to do
      expect(scrubSecrets(output), input).toBe(output);
    }
  });

  it("scrubText adds the wider rules after the server's, and the result is still final to the server", () => {
    const pinned: Array<[string, string]> = [
      ["see 3f2b8a1e-0c4d-4e6f-8a9b-1c2d3e4f5a6b please", "see {id} please"],
      ["back to https://app.example.com/reset?token=abc#x", "back to https://app.example.com/reset"],
      [
        "for carol@example.org password=hunter2 device 3f2b8a1e-0c4d-4e6f-8a9b-1c2d3e4f5a6b",
        "for [REDACTED_EMAIL] password=[REDACTED] device {id}",
      ],
      ["TypeError: x is not a function (v1.2.3)", "TypeError: x is not a function (v1.2.3)"],
    ];
    for (const [input, output] of pinned) {
      expect(scrubText(input), input).toBe(output);
      expect(scrubSecrets(output), input).toBe(output);
    }
  });

  it("lands on the server's fixed point where the server's own single pass is not yet final", () => {
    // The word before the credentials is read as the scheme until the JWT inside it is gone;
    // the server, given this result, finds nothing left to change.
    const once = scrubText("Authorization:client_secret" + JWT + "\t/ = ");
    expect(once).toBe("Authorization:[REDACTED]\t[REDACTED] = ");
    expect(scrubSecrets(once)).toBe(once);
  });

  it("eventFromError reads an Error and describes anything else", () => {
    const error = new RangeError("out of range");
    error.stack = "RangeError: out of range\n    at f (a.js:3:4)";
    expect(eventFromError(error)).toEqual({
      error_type: "RangeError",
      message: "out of range",
      stack: "RangeError: out of range\n    at f (a.js:3:4)",
    });
    expect(eventFromError("just a string")).toEqual({ error_type: "NonError", message: "just a string", stack: "" });
    expect(eventFromError({ name: "QuotaExceeded", message: "storage is full" })).toEqual({
      error_type: "QuotaExceeded",
      message: "storage is full",
      stack: "",
    });
    expect(eventFromError(42)).toEqual({ error_type: "NonError", message: "42", stack: "" });
    expect(eventFromError(undefined)).toEqual({ error_type: "NonError", message: "undefined", stack: "" });
    const nameless = new Error("");
    expect(eventFromError(nameless)).toMatchObject({ error_type: "Error", message: "(no message)" });
  });
});

describe("installation", () => {
  it("installing twice adds one set of hooks", async () => {
    const again = installErrorReporter({ apiBase: "https://elsewhere.example/api/v1" });
    expect(again).toBe(uninstall);
    scriptError();
    await settle(10_000);
    expect(requests()[0]?.url).toBe(ENDPOINT);
    expect(requests()[0]?.events).toHaveLength(1);
  });

  it("the returned function removes everything it added", async () => {
    uninstall();
    expect(console.error).toBe(errorSpy);
    expect(console.warn).toBe(warnSpy);
    // With the reporter gone, jsdom would report the synthetic error event as uncaught; a
    // page always has its own handling, which this stands in for.
    const swallow = (event: Event): void => event.preventDefault();
    window.addEventListener("error", swallow);
    scriptError();
    console.warn("after removal");
    await settle(20_000);
    pageHide();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(beaconMock).not.toHaveBeenCalled();
    window.removeEventListener("error", swallow);

    // and a fresh installation starts clean
    uninstall = installErrorReporter({ apiBase: API });
    scriptError(boom("after-reinstall"));
    await settle(10_000);
    expect(sentEvent().console_log).toBe("");
  });

  it("the reporter adds nothing to the console itself", async () => {
    for (let n = 1; n <= 5; n++) scriptError(boom(String(n)));
    pageHide();
    await settle(60_000);
    expect(errorSpy).toHaveBeenCalledTimes(0);
    expect(warnSpy).toHaveBeenCalledTimes(0);
  });
});

describe("the fallback hash", () => {
  it("where crypto.subtle is unavailable the reference is still a stable id for the error", async () => {
    vi.stubGlobal("crypto", {} as Crypto);
    const first = await reportUiError(boom(), { componentStack: null });
    const again = await reportUiError(boom(), { componentStack: null });
    const other = await reportUiError(boom("2"), { componentStack: null });
    expect(first).toMatch(/^[0-9a-f]{12,}$/);
    expect(first).toHaveLength(12);
    expect(again).toBe(first);
    expect(other).not.toBe(first);
    // not SHA-256: a reference from the fallback is not the one the server computes
    expect(first).not.toBe(UI_BOOM);
  });
});
