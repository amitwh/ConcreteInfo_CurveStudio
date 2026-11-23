import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "../api/client";
import { renderWithQuery } from "../test/render";
import type { ErrorEventDetailOut, ErrorEventOut, ErrorStatus } from "./adminApi";
import { ErrorsPage } from "./ErrorsPage";

const CRASH_ID = "0b1f0d3e-9a6c-4b2e-8f1a-2c3d4e5f6a7b";
const NOTE_ID = "1c2e3f4a-aaaa-4bbb-8ccc-0123456789ab";
const SQL_ID = "2d3f4a5b-bbbb-4ccc-8ddd-1123456789ab";
const NET_ID = "3a4b5c6d-cccc-4ddd-8eee-2123456789ab";
const REFERENCE = "3f9a1c2b7d0e";

const STACK = "TypeError: Cannot read properties of undefined (reading 'x')\n    at FitPanel (FitPanel.tsx:42:11)";
const CONSOLE = "[ERROR] Failed to load resource: 500\n[WARN] deprecated option: bias";
const USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) CurveStudio/0.11.1";

function event(overrides: Partial<ErrorEventOut>): ErrorEventOut {
  return {
    id: "00000000-0000-4000-8000-000000000000",
    source: "client",
    category: "ui",
    severity: "error",
    error_type: "Error",
    message: "Something went wrong",
    fingerprint: "0000000000000000000000000000000000000000000000000000000000000000",
    route: null,
    method: null,
    status_code: null,
    request_id: null,
    app_version: null,
    status: "new",
    occurrence_count: 1,
    first_seen_at: "2026-11-01T10:00:00+05:30",
    last_seen_at: "2026-11-01T10:00:00+05:30",
    ...overrides,
  };
}

/** A fatal crash in the browser, four times. */
const CRASH = event({
  id: CRASH_ID,
  source: "client",
  category: "crash",
  severity: "fatal",
  error_type: "TypeError",
  message: "Cannot read properties of undefined (reading 'x')",
  fingerprint: `${REFERENCE}ff11aa22bb33cc44dd55ee66ff77aa88bb99cc00`,
  route: "/fits/new",
  app_version: "0.11.1",
  occurrence_count: 4,
  first_seen_at: "2026-11-20T10:00:00+05:30",
  last_seen_at: "2026-11-23T14:05:00+05:30",
});
/** What a visitor typed on the recovery screen about the crash above. */
const NOTE = event({
  id: NOTE_ID,
  source: "client",
  category: "ui",
  severity: "error",
  error_type: "UserDescription",
  message: "I was fitting a new model when the page went blank",
  fingerprint: "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee",
  route: "/fits/new",
  app_version: "0.11.1",
  last_seen_at: "2026-11-23T14:06:00+05:30",
});
/** A server-side database error, already acknowledged. */
const SQL = event({
  id: SQL_ID,
  source: "server",
  category: "sql",
  severity: "error",
  error_type: "OperationalError",
  message: "server closed the connection unexpectedly",
  fingerprint: "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b",
  route: "/api/v1/datasets",
  method: "POST",
  status_code: 500,
  request_id: "req-42",
  // 18:30 UTC on the 22nd is midnight on the 23rd in IST
  last_seen_at: "2026-11-22T18:30:00Z",
});
const NET = event({
  id: NET_ID,
  source: "client",
  category: "network",
  severity: "error",
  error_type: "NetworkError",
  message: "Failed to fetch",
  route: "/datasets",
  status: "acknowledged",
});
/** A note about an error this console has never loaded, already ignored. */
const ORPHAN_ID = "4b5c6d7e-dddd-4eee-8fff-323456789abc";
const ORPHAN = event({
  id: ORPHAN_ID,
  source: "client",
  category: "ui",
  severity: "error",
  error_type: "UserDescription",
  message: "The chart would not print",
  fingerprint: "abcdef00abcdef00abcdef00abcdef00abcdef00abcdef00abcdef00abcdef00abcd",
  route: "/report",
  status: "ignored",
});

const EVENTS = [CRASH, NOTE, SQL, NET, ORPHAN];

const CRASH_DETAIL: ErrorEventDetailOut = {
  ...CRASH,
  stack: STACK,
  context_json: { componentStack: "\n    at FitPanel", model: "plastic" },
  console_log: CONSOLE,
  user_agent: USER_AGENT,
};
const NOTE_DETAIL: ErrorEventDetailOut = {
  ...NOTE,
  stack: null,
  context_json: { reference: REFERENCE },
  console_log: null,
  user_agent: USER_AGENT,
};
const NET_DETAIL: ErrorEventDetailOut = {
  ...NET,
  stack: null,
  context_json: null,
  console_log: "",
  user_agent: null,
};
const SQL_DETAIL: ErrorEventDetailOut = {
  ...SQL,
  stack: 'sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) server closed the connection unexpectedly',
  context_json: { job: "autofit", dataset: "d-1" },
  console_log: null,
  user_agent: null,
};
const ORPHAN_DETAIL: ErrorEventDetailOut = {
  ...ORPHAN,
  stack: null,
  context_json: { reference: "0a0a0a0a0a0a" },
  console_log: null,
  user_agent: USER_AGENT,
};
const DETAILS: Record<string, ErrorEventDetailOut> = {
  [CRASH_ID]: CRASH_DETAIL,
  [NOTE_ID]: NOTE_DETAIL,
  [NET_ID]: NET_DETAIL,
  [SQL_ID]: SQL_DETAIL,
  [ORPHAN_ID]: ORPHAN_DETAIL,
};

const BUNDLE = "# Error event\n\n| Field | Value |\n| --- | --- |\n| Id | 0b1f0d3e |\n";

/** A spy on the client's request method, with its call record typed. */
function spyRequest() {
  return vi.spyOn(api, "request");
}
type RequestSpy = ReturnType<typeof spyRequest>;
type TextSpy = ReturnType<typeof spyText>;
function spyText() {
  return vi.spyOn(api, "text");
}

interface Options {
  /** What the server answers to a status change. */
  refuseStatus?: ApiError;
  /** What the server answers to the bundle. */
  refuseBundle?: ApiError;
  /** A total to report on every list, and one event on page 2. */
  paged?: boolean;
}

/**
 * Answers like the API: the list is filtered by status and category over the events'
 * current statuses; a PATCH changes a status; the detail and the bundle are read.
 */
function serve(options: Options = {}): { request: RequestSpy; text: TextSpy } {
  const statusOf = new Map(EVENTS.map((item) => [item.id, item.status] as const));
  const text = spyText();
  if (options.refuseBundle !== undefined) text.mockRejectedValue(options.refuseBundle);
  else text.mockResolvedValue(BUNDLE);
  const request = spyRequest().mockImplementation(async (path, init) => {
    if (path.startsWith("/admin/errors?")) {
      const params = new URLSearchParams(path.slice("/admin/errors?".length));
      const status = params.get("status");
      const category = params.get("category");
      const pageNumber = Number(params.get("page"));
      const matching = EVENTS.filter(
        (item) => statusOf.get(item.id) === status && (category === null || item.category === category),
      ).sort((a, b) => b.last_seen_at.localeCompare(a.last_seen_at));
      const items = pageNumber === 1 ? matching : options.paged ? [NET] : [];
      const total = options.paged && category === null ? 128 : matching.length;
      return { items, total, page: pageNumber, page_size: 25 };
    }
    const id = path.slice("/admin/errors/".length);
    if (init?.method === "PATCH") {
      if (options.refuseStatus !== undefined) throw options.refuseStatus;
      const change = JSON.parse(String(init.body)) as { status: ErrorStatus };
      statusOf.set(id, change.status);
      const base = EVENTS.find((item) => item.id === id);
      if (base === undefined) throw new ApiError(404, "NOT_FOUND", "error event not found", {});
      return { ...base, status: change.status };
    }
    const detail = DETAILS[id];
    if (detail === undefined) throw new ApiError(404, "NOT_FOUND", "error event not found", {});
    return { ...detail, status: statusOf.get(id) ?? detail.status };
  });
  return { request, text };
}

function table(): HTMLElement {
  return screen.getByRole("table", { name: "Error events" });
}

function rowTexts(): string[][] {
  return within(table())
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell").map((cell) => cell.textContent ?? ""));
}

/** The status tab whose name starts with `label`, count included. */
function tab(label: string): HTMLElement {
  return screen.getByRole("tab", { name: new RegExp(`^${label}`) });
}

function listCalls(spy: RequestSpy): string[] {
  return spy.mock.calls.map(([path]) => path).filter((path) => path.startsWith("/admin/errors?"));
}

function patches(spy: RequestSpy): Array<{ path: string; body: unknown }> {
  return spy.mock.calls
    .filter(([, init]) => init?.method === "PATCH")
    .map(([path, init]) => ({ path, body: JSON.parse(String(init?.body)) }));
}

/** Each term with its definition, as "Route/api/v1/datasets", in `container`. */
function facts(container: HTMLElement): string[] {
  const terms = within(container).getAllByRole("term");
  const definitions = within(container).getAllByRole("definition");
  return terms.map((term, index) => `${term.textContent}${definitions[index]?.textContent ?? ""}`);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("ErrorsPage — the list", () => {
  it("shows loading, then the new events, newest first, with every column", async () => {
    serve();
    renderWithQuery(<ErrorsPage />);
    expect(screen.getByRole("status").textContent).toContain("Loading the error events");
    await screen.findByRole("table", { name: "Error events" });

    expect(within(table()).getAllByRole("columnheader").map((th) => th.textContent)).toEqual([
      "Last seen",
      "Category",
      "Severity",
      "Type",
      "Message",
      "Route",
      "Times",
      "Actions",
    ]);
    expect(rowTexts().map((cells) => cells.slice(0, 7))).toEqual([
      [
        "23/11/2026 14:06",
        "UI",
        "Error",
        "Visitor's note",
        "I was fitting a new model when the page went blank",
        "/fits/new",
        "1",
      ],
      [
        "23/11/2026 14:05",
        "Crash",
        "Fatal",
        "TypeError",
        "Cannot read properties of undefined (reading 'x')",
        "/fits/new",
        "4",
      ],
      [
        "23/11/2026 00:00",
        "SQL",
        "Error",
        "OperationalError",
        "server closed the connection unexpectedly",
        "/api/v1/datasets",
        "1",
      ],
    ]);
    expect(screen.getByText("Showing 1–3 of 3")).toBeTruthy();
  });

  it("marks a fatal severity, and keeps a long message to one line with its full text in the title", async () => {
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    const rows = within(table()).getAllByRole("row").slice(1);
    const fatal = within(rows[1] as HTMLElement).getAllByRole("cell")[2] as HTMLElement;
    expect(within(fatal).getByText("Fatal").className).toContain("badge--bad");
    // an ordinary error is not marked
    const ordinary = within(rows[2] as HTMLElement).getAllByRole("cell")[2] as HTMLElement;
    expect(ordinary.textContent).toBe("Error");
    expect(ordinary.querySelector(".badge")).toBeNull();

    const message = within(rows[1] as HTMLElement).getAllByRole("cell")[4] as HTMLElement;
    expect(message.classList.contains("truncate")).toBe(true);
    expect(message.getAttribute("title")).toBe("Cannot read properties of undefined (reading 'x')");
  });

  it("never shows UserDescription: the row reads as a visitor's note with the first words of the note", async () => {
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    expect(screen.queryByText(/UserDescription/)).toBeNull();
    expect(screen.getByText("Visitor's note")).toBeTruthy();
    expect(
      within(table()).getByText("I was fitting a new model when the page went blank"),
    ).toBeTruthy();
  });

  it("names each row's Open button after the event and when it was last seen", async () => {
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    const names = within(table())
      .getAllByRole("button")
      .map((button) => button.getAttribute("aria-label"));
    expect(names).toEqual([
      "Open Visitor's note, last seen 23/11/2026 14:06",
      "Open TypeError, last seen 23/11/2026 14:05",
      "Open OperationalError, last seen 23/11/2026 00:00",
    ]);
  });

  it("shows the total of the current tab on its tab, and no number on the others", async () => {
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    expect(tab("New").textContent).toBe("New3");
    expect(tab("Acknowledged").textContent).toBe("Acknowledged");
    expect(tab("Acknowledged").getAttribute("aria-selected")).toBe("false");
    expect(tab("New").getAttribute("aria-selected")).toBe("true");
  });

  it("each tab lists its own events and asks for them with its status", async () => {
    const user = userEvent.setup();
    const { request } = serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    expect(listCalls(request)).toEqual(["/admin/errors?status=new&page=1"]);

    await user.click(tab("Acknowledged"));
    await waitFor(() => expect(listCalls(request).at(-1)).toBe("/admin/errors?status=acknowledged&page=1"));
    expect(await screen.findByText("Failed to fetch")).toBeTruthy();
    expect(tab("Acknowledged").textContent).toBe("Acknowledged1");
    expect(tab("New").textContent).toBe("New");
    expect(screen.queryByText("Visitor's note")).toBeNull();
    expect(tab("Acknowledged").getAttribute("aria-selected")).toBe("true");
  });

  it("filters by category and starts again from the first page", async () => {
    const user = userEvent.setup();
    const { request } = serve({ paged: true });
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(listCalls(request).at(-1)).toBe("/admin/errors?status=new&page=2"));

    await user.selectOptions(screen.getByLabelText("Category"), "crash");
    await waitFor(() =>
      expect(listCalls(request).at(-1)).toBe("/admin/errors?status=new&category=crash&page=1"),
    );
    await screen.findByText("TypeError");
    expect(screen.queryByText("Visitor's note")).toBeNull();
    expect(screen.getByText("1", { selector: ".tab__count" })).toBeTruthy();
  });

  it("shows an empty state for a tab with nothing in it, and for a category with nothing in it", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(tab("Resolved"));
    expect(await screen.findByText("No resolved events.")).toBeTruthy();
    expect(screen.queryByRole("table")).toBeNull();

    await user.selectOptions(screen.getByLabelText("Category"), "network");
    expect(await screen.findByText("No resolved events in the Network category.")).toBeTruthy();
  });

  it("shows an error with a Retry that fetches the events again", async () => {
    const user = userEvent.setup();
    const { request } = serve();
    const answer = request.getMockImplementation();
    let fail = true;
    request.mockImplementation(async (path, init) => {
      if (fail && path.startsWith("/admin/errors?")) {
        throw new ApiError(500, "INTERNAL_ERROR", "the server failed", null);
      }
      return answer === undefined ? Promise.reject(new Error("nothing behind the spy")) : answer(path, init);
    });
    renderWithQuery(<ErrorsPage />);
    expect((await screen.findByRole("alert")).textContent).toBe("the server failed");
    expect(tab("New").textContent).toBe("New");

    fail = false;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("table", { name: "Error events" })).toBeTruthy();
    expect(listCalls(request)).toHaveLength(2);
  });

  it("asks for the next page", async () => {
    const user = userEvent.setup();
    const { request } = serve({ paged: true });
    renderWithQuery(<ErrorsPage />);
    await screen.findByText("Showing 1–25 of 128");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(listCalls(request).at(-1)).toBe("/admin/errors?status=new&page=2"));
    expect(await screen.findByText("Showing 26–50 of 128")).toBeTruthy();
    expect(screen.getByText("Failed to fetch")).toBeTruthy();
  });
});

describe("ErrorsPage — the status tabs by keyboard", () => {
  it("moves with the arrow keys, selecting as it goes, and jumps with Home and End", async () => {
    const user = userEvent.setup();
    const { request } = serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    expect(tab("New").tabIndex).toBe(0);
    expect(tab("Acknowledged").tabIndex).toBe(-1);

    await user.click(tab("New"));
    await user.keyboard("{ArrowRight}");
    expect(document.activeElement).toBe(tab("Acknowledged"));
    await waitFor(() => expect(listCalls(request).at(-1)).toBe("/admin/errors?status=acknowledged&page=1"));

    await user.keyboard("{End}");
    expect(document.activeElement).toBe(tab("Ignored"));
    await waitFor(() => expect(listCalls(request).at(-1)).toBe("/admin/errors?status=ignored&page=1"));

    await user.keyboard("{ArrowLeft}");
    expect(document.activeElement).toBe(tab("Resolved"));
    await user.keyboard("{Home}");
    expect(document.activeElement).toBe(tab("New"));
    await waitFor(() => expect(listCalls(request).at(-1)).toBe("/admin/errors?status=new&page=1"));
  });
});

describe("ErrorsPage — one event's detail", () => {
  it("opens below the list, takes the focus, and shows every field", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));

    const detail = screen.getByRole("region", { name: "TypeError" });
    expect(document.activeElement).toBe(detail);
    expect(await within(detail).findByText(STACK, { collapseWhitespace: false })).toBeTruthy();

    expect(facts(detail)).toEqual(
      expect.arrayContaining([
        "SourceClient",
        "CategoryCrash",
        "SeverityFatal",
        "TypeTypeError",
        "StatusNew",
        "MessageCannot read properties of undefined (reading 'x')",
        "Route/fits/new",
        "Occurrences4",
        "First seen20/11/2026 10:00",
        "Last seen23/11/2026 14:05",
        "App version0.11.1",
        `Fingerprint${CRASH.fingerprint}`,
        "User agentMozilla/5.0 (X11; Linux x86_64) CurveStudio/0.11.1",
        // nothing was recorded for these
        "Method–",
        "Status code–",
        "Request ID–",
      ]),
    );
    // the console lines in a code block of their own
    expect(within(detail).getByText(CONSOLE, { collapseWhitespace: false }).tagName).toBe("CODE");
  });

  it("shows the stack, the context as formatted JSON and the console lines as scrollable code blocks", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = screen.getByRole("region", { name: "TypeError" });

    const blocks = await within(detail).findAllByRole("region", { name: /code block/ });
    expect(blocks.map((block) => block.getAttribute("aria-label"))).toEqual([
      "Stack code block",
      "Context code block",
      "Console log code block",
    ]);
    for (const block of blocks) expect(block.className).toContain("admin-code");
    expect(blocks[0]?.textContent).toBe(STACK);
    expect(blocks[1]?.textContent).toBe(
      '{\n  "componentStack": "\\n    at FitPanel",\n  "model": "plastic"\n}',
    );
    expect(blocks[2]?.textContent).toBe(CONSOLE);
  });

  it("says where nothing was recorded instead of showing an empty block", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(tab("Acknowledged"));
    await screen.findByText("Failed to fetch");
    await user.click(within(table()).getByRole("button", { name: /^Open NetworkError/ }));
    const detail = screen.getByRole("region", { name: "NetworkError" });

    expect(within(detail).queryByRole("region", { name: /code block/ })).toBeNull();
    expect(facts(detail)).toEqual(
      expect.arrayContaining(["Stack–", "Context–", "Console log–", "User agent–", "Route/datasets"]),
    );
  });

  it("shows the request id", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open OperationalError/ }));
    const detail = await screen.findByRole("region", { name: "OperationalError" });
    expect(within(detail).getByText("req-42").tagName).toBe("CODE");
    expect(facts(detail)).toContain("Request IDreq-42");
    expect(facts(detail)).toContain("MethodPOST");
    expect(facts(detail)).toContain("Status code500");
  });

  it("names the error a visitor's note belongs to, and opens it", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open Visitor/ }));

    const detail = await screen.findByRole("region", { name: "Visitor's note" });
    // the note is prose, and is never shown as an error type
    expect(within(detail).queryByText(/UserDescription/)).toBeNull();
    expect(within(detail).getByText(NOTE.message).className).toContain("admin-note__text");
    expect(within(detail).getByText(/It belongs to/).textContent).toContain("TypeError");
    expect(within(detail).getByText(REFERENCE).tagName).toBe("CODE");

    await user.click(within(detail).getByRole("button", { name: "Open that error" }));
    expect(await screen.findByRole("region", { name: "TypeError" })).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Visitor's note" })).toBeNull();
    // The focus still returns to the row the first detail was opened from.
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(document.activeElement).toBe(within(table()).getByRole("button", { name: /^Open Visitor/ }));
  });

  it("says when the error a note belongs to is not among the events loaded", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    // A note about an error no loaded page carries: the reference names nothing here.
    await user.click(tab("Ignored"));
    await screen.findByText("The chart would not print");
    await user.click(within(table()).getByRole("button", { name: /^Open Visitor/ }));
    const detail = screen.getByRole("region", { name: "Visitor's note" });
    expect(await within(detail).findByText(/not among the events loaded here/)).toBeTruthy();
    expect(within(detail).getByText("0a0a0a0a0a0a").tagName).toBe("CODE");
    expect(within(detail).queryByRole("button", { name: "Open that error" })).toBeNull();
  });

  it("closes on request and returns the focus to the row's Open", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    const trigger = within(table()).getByRole("button", { name: /^Open TypeError/ });
    await user.click(trigger);
    await screen.findByRole("region", { name: "TypeError" });
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("region")).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("choosing the tab and the category that are already chosen changes nothing", async () => {
    const user = userEvent.setup();
    const { request } = serve({ paged: true });
    renderWithQuery(<ErrorsPage />);
    await screen.findByText("Showing 1–25 of 128");
    await user.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("Showing 26–50 of 128");
    await user.click(within(table()).getByRole("button", { name: /^Open NetworkError/ }));
    await screen.findByRole("region", { name: "NetworkError" });
    const before = listCalls(request).length;

    await user.click(tab("New"));
    await user.selectOptions(screen.getByLabelText("Category"), "");
    // no list is fetched again: the page stays, and with it the open detail
    expect(listCalls(request)).toHaveLength(before);
    expect(screen.getByRole("region", { name: "NetworkError" })).toBeTruthy();
    expect(screen.getByText("Showing 26–50 of 128")).toBeTruthy();
  });

  it("closes the open detail when the tab changes", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    await screen.findByRole("region", { name: "TypeError" });
    await user.click(tab("Resolved"));
    await waitFor(() => expect(screen.queryByRole("region")).toBeNull());
  });

  it("shows the server's reason when the event cannot be loaded, with a Retry", async () => {
    const user = userEvent.setup();
    const { request } = serve();
    const answer = request.getMockImplementation();
    let refuse = true;
    request.mockImplementation(async (path, init) => {
      if (refuse && path === `/admin/errors/${CRASH_ID}` && init === undefined) {
        throw new ApiError(404, "NOT_FOUND", "error event not found", { code: "NOT_FOUND" });
      }
      return answer === undefined ? Promise.reject(new Error("nothing behind the spy")) : answer(path, init);
    });
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = screen.getByRole("region");
    expect((await within(detail).findByRole("alert")).textContent).toBe("error event not found");

    refuse = false;
    await user.click(within(detail).getByRole("button", { name: "Retry" }));
    expect(await within(detail).findByText(STACK, { collapseWhitespace: false })).toBeTruthy();
  });
});

describe("ErrorsPage — changing an event's status", () => {
  it("offers Acknowledge, Resolve and Ignore for a new event, and acts at once", async () => {
    const user = userEvent.setup();
    const { request } = serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });
    const actions = within(detail).getAllByRole("button").map((button) => button.textContent);
    expect(actions).toEqual(["Close", "Acknowledge", "Resolve", "Ignore", "Copy as Markdown"]);
    expect(within(detail).queryByRole("button", { name: "Reopen" })).toBeNull();

    await user.click(within(detail).getByRole("button", { name: "Acknowledge" }));
    expect(await within(detail).findByText("Marked as acknowledged.")).toBeTruthy();
    expect(patches(request)).toEqual([{ path: `/admin/errors/${CRASH_ID}`, body: { status: "acknowledged" } }]);
    expect(within(detail).getByText("Acknowledged", { selector: ".badge" }).className).toContain(
      "badge--warning",
    );
    // no longer new, so it leaves the New tab; the detail stays to show what was done
    await waitFor(() => expect(within(table()).queryByText("TypeError")).toBeNull());
    expect(tab("New").textContent).toBe("New2");
    expect(screen.getByRole("region", { name: "TypeError" })).toBeTruthy();
  });

  it("offers Resolve and Ignore for an acknowledged event, and no Acknowledge", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(tab("Acknowledged"));
    await screen.findByText("Failed to fetch");
    await user.click(within(table()).getByRole("button", { name: /^Open NetworkError/ }));
    const detail = await screen.findByRole("region", { name: "NetworkError" });
    const actions = within(detail).getAllByRole("button").map((button) => button.textContent);
    expect(actions).toEqual(["Close", "Resolve", "Ignore", "Copy as Markdown"]);

    await user.click(within(detail).getByRole("button", { name: "Resolve" }));
    expect(await within(detail).findByText("Marked as resolved.")).toBeTruthy();
    expect(within(detail).getByText("Resolved", { selector: ".badge" }).className).toContain("badge--ok");
    expect(within(detail).getByRole("button", { name: "Reopen" })).toBeTruthy();
  });

  it("Reopen sends new for a resolved or ignored event", async () => {
    const user = userEvent.setup();
    const { request } = serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });
    await user.click(within(detail).getByRole("button", { name: "Ignore" }));
    expect(await within(detail).findByText("Marked as ignored.")).toBeTruthy();
    expect(within(detail).getAllByRole("button").map((button) => button.textContent)).toEqual([
      "Close",
      "Reopen",
      "Resolve",
      "Copy as Markdown",
    ]);

    await user.click(within(detail).getByRole("button", { name: "Reopen" }));
    expect(await within(detail).findByText("Reopened.")).toBeTruthy();
    expect(within(detail).getByText("New", { selector: ".badge" })).toBeTruthy();
    expect(patches(request)).toEqual([
      { path: `/admin/errors/${CRASH_ID}`, body: { status: "ignored" } },
      { path: `/admin/errors/${CRASH_ID}`, body: { status: "new" } },
    ]);
    // back among the new events
    await waitFor(() => expect(within(table()).getByText("TypeError")).toBeTruthy());
  });

  it("shows the server's reason for a refused change and leaves the status as it was", async () => {
    const user = userEvent.setup();
    serve({ refuseStatus: new ApiError(500, "INTERNAL_ERROR", "the server failed", null) });
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });
    await user.click(within(detail).getByRole("button", { name: "Resolve" }));

    expect((await within(detail).findByRole("alert")).textContent).toBe("the server failed");
    expect(within(detail).queryByText("Marked as resolved.")).toBeNull();
    expect(within(detail).getByText("New", { selector: ".badge" })).toBeTruthy();
    expect(within(table()).getByText("TypeError")).toBeTruthy();

    // a later attempt replaces the reason
    await user.click(within(detail).getByRole("button", { name: "Acknowledge" }));
    await waitFor(() => expect(within(detail).getByRole("alert").textContent).toBe("the server failed"));
  });
});

describe("ErrorsPage — Copy as Markdown", () => {
  it("fetches the document, puts it on the clipboard and says so", async () => {
    const user = userEvent.setup();
    const { text } = serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });

    await user.click(within(detail).getByRole("button", { name: "Copy as Markdown" }));
    expect(await within(detail).findByText("Copied.")).toBeTruthy();
    expect(await navigator.clipboard.readText()).toBe(BUNDLE);
    expect(text).toHaveBeenCalledWith(`/admin/errors/${CRASH_ID}/bundle`);
    expect(within(detail).queryByRole("textbox")).toBeNull();
  });

  it("shows the document, with Select all, when the clipboard refuses, and does not say Copied", async () => {
    const user = userEvent.setup();
    serve();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(new DOMException("denied", "NotAllowedError"));
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });

    await user.click(within(detail).getByRole("button", { name: "Copy as Markdown" }));
    expect(
      await within(detail).findByText(/clipboard is not available here, so the document is shown below/),
    ).toBeTruthy();
    expect(within(detail).queryByText("Copied.")).toBeNull();
    const shown = within(detail).getByRole("textbox", { name: "Error event as Markdown" }) as HTMLTextAreaElement;
    expect(shown.value).toBe(BUNDLE);
    expect(shown.readOnly).toBe(true);

    const select = vi.spyOn(shown, "select");
    await user.click(within(detail).getByRole("button", { name: "Select all" }));
    expect(select).toHaveBeenCalledTimes(1);
  });

  it("shows the document when there is no clipboard object at all", async () => {
    const user = userEvent.setup();
    serve();
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });

    await user.click(within(detail).getByRole("button", { name: "Copy as Markdown" }));
    expect(await within(detail).findByRole("textbox", { name: "Error event as Markdown" })).toBeTruthy();
    expect(within(detail).queryByText("Copied.")).toBeNull();
  });

  it("shows the server's reason when the document cannot be fetched", async () => {
    const user = userEvent.setup();
    serve({ refuseBundle: new ApiError(404, "NOT_FOUND", "error event not found", { code: "NOT_FOUND" }) });
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    const detail = await screen.findByRole("region", { name: "TypeError" });

    await user.click(within(detail).getByRole("button", { name: "Copy as Markdown" }));
    expect((await within(detail).findByRole("alert")).textContent).toBe("error event not found");
    expect(within(detail).queryByText("Copied.")).toBeNull();
    expect(within(detail).queryByRole("textbox")).toBeNull();
  });

  it("forgets the result when another event is opened", async () => {
    const user = userEvent.setup();
    serve();
    renderWithQuery(<ErrorsPage />);
    await screen.findByRole("table", { name: "Error events" });
    await user.click(within(table()).getByRole("button", { name: /^Open TypeError/ }));
    let detail = await screen.findByRole("region", { name: "TypeError" });
    await user.click(within(detail).getByRole("button", { name: "Copy as Markdown" }));
    await within(detail).findByText("Copied.");

    await user.click(within(table()).getByRole("button", { name: /^Open Visitor/ }));
    detail = await screen.findByRole("region", { name: "Visitor's note" });
    expect(within(detail).queryByText("Copied.")).toBeNull();
  });
});
