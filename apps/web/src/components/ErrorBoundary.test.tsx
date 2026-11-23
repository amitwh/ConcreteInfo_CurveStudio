import { createHash } from "node:crypto";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import {
  installErrorReporter,
  setConsentResolver,
  setErrorReportsEnabled,
  type ErrorReportEvent,
} from "../lib/errorReporter";
import { ErrorBoundary } from "./ErrorBoundary";

const API = "http://localhost:3000/api/v1";
const THANKS = "Thanks. Your note was sent with the reference above.";

/** The reference the server will hold for the error `Bomb` throws (no frames in its stack). */
const REFERENCE = createHash("sha256").update("ui|TypeError|render exploded||/").digest("hex").slice(0, 12);

/** Throws on every render; the stack is fixed so the reference is known. */
function Bomb(): ReactElement {
  const error = new TypeError("render exploded");
  error.stack = "TypeError: render exploded";
  throw error;
}

function renderBoundary(child: ReactElement = <Bomb />): void {
  render(<ErrorBoundary>{child}</ErrorBoundary>);
}

let fetchMock: Mock<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>;
let beaconMock: Mock<(url: string | URL, data?: BodyInit | null) => boolean>;
let uninstall: () => void;

/** The one description the reporter posted. */
function postedDescription(): { events: ErrorReportEvent[] } {
  expect(fetchMock).toHaveBeenCalledTimes(1);
  const [, init] = fetchMock.mock.calls[0] as [RequestInfo | URL, RequestInit];
  return JSON.parse(String(init.body)) as { events: ErrorReportEvent[] };
}

/** What the page's last beacon carried. */
async function beaconsEvents(): Promise<ErrorReportEvent[]> {
  const [, data] = beaconMock.mock.calls[0] as [string | URL, Blob];
  return (JSON.parse(await data.text()) as { events: ErrorReportEvent[] }).events;
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  fetchMock = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>();
  fetchMock.mockImplementation(() => Promise.resolve(new Response(null, { status: 204 })));
  vi.stubGlobal("fetch", fetchMock);
  beaconMock = vi.fn(() => true);
  Object.defineProperty(navigator, "sendBeacon", { configurable: true, value: beaconMock });
  uninstall = installErrorReporter({ apiBase: API });
});

const systemTheme = window.matchMedia;

afterEach(() => {
  uninstall();
  vi.unstubAllGlobals();
  delete (navigator as { sendBeacon?: unknown }).sendBeacon;
  // Every test's screen applies the theme; the next one starts without one.
  delete document.documentElement.dataset.theme;
  window.matchMedia = systemTheme;
});

describe("ErrorBoundary", () => {
  it("renders its children untouched while nothing throws", () => {
    renderBoundary(<p>Child content</p>);
    expect(screen.getByText("Child content")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Something went wrong" })).toBeNull();
  });

  it("a child that throws brings up the recovery screen with a twelve-character reference", async () => {
    renderBoundary();
    expect(screen.getByRole("heading", { level: 1, name: "Something went wrong" })).toBeTruthy();
    expect(screen.getByText(/hit an error/)).toBeTruthy();
    expect(screen.getByText(/your saved work .* is safe/i)).toBeTruthy();

    const code = await screen.findByText(REFERENCE);
    expect(code.textContent).toHaveLength(12);
    expect(screen.getByText("Reference")).toBeTruthy();

    expect(screen.getByRole("button", { name: "Reload" })).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "What were you doing?" })).toBeTruthy();
    // a plain link: the full page load is what recovers the app
    const help = document.querySelector('a[href="/help"]');
    expect(help).not.toBeNull();
    expect(screen.queryByText("Child content")).toBeNull();
  });

  it("the error was reported once, as category ui and severity fatal", async () => {
    renderBoundary();
    await screen.findByText(REFERENCE);
    // the page goes away with the report still queued: the beacon is what carries it
    act(() => {
      window.dispatchEvent(new Event("pagehide"));
    });
    const events = await beaconsEvents();
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      category: "ui",
      severity: "fatal",
      error_type: "TypeError",
      message: "render exploded",
    });
    expect(typeof events[0]?.context.componentStack).toBe("string");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("Reload calls location.reload()", async () => {
    // jsdom's location.reload cannot be redefined; the bare global can stand in for it.
    const reload = vi.fn();
    vi.stubGlobal("location", {
      ...window.location,
      href: window.location.href,
      pathname: window.location.pathname,
      origin: window.location.origin,
      reload,
    });
    const user = userEvent.setup();
    renderBoundary();
    await user.click(screen.getByRole("button", { name: "Reload" }));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("typing a note and pressing Send sends one description event carrying the reference, then thanks the user", async () => {
    const user = userEvent.setup();
    renderBoundary();
    const code = (await screen.findByText(REFERENCE)).textContent;
    const box = screen.getByRole("textbox", { name: "What were you doing?" }) as HTMLTextAreaElement;
    expect(box.maxLength).toBe(1000);

    await user.type(box, "I clicked Save on the fit");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText(THANKS);

    const { events } = postedDescription();
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      category: "ui",
      severity: "error",
      error_type: "UserDescription",
      message: "I clicked Save on the fit",
      context: { reference: code },
    });
  });

  it("does not call a note sent when the server refuses it", async () => {
    fetchMock.mockImplementation(() => Promise.resolve(new Response(null, { status: 429 })));
    const user = userEvent.setup();
    renderBoundary();
    await screen.findByText(REFERENCE);
    await user.type(screen.getByRole("textbox", { name: "What were you doing?" }), "a note");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByRole("alert");
    expect(screen.getByRole("alert").textContent).toMatch(/could not be sent/i);
    expect(screen.queryByText(THANKS)).toBeNull();
  });

  it("an empty note cannot be sent", async () => {
    renderBoundary();
    await screen.findByText(REFERENCE);
    const send = screen.getByRole("button", { name: "Send" }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("with consent off there is no box to type in, and the sentence says where to turn reports on", async () => {
    setErrorReportsEnabled(false);
    renderBoundary();
    expect(screen.getByRole("heading", { level: 1, name: "Something went wrong" })).toBeTruthy();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
    // nothing was reported, so there is no reference to quote
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("Reference")).toBeNull();
    const sentence = screen.getByText(/Error reports are off/i).textContent ?? "";
    expect(sentence).toMatch(/About/);
    expect(sentence).toMatch(/Profile → Privacy/);
    act(() => {
      window.dispatchEvent(new Event("pagehide"));
    });
    expect(beaconMock).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("an error that arrives while consent is still being decided is reported once reports are on", async () => {
    // a crash at startup, before the session (and the account's choice) is known
    setConsentResolver(() => false);
    renderBoundary();
    expect(screen.getByText(/Error reports are off/i)).toBeTruthy();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("Reference")).toBeNull();

    // the session turns out to be a visitor's: the browser's setting now counts
    act(() => {
      setConsentResolver(null);
    });
    expect(screen.getByRole("textbox", { name: "What were you doing?" })).toBeTruthy();
    await screen.findByText(REFERENCE);
    act(() => {
      window.dispatchEvent(new Event("pagehide"));
    });
    const events = await beaconsEvents();
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ category: "ui", severity: "fatal" });
  });

  it("reports switched off while the screen is up take the box away", async () => {
    renderBoundary();
    await screen.findByText(REFERENCE);
    act(() => {
      setErrorReportsEnabled(false);
    });
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(screen.queryByText("Reference")).toBeNull();
    expect(screen.getByText(/Error reports are off/i)).toBeTruthy();
  });

  it("consent that dips off and back on before the report is made still ends in one report", async () => {
    renderBoundary();
    // the fingerprint is still being computed while the answer changes twice
    act(() => {
      setConsentResolver(() => false);
    });
    expect(screen.getByText(/Error reports are off/i)).toBeTruthy();
    act(() => {
      setConsentResolver(null);
    });
    await screen.findByText(REFERENCE);
    act(() => {
      window.dispatchEvent(new Event("pagehide"));
    });
    expect(await beaconsEvents()).toHaveLength(1);
  });

  describe("theme", () => {
    it("applies the stored theme itself when the crash came before the app could", async () => {
      localStorage.setItem("curvestudio.theme", "dark");
      renderBoundary();
      await screen.findByText(REFERENCE);
      expect(document.documentElement.dataset.theme).toBe("dark");
    });

    it("follows the system's theme when nothing is stored", async () => {
      window.matchMedia = ((query: string) => ({
        ...systemTheme.call(window, query),
        matches: /dark/.test(query),
      })) as typeof window.matchMedia;
      renderBoundary();
      await screen.findByText(REFERENCE);
      expect(document.documentElement.dataset.theme).toBe("dark");
    });

    it("leaves alone a theme the app had already applied", async () => {
      document.documentElement.dataset.theme = "light";
      localStorage.setItem("curvestudio.theme", "dark");
      renderBoundary();
      await screen.findByText(REFERENCE);
      expect(document.documentElement.dataset.theme).toBe("light");
    });

    it("still shows the screen when the theme cannot be determined", async () => {
      // the very thing that crashed the app: the screen must not crash on it too
      vi.spyOn(window, "matchMedia").mockImplementation(() => {
        throw new TypeError("matchMedia exploded");
      });
      renderBoundary();
      await screen.findByText(REFERENCE);
      expect(document.documentElement.dataset.theme).toBeUndefined();
    });
  });
});
