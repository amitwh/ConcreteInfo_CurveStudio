import { render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../api/auth";
import { api, type MeOut } from "../api/client";
import {
  errorReportsEnabled,
  setConsentResolver,
  setErrorReportsEnabled,
  subscribeConsent,
} from "../lib/errorReporter";
import { ErrorReportConsent } from "./ErrorReportConsent";

const TOKENS = { access_token: "a2", refresh_token: "r2", expires_in: 900, token_type: "bearer" };

/** A profile with error reports switched off on the account. */
const ME: MeOut = {
  id: "3f2b8a1e-0c4d-4e6f-8a9b-1c2d3e4f5a6b",
  email: "analyst@example.test",
  name: "Analyst",
  created_at: "2026-11-01T10:00:00+05:30",
  email_verified: true,
  role: "member",
  preferences: { error_reports: false, report_byline: false },
  last_login_at: null,
  plan: {
    name: "Free",
    max_datasets: 10,
    max_dataset_bytes: 100_000_000,
    max_fits: 1_00_000,
    max_autofits_per_day: 20,
  },
  usage: { datasets: 1, dataset_bytes: 2000, fits: 2, autofits_today: 0 },
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** A server that restores the session and answers `/me` with `me`. */
function signedInServer(me: MeOut): void {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/auth/refresh")) return Promise.resolve(jsonResponse(200, TOKENS));
      if (url.endsWith("/me")) return Promise.resolve(jsonResponse(200, me));
      return Promise.resolve(jsonResponse(404, { error: "NOT_FOUND", message: "", detail: {} }));
    }),
  );
}

function renderConsent(): void {
  render(
    <AuthProvider>
      <ErrorReportConsent />
    </AuthProvider>,
  );
}

beforeEach(() => {
  setConsentResolver(null);
});

afterEach(() => {
  setConsentResolver(null);
  api.resetForTests();
  vi.unstubAllGlobals();
});

describe("ErrorReportConsent", () => {
  it("renders nothing", () => {
    const view = render(
      <AuthProvider>
        <ErrorReportConsent />
      </AuthProvider>,
    );
    expect(view.container.textContent).toBe("");
  });

  it("a visitor keeps the decision: the browser's own setting is what counts", () => {
    renderConsent();
    setErrorReportsEnabled(false);
    expect(errorReportsEnabled()).toBe(false);
    setErrorReportsEnabled(true);
    expect(errorReportsEnabled()).toBe(true);
  });

  it("a signed-in account's preference decides, whatever the browser says", async () => {
    api.setTokens("a1", "r1");
    setErrorReportsEnabled(true);
    signedInServer(ME);
    renderConsent();
    await waitFor(() => expect(errorReportsEnabled()).toBe(false));
    // the browser's switch does not get a say while the account is signed in
    setErrorReportsEnabled(false);
    expect(errorReportsEnabled()).toBe(false);
    setErrorReportsEnabled(true);
    expect(errorReportsEnabled()).toBe(false);
  });

  it("signing out hands the decision back to the browser", async () => {
    api.setTokens("a1", "r1");
    setErrorReportsEnabled(true);
    signedInServer(ME);
    renderConsent();
    await waitFor(() => expect(errorReportsEnabled()).toBe(false));
    api.onAuthFailure?.();
    await waitFor(() => expect(errorReportsEnabled()).toBe(true));
  });

  it("while a stored session is still being restored, nothing is collected", () => {
    api.setTokens("a1", "r1");
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    renderConsent();
    expect(errorReportsEnabled()).toBe(false);
  });

  it("a visitor with no stored session is never in the dark: the browser's setting counts at once", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    setErrorReportsEnabled(false);
    renderConsent();
    expect(errorReportsEnabled()).toBe(false);
    setErrorReportsEnabled(true);
    expect(errorReportsEnabled()).toBe(true);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("tells the recovery screen when the answer changes", () => {
    renderConsent();
    const told: boolean[] = [];
    const stop = subscribeConsent(() => told.push(errorReportsEnabled()));
    setErrorReportsEnabled(false);
    setErrorReportsEnabled(true);
    expect(told).toEqual([false, true]);
    stop();
    setErrorReportsEnabled(false);
    expect(told).toEqual([false, true]);
  });
});
