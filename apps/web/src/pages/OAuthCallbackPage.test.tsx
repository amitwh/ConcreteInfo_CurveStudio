import { StrictMode } from "react";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { OAuthCallbackPage } from "./OAuthCallbackPage";

const { adoptSession } = vi.hoisted(() => ({ adoptSession: vi.fn() }));

vi.mock("../api/auth", () => ({
  useAuth: () => ({
    status: "anon",
    user: null,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
    adoptSession,
  }),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, oauthExchange: vi.fn() },
  };
});

const tokens = {
  access_token: "access",
  refresh_token: "refresh",
  expires_in: 900,
  token_type: "bearer" as const,
};

function renderCallback(search = "?code=abc") {
  return render(
    <StrictMode>
      <MemoryRouter initialEntries={[`/oauth/callback${search}`]}>
        <Routes>
          <Route path="/oauth/callback" element={<OAuthCallbackPage />} />
          <Route path="/" element={<main>workspace</main>} />
          <Route path="/login" element={<main>login page</main>} />
        </Routes>
      </MemoryRouter>
    </StrictMode>,
  );
}

beforeEach(() => {
  adoptSession.mockReset();
  adoptSession.mockResolvedValue(undefined);
  vi.mocked(api.oauthExchange).mockReset();
});

describe("OAuthCallbackPage", () => {
  it("exchanges the code once, adopts the session and lands in the workspace", async () => {
    vi.mocked(api.oauthExchange).mockResolvedValue(tokens);
    renderCallback();

    expect(await screen.findByText("workspace")).toBeTruthy();
    expect(api.oauthExchange).toHaveBeenCalledTimes(1);
    expect(api.oauthExchange).toHaveBeenCalledWith("abc");
    expect(adoptSession).toHaveBeenCalledTimes(1);
    expect(adoptSession).toHaveBeenCalledWith(tokens);
  });

  it("shows the invalid-link message when the exchange is rejected", async () => {
    vi.mocked(api.oauthExchange).mockRejectedValue(new Error("bad code"));
    renderCallback();

    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid or has expired/i);
    expect(adoptSession).not.toHaveBeenCalled();
  });

  it("shows the invalid-link message when the code is missing", async () => {
    renderCallback("");

    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid or has expired/i);
    expect(api.oauthExchange).not.toHaveBeenCalled();
  });
});
