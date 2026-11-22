import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { AuthProvider } from "../api/auth";
import { LoginPage } from "./LoginPage";

function renderLogin(path = "/login") {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<main>workspace</main>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>,
  );
}

beforeEach(() => {
  // Social discovery is an API call; keep the form tests offline.
  vi.spyOn(api, "oauthProviders").mockResolvedValue([]);
});

describe("LoginPage", () => {
  it("rejects a malformed email without calling the API", async () => {
    const login = vi.spyOn(api, "login");
    renderLogin();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "not-an-email");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/email/i);
    expect(login).not.toHaveBeenCalled();
  });

  it("rejects a password shorter than 10 characters", async () => {
    const login = vi.spyOn(api, "login");
    renderLogin();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "short");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/10/);
    expect(login).not.toHaveBeenCalled();
  });

  it("shows the server's error inline on a 401", async () => {
    vi.spyOn(api, "login").mockRejectedValue(
      new ApiError(401, "UNAUTHORIZED", "invalid email or password", {}),
    );
    renderLogin();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid email or password/i);
  });

  it("navigates to the workspace on success", async () => {
    const me = {
      id: "u1",
      email: "a@b.co",
      name: "",
      created_at: "2026-11-17T05:00:00Z",
      email_verified: true,
    };
    vi.spyOn(api, "login").mockResolvedValue({
      access_token: "a",
      refresh_token: "r",
      expires_in: 900,
      token_type: "bearer",
    });
    vi.spyOn(api, "me").mockResolvedValue(me);
    renderLogin();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(screen.getByText("workspace")).not.toBeNull());
  });

  it("shows the verification message and a resend button on 403 EMAIL_NOT_VERIFIED", async () => {
    vi.spyOn(api, "login").mockRejectedValue(
      new ApiError(403, "FORBIDDEN", "verify your email before signing in", {
        code: "EMAIL_NOT_VERIFIED",
      }),
    );
    renderLogin();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/verify your email/i);
    expect(screen.getByRole("button", { name: "Resend verification email" })).toBeTruthy();
  });

  it("resends the verification email to the address already typed", async () => {
    vi.spyOn(api, "login").mockRejectedValue(
      new ApiError(403, "FORBIDDEN", "verify your email before signing in", {
        code: "EMAIL_NOT_VERIFIED",
      }),
    );
    const resend = vi.spyOn(api, "resendVerification").mockResolvedValue({ message: "sent" });
    renderLogin();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await user.click(await screen.findByRole("button", { name: "Resend verification email" }));
    expect(resend).toHaveBeenCalledWith("a@b.co");
    expect(
      await screen.findByText(/if an unverified account exists, a new verification email/i),
    ).toBeTruthy();
  });

  it("shows a friendly message when redirected back with ?oauth_error=provider", () => {
    renderLogin("/login?oauth_error=provider");
    expect(screen.getByRole("alert").textContent).toMatch(/provider could not complete sign-in/i);
  });

  it("falls back to a generic message for an unknown oauth_error", () => {
    renderLogin("/login?oauth_error=something-new");
    expect(screen.getByRole("alert").textContent).toMatch(
      /social sign-in could not be completed/i,
    );
  });

  it("hides the social buttons when no providers are configured", async () => {
    renderLogin();
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /continue with google/i })).toBeNull();
  });

  it("renders the provider buttons when configured", async () => {
    vi.spyOn(api, "oauthProviders").mockResolvedValue(["google", "microsoft"]);
    renderLogin();
    expect(await screen.findByRole("button", { name: /continue with google/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: /continue with microsoft/i })).toBeTruthy();
  });
});
