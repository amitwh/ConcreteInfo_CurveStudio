import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../api/auth";
import { API_BASE, api, ApiError } from "../api/client";
import { RegisterPage } from "./RegisterPage";

const USER = {
  id: "u1",
  email: "a@b.co",
  name: "",
  created_at: "2026-11-17T05:00:00Z",
  email_verified: false,
};

function renderRegister() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={["/register"]}>
        <Routes>
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/" element={<main>workspace</main>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>,
  );
}

async function submitValid(user: ReturnType<typeof userEvent.setup>): Promise<void> {
  await user.type(screen.getByLabelText("Email"), "a@b.co");
  await user.type(screen.getByLabelText("Password (10–128 characters)"), "longenoughpass");
  await user.click(screen.getByRole("button", { name: "Create account" }));
}

describe("RegisterPage", () => {
  // The page asks the API which social providers are configured as soon as it mounts.
  beforeEach(() => {
    vi.spyOn(api, "oauthProviders").mockResolvedValue([]);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the check-your-inbox panel and does not navigate", async () => {
    vi.spyOn(api, "register").mockResolvedValue(USER);
    renderRegister();
    const user = userEvent.setup();
    await submitValid(user);
    expect(await screen.findByRole("heading", { name: "Check your inbox" })).toBeTruthy();
    expect(screen.getByText(/we sent a verification link to a@b\.co/i)).toBeTruthy();
    expect(screen.queryByText("workspace")).toBeNull();
  });

  it("resends the verification email from the success panel", async () => {
    vi.spyOn(api, "register").mockResolvedValue(USER);
    const resend = vi.spyOn(api, "resendVerification").mockResolvedValue({ message: "sent" });
    renderRegister();
    const user = userEvent.setup();
    await submitValid(user);
    await user.click(
      await screen.findByRole("button", { name: "Resend verification email" }),
    );
    expect(resend).toHaveBeenCalledWith("a@b.co");
    expect(
      await screen.findByText(/if an unverified account exists, a new verification email/i),
    ).toBeTruthy();
  });

  it("maps a 409 to a sign-in hint", async () => {
    vi.spyOn(api, "register").mockRejectedValue(
      new ApiError(409, "CONFLICT", "email already registered", {}),
    );
    renderRegister();
    const user = userEvent.setup();
    await submitValid(user);
    expect((await screen.findByRole("alert")).textContent).toMatch(
      /already exists — sign in instead/i,
    );
  });

  it("rejects a short password without calling the API", async () => {
    const register = vi.spyOn(api, "register").mockResolvedValue(USER);
    renderRegister();
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password (10–128 characters)"), "short");
    await user.click(screen.getByRole("button", { name: "Create account" }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/10 and 128/);
    expect(register).not.toHaveBeenCalled();
  });

  it("keeps the form usable after a server error", async () => {
    vi.spyOn(api, "register").mockRejectedValue(
      new ApiError(422, "VALIDATION_ERROR", "the request was rejected", {}),
    );
    renderRegister();
    const user = userEvent.setup();
    await submitValid(user);
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    expect(screen.getByRole("button", { name: "Create account" })).toBeTruthy();
  });

  it("offers the configured social providers and starts the flow when one is picked", async () => {
    vi.mocked(api.oauthProviders).mockResolvedValue(["google", "microsoft"]);
    // jsdom will not let `location.assign` be replaced, so stand in a whole `location`
    // to capture where the button would send the visitor.
    const assign = vi.fn();
    vi.stubGlobal("location", { assign });
    renderRegister();
    const user = userEvent.setup();
    expect(await screen.findByRole("button", { name: "Continue with Google" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Continue with Microsoft" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Continue with Google" }));
    expect(assign).toHaveBeenCalledWith(`${API_BASE}/auth/oauth/google/start`);
  });

  it("renders the social buttons below the form, outside it", async () => {
    vi.mocked(api.oauthProviders).mockResolvedValue(["google"]);
    renderRegister();
    const social = await screen.findByRole("button", { name: /continue with google/i });
    const form = screen.getByRole("button", { name: "Create account" }).closest("form");
    expect(form).not.toBeNull();
    // Outside the form element, so the buttons can never submit it…
    expect(form?.contains(social)).toBe(false);
    // …and visually after it.
    expect(
      (form as HTMLElement).compareDocumentPosition(social) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("renders only the form when no social providers are configured", async () => {
    renderRegister();
    // Let the (empty) discovery settle before asserting on what is absent.
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /continue with/i })).toBeNull();
    expect(document.querySelector(".oauth-block")).toBeNull();
    const main = screen.getByRole("main");
    expect(main.children).toHaveLength(1);
    expect(main.firstElementChild?.tagName).toBe("FORM");
  });
});
