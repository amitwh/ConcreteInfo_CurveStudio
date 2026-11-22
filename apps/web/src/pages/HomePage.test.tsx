import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { HomePage } from "./HomePage";

const { loginMock } = vi.hoisted(() => ({ loginMock: vi.fn() }));

vi.mock("../api/auth", () => ({
  useAuth: () => ({
    status: "anon",
    user: null,
    login: loginMock,
    register: vi.fn(),
    logout: vi.fn(),
    adoptSession: vi.fn(),
  }),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: { ...actual.api, oauthProviders: vi.fn(() => Promise.resolve([])) },
  };
});

function renderHome() {
  return render(
    <MemoryRouter initialEntries={["/start"]}>
      <Routes>
        <Route path="/start" element={<HomePage />} />
        <Route path="/" element={<main>workspace</main>} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  loginMock.mockReset();
  loginMock.mockResolvedValue(undefined);
  vi.mocked(api.oauthProviders).mockResolvedValue([]);
});

describe("HomePage", () => {
  it("renders the headline, the sign-in card and the section headings", async () => {
    renderHome();

    expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(
      /defensible model/i,
    );
    expect(screen.getByLabelText("Email")).toBeTruthy();
    expect(screen.getByLabelText("Password")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeTruthy();

    expect(screen.getByRole("heading", { name: /fit the data/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /four steps/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /library that covers/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /explain to a reviewer/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /ranking is the result/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /curves engineers/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /one engine/i })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /questions, answered/i })).toBeTruthy();

    // Settle the provider discovery so its state update happens inside act().
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
  });

  it("submits the login form and navigates to the workspace", async () => {
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect(loginMock).toHaveBeenCalledWith("a@b.co", "longenoughpass");
    await waitFor(() => expect(screen.getByText("workspace")).not.toBeNull());
  });

  it("shows the server error inline when sign-in fails", async () => {
    loginMock.mockRejectedValue(
      new ApiError(401, "UNAUTHORIZED", "invalid email or password", {}),
    );
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid email or password/i);
  });

  it("shows the verify-your-email note on 403 EMAIL_NOT_VERIFIED", async () => {
    loginMock.mockRejectedValue(
      new ApiError(403, "FORBIDDEN", "verify your email before signing in", {
        code: "EMAIL_NOT_VERIFIED",
      }),
    );
    const user = userEvent.setup();
    renderHome();

    await user.type(screen.getByLabelText("Email"), "a@b.co");
    await user.type(screen.getByLabelText("Password"), "longenoughpass");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/verify your email/i);
    expect(screen.getByRole("link", { name: "sign in" })).toBeTruthy();
  });

  it("hides the social buttons when no providers are configured", async () => {
    renderHome();
    await waitFor(() => expect(api.oauthProviders).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /continue with google/i })).toBeNull();
  });

  it("shows the social button when a provider is configured", async () => {
    vi.mocked(api.oauthProviders).mockResolvedValue(["google"]);
    renderHome();
    expect(await screen.findByRole("button", { name: /continue with google/i })).toBeTruthy();
  });
});
