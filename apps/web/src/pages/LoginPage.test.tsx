import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { AuthProvider } from "../api/auth";
import { LoginPage } from "./LoginPage";

function renderLogin() {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={["/login"]}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<main>workspace</main>} />
        </Routes>
      </MemoryRouter>
    </AuthProvider>,
  );
}

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
    const me = { id: "u1", email: "a@b.co", name: "", created_at: "2026-11-17T05:00:00Z" };
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
});
