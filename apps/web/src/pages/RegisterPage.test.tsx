import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../api/auth";
import { api, ApiError } from "../api/client";
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
});
