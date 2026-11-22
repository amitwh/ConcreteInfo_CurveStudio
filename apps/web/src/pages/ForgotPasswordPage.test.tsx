import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { ForgotPasswordPage } from "./ForgotPasswordPage";

function renderForgot() {
  return render(
    <MemoryRouter initialEntries={["/forgot-password"]}>
      <Routes>
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/" element={<main>login page</main>} />
      </Routes>
    </MemoryRouter>,
  );
}

async function submitEmail(user: ReturnType<typeof userEvent.setup>, email: string) {
  await user.type(screen.getByLabelText("Email"), email);
  await user.click(screen.getByRole("button", { name: "Send reset link" }));
}

describe("ForgotPasswordPage", () => {
  it("shows the generic success panel once the call resolves", async () => {
    const forgot = vi.spyOn(api, "forgotPassword").mockResolvedValue({ message: "sent" });
    renderForgot();
    const user = userEvent.setup();
    await submitEmail(user, "a@b.co");
    expect(
      await screen.findByText(/if an account exists, a password reset email has been sent/i),
    ).toBeTruthy();
    expect(forgot).toHaveBeenCalledWith("a@b.co");
  });

  it("shows the same panel for an unknown address", async () => {
    vi.spyOn(api, "forgotPassword").mockResolvedValue({ message: "sent" });
    renderForgot();
    const user = userEvent.setup();
    await submitEmail(user, "nobody@example.test");
    expect(
      await screen.findByText(/if an account exists, a password reset email has been sent/i),
    ).toBeTruthy();
  });

  it("shows a generic error on a network failure", async () => {
    vi.spyOn(api, "forgotPassword").mockRejectedValue(
      new ApiError(0, "NETWORK", "could not reach the server — check your connection", null),
    );
    renderForgot();
    const user = userEvent.setup();
    await submitEmail(user, "a@b.co");
    expect((await screen.findByRole("alert")).textContent).toMatch(/could not reach the server/i);
    expect(screen.queryByText(/a password reset email has been sent/i)).toBeNull();
  });

  it("rejects a malformed email without calling the API", async () => {
    const forgot = vi.spyOn(api, "forgotPassword").mockResolvedValue({ message: "sent" });
    renderForgot();
    const user = userEvent.setup();
    await submitEmail(user, "not-an-email");
    expect((await screen.findByRole("alert")).textContent).toMatch(/valid email/i);
    expect(forgot).not.toHaveBeenCalled();
  });
});
