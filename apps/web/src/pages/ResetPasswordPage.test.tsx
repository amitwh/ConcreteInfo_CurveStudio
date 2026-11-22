import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { ResetPasswordPage } from "./ResetPasswordPage";

function renderReset(path = "/reset-password?token=tok") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/reset-password" element={<ResetPasswordPage />} />
        <Route path="/login" element={<main>login page</main>} />
        <Route path="/forgot-password" element={<main>forgot page</main>} />
      </Routes>
    </MemoryRouter>,
  );
}

async function submit(
  user: ReturnType<typeof userEvent.setup>,
  password: string,
  confirm: string,
): Promise<void> {
  await user.type(screen.getByLabelText("New password (10–128 characters)"), password);
  await user.type(screen.getByLabelText("Confirm new password"), confirm);
  await user.click(screen.getByRole("button", { name: "Reset password" }));
}

describe("ResetPasswordPage", () => {
  it("rejects mismatched passwords without calling the API", async () => {
    const reset = vi.spyOn(api, "resetPassword").mockResolvedValue({ message: "ok" });
    renderReset();
    const user = userEvent.setup();
    await submit(user, "longenoughpass", "differentpass");
    expect((await screen.findByRole("alert")).textContent).toMatch(/do not match/i);
    expect(reset).not.toHaveBeenCalled();
  });

  it("rejects a password shorter than 10 characters", async () => {
    const reset = vi.spyOn(api, "resetPassword").mockResolvedValue({ message: "ok" });
    renderReset();
    const user = userEvent.setup();
    await submit(user, "short", "short");
    expect((await screen.findByRole("alert")).textContent).toMatch(/10 and 128/);
    expect(reset).not.toHaveBeenCalled();
  });

  it("submits the token and password, then shows the success panel", async () => {
    const reset = vi.spyOn(api, "resetPassword").mockResolvedValue({ message: "ok" });
    renderReset();
    const user = userEvent.setup();
    await submit(user, "longenoughpass", "longenoughpass");
    expect(
      await screen.findByText(/your password has been reset\. sign in with the new one/i),
    ).toBeTruthy();
    expect(reset).toHaveBeenCalledWith("tok", "longenoughpass");
  });

  it("shows the invalid-link state when the token is missing", async () => {
    const reset = vi.spyOn(api, "resetPassword");
    renderReset("/reset-password");
    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid or has expired/i);
    expect(screen.queryByLabelText("New password (10–128 characters)")).toBeNull();
    expect(reset).not.toHaveBeenCalled();
  });

  it("maps an API 400 INVALID_TOKEN to the invalid-link message", async () => {
    vi.spyOn(api, "resetPassword").mockRejectedValue(
      new ApiError(400, "INVALID_TOKEN", "invalid token", {}),
    );
    renderReset();
    const user = userEvent.setup();
    await submit(user, "longenoughpass", "longenoughpass");
    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid or has expired/i);
  });
});
