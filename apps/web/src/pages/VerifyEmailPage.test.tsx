import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { VerifyEmailPage } from "./VerifyEmailPage";

function renderVerify(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/verify-email" element={<VerifyEmailPage />} />
        <Route path="/" element={<main>login page</main>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("VerifyEmailPage", () => {
  it("calls the API once with the query token and shows success", async () => {
    const verify = vi.spyOn(api, "verifyEmail").mockResolvedValue({
      email: "a@b.co",
      verified: true,
    });
    renderVerify("/verify-email?token=abc");
    expect(await screen.findByText(/your email is verified/i)).toBeTruthy();
    expect(verify).toHaveBeenCalledTimes(1);
    expect(verify).toHaveBeenCalledWith("abc");
  });

  it("shows the invalid state for a rejected token", async () => {
    vi.spyOn(api, "verifyEmail").mockRejectedValue(
      new ApiError(400, "INVALID_TOKEN", "this verification link is invalid or has expired", {}),
    );
    renderVerify("/verify-email?token=stale");
    expect((await screen.findByRole("alert")).textContent).toMatch(/invalid or has expired/i);
  });

  it("shows the missing-token state without calling the API", async () => {
    const verify = vi.spyOn(api, "verifyEmail");
    renderVerify("/verify-email");
    expect((await screen.findByRole("alert")).textContent).toMatch(/missing its token/i);
    expect(verify).not.toHaveBeenCalled();
  });
});
