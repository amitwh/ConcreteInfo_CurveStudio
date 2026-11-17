import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { ErrorBox, errorText } from "./States";

describe("errorText turns any thrown value into a user-facing message", () => {
  it("passes a plain string through (components setError with strings)", () => {
    expect(errorText("every x value must be a finite number")).toBe(
      "every x value must be a finite number",
    );
  });

  it("uses the fallback for an empty string", () => {
    expect(errorText("", "something went wrong")).toBe("something went wrong");
  });

  it("keeps ApiError and Error handling", () => {
    expect(errorText(new ApiError(401, "UNAUTHORIZED", "invalid email or password", {}))).toBe(
      "invalid email or password",
    );
    expect(errorText(new Error("boom"))).toBe("boom");
    expect(errorText({ unexpected: true })).toBe("something went wrong");
  });

  it("renders a plain-string error in the ErrorBox", () => {
    render(<ErrorBox error="every x value must be a finite number" />);
    expect(screen.getByRole("alert").textContent).toBe("every x value must be a finite number");
  });
});
