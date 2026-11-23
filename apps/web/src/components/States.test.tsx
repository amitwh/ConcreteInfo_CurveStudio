import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { ErrorBox, apiErrorText, errorText } from "./States";

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

describe("apiErrorText names the field the server refused", () => {
  // Every API 422 carries the same generic message; the reason is in detail.errors[0].msg.
  const rejected = (detail: unknown): ApiError =>
    new ApiError(422, "VALIDATION_ERROR", "the request is not valid", detail);

  it("prefers detail.errors[0].msg for a 422", () => {
    expect(
      apiErrorText(rejected({ errors: [{ loc: ["body", "name"], msg: "name is required", type: "value_error" }] })),
    ).toBe("name is required");
  });

  it("falls back to the message when a 422 carries no reason to show", () => {
    expect(apiErrorText(rejected({}))).toBe("the request is not valid");
    expect(apiErrorText(rejected({ errors: [] }))).toBe("the request is not valid");
    expect(apiErrorText(rejected({ errors: [{ msg: 42 }] }))).toBe("the request is not valid");
  });

  it("keeps the server's own message for every other status", () => {
    expect(apiErrorText(new ApiError(403, "FORBIDDEN", "the current password is not correct", {}))).toBe(
      "the current password is not correct",
    );
    expect(apiErrorText(new ApiError(429, "RATE_LIMITED", "too many attempts", { errors: [{ msg: "x" }] }))).toBe(
      "too many attempts",
    );
  });

  it("behaves like errorText for anything that is not an ApiError", () => {
    expect(apiErrorText(new Error("boom"))).toBe("boom");
    expect(apiErrorText("a string")).toBe("a string");
    expect(apiErrorText({ unexpected: true }, "something went wrong")).toBe("something went wrong");
  });
});
