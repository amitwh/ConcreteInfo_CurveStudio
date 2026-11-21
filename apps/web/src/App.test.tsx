import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "./App";
import { AuthProvider } from "./api/auth";

function renderApp(entry: string) {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[entry]}>
        <App />
      </MemoryRouter>
    </AuthProvider>,
  );
}

describe("App shell", () => {
  it("has a Help link in the main navigation next to Workspace and About", () => {
    renderApp("/about");
    const nav = screen.getByRole("navigation", { name: "Main" });
    const links = within(nav)
      .getAllByRole("link")
      .map((link) => [link.textContent?.trim(), link.getAttribute("href")]);
    expect(links).toEqual([
      ["Workspace", "/"],
      ["Help", "/help"],
      ["About", "/about"],
    ]);
  });

  it("renders the help page at /help", () => {
    renderApp("/help");
    expect(screen.getByRole("heading", { level: 1, name: /help/i })).toBeTruthy();
  });

  it("gives the icon-only theme toggle an accessible name", () => {
    renderApp("/about");
    const toggle = screen.getByRole("button", { name: /switch to (light|dark) theme/i });
    // Icon-only: the name comes from aria-label, and the decorative svg is hidden.
    expect(toggle.textContent).toBe("");
    const icon = toggle.querySelector("svg");
    expect(icon).not.toBeNull();
    expect(icon?.getAttribute("aria-hidden")).toBe("true");
    expect(icon?.getAttribute("focusable")).toBe("false");
  });
});
