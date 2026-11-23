import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App, { overlayFor } from "./App";

vi.mock("./api/auth", () => ({
  useAuth: () => ({
    status: "authed",
    user: { id: "u1", email: "analyst@example.test", name: "" },
    logout: vi.fn(),
  }),
}));

vi.mock("./components/ThemeToggle", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useTheme: () => ({ theme: "light", toggle: vi.fn() }),
}));

vi.mock("./components/useInstallPrompt", () => ({
  useInstallPrompt: () => ({ canInstall: false, install: vi.fn() }),
}));

// The workspace is the thing that must survive navigation; a marker is enough here.
vi.mock("./pages/WorkspacePage", () => ({
  WorkspacePage: () => <div data-testid="workspace">workspace</div>,
}));
vi.mock("./pages/HelpPage", () => ({ HelpPage: () => <p>Help body</p> }));
vi.mock("./pages/AboutPage", () => ({ AboutPage: () => <p>About body</p> }));
vi.mock("./pages/ProfilePage", () => ({ ProfilePage: () => <p>Profile body</p> }));

afterEach(cleanup);

function renderApp(path = "/"): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("overlayFor", () => {
  it("maps the overlay URLs to their overlays and everything else to none", () => {
    expect(overlayFor("/help")).toBe("help");
    expect(overlayFor("/about")).toBe("about");
    expect(overlayFor("/profile")).toBe("profile");
    expect(overlayFor("/")).toBeNull();
    expect(overlayFor("/anything")).toBeNull();
  });
});

describe("App shell", () => {
  it("opens Help over the workspace without unmounting it", async () => {
    const user = userEvent.setup();
    renderApp();
    expect(screen.getByTestId("workspace")).toBeTruthy();

    await user.click(screen.getByRole("link", { name: "Help" }));
    expect(screen.getByRole("dialog", { name: "Help contents" })).toBeTruthy();
    expect(screen.getByText("Help body")).toBeTruthy();
    // the main interface is still there, underneath the overlay
    expect(screen.getByTestId("workspace")).toBeTruthy();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByTestId("workspace")).toBeTruthy();
  });

  it("does the same for About", async () => {
    const user = userEvent.setup();
    renderApp("/about");
    expect(screen.getByRole("dialog", { name: "About CurveStudio" })).toBeTruthy();
    expect(screen.getByTestId("workspace")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByTestId("workspace")).toBeTruthy();
  });

  it("opens the profile over the workspace without unmounting it", async () => {
    const user = userEvent.setup();
    renderApp("/profile");
    const dialog = screen.getByRole("dialog", { name: "Your profile" });
    expect(dialog.className).toContain("modal__panel--sheet");
    expect(screen.getByText("Profile body")).toBeTruthy();
    // the main interface is still there, underneath the overlay
    expect(screen.getByTestId("workspace")).toBeTruthy();
    // an overlay is open, so the Workspace link is no longer the current page
    expect(screen.getByRole("link", { name: "Workspace" }).getAttribute("aria-current")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByTestId("workspace")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Workspace" }).getAttribute("aria-current")).toBe(
      "page",
    );
  });

  it("shows the workspace tabs and starts a second workspace", async () => {
    const user = userEvent.setup();
    renderApp();
    expect(screen.getByRole("tablist", { name: "Workspaces" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "New workspace" }));
    expect(screen.getAllByRole("button", { name: /^Workspace \d$/ })).toHaveLength(2);
  });
});
