import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { overlayFor } from "./authenticated/AuthenticatedApp";

const { authState } = vi.hoisted(() => ({
  authState: {
    user: { id: "u1", email: "analyst@example.test", name: "", role: "member" } as {
      id: string;
      email: string;
      name: string;
      role: "member" | "admin";
    },
  },
}));

vi.mock("./api/auth", () => ({
  useAuth: () => ({ status: "authed", user: authState.user, logout: vi.fn() }),
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

// The console itself is real here (its guard and its routing are what the app shell relies
// on); only its pages are stand-ins with their real heading.
vi.mock("./admin/OverviewPage", () => ({ OverviewPage: () => <h2>Overview</h2> }));
vi.mock("./admin/InstallsPage", () => ({ InstallsPage: () => <h2>Installs</h2> }));
vi.mock("./admin/AuditPage", () => ({ AuditPage: () => <h2>Audit</h2> }));
vi.mock("./admin/SystemPage", () => ({ SystemPage: () => <h2>System</h2> }));
vi.mock("./admin/PlansPage", () => ({ PlansPage: () => <h2>Plans</h2> }));

beforeEach(() => {
  authState.user = { id: "u1", email: "analyst@example.test", name: "", role: "member" };
});

afterEach(cleanup);

/** Reports the URL the router settled on. */
function LocationProbe(): React.ReactElement {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderApp(path = "/"): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <LocationProbe />
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
    expect(await screen.findByTestId("workspace")).toBeTruthy();

    await user.click(screen.getByRole("link", { name: "Help" }));
    expect(screen.getByRole("dialog", { name: "Help contents" })).toBeTruthy();
    expect(await screen.findByText("Help body")).toBeTruthy();
    // the main interface is still there, underneath the overlay
    expect(screen.getByTestId("workspace")).toBeTruthy();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByTestId("workspace")).toBeTruthy();
  });

  it("does the same for About", async () => {
    const user = userEvent.setup();
    renderApp("/about");
    expect(await screen.findByRole("dialog", { name: "About CurveStudio" })).toBeTruthy();
    expect(await screen.findByTestId("workspace")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByTestId("workspace")).toBeTruthy();
  });

  it("opens the profile over the workspace without unmounting it", async () => {
    const user = userEvent.setup();
    renderApp("/profile");
    const dialog = await screen.findByRole("dialog", { name: "Your profile" });
    expect(dialog.className).toContain("modal__panel--sheet");
    expect(await screen.findByText("Profile body")).toBeTruthy();
    // the main interface is still there, underneath the overlay
    expect(await screen.findByTestId("workspace")).toBeTruthy();
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
    expect(await screen.findByRole("tablist", { name: "Workspaces" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "New workspace" }));
    expect(screen.getAllByRole("button", { name: /^Workspace \d$/ })).toHaveLength(2);
  });
});

describe("App shell — the admin console", () => {
  it("is a page of its own under /admin: the workspace is not rendered beneath it", async () => {
    authState.user = { ...authState.user, role: "admin" };
    renderApp("/admin");
    expect(await screen.findByRole("heading", { level: 1, name: "Admin console" })).toBeTruthy();
    expect(screen.getByRole("heading", { level: 2, name: "Overview" })).toBeTruthy();
    expect(screen.queryByTestId("workspace")).toBeNull();
    expect(screen.queryByRole("tablist", { name: "Workspaces" })).toBeNull();
    expect(screen.getByTestId("location").textContent).toBe("/admin");
  });

  it("routes its own pages without ever mounting the workspace", async () => {
    authState.user = { ...authState.user, role: "admin" };
    renderApp("/admin/audit");
    expect(await screen.findByRole("heading", { level: 2, name: "Audit" })).toBeTruthy();
    expect(screen.queryByTestId("workspace")).toBeNull();
  });

  it("redirects a member from /admin to the workspace route", async () => {
    renderApp("/admin");
    expect(await screen.findByTestId("workspace")).toBeTruthy();
    expect(screen.getByTestId("location").textContent).toBe("/");
    expect(screen.queryByRole("heading", { name: "Admin console" })).toBeNull();
  });

  it("returns to the workspace through Back to CurveStudio", async () => {
    const user = userEvent.setup();
    authState.user = { ...authState.user, role: "admin" };
    renderApp("/admin/plans");
    await user.click(await screen.findByRole("link", { name: "Back to CurveStudio" }));
    expect(await screen.findByTestId("workspace")).toBeTruthy();
    expect(screen.getByTestId("location").textContent).toBe("/");
    expect(screen.queryByRole("navigation", { name: "Admin" })).toBeNull();
    expect(within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name: "Workspace" }).getAttribute("aria-current")).toBe("page");
  });
});
