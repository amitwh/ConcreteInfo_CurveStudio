import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AdminConsole from "./AdminConsole";

const { authState } = vi.hoisted(() => ({
  authState: {
    user: { id: "u1", email: "admin@example.test", name: "The Admin", role: "admin" } as {
      id: string;
      email: string;
      name: string;
      role: "member" | "admin";
    } | null,
  },
}));

vi.mock("../api/auth", () => ({
  useAuth: () => ({ status: "authed", user: authState.user, logout: vi.fn() }),
}));

// The pages are stand-ins with their real heading: the shell's job is who may see it, the
// layout and the routing between them.
vi.mock("./OverviewPage", () => ({ OverviewPage: () => <h2>Overview</h2> }));
vi.mock("./InstallsPage", () => ({ InstallsPage: () => <h2>Installs</h2> }));
vi.mock("./AuditPage", () => ({ AuditPage: () => <h2>Audit</h2> }));
vi.mock("./SystemPage", () => ({ SystemPage: () => <h2>System</h2> }));
vi.mock("./PlansPage", () => ({ PlansPage: () => <h2>Plans</h2> }));

beforeEach(() => {
  authState.user = { id: "u1", email: "admin@example.test", name: "The Admin", role: "admin" };
});

afterEach(cleanup);

/** Reports the URL the router settled on. */
function LocationProbe(): React.ReactElement {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

/** The console under the same parent route the app gives it. */
function renderConsole(path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <LocationProbe />
      <Routes>
        <Route path="/admin/*" element={<AdminConsole />} />
      </Routes>
    </MemoryRouter>,
  );
}

const PAGES = ["Overview", "Installs", "Audit", "System", "Plans"] as const;

describe("AdminConsole — who may see it", () => {
  it("sends a member back to the workspace", () => {
    authState.user = { ...authState.user, role: "member" } as NonNullable<typeof authState.user>;
    renderConsole("/admin");
    expect(screen.getByTestId("location").textContent).toBe("/");
    expect(screen.queryByRole("heading", { name: "Admin console" })).toBeNull();
  });

  it("sends a visitor back to the workspace", () => {
    authState.user = null;
    renderConsole("/admin/installs");
    expect(screen.getByTestId("location").textContent).toBe("/");
    expect(screen.queryByRole("navigation", { name: "Admin" })).toBeNull();
  });
});

describe("AdminConsole — layout", () => {
  it("wears a slim header with a way back and the title", () => {
    renderConsole("/admin");
    const back = screen.getByRole("link", { name: "Back to CurveStudio" });
    expect(back.getAttribute("href")).toBe("/");
    expect(screen.getByRole("heading", { level: 1, name: "Admin console" })).toBeTruthy();
    // the workspace's own header is not here
    expect(screen.queryByRole("tablist", { name: "Workspaces" })).toBeNull();
  });

  it("lists the five pages in order, marks the current one, and names no page that does not exist", () => {
    renderConsole("/admin");
    const nav = screen.getByRole("navigation", { name: "Admin" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((link) => link.textContent)).toEqual([...PAGES]);
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/admin",
      "/admin/installs",
      "/admin/audit",
      "/admin/system",
      "/admin/plans",
    ]);
    expect(within(nav).queryByRole("link", { name: /accounts/i })).toBeNull();
    expect(within(nav).queryByRole("link", { name: /errors/i })).toBeNull();
    expect(within(nav).getByRole("link", { name: "Overview" }).getAttribute("aria-current")).toBe(
      "page",
    );
    expect(within(nav).getAllByRole("link", { current: "page" })).toHaveLength(1);
  });
});

describe("AdminConsole — routing", () => {
  it.each([
    ["/admin", "Overview"],
    ["/admin/installs", "Installs"],
    ["/admin/audit", "Audit"],
    ["/admin/system", "System"],
    ["/admin/plans", "Plans"],
  ])("%s shows the %s page", (path, page) => {
    renderConsole(path);
    expect(screen.getByRole("heading", { level: 2, name: page })).toBeTruthy();
    expect(screen.getAllByRole("heading", { level: 2 })).toHaveLength(1);
    expect(within(screen.getByRole("navigation", { name: "Admin" })).getByRole("link", { name: page }).getAttribute("aria-current")).toBe("page");
  });

  it("renders each page's heading as its navigation link is followed", async () => {
    const user = userEvent.setup();
    renderConsole("/admin");
    for (const page of PAGES) {
      await user.click(within(screen.getByRole("navigation", { name: "Admin" })).getByRole("link", { name: page }));
      expect(screen.getByRole("heading", { level: 2, name: page })).toBeTruthy();
      expect(within(screen.getByRole("navigation", { name: "Admin" })).getByRole("link", { name: page }).getAttribute("aria-current")).toBe("page");
      expect(screen.getByTestId("location").textContent).toBe(page === "Overview" ? "/admin" : `/admin/${page.toLowerCase()}`);
    }
  });

  it.each([["/admin/nope"], ["/admin/installs/extra"], ["/admin/accounts"]])(
    "redirects %s, which is not a page, to the overview",
    (path) => {
      renderConsole(path);
      expect(screen.getByTestId("location").textContent).toBe("/admin");
      expect(screen.getByRole("heading", { level: 2, name: "Overview" })).toBeTruthy();
    },
  );
});
