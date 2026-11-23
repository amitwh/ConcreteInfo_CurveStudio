import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Workspace } from "../workspace/workspaces";
import { AppNav } from "./AppNav";

const { authState } = vi.hoisted(() => ({
  authState: {
    user: { id: "u1", email: "analyst@example.test", name: "", role: "member" },
    logout: vi.fn(),
  },
}));

vi.mock("../api/auth", () => ({
  useAuth: () => ({ status: "authed", user: authState.user, logout: authState.logout }),
}));

vi.mock("./ThemeToggle", () => ({
  useTheme: () => ({ theme: "light", toggle: vi.fn() }),
}));

vi.mock("./useInstallPrompt", () => ({
  useInstallPrompt: () => ({ canInstall: false, install: vi.fn() }),
}));

const WORKSPACES: Workspace[] = [
  { id: "ws-1", name: "Cube tests", datasetId: "ds-1", mode: "auto" },
  { id: "ws-2", name: "Mix trial", datasetId: null, mode: "manual" },
];

beforeEach(() => {
  authState.user = { id: "u1", email: "analyst@example.test", name: "", role: "member" };
  authState.logout.mockReset();
});

afterEach(cleanup);

/** Reports the URL the router settled on, so the Profile item's navigation can be asserted. */
function LocationProbe(): React.ReactElement {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderNav(overrides: Partial<Parameters<typeof AppNav>[0]> = {}) {
  const handlers = {
    onSelectWorkspace: vi.fn(),
    onAddWorkspace: vi.fn(),
    onCloseWorkspace: vi.fn(),
    onRenameWorkspace: vi.fn(),
  };
  const props = {
    workspaces: WORKSPACES,
    activeId: "ws-1",
    overlay: null,
    ...handlers,
    ...overrides,
  } as const;
  const view = render(
    <MemoryRouter>
      <AppNav {...props} />
      <LocationProbe />
    </MemoryRouter>,
  );
  return { ...handlers, ...view };
}

/** The account chip: named after the account it belongs to. */
function accountButton(): HTMLElement {
  return screen.getByRole("button", { name: /analyst@example\.test/ });
}

async function openMenu(user: ReturnType<typeof userEvent.setup>): Promise<HTMLElement> {
  await user.click(accountButton());
  return screen.getByRole("menu");
}

describe("AppNav", () => {
  it("marks the current destination and links Help and About to their URLs", () => {
    renderNav({ overlay: "help" });
    const workspace = screen.getByRole("link", { name: "Workspace" });
    const help = screen.getByRole("link", { name: "Help" });
    const about = screen.getByRole("link", { name: "About" });
    expect(workspace.getAttribute("href")).toBe("/");
    expect(help.getAttribute("href")).toBe("/help");
    expect(about.getAttribute("href")).toBe("/about");
    expect(help.getAttribute("aria-current")).toBe("page");
    expect(workspace.getAttribute("aria-current")).toBeNull();
  });

  it("lists the workspaces and activates one on click", async () => {
    const user = userEvent.setup();
    const { onSelectWorkspace } = renderNav();
    expect(screen.getByRole("tablist", { name: "Workspaces" })).toBeTruthy();
    const active = screen.getByRole("button", { name: "Cube tests" });
    expect(active.getAttribute("aria-current")).toBe("true");
    await user.click(screen.getByRole("button", { name: "Mix trial" }));
    expect(onSelectWorkspace).toHaveBeenCalledWith("ws-2");
  });

  it("adds a workspace with the + control", async () => {
    const user = userEvent.setup();
    const { onAddWorkspace } = renderNav();
    await user.click(screen.getByRole("button", { name: "New workspace" }));
    expect(onAddWorkspace).toHaveBeenCalledTimes(1);
  });

  it("closes a workspace", async () => {
    const user = userEvent.setup();
    const { onCloseWorkspace } = renderNav();
    await user.click(screen.getByRole("button", { name: "Close Mix trial" }));
    expect(onCloseWorkspace).toHaveBeenCalledWith("ws-2");
  });

  it("never offers to close the last workspace", () => {
    renderNav({ workspaces: [WORKSPACES[0] as Workspace] });
    expect(screen.queryByRole("button", { name: /^Close / })).toBeNull();
  });

  it("renames on double-click with Enter, and keeps the old name on Escape", async () => {
    const user = userEvent.setup();
    const { onRenameWorkspace } = renderNav();
    const tab = screen.getByRole("button", { name: "Cube tests" });

    fireEvent.doubleClick(tab);
    const input = screen.getByRole("textbox", { name: /rename cube tests/i });
    await user.clear(input);
    await user.type(input, "Slab trial{Enter}");
    expect(onRenameWorkspace).toHaveBeenCalledWith("ws-1", "Slab trial");

    fireEvent.doubleClick(screen.getByRole("button", { name: "Cube tests" }));
    const again = screen.getByRole("textbox", { name: /rename cube tests/i });
    await user.clear(again);
    await user.type(again, "Discarded{Escape}");
    expect(onRenameWorkspace).toHaveBeenCalledTimes(1);
  });
});

describe("AppNav account menu", () => {
  it("the account chip is a menu button, closed, showing the e-mail in the open", () => {
    renderNav();
    const chip = accountButton();
    expect(chip.tagName).toBe("BUTTON");
    expect(chip.getAttribute("type")).toBe("button");
    expect(chip.getAttribute("aria-haspopup")).toBe("menu");
    expect(chip.getAttribute("aria-expanded")).toBe("false");
    expect(within(chip).getByText("analyst@example.test")).toBeTruthy();
    expect(screen.queryByRole("menu")).toBeNull();
    // sign-out lives in the menu: there is no second, standalone control
    expect(screen.queryByRole("button", { name: "Sign out" })).toBeNull();
  });

  it("shows the name beside the e-mail when the account has one", () => {
    authState.user = { ...authState.user, name: "Analyst One" };
    renderNav();
    const chip = accountButton();
    expect(within(chip).getByText("Analyst One")).toBeTruthy();
    expect(within(chip).getByText("analyst@example.test")).toBeTruthy();
  });

  it("opens a menu with the account as a header and two items, and focuses the first", async () => {
    const user = userEvent.setup();
    authState.user = { ...authState.user, name: "Analyst One" };
    const { container } = renderNav();
    const chip = accountButton();

    const menu = await openMenu(user);
    expect(chip.getAttribute("aria-expanded")).toBe("true");
    const header = container.querySelector(".account-menu__header");
    expect(header?.textContent).toContain("Analyst One");
    expect(header?.textContent).toContain("analyst@example.test");
    const items = within(menu).getAllByRole("menuitem");
    expect(items.map((item) => item.textContent)).toEqual(["Profile", "Sign out"]);
    // opening the menu moves focus to its first item
    expect(document.activeElement).toBe(items[0]);
  });

  it("Profile is a link to /profile and the menu closes behind it", async () => {
    const user = userEvent.setup();
    renderNav();
    const menu = await openMenu(user);
    const profile = within(menu).getByRole("menuitem", { name: "Profile" });
    expect(profile.getAttribute("href")).toBe("/profile");

    await user.click(profile);
    expect(screen.getByTestId("location").textContent).toBe("/profile");
    expect(screen.queryByRole("menu")).toBeNull();
    expect(accountButton().getAttribute("aria-expanded")).toBe("false");
  });

  it("Sign out calls logout", async () => {
    const user = userEvent.setup();
    renderNav();
    const menu = await openMenu(user);
    await user.click(within(menu).getByRole("menuitem", { name: "Sign out" }));
    expect(authState.logout).toHaveBeenCalledTimes(1);
  });

  it("Escape closes the menu and returns focus to the chip", async () => {
    const user = userEvent.setup();
    renderNav();
    const chip = accountButton();
    await openMenu(user);
    expect(document.activeElement).not.toBe(chip);

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).toBeNull();
    expect(chip.getAttribute("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(chip);
    expect(authState.logout).not.toHaveBeenCalled();
  });

  it("closes on a click outside, and toggles on the chip", async () => {
    const user = userEvent.setup();
    const { container } = renderNav();
    await openMenu(user);
    await user.click(container.querySelector("header") as HTMLElement);
    expect(screen.queryByRole("menu")).toBeNull();

    await openMenu(user);
    await user.click(accountButton());
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("moves between the items with the arrow keys", async () => {
    const user = userEvent.setup();
    renderNav();
    const menu = await openMenu(user);
    const [profile, signOut] = within(menu).getAllByRole("menuitem");
    expect(document.activeElement).toBe(profile);

    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(signOut);
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(profile);
    await user.keyboard("{ArrowUp}");
    expect(document.activeElement).toBe(signOut);
  });

  it("never offers an Admin console item, not even to an admin", async () => {
    const user = userEvent.setup();
    authState.user = { ...authState.user, name: "The Admin", role: "admin" };
    renderNav();
    const menu = await openMenu(user);
    expect(within(menu).queryByRole("menuitem", { name: /admin/i })).toBeNull();
    expect(within(menu).getAllByRole("menuitem")).toHaveLength(2);
  });
});
