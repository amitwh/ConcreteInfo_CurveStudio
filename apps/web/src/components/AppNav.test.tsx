import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Workspace } from "../workspace/workspaces";
import { AppNav } from "./AppNav";

afterEach(cleanup);

vi.mock("../api/auth", () => ({
  useAuth: () => ({
    status: "authed",
    user: { id: "u1", email: "analyst@example.test", name: "" },
    logout: vi.fn(),
  }),
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
    </MemoryRouter>,
  );
  return { ...handlers, view };
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

  it("shows the account and never the email in a title-only control", () => {
    renderNav();
    expect(screen.getByText("analyst@example.test")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeTruthy();
  });
});
