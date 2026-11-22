import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const { authState } = vi.hoisted(() => ({
  authState: { status: "anon" as "anon" | "authed" },
}));

vi.mock("./api/auth", () => ({
  useAuth: () => ({
    status: authState.status,
    user:
      authState.status === "authed"
        ? { id: "u1", email: "analyst@example.test", name: "" }
        : null,
    login: vi.fn(),
    register: vi.fn(),
    logout: vi.fn(),
    adoptSession: vi.fn(),
  }),
}));

vi.mock("./components/ThemeToggle", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useTheme: () => ({ theme: "light", toggle: vi.fn() }),
}));

vi.mock("./components/useInstallPrompt", () => ({
  useInstallPrompt: () => ({ canInstall: false, install: vi.fn() }),
}));

// Routing is the subject; the pages themselves have their own tests.
vi.mock("./pages/HomePage", () => ({
  HomePage: () => <main data-testid="home">home</main>,
}));
vi.mock("./pages/LoginPage", () => ({
  LoginPage: () => <main data-testid="login">login</main>,
}));
vi.mock("./pages/WorkspacePage", () => ({
  WorkspacePage: () => <div data-testid="workspace">workspace</div>,
}));

afterEach(cleanup);

beforeEach(() => {
  authState.status = "anon";
});

function renderApp(path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App routing for visitors", () => {
  it("renders the home page at / for an anonymous visitor", () => {
    renderApp("/");
    expect(screen.getByTestId("home")).toBeTruthy();
    expect(screen.queryByTestId("workspace")).toBeNull();
  });

  it("redirects an unknown anonymous path to the home page", () => {
    renderApp("/no-such-page");
    expect(screen.getByTestId("home")).toBeTruthy();
  });

  it("keeps the workspace at / for an authenticated visitor", () => {
    authState.status = "authed";
    renderApp("/");
    expect(screen.getByTestId("workspace")).toBeTruthy();
    expect(screen.queryByTestId("home")).toBeNull();
  });
});
