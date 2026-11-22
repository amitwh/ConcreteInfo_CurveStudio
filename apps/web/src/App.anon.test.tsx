import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
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
vi.mock("./pages/WorkspacePage", () => ({
  WorkspacePage: () => <div data-testid="workspace">workspace</div>,
}));

afterEach(cleanup);

beforeEach(() => {
  authState.status = "anon";
});

/** Reports the URL the router settled on, so redirects can be asserted. */
function LocationProbe(): React.ReactElement {
  const location = useLocation();
  return (
    <div data-testid="location">{`${location.pathname}${location.search}${location.hash}`}</div>
  );
}

function renderApp(path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <LocationProbe />
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

  it("sends a bare /login to the home sign-in card", async () => {
    renderApp("/login");
    await waitFor(() =>
      expect(screen.getByTestId("location").textContent).toBe("/#login-card"),
    );
    expect(screen.getByTestId("home")).toBeTruthy();
  });

  it("keeps the OAuth error when redirecting /login to the home card", async () => {
    renderApp("/login?oauth_error=provider");
    await waitFor(() =>
      expect(screen.getByTestId("location").textContent).toBe(
        "/?oauth_error=provider#login-card",
      ),
    );
    expect(screen.getByTestId("home")).toBeTruthy();
  });

  it("points the header's Sign in button at the home card", () => {
    renderApp("/");
    const signIn = screen.getByRole("link", { name: "Sign in" });
    expect(signIn.getAttribute("href")).toBe("/#login-card");
  });
});
