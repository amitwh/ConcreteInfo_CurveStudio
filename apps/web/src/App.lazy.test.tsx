import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const authState = vi.hoisted(() => ({ status: "anon" as "anon" | "authed" }));

vi.mock("./api/auth", () => ({
  useAuth: () => ({
    status: authState.status,
    user:
      authState.status === "authed"
        ? { id: "u1", email: "analyst@example.test", name: "", role: "member" }
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

// The idle chart-bundle prefetch is that behaviour's own suite (chartPrefetch.test.tsx);
// mocked here so these routing tests never schedule a real Plotly download.
vi.mock("./lib/chartPrefetch", () => ({ useChartPrefetch: () => {} }));

vi.mock("./pages/HomePage", () => ({
  HomePage: () => <main data-testid="home">home</main>,
}));

/**
 * Every lazy page is mocked the same way here: the factory counts the import (a mock
 * factory runs at its module's first import, so the count says when the chunk would have
 * been fetched), and the component it returns suspends on a promise that never settles —
 * which is exactly what a chunk still downloading looks like to React. A route that keeps
 * this loading state passes only if a Suspense boundary with the right fallback wraps it.
 */
const lazyPages = vi.hoisted(() => ({
  imports: {} as Record<string, number>,
  touch(name: string): void {
    this.imports[name] = (this.imports[name] ?? 0) + 1;
  },
  /** Throwing a thenable is how a component suspends; React.lazy does the same inside. */
  pendingPage(): never {
    throw new Promise<never>(() => {});
  },
}));

vi.mock("./pages/WorkspacePage", () => {
  lazyPages.touch("workspace");
  return { WorkspacePage: lazyPages.pendingPage };
});
vi.mock("./pages/HelpPage", () => {
  lazyPages.touch("help");
  return { HelpPage: lazyPages.pendingPage };
});
vi.mock("./pages/AboutPage", () => {
  lazyPages.touch("about");
  return { AboutPage: lazyPages.pendingPage };
});
vi.mock("./pages/ProfilePage", () => {
  lazyPages.touch("profile");
  return { ProfilePage: lazyPages.pendingPage };
});
vi.mock("./pages/RegisterPage", () => {
  lazyPages.touch("register");
  return { RegisterPage: lazyPages.pendingPage };
});
vi.mock("./pages/VerifyEmailPage", () => {
  lazyPages.touch("verify-email");
  return { VerifyEmailPage: lazyPages.pendingPage };
});
vi.mock("./pages/ForgotPasswordPage", () => {
  lazyPages.touch("forgot-password");
  return { ForgotPasswordPage: lazyPages.pendingPage };
});
vi.mock("./pages/ResetPasswordPage", () => {
  lazyPages.touch("reset-password");
  return { ResetPasswordPage: lazyPages.pendingPage };
});
vi.mock("./pages/OAuthCallbackPage", () => {
  lazyPages.touch("oauth-callback");
  return { OAuthCallbackPage: lazyPages.pendingPage };
});

beforeEach(() => {
  authState.status = "anon";
});

afterEach(cleanup);

function renderApp(path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

/** The loading state a route shows: the label text, inside the app's `role="status"`. */
function loadingState(label: string): HTMLElement {
  const state = screen.getByText(label);
  expect(state.getAttribute("role")).toBe("status");
  return state;
}

describe("the on-demand routes and their loading states", () => {
  // First on purpose: a mock factory runs once per file, when its module is first
  // imported, so only a test that comes before any navigation can prove the home page
  // imports none of them.
  it("downloads none of the on-demand pages for the public home", () => {
    renderApp("/");
    expect(screen.getByTestId("home")).toBeTruthy();
    expect(lazyPages.imports).toEqual({});
  });

  it.each([
    { path: "/help", label: "Opening help…", page: "help" },
    { path: "/about", label: "Opening about…", page: "about" },
    { path: "/register", label: "Opening account sign-up…", page: "register" },
    { path: "/verify-email", label: "Opening e-mail verification…", page: "verify-email" },
    { path: "/forgot-password", label: "Opening password recovery…", page: "forgot-password" },
    { path: "/reset-password", label: "Opening password reset…", page: "reset-password" },
    { path: "/oauth/callback", label: "Completing sign-in…", page: "oauth-callback" },
  ])("shows '$label' while the $page page loads ($path)", async ({ path, label, page }) => {
    renderApp(path);
    // the loading state is there the moment the route mounts, before the chunk arrives
    expect(loadingState(label)).toBeTruthy();
    // and the route did fetch the page module (a tick later, as a dynamic import does)
    await waitFor(() => expect(lazyPages.imports[page] ?? 0).toBeGreaterThanOrEqual(1));
  });

  it("shows the workspace's loading state under the signed-in shell", async () => {
    authState.status = "authed";
    renderApp("/");
    expect(loadingState("Opening your workspace…")).toBeTruthy();
    await waitFor(() => expect(lazyPages.imports["workspace"] ?? 0).toBeGreaterThanOrEqual(1));
    // the shell itself is not held back by the page it frames
    expect(await screen.findByRole("tablist", { name: "Workspaces" })).toBeTruthy();
  });

  it.each([
    { path: "/help", title: "Help contents", label: "Opening help…", page: "help" },
    { path: "/about", title: "About CurveStudio", label: "Opening about…", page: "about" },
    { path: "/profile", title: "Your profile", label: "Opening your profile…", page: "profile" },
  ])("opens '$title' over the workspace with '$label' inside", async ({ path, title, label, page }) => {
    authState.status = "authed";
    renderApp(path);
    expect(await screen.findByRole("dialog", { name: title })).toBeTruthy();
    expect(loadingState(label)).toBeTruthy();
    await waitFor(() => expect(lazyPages.imports[page] ?? 0).toBeGreaterThanOrEqual(1));
    // the workspace's own loading state is there too — the sheet opens without waiting
    expect(loadingState("Opening your workspace…")).toBeTruthy();
  });
});
